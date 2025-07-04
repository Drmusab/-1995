"""
Advanced GPU Profiler for AI Assistant
Author: Drmusab
Last Modified: 2025-01-10 14:30:00 UTC

This module provides comprehensive GPU profiling capabilities for the AI assistant,
including real-time monitoring, CUDA kernel profiling, memory tracking, multi-GPU support,
and integration with all core system components.
"""

import asyncio
import threading
import time
import logging
import json
import uuid
from pathlib import Path
from typing import Optional, Dict, Any, List, Union, Set, Callable
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
from contextlib import contextmanager, asynccontextmanager
from collections import defaultdict, deque
import concurrent.futures

# Core imports
from src.core.config.loader import ConfigLoader
from src.core.events.event_bus import EventBus
from src.core.events.event_types import (
    GPUProfilerStarted, GPUProfilerStopped, GPUUtilizationAlert, 
    GPUMemoryAlert, GPUTemperatureAlert, GPUMemoryLeakDetected,
    GPUPerformanceBottleneckDetected, GPUKernelExecutionStarted,
    GPUKernelExecutionCompleted, GPUPowerUsageAlert,
    ComponentHealthChanged, SystemStateChanged
)
from src.core.error_handling import ErrorHandler, handle_exceptions
from src.core.dependency_injection import Container
from src.core.health_check import HealthCheck

# Observability components
from src.observability.monitoring.metrics import MetricsCollector
from src.observability.monitoring.tracing import TraceManager
from src.observability.logging.config import get_logger

# Optional GPU dependencies (with graceful fallbacks)
try:
    import torch
    import torch.profiler
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import nvidia_ml_py3 as nvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False


class GPUProfilingLevel(Enum):
    """GPU profiling detail levels."""
    LIGHTWEIGHT = "lightweight"        # Basic GPU usage tracking
    STANDARD = "standard"              # GPU + memory monitoring
    DETAILED = "detailed"              # + kernel profiling
    COMPREHENSIVE = "comprehensive"     # + power, temperature, detailed analysis


class GPUProfilingMode(Enum):
    """GPU profiling modes."""
    OFF = "off"
    MONITORING = "monitoring"          # Continuous monitoring
    PROFILING = "profiling"           # Detailed profiling sessions
    ADAPTIVE = "adaptive"             # Adaptive based on workload
    DEBUG = "debug"                   # Debug mode with verbose info


class GPUProfilerStatus(Enum):
    """GPU profiler operational status."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    ERROR = "error"


class GPUBottleneckType(Enum):
    """Types of GPU performance bottlenecks."""
    COMPUTE = "compute"
    MEMORY_BANDWIDTH = "memory_bandwidth"
    MEMORY_CAPACITY = "memory_capacity"
    THERMAL = "thermal"
    POWER = "power"
    SYNCHRONIZATION = "synchronization"


@dataclass
class GPUProfilingConfig:
    """Configuration for GPU profiling."""
    level: GPUProfilingLevel = GPUProfilingLevel.STANDARD
    mode: GPUProfilingMode = GPUProfilingMode.MONITORING
    sampling_interval_ms: float = 100.0  # 100ms sampling
    
    # Monitoring thresholds
    utilization_threshold_percent: float = 85.0
    memory_threshold_percent: float = 90.0
    temperature_threshold_celsius: float = 80.0
    power_threshold_percent: float = 95.0
    
    # Data retention
    max_samples: int = 10000
    retention_hours: int = 24
    
    # Output configuration
    enable_real_time_monitoring: bool = True
    enable_kernel_profiling: bool = True
    enable_memory_tracking: bool = True
    enable_power_monitoring: bool = True
    enable_temperature_monitoring: bool = True
    enable_multi_gpu: bool = True
    
    # Integration settings
    integrate_with_tracing: bool = True
    integrate_with_metrics: bool = True
    integrate_with_component_profiling: bool = True
    
    # Advanced features
    enable_bottleneck_detection: bool = True
    enable_optimization_suggestions: bool = True
    enable_leak_detection: bool = True


@dataclass
class GPUDeviceInfo:
    """Information about a GPU device."""
    device_id: int
    name: str
    compute_capability: Optional[str] = None
    total_memory_mb: float = 0.0
    driver_version: Optional[str] = None
    cuda_version: Optional[str] = None
    multiprocessor_count: int = 0
    max_threads_per_block: int = 0
    max_threads_per_multiprocessor: int = 0
    max_shared_memory_per_block: int = 0
    is_available: bool = True


@dataclass 
class GPUMetrics:
    """GPU metrics at a point in time."""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    device_id: int = 0
    
    # Utilization
    utilization_percent: float = 0.0
    memory_utilization_percent: float = 0.0
    
    # Memory
    memory_used_mb: float = 0.0
    memory_free_mb: float = 0.0
    memory_total_mb: float = 0.0
    memory_cached_mb: float = 0.0
    
    # Performance
    temperature_celsius: float = 0.0
    power_draw_watts: float = 0.0
    power_limit_watts: float = 0.0
    clock_graphics_mhz: int = 0
    clock_memory_mhz: int = 0
    
    # Processes
    process_count: int = 0
    active_kernels: int = 0


@dataclass
class GPUKernelInfo:
    """Information about a GPU kernel execution."""
    kernel_id: str
    kernel_name: str
    device_id: int
    component: Optional[str] = None
    
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    end_time: Optional[datetime] = None
    duration_ms: float = 0.0
    
    # Memory transfer
    memory_transferred_mb: float = 0.0
    memory_bandwidth_gbps: float = 0.0
    
    # Performance metrics
    occupancy_percent: float = 0.0
    achieved_occupancy_percent: float = 0.0
    
    # Grid/block information
    grid_size: tuple = field(default_factory=tuple)
    block_size: tuple = field(default_factory=tuple)


@dataclass
class GPUBottleneck:
    """Represents a detected GPU performance bottleneck."""
    bottleneck_id: str
    device_id: int
    bottleneck_type: GPUBottleneckType
    severity: str  # low, medium, high, critical
    
    # Detection info
    detected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    component: Optional[str] = None
    
    # Metrics
    current_value: float = 0.0
    threshold_value: float = 0.0
    impact_factor: float = 0.0  # 0-1 scale
    
    # Details
    description: str = ""
    suggested_actions: List[str] = field(default_factory=list)
    related_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class GPUMemoryLeak:
    """Represents a detected GPU memory leak."""
    leak_id: str
    device_id: int
    component: Optional[str] = None
    
    # Detection info
    detected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    first_seen: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Memory info
    leaked_memory_mb: float = 0.0
    growth_rate_mb_per_hour: float = 0.0
    allocation_pattern: Dict[str, Any] = field(default_factory=dict)
    
    # Context
    stack_trace: Optional[str] = None
    operation_context: Dict[str, Any] = field(default_factory=dict)


class GPUProfiler:
    """
    Advanced GPU Profiler for the AI Assistant.
    
    This profiler provides comprehensive GPU performance monitoring including:
    - Real-time GPU utilization monitoring (compute, memory, temperature)
    - CUDA kernel execution profiling
    - Memory allocation and deallocation tracking
    - Multi-GPU support for distributed processing
    - GPU memory fragmentation analysis
    - Power consumption monitoring
    - Performance bottleneck identification
    - Component-specific profiling
    - Integration with core assistant components
    """
    
    def __init__(self, container: Container):
        """Initialize the GPU profiler."""
        self.container = container
        self.config_loader = container.get(ConfigLoader)
        self.event_bus = container.get(EventBus)
        self.error_handler = container.get(ErrorHandler)
        self.health_check = container.get(HealthCheck)
        self.logger = get_logger(__name__)
        
        # Initialize configuration
        self.config = GPUProfilingConfig(
            level=GPUProfilingLevel(
                self.config_loader.get("gpu_profiler.level", "standard")
            ),
            mode=GPUProfilingMode(
                self.config_loader.get("gpu_profiler.mode", "monitoring")
            ),
            sampling_interval_ms=self.config_loader.get(
                "gpu_profiler.sampling_interval_ms", 100.0
            ),
            utilization_threshold_percent=self.config_loader.get(
                "gpu_profiler.utilization_threshold", 85.0
            ),
            memory_threshold_percent=self.config_loader.get(
                "gpu_profiler.memory_threshold", 90.0
            ),
            temperature_threshold_celsius=self.config_loader.get(
                "gpu_profiler.temperature_threshold", 80.0
            ),
            power_threshold_percent=self.config_loader.get(
                "gpu_profiler.power_threshold", 95.0
            ),
            max_samples=self.config_loader.get("gpu_profiler.max_samples", 10000),
            retention_hours=self.config_loader.get("gpu_profiler.retention_hours", 24),
            enable_real_time_monitoring=self.config_loader.get(
                "gpu_profiler.enable_real_time_monitoring", True
            ),
            enable_kernel_profiling=self.config_loader.get(
                "gpu_profiler.enable_kernel_profiling", True
            ),
            enable_memory_tracking=self.config_loader.get(
                "gpu_profiler.enable_memory_tracking", True
            ),
            enable_power_monitoring=self.config_loader.get(
                "gpu_profiler.enable_power_monitoring", True
            ),
            enable_temperature_monitoring=self.config_loader.get(
                "gpu_profiler.enable_temperature_monitoring", True
            ),
            enable_multi_gpu=self.config_loader.get(
                "gpu_profiler.enable_multi_gpu", True
            ),
            integrate_with_tracing=self.config_loader.get(
                "gpu_profiler.integrate_with_tracing", True
            ),
            integrate_with_metrics=self.config_loader.get(
                "gpu_profiler.integrate_with_metrics", True
            ),
            integrate_with_component_profiling=self.config_loader.get(
                "gpu_profiler.integrate_with_component_profiling", True
            ),
            enable_bottleneck_detection=self.config_loader.get(
                "gpu_profiler.enable_bottleneck_detection", True
            ),
            enable_optimization_suggestions=self.config_loader.get(
                "gpu_profiler.enable_optimization_suggestions", True
            ),
            enable_leak_detection=self.config_loader.get(
                "gpu_profiler.enable_leak_detection", True
            )
        )
        
        # State management
        self.status = GPUProfilerStatus.STOPPED
        self.monitoring_task: Optional[asyncio.Task] = None
        self.profiling_task: Optional[asyncio.Task] = None
        self.stop_event = asyncio.Event()
        
        # GPU device information
        self.gpu_devices: Dict[int, GPUDeviceInfo] = {}
        self.gpu_available = False
        
        # Data storage
        self.metrics_history: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=self.config.max_samples)
        )
        self.kernel_executions: Dict[str, GPUKernelInfo] = {}
        self.active_kernels: Set[str] = set()
        self.detected_bottlenecks: List[GPUBottleneck] = []
        self.detected_leaks: List[GPUMemoryLeak] = []
        
        # Component tracking
        self.component_gpu_usage: Dict[str, Dict[int, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        
        # Initialize GPU capabilities
        self._initialize_gpu_detection()
        
        # Setup monitoring and metrics
        if self.gpu_available:
            self._setup_monitoring()
            self._setup_metrics()
        else:
            self.logger.warning("No GPU detected or GPU libraries unavailable. GPU profiling disabled.")
        
        # Setup event handlers
        self._setup_event_handlers()
        
        # Register health check
        self.health_check.register_component("gpu_profiler", self._health_check_callback)
        
        self.logger.info(f"GPUProfiler initialized with level: {self.config.level.value}, available GPUs: {len(self.gpu_devices)}")

    def _initialize_gpu_detection(self) -> None:
        """Initialize GPU detection and device enumeration."""
        try:
            # Check if any GPU libraries are available
            if not (TORCH_AVAILABLE or PYNVML_AVAILABLE or NVML_AVAILABLE):
                self.logger.warning("No GPU libraries available (torch, pynvml, nvidia-ml-py3)")
                return
            
            # Try PyTorch first
            if TORCH_AVAILABLE and torch.cuda.is_available():
                self._detect_cuda_devices()
                self.gpu_available = True
                self.logger.info(f"Detected {len(self.gpu_devices)} CUDA devices via PyTorch")
            
            # Try NVML/pynvml for additional info
            if PYNVML_AVAILABLE or NVML_AVAILABLE:
                self._enhance_device_info_with_nvml()
                
        except Exception as e:
            self.logger.warning(f"Failed to initialize GPU detection: {str(e)}")
            self.gpu_available = False

    def _detect_cuda_devices(self) -> None:
        """Detect CUDA devices using PyTorch."""
        if not TORCH_AVAILABLE:
            return
            
        try:
            device_count = torch.cuda.device_count()
            for device_id in range(device_count):
                props = torch.cuda.get_device_properties(device_id)
                
                device_info = GPUDeviceInfo(
                    device_id=device_id,
                    name=props.name,
                    compute_capability=f"{props.major}.{props.minor}",
                    total_memory_mb=props.total_memory / (1024 * 1024),
                    multiprocessor_count=props.multi_processor_count,
                    max_threads_per_block=props.max_threads_per_block,
                    max_threads_per_multiprocessor=props.max_threads_per_multi_processor,
                    max_shared_memory_per_block=props.max_shared_memory_per_block,
                    is_available=True
                )
                
                self.gpu_devices[device_id] = device_info
                
        except Exception as e:
            self.logger.error(f"Failed to detect CUDA devices: {str(e)}")

    def _enhance_device_info_with_nvml(self) -> None:
        """Enhance device information using NVML."""
        try:
            if PYNVML_AVAILABLE:
                pynvml.nvmlInit()
                for device_id in self.gpu_devices:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
                    driver_version = pynvml.nvmlSystemGetDriverVersion()
                    self.gpu_devices[device_id].driver_version = driver_version.decode('utf-8')
            elif NVML_AVAILABLE:
                nvml.nvmlInit()
                for device_id in self.gpu_devices:
                    handle = nvml.nvmlDeviceGetHandleByIndex(device_id)
                    driver_version = nvml.nvmlSystemGetDriverVersion()
                    self.gpu_devices[device_id].driver_version = driver_version
                    
        except Exception as e:
            self.logger.debug(f"Could not enhance device info with NVML: {str(e)}")

    def _setup_monitoring(self) -> None:
        """Setup metrics and monitoring."""
        try:
            self.metrics = self.container.get(MetricsCollector)
            self.tracer = self.container.get(TraceManager) if self.config.integrate_with_tracing else None
            
        except Exception as e:
            self.logger.warning(f"Failed to setup monitoring: {str(e)}")
            self.metrics = None
            self.tracer = None

    def _setup_metrics(self) -> None:
        """Register GPU-related metrics with the metrics system."""
        if not self.metrics:
            return
            
        try:
            # GPU utilization metrics
            self.metrics.register_gauge("gpu_utilization_percent")
            self.metrics.register_gauge("gpu_memory_utilization_percent")
            
            # GPU memory metrics  
            self.metrics.register_gauge("gpu_memory_used_mb")
            self.metrics.register_gauge("gpu_memory_free_mb")
            self.metrics.register_gauge("gpu_memory_total_mb")
            self.metrics.register_gauge("gpu_memory_cached_mb")
            
            # GPU performance metrics
            self.metrics.register_gauge("gpu_temperature_celsius")
            self.metrics.register_gauge("gpu_power_draw_watts")
            self.metrics.register_gauge("gpu_clock_graphics_mhz")
            self.metrics.register_gauge("gpu_clock_memory_mhz")
            
            # GPU kernel metrics
            self.metrics.register_counter("gpu_kernel_executions_total")
            self.metrics.register_histogram("gpu_kernel_duration_ms")
            self.metrics.register_histogram("gpu_memory_transfer_mb")
            
            # GPU bottleneck metrics
            self.metrics.register_counter("gpu_bottlenecks_detected_total")
            self.metrics.register_counter("gpu_memory_leaks_detected_total")
            
            # Component-specific metrics
            self.metrics.register_gauge("gpu_component_memory_usage_mb")
            
            self.logger.debug("GPU metrics registered successfully")
            
        except Exception as e:
            self.logger.warning(f"Failed to register GPU metrics: {str(e)}")

    def _setup_event_handlers(self) -> None:
        """Setup event handlers for system integration."""
        try:
            # Processing events for component tracking
            self.event_bus.subscribe("processing_started", self._handle_processing_started)
            self.event_bus.subscribe("processing_completed", self._handle_processing_completed)
            
            # Workflow events for workflow-level GPU tracking
            self.event_bus.subscribe("workflow_started", self._handle_workflow_started)
            self.event_bus.subscribe("workflow_completed", self._handle_workflow_completed)
            self.event_bus.subscribe("workflow_step_started", self._handle_workflow_step_started)
            self.event_bus.subscribe("workflow_step_completed", self._handle_workflow_step_completed)
            
            # Component events
            self.event_bus.subscribe("component_started", self._handle_component_started)
            self.event_bus.subscribe("component_stopped", self._handle_component_stopped)
            
            self.logger.debug("GPU profiler event handlers registered")
            
        except Exception as e:
            self.logger.warning(f"Failed to setup event handlers: {str(e)}")

    async def _health_check_callback(self) -> Dict[str, Any]:
        """Health check callback for the GPU profiler."""
        try:
            health_status = {
                "status": self.status.value,
                "gpu_available": self.gpu_available,
                "gpu_count": len(self.gpu_devices),
                "monitoring_active": self.monitoring_task is not None and not self.monitoring_task.done(),
                "samples_collected": sum(len(history) for history in self.metrics_history.values()),
                "active_kernels": len(self.active_kernels),
                "detected_bottlenecks": len(self.detected_bottlenecks),
                "detected_leaks": len(self.detected_leaks)
            }
            
            # Add per-device status
            if self.gpu_available:
                health_status["devices"] = {}
                for device_id, device_info in self.gpu_devices.items():
                    device_status = {
                        "name": device_info.name,
                        "available": device_info.is_available,
                        "memory_total_mb": device_info.total_memory_mb
                    }
                    
                    # Add latest metrics if available
                    if device_id in self.metrics_history and self.metrics_history[device_id]:
                        latest_metrics = self.metrics_history[device_id][-1]
                        device_status.update({
                            "utilization_percent": latest_metrics.utilization_percent,
                            "memory_used_mb": latest_metrics.memory_used_mb,
                            "temperature_celsius": latest_metrics.temperature_celsius
                        })
                    
                    health_status["devices"][str(device_id)] = device_status
            
            return health_status
            
        except Exception as e:
            self.logger.error(f"GPU profiler health check failed: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "gpu_available": False
            }

    @handle_exceptions
    async def start_profiling(
        self,
        level: Optional[GPUProfilingLevel] = None,
        mode: Optional[GPUProfilingMode] = None
    ) -> str:
        """Start GPU profiling."""
        if not self.gpu_available:
            raise RuntimeError("GPU profiling unavailable - no GPU detected or GPU libraries missing")
            
        if self.status == GPUProfilerStatus.RUNNING:
            self.logger.warning("GPU profiler is already running")
            return "already_running"
            
        try:
            self.status = GPUProfilerStatus.STARTING
            
            # Update configuration if provided
            if level:
                self.config.level = level
            if mode:
                self.config.mode = mode
                
            # Clear previous data
            self.metrics_history.clear()
            self.kernel_executions.clear()
            self.active_kernels.clear()
            self.detected_bottlenecks.clear()
            self.detected_leaks.clear()
            
            # Start monitoring task
            if self.config.enable_real_time_monitoring:
                self.stop_event.clear()
                self.monitoring_task = asyncio.create_task(self._monitoring_loop())
            
            # Start profiling task for detailed modes
            if self.config.level in [GPUProfilingLevel.DETAILED, GPUProfilingLevel.COMPREHENSIVE]:
                self.profiling_task = asyncio.create_task(self._profiling_loop())
            
            self.status = GPUProfilerStatus.RUNNING
            
            # Emit profiler started event
            await self.event_bus.emit(GPUProfilerStarted(
                profiling_level=self.config.level.value,
                gpu_count=len(self.gpu_devices),
                profiler_config=asdict(self.config)
            ))
            
            self.logger.info(f"GPU profiler started with level: {self.config.level.value}, mode: {self.config.mode.value}")
            return "started"
            
        except Exception as e:
            self.status = GPUProfilerStatus.ERROR
            self.logger.error(f"Failed to start GPU profiler: {str(e)}")
            raise

    @handle_exceptions
    async def stop_profiling(self) -> Dict[str, Any]:
        """Stop GPU profiling."""
        if self.status == GPUProfilerStatus.STOPPED:
            self.logger.warning("GPU profiler is already stopped")
            return {"status": "already_stopped"}
            
        try:
            self.status = GPUProfilerStatus.STOPPING
            
            # Signal tasks to stop
            self.stop_event.set()
            
            # Wait for tasks to complete
            if self.monitoring_task and not self.monitoring_task.done():
                try:
                    await asyncio.wait_for(self.monitoring_task, timeout=5.0)
                except asyncio.TimeoutError:
                    self.monitoring_task.cancel()
                    
            if self.profiling_task and not self.profiling_task.done():
                try:
                    await asyncio.wait_for(self.profiling_task, timeout=5.0)
                except asyncio.TimeoutError:
                    self.profiling_task.cancel()
            
            self.status = GPUProfilerStatus.STOPPED
            
            # Collect final statistics
            total_samples = sum(len(history) for history in self.metrics_history.values())
            
            # Emit profiler stopped event
            await self.event_bus.emit(GPUProfilerStopped(
                profiling_duration=0.0,  # TODO: Calculate actual duration
                total_samples=total_samples,
                report_path=None  # TODO: Generate report
            ))
            
            self.logger.info("GPU profiler stopped")
            
            return {
                "status": "stopped",
                "total_samples": total_samples,
                "devices_monitored": len(self.gpu_devices),
                "bottlenecks_detected": len(self.detected_bottlenecks),
                "leaks_detected": len(self.detected_leaks)
            }
            
        except Exception as e:
            self.status = GPUProfilerStatus.ERROR
            self.logger.error(f"Failed to stop GPU profiler: {str(e)}")
            raise

    async def _monitoring_loop(self) -> None:
        """Main monitoring loop for collecting GPU metrics."""
        self.logger.debug("Starting GPU monitoring loop")
        
        while not self.stop_event.is_set():
            try:
                await self._collect_gpu_metrics()
                
                # Check for bottlenecks and alerts
                if self.config.enable_bottleneck_detection:
                    await self._detect_bottlenecks()
                
                # Check for memory leaks
                if self.config.enable_leak_detection:
                    await self._detect_memory_leaks()
                
                # Sleep until next sampling interval
                await asyncio.sleep(self.config.sampling_interval_ms / 1000.0)
                
            except Exception as e:
                self.logger.error(f"Error in GPU monitoring loop: {str(e)}")
                await asyncio.sleep(1.0)  # Brief pause before retrying

    async def _profiling_loop(self) -> None:
        """Main profiling loop for detailed GPU analysis."""
        self.logger.debug("Starting GPU profiling loop")
        
        while not self.stop_event.is_set():
            try:
                # Detailed profiling logic here
                if self.config.enable_kernel_profiling:
                    await self._profile_kernel_executions()
                
                await asyncio.sleep(self.config.sampling_interval_ms / 1000.0)
                
            except Exception as e:
                self.logger.error(f"Error in GPU profiling loop: {str(e)}")
                await asyncio.sleep(1.0)

    async def _collect_gpu_metrics(self) -> None:
        """Collect GPU metrics from all available devices."""
        if not self.gpu_available:
            return
            
        for device_id in self.gpu_devices:
            try:
                metrics = await self._get_device_metrics(device_id)
                self.metrics_history[device_id].append(metrics)
                
                # Update metrics collector
                if self.metrics:
                    self._update_metrics_collector(device_id, metrics)
                
                # Check thresholds and emit alerts
                await self._check_thresholds(device_id, metrics)
                
            except Exception as e:
                self.logger.error(f"Failed to collect metrics for GPU {device_id}: {str(e)}")

    async def _get_device_metrics(self, device_id: int) -> GPUMetrics:
        """Get metrics for a specific GPU device."""
        metrics = GPUMetrics(device_id=device_id)
        
        try:
            # PyTorch metrics
            if TORCH_AVAILABLE and torch.cuda.is_available():
                metrics.memory_used_mb = torch.cuda.memory_allocated(device_id) / (1024 * 1024)
                metrics.memory_cached_mb = torch.cuda.memory_reserved(device_id) / (1024 * 1024)
                metrics.memory_total_mb = self.gpu_devices[device_id].total_memory_mb
                metrics.memory_free_mb = metrics.memory_total_mb - metrics.memory_used_mb
                
                if metrics.memory_total_mb > 0:
                    metrics.memory_utilization_percent = (metrics.memory_used_mb / metrics.memory_total_mb) * 100
            
            # NVML metrics for additional information
            if PYNVML_AVAILABLE:
                handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
                
                # Utilization
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                metrics.utilization_percent = util.gpu
                metrics.memory_utilization_percent = util.memory
                
                # Temperature
                try:
                    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    metrics.temperature_celsius = temp
                except:
                    pass
                
                # Power
                try:
                    power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW to W
                    metrics.power_draw_watts = power
                    
                    power_limit = pynvml.nvmlDeviceGetPowerManagementLimitConstraints(handle)[1] / 1000.0
                    metrics.power_limit_watts = power_limit
                except:
                    pass
                
                # Clock speeds
                try:
                    graphics_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS)
                    memory_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
                    metrics.clock_graphics_mhz = graphics_clock
                    metrics.clock_memory_mhz = memory_clock
                except:
                    pass
                
                # Process count
                try:
                    processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                    metrics.process_count = len(processes)
                except:
                    pass
            
        except Exception as e:
            self.logger.debug(f"Could not collect all metrics for GPU {device_id}: {str(e)}")
        
        return metrics

    def _update_metrics_collector(self, device_id: int, metrics: GPUMetrics) -> None:
        """Update the metrics collector with GPU metrics."""
        try:
            tags = {"device_id": str(device_id), "device_name": self.gpu_devices[device_id].name}
            
            # Update utilization metrics
            self.metrics.set("gpu_utilization_percent", metrics.utilization_percent, tags=tags)
            self.metrics.set("gpu_memory_utilization_percent", metrics.memory_utilization_percent, tags=tags)
            
            # Update memory metrics
            self.metrics.set("gpu_memory_used_mb", metrics.memory_used_mb, tags=tags)
            self.metrics.set("gpu_memory_free_mb", metrics.memory_free_mb, tags=tags)
            self.metrics.set("gpu_memory_total_mb", metrics.memory_total_mb, tags=tags)
            self.metrics.set("gpu_memory_cached_mb", metrics.memory_cached_mb, tags=tags)
            
            # Update performance metrics
            self.metrics.set("gpu_temperature_celsius", metrics.temperature_celsius, tags=tags)
            self.metrics.set("gpu_power_draw_watts", metrics.power_draw_watts, tags=tags)
            self.metrics.set("gpu_clock_graphics_mhz", metrics.clock_graphics_mhz, tags=tags)
            self.metrics.set("gpu_clock_memory_mhz", metrics.clock_memory_mhz, tags=tags)
            
        except Exception as e:
            self.logger.debug(f"Failed to update metrics collector: {str(e)}")

    async def _check_thresholds(self, device_id: int, metrics: GPUMetrics) -> None:
        """Check thresholds and emit alerts if necessary."""
        device_name = self.gpu_devices[device_id].name
        
        # GPU utilization alert
        if metrics.utilization_percent > self.config.utilization_threshold_percent:
            await self.event_bus.emit(GPUUtilizationAlert(
                device_id=device_id,
                device_name=device_name,
                utilization_percent=metrics.utilization_percent,
                threshold_percent=self.config.utilization_threshold_percent,
                alert_type="high"
            ))
        
        # GPU memory alert
        if metrics.memory_utilization_percent > self.config.memory_threshold_percent:
            await self.event_bus.emit(GPUMemoryAlert(
                device_id=device_id,
                device_name=device_name,
                memory_used_mb=metrics.memory_used_mb,
                memory_total_mb=metrics.memory_total_mb,
                usage_percent=metrics.memory_utilization_percent,
                threshold_percent=self.config.memory_threshold_percent
            ))
        
        # Temperature alert
        if metrics.temperature_celsius > self.config.temperature_threshold_celsius:
            await self.event_bus.emit(GPUTemperatureAlert(
                device_id=device_id,
                device_name=device_name,
                temperature_celsius=metrics.temperature_celsius,
                threshold_celsius=self.config.temperature_threshold_celsius
            ))
        
        # Power usage alert
        if (metrics.power_limit_watts > 0 and 
            (metrics.power_draw_watts / metrics.power_limit_watts * 100) > self.config.power_threshold_percent):
            await self.event_bus.emit(GPUPowerUsageAlert(
                device_id=device_id,
                device_name=device_name,
                power_draw_watts=metrics.power_draw_watts,
                power_limit_watts=metrics.power_limit_watts,
                usage_percent=(metrics.power_draw_watts / metrics.power_limit_watts * 100)
            ))

    async def _detect_bottlenecks(self) -> None:
        """Detect GPU performance bottlenecks."""
        # TODO: Implement bottleneck detection logic
        pass

    async def _detect_memory_leaks(self) -> None:
        """Detect GPU memory leaks."""
        # TODO: Implement memory leak detection logic
        pass

    async def _profile_kernel_executions(self) -> None:
        """Profile GPU kernel executions."""
        # TODO: Implement kernel profiling logic
        pass

    # Event handlers
    async def _handle_processing_started(self, event) -> None:
        """Handle processing started events."""
        # TODO: Track component GPU usage
        pass

    async def _handle_processing_completed(self, event) -> None:
        """Handle processing completed events."""
        # TODO: Update component GPU usage statistics
        pass

    async def _handle_workflow_started(self, event) -> None:
        """Handle workflow started events."""
        # TODO: Track workflow-level GPU usage
        pass

    async def _handle_workflow_completed(self, event) -> None:
        """Handle workflow completed events."""
        # TODO: Update workflow GPU usage statistics
        pass

    async def _handle_workflow_step_started(self, event) -> None:
        """Handle workflow step started events."""
        # TODO: Track step-level GPU usage
        pass

    async def _handle_workflow_step_completed(self, event) -> None:
        """Handle workflow step completed events."""
        # TODO: Update step GPU usage statistics
        pass

    async def _handle_component_started(self, event) -> None:
        """Handle component started events."""
        # TODO: Initialize component GPU tracking
        pass

    async def _handle_component_stopped(self, event) -> None:
        """Handle component stopped events."""
        # TODO: Finalize component GPU tracking
        pass

    @handle_exceptions
    async def get_gpu_status(self) -> Dict[str, Any]:
        """Get current GPU status and metrics."""
        if not self.gpu_available:
            return {
                "available": False,
                "error": "No GPU detected or GPU libraries unavailable"
            }
        
        status = {
            "available": True,
            "profiler_status": self.status.value,
            "device_count": len(self.gpu_devices),
            "devices": {}
        }
        
        for device_id, device_info in self.gpu_devices.items():
            device_status = {
                "name": device_info.name,
                "compute_capability": device_info.compute_capability,
                "total_memory_mb": device_info.total_memory_mb,
                "driver_version": device_info.driver_version,
                "is_available": device_info.is_available
            }
            
            # Add latest metrics if available
            if device_id in self.metrics_history and self.metrics_history[device_id]:
                latest_metrics = self.metrics_history[device_id][-1]
                device_status["current_metrics"] = asdict(latest_metrics)
            
            status["devices"][str(device_id)] = device_status
        
        return status

    @handle_exceptions
    async def get_optimization_suggestions(self) -> List[Dict[str, Any]]:
        """Get GPU optimization suggestions based on current metrics and detected issues."""
        suggestions = []
        
        if not self.gpu_available:
            return suggestions
        
        # Analyze recent metrics for each device
        for device_id, device_info in self.gpu_devices.items():
            if device_id not in self.metrics_history or not self.metrics_history[device_id]:
                continue
                
            recent_metrics = list(self.metrics_history[device_id])[-10:]  # Last 10 samples
            
            # High memory usage suggestion
            avg_memory_util = sum(m.memory_utilization_percent for m in recent_metrics) / len(recent_metrics)
            if avg_memory_util > 85:
                suggestions.append({
                    "type": "memory",
                    "severity": "high",
                    "device_id": device_id,
                    "device_name": device_info.name,
                    "message": f"High memory usage ({avg_memory_util:.1f}%) on GPU {device_id}",
                    "suggestion": "Consider using gradient checkpointing, reducing batch size, or clearing cache with torch.cuda.empty_cache()"
                })
            
            # Low utilization suggestion
            avg_gpu_util = sum(m.utilization_percent for m in recent_metrics) / len(recent_metrics)
            if avg_gpu_util < 20:
                suggestions.append({
                    "type": "utilization",
                    "severity": "medium",
                    "device_id": device_id,
                    "device_name": device_info.name,
                    "message": f"Low GPU utilization ({avg_gpu_util:.1f}%) on GPU {device_id}",
                    "suggestion": "Consider increasing batch size, using mixed precision training, or optimizing data loading pipeline"
                })
            
            # High temperature suggestion
            avg_temp = sum(m.temperature_celsius for m in recent_metrics if m.temperature_celsius > 0) / max(1, len([m for m in recent_metrics if m.temperature_celsius > 0]))
            if avg_temp > 75:
                suggestions.append({
                    "type": "thermal",
                    "severity": "high",
                    "device_id": device_id,
                    "device_name": device_info.name,
                    "message": f"High temperature ({avg_temp:.1f}°C) on GPU {device_id}",
                    "suggestion": "Check cooling system, reduce workload intensity, or improve case ventilation"
                })
        
        # Add detected bottlenecks as suggestions
        for bottleneck in self.detected_bottlenecks[-10:]:  # Recent bottlenecks
            suggestions.append({
                "type": "bottleneck",
                "severity": bottleneck.severity,
                "device_id": bottleneck.device_id,
                "device_name": self.gpu_devices[bottleneck.device_id].name,
                "message": f"Performance bottleneck detected: {bottleneck.description}",
                "suggestion": "; ".join(bottleneck.suggested_actions)
            })
        
        # Add memory leak suggestions
        for leak in self.detected_leaks[-5:]:  # Recent leaks
            suggestions.append({
                "type": "memory_leak",
                "severity": "critical",
                "device_id": leak.device_id,
                "device_name": self.gpu_devices[leak.device_id].name,
                "message": f"Memory leak detected: {leak.leaked_memory_mb:.2f} MB leaked",
                "suggestion": "Review memory allocation patterns and ensure proper cleanup of GPU tensors"
            })
        
        return suggestions

    @handle_exceptions
    def get_component_gpu_usage(self, component: Optional[str] = None) -> Dict[str, Any]:
        """Get GPU usage statistics for components."""
        if component:
            if component in self.component_gpu_usage:
                return {
                    "component": component,
                    "usage_by_device": dict(self.component_gpu_usage[component])
                }
            else:
                return {
                    "component": component,
                    "usage_by_device": {},
                    "error": "Component not found"
                }
        else:
            return {
                "all_components": {
                    comp: dict(usage) for comp, usage in self.component_gpu_usage.items()
                }
            }

    def __repr__(self) -> str:
        return f"GPUProfiler(status={self.status.value}, devices={len(self.gpu_devices)}, available={self.gpu_available})"