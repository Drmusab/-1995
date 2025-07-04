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
        if not self.gpu_available:
            return
            
        for device_id in self.gpu_devices:
            try:
                await self._detect_device_bottlenecks(device_id)
            except Exception as e:
                self.logger.error(f"Failed to detect bottlenecks for GPU {device_id}: {str(e)}")

    async def _detect_device_bottlenecks(self, device_id: int) -> None:
        """Detect bottlenecks for a specific GPU device."""
        if device_id not in self.metrics_history or len(self.metrics_history[device_id]) < 5:
            return
            
        recent_metrics = list(self.metrics_history[device_id])[-10:]  # Last 10 samples
        device_info = self.gpu_devices[device_id]
        
        # Compute utilization bottleneck
        avg_utilization = sum(m.utilization_percent for m in recent_metrics) / len(recent_metrics)
        if avg_utilization > 95:
            bottleneck = GPUBottleneck(
                bottleneck_id=str(uuid.uuid4()),
                device_id=device_id,
                bottleneck_type=GPUBottleneckType.COMPUTE,
                severity="high",
                current_value=avg_utilization,
                threshold_value=95.0,
                impact_factor=min(1.0, avg_utilization / 100.0),
                description=f"High compute utilization ({avg_utilization:.1f}%) on GPU {device_id}",
                suggested_actions=[
                    "Consider reducing model complexity",
                    "Optimize kernel implementations",
                    "Use mixed precision training",
                    "Distribute workload across multiple GPUs"
                ],
                related_metrics={
                    "utilization_percent": avg_utilization,
                    "memory_utilization": sum(m.memory_utilization_percent for m in recent_metrics) / len(recent_metrics)
                }
            )
            self.detected_bottlenecks.append(bottleneck)
            
            await self.event_bus.emit(GPUPerformanceBottleneckDetected(
                device_id=device_id,
                device_name=device_info.name,
                bottleneck_type=bottleneck.bottleneck_type.value,
                bottleneck_details=asdict(bottleneck)
            ))
        
        # Memory bandwidth bottleneck
        avg_memory_util = sum(m.memory_utilization_percent for m in recent_metrics) / len(recent_metrics)
        if avg_memory_util > 90 and avg_utilization < 70:
            bottleneck = GPUBottleneck(
                bottleneck_id=str(uuid.uuid4()),
                device_id=device_id,
                bottleneck_type=GPUBottleneckType.MEMORY_BANDWIDTH,
                severity="medium",
                current_value=avg_memory_util,
                threshold_value=90.0,
                impact_factor=min(1.0, avg_memory_util / 100.0),
                description=f"Memory bandwidth bottleneck on GPU {device_id}",
                suggested_actions=[
                    "Optimize memory access patterns",
                    "Use memory coalescing techniques",
                    "Reduce memory transfers between host and device",
                    "Consider using unified memory"
                ],
                related_metrics={
                    "memory_utilization": avg_memory_util,
                    "compute_utilization": avg_utilization
                }
            )
            self.detected_bottlenecks.append(bottleneck)
            
            await self.event_bus.emit(GPUPerformanceBottleneckDetected(
                device_id=device_id,
                device_name=device_info.name,
                bottleneck_type=bottleneck.bottleneck_type.value,
                bottleneck_details=asdict(bottleneck)
            ))
        
        # Memory capacity bottleneck
        avg_memory_used = sum(m.memory_used_mb for m in recent_metrics) / len(recent_metrics)
        memory_usage_percent = (avg_memory_used / device_info.total_memory_mb) * 100
        if memory_usage_percent > 95:
            bottleneck = GPUBottleneck(
                bottleneck_id=str(uuid.uuid4()),
                device_id=device_id,
                bottleneck_type=GPUBottleneckType.MEMORY_CAPACITY,
                severity="critical",
                current_value=memory_usage_percent,
                threshold_value=95.0,
                impact_factor=min(1.0, memory_usage_percent / 100.0),
                description=f"Memory capacity bottleneck on GPU {device_id} ({memory_usage_percent:.1f}% used)",
                suggested_actions=[
                    "Reduce batch size",
                    "Use gradient checkpointing",
                    "Clear unused GPU memory with torch.cuda.empty_cache()",
                    "Consider model parallelization"
                ],
                related_metrics={
                    "memory_used_mb": avg_memory_used,
                    "memory_total_mb": device_info.total_memory_mb,
                    "memory_usage_percent": memory_usage_percent
                }
            )
            self.detected_bottlenecks.append(bottleneck)
            
            await self.event_bus.emit(GPUPerformanceBottleneckDetected(
                device_id=device_id,
                device_name=device_info.name,
                bottleneck_type=bottleneck.bottleneck_type.value,
                bottleneck_details=asdict(bottleneck)
            ))
        
        # Thermal bottleneck
        avg_temp = sum(m.temperature_celsius for m in recent_metrics if m.temperature_celsius > 0) / max(1, len([m for m in recent_metrics if m.temperature_celsius > 0]))
        if avg_temp > 85:
            bottleneck = GPUBottleneck(
                bottleneck_id=str(uuid.uuid4()),
                device_id=device_id,
                bottleneck_type=GPUBottleneckType.THERMAL,
                severity="high",
                current_value=avg_temp,
                threshold_value=85.0,
                impact_factor=min(1.0, avg_temp / 100.0),
                description=f"Thermal bottleneck on GPU {device_id} ({avg_temp:.1f}°C)",
                suggested_actions=[
                    "Improve cooling system",
                    "Reduce GPU workload",
                    "Lower power limits",
                    "Check for dust buildup"
                ],
                related_metrics={
                    "temperature_celsius": avg_temp,
                    "utilization_percent": avg_utilization
                }
            )
            self.detected_bottlenecks.append(bottleneck)
            
            await self.event_bus.emit(GPUPerformanceBottleneckDetected(
                device_id=device_id,
                device_name=device_info.name,
                bottleneck_type=bottleneck.bottleneck_type.value,
                bottleneck_details=asdict(bottleneck)
            ))
        
        # Power bottleneck
        avg_power = sum(m.power_draw_watts for m in recent_metrics if m.power_draw_watts > 0) / max(1, len([m for m in recent_metrics if m.power_draw_watts > 0]))
        avg_power_limit = sum(m.power_limit_watts for m in recent_metrics if m.power_limit_watts > 0) / max(1, len([m for m in recent_metrics if m.power_limit_watts > 0]))
        if avg_power_limit > 0 and (avg_power / avg_power_limit) > 0.98:
            bottleneck = GPUBottleneck(
                bottleneck_id=str(uuid.uuid4()),
                device_id=device_id,
                bottleneck_type=GPUBottleneckType.POWER,
                severity="medium",
                current_value=avg_power,
                threshold_value=avg_power_limit,
                impact_factor=min(1.0, avg_power / avg_power_limit),
                description=f"Power bottleneck on GPU {device_id} ({avg_power:.1f}W / {avg_power_limit:.1f}W)",
                suggested_actions=[
                    "Increase power limit if possible",
                    "Optimize compute workload",
                    "Use power-efficient algorithms",
                    "Consider undervolting"
                ],
                related_metrics={
                    "power_draw_watts": avg_power,
                    "power_limit_watts": avg_power_limit,
                    "power_usage_percent": (avg_power / avg_power_limit) * 100
                }
            )
            self.detected_bottlenecks.append(bottleneck)
            
            await self.event_bus.emit(GPUPerformanceBottleneckDetected(
                device_id=device_id,
                device_name=device_info.name,
                bottleneck_type=bottleneck.bottleneck_type.value,
                bottleneck_details=asdict(bottleneck)
            ))
        
        # Cleanup old bottlenecks (keep last 50)
        self.detected_bottlenecks = self.detected_bottlenecks[-50:]

    async def _detect_memory_leaks(self) -> None:
        """Detect GPU memory leaks."""
        if not self.gpu_available:
            return
            
        for device_id in self.gpu_devices:
            try:
                await self._detect_device_memory_leaks(device_id)
            except Exception as e:
                self.logger.error(f"Failed to detect memory leaks for GPU {device_id}: {str(e)}")

    async def _detect_device_memory_leaks(self, device_id: int) -> None:
        """Detect memory leaks for a specific GPU device."""
        if device_id not in self.metrics_history or len(self.metrics_history[device_id]) < 20:
            return  # Need at least 20 samples for trend analysis
            
        metrics_list = list(self.metrics_history[device_id])
        device_info = self.gpu_devices[device_id]
        
        # Analyze memory usage trend over time
        memory_usage_trend = [m.memory_used_mb for m in metrics_list[-20:]]
        time_points = [(i * self.config.sampling_interval_ms / 1000.0) for i in range(len(memory_usage_trend))]
        
        # Calculate linear regression to detect consistent memory growth
        if len(memory_usage_trend) >= 10:
            # Simple linear regression
            n = len(memory_usage_trend)
            sum_x = sum(time_points)
            sum_y = sum(memory_usage_trend)
            sum_xy = sum(x * y for x, y in zip(time_points, memory_usage_trend))
            sum_x2 = sum(x * x for x in time_points)
            
            # Calculate slope (growth rate)
            if n * sum_x2 - sum_x * sum_x != 0:
                slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
                
                # Convert slope to MB/hour
                growth_rate_mb_per_hour = slope * 3600  # seconds to hours
                
                # Detect potential memory leak
                if growth_rate_mb_per_hour > 10.0:  # Growing by more than 10MB/hour
                    # Check if this is a new leak or existing one
                    existing_leak = None
                    for leak in self.detected_leaks:
                        if (leak.device_id == device_id and 
                            (datetime.now(timezone.utc) - leak.detected_at).total_seconds() < 3600):  # Within last hour
                            existing_leak = leak
                            break
                    
                    if existing_leak:
                        # Update existing leak
                        existing_leak.leaked_memory_mb = memory_usage_trend[-1] - memory_usage_trend[0]
                        existing_leak.growth_rate_mb_per_hour = growth_rate_mb_per_hour
                    else:
                        # Create new leak detection
                        leak = GPUMemoryLeak(
                            leak_id=str(uuid.uuid4()),
                            device_id=device_id,
                            leaked_memory_mb=memory_usage_trend[-1] - memory_usage_trend[0],
                            growth_rate_mb_per_hour=growth_rate_mb_per_hour,
                            allocation_pattern={
                                "trend_slope": slope,
                                "initial_memory_mb": memory_usage_trend[0],
                                "current_memory_mb": memory_usage_trend[-1],
                                "sample_count": len(memory_usage_trend),
                                "time_span_seconds": time_points[-1] - time_points[0]
                            }
                        )
                        self.detected_leaks.append(leak)
                        
                        # Emit memory leak event
                        await self.event_bus.emit(GPUMemoryLeakDetected(
                            device_id=device_id,
                            device_name=device_info.name,
                            leaked_memory_mb=leak.leaked_memory_mb,
                            leak_details=asdict(leak)
                        ))
                        
                        self.logger.warning(
                            f"GPU memory leak detected on device {device_id}: "
                            f"{growth_rate_mb_per_hour:.2f} MB/hour growth rate"
                        )
        
        # Check for sudden memory spikes
        if len(metrics_list) >= 5:
            recent_memory = [m.memory_used_mb for m in metrics_list[-5:]]
            if len(recent_memory) >= 2:
                memory_spike = recent_memory[-1] - recent_memory[0]
                if memory_spike > 500:  # Sudden 500MB+ increase
                    leak = GPUMemoryLeak(
                        leak_id=str(uuid.uuid4()),
                        device_id=device_id,
                        leaked_memory_mb=memory_spike,
                        growth_rate_mb_per_hour=0.0,  # Not a gradual leak
                        allocation_pattern={
                            "type": "spike",
                            "spike_size_mb": memory_spike,
                            "before_mb": recent_memory[0],
                            "after_mb": recent_memory[-1],
                            "time_span_seconds": 5 * self.config.sampling_interval_ms / 1000.0
                        }
                    )
                    self.detected_leaks.append(leak)
                    
                    await self.event_bus.emit(GPUMemoryLeakDetected(
                        device_id=device_id,
                        device_name=device_info.name,
                        leaked_memory_mb=memory_spike,
                        leak_details=asdict(leak)
                    ))
        
        # Cleanup old leak detections (keep last 20)
        self.detected_leaks = self.detected_leaks[-20:]

    async def _profile_kernel_executions(self) -> None:
        """Profile GPU kernel executions."""
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return
            
        try:
            # Use PyTorch profiler for kernel tracking
            if self.config.level in [GPUProfilingLevel.DETAILED, GPUProfilingLevel.COMPREHENSIVE]:
                await self._profile_with_torch_profiler()
            else:
                await self._basic_kernel_monitoring()
                
        except Exception as e:
            self.logger.error(f"Failed to profile kernel executions: {str(e)}")

    async def _profile_with_torch_profiler(self) -> None:
        """Use PyTorch profiler for detailed kernel analysis."""
        try:
            # This would be used in conjunction with actual model execution
            # For now, we'll monitor active contexts and streams
            for device_id in self.gpu_devices:
                if torch.cuda.is_available():
                    # Check for active CUDA contexts
                    try:
                        torch.cuda.set_device(device_id)
                        current_stream = torch.cuda.current_stream(device_id)
                        
                        # Monitor stream synchronization events
                        if hasattr(current_stream, 'query') and not current_stream.query():
                            # Stream is busy, kernel is likely executing
                            kernel_id = f"stream_{device_id}_{int(time.time() * 1000)}"
                            
                            kernel_info = GPUKernelInfo(
                                kernel_id=kernel_id,
                                kernel_name="active_kernel",
                                device_id=device_id
                            )
                            
                            self.kernel_executions[kernel_id] = kernel_info
                            self.active_kernels.add(kernel_id)
                            
                            # Emit kernel started event
                            await self.event_bus.emit(GPUKernelExecutionStarted(
                                device_id=device_id,
                                kernel_name=kernel_info.kernel_name,
                                kernel_id=kernel_id
                            ))
                            
                    except Exception as e:
                        self.logger.debug(f"Could not monitor stream for device {device_id}: {str(e)}")
                        
        except Exception as e:
            self.logger.debug(f"PyTorch profiler monitoring failed: {str(e)}")

    async def _basic_kernel_monitoring(self) -> None:
        """Basic kernel monitoring without detailed profiling."""
        try:
            # Monitor GPU utilization changes to infer kernel activity
            for device_id in self.gpu_devices:
                if device_id not in self.metrics_history or len(self.metrics_history[device_id]) < 2:
                    continue
                    
                recent_metrics = list(self.metrics_history[device_id])[-2:]
                
                # If utilization increased significantly, assume kernel started
                utilization_change = recent_metrics[-1].utilization_percent - recent_metrics[-2].utilization_percent
                
                if utilization_change > 20:  # 20% increase in utilization
                    kernel_id = f"inferred_{device_id}_{int(time.time() * 1000)}"
                    
                    kernel_info = GPUKernelInfo(
                        kernel_id=kernel_id,
                        kernel_name="inferred_kernel",
                        device_id=device_id
                    )
                    
                    self.kernel_executions[kernel_id] = kernel_info
                    self.active_kernels.add(kernel_id)
                    
                    # Update metrics
                    if self.metrics:
                        self.metrics.increment("gpu_kernel_executions_total", 
                                            tags={"device_id": str(device_id)})
                
                # Check for completed kernels
                completed_kernels = []
                for kernel_id in list(self.active_kernels):
                    kernel_info = self.kernel_executions.get(kernel_id)
                    if kernel_info and kernel_info.device_id == device_id:
                        # If utilization dropped, assume kernel completed
                        if utilization_change < -10:  # 10% decrease
                            kernel_info.end_time = datetime.now(timezone.utc)
                            kernel_info.duration_ms = (
                                kernel_info.end_time - kernel_info.start_time
                            ).total_seconds() * 1000
                            
                            completed_kernels.append(kernel_id)
                            
                            # Emit kernel completed event
                            await self.event_bus.emit(GPUKernelExecutionCompleted(
                                device_id=device_id,
                                kernel_name=kernel_info.kernel_name,
                                kernel_id=kernel_id,
                                execution_time_ms=kernel_info.duration_ms
                            ))
                            
                            # Update metrics
                            if self.metrics:
                                self.metrics.record("gpu_kernel_duration_ms", 
                                                  kernel_info.duration_ms,
                                                  tags={"device_id": str(device_id)})
                
                # Remove completed kernels from active set
                for kernel_id in completed_kernels:
                    self.active_kernels.discard(kernel_id)
                    
        except Exception as e:
            self.logger.debug(f"Basic kernel monitoring failed: {str(e)}")

    @contextmanager
    def profile_kernel(self, kernel_name: str, device_id: int = 0, component: Optional[str] = None):
        """Context manager for profiling a specific kernel execution."""
        kernel_id = str(uuid.uuid4())
        
        kernel_info = GPUKernelInfo(
            kernel_id=kernel_id,
            kernel_name=kernel_name,
            device_id=device_id,
            component=component
        )
        
        try:
            # Record start
            self.kernel_executions[kernel_id] = kernel_info
            self.active_kernels.add(kernel_id)
            
            # Emit start event
            asyncio.create_task(self.event_bus.emit(GPUKernelExecutionStarted(
                device_id=device_id,
                kernel_name=kernel_name,
                component=component,
                kernel_id=kernel_id
            )))
            
            yield kernel_info
            
        finally:
            # Record end
            kernel_info.end_time = datetime.now(timezone.utc)
            kernel_info.duration_ms = (
                kernel_info.end_time - kernel_info.start_time
            ).total_seconds() * 1000
            
            self.active_kernels.discard(kernel_id)
            
            # Emit completion event
            asyncio.create_task(self.event_bus.emit(GPUKernelExecutionCompleted(
                device_id=device_id,
                kernel_name=kernel_name,
                component=component,
                kernel_id=kernel_id,
                execution_time_ms=kernel_info.duration_ms
            )))
            
            # Update metrics
            if self.metrics:
                tags = {"device_id": str(device_id), "kernel_name": kernel_name}
                if component:
                    tags["component"] = component
                    
                self.metrics.increment("gpu_kernel_executions_total", tags=tags)
                self.metrics.record("gpu_kernel_duration_ms", kernel_info.duration_ms, tags=tags)

    # Event handlers
    async def _handle_processing_started(self, event) -> None:
        """Handle processing started events."""
        try:
            component = getattr(event, 'component', None) or 'unknown'
            session_id = getattr(event, 'session_id', None)
            
            # Initialize component GPU tracking if not exists
            if component not in self.component_gpu_usage:
                self.component_gpu_usage[component] = defaultdict(float)
            
            # Record baseline GPU usage for this component
            for device_id in self.gpu_devices:
                if device_id in self.metrics_history and self.metrics_history[device_id]:
                    latest_metrics = self.metrics_history[device_id][-1]
                    baseline_memory = latest_metrics.memory_used_mb
                    
                    # Store baseline for tracking delta
                    self.component_gpu_usage[component][f"baseline_{device_id}"] = baseline_memory
            
            self.logger.debug(f"Started GPU tracking for component: {component}")
            
        except Exception as e:
            self.logger.error(f"Failed to handle processing started event: {str(e)}")

    async def _handle_processing_completed(self, event) -> None:
        """Handle processing completed events."""
        try:
            component = getattr(event, 'component', None) or 'unknown'
            session_id = getattr(event, 'session_id', None)
            processing_time = getattr(event, 'processing_time', 0.0)
            
            # Calculate GPU usage delta for this component
            for device_id in self.gpu_devices:
                if device_id in self.metrics_history and self.metrics_history[device_id]:
                    latest_metrics = self.metrics_history[device_id][-1]
                    current_memory = latest_metrics.memory_used_mb
                    
                    baseline_key = f"baseline_{device_id}"
                    if baseline_key in self.component_gpu_usage[component]:
                        baseline_memory = self.component_gpu_usage[component][baseline_key]
                        memory_delta = current_memory - baseline_memory
                        
                        # Update cumulative usage (only positive deltas to avoid negative from cleanup)
                        if memory_delta > 0:
                            self.component_gpu_usage[component][device_id] += memory_delta
                        
                        # Update metrics
                        if self.metrics:
                            self.metrics.set("gpu_component_memory_usage_mb", 
                                           self.component_gpu_usage[component][device_id],
                                           tags={
                                               "component": component,
                                               "device_id": str(device_id)
                                           })
                        
                        # Clean up baseline
                        del self.component_gpu_usage[component][baseline_key]
            
            self.logger.debug(f"Completed GPU tracking for component: {component}, duration: {processing_time:.2f}s")
            
        except Exception as e:
            self.logger.error(f"Failed to handle processing completed event: {str(e)}")

    async def _handle_workflow_started(self, event) -> None:
        """Handle workflow started events."""
        try:
            workflow_id = getattr(event, 'workflow_id', None)
            
            if workflow_id:
                # Track workflow-level GPU usage
                workflow_key = f"workflow_{workflow_id}"
                self.component_gpu_usage[workflow_key] = defaultdict(float)
                
                # Record initial GPU state
                for device_id in self.gpu_devices:
                    if device_id in self.metrics_history and self.metrics_history[device_id]:
                        latest_metrics = self.metrics_history[device_id][-1]
                        self.component_gpu_usage[workflow_key][f"start_{device_id}"] = latest_metrics.memory_used_mb
                
                self.logger.debug(f"Started GPU tracking for workflow: {workflow_id}")
                
        except Exception as e:
            self.logger.error(f"Failed to handle workflow started event: {str(e)}")

    async def _handle_workflow_completed(self, event) -> None:
        """Handle workflow completed events."""
        try:
            workflow_id = getattr(event, 'workflow_id', None)
            execution_time = getattr(event, 'execution_time', 0.0)
            
            if workflow_id:
                workflow_key = f"workflow_{workflow_id}"
                
                # Calculate total GPU usage for workflow
                for device_id in self.gpu_devices:
                    start_key = f"start_{device_id}"
                    if (workflow_key in self.component_gpu_usage and 
                        start_key in self.component_gpu_usage[workflow_key]):
                        
                        if device_id in self.metrics_history and self.metrics_history[device_id]:
                            latest_metrics = self.metrics_history[device_id][-1]
                            start_memory = self.component_gpu_usage[workflow_key][start_key]
                            peak_memory = latest_metrics.memory_used_mb
                            
                            workflow_memory_usage = max(0, peak_memory - start_memory)
                            self.component_gpu_usage[workflow_key][device_id] = workflow_memory_usage
                            
                            # Update metrics
                            if self.metrics:
                                self.metrics.set("gpu_component_memory_usage_mb", 
                                               workflow_memory_usage,
                                               tags={
                                                   "component": workflow_key,
                                                   "device_id": str(device_id)
                                               })
                
                self.logger.debug(f"Completed GPU tracking for workflow: {workflow_id}, duration: {execution_time:.2f}s")
                
        except Exception as e:
            self.logger.error(f"Failed to handle workflow completed event: {str(e)}")

    async def _handle_workflow_step_started(self, event) -> None:
        """Handle workflow step started events."""
        try:
            workflow_id = getattr(event, 'workflow_id', None)
            step_id = getattr(event, 'step_id', None)
            
            if workflow_id and step_id:
                step_key = f"step_{workflow_id}_{step_id}"
                self.component_gpu_usage[step_key] = defaultdict(float)
                
                # Record step start GPU state
                for device_id in self.gpu_devices:
                    if device_id in self.metrics_history and self.metrics_history[device_id]:
                        latest_metrics = self.metrics_history[device_id][-1]
                        self.component_gpu_usage[step_key][f"start_{device_id}"] = latest_metrics.memory_used_mb
                
                self.logger.debug(f"Started GPU tracking for workflow step: {workflow_id}/{step_id}")
                
        except Exception as e:
            self.logger.error(f"Failed to handle workflow step started event: {str(e)}")

    async def _handle_workflow_step_completed(self, event) -> None:
        """Handle workflow step completed events."""
        try:
            workflow_id = getattr(event, 'workflow_id', None)
            step_id = getattr(event, 'step_id', None)
            step_duration = getattr(event, 'step_duration', 0.0)
            
            if workflow_id and step_id:
                step_key = f"step_{workflow_id}_{step_id}"
                
                # Calculate step GPU usage
                for device_id in self.gpu_devices:
                    start_key = f"start_{device_id}"
                    if (step_key in self.component_gpu_usage and 
                        start_key in self.component_gpu_usage[step_key]):
                        
                        if device_id in self.metrics_history and self.metrics_history[device_id]:
                            latest_metrics = self.metrics_history[device_id][-1]
                            start_memory = self.component_gpu_usage[step_key][start_key]
                            end_memory = latest_metrics.memory_used_mb
                            
                            step_memory_usage = max(0, end_memory - start_memory)
                            self.component_gpu_usage[step_key][device_id] = step_memory_usage
                            
                            # Update metrics
                            if self.metrics:
                                self.metrics.set("gpu_component_memory_usage_mb", 
                                               step_memory_usage,
                                               tags={
                                                   "component": step_key,
                                                   "device_id": str(device_id),
                                                   "workflow_id": workflow_id,
                                                   "step_id": step_id
                                               })
                
                self.logger.debug(f"Completed GPU tracking for workflow step: {workflow_id}/{step_id}, duration: {step_duration:.2f}s")
                
        except Exception as e:
            self.logger.error(f"Failed to handle workflow step completed event: {str(e)}")

    async def _handle_component_started(self, event) -> None:
        """Handle component started events."""
        try:
            component_id = getattr(event, 'component_id', None)
            
            if component_id:
                # Initialize component GPU tracking
                if component_id not in self.component_gpu_usage:
                    self.component_gpu_usage[component_id] = defaultdict(float)
                
                self.logger.debug(f"Initialized GPU tracking for component: {component_id}")
                
        except Exception as e:
            self.logger.error(f"Failed to handle component started event: {str(e)}")

    async def _handle_component_stopped(self, event) -> None:
        """Handle component stopped events."""
        try:
            component_id = getattr(event, 'component_id', None)
            
            if component_id and component_id in self.component_gpu_usage:
                # Log final component GPU usage
                total_usage = sum(
                    usage for key, usage in self.component_gpu_usage[component_id].items()
                    if not key.startswith(('baseline_', 'start_'))
                )
                
                self.logger.info(f"Component {component_id} final GPU usage: {total_usage:.2f} MB across all devices")
                
                # Keep component data for historical analysis but mark as stopped
                self.component_gpu_usage[f"{component_id}_stopped"] = self.component_gpu_usage[component_id]
                del self.component_gpu_usage[component_id]
                
        except Exception as e:
            self.logger.error(f"Failed to handle component stopped event: {str(e)}")

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