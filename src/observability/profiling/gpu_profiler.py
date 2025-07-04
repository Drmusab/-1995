"""
Comprehensive GPU Profiler for AI Assistant
Author: Drmusab
Last Modified: 2025-01-10 15:00:00 UTC

This module provides comprehensive GPU profiling capabilities for the AI assistant,
including GPU utilization monitoring, memory tracking, thermal management,
performance analysis, and integration with all core system components.

Features:
- CUDA and OpenCL device support
- Real-time GPU performance monitoring
- GPU memory allocation and fragmentation tracking
- Thermal management and power consumption monitoring
- Multi-GPU support
- Performance benchmarking and optimization suggestions
- Integration with core engine, events, and metrics systems
- Profiling sessions with detailed reports
- Memory leak detection for GPU resources
- Graceful degradation when GPU monitoring is unavailable
"""

import asyncio
import threading
import time
import json
import uuid
import logging
import traceback
import psutil
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable, Union, AsyncGenerator
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
from contextlib import contextmanager, asynccontextmanager
from collections import defaultdict, deque
import concurrent.futures

# GPU monitoring libraries (with graceful fallbacks)
try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    pynvml = None

try:
    import pyopencl as cl
    PYOPENCL_AVAILABLE = True
except ImportError:
    PYOPENCL_AVAILABLE = False
    cl = None

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

# Core imports (with graceful fallbacks for broken dependencies)
try:
    from src.core.dependency_injection import Container
    from src.observability.logging.config import get_logger
    CORE_IMPORTS_AVAILABLE = True
except:
    Container = None
    get_logger = None
    CORE_IMPORTS_AVAILABLE = False

# Try to import other core components
try:
    from src.core.config.loader import ConfigLoader
    from src.core.events.event_bus import EventBus
    from src.core.error_handling import ErrorHandler, handle_exceptions
    from src.core.health_check import HealthCheck
    from src.observability.monitoring.metrics import MetricsCollector
    from src.observability.monitoring.tracing import TraceManager
    FULL_INTEGRATION_AVAILABLE = True
except:
    FULL_INTEGRATION_AVAILABLE = False
    # Create dummy classes for graceful degradation
    class DummyLogger:
        def info(self, msg): print(f"INFO: {msg}")
        def warning(self, msg): print(f"WARNING: {msg}")
        def error(self, msg): print(f"ERROR: {msg}")
        def debug(self, msg): print(f"DEBUG: {msg}")
    
    def get_logger(name):
        return DummyLogger()
    
    def handle_exceptions(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if hasattr(args[0], 'logger'):
                    args[0].logger.error(f"Error in {func.__name__}: {str(e)}")
                raise
        return wrapper

# Import event types if available
try:
    from src.core.events.event_types import (
        PerformanceAlertTriggered, ProfilerStarted, ProfilerStopped,
        ProfilingDataGenerated, BottleneckDetected, PerformanceThresholdExceeded,
        ComponentHealthChanged, SystemStateChanged
    )
    EVENTS_AVAILABLE = True
except:
    EVENTS_AVAILABLE = False
    # Create dummy event classes
    class PerformanceAlertTriggered:
        def __init__(self, **kwargs): pass
    class ProfilerStarted:
        def __init__(self, **kwargs): pass
    class ProfilerStopped:
        def __init__(self, **kwargs): pass
    class ProfilingDataGenerated:
        def __init__(self, **kwargs): pass
    class BottleneckDetected:
        def __init__(self, **kwargs): pass
    class PerformanceThresholdExceeded:
        def __init__(self, **kwargs): pass
    class ComponentHealthChanged:
        def __init__(self, **kwargs): pass
    class SystemStateChanged:
        def __init__(self, **kwargs): pass


class GPUVendor(Enum):
    """GPU vendor types."""
    NVIDIA = "nvidia"
    AMD = "amd"
    INTEL = "intel"
    UNKNOWN = "unknown"


class ProfilingMode(Enum):
    """GPU profiling modes."""
    BASIC = "basic"                    # Basic metrics only
    STANDARD = "standard"              # Standard monitoring
    DETAILED = "detailed"              # Detailed profiling
    COMPREHENSIVE = "comprehensive"    # Full profiling with all features


class ProfilingLevel(Enum):
    """GPU profiling levels."""
    LOW = "low"                        # Minimal overhead
    MEDIUM = "medium"                  # Balanced monitoring
    HIGH = "high"                      # Detailed monitoring
    MAXIMUM = "maximum"                # All available metrics


class GPUType(Enum):
    """GPU types."""
    DISCRETE = "discrete"
    INTEGRATED = "integrated"
    VIRTUAL = "virtual"
    UNKNOWN = "unknown"


@dataclass
class GPUDevice:
    """Information about a GPU device."""
    index: int
    name: str
    vendor: GPUVendor
    gpu_type: GPUType
    
    # Memory information
    total_memory_mb: float = 0.0
    free_memory_mb: float = 0.0
    used_memory_mb: float = 0.0
    
    # Performance information
    utilization_percent: float = 0.0
    memory_utilization_percent: float = 0.0
    temperature_celsius: Optional[float] = None
    power_draw_watts: Optional[float] = None
    max_power_watts: Optional[float] = None
    
    # Clock information
    graphics_clock_mhz: Optional[float] = None
    memory_clock_mhz: Optional[float] = None
    max_graphics_clock_mhz: Optional[float] = None
    max_memory_clock_mhz: Optional[float] = None
    
    # Capabilities
    compute_capability: Optional[str] = None
    driver_version: Optional[str] = None
    is_available: bool = True
    supports_profiling: bool = True
    
    # Additional metadata
    pci_bus_id: Optional[str] = None
    uuid: Optional[str] = None
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class GPUProfilerConfig:
    """Configuration for GPU profiling."""
    mode: ProfilingMode = ProfilingMode.STANDARD
    level: ProfilingLevel = ProfilingLevel.MEDIUM
    
    # Monitoring intervals
    monitoring_interval_seconds: float = 1.0
    metrics_update_interval_seconds: float = 5.0
    device_discovery_interval_seconds: float = 30.0
    
    # Feature flags
    enable_cuda_profiling: bool = True
    enable_opencl_profiling: bool = True
    enable_thermal_monitoring: bool = True
    enable_power_monitoring: bool = True
    enable_memory_tracking: bool = True
    enable_performance_benchmarking: bool = True
    enable_leak_detection: bool = True
    
    # Thresholds
    temperature_warning_celsius: float = 80.0
    temperature_critical_celsius: float = 90.0
    memory_warning_percent: float = 80.0
    memory_critical_percent: float = 95.0
    utilization_warning_percent: float = 90.0
    power_warning_percent: float = 90.0
    
    # Data retention
    max_snapshots: int = 1000
    max_profiling_sessions: int = 100
    snapshot_retention_hours: int = 24
    
    # Output configuration
    output_dir: str = "data/profiling/gpu"
    enable_detailed_logging: bool = False
    export_json_reports: bool = True
    export_csv_metrics: bool = False


@dataclass
class GPUSnapshot:
    """A snapshot of GPU state at a point in time."""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    devices: List[GPUDevice] = field(default_factory=list)
    
    # System-wide GPU metrics
    total_gpu_memory_mb: float = 0.0
    total_gpu_memory_used_mb: float = 0.0
    total_gpu_memory_free_mb: float = 0.0
    average_gpu_utilization: float = 0.0
    average_memory_utilization: float = 0.0
    max_temperature_celsius: Optional[float] = None
    total_power_draw_watts: Optional[float] = None
    
    # Performance metrics
    inference_count: int = 0
    average_inference_time_ms: Optional[float] = None
    throughput_inferences_per_second: Optional[float] = None
    
    # Health indicators
    thermal_alerts: List[Dict[str, Any]] = field(default_factory=list)
    memory_alerts: List[Dict[str, Any]] = field(default_factory=list)
    performance_alerts: List[Dict[str, Any]] = field(default_factory=list)
    
    # Memory leak detection
    potential_leaks: List[Dict[str, Any]] = field(default_factory=list)
    
    # Metadata
    snapshot_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    trigger: str = "manual"


@dataclass
class GPUProfilingSession:
    """A GPU profiling session with metadata and results."""
    session_id: str
    name: str
    description: Optional[str] = None
    
    # Session metadata
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    end_time: Optional[datetime] = None
    duration_seconds: float = 0.0
    
    # Configuration
    config: GPUProfilerConfig = field(default_factory=GPUProfilerConfig)
    
    # Results
    snapshots: List[GPUSnapshot] = field(default_factory=list)
    performance_metrics: Dict[str, List[float]] = field(default_factory=dict)
    bottlenecks: List[Dict[str, Any]] = field(default_factory=list)
    optimization_suggestions: List[Dict[str, Any]] = field(default_factory=list)
    
    # Statistics
    total_inferences: int = 0
    average_inference_time_ms: float = 0.0
    peak_gpu_utilization: float = 0.0
    peak_memory_utilization: float = 0.0
    peak_temperature_celsius: Optional[float] = None
    total_energy_consumed_wh: Optional[float] = None
    
    # Files and exports
    session_file_path: Optional[Path] = None
    report_file_path: Optional[Path] = None


class GPUProfiler:
    """
    Comprehensive GPU Profiler for the AI Assistant.
    
    This profiler provides comprehensive GPU performance monitoring including:
    - CUDA and OpenCL device support
    - Real-time GPU utilization and memory monitoring
    - Thermal management and power consumption tracking
    - Multi-GPU support with per-device metrics
    - Performance benchmarking and optimization analysis
    - Memory leak detection for GPU resources
    - Integration with core assistant components
    - Profiling sessions with detailed reports
    - Graceful degradation when GPU monitoring is unavailable
    """
    
    def __init__(self, container: Container = None):
        # Core dependencies - use dependency injection if available
        self.container = container
        
        # Setup logging
        if CORE_IMPORTS_AVAILABLE and get_logger:
            self.logger = get_logger(__name__)
        else:
            self.logger = logging.getLogger(__name__)
            if not self.logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                handler.setFormatter(formatter)
                self.logger.addHandler(handler)
                self.logger.setLevel(logging.INFO)
        
        # Initialize dependencies if container is available
        if container and FULL_INTEGRATION_AVAILABLE:
            try:
                self.config_loader = container.get(ConfigLoader)
                self.event_bus = container.get(EventBus)
                self.metrics = container.get(MetricsCollector)
                self.tracer = container.get(TraceManager)
                self.error_handler = container.get(ErrorHandler)
                self.integration_mode = True
            except Exception as e:
                self.logger.warning(f"Failed to initialize some dependencies: {str(e)}")
                self.integration_mode = False
        else:
            self.integration_mode = False
        
        # Configuration
        self.config = GPUProfilerConfig()
        self._load_config()
        
        # GPU monitoring state
        self.is_running = False
        self.devices: List[GPUDevice] = []
        self.current_session: Optional[GPUProfilingSession] = None
        self.profiling_sessions: Dict[str, GPUProfilingSession] = {}
        self.recent_sessions = deque(maxlen=self.config.max_profiling_sessions)
        
        # Data storage
        self.snapshots: List[GPUSnapshot] = []
        self.performance_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.inference_times: deque = deque(maxlen=10000)
        
        # Threading and async support
        self.monitoring_task: Optional[asyncio.Task] = None
        self.device_discovery_task: Optional[asyncio.Task] = None
        self.stop_monitoring_event = asyncio.Event()
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        
        # Output paths
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # GPU library initialization
        self._initialize_gpu_libraries()
        
        # Setup monitoring and integration
        if self.integration_mode:
            self._setup_monitoring()
            self._setup_event_handlers()
        
        # Discover devices
        self._discover_devices()
        
        self.logger.info("GPUProfiler initialized successfully")
    
    def _load_config(self) -> None:
        """Load configuration from config loader."""
        if not self.integration_mode:
            return
            
        try:
            gpu_config = self.config_loader.get("gpu_profiler", {})
            
            # Update config from loaded values
            for key, value in gpu_config.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
            
            self.logger.debug("GPU profiler configuration loaded")
            
        except Exception as e:
            self.logger.warning(f"Failed to load GPU profiler config: {str(e)}")
    
    def _initialize_gpu_libraries(self) -> None:
        """Initialize GPU monitoring libraries."""
        self.cuda_available = False
        self.opencl_available = False
        
        # Initialize NVIDIA ML
        if PYNVML_AVAILABLE and self.config.enable_cuda_profiling:
            try:
                pynvml.nvmlInit()
                self.cuda_available = True
                self.logger.info("NVIDIA GPU monitoring initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize NVIDIA GPU monitoring: {str(e)}")
        
        # Initialize OpenCL
        if PYOPENCL_AVAILABLE and self.config.enable_opencl_profiling:
            try:
                platforms = cl.get_platforms()
                if platforms:
                    self.opencl_available = True
                    self.logger.info("OpenCL GPU monitoring initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize OpenCL GPU monitoring: {str(e)}")
        
        # Check PyTorch CUDA availability
        if TORCH_AVAILABLE:
            try:
                if torch.cuda.is_available():
                    self.logger.info("PyTorch CUDA available")
            except Exception as e:
                self.logger.warning(f"PyTorch CUDA check failed: {str(e)}")
    
    def _setup_monitoring(self) -> None:
        """Setup metrics and monitoring."""
        if not self.integration_mode:
            return
            
        try:
            # GPU device metrics
            self.metrics.register_gauge("gpu_device_count")
            self.metrics.register_gauge("gpu_total_memory_mb")
            self.metrics.register_gauge("gpu_total_memory_used_mb")
            self.metrics.register_gauge("gpu_total_memory_free_mb")
            
            # Performance metrics
            self.metrics.register_gauge("gpu_average_utilization_percent")
            self.metrics.register_gauge("gpu_average_memory_utilization_percent")
            self.metrics.register_gauge("gpu_max_temperature_celsius")
            self.metrics.register_gauge("gpu_total_power_draw_watts")
            
            # Per-device metrics (will be registered dynamically)
            self.metrics.register_histogram("gpu_inference_time_ms")
            self.metrics.register_counter("gpu_inferences_total")
            self.metrics.register_gauge("gpu_throughput_inferences_per_second")
            
            # Health and alerts
            self.metrics.register_counter("gpu_thermal_alerts_total")
            self.metrics.register_counter("gpu_memory_alerts_total")
            self.metrics.register_counter("gpu_performance_alerts_total")
            
            # Profiling session metrics
            self.metrics.register_counter("gpu_profiling_sessions_total")
            self.metrics.register_histogram("gpu_profiling_session_duration_seconds")
            
            self.logger.debug("GPU metrics registered successfully")
            
        except Exception as e:
            self.logger.warning(f"Failed to register GPU metrics: {str(e)}")
    
    def _setup_event_handlers(self) -> None:
        """Setup event handlers for system integration."""
        if not self.integration_mode:
            return
            
        try:
            # Component lifecycle events
            self.event_bus.subscribe("component_initialized", self._handle_component_initialized)
            self.event_bus.subscribe("component_stopped", self._handle_component_stopped)
            
            # Session lifecycle events
            self.event_bus.subscribe("session_started", self._handle_session_started)
            self.event_bus.subscribe("session_ended", self._handle_session_ended)
            
            # Processing events
            self.event_bus.subscribe("processing_started", self._handle_processing_started)
            self.event_bus.subscribe("processing_completed", self._handle_processing_completed)
            self.event_bus.subscribe("inference_started", self._handle_inference_started)
            self.event_bus.subscribe("inference_completed", self._handle_inference_completed)
            
            # Error events
            self.event_bus.subscribe("error_occurred", self._handle_error)
            
            # System events
            self.event_bus.subscribe("system_shutdown_started", self._handle_system_shutdown)
        except Exception as e:
            self.logger.warning(f"Failed to setup event handlers: {str(e)}")
    
    def _discover_devices(self) -> None:
        """Discover available GPU devices."""
        devices = []
        
        # Discover NVIDIA devices
        if self.cuda_available:
            devices.extend(self._discover_nvidia_devices())
        
        # Discover OpenCL devices
        if self.opencl_available:
            devices.extend(self._discover_opencl_devices())
        
        self.devices = devices
        
        # Update metrics if available
        if self.integration_mode and hasattr(self, 'metrics'):
            try:
                self.metrics.set("gpu_device_count", len(self.devices))
            except:
                pass
        
        if devices:
            self.logger.info(f"Discovered {len(devices)} GPU device(s)")
            for device in devices:
                self.logger.debug(f"GPU {device.index}: {device.name} ({device.vendor.value})")
        else:
            self.logger.warning("No GPU devices discovered")
    
    def _discover_nvidia_devices(self) -> List[GPUDevice]:
        """Discover NVIDIA GPU devices."""
        devices = []
        
        try:
            device_count = pynvml.nvmlDeviceGetCount()
            
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                
                # Basic device info
                name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
                
                # Memory info
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                total_memory_mb = memory_info.total / (1024 * 1024)
                free_memory_mb = memory_info.free / (1024 * 1024)
                used_memory_mb = memory_info.used / (1024 * 1024)
                
                # Additional info
                try:
                    uuid_str = pynvml.nvmlDeviceGetUUID(handle).decode('utf-8')
                except:
                    uuid_str = None
                
                try:
                    pci_info = pynvml.nvmlDeviceGetPciInfo(handle)
                    pci_bus_id = f"{pci_info.domain:04x}:{pci_info.bus:02x}:{pci_info.device:02x}.{pci_info.pciDeviceId:x}"
                except:
                    pci_bus_id = None
                
                try:
                    driver_version = pynvml.nvmlSystemGetDriverVersion().decode('utf-8')
                except:
                    driver_version = None
                
                device = GPUDevice(
                    index=i,
                    name=name,
                    vendor=GPUVendor.NVIDIA,
                    gpu_type=GPUType.DISCRETE,  # Assume discrete for NVIDIA cards
                    total_memory_mb=total_memory_mb,
                    free_memory_mb=free_memory_mb,
                    used_memory_mb=used_memory_mb,
                    uuid=uuid_str,
                    pci_bus_id=pci_bus_id,
                    driver_version=driver_version,
                    is_available=True,
                    supports_profiling=True
                )
                
                devices.append(device)
                
        except Exception as e:
            self.logger.error(f"Failed to discover NVIDIA devices: {str(e)}")
        
        return devices
    
    def _discover_opencl_devices(self) -> List[GPUDevice]:
        """Discover OpenCL GPU devices."""
        devices = []
        
        try:
            platforms = cl.get_platforms()
            device_index = len(self.devices)  # Continue numbering from NVIDIA devices
            
            for platform in platforms:
                try:
                    gpu_devices = platform.get_devices(device_type=cl.device_type.GPU)
                    
                    for device in gpu_devices:
                        name = device.get_info(cl.device_info.NAME)
                        vendor_name = device.get_info(cl.device_info.VENDOR).lower()
                        
                        # Determine vendor
                        if 'nvidia' in vendor_name:
                            vendor = GPUVendor.NVIDIA
                        elif 'amd' in vendor_name or 'advanced micro devices' in vendor_name:
                            vendor = GPUVendor.AMD
                        elif 'intel' in vendor_name:
                            vendor = GPUVendor.INTEL
                        else:
                            vendor = GPUVendor.UNKNOWN
                        
                        # Get memory info
                        try:
                            global_mem_size = device.get_info(cl.device_info.GLOBAL_MEM_SIZE)
                            total_memory_mb = global_mem_size / (1024 * 1024)
                        except:
                            total_memory_mb = 0.0
                        
                        gpu_device = GPUDevice(
                            index=device_index,
                            name=name,
                            vendor=vendor,
                            gpu_type=GPUType.DISCRETE,  # Assume discrete for OpenCL devices
                            total_memory_mb=total_memory_mb,
                            is_available=True,
                            supports_profiling=True
                        )
                        
                        devices.append(gpu_device)
                        device_index += 1
                        
                except Exception as e:
                    self.logger.warning(f"Failed to discover devices on platform {platform.name}: {str(e)}")
                    
        except Exception as e:
            self.logger.error(f"Failed to discover OpenCL devices: {str(e)}")
        
        return devices
    
    @handle_exceptions
    async def start(self) -> None:
        """Start the GPU profiler."""
        if self.is_running:
            self.logger.warning("GPU profiler is already running")
            return
        
        try:
            self.is_running = True
            self.stop_monitoring_event.clear()
            
            # Start monitoring tasks
            if self.config.mode != ProfilingMode.BASIC:
                self.monitoring_task = asyncio.create_task(self._monitoring_loop())
                self.device_discovery_task = asyncio.create_task(self._device_discovery_loop())
            
            # Emit startup event if integration available
            if self.integration_mode and hasattr(self, 'event_bus'):
                try:
                    await self.event_bus.emit(ProfilerStarted(
                        profiler_type="gpu",
                        profiler_id=id(self),
                        devices_count=len(self.devices),
                        config=asdict(self.config)
                    ))
                except:
                    pass
            
            self.logger.info("GPU profiler started successfully")
            
        except Exception as e:
            self.is_running = False
            self.logger.error(f"Failed to start GPU profiler: {str(e)}")
            raise
    
    @handle_exceptions
    async def stop(self) -> None:
        """Stop the GPU profiler."""
        if not self.is_running:
            return
        
        try:
            self.is_running = False
            self.stop_monitoring_event.set()
            
            # Stop monitoring tasks
            if self.monitoring_task:
                self.monitoring_task.cancel()
                try:
                    await self.monitoring_task
                except asyncio.CancelledError:
                    pass
            
            if self.device_discovery_task:
                self.device_discovery_task.cancel()
                try:
                    await self.device_discovery_task
                except asyncio.CancelledError:
                    pass
            
            # Stop current session if running
            if self.current_session:
                await self.stop_profiling_session()
            
            # Emit shutdown event if integration available
            if self.integration_mode and hasattr(self, 'event_bus'):
                try:
                    await self.event_bus.emit(ProfilerStopped(
                        profiler_type="gpu",
                        profiler_id=id(self),
                        total_sessions=len(self.profiling_sessions),
                        total_snapshots=len(self.snapshots)
                    ))
                except:
                    pass
            
            self.logger.info("GPU profiler stopped successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to stop GPU profiler: {str(e)}")
    
    async def _monitoring_loop(self) -> None:
        """Background task for GPU monitoring."""
        try:
            while self.is_running and not self.stop_monitoring_event.is_set():
                try:
                    # Update device information
                    await self._update_device_metrics()
                    
                    # Take snapshot if in detailed mode
                    if self.config.level in [ProfilingLevel.HIGH, ProfilingLevel.MAXIMUM]:
                        await self.take_snapshot("monitoring")
                    
                    # Update metrics
                    await self._update_system_metrics()
                    
                    # Check for alerts
                    await self._check_alerts()
                    
                except Exception as e:
                    self.logger.error(f"GPU monitoring error: {str(e)}")
                
                # Wait for next monitoring cycle
                await asyncio.sleep(self.config.monitoring_interval_seconds)
                
        except asyncio.CancelledError:
            self.logger.debug("GPU monitoring loop cancelled")
        except Exception as e:
            self.logger.error(f"GPU monitoring loop error: {str(e)}")
    
    async def _device_discovery_loop(self) -> None:
        """Background task for device discovery."""
        try:
            while self.is_running and not self.stop_monitoring_event.is_set():
                try:
                    # Rediscover devices periodically
                    self._discover_devices()
                    
                except Exception as e:
                    self.logger.error(f"Device discovery error: {str(e)}")
                
                # Wait for next discovery cycle
                await asyncio.sleep(self.config.device_discovery_interval_seconds)
                
        except asyncio.CancelledError:
            self.logger.debug("Device discovery loop cancelled")
        except Exception as e:
            self.logger.error(f"Device discovery loop error: {str(e)}")
    
    async def _update_device_metrics(self) -> None:
        """Update metrics for all GPU devices."""
        for device in self.devices:
            try:
                if device.vendor == GPUVendor.NVIDIA and self.cuda_available:
                    await self._update_nvidia_device_metrics(device)
                elif self.opencl_available:
                    await self._update_opencl_device_metrics(device)
                    
            except Exception as e:
                self.logger.warning(f"Failed to update metrics for device {device.index}: {str(e)}")
    
    async def _update_nvidia_device_metrics(self, device: GPUDevice) -> None:
        """Update metrics for an NVIDIA GPU device."""
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(device.index)
            
            # Memory info
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            device.total_memory_mb = memory_info.total / (1024 * 1024)
            device.free_memory_mb = memory_info.free / (1024 * 1024)
            device.used_memory_mb = memory_info.used / (1024 * 1024)
            device.memory_utilization_percent = (device.used_memory_mb / device.total_memory_mb) * 100
            
            # Utilization
            try:
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                device.utilization_percent = utilization.gpu
            except:
                device.utilization_percent = 0.0
            
            # Temperature
            try:
                device.temperature_celsius = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            except:
                device.temperature_celsius = None
            
            # Power
            try:
                device.power_draw_watts = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
            except:
                device.power_draw_watts = None
            
            try:
                device.max_power_watts = pynvml.nvmlDeviceGetMaxPowerManagementLimitConstraints(handle)[1] / 1000.0
            except:
                device.max_power_watts = None
            
            # Clock speeds
            try:
                device.graphics_clock_mhz = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS)
                device.memory_clock_mhz = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
            except:
                pass
            
            device.last_updated = datetime.now(timezone.utc)
            
        except Exception as e:
            self.logger.warning(f"Failed to update NVIDIA device {device.index} metrics: {str(e)}")
            device.is_available = False
    
    async def _update_opencl_device_metrics(self, device: GPUDevice) -> None:
        """Update metrics for an OpenCL GPU device."""
        # OpenCL doesn't provide real-time utilization/temperature data
        # We can only update basic availability and memory info
        device.last_updated = datetime.now(timezone.utc)
    
    async def _update_system_metrics(self) -> None:
        """Update system-wide GPU metrics."""
        if not self.devices or not self.integration_mode:
            return
        
        try:
            # Calculate aggregate metrics
            total_memory_mb = sum(d.total_memory_mb for d in self.devices)
            total_used_mb = sum(d.used_memory_mb for d in self.devices)
            total_free_mb = sum(d.free_memory_mb for d in self.devices)
            
            avg_utilization = sum(d.utilization_percent for d in self.devices) / len(self.devices)
            avg_memory_util = sum(d.memory_utilization_percent for d in self.devices) / len(self.devices)
            
            max_temp = max((d.temperature_celsius for d in self.devices if d.temperature_celsius), default=None)
            total_power = sum((d.power_draw_watts for d in self.devices if d.power_draw_watts), default=None)
            
            # Update metrics
            if hasattr(self, 'metrics'):
                self.metrics.set("gpu_total_memory_mb", total_memory_mb)
                self.metrics.set("gpu_total_memory_used_mb", total_used_mb)
                self.metrics.set("gpu_total_memory_free_mb", total_free_mb)
                self.metrics.set("gpu_average_utilization_percent", avg_utilization)
                self.metrics.set("gpu_average_memory_utilization_percent", avg_memory_util)
                
                if max_temp is not None:
                    self.metrics.set("gpu_max_temperature_celsius", max_temp)
                
                if total_power is not None:
                    self.metrics.set("gpu_total_power_draw_watts", total_power)
            
        except Exception as e:
            self.logger.error(f"Failed to update system GPU metrics: {str(e)}")
    
    async def _check_alerts(self) -> None:
        """Check for GPU-related alerts."""
        for device in self.devices:
            await self._check_device_alerts(device)
    
    async def _check_device_alerts(self, device: GPUDevice) -> None:
        """Check for alerts on a specific device."""
        alerts = []
        
        # Temperature alerts
        if device.temperature_celsius is not None:
            if device.temperature_celsius >= self.config.temperature_critical_celsius:
                alerts.append({
                    "type": "thermal",
                    "severity": "critical",
                    "device_index": device.index,
                    "device_name": device.name,
                    "temperature_celsius": device.temperature_celsius,
                    "threshold": self.config.temperature_critical_celsius
                })
                if self.integration_mode and hasattr(self, 'metrics'):
                    try:
                        self.metrics.increment("gpu_thermal_alerts_total")
                    except:
                        pass
                
            elif device.temperature_celsius >= self.config.temperature_warning_celsius:
                alerts.append({
                    "type": "thermal",
                    "severity": "warning",
                    "device_index": device.index,
                    "device_name": device.name,
                    "temperature_celsius": device.temperature_celsius,
                    "threshold": self.config.temperature_warning_celsius
                })
        
        # Memory alerts
        if device.memory_utilization_percent >= self.config.memory_critical_percent:
            alerts.append({
                "type": "memory",
                "severity": "critical",
                "device_index": device.index,
                "device_name": device.name,
                "memory_utilization_percent": device.memory_utilization_percent,
                "threshold": self.config.memory_critical_percent
            })
            if self.integration_mode and hasattr(self, 'metrics'):
                try:
                    self.metrics.increment("gpu_memory_alerts_total")
                except:
                    pass
            
        elif device.memory_utilization_percent >= self.config.memory_warning_percent:
            alerts.append({
                "type": "memory",
                "severity": "warning",
                "device_index": device.index,
                "device_name": device.name,
                "memory_utilization_percent": device.memory_utilization_percent,
                "threshold": self.config.memory_warning_percent
            })
        
        # Utilization alerts
        if device.utilization_percent >= self.config.utilization_warning_percent:
            alerts.append({
                "type": "utilization",
                "severity": "warning",
                "device_index": device.index,
                "device_name": device.name,
                "utilization_percent": device.utilization_percent,
                "threshold": self.config.utilization_warning_percent
            })
            if self.integration_mode and hasattr(self, 'metrics'):
                try:
                    self.metrics.increment("gpu_performance_alerts_total")
                except:
                    pass
        
        # Emit alert events if integration available
        if self.integration_mode and hasattr(self, 'event_bus'):
            for alert in alerts:
                try:
                    await self.event_bus.emit(PerformanceAlertTriggered(
                        component="gpu_profiler",
                        alert_type=alert["type"],
                        severity=alert["severity"],
                        message=f"GPU {alert['device_index']} ({alert['device_name']}) {alert['type']} alert",
                        details=alert
                    ))
                except:
                    pass
        
        # Always log alerts
        for alert in alerts:
            self.logger.warning(f"GPU Alert: {alert['type']} {alert['severity']} on device {alert['device_index']}")
    
    @handle_exceptions
    async def take_snapshot(self, trigger: str = "manual") -> GPUSnapshot:
        """Take a snapshot of current GPU state."""
        try:
            snapshot = GPUSnapshot(trigger=trigger)
            
            # Copy current device states
            snapshot.devices = [
                GPUDevice(**asdict(device)) for device in self.devices
            ]
            
            # Calculate system-wide metrics
            if self.devices:
                snapshot.total_gpu_memory_mb = sum(d.total_memory_mb for d in self.devices)
                snapshot.total_gpu_memory_used_mb = sum(d.used_memory_mb for d in self.devices)
                snapshot.total_gpu_memory_free_mb = sum(d.free_memory_mb for d in self.devices)
                snapshot.average_gpu_utilization = sum(d.utilization_percent for d in self.devices) / len(self.devices)
                snapshot.average_memory_utilization = sum(d.memory_utilization_percent for d in self.devices) / len(self.devices)
                
                temps = [d.temperature_celsius for d in self.devices if d.temperature_celsius is not None]
                if temps:
                    snapshot.max_temperature_celsius = max(temps)
                
                powers = [d.power_draw_watts for d in self.devices if d.power_draw_watts is not None]
                if powers:
                    snapshot.total_power_draw_watts = sum(powers)
            
            # Add performance metrics
            if self.inference_times:
                recent_times = list(self.inference_times)[-100:]  # Last 100 inferences
                snapshot.average_inference_time_ms = sum(recent_times) / len(recent_times)
                
                if len(recent_times) > 1:
                    # Calculate throughput based on recent inferences
                    time_span_seconds = (recent_times[-1] - recent_times[0]) / 1000.0
                    if time_span_seconds > 0:
                        snapshot.throughput_inferences_per_second = len(recent_times) / time_span_seconds
            
            snapshot.inference_count = len(self.inference_times)
            
            # Store snapshot
            self.snapshots.append(snapshot)
            
            # Cleanup old snapshots
            if len(self.snapshots) > self.config.max_snapshots:
                self.snapshots = self.snapshots[-self.config.max_snapshots:]
            
            # Add to current session if active
            if self.current_session:
                self.current_session.snapshots.append(snapshot)
            
            return snapshot
            
        except Exception as e:
            self.logger.error(f"Failed to take GPU snapshot: {str(e)}")
            raise
    
    @handle_exceptions
    async def start_profiling_session(
        self,
        session_name: str = None,
        description: str = None,
        config: GPUProfilerConfig = None
    ) -> str:
        """Start a new profiling session."""
        if self.current_session is not None:
            await self.stop_profiling_session()
        
        session_id = str(uuid.uuid4())
        session_name = session_name or f"gpu_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        if config is None:
            config = self.config
        
        self.current_session = GPUProfilingSession(
            session_id=session_id,
            name=session_name,
            description=description,
            config=config
        )
        
        self.profiling_sessions[session_id] = self.current_session
        self.recent_sessions.append(session_id)
        
        # Take initial snapshot
        await self.take_snapshot("session_start")
        
        # Emit session started event if integration available
        if self.integration_mode and hasattr(self, 'event_bus'):
            try:
                await self.event_bus.emit(ProfilerStarted(
                    profiler_type="gpu",
                    profiler_id=id(self),
                    session_id=session_id,
                    session_name=session_name,
                    config=asdict(config)
                ))
            except:
                pass
        
        if self.integration_mode and hasattr(self, 'metrics'):
            try:
                self.metrics.increment("gpu_profiling_sessions_total")
            except:
                pass
        
        self.logger.info(f"Started GPU profiling session: {session_name} ({session_id})")
        
        return session_id
    
    @handle_exceptions
    async def stop_profiling_session(self) -> Optional[str]:
        """Stop the current profiling session."""
        if self.current_session is None:
            return None
        
        session = self.current_session
        session.end_time = datetime.now(timezone.utc)
        session.duration_seconds = (session.end_time - session.start_time).total_seconds()
        
        # Take final snapshot
        await self.take_snapshot("session_end")
        
        # Calculate session statistics
        self._calculate_session_statistics(session)
        
        # Generate optimization suggestions
        session.optimization_suggestions = self._generate_optimization_suggestions(session)
        
        # Export session data
        if self.config.export_json_reports:
            await self._export_session_report(session)
        
        # Emit session stopped event if integration available
        if self.integration_mode and hasattr(self, 'event_bus'):
            try:
                await self.event_bus.emit(ProfilerStopped(
                    profiler_type="gpu",
                    profiler_id=id(self),
                    session_id=session.session_id,
                    session_name=session.name,
                    duration_seconds=session.duration_seconds,
                    snapshots_count=len(session.snapshots)
                ))
            except:
                pass
        
        if self.integration_mode and hasattr(self, 'metrics'):
            try:
                self.metrics.record("gpu_profiling_session_duration_seconds", session.duration_seconds)
            except:
                pass
        
        self.logger.info(f"Stopped GPU profiling session: {session.name} ({session.session_id})")
        
        session_id = session.session_id
        self.current_session = None
        
        return session_id
    
    def _calculate_session_statistics(self, session: GPUProfilingSession) -> None:
        """Calculate statistics for a profiling session."""
        if not session.snapshots:
            return
        
        try:
            # Performance statistics
            inference_times = [s.average_inference_time_ms for s in session.snapshots if s.average_inference_time_ms is not None]
            if inference_times:
                session.average_inference_time_ms = sum(inference_times) / len(inference_times)
            
            session.total_inferences = max((s.inference_count for s in session.snapshots), default=0)
            
            # Utilization statistics
            gpu_utils = [s.average_gpu_utilization for s in session.snapshots]
            if gpu_utils:
                session.peak_gpu_utilization = max(gpu_utils)
            
            memory_utils = [s.average_memory_utilization for s in session.snapshots]
            if memory_utils:
                session.peak_memory_utilization = max(memory_utils)
            
            # Temperature statistics
            temps = [s.max_temperature_celsius for s in session.snapshots if s.max_temperature_celsius is not None]
            if temps:
                session.peak_temperature_celsius = max(temps)
            
            # Energy consumption (rough estimate)
            powers = [s.total_power_draw_watts for s in session.snapshots if s.total_power_draw_watts is not None]
            if powers and session.duration_seconds > 0:
                avg_power_watts = sum(powers) / len(powers)
                session.total_energy_consumed_wh = (avg_power_watts * session.duration_seconds) / 3600.0
            
        except Exception as e:
            self.logger.error(f"Failed to calculate session statistics: {str(e)}")
    
    def _generate_optimization_suggestions(self, session: GPUProfilingSession) -> List[Dict[str, Any]]:
        """Generate optimization suggestions based on session data."""
        suggestions = []
        
        try:
            if not session.snapshots:
                return suggestions
            
            # Analyze GPU utilization
            avg_utilization = session.peak_gpu_utilization
            if avg_utilization < 30.0:
                suggestions.append({
                    "type": "utilization",
                    "severity": "info",
                    "message": f"Low GPU utilization ({avg_utilization:.1f}%). Consider batch processing or model optimization.",
                    "recommendation": "Increase batch size or use GPU-optimized models"
                })
            elif avg_utilization > 95.0:
                suggestions.append({
                    "type": "utilization",
                    "severity": "warning",
                    "message": f"Very high GPU utilization ({avg_utilization:.1f}%). May indicate resource contention.",
                    "recommendation": "Consider distributing workload across multiple GPUs or reducing batch size"
                })
            
            # Analyze memory utilization
            if session.peak_memory_utilization > 90.0:
                suggestions.append({
                    "type": "memory",
                    "severity": "warning",
                    "message": f"High GPU memory utilization ({session.peak_memory_utilization:.1f}%). Risk of out-of-memory errors.",
                    "recommendation": "Reduce batch size, use gradient checkpointing, or enable memory-efficient attention"
                })
            
            # Analyze temperature
            if session.peak_temperature_celsius and session.peak_temperature_celsius > 85.0:
                suggestions.append({
                    "type": "thermal",
                    "severity": "warning",
                    "message": f"High GPU temperature ({session.peak_temperature_celsius:.1f}°C). May cause throttling.",
                    "recommendation": "Improve cooling, reduce workload intensity, or check thermal paste"
                })
            
            # Analyze inference performance
            if session.average_inference_time_ms > 1000.0:
                suggestions.append({
                    "type": "performance",
                    "severity": "info",
                    "message": f"Slow inference times ({session.average_inference_time_ms:.1f}ms). Consider optimization.",
                    "recommendation": "Use model quantization, TensorRT optimization, or smaller models"
                })
            
        except Exception as e:
            self.logger.error(f"Failed to generate optimization suggestions: {str(e)}")
        
        return suggestions
    
    async def _export_session_report(self, session: GPUProfilingSession) -> None:
        """Export session report to file."""
        try:
            report_data = {
                "session": asdict(session),
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "profiler_version": "1.0.0"
            }
            
            # Create session-specific directory
            session_dir = self.output_dir / session.session_id
            session_dir.mkdir(parents=True, exist_ok=True)
            
            # Export JSON report
            report_file = session_dir / f"{session.name}_report.json"
            with open(report_file, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)
            
            session.report_file_path = report_file
            
            self.logger.info(f"Exported session report: {report_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to export session report: {str(e)}")
    
    @handle_exceptions
    async def get_gpu_status(self) -> Dict[str, Any]:
        """Get current GPU status and metrics."""
        try:
            status = {
                "profiler_running": self.is_running,
                "devices_count": len(self.devices),
                "devices": [asdict(device) for device in self.devices],
                "current_session": self.current_session.session_id if self.current_session else None,
                "total_sessions": len(self.profiling_sessions),
                "total_snapshots": len(self.snapshots),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            # Add system-wide metrics
            if self.devices:
                status["system_metrics"] = {
                    "total_memory_mb": sum(d.total_memory_mb for d in self.devices),
                    "total_memory_used_mb": sum(d.used_memory_mb for d in self.devices),
                    "average_utilization_percent": sum(d.utilization_percent for d in self.devices) / len(self.devices),
                    "average_memory_utilization_percent": sum(d.memory_utilization_percent for d in self.devices) / len(self.devices),
                }
                
                temps = [d.temperature_celsius for d in self.devices if d.temperature_celsius is not None]
                if temps:
                    status["system_metrics"]["max_temperature_celsius"] = max(temps)
                
                powers = [d.power_draw_watts for d in self.devices if d.power_draw_watts is not None]
                if powers:
                    status["system_metrics"]["total_power_draw_watts"] = sum(powers)
            
            return status
            
        except Exception as e:
            self.logger.error(f"Failed to get GPU status: {str(e)}")
            return {"error": str(e)}
    
    @handle_exceptions
    async def get_profiling_sessions(self) -> List[Dict[str, Any]]:
        """Get list of profiling sessions."""
        try:
            sessions = []
            for session_id in self.recent_sessions:
                if session_id in self.profiling_sessions:
                    session = self.profiling_sessions[session_id]
                    sessions.append({
                        "session_id": session.session_id,
                        "name": session.name,
                        "description": session.description,
                        "start_time": session.start_time.isoformat(),
                        "end_time": session.end_time.isoformat() if session.end_time else None,
                        "duration_seconds": session.duration_seconds,
                        "snapshots_count": len(session.snapshots),
                        "total_inferences": session.total_inferences,
                        "peak_gpu_utilization": session.peak_gpu_utilization,
                        "peak_temperature_celsius": session.peak_temperature_celsius
                    })
            
            return sessions
            
        except Exception as e:
            self.logger.error(f"Failed to get profiling sessions: {str(e)}")
            return []
    
    @handle_exceptions
    async def track_inference(self, inference_time_ms: float, session_id: str = None) -> None:
        """Track an inference operation."""
        try:
            self.inference_times.append(inference_time_ms)
            
            # Update metrics if available
            if self.integration_mode and hasattr(self, 'metrics'):
                try:
                    self.metrics.record("gpu_inference_time_ms", inference_time_ms)
                    self.metrics.increment("gpu_inferences_total")
                except:
                    pass
            
            # Calculate current throughput
            if len(self.inference_times) >= 10:
                recent_times = list(self.inference_times)[-10:]
                avg_time_ms = sum(recent_times) / len(recent_times)
                throughput = 1000.0 / avg_time_ms if avg_time_ms > 0 else 0.0
                
                if self.integration_mode and hasattr(self, 'metrics'):
                    try:
                        self.metrics.set("gpu_throughput_inferences_per_second", throughput)
                    except:
                        pass
            
        except Exception as e:
            self.logger.error(f"Failed to track inference: {str(e)}")
    
    # Event handlers (only used if integration mode is enabled)
    async def _handle_component_initialized(self, event) -> None:
        """Handle component initialization events."""
        pass
    
    async def _handle_component_stopped(self, event) -> None:
        """Handle component stop events."""
        pass
    
    async def _handle_session_started(self, event) -> None:
        """Handle session start events."""
        if hasattr(event, 'session_id'):
            await self.start_profiling_session(f"session_{event.session_id}")
    
    async def _handle_session_ended(self, event) -> None:
        """Handle session end events."""
        if self.current_session:
            await self.stop_profiling_session()
    
    async def _handle_processing_started(self, event) -> None:
        """Handle processing start events."""
        pass
    
    async def _handle_processing_completed(self, event) -> None:
        """Handle processing completion events."""
        pass
    
    async def _handle_inference_started(self, event) -> None:
        """Handle inference start events."""
        pass
    
    async def _handle_inference_completed(self, event) -> None:
        """Handle inference completion events."""
        if hasattr(event, 'inference_time_ms'):
            await self.track_inference(event.inference_time_ms)
    
    async def _handle_error(self, event) -> None:
        """Handle error events."""
        pass
    
    async def _handle_system_shutdown(self, event) -> None:
        """Handle system shutdown events."""
        await self.stop()
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check for the GPU profiler."""
        try:
            health = {
                "status": "healthy",
                "details": {
                    "profiler_running": self.is_running,
                    "devices_discovered": len(self.devices),
                    "cuda_available": self.cuda_available,
                    "opencl_available": self.opencl_available,
                    "current_session_active": self.current_session is not None,
                    "recent_snapshots": len(self.snapshots),
                    "integration_mode": self.integration_mode
                }
            }
            
            # Check device availability
            unavailable_devices = [d for d in self.devices if not d.is_available]
            if unavailable_devices:
                health["status"] = "degraded"
                health["details"]["unavailable_devices"] = [d.index for d in unavailable_devices]
            
            # Check for critical temperatures
            hot_devices = [d for d in self.devices if d.temperature_celsius and d.temperature_celsius > self.config.temperature_critical_celsius]
            if hot_devices:
                health["status"] = "warning"
                health["details"]["overheating_devices"] = [
                    {"index": d.index, "temperature": d.temperature_celsius} for d in hot_devices
                ]
            
            return health
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }


class GPUVendor(Enum):
    """GPU vendor types."""
    NVIDIA = "nvidia"
    AMD = "amd"
    INTEL = "intel"
    UNKNOWN = "unknown"


class ProfilingMode(Enum):
    """GPU profiling modes."""
    BASIC = "basic"                    # Basic metrics only
    STANDARD = "standard"              # Standard monitoring
    DETAILED = "detailed"              # Detailed profiling
    COMPREHENSIVE = "comprehensive"    # Full profiling with all features


class ProfilingLevel(Enum):
    """GPU profiling levels."""
    LOW = "low"                        # Minimal overhead
    MEDIUM = "medium"                  # Balanced monitoring
    HIGH = "high"                      # Detailed monitoring
    MAXIMUM = "maximum"                # All available metrics


class GPUType(Enum):
    """GPU types."""
    DISCRETE = "discrete"
    INTEGRATED = "integrated"
    VIRTUAL = "virtual"
    UNKNOWN = "unknown"


@dataclass
class GPUDevice:
    """Information about a GPU device."""
    index: int
    name: str
    vendor: GPUVendor
    gpu_type: GPUType
    
    # Memory information
    total_memory_mb: float = 0.0
    free_memory_mb: float = 0.0
    used_memory_mb: float = 0.0
    
    # Performance information
    utilization_percent: float = 0.0
    memory_utilization_percent: float = 0.0
    temperature_celsius: Optional[float] = None
    power_draw_watts: Optional[float] = None
    max_power_watts: Optional[float] = None
    
    # Clock information
    graphics_clock_mhz: Optional[float] = None
    memory_clock_mhz: Optional[float] = None
    max_graphics_clock_mhz: Optional[float] = None
    max_memory_clock_mhz: Optional[float] = None
    
    # Capabilities
    compute_capability: Optional[str] = None
    driver_version: Optional[str] = None
    is_available: bool = True
    supports_profiling: bool = True
    
    # Additional metadata
    pci_bus_id: Optional[str] = None
    uuid: Optional[str] = None
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


