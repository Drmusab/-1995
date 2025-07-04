"""
<<<<<<< copilot/fix-1db4cb43-e694-40de-b677-350b778b7162
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

=======
Advanced GPU Profiling and Monitoring System
Author: Drmusab
Last Modified: 2025-05-26 16:34:21 UTC

This module provides comprehensive GPU resource monitoring, profiling, and optimization
for the AI assistant, supporting multiple GPU frameworks (PyTorch, TensorFlow, CUDA),
with real-time metrics collection, bottleneck detection, and seamless integration with
the core event and monitoring systems.
"""

from typing import Optional, Dict, Any, List, Set, Union, Callable, Type, Tuple
import asyncio
import threading
import time
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
from contextlib import asynccontextmanager, contextmanager
import logging
import os
import json
import platform
import subprocess
import re
import uuid
from collections import deque
from functools import wraps
import weakref
import warnings
import traceback

# Conditionally import GPU libraries based on availability
>>>>>>> main
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
<<<<<<< copilot/fix-1db4cb43-e694-40de-b677-350b778b7162
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
=======

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

try:
    import pynvml
    from pynvml import nvmlInit, nvmlShutdown, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False

# Core imports
from src.core.config.loader import ConfigLoader
from src.core.events.event_bus import EventBus
from src.core.events.event_types import (
    GPUUsageThresholdExceeded, GPUMemoryThresholdExceeded, GPUErrorDetected,
    GPUPerformanceOptimized, ModelOptimizationPerformed, ResourceUtilizationChanged,
    ComponentHealthChanged, ProcessingStarted, ProcessingCompleted, ResourceAvailabilityChanged
)
from src.core.error_handling import ErrorHandler, handle_exceptions
from src.core.dependency_injection import Container
from src.core.health_check import HealthCheck

# Observability imports
from src.observability.monitoring.metrics import MetricsCollector
from src.observability.monitoring.tracing import TraceManager
from src.observability.logging.config import get_logger
>>>>>>> main


class GPUVendor(Enum):
    """GPU vendor types."""
    NVIDIA = "nvidia"
    AMD = "amd"
    INTEL = "intel"
<<<<<<< copilot/fix-1db4cb43-e694-40de-b677-350b778b7162
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


=======
    APPLE = "apple"
    UNKNOWN = "unknown"


class GPUFramework(Enum):
    """Supported GPU frameworks."""
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    CUDA = "cuda"
    OPENCL = "opencl"
    METAL = "metal"
    NONE = "none"


class GPUProfilerMode(Enum):
    """Profiling modes for different use cases."""
    MONITORING = "monitoring"          # Continuous lightweight monitoring
    DETAILED = "detailed"              # Detailed profiling for optimization
    MEMORY_FOCUSED = "memory_focused"  # Focus on memory usage and leaks
    TRAINING = "training"              # Optimized for ML training workloads
    INFERENCE = "inference"            # Optimized for inference workloads
    MINIMAL = "minimal"                # Minimal overhead for production


class GPUOptimizationLevel(Enum):
    """Optimization levels for GPU usage."""
    NONE = "none"                # No optimization
    CONSERVATIVE = "conservative" # Safe optimizations only
    BALANCED = "balanced"         # Balance between performance and stability
    AGGRESSIVE = "aggressive"     # Maximize performance, may reduce stability
    AUTOMATIC = "automatic"       # Automatically determine best optimizations


@dataclass
class GPUStats:
    """Comprehensive GPU statistics."""
    # Basic info
    device_id: int
    name: str
    vendor: GPUVendor
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Utilization metrics
    utilization_percent: float = 0.0
    memory_used_mb: float = 0.0
    memory_total_mb: float = 0.0
    memory_percent: float = 0.0
    temperature_c: Optional[float] = None
    power_usage_watts: Optional[float] = None
    power_limit_watts: Optional[float] = None
    
    # Performance metrics
    compute_utilization: Optional[float] = None
    memory_bandwidth_utilization: Optional[float] = None
    pcie_bandwidth_utilization: Optional[float] = None
    sm_occupancy: Optional[float] = None
    
    # Process info
    process_count: int = 0
    process_details: List[Dict[str, Any]] = field(default_factory=list)
    
    # Tensor operations
    active_tensors: int = 0
    tensor_memory_mb: float = 0.0
    
    # Framework-specific
    framework: GPUFramework = GPUFramework.NONE
    framework_version: Optional[str] = None
    cuda_version: Optional[str] = None
    
    # Alerts and status
    throttling_detected: bool = False
    errors_detected: bool = False
    health_status: str = "healthy"


@dataclass
class GPUProfilerConfig:
    """Configuration for the GPU profiler."""
    # Monitoring settings
    enabled: bool = True
    polling_interval_seconds: float = 5.0
    detailed_polling_interval_seconds: float = 30.0
    profiling_mode: GPUProfilerMode = GPUProfilerMode.MONITORING
    
    # Thresholds for alerts
    memory_warning_threshold: float = 85.0  # percentage
    memory_critical_threshold: float = 95.0  # percentage
    utilization_warning_threshold: float = 90.0  # percentage
    temperature_warning_threshold_c: float = 80.0  # Celsius
    temperature_critical_threshold_c: float = 90.0  # Celsius
    
    # History and storage
    history_length: int = 100
    save_history: bool = True
    history_file_path: Optional[str] = None
    
    # Optimization settings
    enable_automatic_optimization: bool = True
    optimization_level: GPUOptimizationLevel = GPUOptimizationLevel.BALANCED
    enable_tensor_fusion: bool = True
    enable_memory_caching: bool = True
    enable_mixed_precision: bool = True
    
    # Frameworks to monitor
    monitor_pytorch: bool = True
    monitor_tensorflow: bool = True
    monitor_cuda_runtime: bool = True
    
    # Reporting and visualization
    enable_dashboard: bool = False
    dashboard_port: int = 8080
    export_metrics: bool = True
    enable_tracing: bool = True
    
    # Integration settings
    report_to_metrics_collector: bool = True
    emit_events: bool = True
    register_health_check: bool = True


class GPUMemoryTracker:
    """Track GPU memory allocations and detect leaks."""
    
    def __init__(self, device_id: int, logger):
        self.device_id = device_id
        self.logger = logger
        self.allocation_history = deque(maxlen=1000)
        self.tensor_references = weakref.WeakValueDictionary()
        self.enabled = False
        self.snapshot_interval = 60  # seconds
        self.last_snapshot_time = time.time()
        self.snapshots = deque(maxlen=100)
    
    def start_tracking(self) -> None:
        """Start memory tracking."""
        if not TORCH_AVAILABLE:
            self.logger.warning("PyTorch not available, cannot track GPU memory")
            return
        
        self.enabled = True
        if hasattr(torch, 'cuda') and torch.cuda.is_available():
            self.logger.info(f"Started GPU memory tracking for device {self.device_id}")
            # Enable PyTorch memory tracking
            torch.cuda.memory.set_per_process_memory_fraction(0.95, self.device_id)
            torch.cuda.memory._record_memory_history(enabled=True, device=self.device_id)
    
    def stop_tracking(self) -> None:
        """Stop memory tracking."""
        if not self.enabled or not TORCH_AVAILABLE:
            return
        
        self.enabled = False
        if hasattr(torch, 'cuda') and torch.cuda.is_available():
            # Disable PyTorch memory tracking
            torch.cuda.memory._record_memory_history(enabled=False, device=self.device_id)
            self.logger.info(f"Stopped GPU memory tracking for device {self.device_id}")
    
    def record_allocation(self, size: int, tensor_id: str, stack_trace: str) -> None:
        """Record a memory allocation."""
        if not self.enabled:
            return
        
        self.allocation_history.append({
            'timestamp': time.time(),
            'operation': 'allocate',
            'size': size,
            'tensor_id': tensor_id,
            'stack_trace': stack_trace
        })
    
    def record_deallocation(self, tensor_id: str) -> None:
        """Record a memory deallocation."""
        if not self.enabled:
            return
        
        self.allocation_history.append({
            'timestamp': time.time(),
            'operation': 'deallocate',
            'tensor_id': tensor_id
        })
    
    def register_tensor(self, tensor, name: Optional[str] = None) -> str:
        """Register a tensor for tracking."""
        if not self.enabled or not TORCH_AVAILABLE:
            return "unknown"
        
        tensor_id = str(uuid.uuid4())
        self.tensor_references[tensor_id] = tensor
        
        if hasattr(tensor, 'numel') and hasattr(tensor, 'element_size'):
            size = tensor.numel() * tensor.element_size()
            stack_trace = ''.join(traceback.format_stack(limit=10))
            self.record_allocation(size, tensor_id, stack_trace)
        
        return tensor_id
    
    def take_snapshot(self) -> Dict[str, Any]:
        """Take a memory snapshot."""
        if not self.enabled or not TORCH_AVAILABLE:
            return {}
        
        now = time.time()
        if now - self.last_snapshot_time < self.snapshot_interval:
            return {}
        
        self.last_snapshot_time = now
        
        if hasattr(torch, 'cuda') and torch.cuda.is_available():
            snapshot = {
                'timestamp': now,
                'allocated_bytes': torch.cuda.memory_allocated(self.device_id),
                'reserved_bytes': torch.cuda.memory_reserved(self.device_id),
                'active_tensors': len(self.tensor_references),
                'allocation_count': len([a for a in self.allocation_history if a['operation'] == 'allocate']),
                'deallocation_count': len([a for a in self.allocation_history if a['operation'] == 'deallocate'])
            }
            
            # Add detailed memory stats if available
            if hasattr(torch.cuda, 'memory_stats'):
                snapshot['memory_stats'] = torch.cuda.memory_stats(self.device_id)
            
            self.snapshots.append(snapshot)
            return snapshot
        
        return {}
    
    def analyze_leaks(self) -> List[Dict[str, Any]]:
        """Analyze potential memory leaks."""
        if not self.enabled or len(self.snapshots) < 2:
            return []
        
        leaks = []
        first_snapshot = self.snapshots[0]
        last_snapshot = self.snapshots[-1]
        
        # Check for significant memory growth
        memory_growth = last_snapshot['allocated_bytes'] - first_snapshot['allocated_bytes']
        time_diff = last_snapshot['timestamp'] - first_snapshot['timestamp']
        
        if memory_growth > 100 * 1024 * 1024 and time_diff > 300:  # 100MB growth over 5 minutes
            # Find allocations without matching deallocations
            allocations = {}
            for record in self.allocation_history:
                if record['operation'] == 'allocate':
                    allocations[record['tensor_id']] = record
                elif record['operation'] == 'deallocate' and record['tensor_id'] in allocations:
                    del allocations[record['tensor_id']]
            
            # Convert remaining allocations to leak reports
            for tensor_id, alloc in allocations.items():
                leaks.append({
                    'tensor_id': tensor_id,
                    'size': alloc['size'],
                    'age_seconds': time.time() - alloc['timestamp'],
                    'stack_trace': alloc['stack_trace']
                })
        
        return leaks


class ModelOptimizer:
    """Optimize models for GPU performance."""
    
    def __init__(self, logger, config: GPUProfilerConfig):
        self.logger = logger
        self.config = config
        self.optimized_models = {}
    
    def optimize_pytorch_model(self, model: Any, optimization_level: GPUOptimizationLevel) -> Any:
        """Optimize a PyTorch model for GPU performance."""
        if not TORCH_AVAILABLE or not hasattr(model, 'to'):
            return model
        
        try:
            original_device = next(model.parameters()).device
            model_id = id(model)
            
            # If already optimized at same or higher level, return cached version
            if model_id in self.optimized_models:
                cached_level = self.optimized_models[model_id]['level']
                if self._level_value(cached_level) >= self._level_value(optimization_level):
                    return self.optimized_models[model_id]['model']
            
            # Apply optimizations based on level
            if optimization_level == GPUOptimizationLevel.NONE:
                return model
            
            optimized_model = model
            
            # Conservative optimizations
            if self._level_value(optimization_level) >= self._level_value(GPUOptimizationLevel.CONSERVATIVE):
                # Convert to inference mode if not training
                if not hasattr(model, 'training') or not model.training:
                    optimized_model = optimized_model.eval()
                
                # Ensure model is on GPU
                if hasattr(torch, 'cuda') and torch.cuda.is_available():
                    optimized_model = optimized_model.to('cuda')
            
            # Balanced optimizations
            if self._level_value(optimization_level) >= self._level_value(GPUOptimizationLevel.BALANCED):
                # Apply mixed precision if supported
                if self.config.enable_mixed_precision and hasattr(torch, 'cuda') and \
                   hasattr(torch.cuda, 'amp') and hasattr(torch.cuda.amp, 'autocast'):
                    original_forward = optimized_model.forward
                    
                    @wraps(original_forward)
                    def forward_with_autocast(*args, **kwargs):
                        with torch.cuda.amp.autocast():
                            return original_forward(*args, **kwargs)
                    
                    optimized_model.forward = forward_with_autocast
                
                # Disable gradient calculation for inference
                if not hasattr(model, 'training') or not model.training:
                    original_forward = optimized_model.forward
                    
                    @wraps(original_forward)
                    def forward_no_grad(*args, **kwargs):
                        with torch.no_grad():
                            return original_forward(*args, **kwargs)
                    
                    optimized_model.forward = forward_no_grad
            
            # Aggressive optimizations
            if self._level_value(optimization_level) >= self._level_value(GPUOptimizationLevel.AGGRESSIVE):
                # Try to apply TorchScript JIT compilation
                if hasattr(torch, 'jit') and hasattr(torch.jit, 'script'):
                    try:
                        # Save original for fallback
                        unoptimized_model = optimized_model
                        optimized_model = torch.jit.script(optimized_model)
                    except Exception as e:
                        self.logger.warning(f"JIT compilation failed: {str(e)}, using unoptimized model")
                        optimized_model = unoptimized_model
                
                # Apply quantization if available
                if hasattr(torch, 'quantization'):
                    try:
                        # Save original for fallback
                        unoptimized_model = optimized_model
                        if hasattr(torch.quantization, 'quantize_dynamic'):
                            optimized_model = torch.quantization.quantize_dynamic(
                                optimized_model, {torch.nn.Linear}, dtype=torch.qint8
                            )
                    except Exception as e:
                        self.logger.warning(f"Quantization failed: {str(e)}, using unoptimized model")
                        optimized_model = unoptimized_model
            
            # Cache optimized model
            self.optimized_models[model_id] = {
                'model': optimized_model,
                'level': optimization_level,
                'original_device': original_device
            }
            
            self.logger.info(f"Optimized PyTorch model at level {optimization_level.value}")
            return optimized_model
            
        except Exception as e:
            self.logger.error(f"Error optimizing PyTorch model: {str(e)}")
            return model
    
    def optimize_tensorflow_model(self, model: Any, optimization_level: GPUOptimizationLevel) -> Any:
        """Optimize a TensorFlow model for GPU performance."""
        if not TF_AVAILABLE:
            return model
        
        try:
            model_id = id(model)
            
            # If already optimized at same or higher level, return cached version
            if model_id in self.optimized_models:
                cached_level = self.optimized_models[model_id]['level']
                if self._level_value(cached_level) >= self._level_value(optimization_level):
                    return self.optimized_models[model_id]['model']
            
            # Apply optimizations based on level
            if optimization_level == GPUOptimizationLevel.NONE:
                return model
            
            optimized_model = model
            
            # Conservative optimizations
            if self._level_value(optimization_level) >= self._level_value(GPUOptimizationLevel.CONSERVATIVE):
                # Ensure model uses GPU memory efficiently
                gpus = tf.config.experimental.list_physical_devices('GPU')
                if gpus:
                    try:
                        for gpu in gpus:
                            tf.config.experimental.set_memory_growth(gpu, True)
                    except RuntimeError as e:
                        self.logger.warning(f"Memory growth setting failed: {str(e)}")
            
            # Balanced optimizations
            if self._level_value(optimization_level) >= self._level_value(GPUOptimizationLevel.BALANCED):
                # Apply mixed precision if supported
                if self.config.enable_mixed_precision and hasattr(tf, 'keras') and \
                   hasattr(tf.keras, 'mixed_precision') and hasattr(tf.keras.mixed_precision, 'set_global_policy'):
                    try:
                        tf.keras.mixed_precision.set_global_policy('mixed_float16')
                    except Exception as e:
                        self.logger.warning(f"Mixed precision setting failed: {str(e)}")
            
            # Aggressive optimizations
            if self._level_value(optimization_level) >= self._level_value(GPUOptimizationLevel.AGGRESSIVE):
                # Convert to TensorFlow Lite if possible
                if hasattr(tf, 'lite') and hasattr(tf.lite, 'TFLiteConverter') and \
                   hasattr(model, 'save') and callable(model.save):
                    try:
                        # Save model temporarily
                        temp_path = f"/tmp/tf_model_{uuid.uuid4()}"
                        model.save(temp_path)
                        
                        # Convert to TFLite
                        converter = tf.lite.TFLiteConverter.from_saved_model(temp_path)
                        converter.optimizations = [tf.lite.Optimize.DEFAULT]
                        tflite_model = converter.convert()
                        
                        # Clean up
                        import shutil
                        if os.path.exists(temp_path):
                            shutil.rmtree(temp_path)
                        
                        # We would need to wrap the TFLite model to match the original API
                        # This is a simplified example
                        class TFLiteWrapper:
                            def __init__(self, tflite_model):
                                self.tflite_model = tflite_model
                                self.interpreter = tf.lite.Interpreter(model_content=tflite_model)
                                self.interpreter.allocate_tensors()
                                self.input_details = self.interpreter.get_input_details()
                                self.output_details = self.interpreter.get_output_details()
                            
                            def __call__(self, inputs):
                                self.interpreter.set_tensor(self.input_details[0]['index'], inputs)
                                self.interpreter.invoke()
                                return self.interpreter.get_tensor(self.output_details[0]['index'])
                        
                        optimized_model = TFLiteWrapper(tflite_model)
                    except Exception as e:
                        self.logger.warning(f"TFLite conversion failed: {str(e)}, using original model")
            
            # Cache optimized model
            self.optimized_models[model_id] = {
                'model': optimized_model,
                'level': optimization_level
            }
            
            self.logger.info(f"Optimized TensorFlow model at level {optimization_level.value}")
            return optimized_model
            
        except Exception as e:
            self.logger.error(f"Error optimizing TensorFlow model: {str(e)}")
            return model
    
    def optimize_tensor_operations(self, tensor_op_func: Callable, optimization_level: GPUOptimizationLevel) -> Callable:
        """Optimize tensor operations for GPU performance."""
        if optimization_level == GPUOptimizationLevel.NONE:
            return tensor_op_func
        
        @wraps(tensor_op_func)
        def optimized_op(*args, **kwargs):
            # Optimize PyTorch operations
            if TORCH_AVAILABLE:
                if 'dtype' not in kwargs and args and hasattr(args[0], 'dtype'):
                    # Use lower precision for aggressive optimization
                    if optimization_level == GPUOptimizationLevel.AGGRESSIVE and self.config.enable_mixed_precision:
                        kwargs['dtype'] = torch.float16
                
                # Ensure tensors are contiguous for better performance
                new_args = []
                for arg in args:
                    if hasattr(arg, 'is_contiguous') and hasattr(arg, 'contiguous') and not arg.is_contiguous():
                        new_args.append(arg.contiguous())
                    else:
                        new_args.append(arg)
                
                # Use tensor cores if available (for matmul operations)
                if tensor_op_func.__name__ in ['matmul', 'mm', 'bmm'] and \
                   hasattr(torch, 'backends') and hasattr(torch.backends, 'cudnn'):
                    torch.backends.cudnn.benchmark = True
                
                # Apply the optimized operation
                with torch.no_grad() if optimization_level != GPUOptimizationLevel.NONE else contextmanager(lambda: (yield))():
                    if self.config.enable_mixed_precision and optimization_level != GPUOptimizationLevel.CONSERVATIVE:
                        with torch.cuda.amp.autocast() if hasattr(torch.cuda, 'amp') else contextmanager(lambda: (yield))():
                            return tensor_op_func(*new_args, **kwargs)
                    else:
                        return tensor_op_func(*new_args, **kwargs)
            
            # Default case when optimizations aren't applicable
            return tensor_op_func(*args, **kwargs)
        
        return optimized_op
    
    def clear_cache(self) -> None:
        """Clear GPU memory cache to free up memory."""
        if TORCH_AVAILABLE and hasattr(torch, 'cuda'):
            try:
                torch.cuda.empty_cache()
                self.logger.debug("Cleared PyTorch CUDA cache")
            except Exception as e:
                self.logger.warning(f"Failed to clear PyTorch CUDA cache: {str(e)}")
        
        if TF_AVAILABLE:
            try:
                if hasattr(tf, 'keras') and hasattr(tf.keras, 'backend') and \
                   hasattr(tf.keras.backend, 'clear_session'):
                    tf.keras.backend.clear_session()
                    self.logger.debug("Cleared TensorFlow session")
            except Exception as e:
                self.logger.warning(f"Failed to clear TensorFlow session: {str(e)}")
    
    def _level_value(self, level: GPUOptimizationLevel) -> int:
        """Convert optimization level to numeric value for comparison."""
        level_values = {
            GPUOptimizationLevel.NONE: 0,
            GPUOptimizationLevel.CONSERVATIVE: 1,
            GPUOptimizationLevel.BALANCED: 2,
            GPUOptimizationLevel.AGGRESSIVE: 3,
            GPUOptimizationLevel.AUTOMATIC: 2  # Default to balanced
        }
        return level_values.get(level, 0)


class EnhancedGPUProfiler:
    """
    Advanced GPU Profiling and Monitoring System.
    
    This profiler provides comprehensive GPU monitoring, profiling, and optimization
    capabilities for AI workloads, supporting multiple frameworks and offering:
    
    - Real-time GPU resource monitoring (utilization, memory, temperature)
    - Memory leak detection and tensor tracking
    - Automatic model optimization for different frameworks
    - Bottleneck identification and performance recommendations
    - Integration with the metrics collection and event systems
    - Support for multi-GPU environments
    - Configurable profiling modes for different use cases
    - Health checks and alerting for GPU issues
    """
    
    def __init__(self, container: Container):
        """
        Initialize the GPU profiler.
        
        Args:
            container: Dependency injection container
        """
        self.container = container
        self.logger = get_logger(__name__)
        
        # Core services
        self.config_loader = container.get(ConfigLoader)
        self.event_bus = container.get(EventBus)
        self.error_handler = container.get(ErrorHandler)
        self.health_check = container.get(HealthCheck)
        
        # Observability
        self.metrics = container.get(MetricsCollector)
        self.tracer = container.get(TraceManager)
        
        # Load configuration
        self.config = self._load_config()
        
        # State management
        self.initialized = False
        self.monitoring_task: Optional[asyncio.Task] = None
        self.optimization_task: Optional[asyncio.Task] = None
        self.shutdown_event = asyncio.Event()
        
        # GPU information
        self.available_gpus: List[Dict[str, Any]] = []
        self.gpu_stats: Dict[int, List[GPUStats]] = {}
        self.memory_trackers: Dict[int, GPUMemoryTracker] = {}
        
        # GPU framework detection
        self.detected_frameworks: Set[GPUFramework] = set()
        
        # Model optimization
        self.model_optimizer = ModelOptimizer(self.logger, self.config)
        
        # Thread safety
        self.stats_lock = threading.Lock()
        
        # Initialize monitoring
        self._setup_gpu_monitoring()
        self._setup_metrics()
        
        self.logger.info("EnhancedGPUProfiler initialized")

    def _load_config(self) -> GPUProfilerConfig:
        """Load configuration from config loader."""
        # Get default config
        config = GPUProfilerConfig()
        
        # Override with values from config loader
        try:
            gpu_config = self.config_loader.get("observability.profiling.gpu", {})
            
            if "enabled" in gpu_config:
                config.enabled = gpu_config["enabled"]
            
            if "polling_interval_seconds" in gpu_config:
                config.polling_interval_seconds = gpu_config["polling_interval_seconds"]
            
            if "profiling_mode" in gpu_config:
                config.profiling_mode = GPUProfilerMode(gpu_config["profiling_mode"])
            
            if "memory_warning_threshold" in gpu_config:
                config.memory_warning_threshold = gpu_config["memory_warning_threshold"]
            
            if "optimization_level" in gpu_config:
                config.optimization_level = GPUOptimizationLevel(gpu_config["optimization_level"])
            
            # Load all other config fields if present
            for field in config.__dataclass_fields__:
                if field in gpu_config and field not in ["profiling_mode", "optimization_level"]:
                    setattr(config, field, gpu_config[field])
            
        except Exception as e:
            self.logger.warning(f"Failed to load GPU profiler config: {str(e)}, using defaults")
        
        return config

    def _setup_gpu_monitoring(self) -> None:
        """Set up GPU monitoring infrastructure."""
        if not self.config.enabled:
            self.logger.info("GPU profiling is disabled in configuration")
            return
        
        # Initialize NVML for NVIDIA GPUs
        if NVML_AVAILABLE:
            try:
                nvmlInit()
                self.logger.info("NVML initialized successfully")
            except Exception as e:
                self.logger.warning(f"Failed to initialize NVML: {str(e)}")
        
        # Detect available GPUs
        self._detect_gpus()
        
        # Detect available frameworks
        self._detect_frameworks()
        
        # Initialize memory trackers for each GPU
        for gpu_info in self.available_gpus:
            device_id = gpu_info["device_id"]
            self.memory_trackers[device_id] = GPUMemoryTracker(device_id, self.logger)
            if self.config.profiling_mode in [
                GPUProfilerMode.MEMORY_FOCUSED, 
                GPUProfilerMode.DETAILED,
                GPUProfilerMode.TRAINING
            ]:
                self.memory_trackers[device_id].start_tracking()

    def _setup_metrics(self) -> None:
        """Set up metrics collection."""
        if not self.config.enabled or not self.config.report_to_metrics_collector:
            return
        
        # Register GPU metrics
        try:
            # Basic metrics
            self.metrics.register_gauge("gpu_utilization_percent")
            self.metrics.register_gauge("gpu_memory_used_mb")
            self.metrics.register_gauge("gpu_memory_total_mb")
            self.metrics.register_gauge("gpu_memory_percent")
            self.metrics.register_gauge("gpu_temperature_c")
            self.metrics.register_gauge("gpu_power_usage_watts")
            
            # Performance metrics
            self.metrics.register_gauge("gpu_tensor_count")
            self.metrics.register_gauge("gpu_process_count")
            self.metrics.register_gauge("gpu_compute_utilization")
            
            # Framework-specific metrics
            if GPUFramework.PYTORCH in self.detected_frameworks:
                self.metrics.register_gauge("gpu_pytorch_tensors")
                self.metrics.register_gauge("gpu_pytorch_memory_mb")
            
            if GPUFramework.TENSORFLOW in self.detected_frameworks:
                self.metrics.register_gauge("gpu_tensorflow_memory_mb")
            
            self.logger.info("GPU metrics registered successfully")
            
        except Exception as e:
            self.logger.warning(f"Failed to register GPU metrics: {str(e)}")

    def _detect_gpus(self) -> None:
        """Detect available GPUs and their capabilities."""
        self.available_gpus = []
        
        # Check NVIDIA GPUs using NVML
        if NVML_AVAILABLE:
            try:
                device_count = pynvml.nvmlDeviceGetCount()
                self.logger.info(f"Detected {device_count} NVIDIA GPUs")
                
                for i in range(device_count):
                    handle = nvmlDeviceGetHandleByIndex(i)
                    name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
                    memory = nvmlDeviceGetMemoryInfo(handle)
                    
                    self.available_gpus.append({
                        "device_id": i,
                        "name": name,
                        "vendor": GPUVendor.NVIDIA,
                        "memory_total_mb": memory.total / (1024 * 1024),
                        "cuda_compute_capability": self._get_cuda_compute_capability(i)
                    })
                    
                    # Initialize stats history
                    self.gpu_stats[i] = []
            except Exception as e:
                self.logger.warning(f"Error detecting NVIDIA GPUs with NVML: {str(e)}")
        
        # Check PyTorch GPUs
        if TORCH_AVAILABLE and hasattr(torch, 'cuda') and torch.cuda.is_available():
            try:
                torch_device_count = torch.cuda.device_count()
                # If we didn't detect any GPUs with NVML, add them from PyTorch
                if not self.available_gpus:
                    for i in range(torch_device_count):
                        name = torch.cuda.get_device_name(i)
                        self.available_gpus.append({
                            "device_id": i,
                            "name": name,
                            "vendor": GPUVendor.NVIDIA,
                            "memory_total_mb": torch.cuda.get_device_properties(i).total_memory / (1024 * 1024)
                        })
                        
                        # Initialize stats history
                        self.gpu_stats[i] = []
                
                self.logger.info(f"PyTorch reports {torch_device_count} available CUDA devices")
            except Exception as e:
                self.logger.warning(f"Error detecting GPUs with PyTorch: {str(e)}")
        
        # Check TensorFlow GPUs
        if TF_AVAILABLE:
            try:
                tf_gpus = tf.config.list_physical_devices('GPU')
                self.logger.info(f"TensorFlow reports {len(tf_gpus)} available GPU devices")
                
                # If we still don't have any GPUs detected, add them from TensorFlow
                if not self.available_gpus and tf_gpus:
                    for i, gpu in enumerate(tf_gpus):
                        # TensorFlow doesn't provide detailed GPU info, so we use generic info
                        self.available_gpus.append({
                            "device_id": i,
                            "name": f"TensorFlow GPU {i}",
                            "vendor": GPUVendor.UNKNOWN,
                            "memory_total_mb": 0  # We can't get this from TF directly
                        })
                        
                        # Initialize stats history
                        self.gpu_stats[i] = []
            except Exception as e:
                self.logger.warning(f"Error detecting GPUs with TensorFlow: {str(e)}")
        
        # Check Apple Metal GPUs
        if platform.system() == 'Darwin' and hasattr(platform, 'mac_ver'):
            try:
                # Check if PyTorch has Metal support
                has_mps = False
                if TORCH_AVAILABLE and hasattr(torch, 'backends') and hasattr(torch.backends, 'mps'):
                    has_mps = torch.backends.mps.is_available()
                
                if has_mps:
                    self.available_gpus.append({
                        "device_id": 0,
                        "name": "Apple M-series GPU",
                        "vendor": GPUVendor.APPLE,
                        "memory_total_mb": 0  # We can't easily get this
                    })
                    
                    # Initialize stats history
                    self.gpu_stats[0] = []
                    
                    self.logger.info("Detected Apple Metal GPU")
            except Exception as e:
                self.logger.warning(f"Error detecting Apple GPU: {str(e)}")
        
        if not self.available_gpus:
            self.logger.warning("No GPUs detected")

    def _detect_frameworks(self) -> None:
        """Detect available GPU frameworks."""
        self.detected_frameworks = set()
        
        # Check PyTorch
        if TORCH_AVAILABLE:
            if hasattr(torch, 'cuda') and torch.cuda.is_available():
                self.detected_frameworks.add(GPUFramework.PYTORCH)
                self.detected_frameworks.add(GPUFramework.CUDA)
                self.logger.info(f"Detected PyTorch {torch.__version__} with CUDA support")
            elif hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.detected_frameworks.add(GPUFramework.PYTORCH)
                self.detected_frameworks.add(GPUFramework.METAL)
                self.logger.info(f"Detected PyTorch {torch.__version__} with Metal support")
            else:
                self.logger.info(f"Detected PyTorch {torch.__version__} without GPU support")
        
        # Check TensorFlow
        if TF_AVAILABLE:
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                self.detected_frameworks.add(GPUFramework.TENSORFLOW)
                # Check if it's using CUDA or ROCm
                if hasattr(tf, 'sysconfig') and hasattr(tf.sysconfig, 'get_build_info'):
                    build_info = tf.sysconfig.get_build_info()
                    if 'cuda' in build_info.get('cuda_version', '').lower():
                        self.detected_frameworks.add(GPUFramework.CUDA)
                    
                self.logger.info(f"Detected TensorFlow {tf.__version__} with GPU support")
            else:
                self.logger.info(f"Detected TensorFlow {tf.__version__} without GPU support")
        
        # Check CUDA directly
        if not GPUFramework.CUDA in self.detected_frameworks:
            cuda_version = self._get_cuda_version()
            if cuda_version:
                self.detected_frameworks.add(GPUFramework.CUDA)
                self.logger.info(f"Detected CUDA {cuda_version}")
        
        if not self.detected_frameworks:
            self.detected_frameworks.add(GPUFramework.NONE)
            self.logger.warning("No GPU frameworks detected")

    def _get_cuda_version(self) -> Optional[str]:
        """Get CUDA version if available."""
        try:
            # Try to get CUDA version from nvcc
            result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
            if result.returncode == 0:
                match = re.search(r'release (\d+\.\d+)', result.stdout)
                if match:
                    return match.group(1)
            
            # Try to get from PyTorch
            if TORCH_AVAILABLE and hasattr(torch, 'version') and hasattr(torch.version, 'cuda'):
                return torch.version.cuda
            
            return None
        except Exception:
            return None

    def _get_cuda_compute_capability(self, device_id: int) -> Optional[str]:
        """Get CUDA compute capability for a device."""
        try:
            if TORCH_AVAILABLE and hasattr(torch, 'cuda') and torch.cuda.is_available():
                if device_id < torch.cuda.device_count():
                    props = torch.cuda.get_device_properties(device_id)
                    return f"{props.major}.{props.minor}"
            return None
        except Exception:
            return None

    async def initialize(self) -> None:
        """Initialize the GPU profiler."""
        if not self.config.enabled or self.initialized:
            return
        
        try:
            self.logger.info("Initializing GPU profiler")
            
            # Register with health check system
            if self.config.register_health_check:
                self.health_check.register_component("gpu_profiler", self._health_check_callback)
            
            # Start monitoring task
            if self.available_gpus:
                self.monitoring_task = asyncio.create_task(self._monitoring_loop())
                self.optimization_task = asyncio.create_task(self._optimization_loop())
                
                # Report GPU availability
                for gpu in self.available_gpus:
                    await self.event_bus.emit(ResourceAvailabilityChanged(
                        resource_type="gpu",
                        resource_id=str(gpu["device_id"]),
                        name=gpu["name"],
                        available=True,
                        details=gpu
                    ))
            
            self.initialized = True
            self.logger.info("GPU profiler initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize GPU profiler: {str(e)}")
            # Don't re-raise to avoid crashing the application if GPU monitoring fails

    @handle_exceptions
    async def shutdown(self) -> None:
        """Shutdown the GPU profiler."""
        if not self.initialized:
            return
        
        self.logger.info("Shutting down GPU profiler")
        
        # Signal monitoring task to stop
        self.shutdown_event.set()
        
        # Wait for tasks to complete
        if self.monitoring_task:
            try:
                await asyncio.wait_for(self.monitoring_task, timeout=5.0)
            except asyncio.TimeoutError:
                self.logger.warning("GPU monitoring task did not shutdown gracefully, cancelling")
                self.monitoring_task.cancel()
        
        if self.optimization_task:
            try:
                await asyncio.wait_for(self.optimization_task, timeout=5.0)
            except asyncio.TimeoutError:
                self.logger.warning("GPU optimization task did not shutdown gracefully, cancelling")
                self.optimization_task.cancel()
        
        # Stop memory trackers
        for tracker in self.memory_trackers.values():
            tracker.stop_tracking()
        
        # Shutdown NVML
        if NVML_AVAILABLE:
            try:
                nvmlShutdown()
                self.logger.info("NVML shutdown successfully")
            except Exception as e:
                self.logger.warning(f"Error shutting down NVML: {str(e)}")
        
        self.initialized = False
        self.logger.info("GPU profiler shutdown completed")

    async def _monitoring_loop(self) -> None:
        """Background task for GPU monitoring."""
        if not self.config.enabled or not self.available_gpus:
            return
        
        self.logger.info("Starting GPU monitoring loop")
        
        while not self.shutdown_event.is_set():
            try:
                # Collect stats for each GPU
                for gpu in self.available_gpus:
                    device_id = gpu["device_id"]
                    stats = await self._collect_gpu_stats(device_id)
                    
                    if stats:
                        # Store in history with thread safety
                        with self.stats_lock:
                            self.gpu_stats[device_id].append(stats)
                            # Limit history length
                            if len(self.gpu_stats[device_id]) > self.config.history_length:
                                self.gpu_stats[device_id].pop(0)
                        
                        # Report metrics
                        if self.config.report_to_metrics_collector:
                            self._report_metrics(stats)
                        
                        # Check thresholds and emit events
                        await self._check_thresholds(stats)
                        
                        # Collect memory leak info in memory-focused mode
                        if self.config.profiling_mode == GPUProfilerMode.MEMORY_FOCUSED:
                            tracker = self.memory_trackers.get(device_id)
                            if tracker and tracker.enabled:
                                tracker.take_snapshot()
                                leaks = tracker.analyze_leaks()
                                if leaks:
                                    self.logger.warning(f"Potential memory leaks detected on GPU {device_id}: {len(leaks)} objects")
                                    for leak in leaks[:5]:  # Log only first 5 to avoid spam
                                        self.logger.warning(f"Leak: {leak['size'] / 1024 / 1024:.2f}MB, age: {leak['age_seconds']:.1f}s")
                
                # Save history to file if configured
                if self.config.save_history and self.config.history_file_path:
                    self._save_history()
                
                # Determine polling interval based on profiling mode
                if self.config.profiling_mode == GPUProfilerMode.MONITORING:
                    polling_interval = self.config.polling_interval_seconds
                elif self.config.profiling_mode == GPUProfilerMode.MINIMAL:
                    polling_interval = max(self.config.polling_interval_seconds * 2, 30.0)
                else:
                    polling_interval = min(self.config.polling_interval_seconds, 5.0)
                
                await asyncio.sleep(polling_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in GPU monitoring loop: {str(e)}")
                await asyncio.sleep(10.0)  # Back off on errors
        
        self.logger.info("GPU monitoring loop stopped")

    async def _optimization_loop(self) -> None:
        """Background task for GPU optimization."""
        if not self.config.enabled or not self.config.enable_automatic_optimization:
            return
        
        self.logger.info("Starting GPU optimization loop")
        
        while not self.shutdown_event.is_set():
            try:
                # Check if memory usage is high and needs optimization
                for gpu in self.available_gpus:
                    device_id = gpu["device_id"]
                    
                    if device_id in self.gpu_stats and self.gpu_stats[device_id]:
                        recent_stats = self.gpu_stats[device_id][-1]
                        
                        # If memory usage is high, try to clear caches
                        if recent_stats.memory_percent > self.config.memory_warning_threshold:
                            self.logger.info(f"High memory usage detected on GPU {device_id}, clearing caches")
                            self.model_optimizer.clear_cache()
                            
                            # Emit optimization event
                            await self.event_bus.emit(GPUPerformanceOptimized(
                                device_id=str(device_id),
                                action="cache_cleared",
                                previous_memory_percent=recent_stats.memory_percent,
                                reason="high_memory_usage"
                            ))
                
                # Sleep longer than monitoring loop
                await asyncio.sleep(60.0)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in GPU optimization loop: {str(e)}")
                await asyncio.sleep(30.0)  # Back off on errors
        
        self.logger.info("GPU optimization loop stopped")

    async def _collect_gpu_stats(self, device_id: int) -> Optional[GPUStats]:
        """Collect statistics for a specific GPU."""
        try:
            stats = None
            
            # NVIDIA GPUs with NVML
            if NVML_AVAILABLE:
                try:
                    handle = nvmlDeviceGetHandleByIndex(device_id)
                    
                    # Get utilization
                    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_util = utilization.gpu
                    memory_util = utilization.memory
                    
                    # Get memory info
                    memory = nvmlDeviceGetMemoryInfo(handle)
                    memory_used = memory.used / (1024 * 1024)  # Convert to MB
                    memory_total = memory.total / (1024 * 1024)
                    memory_percent = (memory.used / memory.total) * 100
                    
                    # Get temperature
                    temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    
                    # Get power usage
                    power_usage = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to watts
                    power_limit = pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0
                    
                    # Get process info
                    processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                    process_count = len(processes)
                    process_details = []
                    
                    for proc in processes:
                        try:
                            process_name = pynvml.nvmlSystemGetProcessName(proc.pid).decode('utf-8')
                            process_details.append({
                                'pid': proc.pid,
                                'name': process_name,
                                'memory_used_mb': proc.usedGpuMemory / (1024 * 1024)
                            })
                        except Exception:
                            pass
                    
                    # Create stats object
                    stats = GPUStats(
                        device_id=device_id,
                        name=pynvml.nvmlDeviceGetName(handle).decode('utf-8'),
                        vendor=GPUVendor.NVIDIA,
                        utilization_percent=gpu_util,
                        memory_used_mb=memory_used,
                        memory_total_mb=memory_total,
                        memory_percent=memory_percent,
                        temperature_c=temperature,
                        power_usage_watts=power_usage,
                        power_limit_watts=power_limit,
                        process_count=process_count,
                        process_details=process_details
                    )
                except Exception as e:
                    self.logger.debug(f"Error collecting NVML stats for GPU {device_id}: {str(e)}")
            
            # PyTorch stats (as fallback or additional info)
            if TORCH_AVAILABLE and hasattr(torch, 'cuda') and torch.cuda.is_available():
                try:
                    if device_id < torch.cuda.device_count():
                        # Get memory info if not already set
                        if stats is None:
                            memory_allocated = torch.cuda.memory_allocated(device_id) / (1024 * 1024)
                            memory_reserved = torch.cuda.memory_reserved(device_id) / (1024 * 1024)
                            
                            # For PyTorch, we use memory_allocated as memory_used
                            # and memory_reserved as memory_total
                            memory_used = memory_allocated
                            memory_total = memory_reserved if memory_reserved > 0 else memory_allocated
                            memory_percent = (memory_used / max(memory_total, 1)) * 100
                            
                            # Create basic stats object
                            stats = GPUStats(
                                device_id=device_id,
                                name=torch.cuda.get_device_name(device_id),
                                vendor=GPUVendor.NVIDIA,
                                utilization_percent=0.0,  # Not available from PyTorch
                                memory_used_mb=memory_used,
                                memory_total_mb=memory_total,
                                memory_percent=memory_percent
                            )
                        
                        # Add PyTorch-specific information
                        stats.framework = GPUFramework.PYTORCH
                        stats.framework_version = torch.__version__
                        stats.cuda_version = torch.version.cuda if hasattr(torch.version, 'cuda') else None
                        
                        # Get tensor information if available
                        if hasattr(torch.cuda, 'memory_stats'):
                            memory_stats = torch.cuda.memory_stats(device_id)
                            stats.active_tensors = memory_stats.get('num_alloc_retries', 0)
                            stats.tensor_memory_mb = memory_allocated
                except Exception as e:
                    self.logger.debug(f"Error collecting PyTorch stats for GPU {device_id}: {str(e)}")
            
            # TensorFlow stats (as additional info)
            if TF_AVAILABLE:
                try:
                    # There's no direct API to get per-GPU stats in TensorFlow
                    # We could parse the output of tf.config.experimental.get_memory_info
                    # but it's not reliable across different TF versions
                    pass
                except Exception:
                    pass
            
            # Apple Metal stats
            if platform.system() == 'Darwin' and hasattr(platform, 'mac_ver'):
                try:
                    if TORCH_AVAILABLE and hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and \
                       torch.backends.mps.is_available() and device_id == 0:
                        
                        # Get basic Metal memory info (very limited)
                        memory_used = 0
                        memory_total = 0
                        
                        # If we have a Metal tensor, we can get some memory info
                        if hasattr(torch, 'mps') and hasattr(torch.mps, 'current_allocated_memory'):
                            memory_used = torch.mps.current_allocated_memory() / (1024 * 1024)
                        
                        # Create basic stats object for Metal
                        stats = GPUStats(
                            device_id=device_id,
                            name="Apple M-series GPU",
                            vendor=GPUVendor.APPLE,
                            utilization_percent=0.0,  # Not available
                            memory_used_mb=memory_used,
                            memory_total_mb=memory_total,
                            memory_percent=0.0 if memory_total == 0 else (memory_used / memory_total) * 100,
                            framework=GPUFramework.METAL
                        )
                except Exception as e:
                    self.logger.debug(f"Error collecting Metal stats: {str(e)}")
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error collecting GPU stats for device {device_id}: {str(e)}")
            return None

    def _report_metrics(self, stats: GPUStats) -> None:
        """Report GPU metrics to the metrics collector."""
        try:
            # Basic metrics
            self.metrics.set("gpu_utilization_percent", stats.utilization_percent, 
                         tags={"device_id": str(stats.device_id), "name": stats.name})
            self.metrics.set("gpu_memory_used_mb", stats.memory_used_mb, 
                         tags={"device_id": str(stats.device_id), "name": stats.name})
            self.metrics.set("gpu_memory_total_mb", stats.memory_total_mb, 
                         tags={"device_id": str(stats.device_id), "name": stats.name})
            self.metrics.set("gpu_memory_percent", stats.memory_percent, 
                         tags={"device_id": str(stats.device_id), "name": stats.name})
            
            # Optional metrics
            if stats.temperature_c is not None:
                self.metrics.set("gpu_temperature_c", stats.temperature_c, 
                             tags={"device_id": str(stats.device_id), "name": stats.name})
            
            if stats.power_usage_watts is not None:
                self.metrics.set("gpu_power_usage_watts", stats.power_usage_watts, 
                             tags={"device_id": str(stats.device_id), "name": stats.name})
            
            # Process metrics
            self.metrics.set("gpu_process_count", stats.process_count, 
                         tags={"device_id": str(stats.device_id), "name": stats.name})
            
            # Framework-specific metrics
            if stats.framework == GPUFramework.PYTORCH:
                self.metrics.set("gpu_pytorch_tensors", stats.active_tensors, 
                             tags={"device_id": str(stats.device_id), "name": stats.name})
                self.metrics.set("gpu_pytorch_memory_mb", stats.tensor_memory_mb, 
                             tags={"device_id": str(stats.device_id), "name": stats.name})
            
        except Exception as e:
            self.logger.warning(f"Failed to report GPU metrics: {str(e)}")

    async def _check_thresholds(self, stats: GPUStats) -> None:
        """Check thresholds and emit events if needed."""
        if not self.config.emit_events:
            return
        
        # Check memory threshold
        if stats.memory_percent > self.config.memory_critical_threshold:
            await self.event_bus.emit(GPUMemoryThresholdExceeded(
                device_id=str(stats.device_id),
                memory_percent=stats.memory_percent,
                threshold=self.config.memory_critical_threshold,
                severity="critical"
            ))
        elif stats.memory_percent > self.config.memory_warning_threshold:
            await self.event_bus.emit(GPUMemoryThresholdExceeded(
                device_id=str(stats.device_id),
                memory_percent=stats.memory_percent,
                threshold=self.config.memory_warning_threshold,
                severity="warning"
            ))
        
        # Check utilization threshold
        if stats.utilization_percent > self.config.utilization_warning_threshold:
            await self.event_bus.emit(GPUUsageThresholdExceeded(
                device_id=str(stats.device_id),
                utilization_percent=stats.utilization_percent,
                threshold=self.config.utilization_warning_threshold,
                severity="warning"
            ))
        
        # Check temperature threshold if available
        if stats.temperature_c is not None:
            if stats.temperature_c > self.config.temperature_critical_threshold_c:
                await self.event_bus.emit(GPUErrorDetected(
                    device_id=str(stats.device_id),
                    error_type="high_temperature",
                    value=stats.temperature_c,
                    threshold=self.config.temperature_critical_threshold_c,
                    severity="critical"
                ))
            elif stats.temperature_c > self.config.temperature_warning_threshold_c:
                await self.event_bus.emit(GPUErrorDetected(
                    device_id=str(stats.device_id),
                    error_type="high_temperature",
                    value=stats.temperature_c,
                    threshold=self.config.temperature_warning_threshold_c,
                    severity="warning"
                ))
        
        # Check for throttling
        if stats.throttling_detected:
            await self.event_bus.emit(GPUErrorDetected(
                device_id=str(stats.device_id),
                error_type="throttling",
                value=True,
                threshold=None,
                severity="warning"
            ))

    def _save_history(self) -> None:
        """Save GPU stats history to file."""
        if not self.config.history_file_path:
            return
        
        try:
            history_data = {}
            
            for device_id, stats_list in self.gpu_stats.items():
                history_data[str(device_id)] = [asdict(stats) for stats in stats_list]
            
            os.makedirs(os.path.dirname(self.config.history_file_path), exist_ok=True)
            
            with open(self.config.history_file_path, 'w') as f:
                json.dump(history_data, f, default=str)
            
        except Exception as e:
            self.logger.warning(f"Failed to save GPU history: {str(e)}")

    async def _health_check_callback(self) -> Dict[str, Any]:
        """Health check callback for the GPU profiler."""
        try:
            gpu_health = {}
            overall_status = "healthy"
            
            for gpu in self.available_gpus:
                device_id = gpu["device_id"]
                
                if device_id in self.gpu_stats and self.gpu_stats[device_id]:
                    recent_stats = self.gpu_stats[device_id][-1]
                    
                    # Determine GPU health status
                    if recent_stats.memory_percent > self.config.memory_critical_threshold:
                        status = "critical"
                        overall_status = "degraded"
                    elif recent_stats.memory_percent > self.config.memory_warning_threshold:
                        status = "warning"
                        if overall_status != "critical":
                            overall_status = "degraded"
                    elif recent_stats.throttling_detected:
                        status = "warning"
                        if overall_status != "critical":
                            overall_status = "degraded"
                    else:
                        status = "healthy"
                    
                    gpu_health[str(device_id)] = {
                        "status": status,
                        "memory_percent": recent_stats.memory_percent,
                        "utilization_percent": recent_stats.utilization_percent,
                        "temperature_c": recent_stats.temperature_c,
                        "processes": recent_stats.process_count
                    }
            
            return {
                "status": overall_status,
                "gpu_count": len(self.available_gpus),
                "gpus": gpu_health,
                "profiling_mode": self.config.profiling_mode.value,
                "frameworks_detected": [f.value for f in self.detected_frameworks]
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

    @contextmanager
    def profile_operation(self, operation_name: str) -> None:
        """Profile a specific operation (for use with 'with' statement)."""
        if not self.config.enabled or not TORCH_AVAILABLE:
            yield
            return
        
        # Start profiling
        start_time = time.time()
        start_memory = {}
        
        # Record initial memory for each GPU
        for device_id in range(torch.cuda.device_count() if torch.cuda.
>>>>>>> main
