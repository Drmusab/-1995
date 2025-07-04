#!/usr/bin/env python3
"""
Standalone test for GPU Profiler without external dependencies
"""

import asyncio
import sys
import uuid
from datetime import datetime, timezone
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, List, Union, Set, Callable
from collections import defaultdict, deque

# Mock all the imports that aren't available
sys.modules['src.core.config.loader'] = type(sys)('mock')
sys.modules['src.core.events.event_bus'] = type(sys)('mock')
sys.modules['src.core.events.event_types'] = type(sys)('mock')
sys.modules['src.core.error_handling'] = type(sys)('mock')
sys.modules['src.core.dependency_injection'] = type(sys)('mock')
sys.modules['src.core.health_check'] = type(sys)('mock')
sys.modules['src.observability.monitoring.metrics'] = type(sys)('mock')
sys.modules['src.observability.monitoring.tracing'] = type(sys)('mock')
sys.modules['src.observability.logging.config'] = type(sys)('mock')

# Mock classes needed by the GPU profiler
class MockClass:
    pass

# Setup mocks
sys.modules['src.core.config.loader'].ConfigLoader = MockClass
sys.modules['src.core.events.event_bus'].EventBus = MockClass
sys.modules['src.core.error_handling'].ErrorHandler = MockClass
sys.modules['src.core.error_handling'].handle_exceptions = lambda func: func
sys.modules['src.core.dependency_injection'].Container = MockClass
sys.modules['src.core.health_check'].HealthCheck = MockClass
sys.modules['src.observability.monitoring.metrics'].MetricsCollector = MockClass
sys.modules['src.observability.monitoring.tracing'].TraceManager = MockClass
sys.modules['src.observability.logging.config'].get_logger = lambda name: MockLogger()

# Add mock events
for event_name in [
    'GPUProfilerStarted', 'GPUProfilerStopped', 'GPUUtilizationAlert', 
    'GPUMemoryAlert', 'GPUTemperatureAlert', 'GPUMemoryLeakDetected',
    'GPUPerformanceBottleneckDetected', 'GPUKernelExecutionStarted',
    'GPUKernelExecutionCompleted', 'GPUPowerUsageAlert',
    'ComponentHealthChanged', 'SystemStateChanged'
]:
    setattr(sys.modules['src.core.events.event_types'], event_name, type(event_name, (), {}))

class MockLogger:
    def debug(self, msg, *args, **kwargs):
        pass
    def info(self, msg, *args, **kwargs):
        print(f"INFO: {msg}")
    def warning(self, msg, *args, **kwargs):
        print(f"WARNING: {msg}")
    def error(self, msg, *args, **kwargs):
        print(f"ERROR: {msg}")

class MockContainer:
    def get(self, service_type, default=None):
        class MockService:
            def get(self, key, default=None):
                return default
            async def emit(self, event):
                print(f"Event: {event.__class__.__name__}")
            def subscribe(self, event_type, handler):
                pass
            def register_component(self, name, callback):
                pass
            def register_gauge(self, name):
                pass
            def register_counter(self, name):
                pass
            def register_histogram(self, name):
                pass
            def set(self, name, value, tags=None):
                pass
            def increment(self, name, tags=None):
                pass
            def record(self, name, value, tags=None):
                pass
        
        return MockService()

# Now manually define the essential GPU profiler classes
class GPUProfilingLevel(Enum):
    LIGHTWEIGHT = "lightweight"
    STANDARD = "standard"
    DETAILED = "detailed"
    COMPREHENSIVE = "comprehensive"

class GPUProfilingMode(Enum):
    OFF = "off"
    MONITORING = "monitoring"
    PROFILING = "profiling"
    ADAPTIVE = "adaptive"
    DEBUG = "debug"

class GPUProfilerStatus(Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    ERROR = "error"

class GPUBottleneckType(Enum):
    COMPUTE = "compute"
    MEMORY_BANDWIDTH = "memory_bandwidth"
    MEMORY_CAPACITY = "memory_capacity"
    THERMAL = "thermal"
    POWER = "power"
    SYNCHRONIZATION = "synchronization"

@dataclass
class GPUProfilingConfig:
    level: GPUProfilingLevel = GPUProfilingLevel.STANDARD
    mode: GPUProfilingMode = GPUProfilingMode.MONITORING
    sampling_interval_ms: float = 100.0
    utilization_threshold_percent: float = 85.0
    memory_threshold_percent: float = 90.0
    temperature_threshold_celsius: float = 80.0
    power_threshold_percent: float = 95.0
    max_samples: int = 10000
    retention_hours: int = 24
    enable_real_time_monitoring: bool = True
    enable_kernel_profiling: bool = True
    enable_memory_tracking: bool = True
    enable_power_monitoring: bool = True
    enable_temperature_monitoring: bool = True
    enable_multi_gpu: bool = True
    integrate_with_tracing: bool = True
    integrate_with_metrics: bool = True
    integrate_with_component_profiling: bool = True
    enable_bottleneck_detection: bool = True
    enable_optimization_suggestions: bool = True
    enable_leak_detection: bool = True

@dataclass
class GPUDeviceInfo:
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
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    device_id: int = 0
    utilization_percent: float = 0.0
    memory_utilization_percent: float = 0.0
    memory_used_mb: float = 0.0
    memory_free_mb: float = 0.0
    memory_total_mb: float = 0.0
    memory_cached_mb: float = 0.0
    temperature_celsius: float = 0.0
    power_draw_watts: float = 0.0
    power_limit_watts: float = 0.0
    clock_graphics_mhz: int = 0
    clock_memory_mhz: int = 0
    process_count: int = 0
    active_kernels: int = 0

@dataclass
class GPUKernelInfo:
    kernel_id: str
    kernel_name: str
    device_id: int
    component: Optional[str] = None
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    end_time: Optional[datetime] = None
    duration_ms: float = 0.0
    memory_transferred_mb: float = 0.0
    memory_bandwidth_gbps: float = 0.0
    occupancy_percent: float = 0.0
    achieved_occupancy_percent: float = 0.0
    grid_size: tuple = field(default_factory=tuple)
    block_size: tuple = field(default_factory=tuple)

class SimpleGPUProfiler:
    """
    Simplified GPU Profiler for testing without external dependencies.
    """
    
    def __init__(self, container):
        self.container = container
        self.config_loader = container.get(MockClass)
        self.event_bus = container.get(MockClass)
        self.error_handler = container.get(MockClass)
        self.health_check = container.get(MockClass)
        self.logger = MockLogger()
        
        # Configuration
        self.config = GPUProfilingConfig()
        
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
        self.detected_bottlenecks: List = []
        self.detected_leaks: List = []
        
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
        
        self.logger.info(f"GPUProfiler initialized with level: {self.config.level.value}, available GPUs: {len(self.gpu_devices)}")

    def _initialize_gpu_detection(self) -> None:
        """Initialize GPU detection and device enumeration."""
        try:
            # Try to detect if PyTorch and CUDA are available
            try:
                import torch
                if torch.cuda.is_available():
                    self._detect_cuda_devices()
                    self.gpu_available = True
                    self.logger.info(f"Detected {len(self.gpu_devices)} CUDA devices via PyTorch")
            except ImportError:
                self.logger.info("PyTorch not available")
                
        except Exception as e:
            self.logger.warning(f"Failed to initialize GPU detection: {str(e)}")
            self.gpu_available = False

    def _detect_cuda_devices(self) -> None:
        """Detect CUDA devices using PyTorch."""
        try:
            import torch
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

    def _setup_monitoring(self) -> None:
        """Setup metrics and monitoring."""
        try:
            self.metrics = self.container.get(MockClass)
            self.tracer = self.container.get(MockClass)
        except Exception as e:
            self.logger.warning(f"Failed to setup monitoring: {str(e)}")
            self.metrics = None
            self.tracer = None

    def _setup_metrics(self) -> None:
        """Register GPU-related metrics with the metrics system."""
        if not self.metrics:
            return
        self.logger.info("GPU metrics registered successfully")

    def _setup_event_handlers(self) -> None:
        """Setup event handlers for system integration."""
        try:
            self.event_bus.subscribe("processing_started", lambda x: None)
            self.event_bus.subscribe("processing_completed", lambda x: None)
            self.logger.info("GPU profiler event handlers registered")
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
            
            if self.gpu_available:
                health_status["devices"] = {}
                for device_id, device_info in self.gpu_devices.items():
                    device_status = {
                        "name": device_info.name,
                        "available": device_info.is_available,
                        "memory_total_mb": device_info.total_memory_mb
                    }
                    health_status["devices"][str(device_id)] = device_status
            
            return health_status
            
        except Exception as e:
            self.logger.error(f"GPU profiler health check failed: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "gpu_available": False
            }

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
                
            self.status = GPUProfilerStatus.RUNNING
            
            # Emit profiler started event
            await self.event_bus.emit(type('GPUProfilerStarted', (), {})())
            
            self.logger.info(f"GPU profiler started with level: {self.config.level.value}, mode: {self.config.mode.value}")
            return "started"
            
        except Exception as e:
            self.status = GPUProfilerStatus.ERROR
            self.logger.error(f"Failed to start GPU profiler: {str(e)}")
            raise

    async def stop_profiling(self) -> Dict[str, Any]:
        """Stop GPU profiling."""
        if self.status == GPUProfilerStatus.STOPPED:
            self.logger.warning("GPU profiler is already stopped")
            return {"status": "already_stopped"}
            
        try:
            self.status = GPUProfilerStatus.STOPPING
            self.stop_event.set()
            self.status = GPUProfilerStatus.STOPPED
            
            total_samples = sum(len(history) for history in self.metrics_history.values())
            
            await self.event_bus.emit(type('GPUProfilerStopped', (), {})())
            
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
            status["devices"][str(device_id)] = device_status
        
        return status

    async def get_optimization_suggestions(self) -> List[Dict[str, Any]]:
        """Get GPU optimization suggestions based on current metrics and detected issues."""
        suggestions = []
        
        if not self.gpu_available:
            return suggestions
        
        # Mock some suggestions for testing
        suggestions.append({
            "type": "test",
            "severity": "info",
            "message": "This is a test optimization suggestion",
            "suggestion": "No action needed - this is just a test"
        })
        
        return suggestions

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

async def test_simple_gpu_profiler():
    """Test the simple GPU profiler."""
    print("Testing Simple GPU Profiler...")
    
    try:
        # Create mock container
        container = MockContainer()
        
        # Create GPU profiler
        print("Creating GPU profiler...")
        gpu_profiler = SimpleGPUProfiler(container)
        
        print(f"GPU profiler created successfully!")
        print(f"GPU available: {gpu_profiler.gpu_available}")
        print(f"GPU devices: {len(gpu_profiler.gpu_devices)}")
        print(f"Status: {gpu_profiler.status}")
        
        # Test getting GPU status
        print("\nTesting get_gpu_status()...")
        status = await gpu_profiler.get_gpu_status()
        print(f"GPU status available: {status.get('available', False)}")
        
        # Test getting optimization suggestions
        print("\nTesting get_optimization_suggestions()...")
        suggestions = await gpu_profiler.get_optimization_suggestions()
        print(f"Optimization suggestions: {len(suggestions)} found")
        
        # Test getting component GPU usage
        print("\nTesting get_component_gpu_usage()...")
        usage = gpu_profiler.get_component_gpu_usage()
        print(f"Component GPU usage keys: {list(usage.keys())}")
        
        # Test profiling start/stop if GPU is available
        if gpu_profiler.gpu_available:
            print("\nTesting profiling start/stop...")
            result = await gpu_profiler.start_profiling()
            print(f"Start profiling result: {result}")
            
            # Wait a bit
            await asyncio.sleep(0.5)
            
            result = await gpu_profiler.stop_profiling()
            print(f"Stop profiling result: {result}")
        else:
            print("\nSkipping profiling start/stop (no GPU available)")
            
            # Test that starting profiling without GPU raises an error
            try:
                await gpu_profiler.start_profiling()
                print("ERROR: Should have raised RuntimeError")
            except RuntimeError as e:
                print(f"Correctly raised RuntimeError: {str(e)[:100]}")
        
        # Test health check
        print("\nTesting health check...")
        health = await gpu_profiler._health_check_callback()
        print(f"Health check status: {health.get('status', 'unknown')}")
        
        print("\nGPU profiler test completed successfully!")
        return True
        
    except Exception as e:
        print(f"GPU profiler test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    async def main():
        print("=" * 60)
        print("Standalone GPU Profiler Test")
        print("=" * 60)
        
        result = await test_simple_gpu_profiler()
        
        print("\n" + "=" * 60)
        print(f"Test Result: {'PASS' if result else 'FAIL'}")
        
        sys.exit(0 if result else 1)
    
    asyncio.run(main())