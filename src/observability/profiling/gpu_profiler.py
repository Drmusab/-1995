"""
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
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

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


class GPUVendor(Enum):
    """GPU vendor types."""
    NVIDIA = "nvidia"
    AMD = "amd"
    INTEL = "intel"
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
