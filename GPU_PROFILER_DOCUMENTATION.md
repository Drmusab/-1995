# GPU Profiler Documentation

## Overview

The GPU Profiler is a comprehensive GPU monitoring and profiling system for the AI Assistant. It provides real-time GPU performance monitoring, memory tracking, thermal management, and optimization suggestions with seamless integration into the AI assistant's core systems.

## Features

### Core Functionality
- **GPU Performance Monitoring**: Real-time tracking of GPU utilization, memory usage, temperature, and power consumption
- **CUDA/OpenCL Support**: Support for NVIDIA CUDA and OpenCL devices with automatic detection
- **Multi-GPU Support**: Monitor multiple GPU configurations simultaneously
- **Real-time Monitoring**: Continuous monitoring with configurable intervals
- **Memory Tracking**: Comprehensive GPU memory allocation, deallocation, and fragmentation tracking
- **Performance Metrics**: Measure inference times, throughput, and efficiency

### Integration Features
- **Core Engine Integration**: Seamless integration with the AI assistant's core processing engine
- **Component Manager**: Automatic registration as a monitored component
- **Event System**: Emit GPU-related events and respond to system events
- **Session Manager**: Track GPU usage per user/processing session
- **Workflow Orchestrator**: Monitor GPU usage during workflow execution
- **Metrics Collection**: Export comprehensive metrics to the monitoring system
- **Health Monitoring**: Continuous health status reporting
- **Configuration**: Environment-specific settings and customization

### Advanced Features
- **Profiling Sessions**: Start/stop profiling sessions with detailed reports
- **Performance Analysis**: Automated bottleneck detection and optimization analysis
- **Memory Leak Detection**: Detect and report GPU memory leaks
- **Thermal Management**: Monitor and alert on temperature thresholds
- **Performance Benchmarking**: Built-in benchmarks for performance validation
- **Resource Optimization**: AI-driven suggestions for GPU resource optimization

### Error Handling & Security
- **Graceful Degradation**: Continue operation when GPU monitoring is unavailable
- **Driver Compatibility**: Handle different GPU driver versions gracefully
- **Fallback Monitoring**: Use system tools when direct GPU access fails
- **Data Sanitization**: Ensure no sensitive data in profiling outputs
- **Access Control**: Secure access to GPU profiling data
- **Audit Logging**: Log all profiling activities

## Architecture

The GPU Profiler follows a modular architecture with the following components:

```
GPUProfiler
├── Device Discovery Layer
│   ├── NVIDIA GPU Detection (via pynvml)
│   ├── OpenCL Device Detection
│   └── PyTorch CUDA Integration
├── Monitoring Engine
│   ├── Real-time Metrics Collection
│   ├── Performance Tracking
│   └── Alert Management
├── Session Management
│   ├── Profiling Sessions
│   ├── Snapshot Management
│   └── Report Generation
├── Integration Layer
│   ├── Event System Integration
│   ├── Metrics System Integration
│   └── Configuration Management
└── Analysis Engine
    ├── Performance Analysis
    ├── Optimization Suggestions
    └── Health Assessment
```

## Installation & Dependencies

### Required Dependencies
- `pynvml` - NVIDIA GPU monitoring (optional, graceful fallback)
- `pyopencl` - OpenCL support (optional, graceful fallback)
- `psutil` - System monitoring
- `torch` - PyTorch CUDA integration (optional)

### Installation
```bash
pip install pynvml pyopencl psutil torch
```

## Usage

### Basic Usage

```python
from src.observability.profiling import GPUProfiler
import asyncio

async def main():
    # Create profiler instance
    profiler = GPUProfiler()
    
    # Start profiler
    await profiler.start()
    
    # Start a profiling session
    session_id = await profiler.start_profiling_session(
        "my_session", 
        "Description of the profiling session"
    )
    
    # Track inference operations
    await profiler.track_inference(150.0)  # 150ms inference time
    
    # Take a snapshot
    snapshot = await profiler.take_snapshot("checkpoint")
    
    # Get current status
    status = await profiler.get_gpu_status()
    print(f"GPU Status: {status}")
    
    # Stop profiling session
    await profiler.stop_profiling_session()
    
    # Stop profiler
    await profiler.stop()
```

### Integration with Dependency Injection

```python
from src.core.dependency_injection import Container
from src.observability.profiling import GPUProfiler

# With dependency injection container
container = Container()
profiler = GPUProfiler(container)  # Full integration mode

# Without container (standalone mode)
profiler = GPUProfiler()  # Graceful fallback mode
```

### Advanced Configuration

```python
from src.observability.profiling.gpu_profiler import GPUProfilerConfig, ProfilingMode, ProfilingLevel

# Custom configuration
config = GPUProfilerConfig(
    mode=ProfilingMode.COMPREHENSIVE,
    level=ProfilingLevel.HIGH,
    monitoring_interval_seconds=0.5,
    temperature_warning_celsius=75.0,
    temperature_critical_celsius=85.0,
    enable_thermal_monitoring=True,
    enable_power_monitoring=True,
    export_json_reports=True
)

# Start session with custom config
session_id = await profiler.start_profiling_session(
    "detailed_session",
    "High-detail profiling",
    config
)
```

## Configuration Options

### GPUProfilerConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mode` | ProfilingMode | STANDARD | Profiling mode (BASIC, STANDARD, DETAILED, COMPREHENSIVE) |
| `level` | ProfilingLevel | MEDIUM | Profiling level (LOW, MEDIUM, HIGH, MAXIMUM) |
| `monitoring_interval_seconds` | float | 1.0 | Interval for device metrics updates |
| `metrics_update_interval_seconds` | float | 5.0 | Interval for system metrics updates |
| `device_discovery_interval_seconds` | float | 30.0 | Interval for device discovery |
| `enable_cuda_profiling` | bool | True | Enable NVIDIA CUDA monitoring |
| `enable_opencl_profiling` | bool | True | Enable OpenCL monitoring |
| `enable_thermal_monitoring` | bool | True | Enable temperature monitoring |
| `enable_power_monitoring` | bool | True | Enable power consumption monitoring |
| `enable_memory_tracking` | bool | True | Enable memory usage tracking |
| `temperature_warning_celsius` | float | 80.0 | Temperature warning threshold |
| `temperature_critical_celsius` | float | 90.0 | Temperature critical threshold |
| `memory_warning_percent` | float | 80.0 | Memory usage warning threshold |
| `memory_critical_percent` | float | 95.0 | Memory usage critical threshold |
| `max_snapshots` | int | 1000 | Maximum snapshots to retain |
| `max_profiling_sessions` | int | 100 | Maximum sessions to retain |
| `output_dir` | str | "data/profiling/gpu" | Output directory for reports |
| `export_json_reports` | bool | True | Enable JSON report export |

## API Reference

### GPUProfiler Class

#### Methods

##### `__init__(container: Container = None)`
Initialize the GPU profiler with optional dependency injection container.

##### `async start() -> None`
Start the GPU profiler and begin monitoring.

##### `async stop() -> None`
Stop the GPU profiler and cleanup resources.

##### `async start_profiling_session(session_name: str = None, description: str = None, config: GPUProfilerConfig = None) -> str`
Start a new profiling session and return the session ID.

##### `async stop_profiling_session() -> Optional[str]`
Stop the current profiling session and return the session ID.

##### `async take_snapshot(trigger: str = "manual") -> GPUSnapshot`
Take a snapshot of the current GPU state.

##### `async track_inference(inference_time_ms: float, session_id: str = None) -> None`
Track an inference operation with its execution time.

##### `async get_gpu_status() -> Dict[str, Any]`
Get current GPU status and metrics.

##### `async get_profiling_sessions() -> List[Dict[str, Any]]`
Get list of recent profiling sessions.

##### `async health_check() -> Dict[str, Any]`
Perform a health check and return status information.

### Data Structures

#### GPUDevice
Represents information about a GPU device.

```python
@dataclass
class GPUDevice:
    index: int
    name: str
    vendor: GPUVendor
    gpu_type: GPUType
    total_memory_mb: float
    free_memory_mb: float
    used_memory_mb: float
    utilization_percent: float
    memory_utilization_percent: float
    temperature_celsius: Optional[float]
    power_draw_watts: Optional[float]
    # ... additional fields
```

#### GPUSnapshot
Represents a snapshot of GPU state at a point in time.

```python
@dataclass
class GPUSnapshot:
    timestamp: datetime
    devices: List[GPUDevice]
    total_gpu_memory_mb: float
    total_gpu_memory_used_mb: float
    average_gpu_utilization: float
    max_temperature_celsius: Optional[float]
    inference_count: int
    # ... additional fields
```

#### GPUProfilingSession
Represents a profiling session with metadata and results.

```python
@dataclass
class GPUProfilingSession:
    session_id: str
    name: str
    description: Optional[str]
    start_time: datetime
    end_time: Optional[datetime]
    duration_seconds: float
    snapshots: List[GPUSnapshot]
    optimization_suggestions: List[Dict[str, Any]]
    # ... additional fields
```

## Integration Points

### Event System Integration

The GPU profiler emits the following events:

- `ProfilerStarted` - When profiler starts
- `ProfilerStopped` - When profiler stops
- `PerformanceAlertTriggered` - When performance alerts are triggered
- `ProfilingDataGenerated` - When profiling data is generated

It also listens to system events:

- `session_started` - Automatically starts profiling
- `session_ended` - Automatically stops profiling
- `inference_completed` - Tracks inference times
- `system_shutdown_started` - Graceful shutdown

### Metrics Integration

The GPU profiler registers the following metrics:

- `gpu_device_count` - Number of GPU devices
- `gpu_total_memory_mb` - Total GPU memory
- `gpu_total_memory_used_mb` - Used GPU memory
- `gpu_average_utilization_percent` - Average GPU utilization
- `gpu_max_temperature_celsius` - Maximum GPU temperature
- `gpu_inference_time_ms` - Inference time histogram
- `gpu_thermal_alerts_total` - Total thermal alerts
- `gpu_profiling_sessions_total` - Total profiling sessions

### Configuration Integration

GPU profiler settings can be configured via the central configuration system:

```yaml
gpu_profiler:
  mode: "comprehensive"
  level: "high"
  monitoring_interval_seconds: 1.0
  enable_thermal_monitoring: true
  temperature_warning_celsius: 75.0
  temperature_critical_celsius: 85.0
  output_dir: "data/profiling/gpu"
```

## Monitoring & Alerts

### Alert Types

1. **Thermal Alerts**
   - Warning: Temperature > 80°C (configurable)
   - Critical: Temperature > 90°C (configurable)

2. **Memory Alerts**
   - Warning: Memory usage > 80% (configurable)
   - Critical: Memory usage > 95% (configurable)

3. **Performance Alerts**
   - Warning: GPU utilization > 90% (configurable)

### Optimization Suggestions

The system automatically generates optimization suggestions based on profiling data:

- **Low GPU Utilization**: Suggests batch processing or model optimization
- **High Memory Usage**: Recommends memory-efficient techniques
- **High Temperature**: Suggests cooling improvements
- **Slow Inference**: Recommends model optimization techniques

## Error Handling

### Graceful Degradation

The GPU profiler is designed to gracefully handle various failure scenarios:

1. **No GPU Hardware**: Continues operation with 0 devices
2. **Driver Issues**: Falls back to basic monitoring
3. **Permission Issues**: Logs warnings and continues
4. **Library Missing**: Disables specific monitoring features

### Error Recovery

- Automatic retry for transient failures
- Fallback to alternative monitoring methods
- Comprehensive error logging and reporting

## Performance Considerations

### Overhead

- **Low Mode**: < 1% CPU overhead
- **Medium Mode**: < 2% CPU overhead  
- **High Mode**: < 5% CPU overhead
- **Maximum Mode**: < 10% CPU overhead

### Memory Usage

- Base memory usage: ~10-20MB
- Per device overhead: ~1-2MB
- Snapshot storage: ~1KB per snapshot

### Recommendations

1. Use **Medium** level for production environments
2. Use **High** or **Maximum** level for debugging
3. Configure appropriate snapshot retention limits
4. Monitor profiler overhead in resource-constrained environments

## Troubleshooting

### Common Issues

#### No GPU Devices Detected

```
WARNING - No GPU devices discovered
```

**Solutions:**
1. Verify GPU drivers are installed
2. Check GPU hardware is present
3. Ensure user has proper permissions
4. Install NVIDIA drivers for CUDA support

#### NVML Library Not Found

```
WARNING - Failed to initialize NVIDIA GPU monitoring: NVML Shared Library Not Found
```

**Solutions:**
1. Install NVIDIA drivers
2. Install `nvidia-ml-py`: `pip install nvidia-ml-py`
3. Set `LD_LIBRARY_PATH` if needed

#### Permission Denied

```
ERROR - Failed to access GPU device
```

**Solutions:**
1. Add user to `video` or `render` group
2. Check device permissions in `/dev/nvidia*`
3. Run with appropriate privileges

### Debug Mode

Enable detailed logging for troubleshooting:

```python
import logging
logging.getLogger('src.observability.profiling.gpu_profiler').setLevel(logging.DEBUG)
```

## Examples

### Complete Example

See `demo_gpu_profiler.py` for a comprehensive demonstration of all features.

### Integration Example

```python
from src.core.dependency_injection import Container
from src.observability.profiling import GPUProfiler

class AIProcessor:
    def __init__(self, container: Container):
        self.gpu_profiler = container.get(GPUProfiler)
        
    async def process_request(self, request):
        # Start profiling session
        session_id = await self.gpu_profiler.start_profiling_session(
            f"request_{request.id}",
            f"Processing request {request.id}"
        )
        
        try:
            # Process AI request
            start_time = time.time()
            result = await self.run_inference(request)
            inference_time = (time.time() - start_time) * 1000
            
            # Track inference
            await self.gpu_profiler.track_inference(inference_time)
            
            return result
            
        finally:
            # Stop profiling session
            await self.gpu_profiler.stop_profiling_session()
```

## License

This GPU Profiler is part of the AI Assistant project and follows the same licensing terms.

## Contributing

When contributing to the GPU profiler:

1. Ensure all tests pass: `python test_gpu_profiler.py`
2. Follow existing code patterns and style
3. Add tests for new functionality
4. Update documentation for API changes
5. Test with and without GPU hardware
6. Verify graceful degradation works properly

## Version History

### v1.0.0 (2025-01-10)
- Initial implementation
- CUDA and OpenCL support
- Comprehensive monitoring and profiling
- Integration with core AI assistant systems
- Advanced optimization suggestions
- Graceful degradation and error handling