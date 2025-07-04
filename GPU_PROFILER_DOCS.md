# GPU Profiler Documentation

The GPU Profiler (`src/observability/profiling/gpu_profiler.py`) provides comprehensive GPU performance monitoring and profiling capabilities for the AI Assistant system.

## Features

### Core Capabilities
- **Real-time GPU monitoring**: Utilization, memory, temperature, power consumption
- **CUDA kernel profiling**: Track kernel execution times and performance
- **Multi-GPU support**: Monitor and profile across multiple GPU devices
- **Memory leak detection**: Identify and track GPU memory leaks
- **Performance bottleneck detection**: Automatically identify performance issues
- **Component-specific profiling**: Track GPU usage by system components
- **Integration with core systems**: Events, metrics, tracing, health checks

### GPU Monitoring Metrics
- GPU compute utilization percentage
- GPU memory utilization and allocation
- GPU temperature monitoring
- Power draw and power limit tracking
- Clock speeds (graphics and memory)
- Active process and kernel counts

### Profiling Levels
- **Lightweight**: Basic GPU usage tracking
- **Standard**: GPU + memory monitoring (default)
- **Detailed**: + kernel profiling and detailed analysis
- **Comprehensive**: + power, temperature, advanced features

### Profiling Modes
- **Monitoring**: Continuous background monitoring
- **Profiling**: Detailed profiling sessions
- **Adaptive**: Automatic mode switching based on workload
- **Debug**: Verbose debugging information

## Configuration

The GPU profiler is configured through the main configuration system:

```yaml
gpu_profiler:
  level: "standard"                    # lightweight, standard, detailed, comprehensive
  mode: "monitoring"                   # monitoring, profiling, adaptive, debug
  sampling_interval_ms: 100.0         # Sampling interval in milliseconds
  utilization_threshold: 85.0         # GPU utilization alert threshold (%)
  memory_threshold: 90.0              # Memory usage alert threshold (%)
  temperature_threshold: 80.0         # Temperature alert threshold (°C)
  power_threshold: 95.0               # Power usage alert threshold (%)
  max_samples: 10000                  # Maximum samples to retain
  retention_hours: 24                 # Data retention period
  
  # Feature toggles
  enable_real_time_monitoring: true
  enable_kernel_profiling: true
  enable_memory_tracking: true
  enable_power_monitoring: true
  enable_temperature_monitoring: true
  enable_multi_gpu: true
  
  # Integration settings
  integrate_with_tracing: true
  integrate_with_metrics: true
  integrate_with_component_profiling: true
  
  # Advanced features
  enable_bottleneck_detection: true
  enable_optimization_suggestions: true
  enable_leak_detection: true
```

## Usage

### Basic Usage

```python
from src.core.dependency_injection import Container
from src.observability.profiling import GPUProfiler

# Initialize with dependency injection container
container = Container()
gpu_profiler = GPUProfiler(container)

# Start profiling
await gpu_profiler.start_profiling()

# Your GPU-intensive code here
# ...

# Stop profiling and get results
results = await gpu_profiler.stop_profiling()
print(f"Profiling completed: {results}")
```

### Advanced Usage

```python
# Start profiling with specific configuration
await gpu_profiler.start_profiling(
    level=GPUProfilingLevel.DETAILED,
    mode=GPUProfilingMode.PROFILING
)

# Get current GPU status
status = await gpu_profiler.get_gpu_status()
print(f"GPU devices: {status['device_count']}")

# Get optimization suggestions
suggestions = await gpu_profiler.get_optimization_suggestions()
for suggestion in suggestions:
    print(f"{suggestion['severity']}: {suggestion['message']}")

# Get component-specific GPU usage
usage = gpu_profiler.get_component_gpu_usage("workflow_engine")
print(f"Workflow engine GPU usage: {usage}")

# Profile specific kernel execution
with gpu_profiler.profile_kernel("my_kernel", device_id=0, component="ml_model") as kernel_info:
    # Your kernel execution code
    result = my_gpu_function()
# Kernel profiling data automatically collected
```

### Integration with Components

The GPU profiler automatically integrates with system events to track component-specific GPU usage:

```python
# GPU usage is automatically tracked when components emit events
await event_bus.emit(ProcessingStarted(component="image_processor"))
# ... GPU processing ...
await event_bus.emit(ProcessingCompleted(component="image_processor"))

# Usage data is automatically collected and available via:
usage = gpu_profiler.get_component_gpu_usage("image_processor")
```

## Events

The GPU profiler emits various events for system integration:

### Profiler Events
- `GPUProfilerStarted`: When profiling begins
- `GPUProfilerStopped`: When profiling ends

### Alert Events
- `GPUUtilizationAlert`: High/low GPU utilization
- `GPUMemoryAlert`: High GPU memory usage
- `GPUTemperatureAlert`: High GPU temperature
- `GPUPowerUsageAlert`: High power consumption

### Performance Events
- `GPUMemoryLeakDetected`: Memory leak detection
- `GPUPerformanceBottleneckDetected`: Performance bottleneck identification

### Kernel Events
- `GPUKernelExecutionStarted`: Kernel execution begins
- `GPUKernelExecutionCompleted`: Kernel execution completes

## Metrics

The profiler registers comprehensive metrics with the metrics system:

### Utilization Metrics
- `gpu_utilization_percent`: GPU compute utilization
- `gpu_memory_utilization_percent`: GPU memory utilization

### Memory Metrics
- `gpu_memory_used_mb`: GPU memory currently used
- `gpu_memory_free_mb`: GPU memory available
- `gpu_memory_total_mb`: Total GPU memory
- `gpu_memory_cached_mb`: GPU memory cached

### Performance Metrics
- `gpu_temperature_celsius`: GPU temperature
- `gpu_power_draw_watts`: Current power consumption
- `gpu_clock_graphics_mhz`: Graphics clock speed
- `gpu_clock_memory_mhz`: Memory clock speed

### Kernel Metrics
- `gpu_kernel_executions_total`: Total kernel executions
- `gpu_kernel_duration_ms`: Kernel execution duration
- `gpu_memory_transfer_mb`: Memory transfer amounts

### Component Metrics
- `gpu_component_memory_usage_mb`: Per-component GPU memory usage

## Health Monitoring

The GPU profiler integrates with the health check system:

```python
# Health check provides comprehensive status
health_status = await gpu_profiler._health_check_callback()
print(f"GPU profiler status: {health_status['status']}")
print(f"GPU devices available: {health_status['gpu_count']}")
print(f"Samples collected: {health_status['samples_collected']}")
```

## Error Handling

The GPU profiler includes comprehensive error handling:

- **Graceful fallback**: Works without GPU libraries or hardware
- **Robust error handling**: Handles GPU driver issues
- **Comprehensive logging**: Detailed error logging
- **System integration**: Uses existing error handling system

## Performance Optimization

### Bottleneck Detection

The profiler automatically detects various performance bottlenecks:

- **Compute bottlenecks**: High GPU utilization
- **Memory bandwidth bottlenecks**: Memory-bound operations
- **Memory capacity bottlenecks**: Insufficient GPU memory
- **Thermal bottlenecks**: Temperature-related throttling
- **Power bottlenecks**: Power limit constraints

### Memory Leak Detection

Advanced memory leak detection includes:

- **Trend analysis**: Linear regression on memory usage
- **Spike detection**: Sudden large memory allocations
- **Growth rate tracking**: MB/hour growth calculations
- **Pattern recognition**: Allocation pattern analysis

### Optimization Suggestions

The profiler provides actionable optimization suggestions:

```python
suggestions = await gpu_profiler.get_optimization_suggestions()
for suggestion in suggestions:
    print(f"Type: {suggestion['type']}")
    print(f"Severity: {suggestion['severity']}")
    print(f"Message: {suggestion['message']}")
    print(f"Suggestion: {suggestion['suggestion']}")
```

## Dependencies

The GPU profiler supports multiple GPU libraries with graceful fallbacks:

### Required Dependencies
- Core system components (Container, EventBus, etc.)

### Optional Dependencies
- **PyTorch**: CUDA device detection and basic monitoring
- **pynvml**: Advanced NVIDIA GPU monitoring
- **nvidia-ml-py3**: Alternative NVIDIA monitoring
- **psutil**: System resource monitoring

### Fallback Behavior
- Works without any GPU libraries (reports no GPU available)
- Gracefully handles missing optional dependencies
- Provides mock implementations for testing

## Testing

The implementation includes comprehensive testing:

- **Unit tests**: Core functionality testing
- **Integration tests**: System integration validation
- **Mock testing**: Testing without GPU hardware
- **Performance tests**: Profiler overhead validation

## Integration Points

The GPU profiler integrates with all major system components:

### Core Engine
- Tracks GPU usage during multimodal processing
- Monitors inference performance

### Workflow Orchestrator
- Profiles GPU usage per workflow step
- Tracks resource allocation

### Skills Factory
- Monitors individual skill GPU consumption
- Tracks execution performance

### Component Manager
- Integrates with health monitoring
- Provides component status

### Session Manager
- Tracks session-level GPU usage
- Manages resource allocation

### Plugin Manager
- Monitors plugin GPU consumption
- Ensures resource limits

## Best Practices

1. **Start profiling early**: Begin profiling before GPU-intensive operations
2. **Use appropriate levels**: Choose profiling level based on needs
3. **Monitor alerts**: Set up alert handling for threshold violations
4. **Regular cleanup**: Use `torch.cuda.empty_cache()` based on suggestions
5. **Component tracking**: Emit proper events for automatic tracking
6. **Performance monitoring**: Regularly review optimization suggestions

## Troubleshooting

### Common Issues

1. **No GPU detected**: Check GPU drivers and libraries
2. **High memory usage**: Review optimization suggestions
3. **Performance bottlenecks**: Analyze bottleneck detection results
4. **Memory leaks**: Monitor leak detection and cleanup patterns

### Debug Mode

Enable debug mode for detailed information:

```python
await gpu_profiler.start_profiling(mode=GPUProfilingMode.DEBUG)
```

This provides verbose logging and detailed profiling data.