# Advanced Memory Profiler for AI Assistant

## Overview

The Advanced Memory Profiler provides comprehensive memory monitoring capabilities for the AI assistant system, including real-time monitoring, leak detection, component-specific tracking, and performance optimization recommendations.

## Features

### Core Capabilities
- **Real-time Memory Monitoring**: Continuous tracking of memory usage with configurable intervals
- **Memory Leak Detection**: Automatic detection and alerting for memory leaks
- **Component-Specific Tracking**: Per-component memory attribution and analysis
- **Session-Based Profiling**: Session-specific memory profiling and analysis
- **Multiple Backend Support**: Uses tracemalloc, psutil (optional), pympler (optional), and system monitoring
- **Performance Integration**: Seamless integration with existing metrics and event systems

### Profiling Capabilities
- Process-level memory monitoring (RSS, VMS, USS, PSS)
- Python object-level tracking
- Memory allocation patterns analysis
- Garbage collection monitoring
- Memory fragmentation analysis
- Memory usage trend analysis
- Memory growth rate detection

## Configuration

The memory profiler is configured through the system's configuration loader:

```python
memory_profiling:
  mode: "standard"  # off, basic, standard, advanced, production, debug
  level: "medium"   # low, medium, high, detailed
  sampling_interval: 1.0  # seconds
  
  # Thresholds (in MB)
  memory_warning_threshold_mb: 1000.0
  memory_critical_threshold_mb: 2000.0
  memory_growth_rate_threshold_mb_per_min: 50.0
  
  # Features
  enable_leak_detection: true
  enable_component_tracking: true
  enable_gc_monitoring: true
  
  # Backends
  enabled_backends:
    - tracemalloc
    - system
    - gc
```

## Usage Examples

### Basic Usage

```python
from src.observability.profiling import AdvancedMemoryProfiler
from src.core.dependency_injection import Container

# Get the memory profiler from the container
container = Container()
memory_profiler = container.get(AdvancedMemoryProfiler)

# Get current memory usage
current_usage = memory_profiler.get_current_memory_usage()
print(f"Current RSS: {current_usage['rss_mb']:.2f} MB")
```

### Profiling Session

```python
# Start a profiling session
session_id = await memory_profiler.start_profiling(
    session_name="my_analysis",
    description="Analyzing memory usage during processing"
)

# ... run your code ...

# Stop profiling and get results
session = await memory_profiler.stop_profiling()
print(f"Peak memory: {session.peak_memory_mb:.2f} MB")
print(f"Detected leaks: {len(session.detected_leaks)}")
```

### Using Context Manager

```python
from src.observability.profiling import memory_profiling_session

async with memory_profiling_session(
    memory_profiler,
    session_name="api_request_analysis"
) as session_id:
    # Your code here
    await process_user_request()
    
# Session is automatically stopped and results are available
```

### Using Decorator

```python
from src.observability.profiling import memory_profile

@memory_profile(memory_profiler, session_name="function_analysis")
async def my_function():
    # Function code here
    pass
```

### Component Memory Tracking

```python
# Get memory usage for all components
component_usage = memory_profiler.get_component_memory_usage()

# Get memory usage for specific component
core_engine_usage = memory_profiler.get_component_memory_usage("core_engine")
print(f"Core engine memory: {core_engine_usage['current_memory_mb']:.2f} MB")
```

### Memory Trends Analysis

```python
# Get memory trends for the last 10 minutes
trends = memory_profiler.get_memory_trends(window_minutes=10)
print(f"Collected {len(trends['timestamps'])} data points")
```

### Leak Detection

```python
# Get active memory leaks
leaks = memory_profiler.get_active_leaks()
for leak in leaks:
    print(f"Leak in {leak['component']}: {leak['growth_rate_mb_per_min']:.2f} MB/min")
    print(f"Recommendation: {leak['recommendation']}")

# Clear specific leak alert
memory_profiler.clear_leak_alerts(leak_id="specific-leak-id")

# Clear all leak alerts
memory_profiler.clear_leak_alerts()
```

### Manual Garbage Collection

```python
# Force garbage collection and get statistics
gc_stats = await memory_profiler.force_garbage_collection()
print(f"Collected {gc_stats['collected_objects']} objects")
print(f"Freed {gc_stats['objects_freed']} objects")
```

## Integration with Core Components

The memory profiler automatically integrates with core AI assistant components:

### Enhanced Core Engine
- Monitors multimodal processing memory usage
- Tracks memory during inference operations

### Component Manager
- Per-component memory attribution
- Component lifecycle memory tracking

### Session Manager
- Session-specific memory profiling
- Memory usage per user session

### Plugin Manager
- Plugin memory monitoring
- Memory isolation tracking

### Workflow Orchestrator
- Workflow memory analysis
- Memory efficiency optimization

## Events

The memory profiler emits events for system integration:

- `MemoryOperationStarted`: When profiling starts
- `MemoryOperationCompleted`: When profiling stops
- `PerformanceThresholdExceeded`: When memory thresholds are exceeded

## Health Checks

The memory profiler registers health checks that monitor:
- Current memory usage vs thresholds
- Active memory leaks
- Profiler operational status

## Metrics

Exported metrics include:
- `memory_profiler_rss_bytes`: Process RSS memory
- `memory_profiler_vms_bytes`: Process VMS memory
- `memory_profiler_heap_objects`: Number of tracked objects
- `memory_profiler_leaks_detected`: Number of detected leaks
- `memory_profiler_gc_collections`: Garbage collection count

## Output Files

The memory profiler generates:
- **Session Reports**: Comprehensive JSON reports with statistics and recommendations
- **Raw Data Exports**: Detailed snapshot data for external analysis
- **Memory Trend Charts**: Visual representation of memory usage over time

## Best Practices

1. **Production Use**: Use `production` mode for minimal overhead
2. **Development**: Use `debug` mode for detailed analysis
3. **Regular Monitoring**: Enable continuous profiling with appropriate intervals
4. **Threshold Tuning**: Adjust thresholds based on your system's memory patterns
5. **Leak Response**: Act quickly on leak detection alerts
6. **Component Analysis**: Use component-specific tracking to identify memory hotspots

## Advanced Configuration

For advanced use cases, you can customize the profiler behavior:

```python
from src.observability.profiling.memory_profiler import MemoryProfilingConfig, MemoryProfilingMode

custom_config = MemoryProfilingConfig(
    mode=MemoryProfilingMode.ADVANCED,
    level=MemoryProfilingLevel.DETAILED,
    sampling_interval=0.5,
    enable_leak_detection=True,
    leak_detection_interval=60,  # Check every minute
    memory_warning_threshold_mb=500.0,
    enable_recommendations=True
)
```

## Troubleshooting

### Common Issues

1. **High Memory Usage**: Check active components and recent leak alerts
2. **Missing Backends**: Install psutil and pympler for enhanced monitoring
3. **Performance Impact**: Reduce sampling frequency or use production mode
4. **Event Loop Errors**: Ensure proper async/await usage in event handlers

### Debug Mode

Enable debug logging for detailed profiler information:

```python
import logging
logging.getLogger('src.observability.profiling.memory_profiler').setLevel(logging.DEBUG)
```

This memory profiler provides comprehensive insights into your AI assistant's memory usage patterns and helps maintain optimal performance through proactive monitoring and leak detection.