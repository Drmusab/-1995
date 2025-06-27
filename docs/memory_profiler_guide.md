# Memory Profiler for AI Assistant

## Overview

The Enhanced Memory Profiler (`EnhancedMemoryProfiler`) provides comprehensive memory profiling and monitoring capabilities for the AI assistant system. It offers real-time memory tracking, leak detection, allocation analysis, and component-specific monitoring with minimal performance impact.

## Features

### Core Capabilities

- **Real-time Memory Monitoring**: Continuous tracking of system and process memory usage
- **Memory Leak Detection**: Advanced algorithms to detect various types of memory leaks
- **Allocation Tracking**: Detailed tracking of memory allocations using `tracemalloc`
- **Component Memory Profiling**: Track memory usage by specific system components
- **Garbage Collection Monitoring**: Monitor Python GC behavior and performance
- **Memory Usage Patterns**: Analyze memory growth, fragmentation, and usage patterns

### Monitoring Features

- **System Memory Metrics**: RSS, VMS, USS, PSS tracking
- **Python Heap Analysis**: Object counts, heap size monitoring
- **Memory Growth Analysis**: Detect unusual growth patterns
- **Peak Memory Detection**: Track memory usage peaks
- **Memory Efficiency Scoring**: Calculate memory usage efficiency

### Detection & Analysis

- **Multiple Leak Detection Algorithms**:
  - Growth-based detection
  - Component-based detection
  - Pattern-based detection
  - Reference cycle detection

- **Root Cause Analysis**: Identify suspected objects and call stacks
- **Optimization Recommendations**: Actionable suggestions for memory optimization

### Integration Features

- **Event-Driven Architecture**: Emit events for memory state changes
- **Metrics Integration**: Export metrics to monitoring systems
- **Health Check Support**: Monitor profiler health and status
- **Session Management**: Organize profiling into sessions
- **Report Generation**: HTML and JSON reports

## Usage

### Basic Usage

```python
from src.observability.profiling.memory_profiler import EnhancedMemoryProfiler
from src.core.dependency_injection import Container

# Initialize profiler
container = Container()
profiler = EnhancedMemoryProfiler(container)

# Start profiling session
session_id = await profiler.start_profiling(
    session_name="My Session",
    description="Memory profiling for feature X"
)

# Your application code here
# ...

# Stop profiling
session = await profiler.stop_profiling()
```

### Component Memory Tracking

```python
# Track memory usage for a specific component
with profiler.track_component_memory("data_processor"):
    # Component operations
    process_data(large_dataset)
    
# Memory usage is automatically recorded
```

### Function Decoration

```python
from src.observability.profiling.memory_profiler import profile_memory

@profile_memory("my_component")
async def process_data(data):
    # Function implementation
    return processed_data
```

### Manual Snapshots

```python
# Take manual memory snapshots
snapshot = await profiler.get_memory_snapshot()
print(f"Memory usage: {snapshot.process_memory['rss'] / (1024*1024):.2f} MB")

# Get current statistics
stats = profiler.get_memory_statistics()
print(f"Status: {stats['status']}")
```

### Leak Detection

```python
# Manual leak detection
leaks = await profiler.detect_memory_leaks()
for leak in leaks:
    print(f"Leak detected: {leak.leak_type} ({leak.severity})")
    for rec in leak.recommendations:
        print(f"  - {rec}")
```

## Configuration

### Memory Profiling Config

```python
from src.observability.profiling.memory_profiler import MemoryProfilingConfig, MemoryProfilingMode

config = MemoryProfilingConfig(
    mode=MemoryProfilingMode.DETAILED,
    monitoring_interval=1.0,  # seconds
    snapshot_interval=30.0,   # seconds
    memory_threshold_mb=1000.0,
    enable_leak_detection=True,
    enable_gc_monitoring=True,
    track_allocations=True
)
```

### Profiling Modes

- **OFF**: Profiling disabled
- **BASIC**: Basic memory usage tracking
- **DETAILED**: Detailed allocation tracking
- **LEAK_DETECTION**: Focus on memory leak detection
- **COMPONENT_TRACKING**: Component-specific memory tracking
- **PRODUCTION**: Optimized for production use

### Profiling Levels

- **LOW**: Basic memory usage only
- **MEDIUM**: Memory usage + allocation patterns
- **HIGH**: Detailed allocation tracking + object lifecycle
- **DETAILED**: Maximum detail with full call stacks

## Event Types

The memory profiler emits various events for integration with the event system:

- `MemoryProfilingStarted`: Profiling session started
- `MemoryProfilingStopped`: Profiling session completed
- `MemorySnapshotTaken`: Memory snapshot captured
- `MemoryLeakDetected`: Memory leak discovered
- `MemoryThresholdExceeded`: Memory usage threshold violated
- `MemoryOptimizationApplied`: Memory optimization performed

## Output and Reporting

### Generated Files

For each profiling session, the following files are generated:

1. **snapshots.json**: Raw snapshot data in JSON format
2. **memory_analysis_report.html**: Comprehensive HTML report
3. **detailed_analysis.json**: Detailed analysis data

### Report Contents

- Memory usage timeline and charts
- Detected memory leaks with severity levels
- Component memory profiles
- Optimization recommendations
- Garbage collection analysis
- Memory efficiency metrics

## Integration Points

### Core Engine Integration

```python
# In core engine initialization
async def _initialize_memory_systems(self):
    # Start memory profiling for core components
    if self.config.enable_memory_profiling:
        session_id = await self.memory_profiler.start_profiling(
            session_name="core_engine_session",
            mode=MemoryProfilingMode.COMPONENT_TRACKING
        )
```

### Component Manager Integration

```python
# Track memory usage per component
async def load_component(self, component_id):
    with self.memory_profiler.track_component_memory(component_id):
        component = await self._load_component_impl(component_id)
        return component
```

### Session Manager Integration

```python
# Track memory per session
async def create_session(self, user_id):
    session_id = str(uuid.uuid4())
    
    with self.memory_profiler.track_component_memory(f"session_{session_id}"):
        session = await self._create_session_impl(user_id)
        return session
```

## Performance Considerations

### Production Recommendations

1. **Use PRODUCTION mode** for minimal overhead
2. **Set appropriate sampling rates** (0.1-0.5 for production)
3. **Limit snapshot frequency** (60+ seconds in production)
4. **Configure retention policies** to prevent disk space issues
5. **Monitor profiler resource usage** itself

### Memory Overhead

- **Basic mode**: ~1-2% memory overhead
- **Detailed mode**: ~5-10% memory overhead
- **Tracemalloc enabled**: ~10-20% memory overhead

### CPU Overhead

- **Basic monitoring**: <1% CPU overhead
- **Detailed profiling**: 2-5% CPU overhead
- **Leak detection**: 1-3% CPU overhead

## Best Practices

### When to Use Memory Profiling

1. **Development**: Always enable for feature development
2. **Testing**: Enable for integration and performance tests
3. **Staging**: Monitor for memory leaks before production
4. **Production**: Use PRODUCTION mode with conservative settings

### Memory Leak Investigation

1. **Start with growth-based detection**
2. **Use component tracking** to isolate issues
3. **Analyze object lifetime patterns**
4. **Review garbage collection metrics**
5. **Implement optimization recommendations**

### Component Integration

1. **Use context managers** for automatic tracking
2. **Implement health check callbacks**
3. **Subscribe to memory events** for reactive responses
4. **Configure appropriate thresholds** for your components

## Examples

See `/tmp/standalone_memory_demo.py` for a complete working example demonstrating:

- Basic memory profiling
- Memory leak detection
- Event handling
- Report generation
- Integration patterns

## Dependencies

### Required
- `psutil`: System and process monitoring
- `tracemalloc`: Python memory allocation tracking (built-in)

### Optional
- `pympler`: Advanced Python memory analysis
- `objgraph`: Object reference analysis

## API Reference

### EnhancedMemoryProfiler

Primary class providing memory profiling functionality.

#### Methods

- `start_profiling(session_name, description, mode, level)`: Start profiling session
- `stop_profiling(session_id)`: Stop profiling session
- `get_memory_snapshot()`: Take immediate memory snapshot
- `detect_memory_leaks()`: Manual leak detection
- `get_memory_statistics()`: Get current statistics
- `track_component_memory(component_name)`: Context manager for component tracking
- `health_check()`: Check profiler health
- `cleanup_old_sessions(retention_hours)`: Clean up old data

### Data Classes

- `MemoryProfilingConfig`: Configuration settings
- `MemorySnapshot`: Point-in-time memory state
- `MemoryLeak`: Detected memory leak information
- `ComponentMemoryProfile`: Component-specific memory profile
- `MemoryProfilingSession`: Complete profiling session data

### Enums

- `MemoryProfilingMode`: Profiling operation modes
- `MemoryProfilingLevel`: Detail levels for profiling
- `MemoryProfilerStatus`: Current profiler status
- `MemoryMetricType`: Types of memory metrics

This memory profiler provides comprehensive monitoring capabilities while maintaining minimal performance impact, making it suitable for both development and production use in the AI assistant system.