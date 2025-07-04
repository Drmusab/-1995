"""
Profiling module for AI Assistant observability.
"""

from .gpu_profiler import GPUProfiler

# Try to import other profilers but gracefully handle failures
_all_profilers = ['GPUProfiler']

try:
    from .memory_profiler import MemoryProfiler
    _all_profilers.append('MemoryProfiler')
except ImportError as e:
    pass  # Memory profiler has import issues

try:
    from .cpu_profiler import EnhancedCPUProfiler
    _all_profilers.append('EnhancedCPUProfiler')
except (ImportError, SyntaxError) as e:
    pass  # CPU profiler has syntax errors

__all__ = _all_profilers
