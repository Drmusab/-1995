"""
Profiling module for the AI Assistant.

This module provides comprehensive profiling capabilities including:
- Memory profiling with the MemoryProfiler
- Performance monitoring and analysis
- Integration with observability systems
"""

from .memory_profiler import (
    MemoryProfiler,
    MemoryProfilingMode,
    MemoryProfilingLevel,
    MemoryCategory,
    MemoryMetric,
    MemorySnapshot,
    MemoryLeak,
    MemoryOptimizationRecommendation,
    MemoryProfilingSession,
    MemoryTracker,
    MemoryLeakDetector,
    MemoryOptimizer,
    profile_memory,
    memory_profiling_context
)

__all__ = [
    "MemoryProfiler",
    "MemoryProfilingMode",
    "MemoryProfilingLevel",
    "MemoryCategory",
    "MemoryMetric",
    "MemorySnapshot",
    "MemoryLeak",
    "MemoryOptimizationRecommendation",
    "MemoryProfilingSession",
    "MemoryTracker",
    "MemoryLeakDetector",
    "MemoryOptimizer",
    "profile_memory",
    "memory_profiling_context"
]