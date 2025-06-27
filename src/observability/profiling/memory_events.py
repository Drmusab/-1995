"""
Memory Profiler Event Types
Author: Drmusab
Last Modified: 2025-01-11 16:00:00 UTC

This module defines event types specific to memory profiling operations.
"""

from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field


@dataclass
class BaseMemoryEvent:
    """Base class for memory profiling events."""
    event_id: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    session_id: str = ""
    component_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryProfilingStarted(BaseMemoryEvent):
    """Memory profiling session started event."""
    session_name: str = ""
    profiling_mode: str = ""
    profiling_level: str = ""
    description: str = ""


@dataclass
class MemoryProfilingStopped(BaseMemoryEvent):
    """Memory profiling session stopped event."""
    session_name: str = ""
    duration_seconds: float = 0.0
    snapshots_collected: int = 0
    leaks_detected: int = 0
    memory_efficiency: float = 0.0


@dataclass
class MemorySnapshotTaken(BaseMemoryEvent):
    """Memory snapshot taken event."""
    snapshot_type: str = ""  # "periodic", "manual", "threshold", etc.
    memory_usage_mb: float = 0.0
    memory_percent: float = 0.0
    object_count: int = 0
    growth_rate_mb_per_min: float = 0.0


@dataclass
class MemoryLeakDetected(BaseMemoryEvent):
    """Memory leak detected event."""
    leak_type: str = ""  # "growth", "allocation", "reference", "pattern"
    severity: str = ""  # "low", "medium", "high", "critical"
    memory_growth_mb: float = 0.0
    growth_rate_mb_per_min: float = 0.0
    duration_minutes: float = 0.0
    suspected_objects: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class MemoryThresholdExceeded(BaseMemoryEvent):
    """Memory threshold exceeded event."""
    threshold_type: str = ""  # "usage", "growth_rate", "fragmentation"
    current_value: float = 0.0
    threshold_value: float = 0.0
    memory_usage_mb: float = 0.0
    memory_percent: float = 0.0


@dataclass
class MemoryOptimizationApplied(BaseMemoryEvent):
    """Memory optimization applied event."""
    optimization_type: str = ""
    memory_saved_mb: float = 0.0
    efficiency_improvement: float = 0.0
    description: str = ""


@dataclass
class MemoryGCEvent(BaseMemoryEvent):
    """Garbage collection event."""
    generation: int = 0
    objects_collected: int = 0
    objects_uncollectable: int = 0
    time_spent_ms: float = 0.0
    memory_freed_mb: float = 0.0


@dataclass
class ComponentMemoryTracked(BaseMemoryEvent):
    """Component memory tracking event."""
    component_name: str = ""
    memory_used_mb: float = 0.0
    peak_memory_mb: float = 0.0
    memory_pattern: str = ""  # "stable", "growing", "oscillating", "leaking"
    efficiency_score: float = 0.0


@dataclass
class MemoryReportGenerated(BaseMemoryEvent):
    """Memory analysis report generated event."""
    report_type: str = ""  # "html", "json", "summary"
    report_path: str = ""
    session_duration_seconds: float = 0.0
    total_snapshots: int = 0
    leaks_found: int = 0


# Event registry for easy access
MEMORY_PROFILER_EVENTS = {
    "MemoryProfilingStarted": MemoryProfilingStarted,
    "MemoryProfilingStopped": MemoryProfilingStopped,
    "MemorySnapshotTaken": MemorySnapshotTaken,
    "MemoryLeakDetected": MemoryLeakDetected,
    "MemoryThresholdExceeded": MemoryThresholdExceeded,
    "MemoryOptimizationApplied": MemoryOptimizationApplied,
    "MemoryGCEvent": MemoryGCEvent,
    "ComponentMemoryTracked": ComponentMemoryTracked,
    "MemoryReportGenerated": MemoryReportGenerated,
}