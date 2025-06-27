"""
Advanced Memory Profiler for AI Assistant
Author: Drmusab
Last Modified: 2025-01-20 12:00:00 UTC

This module provides comprehensive memory profiling capabilities for the AI assistant,
including real-time monitoring, memory leak detection, component-specific tracking,
and integration with all core system components.
"""

import gc
import threading
import time
import asyncio
import tracemalloc
import sys
import os
import weakref
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable, TypeVar, Union, Set, Tuple
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
from contextlib import contextmanager, asynccontextmanager
from collections import defaultdict, deque
import json
import uuid
import logging
import concurrent.futures
from abc import ABC, abstractmethod
import statistics

# Graceful imports for optional dependencies
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    psutil = None
    PSUTIL_AVAILABLE = False

try:
    import py3nvml
    GPU_MONITORING_AVAILABLE = True
except ImportError:
    py3nvml = None
    GPU_MONITORING_AVAILABLE = False

# Core imports
from src.core.config.loader import ConfigLoader
from src.core.events.event_bus import EventBus
from src.core.events.event_types import (
    MemoryProfilerStarted, MemoryProfilerStopped, MemoryLeakDetected,
    MemoryThresholdExceeded, MemoryOptimizationPerformed,
    ComponentHealthChanged, SystemStateChanged
)
from src.core.error_handling import ErrorHandler, handle_exceptions
from src.core.dependency_injection import Container
from src.core.health_check import HealthCheck

# Observability components
from src.observability.monitoring.metrics import MetricsCollector
from src.observability.monitoring.tracing import TraceManager
from src.observability.logging.config import get_logger

# Type definitions
F = TypeVar('F', bound=Callable)


class MemoryProfilingMode(Enum):
    """Memory profiling modes."""
    OFF = "off"
    BASIC = "basic"                    # System memory monitoring only
    STANDARD = "standard"              # Python heap + system memory
    ADVANCED = "advanced"              # Full profiling with leak detection
    COMPONENT_SPECIFIC = "component"   # Component-focused profiling
    REAL_TIME = "real_time"           # Continuous real-time monitoring


class MemoryTrackingLevel(Enum):
    """Memory tracking detail levels."""
    MINIMAL = "minimal"        # Basic memory usage tracking
    STANDARD = "standard"      # Object allocation tracking
    DETAILED = "detailed"      # Line-by-line allocation tracking
    COMPREHENSIVE = "comprehensive"  # Full memory profiling with leak detection


class MemoryProfilerStatus(Enum):
    """Memory profiler operational status."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    ERROR = "error"


class MemoryCategory(Enum):
    """Categories of memory usage."""
    HEAP = "heap"               # Python heap memory
    SYSTEM = "system"           # System memory (RSS, VMS)
    GPU = "gpu"                 # GPU memory
    SHARED = "shared"           # Shared memory
    CACHE = "cache"             # Cache memory
    COMPONENT = "component"     # Component-specific memory


class LeakSeverity(Enum):
    """Memory leak severity levels."""
    LOW = "low"                 # Small, gradual leaks
    MEDIUM = "medium"           # Moderate memory growth
    HIGH = "high"               # Significant memory leaks
    CRITICAL = "critical"       # Severe memory leaks requiring immediate action


@dataclass
class MemoryProfilingConfig:
    """Configuration for memory profiling."""
    mode: MemoryProfilingMode = MemoryProfilingMode.STANDARD
    level: MemoryTrackingLevel = MemoryTrackingLevel.STANDARD
    sampling_interval: float = 5.0  # 5 second sampling
    
    # Thresholds
    heap_threshold_mb: float = 1000.0
    system_threshold_mb: float = 2000.0
    gpu_threshold_mb: float = 1000.0
    leak_threshold_mb: float = 100.0
    
    # Leak detection
    enable_leak_detection: bool = True
    leak_detection_interval: float = 60.0  # 1 minute
    leak_growth_threshold_percent: float = 20.0
    
    # Component tracking
    enable_component_tracking: bool = True
    track_object_allocations: bool = True
    track_call_stacks: bool = False
    
    # Optimization
    enable_auto_optimization: bool = True
    auto_gc_threshold_mb: float = 500.0
    optimization_interval: float = 300.0  # 5 minutes
    
    # Data retention
    max_snapshots: int = 1000
    snapshot_retention_hours: int = 24
    max_profile_size_mb: float = 200.0
    
    # Integration settings
    integrate_with_metrics: bool = True
    integrate_with_health_check: bool = True
    enable_real_time_alerts: bool = True
    
    # Advanced features
    enable_fragmentation_analysis: bool = True
    enable_gc_monitoring: bool = True
    enable_memory_mapping: bool = True


@dataclass
class MemorySnapshot:
    """Point-in-time memory state snapshot."""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    snapshot_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # System memory
    system_memory_mb: float = 0.0
    system_available_mb: float = 0.0
    system_percent: float = 0.0
    
    # Process memory
    process_rss_mb: float = 0.0
    process_vms_mb: float = 0.0
    process_percent: float = 0.0
    
    # Python heap
    heap_size_mb: float = 0.0
    heap_peak_mb: float = 0.0
    heap_allocated_blocks: int = 0
    
    # GPU memory (if available)
    gpu_memory_mb: float = 0.0
    gpu_available_mb: float = 0.0
    gpu_percent: float = 0.0
    
    # Component memory
    component_memory: Dict[str, float] = field(default_factory=dict)
    
    # Garbage collection
    gc_generation_counts: List[int] = field(default_factory=list)
    gc_collection_count: int = 0
    
    # Memory allocations (if tracking enabled)
    top_allocations: List[Dict[str, Any]] = field(default_factory=list)
    
    # Calculated metrics
    memory_fragmentation: float = 0.0
    allocation_rate_mb_per_sec: float = 0.0


@dataclass
class ComponentMemoryProfile:
    """Memory profile for a specific component."""
    component_id: str
    component_type: str
    
    # Memory usage over time
    memory_samples: deque = field(default_factory=lambda: deque(maxlen=1000))
    peak_memory_mb: float = 0.0
    average_memory_mb: float = 0.0
    current_memory_mb: float = 0.0
    
    # Allocation tracking
    allocation_count: int = 0
    deallocation_count: int = 0
    net_allocations: int = 0
    
    # Leak detection
    potential_leaks: List[Dict[str, Any]] = field(default_factory=list)
    leak_growth_rate_mb_per_hour: float = 0.0
    
    # Timing
    first_seen: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Metadata
    tags: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryLeak:
    """Detected memory leak information."""
    leak_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    component_id: Optional[str] = None
    
    # Leak characteristics
    leak_type: str = ""  # gradual, sudden, periodic
    severity: LeakSeverity = LeakSeverity.LOW
    growth_rate_mb_per_hour: float = 0.0
    total_leaked_mb: float = 0.0
    
    # Detection details
    first_detected: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_confirmed: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    detection_confidence: float = 0.0
    
    # Source information
    allocation_source: Optional[str] = None
    call_stack: List[str] = field(default_factory=list)
    
    # Remediation
    suggested_actions: List[str] = field(default_factory=list)
    auto_remediation_attempted: bool = False


@dataclass
class MemoryProfilingSession:
    """A memory profiling session with metadata and results."""
    session_id: str
    name: str
    description: Optional[str] = None
    
    # Session metadata
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    end_time: Optional[datetime] = None
    duration: float = 0.0
    
    # Configuration
    config: MemoryProfilingConfig = field(default_factory=MemoryProfilingConfig)
    
    # Data collection
    snapshots: List[MemorySnapshot] = field(default_factory=list)
    component_profiles: Dict[str, ComponentMemoryProfile] = field(default_factory=dict)
    detected_leaks: List[MemoryLeak] = field(default_factory=list)
    
    # Analysis results
    peak_memory_mb: float = 0.0
    average_memory_mb: float = 0.0
    memory_growth_rate_mb_per_hour: float = 0.0
    fragmentation_score: float = 0.0
    
    # Optimization actions
    optimizations_performed: List[Dict[str, Any]] = field(default_factory=list)
    memory_freed_mb: float = 0.0
    
    # Files and exports
    session_file_path: Optional[Path] = None
    report_path: Optional[Path] = None


class ComponentMemoryTracker:
    """Tracks memory usage for specific components."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.component_profiles: Dict[str, ComponentMemoryProfile] = {}
        self.component_refs: Dict[str, weakref.ref] = {}
        self.tracking_lock = threading.Lock()
    
    def register_component(self, component_id: str, component_type: str, 
                          component_instance: Any = None) -> None:
        """Register a component for memory tracking."""
        with self.tracking_lock:
            profile = ComponentMemoryProfile(
                component_id=component_id,
                component_type=component_type
            )
            self.component_profiles[component_id] = profile
            
            if component_instance is not None:
                self.component_refs[component_id] = weakref.ref(component_instance)
    
    def track_allocation(self, component_id: str, size_bytes: int) -> None:
        """Track memory allocation for a component."""
        with self.tracking_lock:
            if component_id in self.component_profiles:
                profile = self.component_profiles[component_id]
                profile.allocation_count += 1
                profile.net_allocations += 1
                profile.current_memory_mb += size_bytes / (1024 * 1024)
                profile.last_updated = datetime.now(timezone.utc)
    
    def track_deallocation(self, component_id: str, size_bytes: int) -> None:
        """Track memory deallocation for a component."""
        with self.tracking_lock:
            if component_id in self.component_profiles:
                profile = self.component_profiles[component_id]
                profile.deallocation_count += 1
                profile.net_allocations -= 1
                profile.current_memory_mb -= size_bytes / (1024 * 1024)
                profile.last_updated = datetime.now(timezone.utc)
    
    def update_memory_usage(self, component_id: str, memory_mb: float) -> None:
        """Update current memory usage for a component."""
        with self.tracking_lock:
            if component_id in self.component_profiles:
                profile = self.component_profiles[component_id]
                profile.memory_samples.append((datetime.now(timezone.utc), memory_mb))
                profile.current_memory_mb = memory_mb
                profile.peak_memory_mb = max(profile.peak_memory_mb, memory_mb)
                
                # Calculate average
                if profile.memory_samples:
                    recent_samples = list(profile.memory_samples)[-100:]  # Last 100 samples
                    profile.average_memory_mb = statistics.mean([sample[1] for sample in recent_samples])
                
                profile.last_updated = datetime.now(timezone.utc)
    
    def get_component_profile(self, component_id: str) -> Optional[ComponentMemoryProfile]:
        """Get memory profile for a component."""
        with self.tracking_lock:
            return self.component_profiles.get(component_id)
    
    def get_all_profiles(self) -> Dict[str, ComponentMemoryProfile]:
        """Get all component memory profiles."""
        with self.tracking_lock:
            return self.component_profiles.copy()


class MemoryAnalyzer:
    """Analyzes memory usage patterns and provides insights."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def analyze_memory_trend(self, snapshots: List[MemorySnapshot]) -> Dict[str, Any]:
        """Analyze memory usage trends."""
        if len(snapshots) < 2:
            return {"trend": "insufficient_data"}
        
        # Calculate memory growth rate
        recent_snapshots = snapshots[-10:]  # Last 10 snapshots
        if len(recent_snapshots) >= 2:
            time_diff = (recent_snapshots[-1].timestamp - recent_snapshots[0].timestamp).total_seconds()
            memory_diff = recent_snapshots[-1].process_rss_mb - recent_snapshots[0].process_rss_mb
            
            if time_diff > 0:
                growth_rate_mb_per_hour = (memory_diff / time_diff) * 3600
            else:
                growth_rate_mb_per_hour = 0.0
        else:
            growth_rate_mb_per_hour = 0.0
        
        # Determine trend type
        if abs(growth_rate_mb_per_hour) < 1.0:
            trend = "stable"
        elif growth_rate_mb_per_hour > 10.0:
            trend = "rapid_growth"
        elif growth_rate_mb_per_hour > 1.0:
            trend = "gradual_growth"
        elif growth_rate_mb_per_hour < -10.0:
            trend = "rapid_decline"
        else:
            trend = "gradual_decline"
        
        return {
            "trend": trend,
            "growth_rate_mb_per_hour": growth_rate_mb_per_hour,
            "peak_memory_mb": max(s.process_rss_mb for s in snapshots),
            "average_memory_mb": statistics.mean([s.process_rss_mb for s in snapshots]),
            "memory_variance": statistics.variance([s.process_rss_mb for s in snapshots]) if len(snapshots) > 1 else 0
        }
    
    def detect_memory_patterns(self, component_profiles: Dict[str, ComponentMemoryProfile]) -> List[Dict[str, Any]]:
        """Detect memory usage patterns across components."""
        patterns = []
        
        for component_id, profile in component_profiles.items():
            if len(profile.memory_samples) < 10:
                continue
            
            recent_samples = list(profile.memory_samples)[-50:]  # Last 50 samples
            memory_values = [sample[1] for sample in recent_samples]
            
            # Check for consistent growth
            if len(memory_values) >= 10:
                correlation = self._calculate_correlation(range(len(memory_values)), memory_values)
                if correlation > 0.7:
                    patterns.append({
                        "component_id": component_id,
                        "pattern": "consistent_growth",
                        "correlation": correlation,
                        "severity": "high" if correlation > 0.9 else "medium"
                    })
        
        return patterns
    
    def _calculate_correlation(self, x: List[float], y: List[float]) -> float:
        """Calculate Pearson correlation coefficient."""
        if len(x) != len(y) or len(x) < 2:
            return 0.0
        
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_x2 = sum(x[i] ** 2 for i in range(n))
        sum_y2 = sum(y[i] ** 2 for i in range(n))
        
        denominator = ((n * sum_x2 - sum_x ** 2) * (n * sum_y2 - sum_y ** 2)) ** 0.5
        
        if denominator == 0:
            return 0.0
        
        return (n * sum_xy - sum_x * sum_y) / denominator


class MemoryLeakDetector:
    """Detects and analyzes memory leaks."""
    
    def __init__(self, logger: logging.Logger, config: MemoryProfilingConfig):
        self.logger = logger
        self.config = config
        self.detected_leaks: Dict[str, MemoryLeak] = {}
        self.detection_history: deque = deque(maxlen=1000)
    
    def analyze_for_leaks(self, snapshots: List[MemorySnapshot], 
                         component_profiles: Dict[str, ComponentMemoryProfile]) -> List[MemoryLeak]:
        """Analyze memory snapshots and component profiles for leaks."""
        newly_detected = []
        
        # Analyze overall memory growth
        if len(snapshots) >= 10:
            recent_snapshots = snapshots[-10:]
            growth_analysis = self._analyze_memory_growth(recent_snapshots)
            
            if growth_analysis["is_leak"]:
                leak = MemoryLeak(
                    leak_type="gradual",
                    severity=growth_analysis["severity"],
                    growth_rate_mb_per_hour=growth_analysis["growth_rate"],
                    total_leaked_mb=growth_analysis["leaked_amount"],
                    detection_confidence=growth_analysis["confidence"]
                )
                
                leak_key = f"system_{leak.leak_type}"
                if leak_key not in self.detected_leaks:
                    self.detected_leaks[leak_key] = leak
                    newly_detected.append(leak)
        
        # Analyze component-specific leaks
        for component_id, profile in component_profiles.items():
            if len(profile.memory_samples) >= 5:
                component_analysis = self._analyze_component_leak(profile)
                
                if component_analysis["is_leak"]:
                    leak = MemoryLeak(
                        component_id=component_id,
                        leak_type=component_analysis["leak_type"],
                        severity=component_analysis["severity"],
                        growth_rate_mb_per_hour=component_analysis["growth_rate"],
                        total_leaked_mb=component_analysis["leaked_amount"],
                        detection_confidence=component_analysis["confidence"]
                    )
                    
                    leak_key = f"{component_id}_{leak.leak_type}"
                    if leak_key not in self.detected_leaks:
                        self.detected_leaks[leak_key] = leak
                        newly_detected.append(leak)
        
        return newly_detected
    
    def _analyze_memory_growth(self, snapshots: List[MemorySnapshot]) -> Dict[str, Any]:
        """Analyze memory growth patterns for leak detection."""
        if len(snapshots) < 5:
            return {"is_leak": False}
        
        memory_values = [s.process_rss_mb for s in snapshots]
        time_diffs = [(snapshots[i].timestamp - snapshots[i-1].timestamp).total_seconds() 
                     for i in range(1, len(snapshots))]
        
        # Calculate growth rate
        total_time = sum(time_diffs)
        if total_time > 0:
            growth_rate = (memory_values[-1] - memory_values[0]) / total_time * 3600  # MB/hour
        else:
            growth_rate = 0
        
        # Determine if this indicates a leak
        is_leak = (
            growth_rate > self.config.leak_growth_threshold_percent and
            memory_values[-1] > memory_values[0] * 1.1  # At least 10% growth
        )
        
        # Determine severity
        if growth_rate > 100:
            severity = LeakSeverity.CRITICAL
        elif growth_rate > 50:
            severity = LeakSeverity.HIGH
        elif growth_rate > 10:
            severity = LeakSeverity.MEDIUM
        else:
            severity = LeakSeverity.LOW
        
        return {
            "is_leak": is_leak,
            "growth_rate": growth_rate,
            "leaked_amount": max(0, memory_values[-1] - memory_values[0]),
            "severity": severity,
            "confidence": min(1.0, len(snapshots) / 10.0),
            "leak_type": "gradual" if is_leak else "none"
        }
    
    def _analyze_component_leak(self, profile: ComponentMemoryProfile) -> Dict[str, Any]:
        """Analyze component memory profile for leaks."""
        if len(profile.memory_samples) < 3:
            return {"is_leak": False}
        
        recent_samples = list(profile.memory_samples)[-10:]
        memory_values = [sample[1] for sample in recent_samples]
        
        # Simple leak detection based on consistent growth
        if len(memory_values) >= 3:
            is_growing = all(memory_values[i] >= memory_values[i-1] for i in range(1, len(memory_values)))
            growth_amount = memory_values[-1] - memory_values[0]
            
            is_leak = is_growing and growth_amount > 10.0  # 10MB growth
            
            if is_leak:
                # Calculate growth rate
                time_span = (recent_samples[-1][0] - recent_samples[0][0]).total_seconds()
                growth_rate = (growth_amount / time_span * 3600) if time_span > 0 else 0
                
                # Determine severity
                if growth_rate > 50:
                    severity = LeakSeverity.HIGH
                elif growth_rate > 20:
                    severity = LeakSeverity.MEDIUM
                else:
                    severity = LeakSeverity.LOW
                
                return {
                    "is_leak": True,
                    "leak_type": "component_gradual",
                    "growth_rate": growth_rate,
                    "leaked_amount": growth_amount,
                    "severity": severity,
                    "confidence": min(1.0, len(memory_values) / 10.0)
                }
        
        return {"is_leak": False}


class MemoryOptimizer:
    """Performs automatic memory optimization and cleanup."""
    
    def __init__(self, logger: logging.Logger, config: MemoryProfilingConfig):
        self.logger = logger
        self.config = config
        self.optimization_history: List[Dict[str, Any]] = []
    
    def optimize_memory(self, current_snapshot: MemorySnapshot, 
                       force: bool = False) -> Dict[str, Any]:
        """Perform memory optimization if needed."""
        optimization_result = {
            "performed": False,
            "actions": [],
            "memory_freed_mb": 0.0,
            "timestamp": datetime.now(timezone.utc)
        }
        
        if not self.config.enable_auto_optimization and not force:
            return optimization_result
        
        # Check if optimization is needed
        needs_optimization = (
            force or
            current_snapshot.process_rss_mb > self.config.auto_gc_threshold_mb or
            current_snapshot.system_percent > 85.0
        )
        
        if needs_optimization:
            initial_memory = current_snapshot.process_rss_mb
            
            # Perform garbage collection
            gc_result = self._perform_garbage_collection()
            optimization_result["actions"].extend(gc_result["actions"])
            
            # Clear caches if available
            cache_result = self._clear_caches()
            optimization_result["actions"].extend(cache_result["actions"])
            
            # Take a new memory measurement
            if PSUTIL_AVAILABLE:
                try:
                    process = psutil.Process()
                    final_memory = process.memory_info().rss / (1024 * 1024)
                    memory_freed = max(0, initial_memory - final_memory)
                    optimization_result["memory_freed_mb"] = memory_freed
                except Exception as e:
                    self.logger.warning(f"Could not measure memory after optimization: {e}")
            
            optimization_result["performed"] = True
            self.optimization_history.append(optimization_result.copy())
            
            self.logger.info(f"Memory optimization completed, freed {optimization_result['memory_freed_mb']:.2f} MB")
        
        return optimization_result
    
    def _perform_garbage_collection(self) -> Dict[str, Any]:
        """Perform Python garbage collection."""
        actions = []
        
        try:
            # Collect all generations
            collected_counts = []
            for generation in range(3):
                collected = gc.collect(generation)
                collected_counts.append(collected)
                if collected > 0:
                    actions.append(f"gc_generation_{generation}_collected_{collected}")
            
            # Force full collection
            total_collected = gc.collect()
            actions.append(f"gc_full_collected_{total_collected}")
            
            self.logger.debug(f"Garbage collection completed: {actions}")
            
        except Exception as e:
            self.logger.warning(f"Garbage collection failed: {e}")
            actions.append(f"gc_failed_{str(e)}")
        
        return {"actions": actions}
    
    def _clear_caches(self) -> Dict[str, Any]:
        """Clear various caches to free memory."""
        actions = []
        
        try:
            # Clear import cache
            import importlib
            importlib.invalidate_caches()
            actions.append("import_cache_cleared")
            
            # Clear function cache (if available)
            import functools
            # Note: This would clear all functools.lru_cache decorated functions
            # In a real implementation, we'd need to track specific caches
            actions.append("function_caches_noted")
            
        except Exception as e:
            self.logger.warning(f"Cache clearing failed: {e}")
            actions.append(f"cache_clear_failed_{str(e)}")
        
        return {"actions": actions}


class MemoryProfiler:
    """
    Advanced Memory Profiler for the AI Assistant.
    
    This profiler provides comprehensive memory monitoring including:
    - Real-time memory usage tracking across all system components
    - Python heap, GPU memory, and system memory monitoring
    - Memory leak detection and prevention
    - Component-specific memory profiling
    - Automatic memory optimization and cleanup
    - Integration with core assistant components
    """
    
    def __init__(self, container: Container):
        """
        Initialize the enhanced memory profiler.
        
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
        
        # Observability components
        self.metrics = container.get(MetricsCollector)
        self.tracer = container.get(TraceManager)
        
        # Configuration
        profiler_config = self.config_loader.get("profiling.memory", {})
        self.config = MemoryProfilingConfig(**profiler_config)
        
        # State management
        self.status = MemoryProfilerStatus.STOPPED
        self.current_session: Optional[MemoryProfilingSession] = None
        self.profiler_lock = threading.Lock()
        
        # Memory profiling infrastructure
        self.component_tracker = ComponentMemoryTracker(self.logger)
        self.memory_analyzer = MemoryAnalyzer(self.logger)
        self.leak_detector = MemoryLeakDetector(self.logger, self.config)
        self.memory_optimizer = MemoryOptimizer(self.logger, self.config)
        
        # Data storage
        self.memory_snapshots: deque = deque(maxlen=self.config.max_snapshots)
        self.profiling_sessions: Dict[str, MemoryProfilingSession] = {}
        self.recent_sessions = deque(maxlen=10)
        
        # Threading support
        self.monitoring_thread: Optional[threading.Thread] = None
        self.stop_monitoring_event = threading.Event()
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        
        # Output paths
        self.output_dir = Path(self.config_loader.get("profiling.output_dir", "data/profiling"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize tracemalloc if not already started
        if self.config.level in [MemoryTrackingLevel.DETAILED, MemoryTrackingLevel.COMPREHENSIVE]:
            if not tracemalloc.is_tracing():
                tracemalloc.start(10)  # Keep top 10 frames
        
        # Setup monitoring and integrations
        self._setup_monitoring()
        self._setup_event_handlers()
        self._setup_health_check()
        
        # Log availability of optional dependencies
        self._log_dependency_status()
        
        self.logger.info("MemoryProfiler initialized")
    
    def _setup_monitoring(self) -> None:
        """Setup metrics and monitoring."""
        try:
            if self.metrics:
                # Register memory metrics
                self.metrics.register_gauge("memory_profiler_heap_usage_mb", "Current heap memory usage in MB")
                self.metrics.register_gauge("memory_profiler_system_usage_mb", "Current system memory usage in MB")
                self.metrics.register_gauge("memory_profiler_gpu_usage_mb", "Current GPU memory usage in MB")
                self.metrics.register_counter("memory_profiler_leaks_detected", "Number of memory leaks detected")
                self.metrics.register_counter("memory_profiler_optimizations_performed", "Number of memory optimizations performed")
                self.metrics.register_histogram("memory_profiler_allocation_size_mb", "Memory allocation sizes in MB")
                
        except Exception as e:
            self.logger.warning(f"Failed to setup monitoring: {str(e)}")
    
    def _setup_event_handlers(self) -> None:
        """Setup event handlers for system integration."""
        try:
            # Listen for component health changes
            self.event_bus.subscribe("component_health_changed", self._handle_component_health_change)
            
            # Listen for system state changes
            self.event_bus.subscribe("system_state_changed", self._handle_system_state_change)
            
        except Exception as e:
            self.logger.warning(f"Failed to setup event handlers: {str(e)}")
    
    def _setup_health_check(self) -> None:
        """Setup health check integration."""
        try:
            if self.health_check:
                self.health_check.register_component("memory_profiler", self._health_check_callback)
                
        except Exception as e:
            self.logger.warning(f"Failed to setup health check: {str(e)}")
    
    def _log_dependency_status(self) -> None:
        """Log the status of optional dependencies."""
        self.logger.info(f"psutil available: {PSUTIL_AVAILABLE}")
        self.logger.info(f"GPU monitoring available: {GPU_MONITORING_AVAILABLE}")
        
        if not PSUTIL_AVAILABLE:
            self.logger.warning("psutil not available - system memory monitoring will be limited")
        
        if not GPU_MONITORING_AVAILABLE:
            self.logger.info("py3nvml not available - GPU memory monitoring disabled")
    
    @handle_exceptions
    async def start_profiling(
        self,
        session_name: str = None,
        description: str = None,
        mode: MemoryProfilingMode = None,
        level: MemoryTrackingLevel = None
    ) -> str:
        """
        Start memory profiling session.
        
        Args:
            session_name: Name for the profiling session
            description: Session description
            mode: Profiling mode override
            level: Tracking level override
            
        Returns:
            Session ID
        """
        if self.status == MemoryProfilerStatus.RUNNING:
            raise RuntimeError("Memory profiler is already running")
        
        try:
            with self.profiler_lock:
                self.status = MemoryProfilerStatus.STARTING
                
                # Generate session ID and name
                session_id = str(uuid.uuid4())
                session_name = session_name or f"memory_profile_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
                # Use provided config overrides or defaults
                config = MemoryProfilingConfig(
                    mode=mode or self.config.mode,
                    level=level or self.config.level,
                    **{k: v for k, v in asdict(self.config).items() 
                       if k not in ['mode', 'level']}
                )
                
                # Create session
                self.current_session = MemoryProfilingSession(
                    session_id=session_id,
                    name=session_name,
                    description=description,
                    config=config
                )
                
                # Clear previous data
                self.memory_snapshots.clear()
                
                # Start monitoring based on mode
                if config.mode != MemoryProfilingMode.OFF:
                    self._start_monitoring_thread()
                
                # Emit profiler started event
                await self.event_bus.emit(MemoryProfilerStarted(
                    session_id=session_id,
                    session_name=session_name,
                    profiling_mode=config.mode.value
                ))
                
                # Update metrics
                if self.metrics:
                    self.metrics.increment("memory_profiler_sessions_started")
                    self.metrics.set("memory_profiler_active_sessions", 1)
                
                self.status = MemoryProfilerStatus.RUNNING
                
                self.logger.info(f"Started memory profiling session: {session_id} (mode: {config.mode.value})")
                
                return session_id
                
        except Exception as e:
            self.status = MemoryProfilerStatus.ERROR
            self.logger.error(f"Failed to start memory profiling: {str(e)}")
            raise
    
    def _start_monitoring_thread(self) -> None:
        """Start the memory monitoring thread."""
        self.stop_monitoring_event.clear()
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitoring_thread.start()
    
    def _monitoring_loop(self) -> None:
        """Main memory monitoring loop."""
        self.logger.info("Memory monitoring loop started")
        
        try:
            while not self.stop_monitoring_event.is_set():
                try:
                    # Take memory snapshot
                    snapshot = self._take_memory_snapshot()
                    self.memory_snapshots.append(snapshot)
                    
                    # Update metrics
                    self._update_metrics(snapshot)
                    
                    # Check for threshold violations
                    self._check_thresholds(snapshot)
                    
                    # Perform leak detection if enabled
                    if (self.config.enable_leak_detection and 
                        len(self.memory_snapshots) >= 5):
                        self._perform_leak_detection()
                    
                    # Perform optimization if needed
                    if self.config.enable_auto_optimization:
                        self._check_optimization_needed(snapshot)
                    
                except Exception as e:
                    self.logger.error(f"Error in monitoring loop: {str(e)}")
                
                # Wait for next iteration
                self.stop_monitoring_event.wait(self.config.sampling_interval)
                
        except Exception as e:
            self.logger.error(f"Memory monitoring loop failed: {str(e)}")
        finally:
            self.logger.info("Memory monitoring loop stopped")
    
    def _take_memory_snapshot(self) -> MemorySnapshot:
        """Take a comprehensive memory snapshot."""
        snapshot = MemorySnapshot()
        
        try:
            # System memory information
            if PSUTIL_AVAILABLE:
                system_memory = psutil.virtual_memory()
                snapshot.system_memory_mb = system_memory.total / (1024 * 1024)
                snapshot.system_available_mb = system_memory.available / (1024 * 1024)
                snapshot.system_percent = system_memory.percent
                
                # Process memory information
                process = psutil.Process()
                process_memory = process.memory_info()
                snapshot.process_rss_mb = process_memory.rss / (1024 * 1024)
                snapshot.process_vms_mb = process_memory.vms / (1024 * 1024)
                snapshot.process_percent = process.memory_percent()
            
            # Python heap information
            if tracemalloc.is_tracing():
                current, peak = tracemalloc.get_traced_memory()
                snapshot.heap_size_mb = current / (1024 * 1024)
                snapshot.heap_peak_mb = peak / (1024 * 1024)
                
                # Get top allocations
                if self.config.track_call_stacks:
                    top_stats = tracemalloc.take_snapshot().statistics('lineno')
                    snapshot.top_allocations = [
                        {
                            "filename": stat.traceback.format()[0] if stat.traceback.format() else "unknown",
                            "size_mb": stat.size / (1024 * 1024),
                            "count": stat.count
                        }
                        for stat in top_stats[:10]
                    ]
            
            # GPU memory (if available)
            if GPU_MONITORING_AVAILABLE:
                try:
                    py3nvml.nvmlInit()
                    device_count = py3nvml.nvmlDeviceGetCount()
                    if device_count > 0:
                        handle = py3nvml.nvmlDeviceGetHandleByIndex(0)
                        memory_info = py3nvml.nvmlDeviceGetMemoryInfo(handle)
                        snapshot.gpu_memory_mb = (memory_info.total - memory_info.free) / (1024 * 1024)
                        snapshot.gpu_available_mb = memory_info.free / (1024 * 1024)
                        snapshot.gpu_percent = ((memory_info.total - memory_info.free) / memory_info.total) * 100
                except Exception as e:
                    self.logger.debug(f"GPU memory monitoring failed: {e}")
            
            # Garbage collection information
            snapshot.gc_generation_counts = list(gc.get_count())
            snapshot.gc_collection_count = sum(gc.get_stats()[i]['collections'] for i in range(len(gc.get_stats())))
            
            # Component memory (from tracker)
            component_profiles = self.component_tracker.get_all_profiles()
            snapshot.component_memory = {
                comp_id: profile.current_memory_mb 
                for comp_id, profile in component_profiles.items()
            }
            
            # Calculate derived metrics
            if len(self.memory_snapshots) > 0:
                prev_snapshot = self.memory_snapshots[-1]
                time_diff = (snapshot.timestamp - prev_snapshot.timestamp).total_seconds()
                if time_diff > 0:
                    memory_diff = snapshot.heap_size_mb - prev_snapshot.heap_size_mb
                    snapshot.allocation_rate_mb_per_sec = memory_diff / time_diff
            
        except Exception as e:
            self.logger.error(f"Failed to take memory snapshot: {str(e)}")
        
        return snapshot
    
    def _update_metrics(self, snapshot: MemorySnapshot) -> None:
        """Update metrics with snapshot data."""
        try:
            if self.metrics:
                self.metrics.set("memory_profiler_heap_usage_mb", snapshot.heap_size_mb)
                self.metrics.set("memory_profiler_system_usage_mb", snapshot.process_rss_mb)
                self.metrics.set("memory_profiler_gpu_usage_mb", snapshot.gpu_memory_mb)
                
        except Exception as e:
            self.logger.warning(f"Failed to update metrics: {str(e)}")
    
    def _check_thresholds(self, snapshot: MemorySnapshot) -> None:
        """Check memory thresholds and emit alerts."""
        try:
            # Check heap threshold
            if snapshot.heap_size_mb > self.config.heap_threshold_mb:
                asyncio.create_task(self.event_bus.emit(MemoryThresholdExceeded(
                    threshold_type="heap",
                    current_usage_mb=snapshot.heap_size_mb,
                    threshold_mb=self.config.heap_threshold_mb
                )))
            
            # Check system threshold
            if snapshot.process_rss_mb > self.config.system_threshold_mb:
                asyncio.create_task(self.event_bus.emit(MemoryThresholdExceeded(
                    threshold_type="system",
                    current_usage_mb=snapshot.process_rss_mb,
                    threshold_mb=self.config.system_threshold_mb
                )))
            
            # Check GPU threshold
            if snapshot.gpu_memory_mb > self.config.gpu_threshold_mb:
                asyncio.create_task(self.event_bus.emit(MemoryThresholdExceeded(
                    threshold_type="gpu",
                    current_usage_mb=snapshot.gpu_memory_mb,
                    threshold_mb=self.config.gpu_threshold_mb
                )))
                
        except Exception as e:
            self.logger.warning(f"Failed to check thresholds: {str(e)}")
    
    def _perform_leak_detection(self) -> None:
        """Perform memory leak detection."""
        try:
            snapshots = list(self.memory_snapshots)
            component_profiles = self.component_tracker.get_all_profiles()
            
            detected_leaks = self.leak_detector.analyze_for_leaks(snapshots, component_profiles)
            
            for leak in detected_leaks:
                # Emit leak detection event
                asyncio.create_task(self.event_bus.emit(MemoryLeakDetected(
                    component_id=leak.component_id or "",
                    leak_size_mb=leak.total_leaked_mb,
                    leak_type=leak.leak_type,
                    severity=leak.severity.value
                )))
                
                # Update metrics
                if self.metrics:
                    self.metrics.increment("memory_profiler_leaks_detected")
                
                # Add to current session
                if self.current_session:
                    self.current_session.detected_leaks.append(leak)
                
                self.logger.warning(f"Memory leak detected: {leak.leak_type} in {leak.component_id or 'system'}")
                
        except Exception as e:
            self.logger.error(f"Leak detection failed: {str(e)}")
    
    def _check_optimization_needed(self, snapshot: MemorySnapshot) -> None:
        """Check if memory optimization is needed."""
        try:
            if snapshot.process_rss_mb > self.config.auto_gc_threshold_mb:
                optimization_result = self.memory_optimizer.optimize_memory(snapshot)
                
                if optimization_result["performed"]:
                    # Emit optimization event
                    asyncio.create_task(self.event_bus.emit(MemoryOptimizationPerformed(
                        optimization_type="auto_gc",
                        memory_freed_mb=optimization_result["memory_freed_mb"]
                    )))
                    
                    # Update metrics
                    if self.metrics:
                        self.metrics.increment("memory_profiler_optimizations_performed")
                    
                    # Add to current session
                    if self.current_session:
                        self.current_session.optimizations_performed.append(optimization_result)
                        self.current_session.memory_freed_mb += optimization_result["memory_freed_mb"]
                        
        except Exception as e:
            self.logger.error(f"Optimization check failed: {str(e)}")
    
    @handle_exceptions
    async def stop_profiling(self) -> MemoryProfilingSession:
        """
        Stop memory profiling session.
        
        Returns:
            Completed profiling session
        """
        if self.status != MemoryProfilerStatus.RUNNING:
            raise RuntimeError("Memory profiler is not running")
        
        try:
            with self.profiler_lock:
                self.status = MemoryProfilerStatus.STOPPING
                session_start_time = time.perf_counter()
                
                # Stop monitoring thread
                if self.monitoring_thread and self.monitoring_thread.is_alive():
                    self.stop_monitoring_event.set()
                    self.monitoring_thread.join(timeout=5.0)
                
                # Complete session
                session = self.current_session
                if session:
                    session.end_time = datetime.now(timezone.utc)
                    session.duration = (session.end_time - session.start_time).total_seconds()
                    
                    # Store snapshots in session
                    session.snapshots = list(self.memory_snapshots)
                    
                    # Store component profiles
                    session.component_profiles = self.component_tracker.get_all_profiles()
                    
                    # Calculate session statistics
                    if session.snapshots:
                        session.peak_memory_mb = max(s.process_rss_mb for s in session.snapshots)
                        session.average_memory_mb = statistics.mean([s.process_rss_mb for s in session.snapshots])
                        
                        # Calculate memory growth rate
                        if len(session.snapshots) >= 2:
                            time_span = (session.snapshots[-1].timestamp - session.snapshots[0].timestamp).total_seconds()
                            memory_growth = session.snapshots[-1].process_rss_mb - session.snapshots[0].process_rss_mb
                            session.memory_growth_rate_mb_per_hour = (memory_growth / time_span * 3600) if time_span > 0 else 0
                    
                    # Perform final analysis
                    await self._generate_session_analysis(session)
                    
                    # Store session
                    self.profiling_sessions[session.session_id] = session
                    self.recent_sessions.append(session)
                    
                    # Emit profiler stopped event
                    await self.event_bus.emit(MemoryProfilerStopped(
                        session_id=session.session_id,
                        duration=session.duration,
                        peak_memory_mb=session.peak_memory_mb,
                        memory_leaks_detected=len(session.detected_leaks)
                    ))
                    
                    # Update metrics
                    if self.metrics:
                        self.metrics.record("memory_profiler_session_duration_seconds", session.duration)
                        self.metrics.increment("memory_profiler_sessions_completed")
                        self.metrics.set("memory_profiler_active_sessions", 0)
                    
                    self.status = MemoryProfilerStatus.STOPPED
                    self.current_session = None
                    
                    processing_time = time.perf_counter() - session_start_time
                    self.logger.info(
                        f"Stopped memory profiling session: {session.session_id} "
                        f"(duration: {session.duration:.2f}s, processed in: {processing_time:.2f}s)"
                    )
                    
                    return session
                else:
                    raise RuntimeError("No active session to stop")
                    
        except Exception as e:
            self.status = MemoryProfilerStatus.ERROR
            self.logger.error(f"Failed to stop memory profiling: {str(e)}")
            raise
    
    async def _generate_session_analysis(self, session: MemoryProfilingSession) -> None:
        """Generate comprehensive analysis for the session."""
        try:
            # Memory trend analysis
            if session.snapshots:
                trend_analysis = self.memory_analyzer.analyze_memory_trend(session.snapshots)
                session.memory_growth_rate_mb_per_hour = trend_analysis.get("growth_rate_mb_per_hour", 0)
            
            # Pattern detection
            if session.component_profiles:
                patterns = self.memory_analyzer.detect_memory_patterns(session.component_profiles)
                # Store patterns in session metadata or logs
                for pattern in patterns:
                    self.logger.info(f"Memory pattern detected: {pattern}")
            
            # Save session data to file
            await self._save_session_data(session)
            
        except Exception as e:
            self.logger.error(f"Failed to generate session analysis: {str(e)}")
    
    async def _save_session_data(self, session: MemoryProfilingSession) -> None:
        """Save session data to file."""
        try:
            session_file = self.output_dir / f"memory_session_{session.session_id}.json"
            
            # Convert session to serializable format
            session_data = {
                "session_id": session.session_id,
                "name": session.name,
                "description": session.description,
                "start_time": session.start_time.isoformat(),
                "end_time": session.end_time.isoformat() if session.end_time else None,
                "duration": session.duration,
                "config": asdict(session.config),
                "peak_memory_mb": session.peak_memory_mb,
                "average_memory_mb": session.average_memory_mb,
                "memory_growth_rate_mb_per_hour": session.memory_growth_rate_mb_per_hour,
                "detected_leaks": [asdict(leak) for leak in session.detected_leaks],
                "optimizations_performed": session.optimizations_performed,
                "memory_freed_mb": session.memory_freed_mb,
                "snapshot_count": len(session.snapshots),
                "component_count": len(session.component_profiles)
            }
            
            with open(session_file, 'w') as f:
                json.dump(session_data, f, indent=2, default=str)
            
            session.session_file_path = session_file
            self.logger.info(f"Session data saved to {session_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save session data: {str(e)}")
    
    # Event handlers
    async def _handle_component_health_change(self, event) -> None:
        """Handle component health change events."""
        try:
            component_id = getattr(event, 'component_id', None)
            health_status = getattr(event, 'status', None)
            
            if component_id and health_status == 'unhealthy':
                # Check if this component has memory issues
                profile = self.component_tracker.get_component_profile(component_id)
                if profile and profile.current_memory_mb > 100:  # Arbitrary threshold
                    self.logger.warning(f"Unhealthy component {component_id} has high memory usage: {profile.current_memory_mb:.2f} MB")
                    
        except Exception as e:
            self.logger.error(f"Error handling component health change: {str(e)}")
    
    async def _handle_system_state_change(self, event) -> None:
        """Handle system state change events."""
        try:
            state = getattr(event, 'state', None)
            
            if state == 'shutdown':
                # Automatically stop profiling if system is shutting down
                if self.status == MemoryProfilerStatus.RUNNING:
                    self.logger.info("System shutdown detected, stopping memory profiling")
                    await self.stop_profiling()
                    
        except Exception as e:
            self.logger.error(f"Error handling system state change: {str(e)}")
    
    # Health check callback
    async def _health_check_callback(self) -> Dict[str, Any]:
        """Health check callback for the memory profiler."""
        return {
            "status": self.status.value,
            "active_session": self.current_session.session_id if self.current_session else None,
            "total_sessions": len(self.profiling_sessions),
            "current_snapshots": len(self.memory_snapshots),
            "psutil_available": PSUTIL_AVAILABLE,
            "gpu_monitoring_available": GPU_MONITORING_AVAILABLE,
            "tracemalloc_active": tracemalloc.is_tracing()
        }
    
    # Public API methods
    def register_component(self, component_id: str, component_type: str, component_instance: Any = None) -> None:
        """Register a component for memory tracking."""
        self.component_tracker.register_component(component_id, component_type, component_instance)
        self.logger.debug(f"Registered component for memory tracking: {component_id}")
    
    def get_current_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage summary."""
        if self.memory_snapshots:
            latest_snapshot = self.memory_snapshots[-1]
            return {
                "timestamp": latest_snapshot.timestamp.isoformat(),
                "heap_mb": latest_snapshot.heap_size_mb,
                "system_mb": latest_snapshot.process_rss_mb,
                "gpu_mb": latest_snapshot.gpu_memory_mb,
                "system_percent": latest_snapshot.system_percent,
                "component_memory": latest_snapshot.component_memory
            }
        else:
            return {"error": "No memory data available"}
    
    def get_memory_statistics(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics."""
        stats = {
            "profiler_status": self.status.value,
            "total_sessions": len(self.profiling_sessions),
            "active_session": self.current_session.session_id if self.current_session else None,
            "total_snapshots": len(self.memory_snapshots),
            "dependencies": {
                "psutil_available": PSUTIL_AVAILABLE,
                "gpu_monitoring_available": GPU_MONITORING_AVAILABLE,
                "tracemalloc_active": tracemalloc.is_tracing()
            }
        }
        
        if self.memory_snapshots:
            snapshots = list(self.memory_snapshots)
            memory_values = [s.process_rss_mb for s in snapshots]
            
            stats.update({
                "current_memory_mb": memory_values[-1] if memory_values else 0,
                "peak_memory_mb": max(memory_values) if memory_values else 0,
                "average_memory_mb": statistics.mean(memory_values) if memory_values else 0,
                "memory_trend": self.memory_analyzer.analyze_memory_trend(snapshots).get("trend", "unknown")
            })
        
        return stats
    
    async def force_optimization(self) -> Dict[str, Any]:
        """Force memory optimization."""
        if self.memory_snapshots:
            latest_snapshot = self.memory_snapshots[-1]
            return self.memory_optimizer.optimize_memory(latest_snapshot, force=True)
        else:
            return {"error": "No memory data available for optimization"}
    
    def get_session_history(self) -> List[Dict[str, Any]]:
        """Get history of completed profiling sessions."""
        return [
            {
                "session_id": session.session_id,
                "name": session.name,
                "start_time": session.start_time.isoformat(),
                "duration": session.duration,
                "peak_memory_mb": session.peak_memory_mb,
                "leaks_detected": len(session.detected_leaks),
                "optimizations_performed": len(session.optimizations_performed)
            }
            for session in self.recent_sessions
        ]
    
    async def cleanup(self) -> None:
        """Cleanup profiler resources."""
        try:
            if self.status == MemoryProfilerStatus.RUNNING:
                await self.stop_profiling()
            
            if self.monitoring_thread and self.monitoring_thread.is_alive():
                self.stop_monitoring_event.set()
                self.monitoring_thread.join(timeout=5.0)
            
            self.executor.shutdown(wait=True)
            
            self.logger.info("MemoryProfiler cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")


# Decorator for component memory tracking
def track_memory(component_id: str, profiler_instance: MemoryProfiler = None):
    """
    Decorator to track memory usage for specific functions/methods.
    
    Args:
        component_id: Component identifier for tracking
        profiler_instance: Memory profiler instance (optional)
    """
    def decorator(func: F) -> F:
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                # Track memory before and after function execution
                try:
                    if profiler_instance and tracemalloc.is_tracing():
                        current, _ = tracemalloc.get_traced_memory()
                        start_memory = current / (1024 * 1024)
                        
                        result = await func(*args, **kwargs)
                        
                        current, _ = tracemalloc.get_traced_memory()
                        end_memory = current / (1024 * 1024)
                        memory_diff = end_memory - start_memory
                        
                        if memory_diff > 0:
                            profiler_instance.component_tracker.track_allocation(component_id, int(memory_diff * 1024 * 1024))
                        elif memory_diff < 0:
                            profiler_instance.component_tracker.track_deallocation(component_id, int(abs(memory_diff) * 1024 * 1024))
                    else:
                        result = await func(*args, **kwargs)
                    
                    return result
                except Exception as e:
                    # Log but don't interfere with the original function
                    if profiler_instance:
                        profiler_instance.logger.debug(f"Memory tracking failed for {component_id}: {e}")
                    return await func(*args, **kwargs)
            
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                try:
                    if profiler_instance and tracemalloc.is_tracing():
                        current, _ = tracemalloc.get_traced_memory()
                        start_memory = current / (1024 * 1024)
                        
                        result = func(*args, **kwargs)
                        
                        current, _ = tracemalloc.get_traced_memory()
                        end_memory = current / (1024 * 1024)
                        memory_diff = end_memory - start_memory
                        
                        if memory_diff > 0:
                            profiler_instance.component_tracker.track_allocation(component_id, int(memory_diff * 1024 * 1024))
                        elif memory_diff < 0:
                            profiler_instance.component_tracker.track_deallocation(component_id, int(abs(memory_diff) * 1024 * 1024))
                    else:
                        result = func(*args, **kwargs)
                    
                    return result
                except Exception as e:
                    if profiler_instance:
                        profiler_instance.logger.debug(f"Memory tracking failed for {component_id}: {e}")
                    return func(*args, **kwargs)
            
            return sync_wrapper
    
    return decorator