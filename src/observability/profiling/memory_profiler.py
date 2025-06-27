"""
Advanced Memory Profiler for AI Assistant
Author: Drmusab
Last Modified: 2025-01-11 15:30:00 UTC

This module provides comprehensive memory profiling and monitoring capabilities
for the AI assistant system, including real-time monitoring, memory leak detection,
allocation tracking, and integration with all core system components.
"""

import asyncio
import gc
import sys
import os
import psutil
import tracemalloc
import threading
import time
import weakref
import json
import uuid
import pickle
import gzip
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable, TypeVar, Union, Set, Tuple
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
from contextlib import contextmanager, asynccontextmanager
from collections import defaultdict, deque
import concurrent.futures

# Core imports
from src.core.config.loader import ConfigLoader
from src.core.events.event_bus import EventBus
from src.core.events.event_types import (
    PerformanceThresholdExceeded, ComponentHealthChanged, SystemStateChanged,
    MemoryOperationStarted, MemoryOperationCompleted
)
from src.core.error_handling import ErrorHandler, handle_exceptions
from src.core.dependency_injection import Container
from src.core.health_check import HealthCheck

# Observability components
from src.observability.monitoring.metrics import MetricsCollector
from src.observability.monitoring.tracing import TraceManager
from src.observability.logging.config import get_logger

# Optional imports for advanced profiling
try:
    import pympler
    from pympler import tracker, muppy, summary
    PYMPLER_AVAILABLE = True
except ImportError:
    PYMPLER_AVAILABLE = False

try:
    import objgraph
    OBJGRAPH_AVAILABLE = True
except ImportError:
    OBJGRAPH_AVAILABLE = False

# Type definitions
F = TypeVar('F', bound=Callable)


class MemoryProfilingMode(Enum):
    """Memory profiling modes."""
    OFF = "off"
    BASIC = "basic"                    # Basic memory usage tracking
    DETAILED = "detailed"              # Detailed allocation tracking
    LEAK_DETECTION = "leak_detection"  # Focus on memory leak detection
    COMPONENT_TRACKING = "component"   # Component-specific memory tracking
    PRODUCTION = "production"          # Optimized for production use


class MemoryProfilingLevel(Enum):
    """Memory profiling detail levels."""
    LOW = "low"           # Basic memory usage only
    MEDIUM = "medium"     # Memory usage + allocation patterns
    HIGH = "high"         # Detailed allocation tracking + object lifecycle
    DETAILED = "detailed" # Maximum detail with full call stacks


class MemoryProfilerStatus(Enum):
    """Memory profiler operational status."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    ERROR = "error"


class MemoryMetricType(Enum):
    """Types of memory metrics."""
    RSS = "rss"                    # Resident Set Size
    VMS = "vms"                    # Virtual Memory Size
    USS = "uss"                    # Unique Set Size
    PSS = "pss"                    # Proportional Set Size
    HEAP = "heap"                  # Python heap size
    PEAK = "peak"                  # Peak memory usage
    ALLOCATIONS = "allocations"    # Number of allocations
    DEALLOCATIONS = "deallocations" # Number of deallocations
    GROWTH_RATE = "growth_rate"    # Memory growth rate
    FRAGMENTATION = "fragmentation" # Memory fragmentation


@dataclass
class MemoryProfilingConfig:
    """Configuration for memory profiling."""
    mode: MemoryProfilingMode = MemoryProfilingMode.BASIC
    level: MemoryProfilingLevel = MemoryProfilingLevel.MEDIUM
    
    # Monitoring intervals
    monitoring_interval: float = 1.0  # seconds
    snapshot_interval: float = 30.0   # seconds
    gc_monitoring_interval: float = 5.0  # seconds
    
    # Memory thresholds
    memory_threshold_mb: float = 1000.0
    growth_rate_threshold_mb_per_min: float = 50.0
    leak_detection_threshold_mb: float = 100.0
    fragmentation_threshold_percent: float = 30.0
    
    # Data retention
    max_snapshots: int = 100
    snapshot_retention_hours: int = 24
    max_tracked_objects: int = 10000
    
    # Detection settings
    enable_leak_detection: bool = True
    enable_growth_analysis: bool = True
    enable_component_tracking: bool = True
    enable_gc_monitoring: bool = True
    enable_object_tracking: bool = True
    
    # Performance settings
    sampling_rate: float = 1.0  # 1.0 = 100% sampling
    max_stack_depth: int = 10
    track_allocations: bool = True
    
    # Integration settings
    integrate_with_tracing: bool = True
    integrate_with_metrics: bool = True
    emit_events: bool = True


@dataclass
class MemorySnapshot:
    """Represents a memory snapshot at a point in time."""
    snapshot_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # System memory metrics
    process_memory: Dict[str, float] = field(default_factory=dict)
    system_memory: Dict[str, float] = field(default_factory=dict)
    
    # Python-specific metrics
    heap_size: int = 0
    object_count: int = 0
    gc_stats: Dict[str, Any] = field(default_factory=dict)
    
    # Component memory usage
    component_memory: Dict[str, float] = field(default_factory=dict)
    
    # Memory allocation tracking
    allocations: Dict[str, int] = field(default_factory=dict)
    deallocations: Dict[str, int] = field(default_factory=dict)
    
    # Memory patterns
    growth_rate: float = 0.0
    fragmentation_ratio: float = 0.0
    
    # Tracemalloc data (if available)
    tracemalloc_stats: Optional[Dict[str, Any]] = None


@dataclass
class MemoryLeak:
    """Represents a detected memory leak."""
    leak_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    detected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    component: str = ""
    leak_type: str = ""  # "growth", "allocation", "reference"
    severity: str = "medium"  # "low", "medium", "high", "critical"
    
    # Leak characteristics
    memory_growth_mb: float = 0.0
    growth_rate_mb_per_min: float = 0.0
    duration_minutes: float = 0.0
    
    # Root cause analysis
    suspected_objects: List[str] = field(default_factory=list)
    call_stacks: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    # Tracking
    first_detected: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    resolved: bool = False


@dataclass
class ComponentMemoryProfile:
    """Memory profile for a specific component."""
    component_id: str
    component_name: str
    
    # Memory usage stats
    current_memory_mb: float = 0.0
    peak_memory_mb: float = 0.0
    average_memory_mb: float = 0.0
    baseline_memory_mb: float = 0.0
    
    # Allocation stats
    total_allocations: int = 0
    total_deallocations: int = 0
    current_objects: int = 0
    
    # Time tracking
    first_seen: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Patterns and analysis
    memory_pattern: str = "stable"  # "stable", "growing", "oscillating", "leaking"
    efficiency_score: float = 1.0
    
    # Historical data
    memory_history: deque = field(default_factory=lambda: deque(maxlen=1000))


@dataclass
class MemoryProfilingSession:
    """Represents a memory profiling session."""
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_name: str = ""
    description: str = ""
    
    # Session metadata
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    end_time: Optional[datetime] = None
    duration: Optional[timedelta] = None
    
    # Configuration
    config: MemoryProfilingConfig = field(default_factory=MemoryProfilingConfig)
    
    # Data collection
    snapshots: List[MemorySnapshot] = field(default_factory=list)
    component_profiles: Dict[str, ComponentMemoryProfile] = field(default_factory=dict)
    detected_leaks: List[MemoryLeak] = field(default_factory=list)
    
    # Analysis results
    memory_efficiency: float = 1.0
    leak_risk_score: float = 0.0
    optimization_opportunities: List[str] = field(default_factory=list)
    
    # Output paths
    snapshot_file_path: Optional[Path] = None
    report_path: Optional[Path] = None
    analysis_path: Optional[Path] = None


class MemoryTracker:
    """Tracks memory usage for specific components or operations."""
    
    def __init__(self, component_name: str, profiler: 'EnhancedMemoryProfiler'):
        self.component_name = component_name
        self.profiler = profiler
        self.tracking_data: Dict[str, Any] = {}
        self.start_memory = None
        self.active = False
    
    def start_tracking(self):
        """Start memory tracking."""
        if self.profiler.is_profiling():
            self.start_memory = self._get_current_memory()
            self.active = True
    
    def stop_tracking(self) -> Dict[str, float]:
        """Stop memory tracking and return usage statistics."""
        if not self.active or not self.start_memory:
            return {}
        
        end_memory = self._get_current_memory()
        usage_stats = {
            'memory_used_mb': (end_memory.get('rss', 0) - self.start_memory.get('rss', 0)) / (1024 * 1024),
            'peak_memory_mb': end_memory.get('peak', 0) / (1024 * 1024),
            'start_memory_mb': self.start_memory.get('rss', 0) / (1024 * 1024),
            'end_memory_mb': end_memory.get('rss', 0) / (1024 * 1024)
        }
        
        self.active = False
        self.profiler._record_component_memory(self.component_name, usage_stats)
        
        return usage_stats
    
    def _get_current_memory(self) -> Dict[str, float]:
        """Get current memory usage."""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            
            return {
                'rss': memory_info.rss,
                'vms': memory_info.vms,
                'peak': getattr(memory_info, 'peak_wset', memory_info.rss) if hasattr(memory_info, 'peak_wset') else memory_info.rss
            }
        except Exception:
            return {'rss': 0, 'vms': 0, 'peak': 0}


class MemoryLeakDetector:
    """Detects memory leaks using various algorithms."""
    
    def __init__(self, logger, config: MemoryProfilingConfig):
        self.logger = logger
        self.config = config
        self.memory_history: deque = deque(maxlen=1000)
        self.component_baselines: Dict[str, float] = {}
        self.leak_candidates: Dict[str, Dict[str, Any]] = {}
    
    def add_snapshot(self, snapshot: MemorySnapshot) -> List[MemoryLeak]:
        """Add a memory snapshot and analyze for leaks."""
        self.memory_history.append(snapshot)
        detected_leaks = []
        
        # Growth-based leak detection
        if len(self.memory_history) >= 5:
            detected_leaks.extend(self._detect_growth_leaks(snapshot))
        
        # Component-based leak detection
        detected_leaks.extend(self._detect_component_leaks(snapshot))
        
        # Pattern-based leak detection
        if len(self.memory_history) >= 10:
            detected_leaks.extend(self._detect_pattern_leaks(snapshot))
        
        return detected_leaks
    
    def _detect_growth_leaks(self, current_snapshot: MemorySnapshot) -> List[MemoryLeak]:
        """Detect leaks based on memory growth patterns."""
        leaks = []
        
        if len(self.memory_history) < 5:
            return leaks
        
        # Analyze recent memory growth
        recent_snapshots = list(self.memory_history)[-5:]
        memory_values = [s.process_memory.get('rss', 0) for s in recent_snapshots]
        
        # Calculate growth rate
        if len(memory_values) >= 2:
            time_diff = (recent_snapshots[-1].timestamp - recent_snapshots[0].timestamp).total_seconds() / 60
            if time_diff > 0:
                memory_diff_mb = (memory_values[-1] - memory_values[0]) / (1024 * 1024)
                growth_rate = memory_diff_mb / time_diff
                
                if growth_rate > self.config.growth_rate_threshold_mb_per_min:
                    leak = MemoryLeak(
                        component="system",
                        leak_type="growth",
                        severity="high" if growth_rate > self.config.growth_rate_threshold_mb_per_min * 2 else "medium",
                        memory_growth_mb=memory_diff_mb,
                        growth_rate_mb_per_min=growth_rate,
                        duration_minutes=time_diff,
                        recommendations=[
                            f"Memory growing at {growth_rate:.2f} MB/min",
                            "Check for object references not being released",
                            "Review recent code changes for memory management"
                        ]
                    )
                    leaks.append(leak)
        
        return leaks
    
    def _detect_component_leaks(self, current_snapshot: MemorySnapshot) -> List[MemoryLeak]:
        """Detect leaks in specific components."""
        leaks = []
        
        for component, memory_usage in current_snapshot.component_memory.items():
            # Establish baseline if not exists
            if component not in self.component_baselines:
                self.component_baselines[component] = memory_usage
                continue
            
            baseline = self.component_baselines[component]
            memory_increase = memory_usage - baseline
            
            if memory_increase > self.config.leak_detection_threshold_mb:
                leak = MemoryLeak(
                    component=component,
                    leak_type="component",
                    severity="medium",
                    memory_growth_mb=memory_increase,
                    recommendations=[
                        f"Component {component} memory increased by {memory_increase:.2f} MB",
                        f"Review {component} for proper resource cleanup",
                        "Check for cached objects not being released"
                    ]
                )
                leaks.append(leak)
        
        return leaks
    
    def _detect_pattern_leaks(self, current_snapshot: MemorySnapshot) -> List[MemoryLeak]:
        """Detect leaks based on memory usage patterns."""
        leaks = []
        
        if len(self.memory_history) < 10:
            return leaks
        
        # Analyze memory pattern over last 10 snapshots
        recent_memory = [s.process_memory.get('rss', 0) for s in list(self.memory_history)[-10:]]
        
        # Check for consistent upward trend
        increasing_count = sum(1 for i in range(1, len(recent_memory)) if recent_memory[i] > recent_memory[i-1])
        
        if increasing_count >= 8:  # 80% of measurements increasing
            total_increase = (recent_memory[-1] - recent_memory[0]) / (1024 * 1024)
            
            if total_increase > self.config.leak_detection_threshold_mb / 2:
                leak = MemoryLeak(
                    component="system",
                    leak_type="pattern",
                    severity="medium",
                    memory_growth_mb=total_increase,
                    recommendations=[
                        "Consistent memory growth pattern detected",
                        "Memory increasing in 80% of recent measurements",
                        "Investigate for accumulating objects or caches"
                    ]
                )
                leaks.append(leak)
        
        return leaks


class GarbageCollectionMonitor:
    """Monitors Python garbage collection behavior."""
    
    def __init__(self, logger):
        self.logger = logger
        self.gc_stats_history: deque = deque(maxlen=1000)
        self.last_gc_stats = None
    
    def collect_gc_stats(self) -> Dict[str, Any]:
        """Collect current garbage collection statistics."""
        try:
            # Get GC stats
            gc_stats = {
                'collections': list(gc.get_stats()),
                'count': gc.get_count(),
                'threshold': gc.get_threshold(),
                'flags': gc.get_debug() if hasattr(gc, 'get_debug') else 0,
                'objects': len(gc.get_objects()) if hasattr(gc, 'get_objects') else 0
            }
            
            # Add timing if previous stats available
            if self.last_gc_stats:
                for i, (current_gen, last_gen) in enumerate(zip(gc_stats['collections'], self.last_gc_stats['collections'])):
                    gc_stats[f'gen_{i}_collections_delta'] = current_gen.get('collections', 0) - last_gen.get('collections', 0)
            
            self.gc_stats_history.append(gc_stats)
            self.last_gc_stats = gc_stats
            
            return gc_stats
            
        except Exception as e:
            self.logger.error(f"Failed to collect GC stats: {str(e)}")
            return {}
    
    def analyze_gc_performance(self) -> Dict[str, Any]:
        """Analyze garbage collection performance."""
        if not self.gc_stats_history:
            return {}
        
        try:
            recent_stats = list(self.gc_stats_history)[-10:]
            
            # Calculate GC frequency
            total_collections = sum(
                stat['collections'][0].get('collections', 0) if stat['collections'] else 0
                for stat in recent_stats
            )
            
            avg_objects = sum(stat.get('objects', 0) for stat in recent_stats) / len(recent_stats)
            
            analysis = {
                'avg_objects': avg_objects,
                'total_recent_collections': total_collections,
                'gc_pressure': total_collections / len(recent_stats) if recent_stats else 0,
                'threshold_efficiency': self._analyze_threshold_efficiency(recent_stats)
            }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Failed to analyze GC performance: {str(e)}")
            return {}
    
    def _analyze_threshold_efficiency(self, stats_list: List[Dict[str, Any]]) -> Dict[str, float]:
        """Analyze GC threshold efficiency."""
        if not stats_list:
            return {}
        
        try:
            # Simple efficiency analysis based on collection frequency vs threshold
            latest_stats = stats_list[-1]
            threshold = latest_stats.get('threshold', [700, 10, 10])
            count = latest_stats.get('count', [0, 0, 0])
            
            efficiency = {}
            for i, (thresh, cnt) in enumerate(zip(threshold, count)):
                if thresh > 0:
                    efficiency[f'gen_{i}_efficiency'] = 1.0 - (cnt / thresh)
                else:
                    efficiency[f'gen_{i}_efficiency'] = 1.0
            
            return efficiency
            
        except Exception:
            return {}


class ObjectTracker:
    """Tracks Python object lifecycle and references."""
    
    def __init__(self, logger, max_tracked_objects: int = 10000):
        self.logger = logger
        self.max_tracked_objects = max_tracked_objects
        self.tracked_objects: Dict[int, Dict[str, Any]] = {}
        self.object_types: Dict[str, int] = defaultdict(int)
        self.weak_refs: Dict[int, weakref.ref] = {}
    
    def start_tracking(self):
        """Start tracking object allocations."""
        if OBJGRAPH_AVAILABLE:
            self._track_with_objgraph()
        else:
            self._track_basic()
    
    def _track_with_objgraph(self):
        """Track objects using objgraph if available."""
        try:
            # Get most common types
            common_types = objgraph.most_common_types(limit=50)
            
            for type_name, count in common_types:
                self.object_types[type_name] = count
                
        except Exception as e:
            self.logger.error(f"Failed to track with objgraph: {str(e)}")
            self._track_basic()
    
    def _track_basic(self):
        """Basic object tracking without external libraries."""
        try:
            # Count objects by type
            object_counts = defaultdict(int)
            
            for obj in gc.get_objects():
                obj_type = type(obj).__name__
                object_counts[obj_type] += 1
                
                # Track specific objects if under limit
                if len(self.tracked_objects) < self.max_tracked_objects:
                    obj_id = id(obj)
                    if obj_id not in self.tracked_objects:
                        self.tracked_objects[obj_id] = {
                            'type': obj_type,
                            'size': sys.getsizeof(obj),
                            'created_at': datetime.now(timezone.utc),
                            'ref_count': sys.getrefcount(obj)
                        }
                        
                        # Create weak reference to track lifetime
                        try:
                            self.weak_refs[obj_id] = weakref.ref(obj, lambda ref: self._object_destroyed(obj_id))
                        except TypeError:
                            # Some objects don't support weak references
                            pass
            
            self.object_types.update(object_counts)
            
        except Exception as e:
            self.logger.error(f"Failed to track objects: {str(e)}")
    
    def _object_destroyed(self, obj_id: int):
        """Callback when a tracked object is destroyed."""
        if obj_id in self.tracked_objects:
            obj_info = self.tracked_objects.pop(obj_id)
            obj_info['destroyed_at'] = datetime.now(timezone.utc)
            obj_info['lifetime'] = (obj_info['destroyed_at'] - obj_info['created_at']).total_seconds()
    
    def get_object_statistics(self) -> Dict[str, Any]:
        """Get object tracking statistics."""
        return {
            'tracked_objects_count': len(self.tracked_objects),
            'object_types': dict(self.object_types),
            'total_objects': sum(self.object_types.values()),
            'weak_refs_count': len(self.weak_refs)
        }


class EnhancedMemoryProfiler:
    """
    Advanced Memory Profiler for the AI Assistant.
    
    This profiler provides comprehensive memory monitoring including:
    - Real-time memory usage monitoring
    - Memory leak detection with root cause analysis
    - Memory allocation tracking and optimization
    - Component-specific memory profiling
    - Garbage collection monitoring and analysis
    - Memory usage pattern recognition
    - Integration with core assistant components
    - Memory optimization recommendations
    - Historical memory trend analysis
    """
    
    def __init__(self, container: Container):
        """Initialize the memory profiler."""
        self.container = container
        self.config_loader = container.get(ConfigLoader)
        self.event_bus = container.get(EventBus)
        self.health_check = container.get(HealthCheck)
        self.metrics = container.get(MetricsCollector)
        self.trace_manager = container.get(TraceManager)
        self.logger = get_logger(__name__)
        
        # Configuration
        self.config = MemoryProfilingConfig()
        self._load_configuration()
        
        # Status and control
        self.status = MemoryProfilerStatus.STOPPED
        self.current_session: Optional[MemoryProfilingSession] = None
        self.monitoring_task: Optional[asyncio.Task] = None
        self.snapshot_task: Optional[asyncio.Task] = None
        
        # Core components
        self.leak_detector = MemoryLeakDetector(self.logger, self.config)
        self.gc_monitor = GarbageCollectionMonitor(self.logger)
        self.object_tracker = ObjectTracker(self.logger, self.config.max_tracked_objects)
        
        # Data storage
        self.active_sessions: Dict[str, MemoryProfilingSession] = {}
        self.component_trackers: Dict[str, MemoryTracker] = {}
        self.recent_snapshots = deque(maxlen=self.config.max_snapshots)
        
        # Tracemalloc integration
        self.tracemalloc_enabled = False
        
        # Threading and async support
        self.monitoring_lock = threading.Lock()
        self.stop_monitoring_event = threading.Event()
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        
        # Output paths
        self.output_dir = Path(self.config_loader.get("profiling.output_dir", "data/profiling/memory"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Component integration
        self._setup_monitoring()
        self._setup_event_handlers()
        
        self.logger.info("Enhanced Memory Profiler initialized successfully")
    
    def _load_configuration(self) -> None:
        """Load memory profiling configuration."""
        try:
            # Load from config with defaults
            profiling_config = self.config_loader.get("memory_profiling", {})
            
            # Update config with loaded values
            for key, value in profiling_config.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
            
            self.logger.debug(f"Loaded memory profiling configuration: {self.config}")
            
        except Exception as e:
            self.logger.warning(f"Failed to load configuration, using defaults: {str(e)}")
    
    def _setup_monitoring(self) -> None:
        """Setup metrics and monitoring."""
        try:
            if self.metrics:
                # Register memory profiler metrics
                self.metrics.register_gauge("memory_profiler_memory_usage_mb")
                self.metrics.register_gauge("memory_profiler_peak_memory_mb")
                self.metrics.register_counter("memory_profiler_snapshots_total")
                self.metrics.register_counter("memory_profiler_leaks_detected_total")
                self.metrics.register_histogram("memory_profiler_gc_duration_seconds")
                self.metrics.register_gauge("memory_profiler_tracked_objects")
                
        except Exception as e:
            self.logger.warning(f"Failed to setup monitoring: {str(e)}")
    
    def _setup_event_handlers(self) -> None:
        """Setup event handlers for system integration."""
        try:
            if self.event_bus:
                # Subscribe to relevant events
                self.event_bus.subscribe(ComponentHealthChanged, self._handle_component_health_change)
                self.event_bus.subscribe(SystemStateChanged, self._handle_system_state_change)
                
        except Exception as e:
            self.logger.warning(f"Failed to setup event handlers: {str(e)}")
    
    async def _handle_component_health_change(self, event: ComponentHealthChanged) -> None:
        """Handle component health change events."""
        if self.is_profiling():
            # Take a snapshot when component health changes
            await self._take_memory_snapshot(f"component_health_change_{event.component_id}")
    
    async def _handle_system_state_change(self, event: SystemStateChanged) -> None:
        """Handle system state change events."""
        if self.is_profiling():
            # Take a snapshot when system state changes
            await self._take_memory_snapshot(f"system_state_change_{event.new_state}")
    
    @handle_exceptions
    async def start_profiling(
        self,
        session_name: str = None,
        description: str = None,
        mode: MemoryProfilingMode = None,
        level: MemoryProfilingLevel = None
    ) -> str:
        """
        Start memory profiling session.
        
        Args:
            session_name: Name for the profiling session
            description: Description of what is being profiled
            mode: Profiling mode to use
            level: Profiling detail level
            
        Returns:
            Session ID for tracking
        """
        if self.status != MemoryProfilerStatus.STOPPED:
            raise RuntimeError(f"Profiler already running with status: {self.status}")
        
        self.status = MemoryProfilerStatus.STARTING
        
        try:
            # Create new session
            session = MemoryProfilingSession(
                session_name=session_name or f"memory_session_{int(time.time())}",
                description=description or "Memory profiling session"
            )
            
            # Update configuration if provided
            if mode:
                session.config.mode = mode
            if level:
                session.config.level = level
            
            # Enable tracemalloc if configured
            if session.config.track_allocations and not self.tracemalloc_enabled:
                try:
                    tracemalloc.start(session.config.max_stack_depth)
                    self.tracemalloc_enabled = True
                    self.logger.debug("Tracemalloc enabled for memory profiling")
                except Exception as e:
                    self.logger.warning(f"Failed to start tracemalloc: {str(e)}")
            
            # Set current session
            self.current_session = session
            self.active_sessions[session.session_id] = session
            
            # Start monitoring tasks
            await self._start_monitoring_tasks()
            
            # Start object tracking
            if session.config.enable_object_tracking:
                self.object_tracker.start_tracking()
            
            # Take initial snapshot
            await self._take_memory_snapshot("session_start")
            
            self.status = MemoryProfilerStatus.RUNNING
            
            # Emit profiling started event
            if self.event_bus and session.config.emit_events:
                await self.event_bus.emit(MemoryOperationStarted(
                    operation_type="profiling_started",
                    session_id=session.session_id
                ))
            
            self.logger.info(f"Memory profiling started: session {session.session_id}")
            
            return session.session_id
            
        except Exception as e:
            self.status = MemoryProfilerStatus.ERROR
            self.logger.error(f"Failed to start memory profiling: {str(e)}")
            raise
    
    async def stop_profiling(self, session_id: str = None) -> Optional[MemoryProfilingSession]:
        """
        Stop memory profiling session.
        
        Args:
            session_id: ID of session to stop (current session if None)
            
        Returns:
            Completed profiling session
        """
        if self.status not in [MemoryProfilerStatus.RUNNING, MemoryProfilerStatus.PAUSED]:
            return None
        
        self.status = MemoryProfilerStatus.STOPPING
        
        try:
            # Determine session to stop
            session = None
            if session_id:
                session = self.active_sessions.get(session_id)
            else:
                session = self.current_session
            
            if not session:
                self.logger.warning("No active session to stop")
                return None
            
            # Take final snapshot
            await self._take_memory_snapshot("session_end")
            
            # Stop monitoring tasks
            await self._stop_monitoring_tasks()
            
            # Complete session
            session.end_time = datetime.now(timezone.utc)
            session.duration = session.end_time - session.start_time
            
            # Perform final analysis
            await self._analyze_session(session)
            
            # Generate outputs
            await self._generate_session_outputs(session)
            
            # Cleanup
            if session_id and session_id in self.active_sessions:
                del self.active_sessions[session_id]
            
            if self.current_session and self.current_session.session_id == session.session_id:
                self.current_session = None
            
            # Disable tracemalloc if enabled
            if self.tracemalloc_enabled:
                try:
                    tracemalloc.stop()
                    self.tracemalloc_enabled = False
                    self.logger.debug("Tracemalloc disabled")
                except Exception as e:
                    self.logger.warning(f"Failed to stop tracemalloc: {str(e)}")
            
            self.status = MemoryProfilerStatus.STOPPED
            
            # Emit profiling completed event
            if self.event_bus and session.config.emit_events:
                await self.event_bus.emit(MemoryOperationCompleted(
                    operation_type="profiling_completed",
                    session_id=session.session_id
                ))
            
            self.logger.info(f"Memory profiling stopped: session {session.session_id}")
            
            return session
            
        except Exception as e:
            self.status = MemoryProfilerStatus.ERROR
            self.logger.error(f"Failed to stop memory profiling: {str(e)}")
            raise
    
    async def _start_monitoring_tasks(self) -> None:
        """Start background monitoring tasks."""
        try:
            # Start real-time monitoring
            if self.config.monitoring_interval > 0:
                self.monitoring_task = asyncio.create_task(self._monitoring_loop())
            
            # Start snapshot collection
            if self.config.snapshot_interval > 0:
                self.snapshot_task = asyncio.create_task(self._snapshot_loop())
            
        except Exception as e:
            self.logger.error(f"Failed to start monitoring tasks: {str(e)}")
    
    async def _stop_monitoring_tasks(self) -> None:
        """Stop background monitoring tasks."""
        try:
            # Stop monitoring task
            if self.monitoring_task and not self.monitoring_task.done():
                self.monitoring_task.cancel()
                try:
                    await self.monitoring_task
                except asyncio.CancelledError:
                    pass
            
            # Stop snapshot task
            if self.snapshot_task and not self.snapshot_task.done():
                self.snapshot_task.cancel()
                try:
                    await self.snapshot_task
                except asyncio.CancelledError:
                    pass
            
        except Exception as e:
            self.logger.error(f"Failed to stop monitoring tasks: {str(e)}")
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop for real-time memory tracking."""
        while self.status == MemoryProfilerStatus.RUNNING:
            try:
                # Collect real-time memory metrics
                await self._collect_realtime_metrics()
                
                # Check for memory threshold violations
                await self._check_memory_thresholds()
                
                # Monitor garbage collection
                if self.config.enable_gc_monitoring:
                    await self._monitor_garbage_collection()
                
                # Sleep until next monitoring cycle
                await asyncio.sleep(self.config.monitoring_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {str(e)}")
                await asyncio.sleep(self.config.monitoring_interval)
    
    async def _snapshot_loop(self) -> None:
        """Loop for taking periodic memory snapshots."""
        while self.status == MemoryProfilerStatus.RUNNING:
            try:
                # Take memory snapshot
                await self._take_memory_snapshot("periodic")
                
                # Sleep until next snapshot
                await asyncio.sleep(self.config.snapshot_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in snapshot loop: {str(e)}")
                await asyncio.sleep(self.config.snapshot_interval)
    
    async def _collect_realtime_metrics(self) -> None:
        """Collect real-time memory metrics."""
        try:
            # Get process memory info
            process = psutil.Process()
            memory_info = process.memory_info()
            
            # Update metrics
            if self.metrics:
                self.metrics.set_gauge("memory_profiler_memory_usage_mb", memory_info.rss / (1024 * 1024))
                
                # Track peak memory if available
                peak_memory = getattr(memory_info, 'peak_wset', memory_info.rss) if hasattr(memory_info, 'peak_wset') else memory_info.rss
                self.metrics.set_gauge("memory_profiler_peak_memory_mb", peak_memory / (1024 * 1024))
                
                # Track object count
                if self.config.enable_object_tracking:
                    object_stats = self.object_tracker.get_object_statistics()
                    self.metrics.set_gauge("memory_profiler_tracked_objects", object_stats.get('tracked_objects_count', 0))
            
        except Exception as e:
            self.logger.error(f"Failed to collect real-time metrics: {str(e)}")
    
    async def _check_memory_thresholds(self) -> None:
        """Check for memory threshold violations."""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / (1024 * 1024)
            
            # Check memory threshold
            if memory_mb > self.config.memory_threshold_mb:
                if self.event_bus and self.config.emit_events:
                    await self.event_bus.emit(PerformanceThresholdExceeded(
                        metric_name="memory_usage",
                        current_value=memory_mb,
                        threshold_value=self.config.memory_threshold_mb,
                        component_id="memory_profiler",
                        session_id=self.current_session.session_id if self.current_session else ""
                    ))
            
            # Check for rapid memory growth
            if len(self.recent_snapshots) >= 2:
                recent_growth = self._calculate_memory_growth_rate()
                if recent_growth > self.config.growth_rate_threshold_mb_per_min:
                    if self.event_bus and self.config.emit_events:
                        await self.event_bus.emit(PerformanceThresholdExceeded(
                            metric_name="memory_growth_rate",
                            current_value=recent_growth,
                            threshold_value=self.config.growth_rate_threshold_mb_per_min,
                            component_id="memory_profiler",
                            session_id=self.current_session.session_id if self.current_session else ""
                        ))
            
        except Exception as e:
            self.logger.error(f"Failed to check memory thresholds: {str(e)}")
    
    def _calculate_memory_growth_rate(self) -> float:
        """Calculate current memory growth rate in MB/min."""
        if len(self.recent_snapshots) < 2:
            return 0.0
        
        try:
            recent_snapshots = list(self.recent_snapshots)[-5:]  # Last 5 snapshots
            if len(recent_snapshots) < 2:
                return 0.0
            
            first_snapshot = recent_snapshots[0]
            last_snapshot = recent_snapshots[-1]
            
            first_memory = first_snapshot.process_memory.get('rss', 0)
            last_memory = last_snapshot.process_memory.get('rss', 0)
            
            time_diff = (last_snapshot.timestamp - first_snapshot.timestamp).total_seconds() / 60
            
            if time_diff > 0:
                memory_diff_mb = (last_memory - first_memory) / (1024 * 1024)
                return memory_diff_mb / time_diff
            
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Failed to calculate growth rate: {str(e)}")
            return 0.0
    
    async def _monitor_garbage_collection(self) -> None:
        """Monitor garbage collection behavior."""
        try:
            gc_stats = self.gc_monitor.collect_gc_stats()
            
            if gc_stats and self.metrics:
                # Track GC performance metrics
                for i, gen_stats in enumerate(gc_stats.get('collections', [])):
                    if isinstance(gen_stats, dict) and 'collections' in gen_stats:
                        self.metrics.increment(f"memory_profiler_gc_collections_gen_{i}_total", 
                                             gen_stats['collections'])
            
        except Exception as e:
            self.logger.error(f"Failed to monitor garbage collection: {str(e)}")
    
    async def _take_memory_snapshot(self, snapshot_type: str = "manual") -> MemorySnapshot:
        """Take a comprehensive memory snapshot."""
        try:
            snapshot = MemorySnapshot()
            
            # Collect process memory information
            process = psutil.Process()
            memory_info = process.memory_info()
            
            snapshot.process_memory = {
                'rss': memory_info.rss,
                'vms': memory_info.vms,
                'percent': process.memory_percent(),
                'available': psutil.virtual_memory().available,
                'used': psutil.virtual_memory().used
            }
            
            # Add platform-specific memory info
            if hasattr(memory_info, 'uss'):
                snapshot.process_memory['uss'] = memory_info.uss
            if hasattr(memory_info, 'pss'):
                snapshot.process_memory['pss'] = memory_info.pss
            if hasattr(memory_info, 'peak_wset'):
                snapshot.process_memory['peak'] = memory_info.peak_wset
            
            # Collect system memory information
            virtual_memory = psutil.virtual_memory()
            snapshot.system_memory = {
                'total': virtual_memory.total,
                'available': virtual_memory.available,
                'percent': virtual_memory.percent,
                'used': virtual_memory.used,
                'free': virtual_memory.free
            }
            
            # Collect Python heap information
            snapshot.heap_size = sum(sys.getsizeof(obj) for obj in gc.get_objects())
            snapshot.object_count = len(gc.get_objects())
            
            # Collect garbage collection stats
            snapshot.gc_stats = self.gc_monitor.collect_gc_stats()
            
            # Collect component memory usage
            if self.current_session:
                for component_id, tracker in self.component_trackers.items():
                    if tracker.active:
                        current_memory = tracker._get_current_memory()
                        snapshot.component_memory[component_id] = current_memory.get('rss', 0) / (1024 * 1024)
            
            # Collect tracemalloc data if available
            if self.tracemalloc_enabled:
                try:
                    current, peak = tracemalloc.get_traced_memory()
                    snapshot.tracemalloc_stats = {
                        'current': current,
                        'peak': peak,
                        'top_stats': [
                            {
                                'filename': stat.traceback.format()[0] if stat.traceback else 'unknown',
                                'size': stat.size,
                                'count': stat.count
                            }
                            for stat in tracemalloc.take_snapshot().statistics('lineno')[:10]
                        ]
                    }
                except Exception as e:
                    self.logger.warning(f"Failed to collect tracemalloc data: {str(e)}")
            
            # Calculate memory patterns
            if len(self.recent_snapshots) > 0:
                snapshot.growth_rate = self._calculate_snapshot_growth_rate(snapshot)
                snapshot.fragmentation_ratio = self._calculate_fragmentation_ratio(snapshot)
            
            # Add to collections
            self.recent_snapshots.append(snapshot)
            if self.current_session:
                self.current_session.snapshots.append(snapshot)
            
            # Detect memory leaks
            if self.config.enable_leak_detection:
                detected_leaks = self.leak_detector.add_snapshot(snapshot)
                if detected_leaks and self.current_session:
                    self.current_session.detected_leaks.extend(detected_leaks)
                    
                    # Update metrics
                    if self.metrics:
                        self.metrics.increment("memory_profiler_leaks_detected_total", len(detected_leaks))
            
            # Update metrics
            if self.metrics:
                self.metrics.increment("memory_profiler_snapshots_total")
            
            self.logger.debug(f"Memory snapshot taken: {snapshot_type}")
            
            return snapshot
            
        except Exception as e:
            self.logger.error(f"Failed to take memory snapshot: {str(e)}")
            # Return empty snapshot to avoid breaking the flow
            return MemorySnapshot()
    
    def _calculate_snapshot_growth_rate(self, current_snapshot: MemorySnapshot) -> float:
        """Calculate memory growth rate for a snapshot."""
        if not self.recent_snapshots:
            return 0.0
        
        try:
            previous_snapshot = self.recent_snapshots[-1]
            time_diff = (current_snapshot.timestamp - previous_snapshot.timestamp).total_seconds() / 60
            
            if time_diff <= 0:
                return 0.0
            
            current_memory = current_snapshot.process_memory.get('rss', 0)
            previous_memory = previous_snapshot.process_memory.get('rss', 0)
            
            memory_diff_mb = (current_memory - previous_memory) / (1024 * 1024)
            return memory_diff_mb / time_diff
            
        except Exception:
            return 0.0
    
    def _calculate_fragmentation_ratio(self, snapshot: MemorySnapshot) -> float:
        """Calculate memory fragmentation ratio."""
        try:
            vms = snapshot.process_memory.get('vms', 0)
            rss = snapshot.process_memory.get('rss', 0)
            
            if vms > 0 and rss > 0:
                return (vms - rss) / vms
            
            return 0.0
            
        except Exception:
            return 0.0
    
    async def _analyze_session(self, session: MemoryProfilingSession) -> None:
        """Perform comprehensive analysis of a profiling session."""
        try:
            if not session.snapshots:
                return
            
            # Calculate session statistics
            memory_values = [s.process_memory.get('rss', 0) for s in session.snapshots]
            
            if memory_values:
                baseline_memory = memory_values[0] / (1024 * 1024)
                peak_memory = max(memory_values) / (1024 * 1024)
                final_memory = memory_values[-1] / (1024 * 1024)
                avg_memory = sum(memory_values) / len(memory_values) / (1024 * 1024)
                
                # Calculate efficiency score
                if peak_memory > 0:
                    session.memory_efficiency = avg_memory / peak_memory
                
                # Calculate leak risk score
                session.leak_risk_score = self._calculate_leak_risk(session)
                
                # Generate optimization opportunities
                session.optimization_opportunities = self._generate_optimization_recommendations(session)
            
            # Analyze component memory profiles
            await self._analyze_component_profiles(session)
            
            self.logger.debug(f"Session analysis completed for {session.session_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to analyze session: {str(e)}")
    
    def _calculate_leak_risk(self, session: MemoryProfilingSession) -> float:
        """Calculate memory leak risk score (0.0 to 1.0)."""
        try:
            if not session.snapshots or len(session.snapshots) < 2:
                return 0.0
            
            # Factors contributing to leak risk
            risk_factors = []
            
            # Growth rate factor
            memory_values = [s.process_memory.get('rss', 0) for s in session.snapshots]
            if len(memory_values) >= 2:
                total_growth = (memory_values[-1] - memory_values[0]) / (1024 * 1024)
                session_duration = session.duration.total_seconds() / 60 if session.duration else 1
                growth_rate = total_growth / session_duration if session_duration > 0 else 0
                
                # Normalize growth rate (>10 MB/min = high risk)
                growth_risk = min(growth_rate / 10.0, 1.0) if growth_rate > 0 else 0.0
                risk_factors.append(growth_risk)
            
            # Leak detection factor
            leak_count = len(session.detected_leaks)
            leak_risk = min(leak_count / 5.0, 1.0)  # >5 leaks = high risk
            risk_factors.append(leak_risk)
            
            # GC pressure factor
            if session.snapshots:
                recent_gc_stats = session.snapshots[-1].gc_stats
                if recent_gc_stats and 'collections' in recent_gc_stats:
                    total_collections = sum(
                        gen.get('collections', 0) if isinstance(gen, dict) else 0
                        for gen in recent_gc_stats['collections']
                    )
                    # Normalize collections (>100 = high pressure)
                    gc_risk = min(total_collections / 100.0, 1.0)
                    risk_factors.append(gc_risk)
            
            # Calculate overall risk as weighted average
            if risk_factors:
                return sum(risk_factors) / len(risk_factors)
            
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Failed to calculate leak risk: {str(e)}")
            return 0.0
    
    def _generate_optimization_recommendations(self, session: MemoryProfilingSession) -> List[str]:
        """Generate memory optimization recommendations."""
        recommendations = []
        
        try:
            if not session.snapshots:
                return recommendations
            
            # Analyze memory patterns
            memory_values = [s.process_memory.get('rss', 0) for s in session.snapshots]
            if len(memory_values) >= 2:
                total_growth = (memory_values[-1] - memory_values[0]) / (1024 * 1024)
                peak_memory = max(memory_values) / (1024 * 1024)
                
                # High memory usage recommendations
                if peak_memory > self.config.memory_threshold_mb:
                    recommendations.append(f"Peak memory usage ({peak_memory:.1f} MB) exceeds threshold")
                    recommendations.append("Consider implementing memory pooling or caching strategies")
                
                # Memory growth recommendations
                if total_growth > self.config.leak_detection_threshold_mb:
                    recommendations.append(f"Memory grew by {total_growth:.1f} MB during session")
                    recommendations.append("Review object lifecycle management and garbage collection")
                
                # Memory efficiency recommendations
                if session.memory_efficiency < 0.7:
                    recommendations.append(f"Memory efficiency is low ({session.memory_efficiency:.2f})")
                    recommendations.append("Consider optimizing memory allocation patterns")
            
            # Leak-specific recommendations
            if session.detected_leaks:
                high_severity_leaks = [leak for leak in session.detected_leaks if leak.severity in ['high', 'critical']]
                if high_severity_leaks:
                    recommendations.append(f"Found {len(high_severity_leaks)} high-severity memory leaks")
                    recommendations.append("Immediate attention required to prevent system instability")
                
                # Add specific leak recommendations
                for leak in session.detected_leaks[:3]:  # Top 3 leaks
                    recommendations.extend(leak.recommendations)
            
            # GC-related recommendations
            if session.snapshots:
                recent_gc_stats = session.snapshots[-1].gc_stats
                if recent_gc_stats:
                    analysis = self.gc_monitor.analyze_gc_performance()
                    if analysis.get('gc_pressure', 0) > 0.8:
                        recommendations.append("High garbage collection pressure detected")
                        recommendations.append("Consider optimizing object creation and cleanup patterns")
            
            # Component-specific recommendations
            for component_id, profile in session.component_profiles.items():
                if profile.memory_pattern == "leaking":
                    recommendations.append(f"Component {component_id} shows memory leak pattern")
                elif profile.efficiency_score < 0.6:
                    recommendations.append(f"Component {component_id} has low memory efficiency")
            
        except Exception as e:
            self.logger.error(f"Failed to generate recommendations: {str(e)}")
        
        return recommendations[:10]  # Limit to top 10 recommendations
    
    async def _analyze_component_profiles(self, session: MemoryProfilingSession) -> None:
        """Analyze memory profiles for individual components."""
        try:
            for component_id, tracker in self.component_trackers.items():
                if component_id not in session.component_profiles:
                    continue
                
                profile = session.component_profiles[component_id]
                
                # Analyze memory pattern
                if profile.memory_history:
                    memory_values = list(profile.memory_history)
                    
                    # Determine pattern
                    if len(memory_values) >= 3:
                        increasing_trend = sum(1 for i in range(1, len(memory_values)) 
                                             if memory_values[i] > memory_values[i-1])
                        trend_ratio = increasing_trend / (len(memory_values) - 1)
                        
                        if trend_ratio > 0.8:
                            profile.memory_pattern = "growing"
                        elif trend_ratio > 0.6:
                            profile.memory_pattern = "leaking"
                        elif trend_ratio < 0.2:
                            profile.memory_pattern = "stable"
                        else:
                            profile.memory_pattern = "oscillating"
                    
                    # Calculate efficiency score
                    if profile.peak_memory_mb > 0:
                        profile.efficiency_score = profile.average_memory_mb / profile.peak_memory_mb
                
                profile.last_updated = datetime.now(timezone.utc)
            
        except Exception as e:
            self.logger.error(f"Failed to analyze component profiles: {str(e)}")
    
    async def _generate_session_outputs(self, session: MemoryProfilingSession) -> None:
        """Generate output files for the profiling session."""
        try:
            session_dir = self.output_dir / session.session_id
            session_dir.mkdir(parents=True, exist_ok=True)
            
            # Save snapshot data
            snapshot_path = session_dir / "snapshots.json"
            await self._save_snapshot_data(session, snapshot_path)
            session.snapshot_file_path = snapshot_path
            
            # Generate analysis report
            report_path = session_dir / "memory_analysis_report.html"
            await self._generate_analysis_report(session, report_path)
            session.report_path = report_path
            
            # Save detailed analysis
            analysis_path = session_dir / "detailed_analysis.json"
            await self._save_detailed_analysis(session, analysis_path)
            session.analysis_path = analysis_path
            
        except Exception as e:
            self.logger.error(f"Failed to generate session outputs: {str(e)}")
    
    async def _save_snapshot_data(self, session: MemoryProfilingSession, path: Path) -> None:
        """Save snapshot data to JSON file."""
        try:
            snapshot_data = {
                'session_info': {
                    'session_id': session.session_id,
                    'session_name': session.session_name,
                    'start_time': session.start_time.isoformat(),
                    'end_time': session.end_time.isoformat() if session.end_time else None,
                    'duration_seconds': session.duration.total_seconds() if session.duration else None
                },
                'snapshots': []
            }
            
            # Convert snapshots to serializable format
            for snapshot in session.snapshots:
                snapshot_dict = asdict(snapshot)
                snapshot_dict['timestamp'] = snapshot.timestamp.isoformat()
                snapshot_data['snapshots'].append(snapshot_dict)
            
            # Save to file
            with open(path, 'w') as f:
                json.dump(snapshot_data, f, indent=2)
            
        except Exception as e:
            self.logger.error(f"Failed to save snapshot data: {str(e)}")
    
    async def _generate_analysis_report(self, session: MemoryProfilingSession, path: Path) -> None:
        """Generate HTML analysis report."""
        try:
            html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Memory Analysis Report - {session.session_name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
        .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: #f9f9f9; border-radius: 3px; }}
        .leak {{ background-color: #ffe6e6; padding: 10px; margin: 5px 0; border-radius: 3px; }}
        .recommendation {{ background-color: #e6f3ff; padding: 10px; margin: 5px 0; border-radius: 3px; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Memory Analysis Report</h1>
        <h2>{session.session_name}</h2>
        <p><strong>Session ID:</strong> {session.session_id}</p>
        <p><strong>Duration:</strong> {session.duration.total_seconds() if session.duration else 0:.2f} seconds</p>
        <p><strong>Start Time:</strong> {session.start_time.isoformat()}</p>
        <p><strong>End Time:</strong> {session.end_time.isoformat() if session.end_time else 'N/A'}</p>
    </div>
    
    <div class="section">
        <h3>Summary Metrics</h3>
        <div class="metric">
            <strong>Memory Efficiency:</strong> {session.memory_efficiency:.2f}
        </div>
        <div class="metric">
            <strong>Leak Risk Score:</strong> {session.leak_risk_score:.2f}
        </div>
        <div class="metric">
            <strong>Snapshots Collected:</strong> {len(session.snapshots)}
        </div>
        <div class="metric">
            <strong>Leaks Detected:</strong> {len(session.detected_leaks)}
        </div>
    </div>
"""
            
            # Add memory snapshots chart placeholder
            if session.snapshots:
                html_content += """
    <div class="section">
        <h3>Memory Usage Timeline</h3>
        <p><em>Memory usage over time (data available in snapshots.json for visualization)</em></p>
        <table>
            <tr>
                <th>Timestamp</th>
                <th>RSS (MB)</th>
                <th>VMS (MB)</th>
                <th>Memory %</th>
                <th>Objects</th>
            </tr>
"""
                for snapshot in session.snapshots[-10:]:  # Last 10 snapshots
                    rss_mb = snapshot.process_memory.get('rss', 0) / (1024 * 1024)
                    vms_mb = snapshot.process_memory.get('vms', 0) / (1024 * 1024)
                    mem_percent = snapshot.process_memory.get('percent', 0)
                    
                    html_content += f"""
            <tr>
                <td>{snapshot.timestamp.strftime('%H:%M:%S')}</td>
                <td>{rss_mb:.1f}</td>
                <td>{vms_mb:.1f}</td>
                <td>{mem_percent:.1f}%</td>
                <td>{snapshot.object_count}</td>
            </tr>
"""
                html_content += """
        </table>
    </div>
"""
            
            # Add detected leaks
            if session.detected_leaks:
                html_content += """
    <div class="section">
        <h3>Detected Memory Leaks</h3>
"""
                for leak in session.detected_leaks:
                    html_content += f"""
        <div class="leak">
            <h4>Leak: {leak.leak_type} in {leak.component}</h4>
            <p><strong>Severity:</strong> {leak.severity}</p>
            <p><strong>Memory Growth:</strong> {leak.memory_growth_mb:.2f} MB</p>
            <p><strong>Growth Rate:</strong> {leak.growth_rate_mb_per_min:.2f} MB/min</p>
            <p><strong>Duration:</strong> {leak.duration_minutes:.1f} minutes</p>
            <p><strong>Recommendations:</strong></p>
            <ul>
                {''.join(f'<li>{rec}</li>' for rec in leak.recommendations)}
            </ul>
        </div>
"""
                html_content += """
    </div>
"""
            
            # Add optimization recommendations
            if session.optimization_opportunities:
                html_content += """
    <div class="section">
        <h3>Optimization Recommendations</h3>
"""
                for recommendation in session.optimization_opportunities:
                    html_content += f"""
        <div class="recommendation">
            {recommendation}
        </div>
"""
                html_content += """
    </div>
"""
            
            # Add component profiles
            if session.component_profiles:
                html_content += """
    <div class="section">
        <h3>Component Memory Profiles</h3>
        <table>
            <tr>
                <th>Component</th>
                <th>Current (MB)</th>
                <th>Peak (MB)</th>
                <th>Average (MB)</th>
                <th>Pattern</th>
                <th>Efficiency</th>
            </tr>
"""
                for component_id, profile in session.component_profiles.items():
                    html_content += f"""
            <tr>
                <td>{profile.component_name}</td>
                <td>{profile.current_memory_mb:.1f}</td>
                <td>{profile.peak_memory_mb:.1f}</td>
                <td>{profile.average_memory_mb:.1f}</td>
                <td>{profile.memory_pattern}</td>
                <td>{profile.efficiency_score:.2f}</td>
            </tr>
"""
                html_content += """
        </table>
    </div>
"""
            
            html_content += """
</body>
</html>
"""
            
            # Save to file
            with open(path, 'w') as f:
                f.write(html_content)
            
        except Exception as e:
            self.logger.error(f"Failed to generate analysis report: {str(e)}")
    
    async def _save_detailed_analysis(self, session: MemoryProfilingSession, path: Path) -> None:
        """Save detailed analysis data to JSON file."""
        try:
            analysis_data = {
                'session_summary': {
                    'session_id': session.session_id,
                    'memory_efficiency': session.memory_efficiency,
                    'leak_risk_score': session.leak_risk_score,
                    'optimization_opportunities': session.optimization_opportunities
                },
                'detected_leaks': [],
                'component_profiles': {},
                'statistics': {}
            }
            
            # Add leak details
            for leak in session.detected_leaks:
                leak_dict = asdict(leak)
                leak_dict['detected_at'] = leak.detected_at.isoformat()
                leak_dict['first_detected'] = leak.first_detected.isoformat()
                leak_dict['last_updated'] = leak.last_updated.isoformat()
                analysis_data['detected_leaks'].append(leak_dict)
            
            # Add component profiles
            for component_id, profile in session.component_profiles.items():
                profile_dict = asdict(profile)
                profile_dict['first_seen'] = profile.first_seen.isoformat()
                profile_dict['last_updated'] = profile.last_updated.isoformat()
                # Convert deque to list for JSON serialization
                profile_dict['memory_history'] = list(profile.memory_history)
                analysis_data['component_profiles'][component_id] = profile_dict
            
            # Add session statistics
            if session.snapshots:
                memory_values = [s.process_memory.get('rss', 0) for s in session.snapshots]
                analysis_data['statistics'] = {
                    'min_memory_mb': min(memory_values) / (1024 * 1024),
                    'max_memory_mb': max(memory_values) / (1024 * 1024),
                    'avg_memory_mb': sum(memory_values) / len(memory_values) / (1024 * 1024),
                    'total_snapshots': len(session.snapshots),
                    'memory_variance': self._calculate_memory_variance(memory_values)
                }
            
            # Save to file
            with open(path, 'w') as f:
                json.dump(analysis_data, f, indent=2)
            
        except Exception as e:
            self.logger.error(f"Failed to save detailed analysis: {str(e)}")
    
    def _calculate_memory_variance(self, memory_values: List[float]) -> float:
        """Calculate variance in memory usage."""
        if len(memory_values) < 2:
            return 0.0
        
        mean = sum(memory_values) / len(memory_values)
        variance = sum((x - mean) ** 2 for x in memory_values) / len(memory_values)
        return variance / (1024 * 1024 * 1024)  # Convert to MB²
    
    def is_profiling(self) -> bool:
        """Check if profiler is currently running."""
        return self.status == MemoryProfilerStatus.RUNNING
    
    def get_current_session(self) -> Optional[MemoryProfilingSession]:
        """Get the current profiling session."""
        return self.current_session
    
    def get_session(self, session_id: str) -> Optional[MemoryProfilingSession]:
        """Get a specific profiling session."""
        return self.active_sessions.get(session_id)
    
    def list_sessions(self) -> List[MemoryProfilingSession]:
        """List all active profiling sessions."""
        return list(self.active_sessions.values())
    
    @contextmanager
    def track_component_memory(self, component_name: str):
        """Context manager for tracking component memory usage."""
        tracker = MemoryTracker(component_name, self)
        self.component_trackers[component_name] = tracker
        
        try:
            tracker.start_tracking()
            yield tracker
        finally:
            usage_stats = tracker.stop_tracking()
            
            # Update component profile
            if self.current_session:
                if component_name not in self.current_session.component_profiles:
                    self.current_session.component_profiles[component_name] = ComponentMemoryProfile(
                        component_id=component_name,
                        component_name=component_name
                    )
                
                profile = self.current_session.component_profiles[component_name]
                profile.current_memory_mb = usage_stats.get('end_memory_mb', 0)
                profile.peak_memory_mb = max(profile.peak_memory_mb, usage_stats.get('peak_memory_mb', 0))
                profile.memory_history.append(usage_stats.get('end_memory_mb', 0))
                
                # Update average
                if profile.memory_history:
                    profile.average_memory_mb = sum(profile.memory_history) / len(profile.memory_history)
    
    def _record_component_memory(self, component_name: str, usage_stats: Dict[str, float]) -> None:
        """Record memory usage for a component."""
        try:
            if self.current_session:
                if component_name not in self.current_session.component_profiles:
                    self.current_session.component_profiles[component_name] = ComponentMemoryProfile(
                        component_id=component_name,
                        component_name=component_name
                    )
                
                profile = self.current_session.component_profiles[component_name]
                profile.last_updated = datetime.now(timezone.utc)
                
                # Update allocation counts
                profile.total_allocations += 1
                
                # Update memory stats
                memory_used = usage_stats.get('memory_used_mb', 0)
                if memory_used > 0:
                    profile.current_memory_mb = usage_stats.get('end_memory_mb', 0)
                    profile.peak_memory_mb = max(profile.peak_memory_mb, usage_stats.get('peak_memory_mb', 0))
                    profile.memory_history.append(profile.current_memory_mb)
                    
                    # Update average
                    if profile.memory_history:
                        profile.average_memory_mb = sum(profile.memory_history) / len(profile.memory_history)
            
        except Exception as e:
            self.logger.error(f"Failed to record component memory: {str(e)}")
    
    async def get_memory_snapshot(self) -> MemorySnapshot:
        """Get current memory snapshot."""
        return await self._take_memory_snapshot("manual")
    
    async def detect_memory_leaks(self) -> List[MemoryLeak]:
        """Manually trigger memory leak detection."""
        if not self.recent_snapshots:
            await self._take_memory_snapshot("leak_detection")
        
        if self.recent_snapshots:
            return self.leak_detector.add_snapshot(self.recent_snapshots[-1])
        
        return []
    
    def get_memory_statistics(self) -> Dict[str, Any]:
        """Get current memory statistics."""
        try:
            stats = {
                'status': self.status.value,
                'current_session': self.current_session.session_id if self.current_session else None,
                'active_sessions': len(self.active_sessions),
                'snapshots_collected': len(self.recent_snapshots),
                'tracemalloc_enabled': self.tracemalloc_enabled
            }
            
            # Add current memory info
            try:
                process = psutil.Process()
                memory_info = process.memory_info()
                stats['current_memory'] = {
                    'rss_mb': memory_info.rss / (1024 * 1024),
                    'vms_mb': memory_info.vms / (1024 * 1024),
                    'percent': process.memory_percent()
                }
            except Exception:
                pass
            
            # Add object tracking stats
            if self.config.enable_object_tracking:
                stats['object_tracking'] = self.object_tracker.get_object_statistics()
            
            # Add GC stats
            if self.config.enable_gc_monitoring:
                stats['gc_analysis'] = self.gc_monitor.analyze_gc_performance()
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to get memory statistics: {str(e)}")
            return {'status': self.status.value, 'error': str(e)}
    
    async def cleanup_old_sessions(self, retention_hours: int = None) -> int:
        """Clean up old profiling sessions."""
        retention_hours = retention_hours or self.config.snapshot_retention_hours
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=retention_hours)
        
        cleaned_count = 0
        sessions_to_remove = []
        
        try:
            for session_id, session in self.active_sessions.items():
                if session.end_time and session.end_time < cutoff_time:
                    sessions_to_remove.append(session_id)
            
            for session_id in sessions_to_remove:
                del self.active_sessions[session_id]
                cleaned_count += 1
            
            # Clean up old snapshots
            snapshots_to_remove = []
            for i, snapshot in enumerate(self.recent_snapshots):
                if snapshot.timestamp < cutoff_time:
                    snapshots_to_remove.append(i)
            
            # Remove from the end to maintain indices
            for i in reversed(snapshots_to_remove):
                del self.recent_snapshots[i]
                cleaned_count += 1
            
            self.logger.info(f"Cleaned up {cleaned_count} old profiling items")
            
            return cleaned_count
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup old sessions: {str(e)}")
            return 0
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check for the memory profiler."""
        try:
            health = {
                'status': 'healthy',
                'profiler_status': self.status.value,
                'active_sessions': len(self.active_sessions),
                'recent_snapshots': len(self.recent_snapshots),
                'tracemalloc_available': tracemalloc.is_tracing() if hasattr(tracemalloc, 'is_tracing') else self.tracemalloc_enabled,
                'pympler_available': PYMPLER_AVAILABLE,
                'objgraph_available': OBJGRAPH_AVAILABLE
            }
            
            # Check for issues
            issues = []
            
            if self.status == MemoryProfilerStatus.ERROR:
                issues.append("Profiler is in error state")
                health['status'] = 'unhealthy'
            
            if self.config.enable_leak_detection and not self.recent_snapshots:
                issues.append("Leak detection enabled but no snapshots available")
                health['status'] = 'degraded'
            
            if self.config.track_allocations and not self.tracemalloc_enabled:
                issues.append("Allocation tracking configured but tracemalloc not enabled")
                health['status'] = 'degraded'
            
            health['issues'] = issues
            
            return health
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e)
            }


# Decorator for memory profiling
def profile_memory(component_name: str = None):
    """Decorator for automatic memory profiling of functions."""
    def decorator(func: F) -> F:
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                # Try to get profiler from container (simplified approach)
                try:
                    from src.core.dependency_injection import Container
                    container = Container()  # This is simplified - in real use, get from context
                    profiler = container.get(EnhancedMemoryProfiler)
                    
                    if profiler and profiler.is_profiling():
                        with profiler.track_component_memory(component_name or func.__name__):
                            return await func(*args, **kwargs)
                    else:
                        return await func(*args, **kwargs)
                        
                except Exception:
                    # Fallback to normal execution if profiler not available
                    return await func(*args, **kwargs)
            
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                # Try to get profiler from container (simplified approach)
                try:
                    from src.core.dependency_injection import Container
                    container = Container()  # This is simplified - in real use, get from context
                    profiler = container.get(EnhancedMemoryProfiler)
                    
                    if profiler and profiler.is_profiling():
                        with profiler.track_component_memory(component_name or func.__name__):
                            return func(*args, **kwargs)
                    else:
                        return func(*args, **kwargs)
                        
                except Exception:
                    # Fallback to normal execution if profiler not available
                    return func(*args, **kwargs)
            
            return sync_wrapper
    
    return decorator