"""
Advanced Memory Profiler for AI Assistant
Author: Drmusab
Last Modified: 2025-06-27 05:48:00 UTC

This module provides comprehensive memory profiling capabilities for the AI assistant,
including real-time monitoring, memory analysis, leak detection, and integration with
all core system components for memory optimization and insights.
"""

import gc
import os
import sys
import psutil
import threading
import time
import asyncio
import functools
import tracemalloc
import weakref
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable, TypeVar, Union, AsyncGenerator, Set, Tuple
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
from contextlib import contextmanager, asynccontextmanager
from collections import defaultdict, deque
import json
import uuid
import pickle
import gzip
import logging
import concurrent.futures
import statistics

# Core imports - with error handling for missing modules
ConfigLoader = None
EventBus = None
ErrorHandler = None
Container = None
HealthCheck = None
MetricsCollector = None
TraceManager = None

def handle_exceptions(func):
    """Placeholder error handler decorator."""
    return func

def get_logger(name):
    """Get logger instance."""
    import logging
    return logging.getLogger(name)

# Create placeholder event classes
class BaseEvent:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

PerformanceAlertTriggered = BaseEvent
ProfilerStarted = BaseEvent
ProfilerStopped = BaseEvent
ProfilingDataGenerated = BaseEvent
PerformanceThresholdExceeded = BaseEvent
ComponentHealthChanged = BaseEvent
SystemStateChanged = BaseEvent
MemoryOperationStarted = BaseEvent
MemoryOperationCompleted = BaseEvent

# Type definitions
F = TypeVar('F', bound=Callable)


class MemoryProfilingMode(Enum):
    """Memory profiling modes."""
    OFF = "off"
    LIGHTWEIGHT = "lightweight"      # Basic memory monitoring
    STANDARD = "standard"           # Standard memory profiling
    DETAILED = "detailed"           # Detailed memory analysis
    LEAK_DETECTION = "leak_detection"  # Focus on memory leaks
    OPTIMIZATION = "optimization"    # Memory optimization mode
    EMERGENCY = "emergency"         # Emergency profiling for memory issues


class MemoryProfilingLevel(Enum):
    """Memory profiling detail levels."""
    BASIC = "basic"           # Basic memory usage tracking
    COMPONENT = "component"   # Component-level memory tracking
    DETAILED = "detailed"     # Detailed memory analysis
    TRACE = "trace"          # Memory trace analysis with call stacks


class MemoryProfilerStatus(Enum):
    """Memory profiler operational status."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    ERROR = "error"


class MemoryCategory(Enum):
    """Types of memory being monitored."""
    SYSTEM_RSS = "system_rss"           # Resident Set Size
    SYSTEM_VMS = "system_vms"           # Virtual Memory Size
    PYTHON_OBJECTS = "python_objects"   # Python object memory
    COMPONENT_MEMORY = "component_memory"  # Component-specific memory
    CACHE_MEMORY = "cache_memory"       # Cache memory usage
    MODEL_MEMORY = "model_memory"       # AI model memory
    SESSION_MEMORY = "session_memory"   # Session-specific memory
    HEAP_MEMORY = "heap_memory"         # Heap memory
    STACK_MEMORY = "stack_memory"       # Stack memory


class MemoryMetric(Enum):
    """Types of memory metrics."""
    CURRENT_USAGE = "current_usage"
    PEAK_USAGE = "peak_usage"
    ALLOCATED = "allocated"
    DEALLOCATED = "deallocated"
    GROWTH_RATE = "growth_rate"
    FRAGMENTATION = "fragmentation"
    GC_COLLECTIONS = "gc_collections"
    LEAK_DETECTED = "leak_detected"


@dataclass
class MemoryProfilingConfig:
    """Configuration for memory profiling."""
    mode: MemoryProfilingMode = MemoryProfilingMode.STANDARD
    level: MemoryProfilingLevel = MemoryProfilingLevel.COMPONENT
    
    # Monitoring intervals
    monitoring_interval: float = 1.0  # 1 second
    gc_monitoring_interval: float = 5.0  # 5 seconds
    leak_detection_interval: float = 60.0  # 1 minute
    
    # Data retention
    max_memory_snapshots: int = 1000
    max_profile_size_mb: float = 200.0
    profile_retention_hours: int = 48
    
    # Memory thresholds
    memory_warning_threshold_mb: float = 1000.0
    memory_critical_threshold_mb: float = 2000.0
    memory_growth_rate_threshold_mb_per_sec: float = 10.0
    gc_frequency_threshold: int = 100
    
    # Leak detection
    enable_leak_detection: bool = True
    leak_detection_threshold_mb: float = 50.0
    leak_detection_cycles: int = 3
    
    # Component monitoring
    enable_component_tracking: bool = True
    enable_session_tracking: bool = True
    enable_model_tracking: bool = True
    
    # Advanced features
    enable_tracemalloc: bool = True
    tracemalloc_frames: int = 25
    enable_memory_optimization: bool = True
    enable_predictive_analysis: bool = True
    enable_fragmentation_analysis: bool = True
    
    # Integration settings
    integrate_with_tracing: bool = True
    integrate_with_metrics: bool = True
    integrate_with_health_check: bool = True


@dataclass
class MemorySnapshot:
    """Memory snapshot at a point in time."""
    timestamp: datetime
    snapshot_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # System memory
    system_rss: int = 0  # bytes
    system_vms: int = 0  # bytes
    system_available: int = 0  # bytes
    system_used_percent: float = 0.0
    
    # Process memory
    process_rss: int = 0  # bytes
    process_vms: int = 0  # bytes
    process_shared: int = 0  # bytes
    process_text: int = 0  # bytes
    process_data: int = 0  # bytes
    
    # Python memory
    python_objects_count: int = 0
    python_memory_usage: int = 0  # bytes
    gc_generation_counts: List[int] = field(default_factory=list)
    gc_collection_counts: List[int] = field(default_factory=list)
    
    # Component memory
    component_memory: Dict[str, int] = field(default_factory=dict)
    session_memory: Dict[str, int] = field(default_factory=dict)
    model_memory: Dict[str, int] = field(default_factory=dict)
    
    # Memory patterns
    memory_growth_rate: float = 0.0  # bytes per second
    fragmentation_ratio: float = 0.0
    heap_size: int = 0  # bytes
    
    # Performance correlation
    cpu_usage: float = 0.0
    io_operations: int = 0
    thread_count: int = 0


@dataclass
class MemoryLeak:
    """Memory leak detection result."""
    leak_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    detected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Leak details
    source_component: str = ""
    memory_category: MemoryCategory = MemoryCategory.PYTHON_OBJECTS
    leaked_amount: int = 0  # bytes
    growth_rate: float = 0.0  # bytes per second
    
    # Detection details
    detection_confidence: float = 0.0  # 0.0 to 1.0
    detection_cycles: int = 0
    first_detected: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Analysis
    suspected_cause: str = ""
    stack_trace: List[str] = field(default_factory=list)
    related_operations: List[str] = field(default_factory=list)
    
    # Status
    is_active: bool = True
    resolution_attempted: bool = False
    resolution_successful: bool = False


@dataclass
class MemoryOptimizationRecommendation:
    """Memory optimization recommendation."""
    category: MemoryCategory
    recommendation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Recommendation details
    priority: str = "medium"  # low, medium, high, critical
    title: str = ""
    description: str = ""
    
    # Optimization details
    potential_savings_mb: float = 0.0
    implementation_effort: str = "medium"  # low, medium, high
    risk_level: str = "low"  # low, medium, high
    
    # Technical details
    affected_components: List[str] = field(default_factory=list)
    optimization_actions: List[str] = field(default_factory=list)
    code_changes_required: bool = False
    
    # Status
    is_implemented: bool = False
    implementation_date: Optional[datetime] = None
    effectiveness_score: Optional[float] = None


@dataclass
class MemoryProfilingSession:
    """Memory profiling session data."""
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: Optional[str] = None
    description: Optional[str] = None
    
    # Session timing
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    end_time: Optional[datetime] = None
    duration: Optional[float] = None
    
    # Configuration
    config: MemoryProfilingConfig = field(default_factory=MemoryProfilingConfig)
    components_monitored: Set[str] = field(default_factory=set)
    
    # Data collection
    snapshots: List[MemorySnapshot] = field(default_factory=list)
    leaks_detected: List[MemoryLeak] = field(default_factory=list)
    recommendations: List[MemoryOptimizationRecommendation] = field(default_factory=list)
    
    # Analysis results
    peak_memory_usage: int = 0  # bytes
    average_memory_usage: int = 0  # bytes
    memory_efficiency_score: float = 0.0
    leak_count: int = 0
    optimization_potential_mb: float = 0.0
    
    # Status
    status: MemoryProfilerStatus = MemoryProfilerStatus.STOPPED
    error_message: Optional[str] = None


class MemoryTracker:
    """Tracks memory usage for specific components."""
    
    def __init__(self, component_name: str, logger):
        self.component_name = component_name
        self.logger = logger
        self.memory_usage_history = deque(maxlen=1000)
        self.allocation_tracking = {}
        self.peak_usage = 0
        self.baseline_usage = 0
        self._track_allocations = False
        
    def start_tracking(self):
        """Start tracking memory allocations."""
        self.baseline_usage = self._get_current_usage()
        self._track_allocations = True
        
    def stop_tracking(self):
        """Stop tracking memory allocations."""
        self._track_allocations = False
        
    def record_allocation(self, allocation_id: str, size: int):
        """Record a memory allocation."""
        if self._track_allocations:
            self.allocation_tracking[allocation_id] = {
                'size': size,
                'timestamp': datetime.now(timezone.utc)
            }
            
    def record_deallocation(self, allocation_id: str):
        """Record a memory deallocation."""
        if allocation_id in self.allocation_tracking:
            del self.allocation_tracking[allocation_id]
            
    def get_current_usage(self) -> int:
        """Get current memory usage for this component."""
        usage = self._get_current_usage()
        self.memory_usage_history.append({
            'timestamp': datetime.now(timezone.utc),
            'usage': usage
        })
        self.peak_usage = max(self.peak_usage, usage)
        return usage
        
    def _get_current_usage(self) -> int:
        """Internal method to get current memory usage."""
        # This would be implemented with actual memory tracking
        # For now, return a placeholder
        return sum(alloc['size'] for alloc in self.allocation_tracking.values())
        
    def get_memory_trend(self) -> Dict[str, float]:
        """Get memory usage trend analysis."""
        current_usage = self._get_current_usage()
        
        if len(self.memory_usage_history) < 2:
            return {
                'trend': 0.0, 
                'growth_rate': 0.0,
                'peak_usage': self.peak_usage,
                'current_usage': current_usage
            }
            
        recent_usage = [entry['usage'] for entry in list(self.memory_usage_history)[-10:]]
        if len(recent_usage) < 2:
            return {
                'trend': 0.0, 
                'growth_rate': 0.0,
                'peak_usage': self.peak_usage,
                'current_usage': current_usage
            }
            
        # Calculate trend
        x = list(range(len(recent_usage)))
        y = recent_usage
        n = len(x)
        
        if n < 2:
            return {
                'trend': 0.0, 
                'growth_rate': 0.0,
                'peak_usage': self.peak_usage,
                'current_usage': current_usage
            }
            
        # Simple linear regression for trend
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_x2 = sum(x[i] * x[i] for i in range(n))
        
        if (n * sum_x2 - sum_x * sum_x) == 0:
            return {
                'trend': 0.0, 
                'growth_rate': 0.0,
                'peak_usage': self.peak_usage,
                'current_usage': current_usage
            }
            
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        
        return {
            'trend': slope,
            'growth_rate': slope,
            'peak_usage': self.peak_usage,
            'current_usage': recent_usage[-1] if recent_usage else current_usage
        }


class MemoryLeakDetector:
    """Detects memory leaks in the system."""
    
    def __init__(self, config: MemoryProfilingConfig, logger):
        self.config = config
        self.logger = logger
        self.suspected_leaks = {}
        self.confirmed_leaks = []
        self.detection_history = deque(maxlen=100)
        
    async def check_for_leaks(self, snapshot: MemorySnapshot, 
                            component_trackers: Dict[str, MemoryTracker]) -> List[MemoryLeak]:
        """Check for memory leaks based on current snapshot."""
        detected_leaks = []
        
        # Check system memory growth
        system_leak = await self._check_system_memory_leak(snapshot)
        if system_leak:
            detected_leaks.append(system_leak)
            
        # Check component memory leaks
        for component_name, tracker in component_trackers.items():
            component_leak = await self._check_component_memory_leak(component_name, tracker)
            if component_leak:
                detected_leaks.append(component_leak)
                
        # Check Python object leaks
        python_leak = await self._check_python_object_leak(snapshot)
        if python_leak:
            detected_leaks.append(python_leak)
            
        return detected_leaks
        
    async def _check_system_memory_leak(self, snapshot: MemorySnapshot) -> Optional[MemoryLeak]:
        """Check for system-level memory leaks."""
        # Add to detection history
        self.detection_history.append({
            'timestamp': snapshot.timestamp,
            'memory_usage': snapshot.process_rss,
            'growth_rate': snapshot.memory_growth_rate
        })
        
        if len(self.detection_history) < self.config.leak_detection_cycles:
            return None
            
        # Analyze growth pattern
        recent_entries = list(self.detection_history)[-self.config.leak_detection_cycles:]
        growth_rates = [entry['growth_rate'] for entry in recent_entries]
        
        # Check if growth rate is consistently high
        avg_growth = statistics.mean(growth_rates)
        if avg_growth > self.config.leak_detection_threshold_mb * 1024 * 1024:  # Convert MB to bytes
            # Potential leak detected
            leak_id = "system_memory_leak"
            
            if leak_id not in self.suspected_leaks:
                self.suspected_leaks[leak_id] = {
                    'first_detected': snapshot.timestamp,
                    'detection_count': 1,
                    'growth_rates': [avg_growth]
                }
            else:
                self.suspected_leaks[leak_id]['detection_count'] += 1
                self.suspected_leaks[leak_id]['growth_rates'].append(avg_growth)
                
            # Confirm leak if detected multiple times
            if self.suspected_leaks[leak_id]['detection_count'] >= self.config.leak_detection_cycles:
                return MemoryLeak(
                    source_component="system",
                    memory_category=MemoryCategory.SYSTEM_RSS,
                    leaked_amount=int(avg_growth * self.config.leak_detection_interval),
                    growth_rate=avg_growth,
                    detection_confidence=0.8,
                    detection_cycles=self.suspected_leaks[leak_id]['detection_count'],
                    first_detected=self.suspected_leaks[leak_id]['first_detected'],
                    suspected_cause="System memory leak detected through growth rate analysis"
                )
                
        return None
        
    async def _check_component_memory_leak(self, component_name: str, 
                                         tracker: MemoryTracker) -> Optional[MemoryLeak]:
        """Check for component-specific memory leaks."""
        trend_data = tracker.get_memory_trend()
        
        if trend_data['growth_rate'] > self.config.leak_detection_threshold_mb * 1024 * 1024:
            return MemoryLeak(
                source_component=component_name,
                memory_category=MemoryCategory.COMPONENT_MEMORY,
                leaked_amount=int(trend_data['growth_rate'] * self.config.leak_detection_interval),
                growth_rate=trend_data['growth_rate'],
                detection_confidence=0.7,
                detection_cycles=1,
                suspected_cause=f"Component {component_name} showing consistent memory growth"
            )
            
        return None
        
    async def _check_python_object_leak(self, snapshot: MemorySnapshot) -> Optional[MemoryLeak]:
        """Check for Python object memory leaks."""
        # This would analyze Python object counts and memory usage
        # For now, return None as placeholder
        return None


class MemoryOptimizer:
    """Provides memory optimization recommendations."""
    
    def __init__(self, config: MemoryProfilingConfig, logger):
        self.config = config
        self.logger = logger
        self.optimization_history = []
        
    async def analyze_and_recommend(self, session: MemoryProfilingSession,
                                  snapshots: List[MemorySnapshot],
                                  leaks: List[MemoryLeak]) -> List[MemoryOptimizationRecommendation]:
        """Analyze memory usage and provide optimization recommendations."""
        recommendations = []
        
        # Analyze memory usage patterns
        if snapshots:
            # Check for high memory usage
            peak_usage = max(snapshot.process_rss for snapshot in snapshots)
            avg_usage = statistics.mean(snapshot.process_rss for snapshot in snapshots)
            
            if peak_usage > self.config.memory_critical_threshold_mb * 1024 * 1024:
                recommendations.append(MemoryOptimizationRecommendation(
                    category=MemoryCategory.SYSTEM_RSS,
                    priority="critical",
                    title="High Memory Usage Detected",
                    description=f"Peak memory usage of {peak_usage / (1024*1024):.1f}MB exceeds critical threshold",
                    potential_savings_mb=(peak_usage - avg_usage) / (1024*1024),
                    optimization_actions=[
                        "Analyze memory usage patterns",
                        "Implement memory pooling",
                        "Optimize data structures",
                        "Add memory cleanup routines"
                    ]
                ))
                
        # Analyze memory leaks
        for leak in leaks:
            if leak.is_active:
                recommendations.append(MemoryOptimizationRecommendation(
                    category=leak.memory_category,
                    priority="high",
                    title=f"Memory Leak in {leak.source_component}",
                    description=f"Detected memory leak with growth rate of {leak.growth_rate / (1024*1024):.2f}MB/s",
                    potential_savings_mb=leak.leaked_amount / (1024*1024),
                    affected_components=[leak.source_component],
                    optimization_actions=[
                        "Review memory allocation patterns",
                        "Implement proper cleanup routines",
                        "Check for circular references",
                        "Add memory monitoring"
                    ],
                    code_changes_required=True
                ))
                
        # Check GC efficiency
        if snapshots and any(len(snapshot.gc_collection_counts) > 0 for snapshot in snapshots):
            latest_snapshot = snapshots[-1]
            if latest_snapshot.gc_collection_counts:
                total_collections = sum(latest_snapshot.gc_collection_counts)
                if total_collections > self.config.gc_frequency_threshold:
                    recommendations.append(MemoryOptimizationRecommendation(
                        category=MemoryCategory.PYTHON_OBJECTS,
                        priority="medium",
                        title="High Garbage Collection Frequency",
                        description=f"Detected {total_collections} GC collections, which may indicate inefficient memory usage",
                        potential_savings_mb=10.0,  # Estimated
                        optimization_actions=[
                            "Optimize object lifecycle management",
                            "Reduce temporary object creation",
                            "Implement object pooling",
                            "Review circular references"
                        ]
                    ))
                    
        return recommendations


class MemoryProfiler:
    """
    Advanced Memory Profiler for the AI Assistant.
    
    This profiler provides comprehensive memory performance monitoring including:
    - Real-time memory usage monitoring
    - Memory leak detection and analysis
    - Component-specific memory tracking
    - Memory optimization recommendations
    - Integration with core assistant components
    - Memory pattern analysis and prediction
    - Emergency memory profiling for critical situations
    - Historical memory tracking and reporting
    """
    
    def __init__(self, container: Container):
        self.container = container
        self.config_loader = container.get(ConfigLoader) if container else None
        self.event_bus = container.get(EventBus) if container else None
        self.logger = get_logger(__name__)
        self.error_handler = container.get(ErrorHandler) if container else None
        
        # Load configuration
        profiling_config = {}
        if self.config_loader:
            profiling_config = self.config_loader.get("memory_profiling", {})
        self.config = MemoryProfilingConfig(**profiling_config)
        
        # Core components
        self.metrics_collector = container.get(MetricsCollector, default=None) if container else None
        self.trace_manager = container.get(TraceManager, default=None) if container else None
        self.health_check = container.get(HealthCheck, default=None) if container else None
        
        # Memory profiling components
        self.leak_detector = MemoryLeakDetector(self.config, self.logger)
        self.memory_optimizer = MemoryOptimizer(self.config, self.logger)
        
        # Component tracking
        self.component_trackers: Dict[str, MemoryTracker] = {}
        self.session_trackers: Dict[str, MemoryTracker] = {}
        
        # Profiling state
        self.current_session: Optional[MemoryProfilingSession] = None
        self.profiling_sessions: Dict[str, MemoryProfilingSession] = {}
        self.recent_sessions = deque(maxlen=self.config.max_memory_snapshots)
        
        # Threading and monitoring
        self.monitoring_thread: Optional[threading.Thread] = None
        self.stop_monitoring_event = threading.Event()
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)
        
        # Memory data storage
        self.memory_snapshots = deque(maxlen=self.config.max_memory_snapshots)
        self.detected_leaks: List[MemoryLeak] = []
        self.optimization_recommendations: List[MemoryOptimizationRecommendation] = []
        
        # Output paths
        output_dir_path = "data/memory_profiling"
        if self.config_loader:
            output_dir_path = self.config_loader.get("memory_profiling.output_dir", "data/memory_profiling")
        self.output_dir = Path(output_dir_path)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize tracemalloc if enabled
        if self.config.enable_tracemalloc and not tracemalloc.is_tracing():
            tracemalloc.start(self.config.tracemalloc_frames)
            
        # Setup monitoring and event handlers
        self._setup_monitoring()
        self._setup_event_handlers()
        
        self.logger.info("Advanced Memory Profiler initialized successfully")
        
    def _setup_monitoring(self) -> None:
        """Setup metrics and monitoring."""
        if self.metrics_collector:
            # Register memory metrics
            memory_metrics = [
                ("memory_profiler_sessions_total", "counter", "Total memory profiling sessions"),
                ("memory_profiler_snapshots_total", "counter", "Total memory snapshots taken"),
                ("memory_profiler_leaks_detected", "counter", "Memory leaks detected"),
                ("memory_profiler_current_memory_mb", "gauge", "Current memory usage in MB"),
                ("memory_profiler_peak_memory_mb", "gauge", "Peak memory usage in MB"),
                ("memory_profiler_gc_collections", "counter", "Garbage collection counts"),
                ("memory_profiler_optimization_recommendations", "counter", "Optimization recommendations generated"),
            ]
            
            for name, metric_type, description in memory_metrics:
                try:
                    if metric_type == "counter":
                        self.metrics_collector.increment_counter(name, description)
                    elif metric_type == "gauge":
                        self.metrics_collector.set_gauge(name, 0, description)
                except Exception as e:
                    self.logger.warning(f"Failed to register metric {name}: {str(e)}")
                    
    def _setup_event_handlers(self) -> None:
        """Setup event handlers for system integration."""
        if self.event_bus:
            # Subscribe to component lifecycle events
            self.event_bus.subscribe("ComponentHealthChanged", self._handle_component_health_change)
            self.event_bus.subscribe("SystemStateChanged", self._handle_system_state_change)
            self.event_bus.subscribe("MemoryOperationStarted", self._handle_memory_operation_started)
            self.event_bus.subscribe("MemoryOperationCompleted", self._handle_memory_operation_completed)
            
    async def _handle_component_health_change(self, event):
        """Handle component health change events."""
        try:
            component_name = getattr(event, 'component_name', 'unknown')
            health_status = getattr(event, 'is_healthy', True)
            
            if not health_status and component_name in self.component_trackers:
                # Component became unhealthy, check for memory issues
                tracker = self.component_trackers[component_name]
                trend_data = tracker.get_memory_trend()
                
                if trend_data['growth_rate'] > 0:
                    self.logger.warning(f"Component {component_name} unhealthy with memory growth: {trend_data['growth_rate']:.2f} bytes/s")
                    
        except Exception as e:
            self.logger.error(f"Error handling component health change: {str(e)}")
            
    async def _handle_system_state_change(self, event):
        """Handle system state change events."""
        try:
            if hasattr(event, 'state') and event.state == 'high_memory_usage':
                # System experiencing high memory usage
                if not self.is_profiling():
                    await self.start_profiling(
                        session_name="emergency_memory_profiling",
                        description="Emergency profiling due to high memory usage",
                        mode=MemoryProfilingMode.EMERGENCY
                    )
                    
        except Exception as e:
            self.logger.error(f"Error handling system state change: {str(e)}")
            
    async def _handle_memory_operation_started(self, event):
        """Handle memory operation started events."""
        try:
            operation_id = getattr(event, 'operation_id', None)
            component_name = getattr(event, 'component_name', 'unknown')
            
            if operation_id and component_name in self.component_trackers:
                tracker = self.component_trackers[component_name]
                tracker.record_allocation(operation_id, getattr(event, 'size', 0))
                
        except Exception as e:
            self.logger.error(f"Error handling memory operation started: {str(e)}")
            
    async def _handle_memory_operation_completed(self, event):
        """Handle memory operation completed events."""
        try:
            operation_id = getattr(event, 'operation_id', None)
            component_name = getattr(event, 'component_name', 'unknown')
            
            if operation_id and component_name in self.component_trackers:
                tracker = self.component_trackers[component_name]
                tracker.record_deallocation(operation_id)
                
        except Exception as e:
            self.logger.error(f"Error handling memory operation completed: {str(e)}")

    @handle_exceptions
    async def start_profiling(
        self,
        session_name: str = None,
        description: str = None,
        mode: MemoryProfilingMode = None,
        level: MemoryProfilingLevel = None,
        components_to_monitor: Optional[List[str]] = None
    ) -> str:
        """
        Start memory profiling session.
        
        Args:
            session_name: Optional name for the session
            description: Optional description
            mode: Profiling mode override
            level: Profiling level override
            components_to_monitor: Specific components to monitor
            
        Returns:
            Session ID
        """
        if self.is_profiling():
            raise RuntimeError("Memory profiling session already active")
            
        # Create new session
        config_dict = asdict(self.config)
        if mode:
            config_dict['mode'] = mode
        if level:
            config_dict['level'] = level
            
        session = MemoryProfilingSession(
            name=session_name or f"memory_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            description=description,
            config=MemoryProfilingConfig(**config_dict)
        )
        
        if components_to_monitor:
            session.components_monitored.update(components_to_monitor)
            
        self.current_session = session
        self.profiling_sessions[session.session_id] = session
        session.status = MemoryProfilerStatus.STARTING
        
        try:
            # Initialize component trackers for monitored components
            if components_to_monitor:
                for component_name in components_to_monitor:
                    if component_name not in self.component_trackers:
                        self.component_trackers[component_name] = MemoryTracker(component_name, self.logger)
                    self.component_trackers[component_name].start_tracking()
                    
            # Start monitoring thread
            self.stop_monitoring_event.clear()
            self.monitoring_thread = threading.Thread(
                target=self._monitoring_loop,
                name=f"MemoryProfiler-{session.session_id[:8]}",
                daemon=True
            )
            self.monitoring_thread.start()
            
            session.status = MemoryProfilerStatus.RUNNING
            
            # Emit profiling started event
            if self.event_bus:
                await self.event_bus.emit(ProfilerStarted(
                    profiler_type="memory",
                    session_id=session.session_id,
                    configuration=asdict(session.config)
                ))
                
            # Update metrics
            if self.metrics_collector:
                self.metrics_collector.increment_counter("memory_profiler_sessions_total")
                
            self.logger.info(f"Started memory profiling session: {session.session_id}")
            return session.session_id
            
        except Exception as e:
            session.status = MemoryProfilerStatus.ERROR
            session.error_message = str(e)
            self.logger.error(f"Failed to start memory profiling: {str(e)}")
            raise
            
    @handle_exceptions
    async def stop_profiling(self, session_id: Optional[str] = None) -> Optional[MemoryProfilingSession]:
        """
        Stop memory profiling session.
        
        Args:
            session_id: Session to stop (current session if None)
            
        Returns:
            Completed session data
        """
        if not self.is_profiling():
            self.logger.warning("No active memory profiling session to stop")
            return None
            
        session = self.current_session
        if session_id and session_id != session.session_id:
            session = self.profiling_sessions.get(session_id)
            if not session:
                raise ValueError(f"Session {session_id} not found")
                
        session.status = MemoryProfilerStatus.STOPPING
        
        try:
            # Stop monitoring
            self.stop_monitoring_event.set()
            if self.monitoring_thread and self.monitoring_thread.is_alive():
                self.monitoring_thread.join(timeout=5.0)
                
            # Stop component trackers
            for tracker in self.component_trackers.values():
                tracker.stop_tracking()
                
            # Finalize session
            session.end_time = datetime.now(timezone.utc)
            session.duration = (session.end_time - session.start_time).total_seconds()
            session.status = MemoryProfilerStatus.STOPPED
            
            # Generate final analysis
            await self._finalize_session_analysis(session)
            
            # Save session data
            await self._save_session_data(session)
            
            # Emit profiling stopped event
            if self.event_bus:
                await self.event_bus.emit(ProfilerStopped(
                    profiler_type="memory",
                    session_id=session.session_id,
                    duration=session.duration,
                    data_size=len(session.snapshots)
                ))
                
            self.current_session = None
            self.recent_sessions.append(session)
            
            self.logger.info(f"Stopped memory profiling session: {session.session_id}")
            return session
            
        except Exception as e:
            session.status = MemoryProfilerStatus.ERROR
            session.error_message = str(e)
            self.logger.error(f"Failed to stop memory profiling: {str(e)}")
            raise
            
    def is_profiling(self) -> bool:
        """Check if memory profiling is currently active."""
        return (self.current_session is not None and 
                self.current_session.status == MemoryProfilerStatus.RUNNING)
                
    def get_current_session(self) -> Optional[MemoryProfilingSession]:
        """Get the current profiling session."""
        return self.current_session
        
    def get_session(self, session_id: str) -> Optional[MemoryProfilingSession]:
        """Get a specific profiling session."""
        return self.profiling_sessions.get(session_id)
        
    def list_sessions(self) -> List[MemoryProfilingSession]:
        """List all profiling sessions."""
        return list(self.profiling_sessions.values())
        
    async def register_component(self, component_name: str) -> None:
        """Register a component for memory tracking."""
        if component_name not in self.component_trackers:
            self.component_trackers[component_name] = MemoryTracker(component_name, self.logger)
            self.logger.info(f"Registered component for memory tracking: {component_name}")
            
        # If profiling is active, start tracking immediately
        if self.is_profiling():
            self.component_trackers[component_name].start_tracking()
            self.current_session.components_monitored.add(component_name)
            
    async def unregister_component(self, component_name: str) -> None:
        """Unregister a component from memory tracking."""
        if component_name in self.component_trackers:
            self.component_trackers[component_name].stop_tracking()
            del self.component_trackers[component_name]
            self.logger.info(f"Unregistered component from memory tracking: {component_name}")
            
        # Remove from current session if active
        if self.is_profiling() and component_name in self.current_session.components_monitored:
            self.current_session.components_monitored.remove(component_name)
            
    def get_current_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage across all categories."""
        try:
            # System memory
            system_memory = psutil.virtual_memory()
            process = psutil.Process()
            process_memory = process.memory_info()
            
            # Python memory
            python_objects = len(gc.get_objects())
            gc_stats = gc.get_stats()
            
            # Component memory
            component_memory = {}
            for name, tracker in self.component_trackers.items():
                component_memory[name] = tracker.get_current_usage()
                
            return {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'system_memory': {
                    'total': system_memory.total,
                    'available': system_memory.available,
                    'used': system_memory.used,
                    'used_percent': system_memory.percent
                },
                'process_memory': {
                    'rss': process_memory.rss,
                    'vms': process_memory.vms,
                    'shared': getattr(process_memory, 'shared', 0),
                    'text': getattr(process_memory, 'text', 0),
                    'data': getattr(process_memory, 'data', 0)
                },
                'python_memory': {
                    'objects_count': python_objects,
                    'gc_stats': gc_stats
                },
                'component_memory': component_memory
            }
            
        except Exception as e:
            self.logger.error(f"Error getting current memory usage: {str(e)}")
            return {}
            
    async def get_memory_snapshot(self) -> MemorySnapshot:
        """Take a memory snapshot."""
        try:
            # Get system and process memory
            system_memory = psutil.virtual_memory()
            process = psutil.Process()
            process_memory = process.memory_info()
            
            # Get Python memory info
            python_objects = len(gc.get_objects())
            gc_stats = gc.get_stats()
            gc_generation_counts = [stat.get('collections', 0) for stat in gc_stats]
            gc_collection_counts = [stat.get('collected', 0) for stat in gc_stats]
            
            # Get component memory
            component_memory = {}
            for name, tracker in self.component_trackers.items():
                component_memory[name] = tracker.get_current_usage()
                
            # Calculate memory growth rate
            memory_growth_rate = 0.0
            if len(self.memory_snapshots) > 0:
                last_snapshot = self.memory_snapshots[-1]
                time_diff = (datetime.now(timezone.utc) - last_snapshot.timestamp).total_seconds()
                if time_diff > 0:
                    memory_growth_rate = (process_memory.rss - last_snapshot.process_rss) / time_diff
                    
            # Create snapshot
            snapshot = MemorySnapshot(
                timestamp=datetime.now(timezone.utc),
                system_rss=system_memory.used,
                system_vms=system_memory.total,
                system_available=system_memory.available,
                system_used_percent=system_memory.percent,
                process_rss=process_memory.rss,
                process_vms=process_memory.vms,
                process_shared=getattr(process_memory, 'shared', 0),
                process_text=getattr(process_memory, 'text', 0),
                process_data=getattr(process_memory, 'data', 0),
                python_objects_count=python_objects,
                python_memory_usage=process_memory.rss,  # Approximation
                gc_generation_counts=gc_generation_counts,
                gc_collection_counts=gc_collection_counts,
                component_memory=component_memory,
                memory_growth_rate=memory_growth_rate,
                cpu_usage=process.cpu_percent(),
                thread_count=process.num_threads()
            )
            
            self.memory_snapshots.append(snapshot)
            
            # Update metrics
            if self.metrics_collector:
                self.metrics_collector.increment_counter("memory_profiler_snapshots_total")
                self.metrics_collector.set_gauge("memory_profiler_current_memory_mb", 
                                                process_memory.rss / (1024 * 1024))
                                                
            return snapshot
            
        except Exception as e:
            self.logger.error(f"Error taking memory snapshot: {str(e)}")
            raise
            
    def _monitoring_loop(self) -> None:
        """Background monitoring loop."""
        self.logger.info("Started memory monitoring loop")
        
        last_leak_check = time.time()
        last_gc_check = time.time()
        
        try:
            while not self.stop_monitoring_event.is_set():
                try:
                    # Take memory snapshot
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    snapshot = loop.run_until_complete(self.get_memory_snapshot())
                    
                    if self.current_session:
                        self.current_session.snapshots.append(snapshot)
                        
                    # Check for memory leaks periodically
                    current_time = time.time()
                    if current_time - last_leak_check >= self.config.leak_detection_interval:
                        leaks = loop.run_until_complete(
                            self.leak_detector.check_for_leaks(snapshot, self.component_trackers)
                        )
                        for leak in leaks:
                            self._handle_detected_leak(leak)
                        last_leak_check = current_time
                        
                    # Check GC statistics periodically
                    if current_time - last_gc_check >= self.config.gc_monitoring_interval:
                        self._check_gc_efficiency(snapshot)
                        last_gc_check = current_time
                        
                    # Check memory thresholds
                    self._check_memory_thresholds(snapshot)
                    
                    loop.close()
                    
                except Exception as e:
                    self.logger.error(f"Error in memory monitoring loop: {str(e)}")
                    
                # Wait for next monitoring interval
                self.stop_monitoring_event.wait(self.config.monitoring_interval)
                
        except Exception as e:
            self.logger.error(f"Fatal error in memory monitoring loop: {str(e)}")
        finally:
            self.logger.info("Stopped memory monitoring loop")
            
    def _handle_detected_leak(self, leak: MemoryLeak) -> None:
        """Handle a detected memory leak."""
        self.detected_leaks.append(leak)
        
        if self.current_session:
            self.current_session.leaks_detected.append(leak)
            self.current_session.leak_count += 1
            
        # Update metrics
        if self.metrics_collector:
            self.metrics_collector.increment_counter("memory_profiler_leaks_detected")
            
        # Emit leak detection event
        if self.event_bus:
            asyncio.create_task(self.event_bus.emit(PerformanceAlertTriggered(
                alert_type="memory_leak",
                severity="high",
                component=leak.source_component,
                message=f"Memory leak detected: {leak.suspected_cause}",
                metrics={'leaked_amount_mb': leak.leaked_amount / (1024 * 1024)}
            )))
            
        self.logger.warning(f"Memory leak detected in {leak.source_component}: {leak.suspected_cause}")
        
    def _check_gc_efficiency(self, snapshot: MemorySnapshot) -> None:
        """Check garbage collection efficiency."""
        if not snapshot.gc_collection_counts:
            return
            
        total_collections = sum(snapshot.gc_collection_counts)
        
        if self.metrics_collector:
            self.metrics_collector.increment_counter("memory_profiler_gc_collections", total_collections)
            
        if total_collections > self.config.gc_frequency_threshold:
            self.logger.warning(f"High GC activity detected: {total_collections} collections")
            
    def _check_memory_thresholds(self, snapshot: MemorySnapshot) -> None:
        """Check memory usage thresholds."""
        memory_mb = snapshot.process_rss / (1024 * 1024)
        
        if memory_mb > self.config.memory_critical_threshold_mb:
            # Critical memory usage
            if self.event_bus:
                asyncio.create_task(self.event_bus.emit(PerformanceThresholdExceeded(
                    metric_name="memory_usage",
                    current_value=memory_mb,
                    threshold_value=self.config.memory_critical_threshold_mb,
                    severity="critical"
                )))
                
        elif memory_mb > self.config.memory_warning_threshold_mb:
            # Warning memory usage
            if self.event_bus:
                asyncio.create_task(self.event_bus.emit(PerformanceThresholdExceeded(
                    metric_name="memory_usage",
                    current_value=memory_mb,
                    threshold_value=self.config.memory_warning_threshold_mb,
                    severity="warning"
                )))
                
        # Check memory growth rate
        if abs(snapshot.memory_growth_rate) > self.config.memory_growth_rate_threshold_mb_per_sec * 1024 * 1024:
            self.logger.warning(f"High memory growth rate: {snapshot.memory_growth_rate / (1024*1024):.2f} MB/s")
            
    async def _finalize_session_analysis(self, session: MemoryProfilingSession) -> None:
        """Finalize analysis for a completed session."""
        try:
            if not session.snapshots:
                return
                
            # Calculate session statistics
            memory_usages = [snapshot.process_rss for snapshot in session.snapshots]
            session.peak_memory_usage = max(memory_usages)
            session.average_memory_usage = int(statistics.mean(memory_usages))
            
            # Calculate memory efficiency score (0-100)
            if session.peak_memory_usage > 0:
                efficiency = (session.average_memory_usage / session.peak_memory_usage) * 100
                session.memory_efficiency_score = min(100, max(0, efficiency))
                
            # Generate optimization recommendations
            recommendations = await self.memory_optimizer.analyze_and_recommend(
                session, session.snapshots, session.leaks_detected
            )
            session.recommendations = recommendations
            session.optimization_potential_mb = sum(
                rec.potential_savings_mb for rec in recommendations
            )
            
            # Update metrics
            if self.metrics_collector:
                self.metrics_collector.increment_counter(
                    "memory_profiler_optimization_recommendations", 
                    len(recommendations)
                )
                self.metrics_collector.set_gauge(
                    "memory_profiler_peak_memory_mb",
                    session.peak_memory_usage / (1024 * 1024)
                )
                
            self.logger.info(f"Finalized analysis for session {session.session_id}: "
                           f"Peak: {session.peak_memory_usage / (1024*1024):.1f}MB, "
                           f"Efficiency: {session.memory_efficiency_score:.1f}%, "
                           f"Recommendations: {len(recommendations)}")
                           
        except Exception as e:
            self.logger.error(f"Error finalizing session analysis: {str(e)}")
            
    async def _save_session_data(self, session: MemoryProfilingSession) -> None:
        """Save session data to disk."""
        try:
            session_file = self.output_dir / f"memory_session_{session.session_id}.json"
            
            # Convert session to serializable format
            session_data = {
                'session_id': session.session_id,
                'name': session.name,
                'description': session.description,
                'start_time': session.start_time.isoformat(),
                'end_time': session.end_time.isoformat() if session.end_time else None,
                'duration': session.duration,
                'config': asdict(session.config),
                'components_monitored': list(session.components_monitored),
                'snapshots': [asdict(snapshot) for snapshot in session.snapshots],
                'leaks_detected': [asdict(leak) for leak in session.leaks_detected],
                'recommendations': [asdict(rec) for rec in session.recommendations],
                'peak_memory_usage': session.peak_memory_usage,
                'average_memory_usage': session.average_memory_usage,
                'memory_efficiency_score': session.memory_efficiency_score,
                'leak_count': session.leak_count,
                'optimization_potential_mb': session.optimization_potential_mb,
                'status': session.status.value,
                'error_message': session.error_message
            }
            
            # Save to compressed JSON file
            with gzip.open(session_file.with_suffix('.json.gz'), 'wt', encoding='utf-8') as f:
                json.dump(session_data, f, indent=2, default=str)
                
            self.logger.info(f"Saved session data to {session_file.with_suffix('.json.gz')}")
            
        except Exception as e:
            self.logger.error(f"Error saving session data: {str(e)}")
            
    async def generate_memory_report(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Generate a comprehensive memory report."""
        try:
            if session_id:
                session = self.get_session(session_id)
                if not session:
                    raise ValueError(f"Session {session_id} not found")
                sessions = [session]
            else:
                sessions = list(self.profiling_sessions.values())
                
            if not sessions:
                return {'error': 'No sessions found'}
                
            # Generate comprehensive report
            report = {
                'report_id': str(uuid.uuid4()),
                'generated_at': datetime.now(timezone.utc).isoformat(),
                'total_sessions': len(sessions),
                'sessions': []
            }
            
            total_peak_memory = 0
            total_leaks = 0
            total_recommendations = 0
            
            for session in sessions:
                session_summary = {
                    'session_id': session.session_id,
                    'name': session.name,
                    'duration': session.duration,
                    'peak_memory_mb': session.peak_memory_usage / (1024 * 1024) if session.peak_memory_usage else 0,
                    'average_memory_mb': session.average_memory_usage / (1024 * 1024) if session.average_memory_usage else 0,
                    'efficiency_score': session.memory_efficiency_score,
                    'leaks_detected': len(session.leaks_detected),
                    'recommendations': len(session.recommendations),
                    'optimization_potential_mb': session.optimization_potential_mb,
                    'components_monitored': list(session.components_monitored)
                }
                
                report['sessions'].append(session_summary)
                total_peak_memory = max(total_peak_memory, session.peak_memory_usage or 0)
                total_leaks += len(session.leaks_detected)
                total_recommendations += len(session.recommendations)
                
            # Add summary statistics
            report['summary'] = {
                'peak_memory_mb': total_peak_memory / (1024 * 1024),
                'total_leaks_detected': total_leaks,
                'total_recommendations': total_recommendations,
                'average_efficiency_score': statistics.mean([
                    s.memory_efficiency_score for s in sessions if s.memory_efficiency_score > 0
                ]) if sessions else 0
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating memory report: {str(e)}")
            return {'error': str(e)}
            
    async def cleanup_old_data(self) -> None:
        """Clean up old profiling data based on retention policy."""
        try:
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=self.config.profile_retention_hours)
            
            # Remove old sessions
            sessions_to_remove = []
            for session_id, session in self.profiling_sessions.items():
                if session.start_time < cutoff_time:
                    sessions_to_remove.append(session_id)
                    
            for session_id in sessions_to_remove:
                del self.profiling_sessions[session_id]
                self.logger.info(f"Cleaned up old session: {session_id}")
                
            # Remove old session files
            for session_file in self.output_dir.glob("memory_session_*.json.gz"):
                if session_file.stat().st_mtime < cutoff_time.timestamp():
                    session_file.unlink()
                    self.logger.info(f"Removed old session file: {session_file}")
                    
        except Exception as e:
            self.logger.error(f"Error cleaning up old data: {str(e)}")
            
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of the memory profiler."""
        try:
            current_memory = self.get_current_memory_usage()
            process_memory_mb = current_memory.get('process_memory', {}).get('rss', 0) / (1024 * 1024)
            
            return {
                'profiler_status': self.current_session.status.value if self.current_session else 'stopped',
                'sessions_count': len(self.profiling_sessions),
                'components_tracked': len(self.component_trackers),
                'current_memory_mb': process_memory_mb,
                'leaks_detected': len(self.detected_leaks),
                'recommendations_generated': len(self.optimization_recommendations),
                'monitoring_active': self.monitoring_thread.is_alive() if self.monitoring_thread else False,
                'last_snapshot': self.memory_snapshots[-1].timestamp.isoformat() if self.memory_snapshots else None
            }
            
        except Exception as e:
            return {'error': str(e)}


# Decorator for memory profiling
def profile_memory(component_name: str = None, track_allocations: bool = True):
    """
    Decorator for profiling memory usage of functions.
    
    Args:
        component_name: Component name for tracking
        track_allocations: Whether to track individual allocations
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Get profiler from container if available
            try:
                from src.core.dependency_injection import Container
                container = Container()
                profiler = container.get(MemoryProfiler, default=None)
            except:
                profiler = None
                
            if not profiler:
                return await func(*args, **kwargs)
                
            # Get or create component tracker
            comp_name = component_name or func.__name__
            if comp_name not in profiler.component_trackers:
                await profiler.register_component(comp_name)
                
            tracker = profiler.component_trackers[comp_name]
            
            # Record allocation start
            operation_id = str(uuid.uuid4())
            if track_allocations:
                tracker.record_allocation(operation_id, 0)  # Size will be calculated
                
            start_memory = tracker.get_current_usage()
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                
                # Calculate memory used
                end_memory = tracker.get_current_usage()
                duration = time.time() - start_time
                memory_used = end_memory - start_memory
                
                # Update allocation record
                if track_allocations and operation_id in tracker.allocation_tracking:
                    tracker.allocation_tracking[operation_id]['size'] = memory_used
                    
                # Log significant memory usage
                if memory_used > 1024 * 1024:  # > 1MB
                    profiler.logger.info(f"Function {func.__name__} used {memory_used / (1024*1024):.2f}MB in {duration:.2f}s")
                    
                return result
                
            except Exception as e:
                # Record deallocation on error
                if track_allocations:
                    tracker.record_deallocation(operation_id)
                raise
            finally:
                # Record deallocation
                if track_allocations:
                    tracker.record_deallocation(operation_id)
                    
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Simplified version for sync functions
            return func(*args, **kwargs)
            
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        
    return decorator


# Context manager for memory profiling
@asynccontextmanager
async def memory_profiling_context(
    profiler: MemoryProfiler,
    session_name: str = None,
    description: str = None,
    mode: MemoryProfilingMode = MemoryProfilingMode.STANDARD
):
    """
    Context manager for memory profiling sessions.
    
    Args:
        profiler: Memory profiler instance
        session_name: Session name
        description: Session description
        mode: Profiling mode
    """
    session_id = None
    
    try:
        session_id = await profiler.start_profiling(
            session_name=session_name,
            description=description,
            mode=mode
        )
        yield session_id
        
    finally:
        if session_id:
            await profiler.stop_profiling(session_id)