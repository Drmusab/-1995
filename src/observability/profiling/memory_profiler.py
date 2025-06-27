"""
Advanced Memory Profiler for AI Assistant
Author: Drmusab
Last Modified: 2025-01-10 15:30:00 UTC

This module provides comprehensive memory profiling capabilities for the AI assistant,
including real-time monitoring, memory leak detection, per-component tracking, and
integration with all core system components.
"""

import tracemalloc
import threading
import time
import asyncio
import functools
import sys
import gc
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
import logging
import concurrent.futures
import os
import resource

# Core imports
from src.core.config.loader import ConfigLoader
from src.core.events.event_bus import EventBus
from src.core.events.event_types import (
    PerformanceThresholdExceeded, MemoryOperationStarted, MemoryOperationCompleted,
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

# Try to import optional dependencies
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import pympler.tracker
    import pympler.summary
    import pympler.muppy
    PYMPLER_AVAILABLE = True
except ImportError:
    PYMPLER_AVAILABLE = False


class MemoryProfilingMode(Enum):
    """Memory profiling modes."""
    OFF = "off"
    BASIC = "basic"                  # Basic memory usage tracking
    STANDARD = "standard"            # Tracemalloc-based profiling
    ADVANCED = "advanced"            # Full featured with all backends
    PRODUCTION = "production"        # Optimized for production use
    DEBUG = "debug"                  # Debug mode with detailed info


class MemoryProfilingLevel(Enum):
    """Memory profiling detail levels."""
    LOW = "low"           # Process-level memory tracking
    MEDIUM = "medium"     # Component-level profiling
    HIGH = "high"         # Object-level profiling with allocation tracking
    DETAILED = "detailed" # Maximum detail with leak detection


class MemoryBackend(Enum):
    """Memory profiling backends."""
    TRACEMALLOC = "tracemalloc"
    PSUTIL = "psutil"
    PYMPLER = "pympler"
    SYSTEM = "system"
    GC = "gc"


class MemoryMetricType(Enum):
    """Types of memory metrics."""
    RSS = "rss"                      # Resident Set Size
    VMS = "vms"                      # Virtual Memory Size
    USS = "uss"                      # Unique Set Size
    PSS = "pss"                      # Proportional Set Size
    SHARED = "shared"                # Shared memory
    HEAP_SIZE = "heap_size"          # Python heap size
    OBJECTS_COUNT = "objects_count"  # Number of tracked objects
    ALLOCATIONS = "allocations"      # Memory allocations
    DEALLOCATIONS = "deallocations"  # Memory deallocations
    PEAK_MEMORY = "peak_memory"      # Peak memory usage


@dataclass
class MemoryProfilingConfig:
    """Configuration for memory profiling."""
    mode: MemoryProfilingMode = MemoryProfilingMode.STANDARD
    level: MemoryProfilingLevel = MemoryProfilingLevel.MEDIUM
    sampling_interval: float = 1.0  # 1 second sampling for memory
    
    # Backends configuration
    enabled_backends: List[MemoryBackend] = field(default_factory=lambda: [
        MemoryBackend.TRACEMALLOC, MemoryBackend.SYSTEM, MemoryBackend.GC
    ])
    tracemalloc_frames: int = 10
    
    # Data retention
    max_snapshots: int = 1000
    max_profile_size_mb: float = 200.0
    profile_retention_hours: int = 48
    
    # Memory thresholds (in MB)
    memory_warning_threshold_mb: float = 1000.0
    memory_critical_threshold_mb: float = 2000.0
    memory_growth_rate_threshold_mb_per_min: float = 50.0
    
    # Leak detection
    enable_leak_detection: bool = True
    leak_detection_interval: int = 300  # 5 minutes
    leak_threshold_objects: int = 1000
    leak_threshold_size_mb: float = 100.0
    
    # Component tracking
    enable_component_tracking: bool = True
    track_session_memory: bool = True
    track_plugin_memory: bool = True
    
    # Performance optimization
    enable_gc_monitoring: bool = True
    enable_allocation_tracking: bool = True
    max_tracked_allocations: int = 10000
    
    # Integration settings
    integrate_with_tracing: bool = True
    integrate_with_metrics: bool = True
    integrate_with_health_check: bool = True
    
    # Output configuration
    enable_snapshots: bool = True
    enable_trend_analysis: bool = True
    enable_recommendations: bool = True


@dataclass
class MemorySnapshot:
    """A snapshot of memory usage at a specific point in time."""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Process-level memory
    rss_mb: float = 0.0
    vms_mb: float = 0.0
    uss_mb: float = 0.0
    pss_mb: float = 0.0
    shared_mb: float = 0.0
    
    # Python-specific memory
    heap_size_mb: float = 0.0
    objects_count: int = 0
    total_allocations: int = 0
    total_deallocations: int = 0
    
    # Garbage collection
    gc_collections: Dict[int, int] = field(default_factory=dict)
    gc_objects: int = 0
    
    # Component memory breakdown
    component_memory: Dict[str, float] = field(default_factory=dict)
    
    # Top memory consumers
    top_allocations: List[Dict[str, Any]] = field(default_factory=list)
    
    # Session and context info
    session_id: Optional[str] = None
    active_components: List[str] = field(default_factory=list)


@dataclass
class MemoryLeak:
    """Information about a detected memory leak."""
    leak_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    first_detected: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Leak characteristics
    component: str = ""
    leak_type: str = ""  # "objects", "memory", "allocation"
    severity: str = "medium"  # "low", "medium", "high", "critical"
    
    # Growth pattern
    growth_rate_mb_per_min: float = 0.0
    objects_growth_rate: float = 0.0
    total_leaked_mb: float = 0.0
    total_leaked_objects: int = 0
    
    # Source information
    allocation_traceback: List[str] = field(default_factory=list)
    suspected_functions: List[str] = field(default_factory=list)
    
    # Recommendation
    recommendation: str = ""
    confidence: float = 0.0


@dataclass
class ComponentMemoryProfile:
    """Memory profile for a specific component."""
    component_id: str
    component_type: str
    
    # Memory usage statistics
    current_memory_mb: float = 0.0
    peak_memory_mb: float = 0.0
    average_memory_mb: float = 0.0
    memory_growth_trend: float = 0.0
    
    # Allocation patterns
    allocations_per_second: float = 0.0
    deallocations_per_second: float = 0.0
    net_allocation_rate: float = 0.0
    
    # Object tracking
    tracked_objects: int = 0
    object_types: Dict[str, int] = field(default_factory=dict)
    
    # Performance impact
    gc_pressure: float = 0.0
    memory_efficiency: float = 0.0
    
    # Timestamps
    first_seen: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


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
    
    # Memory data
    snapshots: List[MemorySnapshot] = field(default_factory=list)
    component_profiles: Dict[str, ComponentMemoryProfile] = field(default_factory=dict)
    detected_leaks: List[MemoryLeak] = field(default_factory=list)
    
    # Session statistics
    peak_memory_mb: float = 0.0
    average_memory_mb: float = 0.0
    memory_growth_rate: float = 0.0
    total_allocations: int = 0
    total_deallocations: int = 0
    
    # Analysis results
    recommendations: List[str] = field(default_factory=list)
    performance_impact: Dict[str, float] = field(default_factory=dict)
    
    # Output files
    report_path: Optional[Path] = None
    data_export_path: Optional[Path] = None


class AdvancedMemoryProfiler:
    """
    Advanced Memory Profiler for the AI Assistant.
    
    This profiler provides comprehensive memory monitoring including:
    - Real-time memory usage tracking across multiple backends
    - Memory leak detection and alerting
    - Per-component memory attribution and analysis
    - Session-based memory profiling
    - Integration with core assistant components
    - Memory usage predictions and optimization recommendations
    - Garbage collection monitoring and analysis
    - Memory fragmentation tracking
    - Thread-specific memory usage analysis
    - Memory growth trend analysis and forecasting
    """
    
    def __init__(self, container: Container):
        """Initialize the memory profiler with dependency injection."""
        self.container = container
        self.logger = get_logger(__name__)
        
        # Core dependencies
        self.config_loader = self.container.get('ConfigLoader') 
        self.event_bus = self.container.get('EventBus')
        self.metrics = self.container.get('MetricsCollector')
        self.trace_manager = self.container.get('TraceManager')
        self.health_check = self.container.get('HealthCheck')
        self.error_handler = self.container.get('ErrorHandler')
        
        # Configuration
        self.config = self._load_config()
        
        # State management
        self.is_running = False
        self.current_session: Optional[MemoryProfilingSession] = None
        self.stop_event = threading.Event()
        
        # Data storage
        self.memory_snapshots: deque = deque(maxlen=self.config.max_snapshots)
        self.component_profiles: Dict[str, ComponentMemoryProfile] = {}
        self.detected_leaks: List[MemoryLeak] = []
        self.profiling_sessions: Dict[str, MemoryProfilingSession] = {}
        
        # Memory tracking infrastructure
        self.tracked_objects: weakref.WeakSet = weakref.WeakSet()
        self.allocation_tracker: Optional[Any] = None
        self.gc_callbacks: List[Callable] = []
        
        # Threading and async support
        self.monitoring_thread: Optional[threading.Thread] = None
        self.leak_detection_thread: Optional[threading.Thread] = None
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        
        # Backend managers
        self.active_backends: Set[MemoryBackend] = set()
        self._setup_backends()
        
        # Output configuration
        self.output_dir = Path(self.config_loader.get("memory_profiling.output_dir", "data/memory_profiling"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Integration setup
        self._setup_monitoring()
        self._setup_event_handlers()
        self._setup_health_checks()
        
        self.logger.info("Advanced Memory Profiler initialized")
    
    def _load_config(self) -> MemoryProfilingConfig:
        """Load memory profiling configuration."""
        try:
            config_data = self.config_loader.get("memory_profiling", {})
            
            # Convert string enums to enum instances
            if "mode" in config_data:
                config_data["mode"] = MemoryProfilingMode(config_data["mode"])
            if "level" in config_data:
                config_data["level"] = MemoryProfilingLevel(config_data["level"])
            if "enabled_backends" in config_data:
                config_data["enabled_backends"] = [
                    MemoryBackend(backend) for backend in config_data["enabled_backends"]
                ]
            
            return MemoryProfilingConfig(**config_data)
        except Exception as e:
            self.logger.warning(f"Failed to load memory profiling config: {str(e)}")
            return MemoryProfilingConfig()
    
    def _setup_backends(self) -> None:
        """Setup available memory profiling backends."""
        self.active_backends.clear()
        
        for backend in self.config.enabled_backends:
            try:
                if backend == MemoryBackend.TRACEMALLOC:
                    if not tracemalloc.is_tracing():
                        tracemalloc.start(self.config.tracemalloc_frames)
                    self.active_backends.add(backend)
                    self.logger.debug("Tracemalloc backend enabled")
                
                elif backend == MemoryBackend.PSUTIL:
                    if PSUTIL_AVAILABLE:
                        self.active_backends.add(backend)
                        self.logger.debug("Psutil backend enabled")
                    else:
                        self.logger.warning("Psutil backend requested but not available")
                
                elif backend == MemoryBackend.PYMPLER:
                    if PYMPLER_AVAILABLE:
                        self.allocation_tracker = pympler.tracker.SummaryTracker()
                        self.active_backends.add(backend)
                        self.logger.debug("Pympler backend enabled")
                    else:
                        self.logger.warning("Pympler backend requested but not available")
                
                elif backend == MemoryBackend.SYSTEM:
                    self.active_backends.add(backend)
                    self.logger.debug("System backend enabled")
                
                elif backend == MemoryBackend.GC:
                    if self.config.enable_gc_monitoring:
                        self._setup_gc_monitoring()
                        self.active_backends.add(backend)
                        self.logger.debug("GC backend enabled")
                
            except Exception as e:
                self.logger.error(f"Failed to setup {backend.value} backend: {str(e)}")
        
        if not self.active_backends:
            self.logger.warning("No memory profiling backends available")
    
    def _setup_gc_monitoring(self) -> None:
        """Setup garbage collection monitoring."""
        try:
            # Enable garbage collection debugging
            if self.config.level in [MemoryProfilingLevel.HIGH, MemoryProfilingLevel.DETAILED]:
                gc.set_debug(gc.DEBUG_STATS)
            
            # Register GC callback for tracking
            def gc_callback(phase, info):
                self._handle_gc_event(phase, info)
            
            self.gc_callbacks.append(gc_callback)
            
        except Exception as e:
            self.logger.error(f"Failed to setup GC monitoring: {str(e)}")
    
    def _setup_monitoring(self) -> None:
        """Setup metrics and monitoring."""
        try:
            # Register memory metrics
            memory_metrics = [
                ("memory_profiler_rss_bytes", "gauge", "Process RSS memory usage"),
                ("memory_profiler_vms_bytes", "gauge", "Process VMS memory usage"),
                ("memory_profiler_heap_objects", "gauge", "Number of objects in heap"),
                ("memory_profiler_allocations_total", "counter", "Total memory allocations"),
                ("memory_profiler_deallocations_total", "counter", "Total memory deallocations"),
                ("memory_profiler_leaks_detected", "counter", "Number of memory leaks detected"),
                ("memory_profiler_gc_collections", "counter", "Garbage collection count"),
                ("memory_profiler_active_sessions", "gauge", "Active memory profiling sessions"),
            ]
            
            for metric_name, metric_type, description in memory_metrics:
                self.metrics.register_metric(metric_name, metric_type, description)
            
        except Exception as e:
            self.logger.warning(f"Failed to setup monitoring: {str(e)}")
    
    def _setup_event_handlers(self) -> None:
        """Setup event handlers for system integration."""
        try:
            # Listen for component lifecycle events
            self.event_bus.subscribe("component_health_changed", self._handle_component_health_change)
            self.event_bus.subscribe("component_started", self._handle_component_started)
            self.event_bus.subscribe("component_stopped", self._handle_component_stopped)
            
            # Listen for session events
            self.event_bus.subscribe("session_started", self._handle_session_started)
            self.event_bus.subscribe("session_ended", self._handle_session_ended)
            
            # Listen for workflow events
            self.event_bus.subscribe("workflow_started", self._handle_workflow_started)
            self.event_bus.subscribe("workflow_completed", self._handle_workflow_completed)
            
            # Listen for plugin events
            self.event_bus.subscribe("plugin_loaded", self._handle_plugin_loaded)
            self.event_bus.subscribe("plugin_unloaded", self._handle_plugin_unloaded)
            
        except Exception as e:
            self.logger.warning(f"Failed to setup event handlers: {str(e)}")
    
    def _setup_health_checks(self) -> None:
        """Setup health checks for memory monitoring."""
        try:
            # Register memory health check
            self.health_check.register_check(
                "memory_profiler",
                self._health_check,
                interval=30.0,  # Check every 30 seconds
                timeout=5.0
            )
            
        except Exception as e:
            self.logger.warning(f"Failed to setup health checks: {str(e)}")
    
    async def _health_check(self) -> Dict[str, Any]:
        """Perform health check for memory profiler."""
        try:
            snapshot = await self._take_memory_snapshot()
            
            # Check memory thresholds
            warnings = []
            if snapshot.rss_mb > self.config.memory_warning_threshold_mb:
                warnings.append(f"High RSS memory usage: {snapshot.rss_mb:.1f}MB")
            
            if snapshot.rss_mb > self.config.memory_critical_threshold_mb:
                warnings.append(f"Critical RSS memory usage: {snapshot.rss_mb:.1f}MB")
            
            # Check for active leaks
            active_leaks = [leak for leak in self.detected_leaks if leak.severity in ["high", "critical"]]
            if active_leaks:
                warnings.append(f"Active memory leaks detected: {len(active_leaks)}")
            
            status = "unhealthy" if warnings else "healthy"
            
            return {
                "status": status,
                "memory_mb": snapshot.rss_mb,
                "active_backends": [backend.value for backend in self.active_backends],
                "warnings": warnings,
                "profiling_active": self.is_running
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    @handle_exceptions
    async def start_profiling(
        self,
        session_name: str = None,
        description: str = None,
        mode: MemoryProfilingMode = None,
        level: MemoryProfilingLevel = None
    ) -> str:
        """Start memory profiling session."""
        if self.is_running:
            raise RuntimeError("Memory profiling is already running")
        
        # Generate session ID
        session_id = str(uuid.uuid4())
        session_name = session_name or f"memory_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create session configuration
        config = MemoryProfilingConfig(
            mode=mode or self.config.mode,
            level=level or self.config.level,
            **{k: v for k, v in asdict(self.config).items() 
               if k not in ['mode', 'level']}
        )
        
        # Create profiling session
        self.current_session = MemoryProfilingSession(
            session_id=session_id,
            name=session_name,
            description=description,
            config=config
        )
        
        # Reset tracking data
        self.memory_snapshots.clear()
        self.component_profiles.clear()
        self.detected_leaks.clear()
        
        # Setup backends for this session
        self._setup_backends()
        
        # Start monitoring
        self.is_running = True
        self.stop_event.clear()
        
        # Start monitoring thread
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            name=f"MemoryProfiler-{session_id[:8]}",
            daemon=True
        )
        self.monitoring_thread.start()
        
        # Start leak detection if enabled
        if config.enable_leak_detection:
            self.leak_detection_thread = threading.Thread(
                target=self._leak_detection_loop,
                name=f"LeakDetector-{session_id[:8]}",
                daemon=True
            )
            self.leak_detection_thread.start()
        
        # Emit profiler started event
        await self.event_bus.emit(MemoryOperationStarted(
            request_id=session_id,
            operation_type="profiling_started"
        ))
        
        # Update metrics
        self.metrics.increment("memory_profiler_sessions_total")
        self.metrics.set("memory_profiler_active_sessions", 1)
        
        self.logger.info(f"Started memory profiling session: {session_id} ({session_name})")
        return session_id
    
    @handle_exceptions
    async def stop_profiling(self) -> MemoryProfilingSession:
        """Stop memory profiling and return session data."""
        if not self.is_running:
            raise RuntimeError("Memory profiling is not running")
        
        try:
            # Signal threads to stop
            self.stop_event.set()
            
            # Wait for threads to finish
            if self.monitoring_thread and self.monitoring_thread.is_alive():
                self.monitoring_thread.join(timeout=5.0)
            
            if self.leak_detection_thread and self.leak_detection_thread.is_alive():
                self.leak_detection_thread.join(timeout=5.0)
            
            # Finalize session
            if self.current_session:
                self.current_session.end_time = datetime.now(timezone.utc)
                self.current_session.duration = (
                    self.current_session.end_time - self.current_session.start_time
                ).total_seconds()
                
                # Copy collected data to session
                self.current_session.snapshots = list(self.memory_snapshots)
                self.current_session.component_profiles = self.component_profiles.copy()
                self.current_session.detected_leaks = self.detected_leaks.copy()
                
                # Calculate session statistics
                await self._calculate_session_statistics()
                
                # Generate recommendations
                if self.config.enable_recommendations:
                    await self._generate_recommendations()
                
                # Generate report
                await self._generate_session_report()
                
                # Store session
                self.profiling_sessions[self.current_session.session_id] = self.current_session
                
                session = self.current_session
                self.current_session = None
            else:
                raise RuntimeError("No active profiling session found")
            
            # Reset state
            self.is_running = False
            
            # Emit profiler stopped event
            await self.event_bus.emit(MemoryOperationCompleted(
                request_id=session.session_id,
                operations_completed=["profiling_stopped"]
            ))
            
            # Update metrics
            self.metrics.set("memory_profiler_active_sessions", 0)
            
            self.logger.info(f"Stopped memory profiling session: {session.session_id}")
            return session
            
        except Exception as e:
            self.is_running = False
            self.logger.error(f"Error stopping memory profiling: {str(e)}")
            raise
    
    def _monitoring_loop(self) -> None:
        """Main memory monitoring loop."""
        self.logger.debug("Starting memory monitoring loop")
        
        while not self.stop_event.is_set():
            try:
                # Take memory snapshot
                snapshot = asyncio.run(self._take_memory_snapshot())
                self.memory_snapshots.append(snapshot)
                
                # Update component profiles
                asyncio.run(self._update_component_profiles(snapshot))
                
                # Check thresholds
                asyncio.run(self._check_memory_thresholds(snapshot))
                
                # Update metrics
                self._update_metrics(snapshot)
                
                # Sleep until next sampling interval
                if self.stop_event.wait(self.config.sampling_interval):
                    break
                    
            except Exception as e:
                self.logger.error(f"Error in memory monitoring loop: {str(e)}")
                time.sleep(self.config.sampling_interval)
        
        self.logger.debug("Memory monitoring loop stopped")
    
    def _leak_detection_loop(self) -> None:
        """Memory leak detection loop."""
        self.logger.debug("Starting leak detection loop")
        
        while not self.stop_event.is_set():
            try:
                # Run leak detection
                asyncio.run(self._detect_memory_leaks())
                
                # Sleep until next check
                if self.stop_event.wait(self.config.leak_detection_interval):
                    break
                    
            except Exception as e:
                self.logger.error(f"Error in leak detection loop: {str(e)}")
                time.sleep(self.config.leak_detection_interval)
        
        self.logger.debug("Leak detection loop stopped")
    
    async def _take_memory_snapshot(self) -> MemorySnapshot:
        """Take a comprehensive memory snapshot."""
        snapshot = MemorySnapshot()
        
        try:
            # Get process memory info
            if MemoryBackend.PSUTIL in self.active_backends and PSUTIL_AVAILABLE:
                process = psutil.Process()
                memory_info = process.memory_info()
                snapshot.rss_mb = memory_info.rss / 1024 / 1024
                snapshot.vms_mb = memory_info.vms / 1024 / 1024
                
                # Get full memory info if available
                try:
                    full_info = process.memory_full_info()
                    snapshot.uss_mb = full_info.uss / 1024 / 1024
                    snapshot.pss_mb = full_info.pss / 1024 / 1024
                    snapshot.shared_mb = full_info.shared / 1024 / 1024
                except AttributeError:
                    pass  # Not available on all platforms
            
            # Get system memory info
            if MemoryBackend.SYSTEM in self.active_backends:
                try:
                    # Use resource module for basic memory info
                    usage = resource.getrusage(resource.RUSAGE_SELF)
                    if snapshot.rss_mb == 0.0:  # Fallback if psutil not available
                        snapshot.rss_mb = usage.ru_maxrss / 1024  # Convert from KB to MB
                except Exception:
                    pass
            
            # Get Python memory info
            if MemoryBackend.TRACEMALLOC in self.active_backends:
                if tracemalloc.is_tracing():
                    current, peak = tracemalloc.get_traced_memory()
                    snapshot.heap_size_mb = current / 1024 / 1024
                    
                    # Get top allocations
                    top_stats = tracemalloc.take_snapshot().statistics('lineno')
                    snapshot.top_allocations = [
                        {
                            "filename": stat.traceback.format()[-1] if stat.traceback else "unknown",
                            "size_mb": stat.size / 1024 / 1024,
                            "count": stat.count
                        }
                        for stat in top_stats[:10]
                    ]
            
            # Get garbage collection info
            if MemoryBackend.GC in self.active_backends:
                gc_stats = gc.get_stats()
                snapshot.gc_collections = {i: stats['collections'] for i, stats in enumerate(gc_stats)}
                snapshot.gc_objects = len(gc.get_objects())
            
            # Get component memory breakdown if available
            if self.config.enable_component_tracking:
                snapshot.component_memory = await self._get_component_memory_breakdown()
            
            # Set session context
            if self.current_session:
                snapshot.session_id = self.current_session.session_id
                snapshot.active_components = list(self.component_profiles.keys())
            
        except Exception as e:
            self.logger.error(f"Error taking memory snapshot: {str(e)}")
        
        return snapshot
    
    async def _get_component_memory_breakdown(self) -> Dict[str, float]:
        """Get memory usage breakdown by component."""
        component_memory = {}
        
        try:
            # Try to get component manager for active components
            try:
                component_manager = self.container.get('EnhancedComponentManager')
                if hasattr(component_manager, 'get_component_health'):
                    for component_id in component_manager.registered_components:
                        # Estimate component memory usage
                        # This is a simplified approach - in practice you might want
                        # more sophisticated tracking per component
                        component_memory[component_id] = 0.0
            except Exception:
                pass
            
            # Add core engine components if available
            core_components = [
                'core_engine', 'session_manager', 'plugin_manager',
                'workflow_orchestrator', 'interaction_handler'
            ]
            
            for component in core_components:
                try:
                    comp_instance = self.container.get(component)
                    if comp_instance:
                        # Basic memory estimation
                        component_memory[component] = sys.getsizeof(comp_instance) / 1024 / 1024
                except Exception:
                    pass
            
        except Exception as e:
            self.logger.debug(f"Could not get component memory breakdown: {str(e)}")
        
        return component_memory
    
    async def _update_component_profiles(self, snapshot: MemorySnapshot) -> None:
        """Update component memory profiles."""
        try:
            for component_id, memory_mb in snapshot.component_memory.items():
                if component_id not in self.component_profiles:
                    # Create new component profile
                    self.component_profiles[component_id] = ComponentMemoryProfile(
                        component_id=component_id,
                        component_type=self._get_component_type(component_id)
                    )
                
                profile = self.component_profiles[component_id]
                
                # Update memory statistics
                profile.current_memory_mb = memory_mb
                profile.peak_memory_mb = max(profile.peak_memory_mb, memory_mb)
                profile.last_updated = datetime.now(timezone.utc)
                
                # Calculate moving average (simple implementation)
                if hasattr(profile, '_memory_history'):
                    profile._memory_history.append(memory_mb)
                    if len(profile._memory_history) > 100:
                        profile._memory_history.pop(0)
                    profile.average_memory_mb = sum(profile._memory_history) / len(profile._memory_history)
                else:
                    profile._memory_history = [memory_mb]
                    profile.average_memory_mb = memory_mb
                
                # Update object counts if available
                if MemoryBackend.GC in self.active_backends:
                    profile.tracked_objects = snapshot.gc_objects
        
        except Exception as e:
            self.logger.error(f"Error updating component profiles: {str(e)}")
    
    def _get_component_type(self, component_id: str) -> str:
        """Get the type of a component."""
        try:
            component = self.container.get(component_id)
            return type(component).__name__ if component else "unknown"
        except Exception:
            return "unknown"
    
    async def _check_memory_thresholds(self, snapshot: MemorySnapshot) -> None:
        """Check memory thresholds and emit alerts."""
        try:
            # Check warning threshold
            if snapshot.rss_mb > self.config.memory_warning_threshold_mb:
                await self.event_bus.emit(PerformanceThresholdExceeded(
                    metric_name="memory_rss_mb",
                    current_value=snapshot.rss_mb,
                    threshold=self.config.memory_warning_threshold_mb,
                    threshold_type="upper"
                ))
            
            # Check critical threshold
            if snapshot.rss_mb > self.config.memory_critical_threshold_mb:
                await self.event_bus.emit(PerformanceThresholdExceeded(
                    metric_name="memory_rss_mb_critical",
                    current_value=snapshot.rss_mb,
                    threshold=self.config.memory_critical_threshold_mb,
                    threshold_type="upper"
                ))
            
            # Check memory growth rate
            if len(self.memory_snapshots) >= 2:
                prev_snapshot = self.memory_snapshots[-2]
                time_diff = (snapshot.timestamp - prev_snapshot.timestamp).total_seconds() / 60.0  # minutes
                
                if time_diff > 0:
                    growth_rate = (snapshot.rss_mb - prev_snapshot.rss_mb) / time_diff
                    
                    if growth_rate > self.config.memory_growth_rate_threshold_mb_per_min:
                        await self.event_bus.emit(PerformanceThresholdExceeded(
                            metric_name="memory_growth_rate_mb_per_min",
                            current_value=growth_rate,
                            threshold=self.config.memory_growth_rate_threshold_mb_per_min,
                            threshold_type="upper"
                        ))
        
        except Exception as e:
            self.logger.error(f"Error checking memory thresholds: {str(e)}")
    
    def _update_metrics(self, snapshot: MemorySnapshot) -> None:
        """Update metrics with memory data."""
        try:
            self.metrics.set("memory_profiler_rss_bytes", snapshot.rss_mb * 1024 * 1024)
            self.metrics.set("memory_profiler_vms_bytes", snapshot.vms_mb * 1024 * 1024)
            self.metrics.set("memory_profiler_heap_objects", snapshot.gc_objects)
            
            # Update allocation counters
            if hasattr(snapshot, 'total_allocations'):
                self.metrics.set("memory_profiler_allocations_total", snapshot.total_allocations)
            if hasattr(snapshot, 'total_deallocations'):
                self.metrics.set("memory_profiler_deallocations_total", snapshot.total_deallocations)
            
            # Update GC metrics
            if snapshot.gc_collections:
                total_collections = sum(snapshot.gc_collections.values())
                self.metrics.set("memory_profiler_gc_collections", total_collections)
        
        except Exception as e:
            self.logger.error(f"Error updating metrics: {str(e)}")
    
    async def _detect_memory_leaks(self) -> None:
        """Detect potential memory leaks."""
        try:
            if len(self.memory_snapshots) < 3:
                return  # Need at least 3 snapshots for trend analysis
            
            # Analyze memory growth patterns
            recent_snapshots = list(self.memory_snapshots)[-10:]  # Last 10 snapshots
            
            # Check for consistent memory growth
            memory_values = [s.rss_mb for s in recent_snapshots]
            if len(memory_values) >= 3:
                # Simple linear regression to detect growth trend
                x_values = list(range(len(memory_values)))
                n = len(memory_values)
                
                sum_x = sum(x_values)
                sum_y = sum(memory_values)
                sum_xy = sum(x * y for x, y in zip(x_values, memory_values))
                sum_x2 = sum(x * x for x in x_values)
                
                # Calculate slope (growth rate)
                if n * sum_x2 - sum_x * sum_x != 0:
                    slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
                    
                    # Convert slope to MB per minute
                    time_interval = self.config.sampling_interval / 60.0  # Convert to minutes
                    growth_rate_mb_per_min = slope / time_interval
                    
                    # Check if growth rate indicates a leak
                    if growth_rate_mb_per_min > self.config.memory_growth_rate_threshold_mb_per_min / 2:
                        await self._create_memory_leak_alert(
                            leak_type="memory_growth",
                            growth_rate=growth_rate_mb_per_min,
                            component="system",
                            severity=self._calculate_leak_severity(growth_rate_mb_per_min)
                        )
            
            # Check component-specific leaks
            for component_id, profile in self.component_profiles.items():
                if hasattr(profile, '_memory_history') and len(profile._memory_history) >= 5:
                    # Check for consistent growth in component memory
                    recent_values = profile._memory_history[-5:]
                    if all(recent_values[i] <= recent_values[i+1] for i in range(len(recent_values)-1)):
                        # Monotonic increase detected
                        growth = recent_values[-1] - recent_values[0]
                        if growth > 10.0:  # More than 10MB growth
                            await self._create_memory_leak_alert(
                                leak_type="component_growth",
                                growth_rate=growth,
                                component=component_id,
                                severity="medium"
                            )
        
        except Exception as e:
            self.logger.error(f"Error detecting memory leaks: {str(e)}")
    
    def _calculate_leak_severity(self, growth_rate: float) -> str:
        """Calculate severity based on growth rate."""
        if growth_rate > self.config.memory_growth_rate_threshold_mb_per_min * 2:
            return "critical"
        elif growth_rate > self.config.memory_growth_rate_threshold_mb_per_min:
            return "high"
        elif growth_rate > self.config.memory_growth_rate_threshold_mb_per_min / 2:
            return "medium"
        else:
            return "low"
    
    async def _create_memory_leak_alert(self, leak_type: str, growth_rate: float, component: str, severity: str) -> None:
        """Create a memory leak alert."""
        try:
            # Check if we already have a recent alert for this component
            recent_leaks = [
                leak for leak in self.detected_leaks
                if leak.component == component and 
                (datetime.now(timezone.utc) - leak.last_updated).total_seconds() < 300  # 5 minutes
            ]
            
            if recent_leaks:
                # Update existing leak
                leak = recent_leaks[0]
                leak.last_updated = datetime.now(timezone.utc)
                leak.growth_rate_mb_per_min = growth_rate
                leak.severity = severity
            else:
                # Create new leak alert
                leak = MemoryLeak(
                    component=component,
                    leak_type=leak_type,
                    severity=severity,
                    growth_rate_mb_per_min=growth_rate,
                    recommendation=self._generate_leak_recommendation(leak_type, component)
                )
                
                self.detected_leaks.append(leak)
                
                # Emit leak detection event
                await self.event_bus.emit(PerformanceThresholdExceeded(
                    metric_name="memory_leak_detected",
                    current_value=growth_rate,
                    threshold=self.config.memory_growth_rate_threshold_mb_per_min / 2
                ))
                
                # Update metrics
                self.metrics.increment("memory_profiler_leaks_detected")
                
                self.logger.warning(f"Memory leak detected in {component}: {growth_rate:.2f}MB/min")
        
        except Exception as e:
            self.logger.error(f"Error creating memory leak alert: {str(e)}")
    
    def _generate_leak_recommendation(self, leak_type: str, component: str) -> str:
        """Generate recommendation for memory leak."""
        recommendations = {
            "memory_growth": f"Monitor {component} for excessive memory allocation. Consider implementing memory pooling or caching strategies.",
            "component_growth": f"Review {component} for proper resource cleanup. Check for unclosed resources, circular references, or excessive caching.",
            "object_growth": f"Investigate object creation patterns in {component}. Consider using weak references or object pooling."
        }
        
        return recommendations.get(leak_type, f"Monitor {component} memory usage and review for potential leaks.")
    
    # Event handlers for system integration
    async def _handle_component_health_change(self, event: ComponentHealthChanged) -> None:
        """Handle component health change event."""
        try:
            if event.component_id in self.component_profiles:
                profile = self.component_profiles[event.component_id]
                # Could add health-based memory analysis here
                self.logger.debug(f"Component {event.component_id} health changed to {event.new_status}")
        except Exception as e:
            self.logger.error(f"Error handling component health change: {str(e)}")
    
    async def _handle_component_started(self, event) -> None:
        """Handle component started event."""
        try:
            component_id = getattr(event, 'component_id', 'unknown')
            if self.config.enable_component_tracking and component_id != 'unknown':
                # Initialize component profile
                if component_id not in self.component_profiles:
                    self.component_profiles[component_id] = ComponentMemoryProfile(
                        component_id=component_id,
                        component_type=self._get_component_type(component_id)
                    )
                self.logger.debug(f"Started tracking memory for component: {component_id}")
        except Exception as e:
            self.logger.error(f"Error handling component started: {str(e)}")
    
    async def _handle_component_stopped(self, event) -> None:
        """Handle component stopped event."""
        try:
            component_id = getattr(event, 'component_id', 'unknown')
            if component_id in self.component_profiles:
                self.logger.debug(f"Component {component_id} stopped, keeping profile for analysis")
        except Exception as e:
            self.logger.error(f"Error handling component stopped: {str(e)}")
    
    async def _handle_session_started(self, event) -> None:
        """Handle session started event."""
        try:
            if self.config.track_session_memory:
                session_id = getattr(event, 'session_id', 'unknown')
                self.logger.debug(f"Started tracking memory for session: {session_id}")
        except Exception as e:
            self.logger.error(f"Error handling session started: {str(e)}")
    
    async def _handle_session_ended(self, event) -> None:
        """Handle session ended event."""
        try:
            session_id = getattr(event, 'session_id', 'unknown')
            self.logger.debug(f"Session {session_id} ended")
        except Exception as e:
            self.logger.error(f"Error handling session ended: {str(e)}")
    
    async def _handle_workflow_started(self, event) -> None:
        """Handle workflow started event."""
        try:
            workflow_id = getattr(event, 'workflow_id', 'unknown')
            self.logger.debug(f"Workflow {workflow_id} started, monitoring memory impact")
        except Exception as e:
            self.logger.error(f"Error handling workflow started: {str(e)}")
    
    async def _handle_workflow_completed(self, event) -> None:
        """Handle workflow completed event."""
        try:
            workflow_id = getattr(event, 'workflow_id', 'unknown')
            self.logger.debug(f"Workflow {workflow_id} completed")
        except Exception as e:
            self.logger.error(f"Error handling workflow completed: {str(e)}")
    
    async def _handle_plugin_loaded(self, event) -> None:
        """Handle plugin loaded event."""
        try:
            if self.config.track_plugin_memory:
                plugin_id = getattr(event, 'plugin_id', 'unknown')
                if plugin_id != 'unknown':
                    # Initialize plugin memory tracking
                    self.component_profiles[f"plugin_{plugin_id}"] = ComponentMemoryProfile(
                        component_id=f"plugin_{plugin_id}",
                        component_type="plugin"
                    )
                    self.logger.debug(f"Started tracking memory for plugin: {plugin_id}")
        except Exception as e:
            self.logger.error(f"Error handling plugin loaded: {str(e)}")
    
    async def _handle_plugin_unloaded(self, event) -> None:
        """Handle plugin unloaded event."""
        try:
            plugin_id = getattr(event, 'plugin_id', 'unknown')
            profile_key = f"plugin_{plugin_id}"
            if profile_key in self.component_profiles:
                self.logger.debug(f"Plugin {plugin_id} unloaded")
        except Exception as e:
            self.logger.error(f"Error handling plugin unloaded: {str(e)}")
    
    def _handle_gc_event(self, phase: str, info: Dict[str, Any]) -> None:
        """Handle garbage collection event."""
        try:
            if self.is_running:
                self.logger.debug(f"GC event: {phase}, info: {info}")
                # Could add detailed GC analysis here
        except Exception as e:
            self.logger.error(f"Error handling GC event: {str(e)}")
    
    async def _calculate_session_statistics(self) -> None:
        """Calculate statistics for the current session."""
        try:
            if not self.current_session or not self.memory_snapshots:
                return
            
            snapshots = list(self.memory_snapshots)
            
            # Calculate memory statistics
            memory_values = [s.rss_mb for s in snapshots]
            self.current_session.peak_memory_mb = max(memory_values)
            self.current_session.average_memory_mb = sum(memory_values) / len(memory_values)
            
            # Calculate growth rate
            if len(snapshots) >= 2:
                time_diff = (snapshots[-1].timestamp - snapshots[0].timestamp).total_seconds() / 60.0
                if time_diff > 0:
                    memory_diff = snapshots[-1].rss_mb - snapshots[0].rss_mb
                    self.current_session.memory_growth_rate = memory_diff / time_diff
            
            # Calculate allocation statistics
            if snapshots and hasattr(snapshots[-1], 'total_allocations'):
                self.current_session.total_allocations = snapshots[-1].total_allocations
            if snapshots and hasattr(snapshots[-1], 'total_deallocations'):
                self.current_session.total_deallocations = snapshots[-1].total_deallocations
            
            # Calculate performance impact
            self.current_session.performance_impact = {
                "memory_overhead_mb": max(0, self.current_session.peak_memory_mb - memory_values[0]) if memory_values else 0,
                "gc_pressure": len([s for s in snapshots if s.gc_objects > 0]),
                "leak_count": len(self.detected_leaks)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating session statistics: {str(e)}")
    
    async def _generate_recommendations(self) -> None:
        """Generate memory optimization recommendations."""
        try:
            if not self.current_session:
                return
            
            recommendations = []
            
            # Memory usage recommendations
            if self.current_session.peak_memory_mb > self.config.memory_warning_threshold_mb:
                recommendations.append(
                    f"High memory usage detected ({self.current_session.peak_memory_mb:.1f}MB). "
                    "Consider implementing memory pooling or reducing cache sizes."
                )
            
            # Memory growth recommendations
            if self.current_session.memory_growth_rate > self.config.memory_growth_rate_threshold_mb_per_min:
                recommendations.append(
                    f"Memory growth rate is high ({self.current_session.memory_growth_rate:.2f}MB/min). "
                    "Check for memory leaks and implement proper resource cleanup."
                )
            
            # Component-specific recommendations
            for component_id, profile in self.component_profiles.items():
                if profile.peak_memory_mb > 100.0:  # 100MB threshold for components
                    recommendations.append(
                        f"Component '{component_id}' has high memory usage ({profile.peak_memory_mb:.1f}MB). "
                        "Consider optimizing data structures or implementing lazy loading."
                    )
            
            # Garbage collection recommendations
            if any(s.gc_objects > 10000 for s in self.memory_snapshots):
                recommendations.append(
                    "High number of tracked objects detected. Consider using object pooling "
                    "or reducing object creation frequency."
                )
            
            # Leak-specific recommendations
            for leak in self.detected_leaks:
                if leak.severity in ["high", "critical"]:
                    recommendations.append(leak.recommendation)
            
            self.current_session.recommendations = recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {str(e)}")
    
    async def _generate_session_report(self) -> None:
        """Generate a comprehensive session report."""
        try:
            if not self.current_session:
                return
            
            # Create report data structure
            report_data = {
                "session_info": {
                    "session_id": self.current_session.session_id,
                    "name": self.current_session.name,
                    "description": self.current_session.description,
                    "start_time": self.current_session.start_time.isoformat(),
                    "end_time": self.current_session.end_time.isoformat() if self.current_session.end_time else None,
                    "duration_seconds": self.current_session.duration,
                    "config": asdict(self.current_session.config)
                },
                "memory_statistics": {
                    "peak_memory_mb": self.current_session.peak_memory_mb,
                    "average_memory_mb": self.current_session.average_memory_mb,
                    "memory_growth_rate_mb_per_min": self.current_session.memory_growth_rate,
                    "total_allocations": self.current_session.total_allocations,
                    "total_deallocations": self.current_session.total_deallocations,
                    "snapshots_count": len(self.current_session.snapshots)
                },
                "component_profiles": {
                    comp_id: {
                        "type": profile.component_type,
                        "current_memory_mb": profile.current_memory_mb,
                        "peak_memory_mb": profile.peak_memory_mb,
                        "average_memory_mb": profile.average_memory_mb,
                        "tracked_objects": profile.tracked_objects
                    }
                    for comp_id, profile in self.current_session.component_profiles.items()
                },
                "detected_leaks": [
                    {
                        "leak_id": leak.leak_id,
                        "component": leak.component,
                        "leak_type": leak.leak_type,
                        "severity": leak.severity,
                        "growth_rate_mb_per_min": leak.growth_rate_mb_per_min,
                        "recommendation": leak.recommendation,
                        "first_detected": leak.first_detected.isoformat(),
                        "last_updated": leak.last_updated.isoformat()
                    }
                    for leak in self.current_session.detected_leaks
                ],
                "recommendations": self.current_session.recommendations,
                "performance_impact": self.current_session.performance_impact
            }
            
            # Save report to file
            report_filename = f"memory_profile_{self.current_session.session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            report_path = self.output_dir / report_filename
            
            with open(report_path, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)
            
            self.current_session.report_path = report_path
            self.logger.info(f"Memory profiling report saved to: {report_path}")
            
            # Also save raw snapshot data
            data_filename = f"memory_snapshots_{self.current_session.session_id}.json"
            data_path = self.output_dir / data_filename
            
            snapshots_data = [asdict(snapshot) for snapshot in self.current_session.snapshots]
            with open(data_path, 'w') as f:
                json.dump(snapshots_data, f, indent=2, default=str)
            
            self.current_session.data_export_path = data_path
            
        except Exception as e:
            self.logger.error(f"Error generating session report: {str(e)}")
    
    # Public API methods
    def get_current_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage metrics."""
        try:
            # Use create_task instead of asyncio.run when in event loop
            try:
                loop = asyncio.get_running_loop()
                # If we're in an event loop, we need to schedule the coroutine differently
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self._take_memory_snapshot())
                    snapshot = future.result(timeout=5.0)
            except RuntimeError:
                # No event loop running, safe to use asyncio.run
                snapshot = asyncio.run(self._take_memory_snapshot())
            
            return {
                "rss_mb": snapshot.rss_mb,
                "vms_mb": snapshot.vms_mb,
                "uss_mb": snapshot.uss_mb,
                "heap_size_mb": snapshot.heap_size_mb,
                "gc_objects": snapshot.gc_objects
            }
        except Exception as e:
            self.logger.error(f"Error getting current memory usage: {str(e)}")
            return {}
    
    def get_component_memory_usage(self, component_id: str = None) -> Dict[str, Any]:
        """Get memory usage for specific component or all components."""
        try:
            if component_id:
                if component_id in self.component_profiles:
                    profile = self.component_profiles[component_id]
                    return {
                        "component_id": profile.component_id,
                        "component_type": profile.component_type,
                        "current_memory_mb": profile.current_memory_mb,
                        "peak_memory_mb": profile.peak_memory_mb,
                        "average_memory_mb": profile.average_memory_mb,
                        "tracked_objects": profile.tracked_objects
                    }
                else:
                    return {}
            else:
                return {
                    comp_id: {
                        "component_type": profile.component_type,
                        "current_memory_mb": profile.current_memory_mb,
                        "peak_memory_mb": profile.peak_memory_mb,
                        "average_memory_mb": profile.average_memory_mb,
                        "tracked_objects": profile.tracked_objects
                    }
                    for comp_id, profile in self.component_profiles.items()
                }
        except Exception as e:
            self.logger.error(f"Error getting component memory usage: {str(e)}")
            return {}
    
    def get_memory_trends(self, window_minutes: int = 10) -> Dict[str, List[float]]:
        """Get memory usage trends over the specified time window."""
        try:
            now = datetime.now(timezone.utc)
            cutoff_time = now - timedelta(minutes=window_minutes)
            
            recent_snapshots = [
                s for s in self.memory_snapshots 
                if s.timestamp >= cutoff_time
            ]
            
            return {
                "timestamps": [s.timestamp.isoformat() for s in recent_snapshots],
                "rss_mb": [s.rss_mb for s in recent_snapshots],
                "vms_mb": [s.vms_mb for s in recent_snapshots],
                "heap_size_mb": [s.heap_size_mb for s in recent_snapshots],
                "gc_objects": [s.gc_objects for s in recent_snapshots]
            }
        except Exception as e:
            self.logger.error(f"Error getting memory trends: {str(e)}")
            return {}
    
    def get_active_leaks(self) -> List[Dict[str, Any]]:
        """Get currently active memory leaks."""
        try:
            return [
                {
                    "leak_id": leak.leak_id,
                    "component": leak.component,
                    "leak_type": leak.leak_type,
                    "severity": leak.severity,
                    "growth_rate_mb_per_min": leak.growth_rate_mb_per_min,
                    "total_leaked_mb": leak.total_leaked_mb,
                    "recommendation": leak.recommendation,
                    "age_minutes": (datetime.now(timezone.utc) - leak.first_detected).total_seconds() / 60.0
                }
                for leak in self.detected_leaks
            ]
        except Exception as e:
            self.logger.error(f"Error getting active leaks: {str(e)}")
            return []
    
    def clear_leak_alerts(self, leak_id: str = None) -> bool:
        """Clear memory leak alerts."""
        try:
            if leak_id:
                self.detected_leaks = [leak for leak in self.detected_leaks if leak.leak_id != leak_id]
                self.logger.info(f"Cleared leak alert: {leak_id}")
            else:
                self.detected_leaks.clear()
                self.logger.info("Cleared all leak alerts")
            return True
        except Exception as e:
            self.logger.error(f"Error clearing leak alerts: {str(e)}")
            return False
    
    async def force_garbage_collection(self) -> Dict[str, int]:
        """Force garbage collection and return statistics."""
        try:
            # Get pre-GC stats
            pre_objects = len(gc.get_objects())
            
            # Force collection
            collected = gc.collect()
            
            # Get post-GC stats
            post_objects = len(gc.get_objects())
            
            stats = {
                "collected_objects": collected,
                "objects_before": pre_objects,
                "objects_after": post_objects,
                "objects_freed": pre_objects - post_objects
            }
            
            self.logger.info(f"Forced GC: {stats}")
            
            # Emit GC event
            await self.event_bus.emit(PerformanceThresholdExceeded(
                metric_name="manual_gc_triggered",
                current_value=collected,
                threshold=0
            ))
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error forcing garbage collection: {str(e)}")
            return {}
    
    async def shutdown(self) -> None:
        """Shutdown the memory profiler and cleanup resources."""
        try:
            # Stop profiling if running
            if self.is_running:
                await self.stop_profiling()
            
            # Cleanup backends
            if MemoryBackend.TRACEMALLOC in self.active_backends:
                if tracemalloc.is_tracing():
                    tracemalloc.stop()
            
            # Clear tracking data
            self.memory_snapshots.clear()
            self.component_profiles.clear()
            self.detected_leaks.clear()
            self.tracked_objects.clear()
            
            # Shutdown executor
            self.executor.shutdown(wait=True)
            
            self.logger.info("Memory profiler shutdown completed")
            
        except Exception as e:
            self.logger.error(f"Error during memory profiler shutdown: {str(e)}")


# Context managers for convenient profiling
@asynccontextmanager
async def memory_profiling_session(
    profiler: AdvancedMemoryProfiler,
    session_name: str = None,
    description: str = None,
    **kwargs
):
    """Context manager for memory profiling sessions."""
    session_id = await profiler.start_profiling(
        session_name=session_name,
        description=description,
        **kwargs
    )
    
    try:
        yield session_id
    finally:
        session = await profiler.stop_profiling()
        yield session


def memory_profile(
    profiler: AdvancedMemoryProfiler,
    session_name: str = None,
    description: str = None
):
    """Decorator for profiling function memory usage."""
    def decorator(func: F) -> F:
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                async with memory_profiling_session(
                    profiler, 
                    session_name=session_name or f"profile_{func.__name__}",
                    description=description or f"Memory profiling for {func.__name__}"
                ):
                    return await func(*args, **kwargs)
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                async def run_with_profiling():
                    async with memory_profiling_session(
                        profiler,
                        session_name=session_name or f"profile_{func.__name__}",
                        description=description or f"Memory profiling for {func.__name__}"
                    ):
                        return func(*args, **kwargs)
                
                return asyncio.run(run_with_profiling())
            return sync_wrapper
    
    return decorator