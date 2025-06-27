"""
Advanced Memory Profiling System
Author: Drmusab
Last Modified: 2025-05-28 11:32:14 UTC

This module provides comprehensive memory profiling for the AI assistant,
tracking memory usage, detecting leaks, providing snapshots, and monitoring
component-level consumption with minimal performance impact.
"""

from typing import Optional, Dict, Any, List, Set, Callable, Union, TypeVar
import asyncio
import threading
import time
import gc
import sys
import os
import tracemalloc
import psutil
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from enum import Enum
from contextlib import asynccontextmanager
import logging
import weakref
import json
import numpy as np
import torch
from pathlib import Path

# Core imports
from src.core.config.loader import ConfigLoader
from src.core.events.event_bus import EventBus
from src.core.events.event_types import (
    MemoryThresholdExceeded, MemoryLeakDetected, MemorySnapshotCreated,
    SystemStateChanged, ComponentHealthChanged, ErrorOccurred
)
from src.core.error_handling import ErrorHandler, handle_exceptions
from src.core.dependency_injection import Container
from src.core.health_check import HealthCheck

# Observability
from src.observability.monitoring.metrics import MetricsCollector
from src.observability.monitoring.tracing import TraceManager
from src.observability.logging.config import get_logger

# Type definitions
T = TypeVar('T')


class MemoryMetricType(Enum):
    """Types of memory metrics being tracked."""
    PHYSICAL = "physical"        # Physical RAM usage
    VIRTUAL = "virtual"          # Virtual memory usage
    SHARED = "shared"            # Shared memory usage
    PRIVATE = "private"          # Private memory usage
    HEAP = "heap"                # Python heap memory
    TENSOR = "tensor"            # Tensor memory (GPU/CPU)
    CACHED = "cached"            # Cached memory
    FRAGMENTATION = "fragmentation"  # Memory fragmentation ratio
    LEAK = "leak"                # Detected memory leak
    ALLOCATION_RATE = "allocation_rate"  # Memory allocation rate


class ProfilingLevel(Enum):
    """Levels of memory profiling detail."""
    BASIC = "basic"              # Basic system-level metrics
    COMPONENT = "component"      # Per-component memory tracking
    DETAILED = "detailed"        # Detailed object-level tracking
    COMPREHENSIVE = "comprehensive"  # Full memory snapshots and analysis
    MINIMAL = "minimal"          # Minimal impact monitoring


class SnapshotFormat(Enum):
    """Formats for memory snapshots."""
    TEXT = "text"                # Human-readable text format
    JSON = "json"                # JSON format
    FLAMEGRAPH = "flamegraph"    # Flamegraph compatible format
    HEAPMAP = "heapmap"          # Memory heap map
    DIFF = "diff"                # Diff against previous snapshot


@dataclass
class MemoryAllocation:
    """Represents a memory allocation event."""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    size_bytes: int = 0
    location: str = ""           # Code location/stack trace
    object_type: str = ""        # Type of allocated object
    component: Optional[str] = None  # Component responsible
    session_id: Optional[str] = None  # Associated session if any
    request_id: Optional[str] = None  # Associated request if any
    lifetime_seconds: Optional[float] = None  # Object lifetime if known
    allocation_id: Optional[str] = None  # Unique allocation identifier
    is_deallocated: bool = False  # Whether it has been deallocated
    is_leaked: bool = False     # Whether it's considered leaked
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MemorySnapshot:
    """Comprehensive memory snapshot."""
    snapshot_id: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    level: ProfilingLevel = ProfilingLevel.COMPONENT
    
    # System memory
    total_physical_mb: float = 0.0
    available_physical_mb: float = 0.0
    used_physical_mb: float = 0.0
    total_virtual_mb: float = 0.0
    used_virtual_mb: float = 0.0
    
    # Process memory
    process_rss_mb: float = 0.0
    process_vms_mb: float = 0.0
    process_shared_mb: float = 0.0
    process_text_mb: float = 0.0
    process_data_mb: float = 0.0
    process_lib_mb: float = 0.0
    
    # Python memory
    python_allocated_mb: float = 0.0
    python_peak_mb: float = 0.0
    python_gc_objects: int = 0
    python_gc_collections: List[int] = field(default_factory=lambda: [0, 0, 0])
    
    # Component memory
    component_memory: Dict[str, float] = field(default_factory=dict)
    
    # Tensor memory (if using frameworks)
    gpu_allocated_mb: float = 0.0
    gpu_cached_mb: float = 0.0
    gpu_reserved_mb: float = 0.0
    cpu_tensor_mb: float = 0.0
    
    # Tracemalloc data
    top_allocations: List[Dict[str, Any]] = field(default_factory=list)
    allocation_by_type: Dict[str, float] = field(default_factory=dict)
    allocation_by_file: Dict[str, float] = field(default_factory=dict)
    
    # Analysis
    potential_leaks: List[Dict[str, Any]] = field(default_factory=list)
    fragmentation_ratio: float = 0.0
    memory_growth_rate_mb_per_min: float = 0.0
    
    # Metadata
    snapshot_trigger: str = "manual"  # manual, scheduled, threshold, etc.
    snapshot_format: SnapshotFormat = SnapshotFormat.JSON
    snapshot_file: Optional[str] = None


class MemoryProfilerError(Exception):
    """Custom exception for memory profiler operations."""
    
    def __init__(self, message: str, component: Optional[str] = None, 
                 error_code: Optional[str] = None):
        super().__init__(message)
        self.component = component
        self.error_code = error_code
        self.timestamp = datetime.now(timezone.utc)


class MemoryProfiler:
    """
    Advanced Memory Profiling System for the AI Assistant.
    
    Features:
    - Comprehensive memory usage tracking across components
    - Memory leak detection with allocation tracking
    - Automatic snapshots and trend analysis
    - GPU and tensor memory monitoring
    - Component-level memory attribution
    - Low-overhead monitoring for production
    - Detailed profiling for development
    - Integration with metrics, tracing, and event systems
    - Automatic threshold-based alerts
    - Memory optimization suggestions
    """
    
    def __init__(self, container: Container):
        """
        Initialize the memory profiler.
        
        Args:
            container: Dependency injection container
        """
        self.container = container
        self.logger = get_logger(__name__)
        
        # Core services
        self.config = container.get(ConfigLoader)
        self.event_bus = container.get(EventBus)
        self.error_handler = container.get(ErrorHandler)
        self.health_check = container.get(HealthCheck)
        
        # Observability
        self.metrics = container.get(MetricsCollector)
        self.tracer = container.get(TraceManager)
        
        # Configuration
        self.profiling_level = ProfilingLevel(
            self.config.get("memory_profiler.level", "component")
        )
        self.snapshot_interval_seconds = self.config.get(
            "memory_profiler.snapshot_interval", 300
        )
        self.retention_days = self.config.get("memory_profiler.retention_days", 7)
        self.enable_leak_detection = self.config.get(
            "memory_profiler.enable_leak_detection", True
        )
        self.enable_gpu_monitoring = self.config.get(
            "memory_profiler.enable_gpu_monitoring", True
        )
        self.warning_threshold_percent = self.config.get(
            "memory_profiler.warning_threshold", 70
        )
        self.critical_threshold_percent = self.config.get(
            "memory_profiler.critical_threshold", 85
        )
        self.snapshot_dir = Path(self.config.get(
            "memory_profiler.snapshot_dir", "data/profiling/memory"
        ))
        
        # State management
        self.is_running = False
        self.snapshot_task: Optional[asyncio.Task] = None
        self.monitor_task: Optional[asyncio.Task] = None
        self.snapshots: List[MemorySnapshot] = []
        self.allocations: Dict[int, MemoryAllocation] = {}
        self.component_memory: Dict[str, float] = {}
        self.process = psutil.Process()
        
        # Track start time for growth calculations
        self.start_time = datetime.now(timezone.utc)
        self.start_memory_mb = 0.0
        
        # Initialize tracemalloc if detailed profiling
        if self.profiling_level in [ProfilingLevel.DETAILED, ProfilingLevel.COMPREHENSIVE]:
            tracemalloc.start(25)  # Capture 25 frames for detailed stack traces
        
        # Register metrics
        self._register_metrics()
        
        # Ensure snapshot directory exists
        if not self.snapshot_dir.exists():
            self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        
        # Register health check
        self.health_check.register_component("memory_profiler", self._health_check_callback)
        
        self.logger.info(f"MemoryProfiler initialized with level: {self.profiling_level.value}")

    def _register_metrics(self) -> None:
        """Register memory-related metrics with the metrics system."""
        try:
            # System memory metrics
            self.metrics.register_gauge("memory_system_total_mb")
            self.metrics.register_gauge("memory_system_available_mb")
            self.metrics.register_gauge("memory_system_used_mb")
            self.metrics.register_gauge("memory_system_used_percent")
            
            # Process memory metrics
            self.metrics.register_gauge("memory_process_rss_mb")
            self.metrics.register_gauge("memory_process_vms_mb")
            self.metrics.register_gauge("memory_python_allocated_mb")
            
            # Component memory metrics
            self.metrics.register_gauge("memory_component_used_mb")
            
            # GPU memory metrics if enabled
            if self.enable_gpu_monitoring:
                self.metrics.register_gauge("memory_gpu_allocated_mb")
                self.metrics.register_gauge("memory_gpu_cached_mb")
                self.metrics.register_gauge("memory_gpu_utilization_percent")
            
            # Leak detection metrics
            if self.enable_leak_detection:
                self.metrics.register_counter("memory_leaks_detected")
                self.metrics.register_gauge("memory_leaked_mb")
            
            # Allocation metrics
            self.metrics.register_histogram("memory_allocation_size_bytes")
            self.metrics.register_counter("memory_allocations_total")
            
            self.logger.debug("Memory metrics registered successfully")
            
        except Exception as e:
            self.logger.warning(f"Failed to register memory metrics: {str(e)}")

    async def start(self) -> None:
        """Start the memory profiler."""
        if self.is_running:
            self.logger.warning("Memory profiler is already running")
            return
        
        try:
            self.is_running = True
            
            # Collect initial memory state
            self.start_memory_mb = self.process.memory_info().rss / (1024 * 1024)
            
            # Start background tasks
            self.monitor_task = asyncio.create_task(self._memory_monitor_loop())
            self.snapshot_task = asyncio.create_task(self._snapshot_loop())
            
            # Register event handlers
            self._register_event_handlers()
            
            # Take initial snapshot
            await self.take_snapshot("startup")
            
            self.logger.info("Memory profiler started successfully")
            
        except Exception as e:
            self.is_running = False
            self.logger.error(f"Failed to start memory profiler: {str(e)}")
            raise MemoryProfilerError(f"Failed to start: {str(e)}")

    async def stop(self) -> None:
        """Stop the memory profiler."""
        if not self.is_running:
            return
        
        try:
            self.is_running = False
            
            # Cancel background tasks
            if self.monitor_task:
                self.monitor_task.cancel()
            if self.snapshot_task:
                self.snapshot_task.cancel()
            
            # Take final snapshot
            await self.take_snapshot("shutdown")
            
            # Stop tracemalloc
            if tracemalloc.is_tracing():
                tracemalloc.stop()
            
            self.logger.info("Memory profiler stopped successfully")
            
        except Exception as e:
            self.logger.error(f"Error stopping memory profiler: {str(e)}")
            raise MemoryProfilerError(f"Failed to stop: {str(e)}")

    def _register_event_handlers(self) -> None:
        """Register event handlers for system events."""
        # Component lifecycle events for memory tracking
        self.event_bus.subscribe("component_initialized", self._handle_component_initialized)
        self.event_bus.subscribe("component_stopped", self._handle_component_stopped)
        
        # Session lifecycle events
        self.event_bus.subscribe("session_started", self._handle_session_started)
        self.event_bus.subscribe("session_ended", self._handle_session_ended)
        
        # Processing events
        self.event_bus.subscribe("processing_started", self._handle_processing_started)
        self.event_bus.subscribe("processing_completed", self._handle_processing_completed)
        
        # Error events
        self.event_bus.subscribe("error_occurred", self._handle_error)
        
        # System events
        self.event_bus.subscribe("system_shutdown_started", self._handle_system_shutdown)

    @handle_exceptions
    async def take_snapshot(self, trigger: str = "manual") -> MemorySnapshot:
        """
        Take a comprehensive memory snapshot.
        
        Args:
            trigger: What triggered this snapshot
            
        Returns:
            Memory snapshot
        """
        snapshot_id = f"snapshot_{int(time.time())}_{trigger}"
        
        try:
            with self.tracer.trace("memory_snapshot") as span:
                span.set_attributes({
                    "snapshot_id": snapshot_id,
                    "trigger": trigger,
                    "profiling_level": self.profiling_level.value
                })
                
                snapshot = MemorySnapshot(
                    snapshot_id=snapshot_id,
                    level=self.profiling_level,
                    snapshot_trigger=trigger
                )
                
                # Collect system memory info
                system_memory = psutil.virtual_memory()
                snapshot.total_physical_mb = system_memory.total / (1024 * 1024)
                snapshot.available_physical_mb = system_memory.available / (1024 * 1024)
                snapshot.used_physical_mb = system_memory.used / (1024 * 1024)
                
                # Collect process memory
                process_memory = self.process.memory_info()
                snapshot.process_rss_mb = process_memory.rss / (1024 * 1024)
                snapshot.process_vms_mb = process_memory.vms / (1024 * 1024)
                if hasattr(process_memory, 'shared'):
                    snapshot.process_shared_mb = process_memory.shared / (1024 * 1024)
                
                # Python memory info
                snapshot.python_gc_objects = sum(gc.get_count())
                snapshot.python_gc_collections = list(gc.get_count())
                
                # Component memory
                snapshot.component_memory = dict(self.component_memory)
                
                # Tensor memory if enabled and available
                if self.enable_gpu_monitoring and torch.cuda.is_available():
                    snapshot.gpu_allocated_mb = torch.cuda.memory_allocated() / (1024 * 1024)
                    snapshot.gpu_cached_mb = torch.cuda.memory_reserved() / (1024 * 1024)
                    snapshot.gpu_reserved_mb = torch.cuda.max_memory_reserved() / (1024 * 1024)
                
                # Detailed profiling with tracemalloc
                if self.profiling_level in [ProfilingLevel.DETAILED, ProfilingLevel.COMPREHENSIVE] and tracemalloc.is_tracing():
                    await self._add_tracemalloc_data(snapshot)
                
                # Calculate growth rate
                time_diff_minutes = (datetime.now(timezone.utc) - self.start_time).total_seconds() / 60
                if time_diff_minutes > 0:
                    memory_diff_mb = snapshot.process_rss_mb - self.start_memory_mb
                    snapshot.memory_growth_rate_mb_per_min = memory_diff_mb / time_diff_minutes
                
                # Add potential leaks
                if self.enable_leak_detection and len(self.allocations) > 0:
                    potential_leaks = self._detect_memory_leaks()
                    snapshot.potential_leaks = [
                        {
                            "allocation_id": str(allocation.allocation_id),
                            "size_mb": allocation.size_bytes / (1024 * 1024),
                            "object_type": allocation.object_type,
                            "component": allocation.component or "unknown",
                            "lifetime_seconds": (datetime.now(timezone.utc) - allocation.timestamp).total_seconds(),
                            "location": allocation.location
                        }
                        for allocation in potential_leaks
                    ]
                
                # Save snapshot
                await self._save_snapshot(snapshot)
                
                # Add to snapshots list
                self.snapshots.append(snapshot)
                
                # Limit number of in-memory snapshots
                if len(self.snapshots) > 100:  # Keep only the most recent 100 snapshots in memory
                    self.snapshots.pop(0)
                
                # Emit event
                await self.event_bus.emit(MemorySnapshotCreated(
                    snapshot_id=snapshot_id,
                    timestamp=snapshot.timestamp,
                    trigger=trigger,
                    memory_mb=snapshot.process_rss_mb,
                    growth_rate=snapshot.memory_growth_rate_mb_per_min
                ))
                
                self.logger.info(f"Memory snapshot {snapshot_id} created")
                
                return snapshot
                
        except Exception as e:
            self.logger.error(f"Failed to take memory snapshot: {str(e)}")
            raise MemoryProfilerError(f"Failed to take snapshot: {str(e)}")

    async def _add_tracemalloc_data(self, snapshot: MemorySnapshot) -> None:
        """Add tracemalloc data to the snapshot."""
        # Get current tracemalloc snapshot
        tracemalloc_snapshot = tracemalloc.take_snapshot()
        
        # Get top statistics
        top_stats = tracemalloc_snapshot.statistics('lineno')
        
        # Add top allocations
        snapshot.top_allocations = [
            {
                "size_mb": stat.size / (1024 * 1024),
                "count": stat.count,
                "file": str(stat.traceback[0].filename),
                "line": stat.traceback[0].lineno,
                "trace": "\n".join(str(frame) for frame in stat.traceback)
            }
            for stat in top_stats[:50]  # Top 50 allocations
        ]
        
        # Group by type
        allocation_by_type = {}
        for stat in top_stats:
            file_path = stat.traceback[0].filename
            file_name = os.path.basename(file_path)
            if "." in file_name:
                file_type = file_name.split(".")[-1]
            else:
                file_type = "unknown"
            
            if file_type not in allocation_by_type:
                allocation_by_type[file_type] = 0
            allocation_by_type[file_type] += stat.size / (1024 * 1024)
        
        snapshot.allocation_by_type = allocation_by_type
        
        # Group by file
        allocation_by_file = {}
        for stat in top_stats:
            file_path = stat.traceback[0].filename
            file_name = os.path.basename(file_path)
            
            if file_name not in allocation_by_file:
                allocation_by_file[file_name] = 0
            allocation_by_file[file_name] += stat.size / (1024 * 1024)
        
        snapshot.allocation_by_file = allocation_by_file

    async def _save_snapshot(self, snapshot: MemorySnapshot) -> None:
        """Save snapshot to file."""
        try:
            snapshot_file = self.snapshot_dir / f"{snapshot.snapshot_id}.json"
            
            # Serialize snapshot
            snapshot_data = {
                "snapshot_id": snapshot.snapshot_id,
                "timestamp": snapshot.timestamp.isoformat(),
                "level": snapshot.level.value,
                "trigger": snapshot.snapshot_trigger,
                
                # System memory
                "system": {
                    "total_physical_mb": snapshot.total_physical_mb,
                    "available_physical_mb": snapshot.available_physical_mb,
                    "used_physical_mb": snapshot.used_physical_mb,
                    "total_virtual_mb": snapshot.total_virtual_mb,
                    "used_virtual_mb": snapshot.used_virtual_mb,
                },
                
                # Process memory
                "process": {
                    "rss_mb": snapshot.process_rss_mb,
                    "vms_mb": snapshot.process_vms_mb,
                    "shared_mb": snapshot.process_shared_mb,
                    "text_mb": snapshot.process_text_mb,
                    "data_mb": snapshot.process_data_mb,
                    "lib_mb": snapshot.process_lib_mb,
                },
                
                # Python memory
                "python": {
                    "allocated_mb": snapshot.python_allocated_mb,
                    "peak_mb": snapshot.python_peak_mb,
                    "gc_objects": snapshot.python_gc_objects,
                    "gc_collections": snapshot.python_gc_collections,
                },
                
                # Component memory
                "components": snapshot.component_memory,
                
                # Tensor memory
                "tensors": {
                    "gpu_allocated_mb": snapshot.gpu_allocated_mb,
                    "gpu_cached_mb": snapshot.gpu_cached_mb,
                    "gpu_reserved_mb": snapshot.gpu_reserved_mb,
                    "cpu_tensor_mb": snapshot.cpu_tensor_mb,
                },
                
                # Tracemalloc data
                "allocations": {
                    "top": snapshot.top_allocations,
                    "by_type": snapshot.allocation_by_type,
                    "by_file": snapshot.allocation_by_file,
                },
                
                # Analysis
                "analysis": {
                    "potential_leaks": snapshot.potential_leaks,
                    "fragmentation_ratio": snapshot.fragmentation_ratio,
                    "memory_growth_rate_mb_per_min": snapshot.memory_growth_rate_mb_per_min,
                }
            }
            
            # Write to file
            with open(snapshot_file, 'w') as f:
                json.dump(snapshot_data, f, indent=2)
            
            snapshot.snapshot_file = str(snapshot_file)
            
        except Exception as e:
            self.logger.error(f"Failed to save snapshot: {str(e)}")

    @handle_exceptions
    async def get_memory_usage(
        self,
        component: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get current memory usage information.
        
        Args:
            component: Optional component name to get specific memory info
            
        Returns:
            Memory usage information
        """
        try:
            # Get system memory
            system_memory = psutil.virtual_memory()
            
            # Get process memory
            process_memory = self.process.memory_info()
            
            # Build basic response
            memory_info = {
                "system": {
                    "total_mb": system_memory.total / (1024 * 1024),
                    "available_mb": system_memory.available / (1024 * 1024),
                    "used_mb": system_memory.used / (1024 * 1024),
                    "percent": system_memory.percent
                },
                "process": {
                    "rss_mb": process_memory.rss / (1024 * 1024),
                    "vms_mb": process_memory.vms / (1024 * 1024),
                    "percent": self.process.memory_percent()
                },
                "python": {
                    "gc_objects": sum(gc.get_count())
                },
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            # Add component-specific info if requested
            if component:
                if component in self.component_memory:
                    memory_info["component"] = {
                        "name": component,
                        "memory_mb": self.component_memory[component]
                    }
                else:
                    memory_info["component"] = {
                        "name": component,
                        "memory_mb": 0.0,
                        "error": "Component not found"
                    }
            else:
                memory_info["components"] = dict(self.component_memory)
            
            # Add GPU memory if available
            if self.enable_gpu_monitoring and torch.cuda.is_available():
                memory_info["gpu"] = {
                    "allocated_mb": torch.cuda.memory_allocated() / (1024 * 1024),
                    "cached_mb": torch.cuda.memory_reserved() / (1024 * 1024),
                    "device_count": torch.cuda.device_count()
                }
                
                # Per-device info
                memory_info["gpu"]["devices"] = []
                for i in range(torch.cuda.device_count()):
                    device_info = {
                        "index": i,
                        "name": torch.cuda.get_device_name(i),
                        "allocated_mb": torch.cuda.memory_allocated(i) / (1024 * 1024),
                        "cached_mb": torch.cuda.memory_reserved(i) / (1024 * 1024)
                    }
                    memory_info["gpu"]["devices"].append(device_info)
            
            # Add trend data
            if len(self.snapshots) >= 2:
                oldest = self.snapshots[0]
                newest = self.snapshots[-1]
                time_diff_minutes = (newest.timestamp - oldest.timestamp).total_seconds() / 60
                
                if time_diff_minutes > 0:
                    memory_diff = newest.process_rss_mb - oldest.process_rss_mb
                    growth_rate = memory_diff / time_diff_minutes
                    
                    memory_info["trends"] = {
                        "window_minutes": time_diff_minutes,
                        "growth_rate_mb_per_min": growth_rate,
                        "is_growing": growth_rate > 0.1,  # Consider > 0.1 MB/min as growth
                        "snapshot_count": len(self.snapshots)
                    }
            
            return memory_info
            
        except Exception as e:
            self.logger.error(f"Failed to get memory usage: {str(e)}")
            raise MemoryProfilerError(f"Failed to get memory usage: {str(e)}")

    @handle_exceptions
    async def track_component_memory(
        self,
        component_name: str,
        memory_mb: float
    ) -> None:
        """
        Track memory usage for a specific component.
        
        Args:
            component_name: Name of the component
            memory_mb: Memory usage in MB
        """
        self.component_memory[component_name] = memory_mb
        
        # Update component memory metric
        self.metrics.set("memory_component_used_mb", memory_mb, 
                       tags={"component": component_name})

    @handle_exceptions
    async def track_allocation(
        self,
        size_bytes: int,
        object_type: str,
        component: Optional[str] = None,
        session_id: Optional[str] = None,
        location: Optional[str] = None
    ) -> int:
        """
        Track a memory allocation.
        
        Args:
            size_bytes: Size of allocation in bytes
            object_type: Type of allocated object
            component: Component making the allocation
            session_id: Session ID if applicable
            location: Code location string
            
        Returns:
            Allocation ID
        """
        if not self.enable_leak_detection:
            return -1
        
        # Create allocation record
        allocation_id = len(self.allocations) + 1
        allocation = MemoryAllocation(
            size_bytes=size_bytes,
            object_type=object_type,
            component=component,
            session_id=session_id,
            location=location or self._get_caller_location(),
            allocation_id=allocation_id
        )
        
        # Store allocation
        self.allocations[allocation_id] = allocation
        
        # Update metrics
        self.metrics.increment("memory_allocations_total")
        self.metrics.record("memory_allocation_size_bytes", size_bytes)
        
        # Clean up old allocations periodically
        if len(self.allocations) > 10000:
            await self._cleanup_old_allocations()
        
        return allocation_id

    @handle_exceptions
    async def track_deallocation(self, allocation_id: int) -> None:
        """
        Track a memory deallocation.
        
        Args:
            allocation_id: ID of the allocation being freed
        """
        if not self.enable_leak_detection or allocation_id not in self.allocations:
            return
        
        allocation = self.allocations[allocation_id]
        allocation.is_deallocated = True
        allocation.lifetime_seconds = (
            datetime.now(timezone.utc) - allocation.timestamp
        ).total_seconds()

    def _get_caller_location(self) -> str:
        """Get the caller's code location."""
        stack = traceback.extract_stack()
        
        # Skip the current frame and the immediate caller (track_allocation)
        frame = stack[-3]
        return f"{frame.filename}:{frame.lineno} in {frame.name}"

    def _detect_memory_leaks(self) -> List[MemoryAllocation]:
        """Detect potential memory leaks based on allocation patterns."""
        potential_leaks = []
        now = datetime.now(timezone.utc)
        
        # Check for long-lived, non-deallocated allocations
        for allocation in self.allocations.values():
            if allocation.is_deallocated:
                continue
            
            lifetime_seconds = (now - allocation.timestamp).total_seconds()
            
            # Consider as potential leak if:
            # 1. Large allocation (>1MB)
            # 2. Long-lived (>10 minutes)
            # 3. Not deallocated
            if (
                allocation.size_bytes > 1024 * 1024 and 
                lifetime_seconds > 600 and 
                not allocation.is_leaked
            ):
                allocation.is_leaked = True
                potential_leaks.append(allocation)
        
        # If we found potential leaks, emit an event
        if potential_leaks:
            leaked_bytes = sum(a.size_bytes for a in potential_leaks)
            leaked_mb = leaked_bytes / (1024 * 1024)
            
            asyncio.create_task(self.event_bus.emit(MemoryLeakDetected(
                leak_count=len(potential_leaks),
                leaked_bytes=leaked_bytes,
                component=potential_leaks[0].component if potential_leaks else None,
                session_id=potential_leaks[0].session_id if potential_leaks else None
            )))
            
            self.metrics.increment("memory_leaks_detected", count=len(potential_leaks))
            self.metrics.set("memory_leaked_mb", leaked_mb)
            
            self.logger.warning(
                f"Detected {len(potential_leaks)} potential memory leaks totaling {leaked_mb:.2f} MB"
            )
        
        return potential_leaks

    async def _cleanup_old_allocations(self) -> None:
        """Clean up old allocation records to prevent memory buildup."""
        now = datetime.now(timezone.utc)
        to_remove = []
        
        # Find deallocated allocations older than 1 hour
        for allocation_id, allocation in self.allocations.items():
            if (
                allocation.is_deallocated and 
                (now - allocation.timestamp).total_seconds() > 3600
            ):
                to_remove.append(allocation_id)
        
        # Remove them
        for allocation_id in to_remove:
            del self.allocations[allocation_id]
        
        self.logger.debug(f"Cleaned up {len(to_remove)} old allocation records")

    async def _cleanup_old_snapshots(self) -> None:
        """Clean up old snapshot files."""
        try:
            retention_time = datetime.now(timezone.utc) - timedelta(days=self.retention_days)
            retention_timestamp = int(retention_time.timestamp())
            
            # List snapshot files
            snapshot_files = list(self.snapshot_dir.glob("*.json"))
            
            # Identify old files
            for snapshot_file in snapshot_files:
                try:
                    # Extract timestamp from filename
                    filename = snapshot_file.name
                    if "_" in filename:
                        timestamp_str = filename.split("_")[1]
                        timestamp = int(timestamp_str)
                        
                        if timestamp < retention_timestamp:
                            snapshot_file.unlink()
                except Exception:
                    continue
            
            self.logger.debug("Cleaned up old snapshot files")
            
        except Exception as e:
            self.logger.warning(f"Failed to clean up old snapshots: {str(e)}")

    async def _memory_monitor_loop(self) -> None:
        """Background task for continuous memory monitoring."""
        try:
            while self.is_running:
                try:
                    # Get current memory usage
                    system_memory = psutil.virtual_memory()
                    process_memory = self.process.memory_info()
                    
                    # Update metrics
                    self.metrics.set("memory_system_total_mb", system_memory.total / (1024 * 1024))
                    self.metrics.set("memory_system_available_mb", system_memory.available / (1024 * 1024))
                    self.metrics.set("memory_system_used_mb", system_memory.used / (1024 * 1024))
                    self.metrics.set("memory_system_used_percent", system_memory.percent)
                    
                    self.metrics.set("memory_process_rss_mb", process_memory.rss / (1024 * 1024))
                    self.metrics.set("memory_process_vms_mb", process_memory.vms / (1024 * 1024))
                    
                    # Check for GPU memory if enabled
                    if self.enable_gpu_monitoring and torch.cuda.is_available():
                        gpu_allocated = torch.cuda.memory_allocated() / (1024 * 1024)
                        gpu_cached = torch.cuda.memory_reserved() / (1024 * 1024)
                        
                        self.metrics.set("memory_gpu_allocated_mb", gpu_allocated)
                        self.metrics.set("memory_gpu_cached_mb", gpu_cached)
                    
                    # Check for memory thresholds
                    if system_memory.percent >= self.critical_threshold_percent:
                        await self.event_bus.emit(MemoryThresholdExceeded(
                            threshold_type="critical",
                            current_percent=system_memory.percent,
                            threshold_percent=self.critical_threshold_percent,
                            available_mb=system_memory.available / (1024 * 1024)
                        ))
                        
                        # Take snapshot on critical threshold
                        await self.take_snapshot("critical_threshold")
                        
                    elif system_memory.percent >= self.warning_threshold_percent:
                        await self.event_bus.emit(MemoryThresholdExceeded(
                            threshold_type="warning",
                            current_percent=system_memory.percent,
                            threshold_percent=self.warning_threshold_percent,
                            available_mb=system_memory.available / (1024 * 1024)
                        ))
                    
                    # Look for memory leaks
                    if self.enable_leak_detection:
                        self._detect_memory_leaks()
                    
                    # Sleep interval - shorter at higher memory usage
                    if system_memory.percent >= self.warning_threshold_percent:
                        await asyncio.sleep(5)  # Monitor more frequently when memory usage is high
                    else:
                        await asyncio.sleep(15)
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    self.logger.error(f"Error in memory monitor loop: {str(e)}")
                    await asyncio.sleep(30)
                    
        except asyncio.CancelledError:
            self.logger.info("Memory monitor loop cancelled")
        except Exception as e:
            self.logger.error(f"Memory monitor loop failed: {str(e)}")

    async def _snapshot_loop(self) -> None:
        """Background task for periodic memory snapshots."""
        try:
            # Wait a bit before first snapshot
            await asyncio.sleep(30)
            
            while self.is_running:
                try:
                    # Take snapshot
                    await self.take_snapshot("scheduled")
                    
                    # Cleanup old snapshots
                    await self._cleanup_old_snapshots()
                    
                    # Sleep until next snapshot
                    await asyncio.sleep(self.snapshot_interval_seconds)
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    self.logger.error(f"Error in snapshot loop: {str(e)}")
                    await asyncio.sleep(60)  # Retry after a minute
                    
        except asyncio.CancelledError:
            self.logger.info("Snapshot loop cancelled")
        except Exception as e:
            self.logger.error(f"Snapshot loop failed: {str(e)}")

    @asynccontextmanager
    async def profile_scope(
        self,
        name: str,
        component: Optional[str] = None,
        session_id: Optional[str] = None
    ):
        """
        Context manager for profiling a scope of code.
        
        Args:
            name: Name of the profiling scope
            component: Component name
            session_id: Session ID if applicable
        """
        start_time = time.time()
        initial_memory = psutil.Process().memory_info().rss
        
        # Start the scope
        with self.tracer.trace(f"memory_profile_{name}") as span:
            span.set_attributes({
                "component": component or "unknown",
                "session_id": session_id or "unknown"
            })
            
            try:
                # Execute the scope
                yield
            finally:
                # Calculate memory change
                final_memory = psutil.Process().memory_info().rss
                memory_change = final_memory - initial_memory
                elapsed_time = time.time() - start_time
                
                # Record metrics
                if memory_change > 0:
                    self.metrics.record("memory_allocation_size_bytes", memory_change)
                
                # Track the allocation if significant
                if memory_change > 1024 * 1024:  # Only track allocations > 1MB
                    await self.track_allocation(
                        size_bytes=memory_change,
                        object_type=name,
                        component=component,
                        session_id=session_id
                    )
                
                # Update span
                span.set_attributes({
                    "memory_change_bytes": memory_change,
                    "execution_time_seconds": elapsed_time
                })

    async def _handle_component_initialized(self, event) -> None:
        """Handle component initialized events."""
        # Start tracking component memory
        self.component_memory[event.component_id] = 0.0

    async def _handle_component_stopped(self, event) -> None:
        """Handle component stopped events."""
        # Clean up component memory tracking
        if event.component_id in self.component_memory:
            del self.component_memory[event.component_id]

    async def _handle_session_started(self, event) -> None:
        """Handle session started events."""
        # Track session memory baseline
        baseline = psutil.Process().memory_info().rss
        
        # Store in working memory
        self.component_memory[f"session_{event.session_id}"] = 0.0

    async def _handle_session_ended(self, event) -> None:
        """Handle session ended events."""
        # Clean up session memory tracking
        if f"session_{event.session_id}" in self.component_memory:
            del self.component_memory[f"session_{event.session_id}"]

    async def _handle_processing_started(self, event) -> None:
        """Handle processing started events."""
        # Take snapshot on significant processing operations
        if self.profiling_level == ProfilingLevel.COMPREHENSIVE:
            await self.take_snapshot(f"processing_{event.request_id}")

    async def _handle_processing_completed(self, event) -> None:
        """Handle processing completed events."""
        pass

    async def _handle_error(self, event) -> None:
        """Handle error events."""
        # Take snapshot on critical errors
        if event.severity == "critical":
            await self.take_snapshot(f"error_{event.error_type}")

    async def _handle_system_shutdown(self, event) -> None:
        """Handle system shutdown events."""
        # Take final snapshot and stop
        await self.take_snapshot("system_shutdown")
        await self.stop()

    async def _health_check_callback(self) -> Dict[str, Any]:
        """Health check callback for the memory profiler."""
        try:
            system_memory = psutil.virtual_memory()
            process_memory = self.process.memory_info()
            
            is_healthy = system_memory.percent < self.warning_threshold_percent
            
            return {
                "status": "healthy" if is_healthy else "degraded",
                "system_memory_percent": system_memory.percent,
                "process_memory_mb": process_memory.rss / (1024 * 1024),
                "warning_threshold": self.warning_threshold_percent,
                "critical_threshold": self.critical_threshold_percent,
                "snapshot_count": len(self.snapshots),
                "profiling_level": self.profiling_level.value,
                "leak_detection_enabled": self.enable_leak_detection
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }

    @handle_exceptions
    def get_memory_optimization_suggestions(self) -> List[Dict[str, Any]]:
        """
        Get memory optimization suggestions based on profiling data.
        
        Returns:
            List of optimization suggestions
        """
        suggestions = []
        
        # Only provide suggestions if we have enough data
        if len(self.snapshots) < 2:
            return [{"message": "Not enough snapshot data for optimization suggestions"}]
        
        latest_snapshot = self.snapshots[-1]
        
        # Check overall memory growth
        if latest_snapshot.memory_growth_rate_mb_per_min > 0.5:
            suggestions.append({
                "type": "growth",
                "severity": "high" if latest_snapshot.memory_growth_rate_mb_per_min > 2.0 else "medium",
                "message": f"Memory is growing at {latest_snapshot.memory_growth_rate_mb_per_min:.2f} MB/minute",
                "suggestion": "Check for memory leaks or resource cleanup issues"
            })
        
        # Check component memory distribution
        if latest_snapshot.component_memory:
            total_component_mb = sum(latest_snapshot.component_memory.values())
            
            for component, memory_mb in latest_snapshot.component_memory.items():
                if memory_mb > 100 and memory_mb / total_component_mb > 0.3:
                    suggestions.append({
                        "type": "component",
                        "severity": "medium",
                        "component": component,
                        "message": f"Component {component} is using {memory_mb:.2f} MB ({memory_mb/total_component_mb*100:.1f}% of tracked memory)",
                        "suggestion": "Consider optimizing memory usage in this component"
                    })
        
        # Check for GPU memory optimization if applicable
        if self.enable_gpu_monitoring and latest_snapshot.gpu_allocated_mb > 0:
            if latest_snapshot.gpu_cached_mb > 2 * latest_snapshot.gpu_allocated_mb:
                suggestions.append({
                    "type": "gpu",
                    "severity": "medium",
                    "message": f"GPU has {latest_snapshot.gpu_cached_mb:.2f} MB cached but only {latest_snapshot.gpu_allocated_mb:.2f} MB allocated",
                    "suggestion": "Consider using torch.cuda.empty_cache() to free unused GPU memory"
                })
        
        # Check for potential leaks
        if latest_snapshot.potential_leaks:
            leak_types = {}
            for leak in latest_snapshot.potential_leaks:
                leak_type = leak.get("object_type", "unknown")
                if leak_type not in leak_types:
                    leak_types[leak_type] = 0
                leak_types[leak_type] += 1
            
            for leak_type, count in leak_types.items():
                suggestions.append({
                    "type": "leak",
                    "severity": "high",
                    "message": f"Detected {count} potential memory leaks of type {leak_type}",
                    "suggestion": "Check for unclosed resources or circular references"
                })
        
        # If using Python objects heavily
        if latest_snapshot.python_gc_objects > 1000000:
            suggestions.append({
                "type": "gc",
                "severity": "medium",
                "message": f"High number of Python objects: {latest_snapshot.python_gc_objects}",
                "suggestion": "Consider manually triggering garbage collection (gc.collect())"
            })
        
        # If no specific issues found but memory usage is high
        if (
            not suggestions and 
            latest_snapshot.process_rss_mb > 0.7 * latest_snapshot.total_physical_mb
        ):
            suggestions.append({
                "type": "general",
                "severity": "medium",
                "message": f"High overall memory usage: {latest_snapshot.process_rss_mb:.2f} MB",
                "suggestion": "Consider implementing memory limits or scaling resources"
            })
        
        return suggestions

    def __del__(self):
        """Destructor to ensure proper cleanup."""
        try:
            # Stop tracemalloc if running
            if tracemalloc.is_tracing():
                tracemalloc.stop()
        except Exception:
            pass  # Ignore cleanup errors in destructor
