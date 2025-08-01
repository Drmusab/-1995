"""
Advanced Cache Strategy System
Author: Drmusab
Last Modified: 2025-01-18 09:56:47 UTC

This module provides comprehensive caching strategies for the AI assistant,
including multi-level caching, intelligent cache policies, performance optimization,
and seamless integration with all core system components.
"""

import base64
import gc
import hashlib
import json
import logging
import pickle
import sys
import threading
import time
import uuid
import weakref
import zlib
from abc import ABC, abstractmethod
from collections import OrderedDict, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    AsyncGenerator,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Set,
    Type,
    TypeVar,
    Union,
)

import asyncio
import psutil

# Core imports
from src.core.config.loader import ConfigLoader
from src.core.dependency_injection import Container
from src.core.error_handling import ErrorHandler, handle_exceptions
from src.core.events.event_bus import EventBus
from src.core.events.event_types import (
    CacheCleanup,
    CacheCoherenceEvent,
    CacheEviction,
    CacheHealthCheck,
    CacheHit,
    CacheInvalidation,
    CacheMiss,
    CacheWarming,
    ComponentHealthChanged,
    ErrorOccurred,
    MemoryPressureAlert,
    ProcessingCompleted,
    SessionEnded,
    SessionStarted,
    SystemPerformanceEvent,
    WorkflowCompleted,
)
from src.core.health_check import HealthCheck
from src.integrations.cache.local_cache import LocalCache

# Cache backends
from src.integrations.cache.redis_cache import RedisCache
from src.observability.logging.config import get_logger

# Observability
from src.observability.monitoring.metrics import MetricsCollector
from src.observability.monitoring.tracing import TraceManager

# Type definitions
T = TypeVar("T")


class CacheLevel(Enum):
    """Cache levels in the hierarchy."""

    L1_MEMORY = "l1_memory"  # In-process memory cache
    L2_REDIS = "l2_redis"  # Redis cache
    L3_DISK = "l3_disk"  # Disk-based cache
    L4_NETWORK = "l4_network"  # Network/CDN cache


class CacheStrategy(Enum):
    """Cache eviction and management strategies."""

    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    FIFO = "fifo"  # First In First Out
    TTL = "ttl"  # Time To Live
    ADAPTIVE = "adaptive"  # Adaptive based on access patterns
    CONTENT_AWARE = "content_aware"  # Based on content importance
    HYBRID = "hybrid"  # Combination of strategies


class CacheOperation(Enum):
    """Types of cache operations."""

    GET = "get"
    SET = "set"
    DELETE = "delete"
    INVALIDATE = "invalidate"
    WARM = "warm"
    CLEANUP = "cleanup"
    EVICT = "evict"


class CachePriority(Enum):
    """Cache priority levels."""

    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3
    PERMANENT = 4  # Never evict


class CacheCoherenceMode(Enum):
    """Cache coherence modes for distributed caching."""

    STRONG = "strong"  # Strong consistency
    EVENTUAL = "eventual"  # Eventual consistency
    WEAK = "weak"  # Weak consistency
    SESSION = "session"  # Session-based consistency


@dataclass
class CacheEntry:
    """Represents a cache entry with metadata."""

    key: str
    value: Any
    ttl: Optional[float] = None
    priority: CachePriority = CachePriority.NORMAL

    # Timestamps
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_accessed: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_modified: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Access tracking
    access_count: int = 0
    access_frequency: float = 0.0

    # Size and cost
    size_bytes: int = 0
    computation_cost: float = 0.0

    # Content metadata
    content_type: Optional[str] = None
    content_hash: Optional[str] = None
    compression_ratio: float = 1.0

    # Dependencies and invalidation
    dependencies: Set[str] = field(default_factory=set)
    invalidation_tags: Set[str] = field(default_factory=set)

    # Version and consistency
    version: int = 1
    etag: Optional[str] = None

    # Custom metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CachePolicy:
    """Cache policy configuration."""

    name: str
    strategy: CacheStrategy = CacheStrategy.LRU
    max_size: int = 1000
    max_memory_mb: float = 100.0
    default_ttl: Optional[float] = 3600.0  # 1 hour

    # Performance settings
    enable_compression: bool = True
    compression_threshold: int = 1024  # Compress if larger than 1KB
    enable_serialization: bool = True

    # Eviction settings
    eviction_batch_size: int = 10
    eviction_threshold: float = 0.8  # Start eviction at 80% capacity
    low_memory_threshold: float = 0.9  # Emergency eviction at 90%

    # Preloading and warming
    enable_warming: bool = True
    warming_patterns: List[str] = field(default_factory=list)
    preload_on_startup: bool = False

    # Coherence settings
    coherence_mode: CacheCoherenceMode = CacheCoherenceMode.EVENTUAL
    sync_interval: float = 30.0

    # Monitoring
    enable_monitoring: bool = True
    enable_detailed_metrics: bool = False

    # Custom configurations
    custom_settings: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CacheStatistics:
    """Cache performance statistics."""

    # Basic metrics
    hit_count: int = 0
    miss_count: int = 0
    eviction_count: int = 0

    # Derived metrics
    hit_rate: float = 0.0
    miss_rate: float = 0.0

    # Size and memory
    total_entries: int = 0
    total_size_bytes: int = 0
    memory_usage_mb: float = 0.0

    # Performance metrics
    avg_get_time_ms: float = 0.0
    avg_set_time_ms: float = 0.0
    throughput_ops_per_sec: float = 0.0

    # Cost savings
    computation_time_saved: float = 0.0
    cost_savings_percentage: float = 0.0

    # Last updated
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class CacheError(Exception):
    """Custom exception for cache operations."""

    def __init__(
        self,
        message: str,
        cache_key: Optional[str] = None,
        operation: Optional[str] = None,
        error_code: Optional[str] = None,
    ):
        super().__init__(message)
        self.cache_key = cache_key
        self.operation = operation
        self.error_code = error_code
        self.timestamp = datetime.now(timezone.utc)


class CacheBackend(Protocol):
    """Protocol for cache backend implementations."""

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        ...

    async def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Set value in cache."""
        ...

    async def delete(self, key: str) -> bool:
        """Delete value from cache."""
        ...

    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        ...

    async def clear(self) -> None:
        """Clear all cache entries."""
        ...

    async def keys(self, pattern: str = "*") -> List[str]:
        """Get keys matching pattern."""
        ...


class CacheSerializer:
    """Handles serialization and compression of cache values."""

    def __init__(self, enable_compression: bool = True, compression_threshold: int = 1024):
        self.enable_compression = enable_compression
        self.compression_threshold = compression_threshold
        self.logger = get_logger(__name__)

    def serialize(self, value: Any) -> bytes:
        """Serialize value to bytes."""
        try:
            # Pickle the value
            data = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)

            # Compress if enabled and data is large enough
            if self.enable_compression and len(data) > self.compression_threshold:
                compressed = zlib.compress(data, level=6)
                # Add compression marker
                return b"COMPRESSED:" + compressed

            return data

        except Exception as e:
            raise CacheError(f"Serialization failed: {str(e)}")

    def deserialize(self, data: bytes) -> Any:
        """Deserialize bytes to value."""
        try:
            # Check for compression marker
            if data.startswith(b"COMPRESSED:"):
                compressed_data = data[11:]  # Remove marker
                data = zlib.decompress(compressed_data)

            # Unpickle the value
            return pickle.loads(data)

        except Exception as e:
            raise CacheError(f"Deserialization failed: {str(e)}")

    def calculate_size(self, value: Any) -> int:
        """Calculate serialized size of value."""
        try:
            return len(self.serialize(value))
        except Exception:
            return sys.getsizeof(value)


class MemoryCacheBackend:
    """High-performance in-memory cache backend."""

    def __init__(self, max_size: int = 1000, strategy: CacheStrategy = CacheStrategy.LRU):
        self.max_size = max_size
        self.strategy = strategy
        self.data: Dict[str, CacheEntry] = {}
        self.access_order: OrderedDict = OrderedDict()
        self.frequency_counter: Dict[str, int] = defaultdict(int)
        self.lock = asyncio.Lock()
        self.logger = get_logger(__name__)

    async def get(self, key: str) -> Optional[Any]:
        """Get value from memory cache."""
        async with self.lock:
            if key not in self.data:
                return None

            entry = self.data[key]

            # Check TTL
            if entry.ttl and self._is_expired(entry):
                await self._remove_entry(key)
                return None

            # Update access statistics
            entry.last_accessed = datetime.now(timezone.utc)
            entry.access_count += 1
            self._update_access_order(key)

            return entry.value

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[float] = None,
        priority: CachePriority = CachePriority.NORMAL,
    ) -> None:
        """Set value in memory cache."""
        async with self.lock:
            # Create cache entry
            entry = CacheEntry(
                key=key, value=value, ttl=ttl, priority=priority, size_bytes=sys.getsizeof(value)
            )

            # Check if we need to evict
            if len(self.data) >= self.max_size and key not in self.data:
                await self._evict_entries(1)

            # Store entry
            self.data[key] = entry
            self._update_access_order(key)

    async def delete(self, key: str) -> bool:
        """Delete value from memory cache."""
        async with self.lock:
            return await self._remove_entry(key)

    async def exists(self, key: str) -> bool:
        """Check if key exists in memory cache."""
        async with self.lock:
            return key in self.data and not self._is_expired(self.data[key])

    async def clear(self) -> None:
        """Clear all entries from memory cache."""
        async with self.lock:
            self.data.clear()
            self.access_order.clear()
            self.frequency_counter.clear()

    async def keys(self, pattern: str = "*") -> List[str]:
        """Get keys matching pattern."""
        async with self.lock:
            # Simple pattern matching (could be enhanced with regex)
            if pattern == "*":
                return list(self.data.keys())
            else:
                return [k for k in self.data.keys() if pattern in k]

    def _is_expired(self, entry: CacheEntry) -> bool:
        """Check if entry is expired."""
        if not entry.ttl:
            return False

        age = (datetime.now(timezone.utc) - entry.created_at).total_seconds()
        return age > entry.ttl

    def _update_access_order(self, key: str) -> None:
        """Update access order for LRU strategy."""
        if self.strategy == CacheStrategy.LRU:
            self.access_order.move_to_end(key)
        elif self.strategy == CacheStrategy.LFU:
            self.frequency_counter[key] += 1

    async def _remove_entry(self, key: str) -> bool:
        """Remove entry from cache."""
        if key in self.data:
            del self.data[key]
            self.access_order.pop(key, None)
            self.frequency_counter.pop(key, None)
            return True
        return False

    async def _evict_entries(self, count: int) -> None:
        """Evict entries based on strategy."""
        if self.strategy == CacheStrategy.LRU:
            await self._evict_lru(count)
        elif self.strategy == CacheStrategy.LFU:
            await self._evict_lfu(count)
        elif self.strategy == CacheStrategy.TTL:
            await self._evict_expired()

    async def _evict_lru(self, count: int) -> None:
        """Evict least recently used entries."""
        keys_to_evict = []

        for key in self.access_order:
            entry = self.data[key]
            if entry.priority != CachePriority.PERMANENT:
                keys_to_evict.append(key)
                if len(keys_to_evict) >= count:
                    break

        for key in keys_to_evict:
            await self._remove_entry(key)

    async def _evict_lfu(self, count: int) -> None:
        """Evict least frequently used entries."""
        # Sort by frequency (ascending)
        sorted_keys = sorted(
            self.data.keys(), key=lambda k: (self.data[k].priority.value, self.frequency_counter[k])
        )

        evicted = 0
        for key in sorted_keys:
            if self.data[key].priority != CachePriority.PERMANENT:
                await self._remove_entry(key)
                evicted += 1
                if evicted >= count:
                    break

    async def _evict_expired(self) -> None:
        """Evict expired entries."""
        expired_keys = [key for key, entry in self.data.items() if self._is_expired(entry)]

        for key in expired_keys:
            await self._remove_entry(key)


class CacheWarmingService:
    """Service for intelligent cache warming and preloading."""

    def __init__(self, cache_strategy: "CacheStrategy", logger):
        self.cache_strategy = cache_strategy
        self.logger = logger
        self.warming_patterns: Dict[str, List[str]] = {}
        self.warming_functions: Dict[str, Callable] = {}
        self.warming_schedules: Dict[str, Dict[str, Any]] = {}

    def register_warming_pattern(
        self, name: str, patterns: List[str], warming_function: Callable
    ) -> None:
        """Register a cache warming pattern."""
        self.warming_patterns[name] = patterns
        self.warming_functions[name] = warming_function

    async def warm_cache(self, pattern_name: str, context: Dict[str, Any] = None) -> None:
        """Warm cache using registered pattern."""
        if pattern_name not in self.warming_functions:
            raise CacheError(f"Warming pattern {pattern_name} not registered")

        try:
            warming_func = self.warming_functions[pattern_name]
            patterns = self.warming_patterns[pattern_name]

            # Execute warming function
            if asyncio.iscoroutinefunction(warming_func):
                await warming_func(patterns, context or {})
            else:
                warming_func(patterns, context or {})

            self.logger.info(f"Cache warming completed for pattern: {pattern_name}")

        except Exception as e:
            self.logger.error(f"Cache warming failed for pattern {pattern_name}: {str(e)}")
            raise CacheError(f"Cache warming failed: {str(e)}")

    async def schedule_warming(self, pattern_name: str, interval_seconds: float) -> None:
        """Schedule periodic cache warming."""
        self.warming_schedules[pattern_name] = {
            "interval": interval_seconds,
            "last_run": datetime.now(timezone.utc),
            "task": asyncio.create_task(self._warming_loop(pattern_name, interval_seconds)),
        }

    async def _warming_loop(self, pattern_name: str, interval: float) -> None:
        """Background task for periodic cache warming."""
        while True:
            try:
                await asyncio.sleep(interval)
                await self.warm_cache(pattern_name)
                self.warming_schedules[pattern_name]["last_run"] = datetime.now(timezone.utc)

            except Exception as e:
                self.logger.error(f"Scheduled warming failed for {pattern_name}: {str(e)}")


class CacheCoherenceManager:
    """Manages cache coherence across distributed instances."""

    def __init__(self, event_bus: EventBus, node_id: str):
        self.event_bus = event_bus
        self.node_id = node_id
        self.logger = get_logger(__name__)
        self.invalidation_queue: asyncio.Queue = asyncio.Queue()
        self.coherence_callbacks: Dict[str, List[Callable]] = defaultdict(list)

    async def invalidate_key(self, key: str, node_id: Optional[str] = None) -> None:
        """Invalidate a key across all nodes."""
        await self.event_bus.emit(
            CacheInvalidation(
                cache_key=key, node_id=node_id or self.node_id, timestamp=datetime.now(timezone.utc)
            )
        )

    async def invalidate_pattern(self, pattern: str, node_id: Optional[str] = None) -> None:
        """Invalidate keys matching pattern across all nodes."""
        await self.event_bus.emit(
            CacheInvalidation(
                cache_key=pattern,
                pattern=True,
                node_id=node_id or self.node_id,
                timestamp=datetime.now(timezone.utc),
            )
        )

    def register_coherence_callback(self, cache_name: str, callback: Callable) -> None:
        """Register callback for coherence events."""
        self.coherence_callbacks[cache_name].append(callback)

    async def handle_invalidation_event(self, event) -> None:
        """Handle cache invalidation events."""
        if event.node_id != self.node_id:  # Don't process our own events
            # Notify registered callbacks
            for callbacks in self.coherence_callbacks.values():
                for callback in callbacks:
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(event)
                        else:
                            callback(event)
                    except Exception as e:
                        self.logger.error(f"Coherence callback failed: {str(e)}")


class EnhancedCacheStrategy:
    """
    Advanced Cache Strategy System for the AI Assistant.

    This system provides:
    - Multi-level caching with intelligent routing
    - Adaptive cache strategies based on usage patterns
    - Event-driven cache invalidation and coherence
    - Performance monitoring and optimization
    - Integration with all core system components
    - Memory pressure handling and resource management
    - Cache warming and preloading capabilities
    - Comprehensive error handling and recovery
    """

    def __init__(self, container: Container):
        """
        Initialize the enhanced cache strategy.

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

        # Cache backends
        self.cache_backends: Dict[CacheLevel, CacheBackend] = {}
        self.cache_policies: Dict[str, CachePolicy] = {}
        self.cache_statistics: Dict[str, CacheStatistics] = {}

        # Core components
        self.serializer = CacheSerializer()
        self.warming_service = CacheWarmingService(self, self.logger)

        # State management
        self.node_id = self.config.get("cache.node_id", f"node_{uuid.uuid4().hex[:8]}")
        self.coherence_manager = CacheCoherenceManager(self.event_bus, self.node_id)

        # Performance tracking
        self.operation_times: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.memory_pressure_level = 0.0
        self.last_cleanup_time = datetime.now(timezone.utc)

        # Configuration
        self.enable_multi_level = self.config.get("cache.enable_multi_level", True)
        self.enable_adaptive_strategy = self.config.get("cache.enable_adaptive", True)
        self.enable_coherence = self.config.get("cache.enable_coherence", True)
        self.cleanup_interval = self.config.get("cache.cleanup_interval", 300.0)

        # Background tasks
        self.background_tasks: List[asyncio.Task] = []

        # Setup cache backends and policies
        self._setup_cache_backends()
        self._setup_default_policies()
        self._setup_monitoring()

        # Register health check
        self.health_check.register_component("cache_strategy", self._health_check_callback)

        self.logger.info("EnhancedCacheStrategy initialized successfully")

    def _setup_cache_backends(self) -> None:
        """Setup cache backends for different levels."""
        try:
            # L1 Memory Cache
            memory_config = self.config.get("cache.memory", {})
            self.cache_backends[CacheLevel.L1_MEMORY] = MemoryCacheBackend(
                max_size=memory_config.get("max_size", 1000),
                strategy=CacheStrategy(memory_config.get("strategy", "lru")),
            )

            # L2 Redis Cache
            if self.config.get("cache.redis.enabled", True):
                try:
                    redis_cache = self.container.get(RedisCache)
                    self.cache_backends[CacheLevel.L2_REDIS] = redis_cache
                except Exception as e:
                    self.logger.warning(f"Redis cache not available: {str(e)}")

            # L3 Local Cache (disk)
            if self.config.get("cache.local.enabled", True):
                try:
                    local_cache = self.container.get(LocalCache)
                    self.cache_backends[CacheLevel.L3_DISK] = local_cache
                except Exception as e:
                    self.logger.warning(f"Local cache not available: {str(e)}")

            self.logger.info(f"Setup {len(self.cache_backends)} cache backends")

        except Exception as e:
            self.logger.error(f"Failed to setup cache backends: {str(e)}")

    def _setup_default_policies(self) -> None:
        """Setup default cache policies for different components."""
        # Core engine cache policy
        self.cache_policies["core_engine"] = CachePolicy(
            name="core_engine",
            strategy=CacheStrategy.ADAPTIVE,
            max_size=5000,
            max_memory_mb=200.0,
            default_ttl=1800.0,  # 30 minutes
            enable_warming=True,
            warming_patterns=["processing_results", "model_outputs"],
        )

        # Session manager cache policy
        self.cache_policies["session_manager"] = CachePolicy(
            name="session_manager",
            strategy=CacheStrategy.LRU,
            max_size=2000,
            max_memory_mb=100.0,
            default_ttl=3600.0,  # 1 hour
            coherence_mode=CacheCoherenceMode.STRONG,
        )

        # Component manager cache policy
        self.cache_policies["component_manager"] = CachePolicy(
            name="component_manager",
            strategy=CacheStrategy.LFU,
            max_size=1000,
            max_memory_mb=50.0,
            default_ttl=7200.0,  # 2 hours
            enable_warming=True,
        )

        # Workflow orchestrator cache policy
        self.cache_policies["workflow_orchestrator"] = CachePolicy(
            name="workflow_orchestrator",
            strategy=CacheStrategy.TTL,
            max_size=3000,
            max_memory_mb=150.0,
            default_ttl=900.0,  # 15 minutes
            enable_monitoring=True,
            enable_detailed_metrics=True,
        )

        # Interaction handler cache policy
        self.cache_policies["interaction_handler"] = CachePolicy(
            name="interaction_handler",
            strategy=CacheStrategy.HYBRID,
            max_size=2500,
            max_memory_mb=120.0,
            default_ttl=600.0,  # 10 minutes
            coherence_mode=CacheCoherenceMode.SESSION,
        )

        # Plugin manager cache policy
        self.cache_policies["plugin_manager"] = CachePolicy(
            name="plugin_manager",
            strategy=CacheStrategy.LRU,
            max_size=500,
            max_memory_mb=30.0,
            default_ttl=14400.0,  # 4 hours
            preload_on_startup=True,
        )

        # Initialize statistics for each policy
        for policy_name in self.cache_policies:
            self.cache_statistics[policy_name] = CacheStatistics()

    def _setup_monitoring(self) -> None:
        """Setup monitoring and metrics collection."""
        try:
            # Register cache metrics
            self.metrics.register_counter("cache_operations_total")
            self.metrics.register_counter("cache_hits_total")
            self.metrics.register_counter("cache_misses_total")
            self.metrics.register_counter("cache_evictions_total")
            self.metrics.register_histogram("cache_operation_duration_seconds")
            self.metrics.register_gauge("cache_size_bytes")
            self.metrics.register_gauge("cache_memory_usage_mb")
            self.metrics.register_gauge("cache_hit_rate")
            self.metrics.register_counter("cache_coherence_events_total")

        except Exception as e:
            self.logger.warning(f"Failed to setup monitoring: {str(e)}")

    async def initialize(self) -> None:
        """Initialize the cache strategy system."""
        try:
            # Initialize cache backends
            for level, backend in self.cache_backends.items():
                if hasattr(backend, "initialize"):
                    await backend.initialize()

            # Register event handlers
            await self._register_event_handlers()

            # Start background tasks
            await self._start_background_tasks()

            # Setup cache warming patterns
            await self._setup_warming_patterns()

            # Preload caches if configured
            await self._preload_caches()

            self.logger.info("CacheStrategy initialization completed")

        except Exception as e:
            self.logger.error(f"Failed to initialize CacheStrategy: {str(e)}")
            raise CacheError(f"Initialization failed: {str(e)}")

    async def _register_event_handlers(self) -> None:
        """Register event handlers for cache management."""
        # Cache coherence events
        self.event_bus.subscribe(
            "cache_invalidation", self.coherence_manager.handle_invalidation_event
        )

        # Session events for cache cleanup
        self.event_bus.subscribe("session_ended", self._handle_session_ended)

        # Component health events
        self.event_bus.subscribe("component_health_changed", self._handle_component_health_change)

        # Memory pressure events
        self.event_bus.subscribe("memory_pressure_alert", self._handle_memory_pressure)

        # Processing completion events for cache warming
        self.event_bus.subscribe("processing_completed", self._handle_processing_completed)

    async def _start_background_tasks(self) -> None:
        """Start background maintenance tasks."""
        # Cache cleanup task
        self.background_tasks.append(asyncio.create_task(self._cleanup_loop()))

        # Performance monitoring task
        self.background_tasks.append(asyncio.create_task(self._performance_monitoring_loop()))

        # Memory pressure monitoring
        self.background_tasks.append(asyncio.create_task(self._memory_monitoring_loop()))

        # Cache coherence sync task
        if self.enable_coherence:
            self.background_tasks.append(asyncio.create_task(self._coherence_sync_loop()))

    async def _setup_warming_patterns(self) -> None:
        """Setup cache warming patterns."""
        # Core engine warming patterns
        self.warming_service.register_warming_pattern(
            "processing_results",
            ["process:*", "model:output:*", "workflow:result:*"],
            self._warm_processing_results,
        )

        # Session data warming
        self.warming_service.register_warming_pattern(
            "session_data", ["session:*:context", "user:*:preferences"], self._warm_session_data
        )

        # Component data warming
        self.warming_service.register_warming_pattern(
            "component_data", ["component:*:config", "component:*:state"], self._warm_component_data
        )

    async def _preload_caches(self) -> None:
        """Preload caches based on policies."""
        for policy_name, policy in self.cache_policies.items():
            if policy.preload_on_startup:
                try:
                    await self.warming_service.warm_cache(f"{policy_name}_startup")
                except Exception as e:
                    self.logger.warning(f"Failed to preload cache {policy_name}: {str(e)}")

    @handle_exceptions
    async def get(
        self, key: str, namespace: str = "default", levels: Optional[List[CacheLevel]] = None
    ) -> Optional[Any]:
        """
        Get value from cache with multi-level lookup.

        Args:
            key: Cache key
            namespace: Cache namespace
            levels: Cache levels to search (defaults to all)

        Returns:
            Cached value or None if not found
        """
        start_time = time.time()
        full_key = self._build_cache_key(key, namespace)

        if levels is None:
            levels = list(self.cache_backends.keys())

        try:
            with self.tracer.trace("cache_get") as span:
                span.set_attributes(
                    {
                        "cache_key": full_key,
                        "namespace": namespace,
                        "levels": [level.value for level in levels],
                    }
                )

                # Search through cache levels
                for level in levels:
                    if level not in self.cache_backends:
                        continue

                    backend = self.cache_backends[level]

                    try:
                        value = await backend.get(full_key)
                        if value is not None:
                            # Cache hit
                            await self._record_cache_hit(namespace, level, full_key)

                            # Promote to higher levels if multi-level enabled
                            if self.enable_multi_level and level != CacheLevel.L1_MEMORY:
                                await self._promote_to_higher_levels(full_key, value, level)

                            operation_time = time.time() - start_time
                            self._record_operation_time("get", operation_time)

                            return value

                    except Exception as e:
                        self.logger.warning(f"Cache get failed for level {level}: {str(e)}")
                        continue

                # Cache miss
                await self._record_cache_miss(namespace, full_key)

                operation_time = time.time() - start_time
                self._record_operation_time("get", operation_time)

                return None

        except Exception as e:
            self.logger.error(f"Cache get operation failed: {str(e)}")
            raise CacheError(f"Get operation failed: {str(e)}", full_key, "get")

    @handle_exceptions
    async def set(
        self,
        key: str,
        value: Any,
        namespace: str = "default",
        ttl: Optional[float] = None,
        priority: CachePriority = CachePriority.NORMAL,
        levels: Optional[List[CacheLevel]] = None,
        tags: Optional[Set[str]] = None,
    ) -> None:
        """
        Set value in cache across specified levels.

        Args:
            key: Cache key
            value: Value to cache
            namespace: Cache namespace
            ttl: Time to live in seconds
            priority: Cache priority
            levels: Cache levels to store in
            tags: Invalidation tags
        """
        start_time = time.time()
        full_key = self._build_cache_key(key, namespace)

        if levels is None:
            levels = list(self.cache_backends.keys())

        try:
            with self.tracer.trace("cache_set") as span:
                span.set_attributes(
                    {
                        "cache_key": full_key,
                        "namespace": namespace,
                        "ttl": ttl,
                        "priority": priority.value,
                        "levels": [level.value for level in levels],
                    }
                )

                # Get policy for namespace
                policy = self.cache_policies.get(namespace, self.cache_policies.get("default"))
                effective_ttl = ttl or (policy.default_ttl if policy else None)

                # Store in specified levels
                tasks = []
                for level in levels:
                    if level in self.cache_backends:
                        backend = self.cache_backends[level]
                        task = self._store_in_backend(
                            backend, full_key, value, effective_ttl, priority
                        )
                        tasks.append(task)

                # Execute storage operations
                results = await asyncio.gather(*tasks, return_exceptions=True)

                # Check for errors
                errors = [r for r in results if isinstance(r, Exception)]
                if errors:
                    self.logger.warning(f"Some cache set operations failed: {errors}")

                # Update statistics
                await self._record_cache_set(namespace, full_key, value)

                # Store invalidation tags if provided
                if tags:
                    await self._store_invalidation_tags(full_key, tags)

                # Emit coherence event if enabled
                if self.enable_coherence:
                    await self.event_bus.emit(
                        CacheCoherenceEvent(
                            operation="set",
                            cache_key=full_key,
                            node_id=self.node_id,
                            timestamp=datetime.now(timezone.utc),
                        )
                    )

                operation_time = time.time() - start_time
                self._record_operation_time("set", operation_time)

        except Exception as e:
            self.logger.error(f"Cache set operation failed: {str(e)}")
            raise CacheError(f"Set operation failed: {str(e)}", full_key, "set")

    async def _store_in_backend(
        self,
        backend: CacheBackend,
        key: str,
        value: Any,
        ttl: Optional[float],
        priority: CachePriority,
    ) -> None:
        """Store value in specific backend."""
        try:
            if hasattr(backend, "set_with_priority"):
                await backend.set_with_priority(key, value, ttl, priority)
            else:
                await backend.set(key, value, ttl)
        except Exception as e:
            self.logger.warning(f"Failed to store in backend: {str(e)}")
            raise

    @handle_exceptions
    async def delete(
        self, key: str, namespace: str = "default", levels: Optional[List[CacheLevel]] = None
    ) -> bool:
        """
        Delete value from cache across specified levels.

        Args:
            key: Cache key to delete
            namespace: Cache namespace
            levels: Cache levels to delete from

        Returns:
            True if key was deleted from at least one level
        """
        full_key = self._build_cache_key(key, namespace)

        if levels is None:
            levels = list(self.cache_backends.keys())

        try:
            deleted = False

            # Delete from specified levels
            for level in levels:
                if level in self.cache_backends:
                    backend = self.cache_backends[level]
                    try:
                        result = await backend.delete(full_key)
                        deleted = deleted or result
                    except Exception as e:
                        self.logger.warning(f"Failed to delete from level {level}: {str(e)}")

            # Emit coherence event if enabled
            if self.enable_coherence and deleted:
                await self.event_bus.emit(
                    CacheCoherenceEvent(
                        operation="delete",
                        cache_key=full_key,
                        node_id=self.node_id,
                        timestamp=datetime.now(timezone.utc),
                    )
                )

            return deleted

        except Exception as e:
            self.logger.error(f"Cache delete operation failed: {str(e)}")
            raise CacheError(f"Delete operation failed: {str(e)}", full_key, "delete")

    @handle_exceptions
    async def invalidate_by_tag(self, tag: str, namespace: str = "default") -> int:
        """
        Invalidate all cache entries with a specific tag.

        Args:
            tag: Invalidation tag
            namespace: Cache namespace

        Returns:
            Number of entries invalidated
        """
        try:
            invalidated_count = 0

            # Find keys with the tag
            for level in self.cache_backends:
                backend = self.cache_backends[level]

                try:
                    # Get all keys and check for tag
                    keys = await backend.keys(f"{namespace}:*")

                    for key in keys:
                        # Check if key has the tag (would need tag storage)
                        if await self._key_has_tag(key, tag):
                            await backend.delete(key)
                            invalidated_count += 1

                except Exception as e:
                    self.logger.warning(f"Tag invalidation failed for level {level}: {str(e)}")

            # Emit invalidation event
            await self.coherence_manager.invalidate_pattern(f"tag:{tag}")

            return invalidated_count

        except Exception as e:
            self.logger.error(f"Tag invalidation failed: {str(e)}")
            raise CacheError(f"Tag invalidation failed: {str(e)}")

    @handle_exceptions
    async def invalidate_namespace(self, namespace: str) -> int:
        """
        Invalidate all cache entries in a namespace.

        Args:
            namespace: Namespace to invalidate

        Returns:
            Number of entries invalidated
        """
        try:
            invalidated_count = 0

            # Clear namespace from all levels
            for level in self.cache_backends:
                backend = self.cache_backends[level]

                try:
                    keys = await backend.keys(f"{namespace}:*")

                    for key in keys:
                        await backend.delete(key)
                        invalidated_count += 1

                except Exception as e:
                    self.logger.warning(
                        f"Namespace invalidation failed for level {level}: {str(e)}"
                    )

            # Emit invalidation event
            await self.coherence_manager.invalidate_pattern(f"{namespace}:*")

            return invalidated_count

        except Exception as e:
            self.logger.error(f"Namespace invalidation failed: {str(e)}")
            raise CacheError(f"Namespace invalidation failed: {str(e)}")

    @handle_exceptions
    async def warm_cache(
        self,
        keys: List[str],
        namespace: str = "default",
        warming_function: Optional[Callable] = None,
    ) -> int:
        """
        Warm cache with specified keys.

        Args:
            keys: Keys to warm
            namespace: Cache namespace
            warming_function: Function to generate values

        Returns:
            Number of keys warmed
        """
        try:
            warmed_count = 0

            for key in keys:
                full_key = self._build_cache_key(key, namespace)

                # Check if already cached
                if await self.get(key, namespace) is not None:
                    continue

                # Generate value using warming function
                if warming_function:
                    try:
                        if asyncio.iscoroutinefunction(warming_function):
                            value = await warming_function(key)
                        else:
                            value = warming_function(key)

                        if value is not None:
                            await self.set(key, value, namespace)
                            warmed_count += 1

                    except Exception as e:
                        self.logger.warning(f"Failed to warm key {key}: {str(e)}")

            # Emit warming event
            await self.event_bus.emit(
                CacheWarming(namespace=namespace, keys_warmed=warmed_count, total_keys=len(keys))
            )

            return warmed_count

        except Exception as e:
            self.logger.error(f"Cache warming failed: {str(e)}")
            raise CacheError(f"Cache warming failed: {str(e)}")

    async def get_statistics(self, namespace: Optional[str] = None) -> Dict[str, CacheStatistics]:
        """
        Get cache statistics.

        Args:
            namespace: Specific namespace or all if None

        Returns:
            Cache statistics
        """
        if namespace:
            return {namespace: self.cache_statistics.get(namespace, CacheStatistics())}
        else:
            return dict(self.cache_statistics)

    async def get_cache_status(self) -> Dict[str, Any]:
        """Get comprehensive cache system status."""
        try:
            status = {
                "node_id": self.node_id,
                "levels": [],
                "policies": len(self.cache_policies),
                "memory_pressure": self.memory_pressure_level,
                "last_cleanup": self.last_cleanup_time.isoformat(),
                "background_tasks": len(self.background_tasks),
                "overall_hit_rate": self._calculate_overall_hit_rate(),
            }

            # Add level status
            for level, backend in self.cache_backends.items():
                level_status = {
                    "level": level.value,
                    "available": True,
                    "backend_type": type(backend).__name__,
                }

                try:
                    if hasattr(backend, "get_status"):
                        level_status.update(await backend.get_status())
                except Exception:
                    level_status["available"] = False

                status["levels"].append(level_status)

            return status

        except Exception as e:
            return {"error": str(e), "status": "unhealthy"}

    def _build_cache_key(self, key: str, namespace: str) -> str:
        """Build full cache key with namespace."""
        return f"{namespace}:{key}"

    async def _promote_to_higher_levels(
        self, key: str, value: Any, current_level: CacheLevel
    ) -> None:
        """Promote cache entry to higher levels."""
        higher_levels = []

        if current_level == CacheLevel.L3_DISK:
            higher_levels = [CacheLevel.L2_REDIS, CacheLevel.L1_MEMORY]
        elif current_level == CacheLevel.L2_REDIS:
            higher_levels = [CacheLevel.L1_MEMORY]

        for level in higher_levels:
            if level in self.cache_backends:
                try:
                    await self.cache_backends[level].set(key, value)
                except Exception as e:
                    self.logger.warning(f"Failed to promote to level {level}: {str(e)}")

    async def _record_cache_hit(self, namespace: str, level: CacheLevel, key: str) -> None:
        """Record cache hit metrics."""
        stats = self.cache_statistics.get(namespace)
        if stats:
            stats.hit_count += 1
            stats.hit_rate = stats.hit_count / max(stats.hit_count + stats.miss_count, 1)

        self.metrics.increment(
            "cache_hits_total", tags={"namespace": namespace, "level": level.value}
        )

        await self.event_bus.emit(
            CacheHit(
                cache_key=key,
                namespace=namespace,
                level=level.value,
                timestamp=datetime.now(timezone.utc),
            )
        )

    async def _record_cache_miss(self, namespace: str, key: str) -> None:
        """Record cache miss metrics."""
        stats = self.cache_statistics.get(namespace)
        if stats:
            stats.miss_count += 1
            stats.miss_rate = stats.miss_count / max(stats.hit_count + stats.miss_count, 1)

        self.metrics.increment("cache_misses_total", tags={"namespace": namespace})

        await self.event_bus.emit(
            CacheMiss(cache_key=key, namespace=namespace, timestamp=datetime.now(timezone.utc))
        )

    async def _record_cache_set(self, namespace: str, key: str, value: Any) -> None:
        """Record cache set metrics."""
        stats = self.cache_statistics.get(namespace)
        if stats:
            stats.total_entries += 1
            value_size = self.serializer.calculate_size(value)
            stats.total_size_bytes += value_size

        self.metrics.increment(
            "cache_operations_total", tags={"namespace": namespace, "operation": "set"}
        )

    def _record_operation_time(self, operation: str, duration: float) -> None:
        """Record operation timing metrics."""
        self.operation_times[operation].append(duration)
        self.metrics.record(
            "cache_operation_duration_seconds", duration, tags={"operation": operation}
        )

    async def _store_invalidation_tags(self, key: str, tags: Set[str]) -> None:
        """Store invalidation tags for a key."""
        # This would store tag mappings in a dedicated store
        # For now, implemented as metadata in memory backend
        pass

    async def _key_has_tag(self, key: str, tag: str) -> bool:
        """Check if a key has a specific invalidation tag."""
        # This would check tag mappings
        # For now, simple pattern matching
        return tag in key

    def _calculate_overall_hit_rate(self) -> float:
        """Calculate overall hit rate across all namespaces."""
        total_hits = sum(stats.hit_count for stats in self.cache_statistics.values())
        total_requests = sum(
            stats.hit_count + stats.miss_count for stats in self.cache_statistics.values()
        )

        return total_hits / max(total_requests, 1)

    async def _warm_processing_results(self, patterns: List[str], context: Dict[str, Any]) -> None:
        """Warm cache with processing results."""
        # Implementation would depend on core engine integration
        pass

    async def _warm_session_data(self, patterns: List[str], context: Dict[str, Any]) -> None:
        """Warm cache with session data."""
        # Implementation would depend on session manager integration
        pass

    async def _warm_component_data(self, patterns: List[str], context: Dict[str, Any]) -> None:
        """Warm cache with component data."""
        # Implementation would depend on component manager integration
        pass

    async def _cleanup_loop(self) -> None:
        """Background task for cache cleanup."""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval)

                # Cleanup expired entries
                cleanup_count = 0

                for level, backend in self.cache_backends.items():
                    try:
                        if hasattr(backend, "cleanup_expired"):
                            count = await backend.cleanup_expired()
                            cleanup_count += count
                    except Exception as e:
                        self.logger.warning(f"Cleanup failed for level {level}: {str(e)}")

                self.last_cleanup_time = datetime.now(timezone.utc)

                if cleanup_count > 0:
                    await self.event_bus.emit(
                        CacheCleanup(
                            entries_cleaned=cleanup_count, timestamp=self.last_cleanup_time
                        )
                    )

            except Exception as e:
                self.logger.error(f"Cleanup loop error: {str(e)}")
                await asyncio.sleep(60)

    async def _performance_monitoring_loop(self) -> None:
        """Background task for performance monitoring."""
        while True:
            try:
                await asyncio.sleep(30)  # Monitor every 30 seconds

                # Update cache statistics
                for namespace, stats in self.cache_statistics.items():
                    stats.last_updated = datetime.now(timezone.utc)

                    # Calculate derived metrics
                    if stats.hit_count + stats.miss_count > 0:
                        stats.hit_rate = stats.hit_count / (stats.hit_count + stats.miss_count)
                        stats.miss_rate = 1.0 - stats.hit_rate

                    # Update metrics
                    self.metrics.set(
                        "cache_hit_rate", stats.hit_rate, tags={"namespace": namespace}
                    )
                    self.metrics.set(
                        "cache_size_bytes", stats.total_size_bytes, tags={"namespace": namespace}
                    )

                # Calculate average operation times
                for operation, times in self.operation_times.items():
                    if times:
                        avg_time = sum(times) / len(times)
                        setattr(
                            self.cache_statistics.get("default", CacheStatistics()),
                            f"avg_{operation}_time_ms",
                            avg_time * 1000,
                        )

            except Exception as e:
                self.logger.error(f"Performance monitoring error: {str(e)}")
                await asyncio.sleep(30)

    async def _memory_monitoring_loop(self) -> None:
        """Background task for memory pressure monitoring."""
        while True:
            try:
                await asyncio.sleep(10)  # Check every 10 seconds

                # Get system memory usage
                memory_info = psutil.virtual_memory()
                self.memory_pressure_level = memory_info.percent / 100.0

                # Check for memory pressure
                if self.memory_pressure_level > 0.85:  # 85% memory usage
                    await self.event_bus.emit(
                        MemoryPressureAlert(
                            level=self.memory_pressure_level,
                            available_mb=memory_info.available / (1024 * 1024),
                            timestamp=datetime.now(timezone.utc),
                        )
                    )

                    # Trigger aggressive cleanup
                    await self._handle_memory_pressure_internal()

            except Exception as e:
                self.logger.error(f"Memory monitoring error: {str(e)}")
                await asyncio.sleep(10)

    async def _coherence_sync_loop(self) -> None:
        """Background task for cache coherence synchronization."""
        while True:
            try:
                await asyncio.sleep(30)  # Sync every 30 seconds

                # Implement coherence synchronization logic
                # This would depend on the specific coherence strategy

            except Exception as e:
                self.logger.error(f"Coherence sync error: {str(e)}")
                await asyncio.sleep(30)

    async def _handle_session_ended(self, event) -> None:
        """Handle session ended events for cache cleanup."""
        try:
            # Invalidate session-specific cache entries
            await self.invalidate_namespace(f"session:{event.session_id}")
        except Exception as e:
            self.logger.error(f"Session cache cleanup failed: {str(e)}")

    async def _handle_component_health_change(self, event) -> None:
        """Handle component health change events."""
        if not event.healthy:
            # Component is unhealthy, might need to invalidate its cache
            await self.invalidate_namespace(f"component:{event.component}")

    async def _handle_memory_pressure(self, event) -> None:
        """Handle memory pressure alert events."""
        await self._handle_memory_pressure_internal()

    async def _handle_memory_pressure_internal(self) -> None:
        """Handle memory pressure by aggressive cache cleanup."""
        try:
            # Evict from memory cache first
            if CacheLevel.L1_MEMORY in self.cache_backends:
                memory_backend = self.cache_backends[CacheLevel.L1_MEMORY]
                if hasattr(memory_backend, "_evict_entries"):
                    await memory_backend._evict_entries(memory_backend.max_size // 4)

            # Force garbage collection
            gc.collect()

            self.logger.info("Performed emergency cache cleanup due to memory pressure")

        except Exception as e:
            self.logger.error(f"Memory pressure handling failed: {str(e)}")

    async def _handle_processing_completed(self, event) -> None:
        """Handle processing completed events for cache warming."""
        try:
            # This could trigger warming of related cache entries
            if hasattr(event, "result_type") and event.result_type in [
                "model_output",
                "workflow_result",
            ]:
                # Schedule warming of related patterns
                pass
        except Exception as e:
            self.logger.error(f"Processing completed cache warming failed: {str(e)}")

    async def _health_check_callback(self) -> Dict[str, Any]:
        """Health check callback for cache strategy."""
        try:
            healthy_backends = 0
            total_backends = len(self.cache_backends)

            for level, backend in self.cache_backends.items():
                try:
                    if hasattr(backend, "ping"):
                        await backend.ping()
                    healthy_backends += 1
                except Exception:
                    pass

            return {
                "status": "healthy" if healthy_backends == total_backends else "degraded",
                "healthy_backends": healthy_backends,
                "total_backends": total_backends,
                "memory_pressure": self.memory_pressure_level,
                "overall_hit_rate": self._calculate_overall_hit_rate(),
                "active_namespaces": len(self.cache_statistics),
            }

        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    async def shutdown(self) -> None:
        """Gracefully shutdown the cache strategy system."""
        try:
            self.logger.info("Starting cache strategy shutdown...")

            # Cancel background tasks
            for task in self.background_tasks:
                task.cancel()

            if self.background_tasks:
                await asyncio.gather(*self.background_tasks, return_exceptions=True)

            # Final cleanup
            for level, backend in self.cache_backends.items():
                try:
                    if hasattr(backend, "cleanup"):
                        await backend.cleanup()
                except Exception as e:
                    self.logger.error(f"Error cleaning up backend {level}: {str(e)}")

            self.logger.info("Cache strategy shutdown completed")

        except Exception as e:
            self.logger.error(f"Error during cache strategy shutdown: {str(e)}")

    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            # Cancel any remaining tasks
            for task in self.background_tasks:
                if not task.done():
                    task.cancel()
        except Exception:
            pass  # Ignore cleanup errors in destructor
