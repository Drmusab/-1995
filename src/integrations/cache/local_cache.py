"""
Advanced Local Cache Implementation for AI Assistant
Author: Drmusab
Last Modified: 2025-01-13 21:04:55 UTC

This module provides a comprehensive local caching system with advanced features
including TTL support, LRU eviction, compression, thread safety, and seamless
integration with all core system components.
"""

from pathlib import Path
from typing import Optional, Dict, Any, List, Set, Callable, Type, Union, AsyncGenerator, TypeVar, Generic
import asyncio
import threading
import time
import pickle
import gzip
import lz4.frame
import hashlib
import json
import weakref
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
from contextlib import asynccontextmanager
import uuid
import logging
from collections import OrderedDict, defaultdict
from abc import ABC, abstractmethod
import psutil
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor

# Core imports
from src.core.config.loader import ConfigLoader
from src.core.events.event_bus import EventBus
from src.core.events.event_types import (
    CacheHit, CacheMiss, CacheEviction, CacheExpired, CacheCleared,
    CacheWarmed, CacheInvalidated, CacheError, CacheHealthChanged,
    SystemStateChanged, ComponentHealthChanged
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
K = TypeVar('K')
V = TypeVar('V')


class CachePolicy(Enum):
    """Cache eviction policies."""
    LRU = "lru"                 # Least Recently Used
    LFU = "lfu"                 # Least Frequently Used
    FIFO = "fifo"               # First In, First Out
    RANDOM = "random"           # Random eviction
    TTL_FIRST = "ttl_first"     # TTL-based eviction first


class CompressionType(Enum):
    """Compression algorithms."""
    NONE = "none"
    GZIP = "gzip"
    LZ4 = "lz4"
    PICKLE = "pickle"


class CacheNamespace(Enum):
    """Cache namespace types."""
    SESSION = "session"
    COMPONENT = "component"
    WORKFLOW = "workflow"
    PROCESSING = "processing"
    MEMORY = "memory"
    PLUGIN = "plugin"
    USER = "user"
    SYSTEM = "system"
    TEMPORARY = "temporary"


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_accessed: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    access_count: int = 0
    ttl_seconds: Optional[float] = None
    size_bytes: int = 0
    compressed: bool = False
    compression_type: CompressionType = CompressionType.NONE
    namespace: str = "default"
    tags: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_expired(self) -> bool:
        """Check if the entry has expired."""
        if self.ttl_seconds is None:
            return False
        
        age = (datetime.now(timezone.utc) - self.created_at).total_seconds()
        return age > self.ttl_seconds
    
    @property
    def age_seconds(self) -> float:
        """Get the age of the entry in seconds."""
        return (datetime.now(timezone.utc) - self.created_at).total_seconds()
    
    def update_access(self) -> None:
        """Update access tracking."""
        self.last_accessed = datetime.now(timezone.utc)
        self.access_count += 1


@dataclass
class CacheConfiguration:
    """Configuration for cache instances."""
    max_size: int = 1000                    # Maximum number of entries
    max_memory_mb: float = 256.0            # Maximum memory usage in MB
    default_ttl_seconds: float = 3600.0     # Default TTL (1 hour)
    eviction_policy: CachePolicy = CachePolicy.LRU
    compression_enabled: bool = True
    compression_type: CompressionType = CompressionType.LZ4
    compression_threshold: int = 1024       # Compress if larger than this
    enable_statistics: bool = True
    enable_warmup: bool = True
    warmup_batch_size: int = 100
    cleanup_interval_seconds: float = 300.0 # Cleanup every 5 minutes
    performance_monitoring: bool = True
    thread_pool_size: int = 4


@dataclass
class CacheStatistics:
    """Cache performance statistics."""
    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    evictions: int = 0
    expirations: int = 0
    compressions: int = 0
    decompressions: int = 0
    memory_usage_bytes: int = 0
    average_access_time_ms: float = 0.0
    hit_ratio: float = 0.0
    compression_ratio: float = 0.0
    last_reset: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def calculate_metrics(self) -> None:
        """Calculate derived metrics."""
        if self.total_requests > 0:
            self.hit_ratio = self.cache_hits / self.total_requests
        else:
            self.hit_ratio = 0.0
        
        if self.compressions > 0:
            self.compression_ratio = self.compressions / (self.compressions + self.decompressions)
        else:
            self.compression_ratio = 0.0


class CacheError(Exception):
    """Custom exception for cache operations."""
    
    def __init__(self, message: str, cache_name: Optional[str] = None, 
                 key: Optional[str] = None, error_code: Optional[str] = None):
        super().__init__(message)
        self.cache_name = cache_name
        self.key = key
        self.error_code = error_code
        self.timestamp = datetime.now(timezone.utc)


class CacheCompressor:
    """Handles compression and decompression of cache values."""
    
    def __init__(self, compression_type: CompressionType = CompressionType.LZ4):
        self.compression_type = compression_type
        self.logger = get_logger(__name__)
    
    def compress(self, data: Any) -> bytes:
        """Compress data using the configured algorithm."""
        try:
            # Serialize to bytes first
            if isinstance(data, bytes):
                serialized = data
            else:
                serialized = pickle.dumps(data)
            
            if self.compression_type == CompressionType.GZIP:
                return gzip.compress(serialized)
            elif self.compression_type == CompressionType.LZ4:
                return lz4.frame.compress(serialized)
            elif self.compression_type == CompressionType.PICKLE:
                return serialized  # Already pickled
            else:
                return serialized
                
        except Exception as e:
            self.logger.error(f"Compression failed: {str(e)}")
            raise CacheError(f"Compression failed: {str(e)}")
    
    def decompress(self, compressed_data: bytes) -> Any:
        """Decompress data using the configured algorithm."""
        try:
            if self.compression_type == CompressionType.GZIP:
                decompressed = gzip.decompress(compressed_data)
            elif self.compression_type == CompressionType.LZ4:
                decompressed = lz4.frame.decompress(compressed_data)
            elif self.compression_type == CompressionType.PICKLE:
                decompressed = compressed_data
            else:
                decompressed = compressed_data
            
            # Deserialize from bytes
            try:
                return pickle.loads(decompressed)
            except:
                return decompressed  # Return as bytes if unpickling fails
                
        except Exception as e:
            self.logger.error(f"Decompression failed: {str(e)}")
            raise CacheError(f"Decompression failed: {str(e)}")


class CacheEvictionStrategy(ABC):
    """Abstract base for cache eviction strategies."""
    
    @abstractmethod
    def select_victim(self, entries: Dict[str, CacheEntry]) -> Optional[str]:
        """Select an entry for eviction."""
        pass


class LRUEvictionStrategy(CacheEvictionStrategy):
    """Least Recently Used eviction strategy."""
    
    def select_victim(self, entries: Dict[str, CacheEntry]) -> Optional[str]:
        """Select the least recently used entry."""
        if not entries:
            return None
        
        oldest_key = None
        oldest_time = datetime.now(timezone.utc)
        
        for key, entry in entries.items():
            if entry.last_accessed < oldest_time:
                oldest_time = entry.last_accessed
                oldest_key = key
        
        return oldest_key


class LFUEvictionStrategy(CacheEvictionStrategy):
    """Least Frequently Used eviction strategy."""
    
    def select_victim(self, entries: Dict[str, CacheEntry]) -> Optional[str]:
        """Select the least frequently used entry."""
        if not entries:
            return None
        
        min_count = float('inf')
        victim_key = None
        
        for key, entry in entries.items():
            if entry.access_count < min_count:
                min_count = entry.access_count
                victim_key = key
        
        return victim_key


class FIFOEvictionStrategy(CacheEvictionStrategy):
    """First In, First Out eviction strategy."""
    
    def select_victim(self, entries: Dict[str, CacheEntry]) -> Optional[str]:
        """Select the oldest entry by creation time."""
        if not entries:
            return None
        
        oldest_key = None
        oldest_time = datetime.now(timezone.utc)
        
        for key, entry in entries.items():
            if entry.created_at < oldest_time:
                oldest_time = entry.created_at
                oldest_key = key
        
        return oldest_key


class TTLFirstEvictionStrategy(CacheEvictionStrategy):
    """TTL-based eviction strategy (expires first)."""
    
    def select_victim(self, entries: Dict[str, CacheEntry]) -> Optional[str]:
        """Select entries that are closest to expiration."""
        if not entries:
            return None
        
        # First, find expired entries
        current_time = datetime.now(timezone.utc)
        for key, entry in entries.items():
            if entry.is_expired:
                return key
        
        # If no expired entries, find the one closest to expiration
        closest_expiry = None
        victim_key = None
        
        for key, entry in entries.items():
            if entry.ttl_seconds is not None:
                expires_at = entry.created_at + timedelta(seconds=entry.ttl_seconds)
                if closest_expiry is None or expires_at < closest_expiry:
                    closest_expiry = expires_at
                    victim_key = key
        
        return victim_key


class LocalCache(Generic[K, V]):
    """
    High-performance local cache with advanced features.
    
    Features:
    - Multiple eviction policies (LRU, LFU, FIFO, TTL-based)
    - Configurable TTL support
    - Compression for large objects
    - Thread-safe operations
    - Performance monitoring
    - Memory usage tracking
    - Namespace support
    - Event-driven notifications
    """
    
    def __init__(self, 
                 name: str,
                 config: Optional[CacheConfiguration] = None,
                 event_bus: Optional[EventBus] = None,
                 metrics: Optional[MetricsCollector] = None):
        """
        Initialize the local cache.
        
        Args:
            name: Cache instance name
            config: Cache configuration
            event_bus: Event bus for notifications
            metrics: Metrics collector
        """
        self.name = name
        self.config = config or CacheConfiguration()
        self.event_bus = event_bus
        self.metrics = metrics
        self.logger = get_logger(f"cache_{name}")
        
        # Internal storage
        self._entries: Dict[str, CacheEntry] = {}
        self._access_order: OrderedDict[str, None] = OrderedDict()
        self._size_tracking: Dict[str, int] = {}
        
        # Thread safety
        self._lock = threading.RLock()
        self._stats_lock = threading.Lock()
        
        # Components
        self._compressor = CacheCompressor(self.config.compression_type)
        self._eviction_strategy = self._create_eviction_strategy()
        
        # Statistics
        self.statistics = CacheStatistics()
        
        # Background tasks
        self._cleanup_task: Optional[asyncio.Task] = None
        self._monitoring_task: Optional[asyncio.Task] = None
        
        # Thread pool for async operations
        self._thread_pool = ThreadPoolExecutor(
            max_workers=self.config.thread_pool_size,
            thread_name_prefix=f"cache_{name}"
        )
        
        # Start background tasks
        if self.config.cleanup_interval_seconds > 0:
            asyncio.create_task(self._start_cleanup_loop())
        
        if self.config.performance_monitoring:
            asyncio.create_task(self._start_monitoring_loop())
        
        self.logger.info(f"LocalCache '{name}' initialized with {self.config.eviction_policy.value} eviction policy")

    def _create_eviction_strategy(self) -> CacheEvictionStrategy:
        """Create the appropriate eviction strategy."""
        strategy_map = {
            CachePolicy.LRU: LRUEvictionStrategy,
            CachePolicy.LFU: LFUEvictionStrategy,
            CachePolicy.FIFO: FIFOEvictionStrategy,
            CachePolicy.TTL_FIRST: TTLFirstEvictionStrategy
        }
        
        strategy_class = strategy_map.get(self.config.eviction_policy, LRUEvictionStrategy)
        return strategy_class()

    @handle_exceptions
    def get(self, key: K, default: Optional[V] = None) -> Optional[V]:
        """
        Get a value from the cache.
        
        Args:
            key: Cache key
            default: Default value if key not found
            
        Returns:
            Cached value or default
        """
        start_time = time.time()
        str_key = str(key)
        
        with self._lock:
            try:
                self._update_statistics('request')
                
                # Check if key exists
                if str_key not in self._entries:
                    self._update_statistics('miss')
                    if self.event_bus:
                        asyncio.create_task(self.event_bus.emit(CacheMiss(
                            cache_name=self.name,
                            key=str_key
                        )))
                    return default
                
                entry = self._entries[str_key]
                
                # Check if expired
                if entry.is_expired:
                    self._remove_entry(str_key)
                    self._update_statistics('miss')
                    self._update_statistics('expiration')
                    
                    if self.event_bus:
                        asyncio.create_task(self.event_bus.emit(CacheExpired(
                            cache_name=self.name,
                            key=str_key,
                            age_seconds=entry.age_seconds
                        )))
                    
                    return default
                
                # Update access tracking
                entry.update_access()
                self._update_access_order(str_key)
                
                # Decompress if needed
                value = self._decompress_value(entry)
                
                self._update_statistics('hit')
                
                # Update access time tracking
                access_time = (time.time() - start_time) * 1000  # Convert to ms
                self._update_access_time(access_time)
                
                if self.event_bus:
                    asyncio.create_task(self.event_bus.emit(CacheHit(
                        cache_name=self.name,
                        key=str_key,
                        access_time_ms=access_time
                    )))
                
                return value
                
            except Exception as e:
                self.logger.error(f"Cache get error for key {str_key}: {str(e)}")
                if self.event_bus:
                    asyncio.create_task(self.event_bus.emit(CacheError(
                        cache_name=self.name,
                        key=str_key,
                        error_message=str(e)
                    )))
                return default

    @handle_exceptions
    def put(self, key: K, value: V, ttl_seconds: Optional[float] = None, 
            namespace: str = "default", tags: Optional[Set[str]] = None,
            metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Put a value into the cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl_seconds: Time to live in seconds
            namespace: Cache namespace
            tags: Entry tags for categorization
            metadata: Additional metadata
            
        Returns:
            True if successfully cached
        """
        str_key = str(key)
        
        with self._lock:
            try:
                # Check if we need to evict
                self._ensure_capacity()
                
                # Compress if needed
                compressed_value, is_compressed, size_bytes = self._compress_value(value)
                
                # Create cache entry
                entry = CacheEntry(
                    key=str_key,
                    value=compressed_value,
                    ttl_seconds=ttl_seconds or self.config.default_ttl_seconds,
                    size_bytes=size_bytes,
                    compressed=is_compressed,
                    compression_type=self.config.compression_type if is_compressed else CompressionType.NONE,
                    namespace=namespace,
                    tags=tags or set(),
                    metadata=metadata or {}
                )
                
                # Store entry
                self._entries[str_key] = entry
                self._update_access_order(str_key)
                self._size_tracking[str_key] = size_bytes
                
                # Update statistics
                self._update_memory_usage()
                
                if is_compressed:
                    self._update_statistics('compression')
                
                self.logger.debug(f"Cached key '{str_key}' in namespace '{namespace}' "
                                f"(size: {size_bytes} bytes, compressed: {is_compressed})")
                
                return True
                
            except Exception as e:
                self.logger.error(f"Cache put error for key {str_key}: {str(e)}")
                if self.event_bus:
                    asyncio.create_task(self.event_bus.emit(CacheError(
                        cache_name=self.name,
                        key=str_key,
                        error_message=str(e)
                    )))
                return False

    @handle_exceptions
    def delete(self, key: K) -> bool:
        """
        Delete a key from the cache.
        
        Args:
            key: Cache key to delete
            
        Returns:
            True if key was deleted
        """
        str_key = str(key)
        
        with self._lock:
            if str_key in self._entries:
                self._remove_entry(str_key)
                self.logger.debug(f"Deleted cache key: {str_key}")
                return True
            return False

    @handle_exceptions
    def clear(self, namespace: Optional[str] = None) -> int:
        """
        Clear cache entries.
        
        Args:
            namespace: Optional namespace to clear (clears all if None)
            
        Returns:
            Number of entries cleared
        """
        with self._lock:
            if namespace is None:
                # Clear all entries
                count = len(self._entries)
                self._entries.clear()
                self._access_order.clear()
                self._size_tracking.clear()
                self.statistics.memory_usage_bytes = 0
                
                if self.event_bus:
                    asyncio.create_task(self.event_bus.emit(CacheCleared(
                        cache_name=self.name,
                        entries_cleared=count
                    )))
                
                self.logger.info(f"Cleared all {count} cache entries")
                return count
            else:
                # Clear specific namespace
                keys_to_remove = [
                    key for key, entry in self._entries.items()
                    if entry.namespace == namespace
                ]
                
                for key in keys_to_remove:
                    self._remove_entry(key)
                
                if self.event_bus:
                    asyncio.create_task(self.event_bus.emit(CacheCleared(
                        cache_name=self.name,
                        namespace=namespace,
                        entries_cleared=len(keys_to_remove)
                    )))
                
                self.logger.info(f"Cleared {len(keys_to_remove)} entries from namespace '{namespace}'")
                return len(keys_to_remove)

    @handle_exceptions
    def exists(self, key: K) -> bool:
        """
        Check if a key exists in the cache.
        
        Args:
            key: Cache key to check
            
        Returns:
            True if key exists and is not expired
        """
        str_key = str(key)
        
        with self._lock:
            if str_key not in self._entries:
                return False
            
            entry = self._entries[str_key]
            if entry.is_expired:
                self._remove_entry(str_key)
                return False
            
            return True

    @handle_exceptions
    def size(self, namespace: Optional[str] = None) -> int:
        """
        Get the number of entries in the cache.
        
        Args:
            namespace: Optional namespace filter
            
        Returns:
            Number of entries
        """
        with self._lock:
            if namespace is None:
                return len(self._entries)
            else:
                return sum(1 for entry in self._entries.values() 
                          if entry.namespace == namespace)

    @handle_exceptions
    def keys(self, namespace: Optional[str] = None, pattern: Optional[str] = None) -> List[str]:
        """
        Get cache keys.
        
        Args:
            namespace: Optional namespace filter
            pattern: Optional key pattern (simple wildcard matching)
            
        Returns:
            List of cache keys
        """
        with self._lock:
            keys = []
            
            for key, entry in self._entries.items():
                # Check namespace filter
                if namespace is not None and entry.namespace != namespace:
                    continue
                
                # Check pattern filter
                if pattern is not None:
                    import fnmatch
                    if not fnmatch.fnmatch(key, pattern):
                        continue
                
                # Check if not expired
                if not entry.is_expired:
                    keys.append(key)
            
            return keys

    @handle_exceptions
    def get_entry_info(self, key: K) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a cache entry.
        
        Args:
            key: Cache key
            
        Returns:
            Entry information dictionary or None
        """
        str_key = str(key)
        
        with self._lock:
            if str_key not in self._entries:
                return None
            
            entry = self._entries[str_key]
            
            return {
                'key': entry.key,
                'created_at': entry.created_at.isoformat(),
                'last_accessed': entry.last_accessed.isoformat(),
                'access_count': entry.access_count,
                'ttl_seconds': entry.ttl_seconds,
                'age_seconds': entry.age_seconds,
                'size_bytes': entry.size_bytes,
                'compressed': entry.compressed,
                'compression_type': entry.compression_type.value,
                'namespace': entry.namespace,
                'tags': list(entry.tags),
                'metadata': entry.metadata,
                'is_expired': entry.is_expired
            }

    @handle_exceptions
    async def warm_cache(self, data_loader: Callable[[List[K]], Dict[K, V]], 
                        keys: List[K], batch_size: Optional[int] = None) -> int:
        """
        Warm the cache with data from a loader function.
        
        Args:
            data_loader: Function that loads data for given keys
            keys: Keys to warm
            batch_size: Batch size for loading
            
        Returns:
            Number of entries warmed
        """
        if not self.config.enable_warmup:
            return 0
        
        batch_size = batch_size or self.config.warmup_batch_size
        warmed_count = 0
        
        try:
            # Process in batches
            for i in range(0, len(keys), batch_size):
                batch_keys = keys[i:i + batch_size]
                
                # Load data
                try:
                    data = await asyncio.get_event_loop().run_in_executor(
                        self._thread_pool,
                        lambda: data_loader(batch_keys)
                    )
                    
                    # Cache loaded data
                    for key, value in data.items():
                        if self.put(key, value):
                            warmed_count += 1
                            
                except Exception as e:
                    self.logger.warning(f"Cache warming batch failed: {str(e)}")
                    continue
            
            if self.event_bus:
                await self.event_bus.emit(CacheWarmed(
                    cache_name=self.name,
                    entries_warmed=warmed_count
                ))
            
            self.logger.info(f"Cache warming completed: {warmed_count} entries loaded")
            return warmed_count
            
        except Exception as e:
            self.logger.error(f"Cache warming failed: {str(e)}")
            return warmed_count

    @handle_exceptions
    def invalidate_by_tags(self, tags: Set[str]) -> int:
        """
        Invalidate cache entries by tags.
        
        Args:
            tags: Tags to match for invalidation
            
        Returns:
            Number of entries invalidated
        """
        with self._lock:
            keys_to_remove = []
            
            for key, entry in self._entries.items():
                if tags.intersection(entry.tags):
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                self._remove_entry(key)
            
            if keys_to_remove and self.event_bus:
                asyncio.create_task(self.event_bus.emit(CacheInvalidated(
                    cache_name=self.name,
                    invalidation_reason="tags",
                    entries_invalidated=len(keys_to_remove)
                )))
            
            self.logger.info(f"Invalidated {len(keys_to_remove)} entries by tags: {tags}")
            return len(keys_to_remove)

    @handle_exceptions
    def invalidate_by_namespace(self, namespace: str) -> int:
        """
        Invalidate all entries in a namespace.
        
        Args:
            namespace: Namespace to invalidate
            
        Returns:
            Number of entries invalidated
        """
        return self.clear(namespace)

    def get_statistics(self) -> CacheStatistics:
        """Get cache statistics."""
        with self._stats_lock:
            stats = CacheStatistics(
                total_requests=self.statistics.total_requests,
                cache_hits=self.statistics.cache_hits,
                cache_misses=self.statistics.cache_misses,
                evictions=self.statistics.evictions,
                expirations=self.statistics.expirations,
                compressions=self.statistics.compressions,
                decompressions=self.statistics.decompressions,
                memory_usage_bytes=self.statistics.memory_usage_bytes,
                average_access_time_ms=self.statistics.average_access_time_ms,
                last_reset=self.statistics.last_reset
            )
            stats.calculate_metrics()
            return stats

    def reset_statistics(self) -> None:
        """Reset cache statistics."""
        with self._stats_lock:
            self.statistics = CacheStatistics()

    def get_memory_usage(self) -> Dict[str, Any]:
        """Get detailed memory usage information."""
        with self._lock:
            total_size = sum(self._size_tracking.values())
            entry_count = len(self._entries)
            
            # Calculate average sizes
            avg_entry_size = total_size / max(entry_count, 1)
            
            # Memory breakdown by namespace
            namespace_sizes = defaultdict(int)
            namespace_counts = defaultdict(int)
            
            for entry in self._entries.values():
                namespace_sizes[entry.namespace] += entry.size_bytes
                namespace_counts[entry.namespace] += 1
            
            return {
                'total_size_bytes': total_size,
                'total_size_mb': total_size / (1024 * 1024),
                'entry_count': entry_count,
                'average_entry_size_bytes': avg_entry_size,
                'memory_limit_mb': self.config.max_memory_mb,
                'memory_utilization': total_size / (self.config.max_memory_mb * 1024 * 1024),
                'namespaces': {
                    ns: {
                        'size_bytes': size,
                        'entry_count': namespace_counts[ns],
                        'average_size': size / max(namespace_counts[ns], 1)
                    }
                    for ns, size in namespace_sizes.items()
                }
            }

    def _compress_value(self, value: Any) -> tuple[Any, bool, int]:
        """Compress value if needed."""
        try:
            # Calculate size
            serialized = pickle.dumps(value)
            size_bytes = len(serialized)
            
            # Check if compression is enabled and threshold is met
            if (self.config.compression_enabled and 
                size_bytes > self.config.compression_threshold):
                
                compressed = self._compressor.compress(value)
                compressed_size = len(compressed)
                
                # Only use compression if it actually reduces size
                if compressed_size < size_bytes:
                    return compressed, True, compressed_size
                else:
                    return value, False, size_bytes
            else:
                return value, False, size_bytes
                
        except Exception as e:
            self.logger.warning(f"Compression failed, storing uncompressed: {str(e)}")
            return value, False, sys.getsizeof(value)

    def _decompress_value(self, entry: CacheEntry) -> Any:
        """Decompress value if needed."""
        if entry.compressed:
            try:
                decompressed = self._compressor.decompress(entry.value)
                self._update_statistics('decompression')
                return decompressed
            except Exception as e:
                self.logger.error(f"Decompression failed for key {entry.key}: {str(e)}")
                raise CacheError(f"Decompression failed: {str(e)}", self.name, entry.key)
        else:
            return entry.value

    def _ensure_capacity(self) -> None:
        """Ensure cache has capacity for new entries."""
        # Check entry count limit
        while len(self._entries) >= self.config.max_size:
            self._evict_entry()
        
        # Check memory limit
        current_memory_mb = sum(self._size_tracking.values()) / (1024 * 1024)
        while current_memory_mb > self.config.max_memory_mb:
            if not self._evict_entry():
                break  # No more entries to evict
            current_memory_mb = sum(self._size_tracking.values()) / (1024 * 1024)

    def _evict_entry(self) -> bool:
        """Evict an entry based on the eviction policy."""
        # First, remove expired entries
        expired_keys = [
            key for key, entry in self._entries.items()
            if entry.is_expired
        ]
        
        if expired_keys:
            key_to_remove = expired_keys[0]
            self._remove_entry(key_to_remove)
            self._update_statistics('expiration')
            return True
        
        # Use eviction strategy
        victim_key = self._eviction_strategy.select_victim(self._entries)
        
        if victim_key:
            self._remove_entry(victim_key)
            self._update_statistics('eviction')
            
            if self.event_bus:
                asyncio.create_task(self.event_bus.emit(CacheEviction(
                    cache_name=self.name,
                    key=victim_key,
                    eviction_reason=self.config.eviction_policy.value
                )))
            
            return True
        
        return False

    def _remove_entry(self, key: str) -> None:
        """Remove an entry from the cache."""
        if key in self._entries:
            del self._entries[key]
        
        if key in self._access_order:
            del self._access_order[key]
        
        if key in self._size_tracking:
            del self._size_tracking[key]

    def _update_access_order(self, key: str) -> None:
        """Update access order for LRU tracking."""
        if key in self._access_order:
            del self._access_order[key]
        self._access_order[key] = None

    def _update_statistics(self, stat_type: str) -> None:
        """Update cache statistics."""
        with self._stats_lock:
            if stat_type == 'request':
                self.statistics.total_requests += 1
            elif stat_type == 'hit':
                self.statistics.cache_hits += 1
            elif stat_type == 'miss':
                self.statistics.cache_misses += 1
            elif stat_type == 'eviction':
                self.statistics.evictions += 1
            elif stat_type == 'expiration':
                self.statistics.expirations += 1
            elif stat_type == 'compression':
                self.statistics.compressions += 1
            elif stat_type == 'decompression':
                self.statistics.decompressions += 1

    def _update_access_time(self, access_time_ms: float) -> None:
        """Update average access time."""
        with self._stats_lock:
            current_avg = self.statistics.average_access_time_ms
            total_requests = self.statistics.total_requests
            
            # Calculate running average
            if total_requests > 0:
                self.statistics.average_access_time_ms = (
                    (current_avg * (total_requests - 1) + access_time_ms) / total_requests
                )

    def _update_memory_usage(self) -> None:
        """Update memory usage statistics."""
        with self._stats_lock:
            self.statistics.memory_usage_bytes = sum(self._size_tracking.values())

    async def _start_cleanup_loop(self) -> None:
        """Start the cleanup background task."""
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def _cleanup_loop(self) -> None:
        """Background cleanup task."""
        while True:
            try:
                await asyncio.sleep(self.config.cleanup_interval_seconds)
                
                # Clean up expired entries
                with self._lock:
                    expired_keys = [
                        key for key, entry in self._entries.items()
                        if entry.is_expired
                    ]
                    
                    for key in expired_keys:
                        self._remove_entry(key)
                        self._update_statistics('expiration')
                    
                    if expired_keys:
                        self.logger.debug(f"Cleaned up {len(expired_keys)} expired entries")
                
                # Update memory usage
                self._update_memory_usage()
                
            except Exception as e:
                self.logger.error(f"Cleanup loop error: {str(e)}")

    async def _start_monitoring_loop(self) -> None:
        """Start the monitoring background task."""
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())

    async def _monitoring_loop(self) -> None:
        """Background monitoring task."""
        while True:
            try:
                await asyncio.sleep(60)  # Monitor every minute
                
                # Update metrics if available
                if self.metrics:
                    stats = self.get_statistics()
                    
                    self.metrics.set(f"cache_entries", len(self._entries), 
                                   tags={'cache': self.name})
                    self.metrics.set(f"cache_memory_usage_bytes", stats.memory_usage_bytes,
                                   tags={'cache': self.name})
                    self.metrics.set(f"cache_hit_ratio", stats.hit_ratio,
                                   tags={'cache': self.name})
                    self.metrics.set(f"cache_compression_ratio", stats.compression_ratio,
                                   tags={'cache': self.name})
                    self.metrics.set(f"cache_average_access_time_ms", stats.average_access_time_ms,
                                   tags={'cache': self.name})
                
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {str(e)}")

    async def cleanup(self) -> None:
        """Cleanup cache resources."""
        try:
            # Cancel background tasks
            if self._cleanup_task:
                self._cleanup_task.cancel()
            
            if self._monitoring_task:
                self._monitoring_task.cancel()
            
            # Shutdown thread pool
            self._thread_pool.shutdown(wait=True)
            
            # Clear cache
            self.clear()
            
            self.logger.info(f"Cache '{self.name}' cleaned up successfully")
            
        except Exception as e:
            self.logger.error(f"Cache cleanup error: {str(e)}")

    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            if hasattr(self, '_thread_pool'):
                self._thread_pool.shutdown(wait=False)
        except Exception:
            pass  # Ignore cleanup errors in destructor


class LocalCacheManager:
    """
    Advanced Local Cache Manager for the AI Assistant.
    
    This manager coordinates multiple cache instances and provides:
    - Cache instance management and configuration
    - Namespace-aware caching
    - Integration with core system components
    - Performance monitoring and optimization
    - Event-driven cache operations
    - Memory management and optimization
    - Health monitoring and alerts
    """
    
    def __init__(self, container: Container):
        """
        Initialize the local cache manager.
        
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
        
        # Cache instances
        self._caches: Dict[str, LocalCache] = {}
        self._cache_configs: Dict[str, CacheConfiguration] = {}
        self._default_config = self._create_default_config()
        
        # Manager state
        self._manager_lock = threading.RLock()
        self._is_initialized = False
        
        # Performance tracking
        self._global_stats = CacheStatistics()
        
        # Setup monitoring
        self._setup_monitoring()
        
        # Register health check
        self.health_check.register_component("cache_manager", self._health_check_callback)
        
        self.logger.info("LocalCacheManager initialized")

    def _create_default_config(self) -> CacheConfiguration:
        """Create default cache configuration."""
        return CacheConfiguration(
            max_size=self.config.get("cache.default_max_size", 1000),
            max_memory_mb=self.config.get("cache.default_max_memory_mb", 256.0),
            default_ttl_seconds=self.config.get("cache.default_ttl_seconds", 3600.0),
            eviction_policy=CachePolicy(self.config.get("cache.default_eviction_policy", "lru")),
            compression_enabled=self.config.get("cache.compression_enabled", True),
            compression_type=CompressionType(self.config.get("cache.compression_type", "lz4")),
            compression_threshold=self.config.get("cache.compression_threshold", 1024),
            enable_statistics=self.config.get("cache.enable_statistics", True),
            cleanup_interval_seconds=self.config.get("cache.cleanup_interval_seconds", 300.0),
            performance_monitoring=self.config.get("cache.performance_monitoring", True)
        )

    def _setup_monitoring(self) -> None:
        """Setup monitoring and metrics."""
        try:
            # Register cache manager metrics
            self.metrics.register_counter("cache_manager_operations_total")
            self.metrics.register_gauge("cache_manager_instances")
            self.metrics.register_gauge("cache_manager_total_memory_usage_bytes")
            self.metrics.register_histogram("cache_manager_operation_duration_seconds")
            
        except Exception as e:
            self.logger.warning(f"Failed to setup monitoring: {str(e)}")

    async def initialize(self) -> None:
        """Initialize the cache manager."""
        try:
            with self._manager_lock:
                if self._is_initialized:
                    return
                
                # Create standard cache instances
                await self._create_standard_caches()
                
                # Start background tasks
                asyncio.create_task(self._global_monitoring_loop())
                
                self._is_initialized = True
                
                self.logger.info("LocalCacheManager initialization completed")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize LocalCacheManager: {str(e)}")
            raise CacheError(f"Cache manager initialization failed: {str(e)}")

    async def _create_standard_caches(self) -> None:
        """Create standard cache instances for system components."""
        # Session cache
        session_config = CacheConfiguration(
            max_size=self.config.get("cache.session.max_size", 500),
            max_memory_mb=self.config.get("cache.session.max_memory_mb", 128.0),
            default_ttl_seconds=self.config.get("cache.session.ttl_seconds", 1800.0),  # 30 minutes
            eviction_policy=CachePolicy.LRU
        )
        await self.create_cache("session", session_config)
        
        # Component cache
        component_config = CacheConfiguration(
            max_size=self.config.get("cache.component.max_size", 200),
            max_memory_mb=self.config.get("cache.component.max_memory_mb", 64.0),
            default_ttl_seconds=self.config.get("cache.component.ttl_seconds", 7200.0),  # 2 hours
            eviction_policy=CachePolicy.LFU
        )
        await self.create_cache("component", component_config)
        
        # Workflow cache
        workflow_config = CacheConfiguration(
            max_size=self.config.get("cache.workflow.max_size", 300),
            max_memory_mb=self.config.get("cache.workflow.max_memory_mb", 96.0),
            default_ttl_seconds=self.config.get("cache.workflow.ttl_seconds", 3600.0),  # 1 hour
            eviction_policy=CachePolicy.TTL_FIRST
        )
        await self.create_cache("workflow", workflow_config)
        
        # Processing cache
        processing_config = CacheConfiguration(
            max_size=self.config.get("cache.processing.max_size", 1000),
            max_memory_mb=self.config.get("cache.processing.max_memory_mb", 256.0),
            default_ttl_seconds=self.config.get("cache.processing.ttl_seconds", 1800.0),  # 30 minutes
            eviction_policy=CachePolicy.LRU,
            compression_enabled=True
        )
        await self.create_cache("processing", processing_config)
        
        # Memory cache
        memory_config = CacheConfiguration(
            max_size=self.config.get("cache.memory.max_size", 800),
            max_memory_mb=self.config.get("cache.memory.max_memory_mb", 192.0),
            default_ttl_seconds=self.config.get("cache.memory.ttl_seconds", 5400.0),  # 1.5 hours
            eviction_policy=CachePolicy.LFU
        )
        await self.create_cache("memory", memory_config)
        
        # Plugin cache
        plugin_config = CacheConfiguration(
            max_size=self.config.get("cache.plugin.max_size", 150),
            max_memory_mb=self.config.get("cache.plugin.max_memory_mb", 48.0),
            default_ttl_seconds=self.config.get("cache.plugin.ttl_seconds", 3600.0),  # 1 hour
            eviction_policy=CachePolicy.LRU
        )
        await self.create_cache("plugin", plugin_config)

    @handle_exceptions
    async def create_cache(self, name: str, config: Optional[CacheConfiguration] = None) -> LocalCache:
        """
        Create a new cache instance.
        
        Args:
            name: Cache name
            config: Cache configuration
            
        Returns:
            Created cache instance
        """
        with self._manager_lock:
            if name in self._caches:
                raise CacheError(f"Cache '{name}' already exists")
            
            cache_config = config or self._default_config
            self._cache_configs[name] = cache_config
            
            cache = LocalCache(
                name=name,
                config=cache_config,
                event_bus=self.event_bus,
                metrics=self.metrics
            )
            
            self._caches[name] = cache
            
            # Update metrics
            if self.metrics:
                self.metrics.set("cache_manager_instances", len(self._caches))
            
            self.logger.info(f"Created cache instance: {name}")
            return cache

    @handle_exceptions
    def get_cache(self, name: str) -> LocalCache:
        """
        Get a cache instance by name.
        
        Args:
            name: Cache name
            
        Returns:
            Cache instance
            
        Raises:
            CacheError: If cache not found
        """
        if name not in self._caches:
            raise CacheError(f"Cache '{name}' not found")
        
        return self._caches[name]

    @handle_exceptions
    async def delete_cache(self, name: str) -> None:
        """
        Delete a cache instance.
        
        Args:
            name: Cache name to delete
        """
        with self._manager_lock:
            if name not in self._caches:
                raise CacheError(f"Cache '{name}' not found")
            
            cache = self._caches[name]
            await cache.cleanup()
            
            del self._caches[name]
            del self._cache_configs[name]
            
            # Update metrics
            if self.metrics:
                self.metrics.set("cache_manager_instances", len(self._caches))
            
            self.logger.info(f"Deleted cache instance: {name}")

    @handle_exceptions
    def list_caches(self) -> List[Dict[str, Any]]:
        """
        List all cache instances with their status.
        
        Returns:
            List of cache information
        """
        caches_info = []
        
        for name, cache in self._caches.items():
            try:
                stats = cache.get_statistics()
                memory_info = cache.get_memory_usage()
                
                caches_info.append({
                    'name': name,
                    'size': cache.size(),
                    'memory_usage_mb': memory_info['total_size_mb'],
                    'memory_limit_mb': cache.config.max_memory_mb,
                    'hit_ratio': stats.hit_ratio,
                    'total_requests': stats.total_requests,
                    'eviction_policy': cache.config.eviction_policy.value,
                    'compression_enabled': cache.config.compression_enabled,
                    'namespaces': list(memory_info['namespaces'].keys())
                })
                
            except Exception as e:
                caches_info.append({
                    'name': name,
                    'error': str(e)
                })
        
        return caches_info

    @handle_exceptions
    def get_global_statistics(self) -> Dict[str, Any]:
        """
        Get global cache statistics across all instances.
        
        Returns:
            Global statistics dictionary
        """
        total_stats = {
            'total_caches': len(self._caches),
            'total_entries': 0,
            'total_memory_usage_mb': 0.0,
            'total_requests': 0,
            'total_hits': 0,
            'total_misses': 0,
            'total_evictions': 0,
            'total_expirations': 0,
            'global_hit_ratio': 0.0,
            'cache_breakdown': {}
        }
        
        for name, cache in self._caches.items():
            try:
                stats = cache.get_statistics()
                memory_info = cache.get_memory_usage()
                
                total_stats['total_entries'] += cache.size()
                total_stats['total_memory_usage_mb'] += memory_info['total_size_mb']
                total_stats['total_requests'] += stats.total_requests
                total_stats['total_hits'] += stats.cache_hits
                total_stats['total_misses'] += stats.cache_misses
                total_stats['total_evictions'] += stats.evictions
                total_stats['total_expirations'] += stats.expirations
                
                total_stats['cache_breakdown'][name] = {
                    'entries': cache.size(),
                    'memory_mb': memory_info['total_size_mb'],
                    'hit_ratio': stats.hit_ratio,
                    'requests': stats.total_requests
                }
                
            except Exception as e:
                self.logger.warning(f"Failed to get stats for cache {name}: {str(e)}")
        
        # Calculate global hit ratio
        if total_stats['total_requests'] > 0:
            total_stats['global_hit_ratio'] = (
                total_stats['total_hits'] / total_stats['total_requests']
            )
        
        return total_stats

    @handle_exceptions
    async def clear_all_caches(self, namespace: Optional[str] = None) -> Dict[str, int]:
        """
        Clear all cache instances.
        
        Args:
            namespace: Optional namespace to clear from all caches
            
        Returns:
            Dictionary of cache names and entries cleared
        """
        results = {}
        
        for name, cache in self._caches.items():
            try:
                cleared = cache.clear(namespace)
                results[name] = cleared
            except Exception as e:
                self.logger.error(f"Failed to clear cache {name}: {str(e)}")
                results[name] = 0
        
        self.logger.info(f"Cleared caches: {results}")
        return results

    @handle_exceptions
    async def warm_all_caches(self, warmup_functions: Dict[str, Callable]) -> Dict[str, int]:
        """
        Warm multiple caches using provided functions.
        
        Args:
            warmup_functions: Dictionary of cache names to warmup functions
            
        Returns:
            Dictionary of cache names and entries warmed
        """
        results = {}
        
        for cache_name, warmup_func in warmup_functions.items():
            if cache_name in self._caches:
                try:
                    cache = self._caches[cache_name]
                    # This would need to be implemented based on specific warmup requirements
                    # For now, just log the attempt
                    self.logger.info(f"Warming cache: {cache_name}")
                    results[cache_name] = 0  # Placeholder
                except Exception as e:
                    self.logger.error(f"Failed to warm cache {cache_name}: {str(e)}")
                    results[cache_name] = 0
        
        return results

    # Integration methods for core components

    @handle_exceptions
    def cache_session_data(self, session_id: str, data: Any, ttl_seconds: Optional[float] = None) -> bool:
        """Cache session data."""
        try:
            session_cache = self.get_cache("session")
            return session_cache.put(
                key=session_id,
                value=data,
                ttl_seconds=ttl_seconds,
                namespace=CacheNamespace.SESSION.value
            )
        except Exception as e:
            self.logger.error(f"Failed to cache session data: {str(e)}")
            return False

    @handle_exceptions
    def get_session_data(self, session_id: str) -> Optional[Any]:
        """Get cached session data."""
        try:
            session_cache = self.get_cache("session")
            return session_cache.get(session_id)
        except Exception as e:
            self.logger.error(f"Failed to get session data: {str(e)}")
            return None

    @handle_exceptions
    def cache_component_result(self, component_id: str, input_hash: str, result: Any, 
                              ttl_seconds: Optional[float] = None) -> bool:
        """Cache component processing result."""
        try:
            component_cache = self.get_cache("component")
            cache_key = f"{component_id}:{input_hash}"
            return component_cache.put(
                key=cache_key,
                value=result,
                ttl_seconds=ttl_seconds,
                namespace=CacheNamespace.COMPONENT.value,
                tags={component_id, "processing_result"}
            )
        except Exception as e:
            self.logger.error(f"Failed to cache component result: {str(e)}")
            return False

    @handle_exceptions
    def get_component_result(self, component_id: str, input_hash: str) -> Optional[Any]:
        """Get cached component result."""
        try:
            component_cache = self.get_cache("component")
            cache_key = f"{component_id}:{input_hash}"
            return component_cache.get(cache_key)
        except Exception as e:
            self.logger.error(f"Failed to get component result: {str(e)}")
            return None

    @handle_exceptions
    def cache_workflow_state(self, workflow_id: str, execution_id: str, state: Any,
                            ttl_seconds: Optional[float] = None) -> bool:
        """Cache workflow execution state."""
        try:
            workflow_cache = self.get_cache("workflow")
            cache_key = f"{workflow_id}:{execution_id}"
            return workflow_cache.put(
                key=cache_key,
                value=state,
                ttl_seconds=ttl_seconds,
                namespace=CacheNamespace.WORKFLOW.value,
                tags={workflow_id, "workflow_state"}
            )
        except Exception as e:
            self.logger.error(f"Failed to cache workflow state: {str(e)}")
            return False

    @handle_exceptions
    def get_workflow_state(self, workflow_id: str, execution_id: str) -> Optional[Any]:
        """Get cached workflow state."""
        try:
            workflow_cache = self.get_cache("workflow")
            cache_key = f"{workflow_id}:{execution_id}"
            return workflow_cache.get(cache_key)
        except Exception as e:
            self.logger.error(f"Failed to get workflow state: {str(e)}")
            return None

    @handle_exceptions
    def cache_processing_result(self, request_hash: str, result: Any, 
                               ttl_seconds: Optional[float] = None) -> bool:
        """Cache processing pipeline result."""
        try:
            processing_cache = self.get_cache("processing")
            return processing_cache.put(
                key=request_hash,
                value=result,
                ttl_seconds=ttl_seconds,
                namespace=CacheNamespace.PROCESSING.value,
                tags={"processing_result"}
            )
        except Exception as e:
            self.logger.error(f"Failed to cache processing result: {str(e)}")
            return False

    @handle_exceptions
    def get_processing_result(self, request_hash: str) -> Optional[Any]:
        """Get cached processing result."""
        try:
            processing_cache = self.get_cache("processing")
            return processing_cache.get(request_hash)
        except Exception as e:
            self.logger.error(f"Failed to get processing result: {str(e)}")
            return None

    @handle_exceptions
    def cache_memory_data(self, memory_id: str, data: Any, ttl_seconds: Optional[float] = None) -> bool:
        """Cache memory system data."""
        try:
            memory_cache = self.get_cache("memory")
            return memory_cache.put(
                key=memory_id,
                value=data,
                ttl_seconds=ttl_seconds,
                namespace=CacheNamespace.MEMORY.value,
                tags={"memory_data"}
            )
        except Exception as e:
            self.logger.error(f"Failed to cache memory data: {str(e)}")
            return False

    @handle_exceptions
    def get_memory_data(self, memory_id: str) -> Optional[Any]:
        """Get cached memory data."""
        try:
            memory_cache = self.get_cache("memory")
            return memory_cache.get(memory_id)
        except Exception as e:
            self.logger.error(f"Failed to get memory data: {str(e)}")
            return None

    @handle_exceptions
    def cache_plugin_data(self, plugin_id: str, data_key: str, data: Any,
                         ttl_seconds: Optional[float] = None) -> bool:
        """Cache plugin data."""
        try:
            plugin_cache = self.get_cache("plugin")
            cache_key = f"{plugin_id}:{data_key}"
            return plugin_cache.put(
                key=cache_key,
                value=data,
                ttl_seconds=ttl_seconds,
                namespace=CacheNamespace.PLUGIN.value,
                tags={plugin_id, "plugin_data"}
            )
        except Exception as e:
            self.logger.error(f"Failed to cache plugin data: {str(e)}")
            return False

    @handle_exceptions
    def get_plugin_data(self, plugin_id: str, data_key: str) -> Optional[Any]:
        """Get cached plugin data."""
        try:
            plugin_cache = self.get_cache("plugin")
            cache_key = f"{plugin_id}:{data_key}"
            return plugin_cache.get(cache_key)
        except Exception as e:
            self.logger.error(f"Failed to get plugin data: {str(e)}")
            return None

    @handle_exceptions
    def invalidate_component_cache(self, component_id: str) -> int:
        """Invalidate all cached data for a component."""
        try:
            component_cache = self.get_cache("component")
            return component_cache.invalidate_by_tags({component_id})
        except Exception as e:
            self.logger.error(f"Failed to invalidate component cache: {str(e)}")
            return 0

    @handle_exceptions
    def invalidate_workflow_cache(self, workflow_id: str) -> int:
        """Invalidate cache entries for a specific workflow."""
        try:
            invalidated = 0
            keys_to_remove = []
            
            # Find workflow-related cache entries
            for key in self.cache:
                if isinstance(key, str) and workflow_id in key:
                    keys_to_remove.append(key)
            
            # Remove found entries
            for key in keys_to_remove:
                del self.cache[key]
                invalidated += 1
            
            self.logger.debug(f"Invalidated {invalidated} cache entries for workflow {workflow_id}")
            return invalidated
            
        except Exception as e:
            self.logger.error(f"Failed to invalidate workflow cache: {str(e)}")
            return 0
