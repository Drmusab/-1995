"""
Memory Cache Manager
Author: Drmusab
Last Modified: 2025-07-05 10:36:03 UTC

This module provides caching functionality for the AI assistant's memory systems,
improving performance by reducing database and storage access latency. It supports
multiple caching strategies and backends, including Redis and local in-memory caching.
"""

import hashlib
import json
import logging
import pickle
import time
import traceback
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Dict, Generic, List, Optional, Set, Tuple, TypeVar, Union

import asyncio

# Core imports
from src.core.config.loader import ConfigLoader
from src.core.dependency_injection import Container
from src.core.error_handling import ErrorHandler, handle_exceptions
from src.core.events.event_bus import EventBus
from src.core.events.event_types import (
    CacheBackupCompleted,
    CacheBackupStarted,
    CacheCleared,
    CacheEvicted,
    CacheHit,
    CacheMiss,
    CacheRestoreCompleted,
    CacheRestoreStarted,
    CacheStored,
    ErrorOccurred,
)
from src.core.health_check import HealthCheck
from src.integrations.cache.cache_strategy import CacheStrategy
from src.integrations.cache.local_cache import LocalCache

# Integration imports
from src.integrations.cache.redis_cache import RedisCache

# Memory imports
from src.memory.core_memory.base_memory import MemoryItem, MemoryType
from src.observability.logging.config import get_logger

# Observability
from src.observability.monitoring.metrics import MetricsCollector
from src.observability.monitoring.tracing import TraceManager


class CacheLevel(Enum):
    """Cache hierarchy levels."""

    L1 = "l1"  # Fastest, usually in-memory
    L2 = "l2"  # Medium speed, may be Redis or similar
    L3 = "l3"  # Slowest, may be disk-based


class CachePolicy(Enum):
    """Caching policies."""

    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    FIFO = "fifo"  # First In First Out
    TTL = "ttl"  # Time To Live
    ADAPTIVE = "adaptive"  # Adaptive based on access patterns


@dataclass
class CacheConfig:
    """Configuration for memory cache."""

    enabled: bool = True
    max_size_l1: int = 1000
    max_size_l2: int = 10000
    ttl_l1: int = 300  # seconds
    ttl_l2: int = 3600  # seconds
    policy: CachePolicy = CachePolicy.LRU
    backup_interval: int = 3600  # seconds
    backup_path: str = "data/cache/memory_cache_backup"
    use_redis: bool = True
    redis_prefix: str = "memory:"
    compression_enabled: bool = True
    compression_level: int = 6  # 0-9, higher is more compression
    stats_interval: int = 60  # seconds


class CacheKey:
    """Utility for generating consistent cache keys."""

    @staticmethod
    def for_memory_item(memory_id: str, memory_type: MemoryType = None) -> str:
        """Generate a cache key for a memory item."""
        if memory_type:
            return f"memory:{memory_type.value}:{memory_id}"
        return f"memory:item:{memory_id}"

    @staticmethod
    def for_query(query_str: str, memory_type: Optional[MemoryType] = None) -> str:
        """Generate a cache key for a query."""
        # Hash the query string for consistent length keys
        query_hash = hashlib.md5(query_str.encode()).hexdigest()
        if memory_type:
            return f"query:{memory_type.value}:{query_hash}"
        return f"query:{query_hash}"

    @staticmethod
    def for_embedding(text: str) -> str:
        """Generate a cache key for text embeddings."""
        # Use first 100 chars plus hash for rest to keep key size reasonable
        if len(text) > 100:
            text_prefix = text[:100]
            text_hash = hashlib.md5(text[100:].encode()).hexdigest()
            return f"embed:{text_prefix}:{text_hash}"
        return f"embed:{text}"

    @staticmethod
    def for_session(session_id: str, subtype: str = None) -> str:
        """Generate a cache key for session data."""
        if subtype:
            return f"session:{session_id}:{subtype}"
        return f"session:{session_id}"

    @staticmethod
    def for_user(user_id: str, subtype: str = None) -> str:
        """Generate a cache key for user data."""
        if subtype:
            return f"user:{user_id}:{subtype}"
        return f"user:{user_id}"


class MemoryCacheEntry:
    """A single entry in the memory cache."""

    def __init__(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        memory_type: Optional[MemoryType] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize a cache entry.

        Args:
            key: Cache key
            value: Cached value
            ttl: Time-to-live in seconds
            memory_type: Type of memory
            metadata: Additional metadata
        """
        self.key = key
        self.value = value
        self.memory_type = memory_type
        self.metadata = metadata or {}

        # Set timestamps
        self.created_at = time.time()
        self.accessed_at = self.created_at
        self.access_count = 0

        # Set expiry
        self.ttl = ttl
        self.expires_at = (self.created_at + ttl) if ttl else None

    def touch(self) -> None:
        """Update access time and count."""
        self.accessed_at = time.time()
        self.access_count += 1

    def is_expired(self) -> bool:
        """Check if entry is expired."""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at

    def get_age(self) -> float:
        """Get age of entry in seconds."""
        return time.time() - self.created_at

    def get_size(self) -> int:
        """Estimate size of entry in bytes."""
        try:
            # Get size of key
            key_size = len(self.key.encode("utf-8"))

            # Try to get size of value
            if hasattr(self.value, "__sizeof__"):
                value_size = self.value.__sizeof__()
            else:
                # Fallback to pickle size
                value_size = len(pickle.dumps(self.value))

            # Add overhead for metadata
            metadata_size = len(pickle.dumps(self.metadata))

            return key_size + value_size + metadata_size
        except Exception:
            # Fallback if we can't determine size
            return 1024  # Assume 1KB


class MemoryCacheManager:
    """
    Memory cache manager for the AI assistant.

    This class provides caching functionality for memory operations,
    improving performance by reducing database and storage access.
    It supports multiple caching strategies and can use Redis for
    distributed caching when available.
    """

    def __init__(self, container: Container):
        """
        Initialize the cache manager.

        Args:
            container: Dependency injection container
        """
        self.container = container
        self.logger = get_logger(__name__)

        # Core components
        self.config_loader = container.get(ConfigLoader)
        self.event_bus = container.get(EventBus)
        self.error_handler = container.get(ErrorHandler)
        self.health_check = container.get(HealthCheck)

        # Load configuration
        cache_config = self.config_loader.get("memory.cache", {})
        self.config = CacheConfig(
            enabled=cache_config.get("enabled", True),
            max_size_l1=cache_config.get("max_size_l1", 1000),
            max_size_l2=cache_config.get("max_size_l2", 10000),
            ttl_l1=cache_config.get("ttl_l1", 300),
            ttl_l2=cache_config.get("ttl_l2", 3600),
            policy=CachePolicy(cache_config.get("policy", "lru")),
            backup_interval=cache_config.get("backup_interval", 3600),
            backup_path=cache_config.get("backup_path", "data/cache/memory_cache_backup"),
            use_redis=cache_config.get("use_redis", True),
            redis_prefix=cache_config.get("redis_prefix", "memory:"),
            compression_enabled=cache_config.get("compression_enabled", True),
            compression_level=cache_config.get("compression_level", 6),
            stats_interval=cache_config.get("stats_interval", 60),
        )

        # Initialize caches
        self._l1_cache: OrderedDict[str, MemoryCacheEntry] = OrderedDict()
        self._l1_size_bytes = 0

        # Get Redis cache if available
        self.redis_cache = None
        if self.config.use_redis:
            try:
                self.redis_cache = container.get(RedisCache)
                self.logger.info("Redis cache initialized for L2 caching")
            except Exception:
                self.logger.warning("Redis cache not available, using local cache for L2")

        # Get local cache as fallback for L2
        if not self.redis_cache:
            try:
                self.local_cache = container.get(LocalCache)
                self.logger.info("Local cache initialized for L2 caching")
            except Exception:
                self.logger.warning("Local cache not available, L2 caching disabled")
                self.local_cache = None

        # Cache statistics
        self._stats = {
            "hits_l1": 0,
            "hits_l2": 0,
            "misses": 0,
            "stores": 0,
            "evictions": 0,
            "size_bytes": 0,
            "entry_count": 0,
        }

        # Get metrics collector if available
        try:
            self.metrics = container.get(MetricsCollector)
            # Register metrics
            self.metrics.register_counter("memory_cache_hits_total")
            self.metrics.register_counter("memory_cache_misses_total")
            self.metrics.register_counter("memory_cache_stores_total")
            self.metrics.register_counter("memory_cache_evictions_total")
            self.metrics.register_gauge("memory_cache_size_bytes")
            self.metrics.register_gauge("memory_cache_entry_count")
            self.metrics.register_histogram("memory_cache_get_latency_seconds")
            self.metrics.register_histogram("memory_cache_set_latency_seconds")
        except Exception:
            self.logger.warning("MetricsCollector not available, metrics disabled")
            self.metrics = None

        # Register health check
        self.health_check.register_component("memory_cache", self._health_check)

        # Schedule background tasks
        self._tasks = []

        self.logger.info("MemoryCacheManager initialized")

    async def start(self) -> None:
        """Start background tasks."""
        if self.config.enabled:
            # Schedule cache maintenance
            self._tasks.append(asyncio.create_task(self._maintenance_loop()))

            # Schedule stats collection
            self._tasks.append(asyncio.create_task(self._stats_loop()))

            # Schedule backup if needed
            if self.config.backup_interval > 0:
                self._tasks.append(asyncio.create_task(self._backup_loop()))

        self.logger.info("MemoryCacheManager background tasks started")

    async def stop(self) -> None:
        """Stop background tasks."""
        for task in self._tasks:
            if not task.done():
                task.cancel()

        # Wait for tasks to complete
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

        self.logger.info("MemoryCacheManager stopped")

    async def get(self, key: str, default: Any = None, level: CacheLevel = CacheLevel.L1) -> Any:
        """
        Get an item from the cache.

        Args:
            key: Cache key
            default: Default value if not found
            level: Cache level to check

        Returns:
            Cached value or default
        """
        if not self.config.enabled:
            return default

        start_time = time.time()

        try:
            # Check L1 cache first (always check L1 regardless of level)
            if key in self._l1_cache:
                entry = self._l1_cache[key]

                # Check if expired
                if entry.is_expired():
                    # Remove from L1
                    self._remove_from_l1(key)
                else:
                    # Update access time and move to end of LRU
                    entry.touch()
                    self._l1_cache.move_to_end(key)

                    # Update statistics
                    self._stats["hits_l1"] += 1

                    # Emit event
                    await self.event_bus.emit(
                        CacheHit(
                            key=key, level=CacheLevel.L1.value, latency=time.time() - start_time
                        )
                    )

                    # Update metrics
                    if self.metrics:
                        self.metrics.increment("memory_cache_hits_total", tags={"level": "l1"})
                        self.metrics.record(
                            "memory_cache_get_latency_seconds", time.time() - start_time
                        )

                    return entry.value

            # Check L2 if requested and L1 miss
            if level in [CacheLevel.L2, CacheLevel.L3]:
                # Try Redis if available
                if self.redis_cache:
                    redis_key = f"{self.config.redis_prefix}{key}"
                    value = await self.redis_cache.get(redis_key)

                    if value is not None:
                        # Try to deserialize
                        try:
                            if isinstance(value, bytes):
                                value = pickle.loads(value)
                            elif isinstance(value, str):
                                try:
                                    value = json.loads(value)
                                except json.JSONDecodeError:
                                    pass  # Keep as string
                        except Exception as e:
                            self.logger.warning(f"Failed to deserialize cache value: {str(e)}")

                        # Store in L1 for future access
                        await self.set(key, value, level=CacheLevel.L1)

                        # Update statistics
                        self._stats["hits_l2"] += 1

                        # Emit event
                        await self.event_bus.emit(
                            CacheHit(
                                key=key, level=CacheLevel.L2.value, latency=time.time() - start_time
                            )
                        )

                        # Update metrics
                        if self.metrics:
                            self.metrics.increment("memory_cache_hits_total", tags={"level": "l2"})
                            self.metrics.record(
                                "memory_cache_get_latency_seconds", time.time() - start_time
                            )

                        return value

                # Try local cache if Redis not available
                elif self.local_cache:
                    value = await self.local_cache.get(key)

                    if value is not None:
                        # Store in L1 for future access
                        await self.set(key, value, level=CacheLevel.L1)

                        # Update statistics
                        self._stats["hits_l2"] += 1

                        # Emit event
                        await self.event_bus.emit(
                            CacheHit(
                                key=key, level=CacheLevel.L2.value, latency=time.time() - start_time
                            )
                        )

                        # Update metrics
                        if self.metrics:
                            self.metrics.increment("memory_cache_hits_total", tags={"level": "l2"})
                            self.metrics.record(
                                "memory_cache_get_latency_seconds", time.time() - start_time
                            )

                        return value

            # Cache miss
            self._stats["misses"] += 1

            # Emit event
            await self.event_bus.emit(CacheMiss(key=key, latency=time.time() - start_time))

            # Update metrics
            if self.metrics:
                self.metrics.increment("memory_cache_misses_total")
                self.metrics.record("memory_cache_get_latency_seconds", time.time() - start_time)

            return default

        except Exception as e:
            self.logger.error(f"Error getting from cache: {str(e)}")
            return default

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        level: CacheLevel = CacheLevel.L1,
        memory_type: Optional[MemoryType] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Set an item in the cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (None for default)
            level: Cache level
            memory_type: Type of memory
            metadata: Additional metadata

        Returns:
            True if successful
        """
        if not self.config.enabled:
            return False

        start_time = time.time()

        try:
            # Set TTL based on level if not specified
            if ttl is None:
                if level == CacheLevel.L1:
                    ttl = self.config.ttl_l1
                else:
                    ttl = self.config.ttl_l2

            # Always set in L1 cache
            self._set_in_l1(key, value, ttl, memory_type, metadata)

            # Store in L2 if requested
            if level in [CacheLevel.L2, CacheLevel.L3]:
                # Use Redis if available
                if self.redis_cache:
                    redis_key = f"{self.config.redis_prefix}{key}"

                    # Serialize value
                    try:
                        if isinstance(value, (dict, list)):
                            serialized = json.dumps(value)
                        else:
                            serialized = pickle.dumps(value)
                    except Exception as e:
                        self.logger.warning(f"Failed to serialize cache value: {str(e)}")
                        serialized = str(value)

                    # Store in Redis
                    await self.redis_cache.set(redis_key, serialized, ttl)

                # Use local cache if Redis not available
                elif self.local_cache:
                    await self.local_cache.set(key, value, ttl)

            # Update statistics
            self._stats["stores"] += 1

            # Emit event
            await self.event_bus.emit(
                CacheStored(
                    key=key,
                    level=level.value,
                    ttl=ttl,
                    size_bytes=len(pickle.dumps(value)) if hasattr(value, "__sizeof__") else 0,
                    latency=time.time() - start_time,
                )
            )

            # Update metrics
            if self.metrics:
                self.metrics.increment("memory_cache_stores_total")
                self.metrics.gauge("memory_cache_size_bytes", self._l1_size_bytes)
                self.metrics.gauge("memory_cache_entry_count", len(self._l1_cache))
                self.metrics.record("memory_cache_set_latency_seconds", time.time() - start_time)

            return True

        except Exception as e:
            self.logger.error(f"Error setting cache value: {str(e)}")
            return False

    async def delete(self, key: str, level: CacheLevel = CacheLevel.L1) -> bool:
        """
        Delete an item from the cache.

        Args:
            key: Cache key
            level: Cache level

        Returns:
            True if successful
        """
        if not self.config.enabled:
            return False

        success = False

        # Remove from L1
        if level in [CacheLevel.L1, CacheLevel.L2, CacheLevel.L3]:
            if key in self._l1_cache:
                self._remove_from_l1(key)
                success = True

        # Remove from L2 if requested
        if level in [CacheLevel.L2, CacheLevel.L3]:
            # Use Redis if available
            if self.redis_cache:
                redis_key = f"{self.config.redis_prefix}{key}"
                await self.redis_cache.delete(redis_key)
                success = True

            # Use local cache if Redis not available
            elif self.local_cache:
                await self.local_cache.delete(key)
                success = True

        if success:
            # Emit event
            await self.event_bus.emit(
                CacheEvicted(key=key, level=level.value, reason="explicit_delete")
            )

        return success

    async def get_memory_item(
        self, memory_id: str, memory_type: Optional[MemoryType] = None
    ) -> Optional[MemoryItem]:
        """
        Get a memory item from cache.

        Args:
            memory_id: Memory identifier
            memory_type: Memory type (for more specific caching)

        Returns:
            Memory item if found, None otherwise
        """
        # Generate cache key
        key = CacheKey.for_memory_item(memory_id, memory_type)

        # Get from cache
        return await self.get(key)

    async def set_memory_item(self, memory_item: MemoryItem, ttl: Optional[int] = None) -> bool:
        """
        Cache a memory item.

        Args:
            memory_item: Memory item to cache
            ttl: Time-to-live in seconds

        Returns:
            True if successful
        """
        # Generate cache key
        key = CacheKey.for_memory_item(memory_item.memory_id, memory_item.memory_type)

        # Set in cache
        return await self.set(
            key,
            memory_item,
            ttl=ttl,
            memory_type=memory_item.memory_type,
            metadata={"owner_id": memory_item.owner_id},
        )

    async def invalidate_memory_item(
        self, memory_id: str, memory_type: Optional[MemoryType] = None
    ) -> bool:
        """
        Invalidate a cached memory item.

        Args:
            memory_id: Memory identifier
            memory_type: Memory type

        Returns:
            True if successful
        """
        # Generate cache key
        key = CacheKey.for_memory_item(memory_id, memory_type)

        # Delete from cache
        return await self.delete(key, level=CacheLevel.L2)

    async def get_cached_query_result(
        self, query: str, memory_type: Optional[MemoryType] = None
    ) -> Optional[Any]:
        """
        Get cached query result.

        Args:
            query: Query string
            memory_type: Memory type

        Returns:
            Cached result if found, None otherwise
        """
        # Generate cache key
        key = CacheKey.for_query(query, memory_type)

        # Get from cache
        return await self.get(key)

    async def set_cached_query_result(
        self,
        query: str,
        result: Any,
        memory_type: Optional[MemoryType] = None,
        ttl: Optional[int] = None,
    ) -> bool:
        """
        Cache a query result.

        Args:
            query: Query string
            result: Query result
            memory_type: Memory type
            ttl: Time-to-live in seconds

        Returns:
            True if successful
        """
        # Generate cache key
        key = CacheKey.for_query(query, memory_type)

        # Set shorter TTL for query results
        if ttl is None:
            ttl = min(60, self.config.ttl_l1)  # Max 1 minute for query results

        # Set in cache
        return await self.set(key, result, ttl=ttl, memory_type=memory_type)

    async def get_cached_embedding(self, text: str) -> Optional[List[float]]:
        """
        Get cached embedding for text.

        Args:
            text: Text to get embedding for

        Returns:
            Cached embedding if found, None otherwise
        """
        # Generate cache key
        key = CacheKey.for_embedding(text)

        # Get from cache
        return await self.get(key, level=CacheLevel.L2)

    async def set_cached_embedding(
        self, text: str, embedding: List[float], ttl: Optional[int] = None
    ) -> bool:
        """
        Cache an embedding.

        Args:
            text: Text
            embedding: Vector embedding
            ttl: Time-to-live in seconds

        Returns:
            True if successful
        """
        # Generate cache key
        key = CacheKey.for_embedding(text)

        # Set longer TTL for embeddings
        if ttl is None:
            ttl = self.config.ttl_l2  # Use L2 TTL for embeddings

        # Set in cache (L2 for embeddings)
        return await self.set(key, embedding, ttl=ttl, level=CacheLevel.L2)

    async def cache_session_data(
        self, session_id: str, data: Any, subtype: Optional[str] = None, ttl: Optional[int] = None
    ) -> bool:
        """
        Cache session-specific data.

        Args:
            session_id: Session identifier
            data: Data to cache
            subtype: Optional subtype
            ttl: Time-to-live in seconds

        Returns:
            True if successful
        """
        # Generate cache key
        key = CacheKey.for_session(session_id, subtype)

        # Set in cache
        return await self.set(key, data, ttl=ttl, metadata={"session_id": session_id})

    async def get_session_data(
        self, session_id: str, subtype: Optional[str] = None
    ) -> Optional[Any]:
        """
        Get cached session data.

        Args:
            session_id: Session identifier
            subtype: Optional subtype

        Returns:
            Cached data if found, None otherwise
        """
        # Generate cache key
        key = CacheKey.for_session(session_id, subtype)

        # Get from cache
        return await self.get(key)

    async def invalidate_session_data(self, session_id: str, subtype: Optional[str] = None) -> bool:
        """
        Invalidate cached session data.

        Args:
            session_id: Session identifier
            subtype: Optional subtype

        Returns:
            True if successful
        """
        # Generate cache key
        key = CacheKey.for_session(session_id, subtype)

        # Delete from cache
        return await self.delete(key, level=CacheLevel.L2)

    async def clear_all(self) -> bool:
        """
        Clear all cached data.

        Returns:
            True if successful
        """
        # Clear L1 cache
        self._l1_cache.clear()
        self._l1_size_bytes = 0

        # Clear L2 cache
        if self.redis_cache:
            # Clear all keys with our prefix
            try:
                await self.redis_cache.clear_pattern(f"{self.config.redis_prefix}*")
            except Exception as e:
                self.logger.error(f"Error clearing Redis cache: {str(e)}")

        if self.local_cache:
            await self.local_cache.clear()

        # Reset statistics
        self._reset_stats()

        # Emit event
        await self.event_bus.emit(CacheCleared())

        # Update metrics
        if self.metrics:
            self.metrics.gauge("memory_cache_size_bytes", 0)
            self.metrics.gauge("memory_cache_entry_count", 0)

        return True

    async def backup(self, path: Optional[str] = None) -> bool:
        """
        Backup cache to disk.

        Args:
            path: Backup path (default from config)

        Returns:
            True if successful
        """
        if not path:
            path = self.config.backup_path

        try:
            # Emit event
            await self.event_bus.emit(CacheBackupStarted(path=path))

            # Create serializable view of L1 cache
            cache_data = {"timestamp": datetime.now(timezone.utc).isoformat(), "entries": {}}

            for key, entry in self._l1_cache.items():
                try:
                    # Only backup entries that aren't expired
                    if not entry.is_expired():
                        cache_data["entries"][key] = {
                            "value": entry.value,
                            "created_at": entry.created_at,
                            "accessed_at": entry.accessed_at,
                            "access_count": entry.access_count,
                            "ttl": entry.ttl,
                            "expires_at": entry.expires_at,
                            "memory_type": entry.memory_type.value if entry.memory_type else None,
                            "metadata": entry.metadata,
                        }
                except Exception as e:
                    self.logger.warning(f"Failed to serialize cache entry {key}: {str(e)}")

            # Save to file
            import os

            os.makedirs(os.path.dirname(path), exist_ok=True)

            with open(path, "wb") as f:
                pickle.dump(cache_data, f)

            # Emit event
            await self.event_bus.emit(
                CacheBackupCompleted(path=path, entry_count=len(cache_data["entries"]))
            )

            self.logger.info(f"Cache backup completed with {len(cache_data['entries'])} entries")
            return True

        except Exception as e:
            self.logger.error(f"Cache backup failed: {str(e)}")
            await self.event_bus.emit(
                ErrorOccurred(
                    component="memory_cache", error_type="backup_failed", error_message=str(e)
                )
            )
            return False

    async def restore(self, path: Optional[str] = None) -> bool:
        """
        Restore cache from backup.

        Args:
            path: Backup path (default from config)

        Returns:
            True if successful
        """
        if not path:
            path = self.config.backup_path

        try:
            # Emit event
            await self.event_bus.emit(CacheRestoreStarted(path=path))

            # Load from file
            with open(path, "rb") as f:
                cache_data = pickle.load(f)

            # Clear current cache
            await self.clear_all()

            # Restore entries
            restored_count = 0
            for key, entry_data in cache_data.get("entries", {}).items():
                try:
                    # Check if expired
                    if entry_data.get("expires_at") and time.time() > entry_data["expires_at"]:
                        continue

                    # Convert memory_type
                    memory_type = None
                    if entry_data.get("memory_type"):
                        try:
                            memory_type = MemoryType(entry_data["memory_type"])
                        except ValueError:
                            pass

                    # Create new entry
                    ttl = entry_data.get("ttl")
                    if ttl and entry_data.get("created_at"):
                        # Adjust TTL based on creation time
                        elapsed = time.time() - entry_data["created_at"]
                        ttl = max(1, ttl - elapsed)

                    # Add to cache
                    await self.set(
                        key,
                        entry_data["value"],
                        ttl=ttl,
                        memory_type=memory_type,
                        metadata=entry_data.get("metadata", {}),
                    )

                    restored_count += 1

                except Exception as e:
                    self.logger.warning(f"Failed to restore cache entry {key}: {str(e)}")

            # Emit event
            await self.event_bus.emit(CacheRestoreCompleted(path=path, entry_count=restored_count))

            self.logger.info(f"Cache restore completed with {restored_count} entries")
            return True

        except Exception as e:
            self.logger.error(f"Cache restore failed: {str(e)}")
            await self.event_bus.emit(
                ErrorOccurred(
                    component="memory_cache", error_type="restore_failed", error_message=str(e)
                )
            )
            return False

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary of statistics
        """
        stats = self._stats.copy()

        # Add hit rate
        total_requests = stats["hits_l1"] + stats["hits_l2"] + stats["misses"]
        if total_requests > 0:
            stats["hit_rate"] = (stats["hits_l1"] + stats["hits_l2"]) / total_requests
        else:
            stats["hit_rate"] = 0.0

        # Add additional information
        stats["cache_enabled"] = self.config.enabled
        stats["l1_max_size"] = self.config.max_size_l1
        stats["l1_entry_count"] = len(self._l1_cache)
        stats["l1_size_bytes"] = self._l1_size_bytes
        stats["l1_ttl"] = self.config.ttl_l1
        stats["redis_available"] = self.redis_cache is not None
        stats["local_l2_available"] = self.local_cache is not None

        return stats

    def _set_in_l1(
        self,
        key: str,
        value: Any,
        ttl: int,
        memory_type: Optional[MemoryType] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Set an item in the L1 cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds
            memory_type: Type of memory
            metadata: Additional metadata
        """
        # Check if entry already exists
        if key in self._l1_cache:
            old_entry = self._l1_cache[key]
            old_size = old_entry.get_size()
            self._l1_size_bytes -= old_size

        # Create new entry
        entry = MemoryCacheEntry(
            key=key, value=value, ttl=ttl, memory_type=memory_type, metadata=metadata
        )

        # Calculate size
        entry_size = entry.get_size()

        # Check if we need to evict items
        while (
            len(self._l1_cache) >= self.config.max_size_l1
            or self._l1_size_bytes + entry_size > 1024 * 1024 * 100
        ):  # 100MB max
            if not self._l1_cache:
                break

            # Evict based on policy
            self._evict_from_l1()

        # Add to cache
        self._l1_cache[key] = entry
        self._l1_size_bytes += entry_size

        # Update entry count in stats
        self._stats["entry_count"] = len(self._l1_cache)
        self._stats["size_bytes"] = self._l1_size_bytes

    def _remove_from_l1(self, key: str) -> None:
        """
        Remove an item from the L1 cache.

        Args:
            key: Cache key
        """
        if key in self._l1_cache:
            # Get entry size
            entry = self._l1_cache[key]
            entry_size = entry.get_size()

            # Remove from cache
            del self._l1_cache[key]

            # Update size
            self._l1_size_bytes -= entry_size

            # Update entry count in stats
            self._stats["entry_count"] = len(self._l1_cache)
            self._stats["size_bytes"] = self._l1_size_bytes

            # Increment evictions
            self._stats["evictions"] += 1

    def _evict_from_l1(self) -> None:
        """Evict an item from the L1 cache based on policy."""
        if not self._l1_cache:
            return

        # Different eviction strategies
        if self.config.policy == CachePolicy.LRU:
            # Least Recently Used - evict first item (OrderedDict keeps insertion order)
            key, _ = next(iter(self._l1_cache.items()))
            self._remove_from_l1(key)

        elif self.config.policy == CachePolicy.LFU:
            # Least Frequently Used - find entry with lowest access_count
            min_count = float("inf")
            min_key = None

            for key, entry in self._l1_cache.items():
                if entry.access_count < min_count:
                    min_count = entry.access_count
                    min_key = key

            if min_key:
                self._remove_from_l1(min_key)

        elif self.config.policy == CachePolicy.FIFO:
            # First In First Out - same as LRU for our implementation
            key, _ = next(iter(self._l1_cache.items()))
            self._remove_from_l1(key)

        elif self.config.policy == CachePolicy.TTL:
            # Evict based on TTL - closest to expiry
            min_expires = float("inf")
            min_key = None

            for key, entry in self._l1_cache.items():
                if entry.expires_at and entry.expires_at < min_expires:
                    min_expires = entry.expires_at
                    min_key = key

            if min_key:
                self._remove_from_l1(min_key)
            else:
                # Fallback to LRU if no expiring entries
                key, _ = next(iter(self._l1_cache.items()))
                self._remove_from_l1(key)

        elif self.config.policy == CachePolicy.ADAPTIVE:
            # Combination of frequency, recency, and size
            # Score = (1/access_count) * age * size
            now = time.time()
            max_score = -float("inf")
            max_key = None

            for key, entry in self._l1_cache.items():
                age = now - entry.accessed_at
                count = max(1, entry.access_count)
                size = entry.get_size()

                score = (1 / count) * age * size

                if score > max_score:
                    max_score = score
                    max_key = key

            if max_key:
                self._remove_from_l1(max_key)
            else:
                # Fallback to LRU
                key, _ = next(iter(self._l1_cache.items()))
                self._remove_from_l1(key)

        else:
            # Default to LRU
            key, _ = next(iter(self._l1_cache.items()))
            self._remove_from_l1(key)

    def _reset_stats(self) -> None:
        """Reset cache statistics."""
        self._stats = {
            "hits_l1": 0,
            "hits_l2": 0,
            "misses": 0,
            "stores": 0,
            "evictions": 0,
            "size_bytes": self._l1_size_bytes,
            "entry_count": len(self._l1_cache),
        }

    async def _maintenance_loop(self) -> None:
        """Background task for cache maintenance."""
        try:
            while True:
                # Run every minute
                await asyncio.sleep(60)

                # Skip if cache disabled
                if not self.config.enabled:
                    continue

                # Clean expired entries
                await self._clean_expired()

        except asyncio.CancelledError:
            self.logger.info("Cache maintenance task cancelled")
        except Exception as e:
            self.logger.error(f"Error in cache maintenance: {str(e)}")
            traceback.print_exc()

    async def _stats_loop(self) -> None:
        """Background task for stats collection."""
        try:
            while True:
                # Run at configured interval
                await asyncio.sleep(self.config.stats_interval)

                # Skip if cache disabled
                if not self.config.enabled:
                    continue

                # Update metrics
                if self.metrics:
                    stats = self.get_stats()
                    self.metrics.gauge("memory_cache_size_bytes", stats["size_bytes"])
                    self.metrics.gauge("memory_cache_entry_count", stats["entry_count"])

        except asyncio.CancelledError:
            self.logger.info("Cache stats task cancelled")
        except Exception as e:
            self.logger.error(f"Error in cache stats: {str(e)}")
            traceback.print_exc()

    async def _backup_loop(self) -> None:
        """Background task for automatic backups."""
        try:
            while True:
                # Run at configured interval
                await asyncio.sleep(self.config.backup_interval)

                # Skip if cache disabled
                if not self.config.enabled:
                    continue

                # Perform backup
                await self.backup()

        except asyncio.CancelledError:
            self.logger.info("Cache backup task cancelled")
        except Exception as e:
            self.logger.error(f"Error in cache backup: {str(e)}")
            traceback.print_exc()

    async def _clean_expired(self) -> int:
        """
        Clean expired entries from cache.

        Returns:
            Number of entries removed
        """
        removed = 0

        # Check L1 cache
        expired_keys = []
        for key, entry in self._l1_cache.items():
            if entry.is_expired():
                expired_keys.append(key)

        # Remove expired entries
        for key in expired_keys:
            self._remove_from_l1(key)
            removed += 1

        # Log result
        if removed > 0:
            self.logger.debug(f"Removed {removed} expired cache entries")

        return removed

    async def _health_check(self) -> Tuple[bool, Dict[str, Any]]:
        """
        Health check callback.

        Returns:
            Tuple of (is_healthy, details)
        """
        # Basic health check
        is_healthy = True
        details = {
            "enabled": self.config.enabled,
            "entry_count": len(self._l1_cache),
            "size_mb": round(self._l1_size_bytes / (1024 * 1024), 2),
            "hit_rate": 0.0,
        }

        # Calculate hit rate
        total_requests = self._stats["hits_l1"] + self._stats["hits_l2"] + self._stats["misses"]
        if total_requests > 0:
            details["hit_rate"] = round(
                (self._stats["hits_l1"] + self._stats["hits_l2"]) / total_requests, 2
            )

        # Check Redis connection if used
        if self.config.use_redis and self.redis_cache:
            try:
                redis_ok = await self.redis_cache.ping()
                details["redis_connected"] = redis_ok
                if not redis_ok:
                    is_healthy = False
            except Exception:
                details["redis_connected"] = False
                is_healthy = False

        return is_healthy, details
