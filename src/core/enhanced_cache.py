"""
Enhanced caching system for performance optimization
Author: Drmusab
Last Modified: 2025-08-10

This module provides multiple caching strategies to improve performance:
- In-memory LRU cache
- Distributed Redis cache
- Function result caching
- Database query caching
- Smart cache invalidation
"""

import asyncio
import hashlib
import json
import pickle
import time
import threading
import weakref
from collections import OrderedDict, defaultdict
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from src.core.lazy_imports import lazy_import
from src.observability.logging.config import get_logger

# Lazy imports
redis = lazy_import('redis')


@dataclass
class CacheStats:
    """Statistics for cache performance."""
    
    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    evictions: int = 0
    total_size: int = 0
    hit_rate: float = 0.0
    
    def update_hit_rate(self):
        """Update hit rate calculation."""
        total = self.hits + self.misses
        self.hit_rate = (self.hits / total) if total > 0 else 0.0


@dataclass 
class CacheEntry:
    """Single cache entry with metadata."""
    
    key: str
    value: Any
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    ttl: Optional[float] = None
    size: int = 0
    
    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.ttl is None:
            return False
        return time.time() > (self.created_at + self.ttl)
    
    def touch(self):
        """Update last accessed time."""
        self.last_accessed = time.time()
        self.access_count += 1


class EnhancedLRUCache:
    """
    Enhanced LRU cache with TTL, size limits, and statistics.
    Thread-safe implementation with advanced features.
    """
    
    def __init__(self, 
                 max_size: int = 1000,
                 default_ttl: Optional[float] = None,
                 max_memory_mb: Optional[float] = None):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.max_memory_bytes = max_memory_mb * 1024 * 1024 if max_memory_mb else None
        
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        self._stats = CacheStats()
        self._current_memory = 0
        
        self.logger = get_logger(f"{__name__}.LRUCache")
        
        # Background cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None
        self._start_cleanup_task()
    
    def _start_cleanup_task(self):
        """Start background cleanup task for expired entries."""
        try:
            loop = asyncio.get_event_loop()
            self._cleanup_task = loop.create_task(self._cleanup_loop())
        except RuntimeError:
            # No event loop, cleanup will happen on access
            pass
    
    async def _cleanup_loop(self):
        """Background task to clean up expired entries."""
        while True:
            try:
                await asyncio.sleep(60)  # Clean up every minute
                self._cleanup_expired()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in cache cleanup: {str(e)}")
    
    def _cleanup_expired(self):
        """Remove expired entries from cache."""
        with self._lock:
            expired_keys = []
            
            for key, entry in self._cache.items():
                if entry.is_expired():
                    expired_keys.append(key)
            
            for key in expired_keys:
                self._remove_entry(key)
                self._stats.evictions += 1
    
    def _calculate_size(self, value: Any) -> int:
        """Calculate approximate size of value in bytes."""
        try:
            if isinstance(value, (str, bytes)):
                return len(value)
            elif isinstance(value, (int, float, bool)):
                return 8
            elif isinstance(value, (list, tuple)):
                return sum(self._calculate_size(item) for item in value)
            elif isinstance(value, dict):
                return sum(
                    self._calculate_size(k) + self._calculate_size(v) 
                    for k, v in value.items()
                )
            else:
                # Use pickle size as approximation
                return len(pickle.dumps(value))
        except Exception:
            return 1024  # Default size if calculation fails
    
    def _make_room(self, required_size: int):
        """Make room in cache by evicting LRU entries."""
        while (
            (self.max_memory_bytes and self._current_memory + required_size > self.max_memory_bytes) or
            (len(self._cache) >= self.max_size)
        ):
            if not self._cache:
                break
                
            # Remove least recently used (first item in OrderedDict)
            oldest_key = next(iter(self._cache))
            self._remove_entry(oldest_key)
            self._stats.evictions += 1
    
    def _remove_entry(self, key: str):
        """Remove an entry and update memory usage."""
        if key in self._cache:
            entry = self._cache.pop(key)
            self._current_memory -= entry.size
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache."""
        with self._lock:
            if key not in self._cache:
                self._stats.misses += 1
                self._stats.update_hit_rate()
                return default
            
            entry = self._cache[key]
            
            # Check if expired
            if entry.is_expired():
                self._remove_entry(key)
                self._stats.misses += 1
                self._stats.update_hit_rate()
                return default
            
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            entry.touch()
            
            self._stats.hits += 1
            self._stats.update_hit_rate()
            return entry.value
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """Set value in cache."""
        with self._lock:
            # Calculate size
            size = self._calculate_size(value)
            
            # Remove existing entry if present
            if key in self._cache:
                self._remove_entry(key)
            
            # Make room for new entry
            self._make_room(size)
            
            # Create new entry
            entry = CacheEntry(
                key=key,
                value=value,
                ttl=ttl or self.default_ttl,
                size=size
            )
            
            self._cache[key] = entry
            self._current_memory += size
            self._stats.sets += 1
            self._stats.total_size = len(self._cache)
            
            return True
    
    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        with self._lock:
            if key in self._cache:
                self._remove_entry(key)
                self._stats.deletes += 1
                self._stats.total_size = len(self._cache)
                return True
            return False
    
    def clear(self):
        """Clear all entries from cache."""
        with self._lock:
            self._cache.clear()
            self._current_memory = 0
            self._stats.total_size = 0
    
    def keys(self) -> List[str]:
        """Get all keys in cache."""
        with self._lock:
            return list(self._cache.keys())
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        with self._lock:
            stats = CacheStats(
                hits=self._stats.hits,
                misses=self._stats.misses,
                sets=self._stats.sets,
                deletes=self._stats.deletes,
                evictions=self._stats.evictions,
                total_size=len(self._cache),
                hit_rate=self._stats.hit_rate
            )
            return stats
    
    def __len__(self) -> int:
        return len(self._cache)
    
    def __contains__(self, key: str) -> bool:
        with self._lock:
            if key not in self._cache:
                return False
            return not self._cache[key].is_expired()


class RedisCache:
    """
    Redis-based distributed cache for shared caching across instances.
    """
    
    def __init__(self, 
                 redis_url: str = "redis://localhost:6379",
                 key_prefix: str = "ai_assistant:",
                 default_ttl: int = 3600,
                 serializer: str = "json"):
        self.redis_url = redis_url
        self.key_prefix = key_prefix
        self.default_ttl = default_ttl
        self.serializer = serializer
        
        self._redis_client = None
        self._stats = CacheStats()
        self.logger = get_logger(f"{__name__}.RedisCache")
    
    async def _get_redis_client(self):
        """Get Redis client with lazy initialization."""
        if self._redis_client is None:
            try:
                redis_module = redis._resolve()
                self._redis_client = redis_module.from_url(
                    self.redis_url,
                    decode_responses=True,
                    socket_connect_timeout=5,
                    socket_timeout=5,
                    retry_on_timeout=True,
                    encoding='utf-8'
                )
                # Test connection
                await self._redis_client.ping()
                self.logger.info(f"Connected to Redis: {self.redis_url}")
            except Exception as e:
                self.logger.error(f"Failed to connect to Redis: {str(e)}")
                raise
        
        return self._redis_client
    
    def _serialize(self, value: Any) -> str:
        """Serialize value for storage."""
        if self.serializer == "json":
            return json.dumps(value, default=str)
        elif self.serializer == "pickle":
            return pickle.dumps(value).hex()
        else:
            return str(value)
    
    def _deserialize(self, value: str) -> Any:
        """Deserialize value from storage."""
        if self.serializer == "json":
            return json.loads(value)
        elif self.serializer == "pickle":
            return pickle.loads(bytes.fromhex(value))
        else:
            return value
    
    def _make_key(self, key: str) -> str:
        """Create full cache key with prefix."""
        return f"{self.key_prefix}{key}"
    
    async def get(self, key: str, default: Any = None) -> Any:
        """Get value from Redis cache."""
        try:
            client = await self._get_redis_client()
            full_key = self._make_key(key)
            
            value = await client.get(full_key)
            
            if value is None:
                self._stats.misses += 1
                self._stats.update_hit_rate()
                return default
            
            deserialized = self._deserialize(value)
            self._stats.hits += 1
            self._stats.update_hit_rate()
            return deserialized
            
        except Exception as e:
            self.logger.error(f"Redis get error for key {key}: {str(e)}")
            self._stats.misses += 1
            self._stats.update_hit_rate()
            return default
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in Redis cache."""
        try:
            client = await self._get_redis_client()
            full_key = self._make_key(key)
            
            serialized = self._serialize(value)
            ttl = ttl or self.default_ttl
            
            result = await client.setex(full_key, ttl, serialized)
            
            self._stats.sets += 1
            return bool(result)
            
        except Exception as e:
            self.logger.error(f"Redis set error for key {key}: {str(e)}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete key from Redis cache."""
        try:
            client = await self._get_redis_client()
            full_key = self._make_key(key)
            
            result = await client.delete(full_key)
            
            self._stats.deletes += 1
            return result > 0
            
        except Exception as e:
            self.logger.error(f"Redis delete error for key {key}: {str(e)}")
            return False
    
    async def clear(self, pattern: str = "*") -> int:
        """Clear cache entries matching pattern."""
        try:
            client = await self._get_redis_client()
            full_pattern = self._make_key(pattern)
            
            keys = await client.keys(full_pattern)
            if keys:
                result = await client.delete(*keys)
                return result
            return 0
            
        except Exception as e:
            self.logger.error(f"Redis clear error: {str(e)}")
            return 0
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in Redis cache."""
        try:
            client = await self._get_redis_client()
            full_key = self._make_key(key)
            
            result = await client.exists(full_key)
            return result > 0
            
        except Exception as e:
            self.logger.error(f"Redis exists error for key {key}: {str(e)}")
            return False
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        return self._stats


class TieredCache:
    """
    Tiered caching system that combines in-memory and distributed caches.
    
    Uses L1 (memory) and L2 (Redis) cache levels for optimal performance.
    """
    
    def __init__(self,
                 l1_cache: Optional[EnhancedLRUCache] = None,
                 l2_cache: Optional[RedisCache] = None,
                 enable_write_through: bool = True,
                 enable_write_back: bool = False):
        
        self.l1_cache = l1_cache or EnhancedLRUCache(max_size=1000)
        self.l2_cache = l2_cache
        self.enable_write_through = enable_write_through
        self.enable_write_back = enable_write_back
        
        self.logger = get_logger(f"{__name__}.TieredCache")
        self._write_back_queue: Dict[str, Any] = {}
        self._write_back_task: Optional[asyncio.Task] = None
        
        if enable_write_back and l2_cache:
            self._start_write_back_task()
    
    def _start_write_back_task(self):
        """Start background write-back task."""
        try:
            loop = asyncio.get_event_loop()
            self._write_back_task = loop.create_task(self._write_back_loop())
        except RuntimeError:
            pass
    
    async def _write_back_loop(self):
        """Background task for write-back operations."""
        while True:
            try:
                await asyncio.sleep(5)  # Write back every 5 seconds
                await self._flush_write_back_queue()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in write-back loop: {str(e)}")
    
    async def _flush_write_back_queue(self):
        """Flush pending write-back operations."""
        if not self._write_back_queue or not self.l2_cache:
            return
        
        items = list(self._write_back_queue.items())
        self._write_back_queue.clear()
        
        for key, value in items:
            try:
                await self.l2_cache.set(key, value)
            except Exception as e:
                self.logger.error(f"Write-back failed for key {key}: {str(e)}")
    
    async def get(self, key: str, default: Any = None) -> Any:
        """Get value from tiered cache (L1 -> L2)."""
        # Try L1 cache first
        value = self.l1_cache.get(key, None)
        if value is not None:
            return value
        
        # Try L2 cache
        if self.l2_cache:
            value = await self.l2_cache.get(key, None)
            if value is not None:
                # Populate L1 cache
                self.l1_cache.set(key, value)
                return value
        
        return default
    
    async def set(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """Set value in tiered cache."""
        # Always set in L1
        self.l1_cache.set(key, value, ttl)
        
        # Handle L2 cache based on strategy
        if self.l2_cache:
            if self.enable_write_through:
                # Write through: immediately write to L2
                try:
                    await self.l2_cache.set(key, value, int(ttl) if ttl else None)
                except Exception as e:
                    self.logger.error(f"Write-through failed for key {key}: {str(e)}")
            
            elif self.enable_write_back:
                # Write back: queue for later write
                self._write_back_queue[key] = value
        
        return True
    
    async def delete(self, key: str) -> bool:
        """Delete key from all cache levels."""
        # Delete from L1
        l1_result = self.l1_cache.delete(key)
        
        # Delete from L2
        l2_result = True
        if self.l2_cache:
            try:
                l2_result = await self.l2_cache.delete(key)
            except Exception as e:
                self.logger.error(f"L2 delete failed for key {key}: {str(e)}")
                l2_result = False
        
        # Remove from write-back queue
        self._write_back_queue.pop(key, None)
        
        return l1_result or l2_result
    
    async def clear(self):
        """Clear all cache levels."""
        self.l1_cache.clear()
        self._write_back_queue.clear()
        
        if self.l2_cache:
            try:
                await self.l2_cache.clear()
            except Exception as e:
                self.logger.error(f"L2 clear failed: {str(e)}")
    
    def get_stats(self) -> Dict[str, CacheStats]:
        """Get statistics for all cache levels."""
        stats = {
            "l1": self.l1_cache.get_stats()
        }
        
        if self.l2_cache:
            stats["l2"] = self.l2_cache.get_stats()
        
        return stats


def cache_result(
    cache_instance: Union[EnhancedLRUCache, RedisCache, TieredCache],
    ttl: Optional[float] = None,
    key_func: Optional[Callable] = None,
    skip_cache_if: Optional[Callable] = None
):
    """
    Decorator for caching function results.
    
    Args:
        cache_instance: Cache instance to use
        ttl: Time to live for cached results
        key_func: Function to generate cache key (defaults to function name + args hash)
        skip_cache_if: Function that returns True if caching should be skipped
    """
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # Default key generation
                func_name = f"{func.__module__}.{func.__name__}"
                args_str = str(args) + str(sorted(kwargs.items()))
                key_hash = hashlib.md5(args_str.encode()).hexdigest()[:16]
                cache_key = f"{func_name}:{key_hash}"
            
            # Check if we should skip cache
            if skip_cache_if and skip_cache_if(*args, **kwargs):
                return await func(*args, **kwargs)
            
            # Try to get from cache
            cached_result = await cache_instance.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = await func(*args, **kwargs)
            await cache_instance.set(cache_key, result, ttl)
            
            return result
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                func_name = f"{func.__module__}.{func.__name__}"
                args_str = str(args) + str(sorted(kwargs.items()))
                key_hash = hashlib.md5(args_str.encode()).hexdigest()[:16]
                cache_key = f"{func_name}:{key_hash}"
            
            # Check if we should skip cache
            if skip_cache_if and skip_cache_if(*args, **kwargs):
                return func(*args, **kwargs)
            
            # Try to get from cache (sync only for in-memory cache)
            if hasattr(cache_instance, 'get') and not asyncio.iscoroutinefunction(cache_instance.get):
                cached_result = cache_instance.get(cache_key)
                if cached_result is not None:
                    return cached_result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            
            # Cache result (sync only for in-memory cache)
            if hasattr(cache_instance, 'set') and not asyncio.iscoroutinefunction(cache_instance.set):
                cache_instance.set(cache_key, result, ttl)
            
            return result
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


# Global cache instances
_global_memory_cache: Optional[EnhancedLRUCache] = None
_global_redis_cache: Optional[RedisCache] = None
_global_tiered_cache: Optional[TieredCache] = None


def get_memory_cache() -> EnhancedLRUCache:
    """Get global in-memory cache instance."""
    global _global_memory_cache
    
    if _global_memory_cache is None:
        _global_memory_cache = EnhancedLRUCache(
            max_size=10000,
            default_ttl=3600,  # 1 hour
            max_memory_mb=256   # 256 MB
        )
    
    return _global_memory_cache


def get_redis_cache(redis_url: str = "redis://localhost:6379") -> RedisCache:
    """Get global Redis cache instance."""
    global _global_redis_cache
    
    if _global_redis_cache is None:
        _global_redis_cache = RedisCache(
            redis_url=redis_url,
            key_prefix="ai_assistant:",
            default_ttl=3600
        )
    
    return _global_redis_cache


def get_tiered_cache(redis_url: str = "redis://localhost:6379") -> TieredCache:
    """Get global tiered cache instance."""
    global _global_tiered_cache
    
    if _global_tiered_cache is None:
        memory_cache = get_memory_cache()
        redis_cache = get_redis_cache(redis_url)
        
        _global_tiered_cache = TieredCache(
            l1_cache=memory_cache,
            l2_cache=redis_cache,
            enable_write_through=True,
            enable_write_back=False
        )
    
    return _global_tiered_cache


# Convenience decorators using global caches
def memory_cache(ttl: Optional[float] = None, key_func: Optional[Callable] = None):
    """Decorator for caching with in-memory cache."""
    return cache_result(get_memory_cache(), ttl=ttl, key_func=key_func)


def redis_cache_decorator(ttl: Optional[float] = None, key_func: Optional[Callable] = None):
    """Decorator for caching with Redis cache."""
    return cache_result(get_redis_cache(), ttl=ttl, key_func=key_func)


def tiered_cache(ttl: Optional[float] = None, key_func: Optional[Callable] = None):
    """Decorator for caching with tiered cache."""
    return cache_result(get_tiered_cache(), ttl=ttl, key_func=key_func)