"""
Advanced Redis Cache Implementation for AI Assistant
Author: Drmusab
Last Modified: 2025-06-13 20:42:24 UTC

This module provides comprehensive Redis caching capabilities for the AI assistant,
supporting session management, memory operations, workflow caching, component states,
plugin coordination, and LLM response caching with high performance and reliability.
"""

from pathlib import Path
from typing import Optional, Dict, Any, List, Set, Callable, Type, Union, AsyncGenerator, TypeVar, Tuple
import asyncio
import threading
import time
import json
import pickle
import hashlib
import zlib
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
from contextlib import asynccontextmanager
import uuid
import weakref
from abc import ABC, abstractmethod
import logging
from concurrent.futures import ThreadPoolExecutor
import numpy as np

# Redis imports
try:
    import redis.asyncio as redis
    from redis.asyncio import Redis, ConnectionPool, RedisCluster
    from redis.exceptions import (
        ConnectionError, TimeoutError, AuthenticationError,
        ResponseError, RedisError, ClusterError
    )
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None
    Redis = None
    ConnectionPool = None
    RedisCluster = None

# Core imports
from src.core.config.loader import ConfigLoader
from src.core.events.event_bus import EventBus
from src.core.events.event_types import (
    CacheHit, CacheMiss, CacheEviction, CacheError, CacheHealthChanged,
    ComponentHealthChanged, SessionStateChanged, WorkflowCompleted,
    MemoryOperationCompleted, ErrorOccurred, SystemStateChanged
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
CacheValue = Union[str, bytes, int, float, bool, dict, list, None]


class CacheStrategy(Enum):
    """Cache eviction strategies."""
    LRU = "lru"           # Least Recently Used
    LFU = "lfu"           # Least Frequently Used
    TTL = "ttl"           # Time To Live
    FIFO = "fifo"         # First In, First Out
    RANDOM = "random"     # Random eviction
    ADAPTIVE = "adaptive" # Adaptive based on usage patterns


class SerializationFormat(Enum):
    """Data serialization formats."""
    JSON = "json"         # JSON serialization
    PICKLE = "pickle"     # Python pickle
    MSGPACK = "msgpack"   # MessagePack (if available)
    COMPRESSED = "compressed"  # Compressed pickle
    RAW = "raw"          # Raw string/bytes


class CacheNamespace(Enum):
    """Cache namespaces for different data types."""
    SESSION = "session"
    MEMORY = "memory"
    WORKFLOW = "workflow"
    COMPONENT = "component"
    PLUGIN = "plugin"
    LLM = "llm"
    USER = "user"
    SYSTEM = "system"
    TEMPORARY = "temp"


@dataclass
class CacheConfiguration:
    """Redis cache configuration."""
    # Connection settings
    host: str = "localhost"
    port: int = 6379
    password: Optional[str] = None
    username: Optional[str] = None
    database: int = 0
    ssl: bool = False
    ssl_cert_file: Optional[str] = None
    ssl_key_file: Optional[str] = None
    ssl_ca_certs: Optional[str] = None
    
    # Connection pool settings
    max_connections: int = 100
    min_connections: int = 5
    connection_timeout: float = 5.0
    socket_timeout: float = 5.0
    socket_keepalive: bool = True
    socket_keepalive_options: Dict[str, int] = field(default_factory=dict)
    
    # Cluster settings
    cluster_mode: bool = False
    cluster_nodes: List[Dict[str, Any]] = field(default_factory=list)
    cluster_max_redirections: int = 16
    cluster_read_from_replicas: bool = True
    
    # Cache behavior
    default_ttl: int = 3600  # 1 hour
    max_ttl: int = 86400     # 24 hours
    key_prefix: str = "ai_assistant"
    default_serialization: SerializationFormat = SerializationFormat.JSON
    compression_enabled: bool = True
    compression_threshold: int = 1024  # Compress data larger than 1KB
    
    # Performance settings
    pipeline_size: int = 100
    batch_size: int = 50
    max_memory_usage: int = 1024 * 1024 * 1024  # 1GB
    eviction_policy: CacheStrategy = CacheStrategy.LRU
    
    # Health and monitoring
    health_check_interval: float = 30.0
    performance_monitoring: bool = True
    enable_metrics: bool = True
    
    # Retry and error handling
    max_retries: int = 3
    retry_delay: float = 0.1
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: float = 60.0


@dataclass
class CacheEntry:
    """Represents a cache entry with metadata."""
    key: str
    value: Any
    namespace: CacheNamespace
    ttl: Optional[int] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_accessed: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    access_count: int = 0
    serialization_format: SerializationFormat = SerializationFormat.JSON
    compressed: bool = False
    size_bytes: int = 0
    tags: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CacheStatistics:
    """Cache performance statistics."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    errors: int = 0
    total_operations: int = 0
    total_memory_bytes: int = 0
    average_response_time: float = 0.0
    hit_ratio: float = 0.0
    memory_usage_ratio: float = 0.0
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class CacheError(Exception):
    """Custom exception for cache operations."""
    
    def __init__(self, message: str, operation: Optional[str] = None, 
                 key: Optional[str] = None, error_code: Optional[str] = None):
        super().__init__(message)
        self.operation = operation
        self.key = key
        self.error_code = error_code
        self.timestamp = datetime.now(timezone.utc)


class CircuitBreaker:
    """Circuit breaker for Redis connection failures."""
    
    def __init__(self, failure_threshold: int = 5, timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.lock = threading.Lock()
    
    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        with self.lock:
            if self.state == "OPEN":
                if self._should_attempt_reset():
                    self.state = "HALF_OPEN"
                else:
                    raise CacheError("Circuit breaker is OPEN", error_code="CIRCUIT_BREAKER_OPEN")
            
            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
            except Exception as e:
                self._on_failure()
                raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset."""
        if self.last_failure_time is None:
            return True
        
        time_since_failure = (datetime.now(timezone.utc) - self.last_failure_time).total_seconds()
        return time_since_failure >= self.timeout
    
    def _on_success(self) -> None:
        """Handle successful operation."""
        self.failure_count = 0
        self.state = "CLOSED"
    
    def _on_failure(self) -> None:
        """Handle failed operation."""
        self.failure_count += 1
        self.last_failure_time = datetime.now(timezone.utc)
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"


class Serializer:
    """Advanced serialization handler for cache data."""
    
    def __init__(self, compression_enabled: bool = True, compression_threshold: int = 1024):
        self.compression_enabled = compression_enabled
        self.compression_threshold = compression_threshold
        self.logger = get_logger(__name__)
    
    def serialize(self, data: Any, format_type: SerializationFormat) -> Tuple[bytes, bool]:
        """
        Serialize data to bytes.
        
        Returns:
            Tuple of (serialized_data, is_compressed)
        """
        try:
            if format_type == SerializationFormat.JSON:
                serialized = json.dumps(data, default=self._json_serializer).encode('utf-8')
            elif format_type == SerializationFormat.PICKLE:
                serialized = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
            elif format_type == SerializationFormat.RAW:
                if isinstance(data, (str, bytes)):
                    serialized = data.encode('utf-8') if isinstance(data, str) else data
                else:
                    raise ValueError("RAW format only supports str or bytes")
            else:
                # Default to pickle for unknown formats
                serialized = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Apply compression if enabled and data is large enough
            is_compressed = False
            if (self.compression_enabled and 
                len(serialized) > self.compression_threshold):
                serialized = zlib.compress(serialized)
                is_compressed = True
            
            return serialized, is_compressed
            
        except Exception as e:
            raise CacheError(f"Serialization failed: {str(e)}", operation="serialize")
    
    def deserialize(self, data: bytes, format_type: SerializationFormat, 
                   is_compressed: bool = False) -> Any:
        """Deserialize bytes back to Python object."""
        try:
            # Decompress if necessary
            if is_compressed:
                data = zlib.decompress(data)
            
            if format_type == SerializationFormat.JSON:
                return json.loads(data.decode('utf-8'))
            elif format_type == SerializationFormat.PICKLE:
                return pickle.loads(data)
            elif format_type == SerializationFormat.RAW:
                return data.decode('utf-8')
            else:
                # Default to pickle
                return pickle.loads(data)
                
        except Exception as e:
            raise CacheError(f"Deserialization failed: {str(e)}", operation="deserialize")
    
    def _json_serializer(self, obj: Any) -> Any:
        """Custom JSON serializer for complex objects."""
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, set):
            return list(obj)
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        else:
            return str(obj)


class RedisCache:
    """
    Advanced Redis Cache Implementation for AI Assistant.
    
    This cache provides high-performance data storage and retrieval for all
    core system components including sessions, memory systems, workflows,
    components, plugins, and LLM responses.
    
    Features:
    - High-performance async operations with connection pooling
    - Advanced serialization with compression support
    - Multiple eviction strategies and TTL management
    - Redis Cluster support for horizontal scaling
    - Circuit breaker pattern for fault tolerance
    - Comprehensive metrics and health monitoring
    - Event-driven cache invalidation
    - Namespace-based data organization
    - Background maintenance and optimization
    """
    
    def __init__(self, container: Container, config: Optional[CacheConfiguration] = None):
        """
        Initialize the Redis cache.
        
        Args:
            container: Dependency injection container
            config: Cache configuration (uses defaults if None)
        """
        if not REDIS_AVAILABLE:
            raise ImportError("Redis package is required but not installed")
        
        self.container = container
        self.config = config or CacheConfiguration()
        self.logger = get_logger(__name__)
        
        # Core services
        self.config_loader = container.get(ConfigLoader)
        self.event_bus = container.get(EventBus)
        self.error_handler = container.get(ErrorHandler)
        self.health_check = container.get(HealthCheck)
        
        # Observability
        try:
            self.metrics = container.get(MetricsCollector)
            self.tracer = container.get(TraceManager)
        except Exception:
            self.metrics = None
            self.tracer = None
        
        # Redis connections
        self.redis_client: Optional[Union[Redis, RedisCluster]] = None
        self.connection_pool: Optional[ConnectionPool] = None
        
        # Cache components
        self.serializer = Serializer(
            self.config.compression_enabled,
            self.config.compression_threshold
        )
        self.circuit_breaker = CircuitBreaker(
            self.config.circuit_breaker_threshold,
            self.config.circuit_breaker_timeout
        )
        
        # State management
        self.is_connected = False
        self.is_healthy = False
        self.last_health_check: Optional[datetime] = None
        
        # Statistics and monitoring
        self.statistics = CacheStatistics()
        self.performance_history: deque = deque(maxlen=1000)
        
        # Background tasks
        self.background_tasks: List[asyncio.Task] = []
        self.shutdown_event = asyncio.Event()
        
        # Key management
        self.active_keys: Set[str] = set()
        self.key_locks: Dict[str, asyncio.Lock] = {}
        
        # Load configuration overrides
        self._load_config_overrides()
        
        # Setup monitoring
        self._setup_monitoring()
        
        # Register health check
        self.health_check.register_component("redis_cache", self._health_check_callback)
        
        self.logger.info("RedisCache initialized successfully")

    def _load_config_overrides(self) -> None:
        """Load configuration overrides from config loader."""
        try:
            redis_config = self.config_loader.get("redis", {})
            
            # Override configuration with values from config loader
            for key, value in redis_config.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
            
            self.logger.debug("Loaded Redis configuration overrides")
            
        except Exception as e:
            self.logger.warning(f"Failed to load config overrides: {str(e)}")

    def _setup_monitoring(self) -> None:
        """Setup monitoring and metrics collection."""
        if not self.metrics:
            return
        
        try:
            # Register cache metrics
            self.metrics.register_counter("cache_operations_total")
            self.metrics.register_counter("cache_hits_total")
            self.metrics.register_counter("cache_misses_total")
            self.metrics.register_counter("cache_errors_total")
            self.metrics.register_counter("cache_evictions_total")
            self.metrics.register_histogram("cache_operation_duration_seconds")
            self.metrics.register_gauge("cache_memory_usage_bytes")
            self.metrics.register_gauge("cache_connection_pool_size")
            self.metrics.register_gauge("cache_hit_ratio")
            
        except Exception as e:
            self.logger.warning(f"Failed to setup monitoring: {str(e)}")

    async def initialize(self) -> None:
        """Initialize Redis connection and start background tasks."""
        try:
            await self._create_redis_connection()
            await self._verify_connection()
            
            # Start background tasks
            self.background_tasks.extend([
                asyncio.create_task(self._health_monitor_loop()),
                asyncio.create_task(self._statistics_update_loop()),
                asyncio.create_task(self._maintenance_loop()),
                asyncio.create_task(self._key_expiration_monitor())
            ])
            
            # Register event handlers
            await self._register_event_handlers()
            
            self.is_connected = True
            self.is_healthy = True
            
            self.logger.info("Redis cache initialized and connected successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Redis cache: {str(e)}")
            raise CacheError(f"Initialization failed: {str(e)}")

    async def _create_redis_connection(self) -> None:
        """Create Redis connection based on configuration."""
        try:
            if self.config.cluster_mode:
                # Redis Cluster mode
                startup_nodes = self.config.cluster_nodes or [
                    {"host": self.config.host, "port": self.config.port}
                ]
                
                self.redis_client = RedisCluster(
                    startup_nodes=startup_nodes,
                    password=self.config.password,
                    username=self.config.username,
                    socket_timeout=self.config.socket_timeout,
                    socket_connect_timeout=self.config.connection_timeout,
                    socket_keepalive=self.config.socket_keepalive,
                    socket_keepalive_options=self.config.socket_keepalive_options,
                    max_connections=self.config.max_connections,
                    ssl=self.config.ssl,
                    ssl_cert_file=self.config.ssl_cert_file,
                    ssl_key_file=self.config.ssl_key_file,
                    ssl_ca_certs=self.config.ssl_ca_certs,
                    read_from_replicas=self.config.cluster_read_from_replicas,
                    max_redirections=self.config.cluster_max_redirections,
                    decode_responses=False  # We handle encoding/decoding manually
                )
            else:
                # Standard Redis mode
                self.connection_pool = ConnectionPool(
                    host=self.config.host,
                    port=self.config.port,
                    password=self.config.password,
                    username=self.config.username,
                    db=self.config.database,
                    socket_timeout=self.config.socket_timeout,
                    socket_connect_timeout=self.config.connection_timeout,
                    socket_keepalive=self.config.socket_keepalive,
                    socket_keepalive_options=self.config.socket_keepalive_options,
                    max_connections=self.config.max_connections,
                    ssl=self.config.ssl,
                    ssl_cert_file=self.config.ssl_cert_file,
                    ssl_key_file=self.config.ssl_key_file,
                    ssl_ca_certs=self.config.ssl_ca_certs,
                    decode_responses=False
                )
                
                self.redis_client = Redis(connection_pool=self.connection_pool)
                
        except Exception as e:
            raise CacheError(f"Failed to create Redis connection: {str(e)}")

    async def _verify_connection(self) -> None:
        """Verify Redis connection is working."""
        try:
            await self.redis_client.ping()
            self.logger.info("Redis connection verified successfully")
        except Exception as e:
            raise CacheError(f"Redis connection verification failed: {str(e)}")

    async def _register_event_handlers(self) -> None:
        """Register event handlers for cache invalidation."""
        # Session events
        self.event_bus.subscribe("session_ended", self._handle_session_ended)
        self.event_bus.subscribe("session_expired", self._handle_session_expired)
        
        # Workflow events
        self.event_bus.subscribe("workflow_completed", self._handle_workflow_completed)
        self.event_bus.subscribe("workflow_failed", self._handle_workflow_failed)
        
        # Component events
        self.event_bus.subscribe("component_health_changed", self._handle_component_health_changed)
        
        # Memory events
        self.event_bus.subscribe("memory_operation_completed", self._handle_memory_operation)
        
        # System events
        self.event_bus.subscribe("system_shutdown_started", self._handle_system_shutdown)

    def _build_key(self, namespace: CacheNamespace, key: str, **kwargs) -> str:
        """Build a namespaced cache key."""
        parts = [self.config.key_prefix, namespace.value, key]
        
        # Add additional key components
        for k, v in kwargs.items():
            if v is not None:
                parts.append(f"{k}:{v}")
        
        return ":".join(parts)

    def _get_key_lock(self, key: str) -> asyncio.Lock:
        """Get or create a lock for a specific key."""
        if key not in self.key_locks:
            self.key_locks[key] = asyncio.Lock()
        return self.key_locks[key]

    @handle_exceptions
    async def set(
        self,
        key: str,
        value: Any,
        namespace: CacheNamespace = CacheNamespace.SYSTEM,
        ttl: Optional[int] = None,
        serialization: Optional[SerializationFormat] = None,
        tags: Optional[Set[str]] = None,
        **kwargs
    ) -> bool:
        """
        Set a value in the cache.
        
        Args:
            key: Cache key
            value: Value to cache
            namespace: Cache namespace
            ttl: Time to live in seconds
            serialization: Serialization format
            tags: Optional tags for the entry
            **kwargs: Additional key components
            
        Returns:
            Success status
        """
        if not self.is_connected:
            await self.initialize()
        
        operation_start = time.time()
        cache_key = self._build_key(namespace, key, **kwargs)
        
        async with self._get_key_lock(cache_key):
            try:
                with self.tracer.trace("cache_set") if self.tracer else None as span:
                    if span:
                        span.set_attributes({
                            "cache.operation": "set",
                            "cache.key": cache_key,
                            "cache.namespace": namespace.value
                        })
                    
                    # Serialize data
                    serialization_format = serialization or self.config.default_serialization
                    serialized_data, is_compressed = self.serializer.serialize(value, serialization_format)
                    
                    # Calculate TTL
                    ttl_seconds = ttl or self.config.default_ttl
                    ttl_seconds = min(ttl_seconds, self.config.max_ttl)
                    
                    # Create cache entry metadata
                    entry_metadata = {
                        'namespace': namespace.value,
                        'serialization_format': serialization_format.value,
                        'compressed': is_compressed,
                        'created_at': datetime.now(timezone.utc).isoformat(),
                        'tags': list(tags) if tags else [],
                        'size_bytes': len(serialized_data)
                    }
                    
                    # Store both data and metadata
                    pipe = self.redis_client.pipeline()
                    pipe.setex(cache_key, ttl_seconds, serialized_data)
                    pipe.setex(f"{cache_key}:meta", ttl_seconds, json.dumps(entry_metadata))
                    
                    # Add to tag indexes if tags provided
                    if tags:
                        for tag in tags:
                            tag_key = self._build_key(CacheNamespace.SYSTEM, f"tag:{tag}")
                            pipe.sadd(tag_key, cache_key)
                            pipe.expire(tag_key, ttl_seconds)
                    
                    await pipe.execute()
                    
                    # Track the key
                    self.active_keys.add(cache_key)
                    
                    # Update statistics
                    operation_time = time.time() - operation_start
                    self.statistics.total_operations += 1
                    self.statistics.total_memory_bytes += len(serialized_data)
                    self._update_performance_history("set", operation_time, True)
                    
                    # Update metrics
                    if self.metrics:
                        self.metrics.increment("cache_operations_total", tags={"operation": "set"})
                        self.metrics.record("cache_operation_duration_seconds", operation_time)
                    
                    # Emit cache event
                    await self.event_bus.emit(CacheHit(
                        operation="set",
                        key=cache_key,
                        namespace=namespace.value,
                        hit=True,
                        response_time=operation_time
                    ))
                    
                    self.logger.debug(f"Cached value for key: {cache_key}")
                    return True
                    
            except Exception as e:
                operation_time = time.time() - operation_start
                self.statistics.errors += 1
                self._update_performance_history("set", operation_time, False)
                
                if self.metrics:
                    self.metrics.increment("cache_errors_total", tags={"operation": "set"})
                
                await self.event_bus.emit(CacheError(
                    operation="set",
                    key=cache_key,
                    error_message=str(e)
                ))
                
                self.logger.error(f"Failed to set cache key {cache_key}: {str(e)}")
                raise CacheError(f"Set operation failed: {str(e)}", "set", cache_key)

    @handle_exceptions
    async def get(
        self,
        key: str,
        namespace: CacheNamespace = CacheNamespace.SYSTEM,
        default: Any = None,
        **kwargs
    ) -> Any:
        """
        Get a value from the cache.
        
        Args:
            key: Cache key
            namespace: Cache namespace
            default: Default value if key not found
            **kwargs: Additional key components
            
        Returns:
            Cached value or default
        """
        if not self.is_connected:
            await self.initialize()
        
        operation_start = time.time()
        cache_key = self._build_key(namespace, key, **kwargs)
        
        try:
            with self.tracer.trace("cache_get") if self.tracer else None as span:
                if span:
                    span.set_attributes({
                        "cache.operation": "get",
                        "cache.key": cache_key,
                        "cache.namespace": namespace.value
                    })
                
                # Get data and metadata
                pipe = self.redis_client.pipeline()
                pipe.get(cache_key)
                pipe.get(f"{cache_key}:meta")
                results = await pipe.execute()
                
                serialized_data, metadata_json = results
                
                if serialized_data is None:
                    # Cache miss
                    operation_time = time.time() - operation_start
                    self.statistics.misses += 1
                    self.statistics.total_operations += 1
                    self._update_performance_history("get", operation_time, False)
                    
                    if self.metrics:
                        self.metrics.increment("cache_misses_total")
                        self.metrics.increment("cache_operations_total", tags={"operation": "get"})
                    
                    await self.event_bus.emit(CacheMiss(
                        operation="get",
                        key=cache_key,
                        namespace=namespace.value
                    ))
                    
                    return default
                
                # Parse metadata
                if metadata_json:
                    try:
                        metadata = json.loads(metadata_json)
                        serialization_format = SerializationFormat(metadata['serialization_format'])
                        is_compressed = metadata['compressed']
                    except (json.JSONDecodeError, KeyError, ValueError):
                        # Fallback to default serialization
                        serialization_format = self.config.default_serialization
                        is_compressed = False
                else:
                    serialization_format = self.config.default_serialization
                    is_compressed = False
                
                # Deserialize data
                value = self.serializer.deserialize(serialized_data, serialization_format, is_compressed)
                
                # Update access information
                await self.redis_client.hincrby(f"{cache_key}:stats", "access_count", 1)
                await self.redis_client.hset(
                    f"{cache_key}:stats", 
                    "last_accessed", 
                    datetime.now(timezone.utc).isoformat()
                )
                
                # Update statistics
                operation_time = time.time() - operation_start
                self.statistics.hits += 1
                self.statistics.total_operations += 1
                self._update_performance_history("get", operation_time, True)
                
                # Update metrics
                if self.metrics:
                    self.metrics.increment("cache_hits_total")
                    self.metrics.increment("cache_operations_total", tags={"operation": "get"})
                    self.metrics.record("cache_operation_duration_seconds", operation_time)
                
                # Emit cache event
                await self.event_bus.emit(CacheHit(
                    operation="get",
                    key=cache_key,
                    namespace=namespace.value,
                    hit=True,
                    response_time=operation_time
                ))
                
                return value
                
        except Exception as e:
            operation_time = time.time() - operation_start
            self.statistics.errors += 1
            self._update_performance_history("get", operation_time, False)
            
            if self.metrics:
                self.metrics.increment("cache_errors_total", tags={"operation": "get"})
            
            await self.event_bus.emit(CacheError(
                operation="get",
                key=cache_key,
                error_message=str(e)
            ))
            
            self.logger.error(f"Failed to get cache key {cache_key}: {str(e)}")
            return default

    @handle_exceptions
    async def delete(
        self,
        key: str,
        namespace: CacheNamespace = CacheNamespace.SYSTEM,
        **kwargs
    ) -> bool:
        """
        Delete a key from the cache.
        
        Args:
            key: Cache key
            namespace: Cache namespace
            **kwargs: Additional key components
            
        Returns:
            Success status
        """
        if not self.is_connected:
            await self.initialize()
        
        operation_start = time.time()
        cache_key = self._build_key(namespace, key, **kwargs)
        
        async with self._get_key_lock(cache_key):
            try:
                with self.tracer.trace("cache_delete") if self.tracer else None:
                    # Delete main key, metadata, and stats
                    pipe = self.redis_client.pipeline()
                    pipe.delete(cache_key)
                    pipe.delete(f"{cache_key}:meta")
                    pipe.delete(f"{cache_key}:stats")
                    results = await pipe.execute()
                    
                    deleted_count = sum(results)
                    success = deleted_count > 0
                    
                    # Remove from active keys
                    self.active_keys.discard(cache_key)
                    
                    # Remove key lock if it exists
                    self.key_locks.pop(cache_key, None)
                    
                    # Update statistics
                    operation_time = time.time() - operation_start
                    self.statistics.total_operations += 1
                    self._update_performance_history("delete", operation_time, success)
                    
                    # Update metrics
                    if self.metrics:
                        self.metrics.increment("cache_operations_total", tags={"operation": "delete"})
                        self.metrics.record("cache_operation_duration_seconds", operation_time)
                    
                    if success:
                        self.logger.debug(f"Deleted cache key: {cache_key}")
                    else:
                        self.logger.debug(f"Cache key not found for deletion: {cache_key}")
                    
                    return success
                    
            except Exception as e:
                operation_time = time.time() - operation_start
                self.statistics.errors += 1
                self._update_performance_history("delete", operation_time, False)
                
                if self.metrics:
                    self.metrics.increment("cache_errors_total", tags={"operation": "delete"})
                
                self.logger.error(f"Failed to delete cache key {cache_key}: {str(e)}")
                raise CacheError(f"Delete operation failed: {str(e)}", "delete", cache_key)

    @handle_exceptions
    async def exists(
        self,
        key: str,
        namespace: CacheNamespace = CacheNamespace.SYSTEM,
        **kwargs
    ) -> bool:
        """
        Check if a key exists in the cache.
        
        Args:
            key: Cache key
            namespace: Cache namespace
            **kwargs: Additional key components
            
        Returns:
            True if key exists
        """
        if not self.is_connected:
            await self.initialize()
        
        cache_key = self._build_key(namespace, key, **kwargs)
        
        try:
            result = await self.redis_client.exists(cache_key)
            return bool(result)
        except Exception as e:
            self.logger.error(f"Failed to check existence of cache key {cache_key}: {str(e)}")
            return False

    @handle_exceptions
    async def expire(
        self,
        key: str,
        ttl: int,
        namespace: CacheNamespace = CacheNamespace.SYSTEM,
        **kwargs
    ) -> bool:
        """
        Set expiration time for a key.
        
        Args:
            key: Cache key
            ttl: Time to live in seconds
            namespace: Cache namespace
            **kwargs: Additional key components
            
        Returns:
            Success status
        """
        if not self.is_connected:
            await self.initialize()
        
        cache_key = self._build_key(namespace, key, **kwargs)
        
        try:
            result = await self.redis_client.expire(cache_key, ttl)
            if result:
                await self.redis_client.expire(f"{cache_key}:meta", ttl)
                await self.redis_client.expire(f"{cache_key}:stats", ttl)
            
            return bool(result)
        except Exception as e:
            self.logger.error(f"Failed to set expiration for cache key {cache_key}: {str(e)}")
            return False

    @handle_exceptions
    async def get_many(
        self,
        keys: List[str],
        namespace: CacheNamespace = CacheNamespace.SYSTEM,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Get multiple values from the cache.
        
        Args:
            keys: List of cache keys
            namespace: Cache namespace
            **kwargs: Additional key components
            
        Returns:
            Dictionary of key-value pairs
        """
        if not self.is_connected:
            await self.initialize()
        
        if not keys:
            return {}
        
        operation_start = time.time()
        cache_keys = [self._build_key(namespace, key, **kwargs) for key in keys]
        
        try:
            with self.tracer.trace("cache_get_many") if self.tracer else None:
                # Get all data and metadata in batches
                results = {}
                
                for i in range(0, len(cache_keys), self.config.batch_size):
                    batch_keys = cache_keys[i:i + self.config.batch_size]
                    
                    # Create pipeline for batch
                    pipe = self.redis_client.pipeline()
                    for cache_key in batch_keys:
                        pipe.get(cache_key)
                        pipe.get(f"{cache_key}:meta")
                    
                    batch_results = await pipe.execute()
                    
                    # Process batch results
                    for j, cache_key in enumerate(batch_keys):
                        data_idx = j * 2
                        meta_idx = j * 2 + 1
                        
                        serialized_data = batch_results[data_idx]
                        metadata_json = batch_results[meta_idx]
                        
                        if serialized_data is not None:
                            try:
                                # Parse metadata
                                if metadata_json:
                                    metadata = json.loads(metadata_json)
                                    serialization_format = SerializationFormat(metadata['serialization_format'])
                                    is_compressed = metadata['compressed']
                                else:
                                    serialization_format = self.config.default_serialization
                                    is_compressed = False
                                
                                # Deserialize data
                                value = self.serializer.deserialize(
                                    serialized_data, serialization_format, is_compressed
                                )
                                
                                # Map back to original key
                                original_key = keys[cache_keys.index(cache_key)]
                                results[original_key] = value
                                
                                self.statistics.hits += 1
                                
                            except Exception as e:
                                self.logger.warning(f"Failed to deserialize cache key {cache_key}: {str(e)}")
                                self.statistics.errors += 1
                        else:
                            self.statistics.misses += 1
                
                # Update statistics
                operation_time = time.time() - operation_start
                self.statistics.total_operations += len(keys)
                self._update_performance_history("get_many", operation_time, len(results) > 0)
                
                # Update metrics
                if self.metrics:
                    self.metrics.increment("cache_operations_total", 
                                         tags={"operation": "get_many"}, 
                                         value=len(keys))
                    self.metrics.record("cache_operation_duration_seconds", operation_time)
                
                return results
                
        except Exception as e:
            operation_time = time.time() - operation_start
            self.statistics.errors += len(keys)
            self._update_performance_history("get_many", operation_time, False)
            
            if self.metrics:
                self.metrics.increment("cache_errors_total", tags={"operation": "get_many"})
            
            self.logger.error(f"Failed to get multiple cache keys: {str(e)}")
            return {}

    @handle_exceptions
    async def set_many(
        self,
        mapping: Dict[str, Any],
        namespace: CacheNamespace = CacheNamespace.SYSTEM,
        ttl: Optional[int] = None,
        **kwargs
    ) -> bool:
        """
        Set multiple values in the cache.
        
        Args:
            mapping: Dictionary of key-value pairs
            namespace: Cache namespace
            ttl: Time to live in seconds
            **kwargs: Additional key components
            
        Returns:
            Success status
        """
        if not self.is_connected:
            await self.initialize()
        
        if not mapping:
            return True
        
        operation_start = time.time()
        ttl_seconds = ttl or self.config.default_ttl
        
        try:
            with self.tracer.trace("cache_set_many") if self.tracer else None:
                # Process in batches
                for i in range(0, len(mapping), self.config.batch_size):
                    batch_items = list(mapping.items())[i:i + self.config.batch_size]
                    
                    pipe = self.redis_client.pipeline()
                    
                    for key, value in batch_items:
                        cache_key = self._build_key(namespace, key, **kwargs)
                        
                        # Serialize data
                        serialized_data, is_compressed = self.serializer.serialize(
                            value, self.config.default_serialization
                        )
                        
                        # Create metadata
                        entry_metadata = {
                            'namespace': namespace.value,
                            'serialization_format': self.config.default_serialization.value,
                            'compressed': is_compressed,
                            'created_at': datetime.now(timezone.utc).isoformat(),
                            'size_bytes': len(serialized_data)
                        }
                        
                        # Add to pipeline
                        pipe.setex(cache_key, ttl_seconds, serialized_data)
                        pipe.setex(f"{cache_key}:meta", ttl_seconds, json.dumps(entry_metadata))
                        
                        # Track the key
                        self.active_keys.add(cache_key)
                    
                    await pipe.execute()
                
                # Update statistics
                operation_time = time.time() - operation_start
                self.statistics.total_operations += len(mapping)
                self._update_performance_history("set_many", operation_time, True)
                
                # Update metrics
                if self.metrics:
                    self.metrics.increment("cache_operations_total", 
                                         tags={"operation": "set_many"}, 
                                         value=len(mapping))
                    self.metrics.record("cache_operation_duration_seconds", operation_time)
                
                self.logger.debug(f"Set {len(mapping)} cache entries")
                return True
                
        except Exception as e:
            operation_time = time.time() - operation_start
            self.statistics.errors += len(mapping)
            self._update_performance_history("set_many", operation_time, False)
            
            if self.metrics:
                self.metrics.increment("cache_errors_total", tags={"operation": "set_many"})
            
            self.logger.error(f"Failed to set multiple cache keys: {str(e)}")
            raise CacheError(f"Set many operation failed: {str(e)}", "set_many")

    @handle_exceptions
    async def delete_by_pattern(
        self,
        pattern: str,
        namespace: CacheNamespace = CacheNamespace.SYSTEM,
        **kwargs
    ) -> int:
        """
        Delete keys matching a pattern.
        
        Args:
            pattern: Key pattern (supports wildcards)
            namespace: Cache namespace
            **kwargs: Additional key components
            
        Returns:
            Number of keys deleted
        """
        if not self.is_connected:
            await self.initialize()
        
        search_pattern = self._build_key(namespace, pattern, **kwargs)
        
        try:
            # Find matching keys
            matching_keys = []
            cursor = 0
            
            while True:
                cursor, keys = await self.redis_client.scan(
                    cursor=cursor,
                    match=search_pattern,
                    count=self.config.batch_size
                )
                
                matching_keys.extend([key.decode() if isinstance(key, bytes) else key for key in keys])
                
                if cursor == 0:
                    break
            
            if not matching_keys:
                return 0
            
            # Delete keys in batches
            deleted_count = 0
            for i in range(0, len(matching_keys), self.config.batch_size):
                batch_keys = matching_keys[i:i + self.config.batch_size]
                
                pipe = self.redis_client.pipeline()
                for key in batch_keys:
                    pipe.delete(key)
                    pipe.delete(f"{key}:meta")
                    pipe.delete(f"{key}:stats")
                    
                    # Remove from tracking
                    self.active_keys.discard(key)
                    self.key_locks.pop(key, None)
                
                results = await pipe.execute()
                deleted_count += sum(1 for r in results if r > 0)
            
            # Emit eviction event
            await self.event_bus.emit(CacheEviction(
                pattern=search_pattern,
                keys_evicted=deleted_count,
                namespace=namespace.value
            ))
            
            self.logger.info(f"Deleted {deleted_count} keys matching pattern: {search_pattern}")
            return deleted_count
            
        except Exception as e:
            self.logger.error(f"Failed to delete keys by pattern {search_pattern}: {str(e)}")
            raise CacheError(f"Delete by pattern failed: {str(e)}", "delete_by_pattern")

    @handle_exceptions
    async def delete_by_tags(
        self,
        tags: Set[str],
        namespace: CacheNamespace = CacheNamespace.SYSTEM
    ) -> int:
        """
        Delete keys by tags.
        
        Args:
            tags: Set of tags
            namespace: Cache namespace
            
        Returns:
            Number of keys deleted
        """
        if not self.is_connected:
            await self.initialize()
        
        try:
            deleted_count = 0
            
            for tag in tags:
                tag_key = self._build_key(CacheNamespace.SYSTEM, f"tag:{tag}")
                
                # Get all keys with this tag
                tagged_keys = await self.redis_client.smembers(tag_key)
                
                if tagged_keys:
                    # Delete keys in batches
                    for i in range(0, len(tagged_keys), self.config.batch_size):
                        batch_keys = list(tagged_keys)[i:i + self.config.batch_size]
                        
                        pipe = self.redis_client.pipeline()
                        for key in batch_keys:
                            key_str = key.decode() if isinstance(key, bytes) else key
                            pipe.delete(key_str)
                            pipe.delete(f"{key_str}:meta")
                            pipe.delete(f"{key_str}:stats")
                            
                            # Remove from tracking
                            self.active_keys.discard(key_str)
                            self.key_locks.pop(key_str, None)
                        
                        results = await pipe.execute()
                        deleted_count += sum(1 for r in results if r > 0)
                
                # Delete the tag set itself
                await self.redis_client.delete(tag_key)
            
            self.logger.info(f"Deleted {deleted_count} keys with tags: {tags}")
            return deleted_count
            
        except Exception as e:
            self.logger.error(f"Failed to delete keys by tags {tags}: {str(e)}")
            raise CacheError(f"Delete by tags failed: {str(e)}", "delete_by_tags")

    @handle_exceptions
    async def clear_namespace(self, namespace: CacheNamespace) -> int:
        """
        Clear all keys in a namespace.
        
        Args:
            namespace: Cache namespace to clear
            
        Returns:
            Number of keys deleted
        """
        pattern = f"{self.config.key_prefix}:{namespace.value}:*"
        return await self.delete_by_pattern(pattern.split(":")[-1], namespace)

    @handle_exceptions
    async def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        try:
            # Get Redis info
            redis_info = {}
            if not self.config.cluster_mode:
                redis_info = await self.redis_client.info()
            
            # Calculate hit ratio
            total_requests = self.statistics.hits + self.statistics.misses
            hit_ratio = self.statistics.hits / max(total_requests, 1)
            
            # Get memory usage
            memory_usage = redis_info.get('used_memory', 0) if redis_info else 0
            
            # Recent performance
            recent_operations = list(self.performance_history)[-100:]  # Last 100 operations
            avg_response_time = (
                sum(op['duration'] for op in recent_operations) / max(len(recent_operations), 1)
            )
            
            return {
                'hits': self.statistics.hits,
                'misses': self.statistics.misses,
                'hit_ratio': hit_ratio,
                'total_operations': self.statistics.total_operations,
                'errors': self.statistics.errors,
                'evictions': self.statistics.evictions,
                'memory_usage_bytes': memory_usage,
                'average_response_time': avg_response_time,
                'active_keys': len(self.active_keys),
                'connection_pool_size': getattr(self.connection_pool, 'created_connections', 0),
                'is_connected': self.is_connected,
                'is_healthy': self.is_healthy,
                'last_health_check': self.last_health_check.isoformat() if self.last_health_check else None,
                'redis_info': redis_info
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get cache statistics: {str(e)}")
            return {
                'error': str(e),
                'is_connected': self.is_connected,
                'is_healthy': self.is_healthy
            }

    def _update_performance_history(self, operation: str, duration: float, success: bool) -> None:
        """Update performance history."""
        self.performance_history.append({
            'operation': operation,
            'duration': duration,
            'success': success,
            'timestamp': datetime.now(timezone.utc)
        })

    # Event handlers
    async def _handle_session_ended(self, event) -> None:
        """Handle session ended events."""
        try:
            session_pattern = f"session:{event.session_id}:*"
            deleted = await self.delete_by_pattern(session_pattern, CacheNamespace.SESSION)
            self.logger.debug(f"Cleaned up {deleted} session cache entries for {event.session_id}")
        except Exception as e:
            self.logger.warning(f"Failed to cleanup session cache: {str(e)}")

    async def _handle_session_expired(self, event) -> None:
        """Handle session expired events."""
        await self._handle_session_ended(event)

    async def _handle_workflow_completed(self, event) -> None:
        """Handle workflow completion events."""
        try:
            # Keep workflow results for a shorter time
            workflow_keys = await self.redis_client.keys(
                f"{self.config.key_prefix}:workflow:{event.workflow_id}:*"
            )
            
            if workflow_keys:
                # Set shorter TTL for completed workflow data
                pipe = self.redis_client.pipeline()
                for key in workflow_keys:
                    pipe.expire(key, 3600)  # 1 hour
                await pipe.execute()
                
        except Exception as e:
            self.logger.warning(f"Failed to update workflow cache TTL: {str(e)}")

    async def _handle_workflow_failed(self, event) -> None:
        """Handle workflow failure events."""
        try:
            # Clean up failed workflow data after a delay
            workflow_pattern = f"workflow:{event.workflow_id}:*"
            await asyncio.sleep(300)  # Wait 5 minutes
            deleted = await self.delete_by_pattern(workflow_pattern, CacheNamespace.WORKFLOW)
            self.logger.debug(f"Cleaned up {deleted} failed workflow cache entries")
        except Exception as e:
            self.logger.warning(f"Failed to cleanup failed workflow cache: {str(e)}")

    async def _handle_component_health_changed(self, event) -> None:
        """Handle component health change events."""
        try:
            # Cache component health status
            await self.set(
                f"health:{event.component}",
                {
                    'healthy': event.healthy,
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'details': getattr(event, 'details', {})
                },
                namespace=CacheNamespace.COMPONENT,
                ttl=300  # 5 minutes
            )
        except Exception as e:
            self.logger.warning(f"Failed to cache component health: {str(e)}")

    async def _handle_memory_operation(self, event) -> None:
        """Handle memory operation events."""
        try:
            # Cache memory operation results with appropriate TTL
            ttl = 3600  # 1 hour for memory operations
            
            if hasattr(event, 'result') and event.result:
                await self.set(
                    f"memory_op:{event.operation_id}",
                    event.result,
                    namespace=CacheNamespace.MEMORY,
                    ttl=ttl
                )
        except Exception as e:
            self.logger.warning(f"Failed to cache memory operation result: {str(e)}")

    async def _handle_system_shutdown(self, event) -> None:
        """Handle system shutdown events."""
        await self.shutdown()

    # Background tasks
    async def _health_monitor_loop(self) -> None:
        """Background task for health monitoring."""
        while not self.shutdown_event.is_set():
            try:
                await self._perform_health_check()
                await asyncio.sleep(self.config.health_check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Health monitor error: {str(e)}")
                await asyncio.sleep(self.config.health_check_interval)

    async def _perform_health_check(self) -> None:
        """Perform comprehensive health check."""
        try:
            # Test basic connectivity
            start_time = time.time()
            await self.redis_client.ping()
            response_time = time.time() - start_time
            
            # Check memory usage
            if not self.config.cluster_mode:
                info = await self.redis_client.info('memory')
                memory_usage = info.get('used_memory', 0)
                memory_limit = self.config.max_memory_usage
                memory_ratio = memory_usage / memory_limit if memory_limit > 0 else 0
            else:
                memory_ratio = 0  # Can't easily check in cluster mode
            
            # Update health status
            is_healthy = (
                response_time < 1.0 and  # Response time under 1 second
                memory_ratio < 0.9       # Memory usage under 90%
            )
            
            if self.is_healthy != is_healthy:
                self.is_healthy = is_healthy
                
                await self.event_bus.emit(CacheHealthChanged(
                    healthy=is_healthy,
                    response_time=response_time,
                    memory_ratio=memory_ratio
                ))
            
            self.last_health_check = datetime.now(timezone.utc)
            
            # Update metrics
            if self.metrics:
                self.metrics.set("cache_response_time_seconds", response_time)
                self.metrics.set("cache_memory_usage_ratio", memory_ratio)
                self.metrics.set("cache_healthy", 1 if is_healthy else 0)
            
        except Exception as e:
            self.is_healthy = False
            self.logger.error(f"Health check failed: {str(e)}")
            
            await self.event_bus.emit(CacheHealthChanged(
                healthy=False,
                error_message=str(e)
            ))

    async def _statistics_update_loop(self) -> None:
        """Background task for updating statistics."""
        while not self.shutdown_event.is_set():
            try:
                # Update hit ratio
                total_requests = self.statistics.hits + self.statistics.misses
                self.statistics.hit_ratio = self.statistics.hits / max(total_requests, 1)
                
                # Update average response time
                if self.performance_history:
                    recent_ops = list(self.performance_history)[-100:]
                    self.statistics.average_response_time = (
                        sum(op['duration'] for op in recent_ops) / len(recent_ops)
                    )
                
                # Update metrics
                if self.metrics:
                    self.metrics.set("cache_hit_ratio", self.statistics.hit_ratio)
                    self.metrics.set("cache_total_operations", self.statistics.total_operations)
                    self.metrics.set("cache_active_keys", len(self.active_keys))
                
                self.statistics.last_updated = datetime.now(timezone.utc)
                
                await asyncio.sleep(30)  # Update every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Statistics update error: {str(e)}")
                await asyncio.sleep(30)

    async def _maintenance_loop(self) -> None:
        """Background maintenance tasks."""
        while not self.shutdown_event.is_set():
            try:
                # Clean up old performance history
                if len(self.performance_history) > 500:
                    # Keep only recent entries
                    recent_entries = list(self.performance_history)[-500:]
                    self.performance_history.clear()
                    self.performance_history.extend(recent_entries)
                
                # Clean up old key locks
                current_time = datetime.now(timezone.utc)
                old_locks = []
                
                for key, lock in self.key_locks.items():
                    if not lock.locked() and key not in self.active_keys:
                        old_locks.append(key)
                
                for key in old_locks:
                    self.key_locks.pop(key, None)
                
                if old_locks:
                    self.logger.debug(f"Cleaned up {len(old_locks)} old key locks")
                
                await asyncio.sleep(300)  # Run every 5 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Maintenance error: {str(e)}")
                await asyncio.sleep(300)

    async def _key_expiration_monitor(self) -> None:
        """Monitor and clean up expired keys."""
        while not self.shutdown_event.is_set():
            try:
                # Check for expired keys in batches
                keys_to_check = list(self.active_keys)
                
                for i in range(0, len(keys_to_check), self.config.batch_size):
                    batch_keys = keys_to_check[i:i + self.config.batch_size]
                    
                    pipe = self.redis_client.pipeline()
                    for key in batch_keys:
                        pipe.exists(key)
                    
                    results = await pipe.execute()
                    
                    # Remove keys that no longer exist
                    for key, exists in zip(batch_keys, results):
                        if not exists:
                            self.active_keys.discard(key)
                            self.key_locks.pop(key, None)
                
                await asyncio.sleep(60)  # Check every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Key expiration monitor error: {str(e)}")
                await asyncio.sleep(60)

    async def _health_check_callback(self) -> Dict[str, Any]:
        """Health check callback for the cache system."""
        try:
            stats = await self.get_statistics()
            
            return {
                "status": "healthy" if self.is_healthy else "degraded",
                "connected": self.is_connected,
                "hit_ratio": stats.get('hit_ratio', 0),
                "total_operations": stats.get('total_operations', 0),
                "active_keys": stats.get('active_keys', 0),
                "memory_usage_bytes": stats.get('memory_usage_bytes', 0),
                "average_response_time": stats.get('average_response_time', 0)
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "connected": False
