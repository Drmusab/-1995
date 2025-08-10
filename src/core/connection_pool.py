"""
Connection pooling and resource management for improved performance
Author: Drmusab
Last Modified: 2025-08-10

This module provides connection pooling and resource management to reduce
the overhead of creating and destroying connections for external services.
"""

import asyncio
import logging
import time
import weakref
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union, AsyncContextManager
from urllib.parse import urlparse

import aiohttp
from src.core.lazy_imports import lazy_import
from src.observability.logging.config import get_logger


# Lazy imports for optional dependencies
redis = lazy_import('redis')
sqlalchemy = lazy_import('sqlalchemy')


@dataclass
class ConnectionConfig:
    """Configuration for connection pooling."""
    
    # HTTP connection pooling
    http_max_connections: int = 100
    http_max_connections_per_host: int = 30
    http_connection_timeout: float = 10.0
    http_request_timeout: float = 30.0
    
    # Redis connection pooling
    redis_max_connections: int = 20
    redis_connection_timeout: float = 5.0
    redis_socket_keepalive: bool = True
    redis_retry_on_timeout: bool = True
    
    # Database connection pooling
    db_pool_size: int = 5
    db_max_overflow: int = 10
    db_pool_timeout: float = 30.0
    db_pool_recycle: int = 3600
    
    # General settings
    enable_connection_reuse: bool = True
    connection_idle_timeout: float = 60.0
    enable_health_checks: bool = True
    health_check_interval: float = 30.0


class ConnectionPool:
    """Base class for connection pooling."""
    
    def __init__(self, name: str, config: ConnectionConfig):
        self.name = name
        self.config = config
        self.logger = get_logger(f"{__name__}.{name}")
        self._active_connections = 0
        self._total_connections_created = 0
        self._connection_errors = 0
        self._last_health_check = 0
        
    async def get_connection(self):
        """Get a connection from the pool."""
        raise NotImplementedError
        
    async def release_connection(self, connection):
        """Release a connection back to the pool."""
        raise NotImplementedError
        
    async def close_all(self):
        """Close all connections in the pool."""
        raise NotImplementedError
        
    def get_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics."""
        return {
            "name": self.name,
            "active_connections": self._active_connections,
            "total_created": self._total_connections_created,
            "connection_errors": self._connection_errors,
            "last_health_check": self._last_health_check,
        }


class HTTPConnectionPool(ConnectionPool):
    """HTTP connection pool using aiohttp."""
    
    def __init__(self, config: ConnectionConfig):
        super().__init__("http", config)
        self._session: Optional[aiohttp.ClientSession] = None
        self._connector: Optional[aiohttp.TCPConnector] = None
        
    async def _ensure_session(self):
        """Ensure HTTP session is created."""
        if self._session is None or self._session.closed:
            self._connector = aiohttp.TCPConnector(
                limit=self.config.http_max_connections,
                limit_per_host=self.config.http_max_connections_per_host,
                enable_cleanup_closed=True,
                keepalive_timeout=self.config.connection_idle_timeout,
                ttl_dns_cache=300,  # 5 minutes DNS cache
            )
            
            timeout = aiohttp.ClientTimeout(
                total=self.config.http_request_timeout,
                connect=self.config.http_connection_timeout,
            )
            
            self._session = aiohttp.ClientSession(
                connector=self._connector,
                timeout=timeout,
                headers={"User-Agent": "AI-Assistant/1.0"},
            )
            
            self.logger.info("Created new HTTP session with connection pooling")
    
    @asynccontextmanager
    async def get_connection(self):
        """Get an HTTP session for making requests."""
        await self._ensure_session()
        self._active_connections += 1
        
        try:
            yield self._session
        except Exception as e:
            self._connection_errors += 1
            self.logger.error(f"HTTP connection error: {str(e)}")
            raise
        finally:
            self._active_connections -= 1
    
    async def request(self, method: str, url: str, **kwargs) -> aiohttp.ClientResponse:
        """Make an HTTP request with connection pooling."""
        async with self.get_connection() as session:
            try:
                response = await session.request(method, url, **kwargs)
                return response
            except Exception as e:
                self.logger.error(f"HTTP request failed: {method} {url} - {str(e)}")
                raise
    
    async def get(self, url: str, **kwargs) -> aiohttp.ClientResponse:
        """Make a GET request."""
        return await self.request("GET", url, **kwargs)
    
    async def post(self, url: str, **kwargs) -> aiohttp.ClientResponse:
        """Make a POST request."""
        return await self.request("POST", url, **kwargs)
    
    async def put(self, url: str, **kwargs) -> aiohttp.ClientResponse:
        """Make a PUT request."""
        return await self.request("PUT", url, **kwargs)
    
    async def delete(self, url: str, **kwargs) -> aiohttp.ClientResponse:
        """Make a DELETE request."""
        return await self.request("DELETE", url, **kwargs)
    
    async def close_all(self):
        """Close all HTTP connections."""
        if self._session and not self._session.closed:
            await self._session.close()
            self.logger.info("Closed HTTP session")
        
        if self._connector:
            await self._connector.close()


class RedisConnectionPool(ConnectionPool):
    """Redis connection pool."""
    
    def __init__(self, config: ConnectionConfig, redis_url: str = "redis://localhost:6379"):
        super().__init__("redis", config)
        self.redis_url = redis_url
        self._pool: Optional[Any] = None  # redis.ConnectionPool
        
    async def _ensure_pool(self):
        """Ensure Redis connection pool is created."""
        if self._pool is None:
            try:
                # Use lazy imported redis
                redis_module = redis._resolve()
                
                parsed_url = urlparse(self.redis_url)
                
                self._pool = redis_module.ConnectionPool(
                    host=parsed_url.hostname or "localhost",
                    port=parsed_url.port or 6379,
                    db=int(parsed_url.path[1:]) if parsed_url.path and len(parsed_url.path) > 1 else 0,
                    password=parsed_url.password,
                    max_connections=self.config.redis_max_connections,
                    socket_connect_timeout=self.config.redis_connection_timeout,
                    socket_keepalive=self.config.redis_socket_keepalive,
                    retry_on_timeout=self.config.redis_retry_on_timeout,
                )
                
                self.logger.info(f"Created Redis connection pool: {self.redis_url}")
                
            except Exception as e:
                self.logger.error(f"Failed to create Redis pool: {str(e)}")
                raise
    
    @asynccontextmanager
    async def get_connection(self):
        """Get a Redis connection from the pool."""
        await self._ensure_pool()
        
        try:
            # Use lazy imported redis
            redis_module = redis._resolve()
            connection = redis_module.Redis(connection_pool=self._pool)
            
            self._active_connections += 1
            self._total_connections_created += 1
            
            yield connection
            
        except Exception as e:
            self._connection_errors += 1
            self.logger.error(f"Redis connection error: {str(e)}")
            raise
        finally:
            self._active_connections -= 1
    
    async def close_all(self):
        """Close all Redis connections."""
        if self._pool:
            self._pool.disconnect()
            self.logger.info("Closed Redis connection pool")


class DatabaseConnectionPool(ConnectionPool):
    """Database connection pool using SQLAlchemy."""
    
    def __init__(self, config: ConnectionConfig, database_url: str):
        super().__init__("database", config)
        self.database_url = database_url
        self._engine: Optional[Any] = None  # sqlalchemy.Engine
        
    async def _ensure_engine(self):
        """Ensure database engine is created."""
        if self._engine is None:
            try:
                # Use lazy imported sqlalchemy
                sqlalchemy_module = sqlalchemy._resolve()
                
                self._engine = sqlalchemy_module.create_engine(
                    self.database_url,
                    pool_size=self.config.db_pool_size,
                    max_overflow=self.config.db_max_overflow,
                    pool_timeout=self.config.db_pool_timeout,
                    pool_recycle=self.config.db_pool_recycle,
                    pool_pre_ping=True,  # Verify connections before use
                    echo=False,  # Set to True for SQL debugging
                )
                
                self.logger.info("Created database connection pool")
                
            except Exception as e:
                self.logger.error(f"Failed to create database engine: {str(e)}")
                raise
    
    @asynccontextmanager
    async def get_connection(self):
        """Get a database connection from the pool."""
        await self._ensure_engine()
        
        try:
            connection = self._engine.connect()
            self._active_connections += 1
            self._total_connections_created += 1
            
            yield connection
            
        except Exception as e:
            self._connection_errors += 1
            self.logger.error(f"Database connection error: {str(e)}")
            raise
        finally:
            if 'connection' in locals():
                connection.close()
            self._active_connections -= 1
    
    async def close_all(self):
        """Close all database connections."""
        if self._engine:
            self._engine.dispose()
            self.logger.info("Closed database connection pool")


class ConnectionManager:
    """
    Central manager for all connection pools.
    
    Provides unified access to HTTP, Redis, and database connections
    with built-in pooling, monitoring, and health checks.
    """
    
    def __init__(self, config: ConnectionConfig = None):
        self.config = config or ConnectionConfig()
        self.logger = get_logger(__name__)
        
        # Connection pools
        self._http_pool: Optional[HTTPConnectionPool] = None
        self._redis_pools: Dict[str, RedisConnectionPool] = {}
        self._db_pools: Dict[str, DatabaseConnectionPool] = {}
        
        # Health check task
        self._health_check_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
        
    async def initialize(self):
        """Initialize the connection manager."""
        self.logger.info("Initializing connection manager")
        
        # Create HTTP pool
        self._http_pool = HTTPConnectionPool(self.config)
        
        # Start health checks if enabled
        if self.config.enable_health_checks:
            self._health_check_task = asyncio.create_task(self._health_check_loop())
        
        self.logger.info("Connection manager initialized")
    
    async def shutdown(self):
        """Shutdown all connection pools."""
        self.logger.info("Shutting down connection manager")
        
        # Signal shutdown
        self._shutdown_event.set()
        
        # Cancel health check task
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        # Close all pools
        if self._http_pool:
            await self._http_pool.close_all()
        
        for pool in self._redis_pools.values():
            await pool.close_all()
        
        for pool in self._db_pools.values():
            await pool.close_all()
        
        self.logger.info("Connection manager shutdown complete")
    
    def get_http_pool(self) -> HTTPConnectionPool:
        """Get the HTTP connection pool."""
        if self._http_pool is None:
            self._http_pool = HTTPConnectionPool(self.config)
        return self._http_pool
    
    def get_redis_pool(self, name: str = "default", redis_url: str = "redis://localhost:6379") -> RedisConnectionPool:
        """Get a Redis connection pool."""
        if name not in self._redis_pools:
            self._redis_pools[name] = RedisConnectionPool(self.config, redis_url)
        return self._redis_pools[name]
    
    def get_database_pool(self, name: str, database_url: str) -> DatabaseConnectionPool:
        """Get a database connection pool."""
        if name not in self._db_pools:
            self._db_pools[name] = DatabaseConnectionPool(self.config, database_url)
        return self._db_pools[name]
    
    # Convenience methods for HTTP requests
    async def http_get(self, url: str, **kwargs) -> aiohttp.ClientResponse:
        """Make an HTTP GET request with connection pooling."""
        return await self.get_http_pool().get(url, **kwargs)
    
    async def http_post(self, url: str, **kwargs) -> aiohttp.ClientResponse:
        """Make an HTTP POST request with connection pooling."""
        return await self.get_http_pool().post(url, **kwargs)
    
    async def http_put(self, url: str, **kwargs) -> aiohttp.ClientResponse:
        """Make an HTTP PUT request with connection pooling."""
        return await self.get_http_pool().put(url, **kwargs)
    
    async def http_delete(self, url: str, **kwargs) -> aiohttp.ClientResponse:
        """Make an HTTP DELETE request with connection pooling."""
        return await self.get_http_pool().delete(url, **kwargs)
    
    # Convenience methods for Redis
    @asynccontextmanager
    async def redis_connection(self, name: str = "default", redis_url: str = "redis://localhost:6379"):
        """Get a Redis connection with automatic cleanup."""
        pool = self.get_redis_pool(name, redis_url)
        async with pool.get_connection() as conn:
            yield conn
    
    # Convenience methods for database
    @asynccontextmanager
    async def database_connection(self, name: str, database_url: str):
        """Get a database connection with automatic cleanup."""
        pool = self.get_database_pool(name, database_url)
        async with pool.get_connection() as conn:
            yield conn
    
    async def _health_check_loop(self):
        """Background task for health checking connections."""
        while not self._shutdown_event.is_set():
            try:
                await self._perform_health_checks()
                await asyncio.sleep(self.config.health_check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in health check loop: {str(e)}")
                await asyncio.sleep(self.config.health_check_interval)
    
    async def _perform_health_checks(self):
        """Perform health checks on all connection pools."""
        current_time = time.time()
        
        # Check HTTP pool
        if self._http_pool:
            self._http_pool._last_health_check = current_time
        
        # Check Redis pools
        for pool in self._redis_pools.values():
            try:
                async with pool.get_connection() as redis_conn:
                    await redis_conn.ping()
                pool._last_health_check = current_time
            except Exception as e:
                self.logger.warning(f"Redis health check failed for {pool.name}: {str(e)}")
        
        # Check database pools
        for pool in self._db_pools.values():
            try:
                async with pool.get_connection() as db_conn:
                    # Simple health check query
                    db_conn.execute("SELECT 1")
                pool._last_health_check = current_time
            except Exception as e:
                self.logger.warning(f"Database health check failed for {pool.name}: {str(e)}")
    
    def get_all_stats(self) -> Dict[str, Any]:
        """Get statistics for all connection pools."""
        stats = {
            "http": self._http_pool.get_stats() if self._http_pool else None,
            "redis": {name: pool.get_stats() for name, pool in self._redis_pools.items()},
            "database": {name: pool.get_stats() for name, pool in self._db_pools.items()},
        }
        return stats


# Global connection manager instance
_connection_manager: Optional[ConnectionManager] = None


async def get_connection_manager() -> ConnectionManager:
    """Get the global connection manager instance."""
    global _connection_manager
    
    if _connection_manager is None:
        _connection_manager = ConnectionManager()
        await _connection_manager.initialize()
    
    return _connection_manager


async def shutdown_connection_manager():
    """Shutdown the global connection manager."""
    global _connection_manager
    
    if _connection_manager:
        await _connection_manager.shutdown()
        _connection_manager = None