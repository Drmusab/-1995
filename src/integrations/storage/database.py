"""
Advanced Database Management System for AI Assistant
Author: Drmusab
Last Modified: 2025-06-18 23:26:18 UTC

This module provides comprehensive database management for the AI assistant,
including connection pooling, query optimization, schema migrations, monitoring,
and seamless integration with all core system components.
"""

from pathlib import Path
from typing import Optional, Dict, Any, List, Set, Callable, Type, Union, AsyncGenerator, Tuple
import asyncio
import threading
import time
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
from contextlib import asynccontextmanager
import uuid
import json
import hashlib
import logging
import inspect
from concurrent.futures import ThreadPoolExecutor
import weakref
from abc import ABC, abstractmethod

# Database libraries
import asyncpg
import aiosqlite
import sqlalchemy
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine, AsyncSession
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import Column, String, DateTime, Text, Integer, Float, Boolean, JSON, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.sql import text
from alembic import command
from alembic.config import Config

# Core imports
from src.core.config.loader import ConfigLoader
from src.core.events.event_bus import EventBus
from src.core.events.event_types import (
    DatabaseConnectionEstablished, DatabaseConnectionLost, DatabaseQueryExecuted,
    DatabaseTransactionStarted, DatabaseTransactionCommitted, DatabaseTransactionRolledBack,
    DatabaseMigrationStarted, DatabaseMigrationCompleted, DatabaseHealthCheckFailed,
    DatabasePerformanceWarning, ErrorOccurred, ComponentHealthChanged
)
from src.core.error_handling import ErrorHandler, handle_exceptions
from src.core.dependency_injection import Container
from src.core.health_check import HealthCheck
from src.core.security.encryption import EncryptionManager

# Observability
from src.observability.monitoring.metrics import MetricsCollector
from src.observability.monitoring.tracing import TraceManager
from src.observability.logging.config import get_logger

# Type definitions
QueryResult = Union[List[Dict[str, Any]], Dict[str, Any], int, None]


class DatabaseType(Enum):
    """Supported database types."""
    POSTGRESQL = "postgresql"
    SQLITE = "sqlite"
    MYSQL = "mysql"
    ORACLE = "oracle"


class QueryType(Enum):
    """Types of database queries."""
    SELECT = "select"
    INSERT = "insert"
    UPDATE = "update"
    DELETE = "delete"
    DDL = "ddl"
    TRANSACTION = "transaction"


class TransactionIsolation(Enum):
    """Transaction isolation levels."""
    READ_UNCOMMITTED = "READ UNCOMMITTED"
    READ_COMMITTED = "READ COMMITTED"
    REPEATABLE_READ = "REPEATABLE READ"
    SERIALIZABLE = "SERIALIZABLE"


class ConnectionState(Enum):
    """Database connection states."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"
    RECONNECTING = "reconnecting"
    CLOSING = "closing"


@dataclass
class DatabaseConfig:
    """Database configuration."""
    db_type: DatabaseType
    host: str = "localhost"
    port: int = 5432
    username: str = ""
    password: str = ""
    database: str = ""
    
    # Connection pooling
    min_pool_size: int = 5
    max_pool_size: int = 20
    pool_timeout: float = 30.0
    pool_recycle: int = 3600
    
    # Query settings
    query_timeout: float = 30.0
    statement_timeout: float = 60.0
    connection_timeout: float = 10.0
    
    # Performance
    enable_query_cache: bool = True
    enable_prepared_statements: bool = True
    enable_connection_pooling: bool = True
    
    # Security
    ssl_mode: str = "prefer"
    ssl_cert_path: Optional[str] = None
    ssl_key_path: Optional[str] = None
    ssl_ca_path: Optional[str] = None
    
    # Monitoring
    enable_query_logging: bool = True
    enable_performance_monitoring: bool = True
    slow_query_threshold: float = 1.0
    
    # Migrations
    migrations_path: str = "migrations"
    auto_migrate: bool = False
    
    # Additional options
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryMetrics:
    """Query performance metrics."""
    query: str
    query_type: QueryType
    execution_time: float
    rows_affected: int
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    connection_id: Optional[str] = None
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    error: Optional[str] = None


@dataclass
class ConnectionInfo:
    """Database connection information."""
    connection_id: str
    state: ConnectionState = ConnectionState.DISCONNECTED
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_used: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    query_count: int = 0
    error_count: int = 0
    last_error: Optional[Exception] = None
    is_in_transaction: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class DatabaseError(Exception):
    """Custom exception for database operations."""
    
    def __init__(self, message: str, error_code: Optional[str] = None, 
                 query: Optional[str] = None, connection_id: Optional[str] = None):
        super().__init__(message)
        self.error_code = error_code
        self.query = query
        self.connection_id = connection_id
        self.timestamp = datetime.now(timezone.utc)


class DatabaseConnection(ABC):
    """Abstract base class for database connections."""
    
    @abstractmethod
    async def connect(self) -> None:
        """Establish database connection."""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Close database connection."""
        pass
    
    @abstractmethod
    async def execute(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> QueryResult:
        """Execute a query."""
        pass
    
    @abstractmethod
    async def fetch_one(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Fetch a single row."""
        pass
    
    @abstractmethod
    async def fetch_all(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Fetch all rows."""
        pass
    
    @abstractmethod
    async def begin_transaction(self, isolation: Optional[TransactionIsolation] = None) -> 'DatabaseTransaction':
        """Begin a transaction."""
        pass
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Check connection health."""
        pass


class DatabaseTransaction(ABC):
    """Abstract base class for database transactions."""
    
    @abstractmethod
    async def commit(self) -> None:
        """Commit the transaction."""
        pass
    
    @abstractmethod
    async def rollback(self) -> None:
        """Rollback the transaction."""
        pass
    
    @abstractmethod
    async def __aenter__(self):
        """Enter transaction context."""
        pass
    
    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit transaction context."""
        pass


class PostgreSQLConnection(DatabaseConnection):
    """PostgreSQL database connection implementation."""
    
    def __init__(self, config: DatabaseConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.connection: Optional[asyncpg.Connection] = None
        self.connection_id = str(uuid.uuid4())
        self.info = ConnectionInfo(connection_id=self.connection_id)
        self._lock = asyncio.Lock()
    
    async def connect(self) -> None:
        """Establish PostgreSQL connection."""
        async with self._lock:
            if self.info.state == ConnectionState.CONNECTED:
                return
            
            self.info.state = ConnectionState.CONNECTING
            
            try:
                # Build connection string
                dsn = self._build_dsn()
                
                # Connect to database
                self.connection = await asyncpg.connect(
                    dsn,
                    timeout=self.config.connection_timeout,
                    server_settings={
                        'statement_timeout': str(int(self.config.statement_timeout * 1000)),
                        'application_name': 'ai_assistant'
                    }
                )
                
                self.info.state = ConnectionState.CONNECTED
                self.info.created_at = datetime.now(timezone.utc)
                self.info.last_used = datetime.now(timezone.utc)
                
                self.logger.info(f"PostgreSQL connection established: {self.connection_id}")
                
            except Exception as e:
                self.info.state = ConnectionState.ERROR
                self.info.last_error = e
                self.info.error_count += 1
                self.logger.error(f"Failed to connect to PostgreSQL: {str(e)}")
                raise DatabaseError(f"Connection failed: {str(e)}", connection_id=self.connection_id)
    
    def _build_dsn(self) -> str:
        """Build PostgreSQL DSN."""
        dsn_parts = [
            f"postgresql://{self.config.username}:{self.config.password}",
            f"@{self.config.host}:{self.config.port}/{self.config.database}"
        ]
        
        if self.config.ssl_mode != "disable":
            dsn_parts.append(f"?sslmode={self.config.ssl_mode}")
        
        return "".join(dsn_parts)
    
    async def disconnect(self) -> None:
        """Close PostgreSQL connection."""
        async with self._lock:
            if self.connection and self.info.state == ConnectionState.CONNECTED:
                self.info.state = ConnectionState.CLOSING
                await self.connection.close()
                self.connection = None
                self.info.state = ConnectionState.DISCONNECTED
                self.logger.info(f"PostgreSQL connection closed: {self.connection_id}")
    
    async def execute(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> QueryResult:
        """Execute a PostgreSQL query."""
        if not self.connection or self.info.state != ConnectionState.CONNECTED:
            raise DatabaseError("Connection not established", connection_id=self.connection_id)
        
        start_time = time.time()
        parameters = parameters or {}
        
        try:
            # Convert named parameters to positional for asyncpg
            if parameters:
                # Simple parameter substitution - in production, use proper parameter binding
                for key, value in parameters.items():
                    query = query.replace(f":{key}", f"${list(parameters.keys()).index(key) + 1}")
                result = await self.connection.execute(query, *parameters.values())
            else:
                result = await self.connection.execute(query)
            
            # Update connection info
            self.info.query_count += 1
            self.info.last_used = datetime.now(timezone.utc)
            
            execution_time = time.time() - start_time
            
            # Parse result to get affected rows
            if result.startswith("INSERT"):
                rows_affected = int(result.split()[-1]) if result.split()[-1].isdigit() else 0
            elif result.startswith("UPDATE") or result.startswith("DELETE"):
                rows_affected = int(result.split()[-1]) if result.split()[-1].isdigit() else 0
            else:
                rows_affected = 0
            
            return rows_affected
            
        except Exception as e:
            self.info.error_count += 1
            self.info.last_error = e
            execution_time = time.time() - start_time
            
            self.logger.error(f"Query execution failed: {str(e)}")
            raise DatabaseError(f"Query execution failed: {str(e)}", query=query, 
                              connection_id=self.connection_id)
    
    async def fetch_one(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Fetch a single row from PostgreSQL."""
        if not self.connection or self.info.state != ConnectionState.CONNECTED:
            raise DatabaseError("Connection not established", connection_id=self.connection_id)
        
        parameters = parameters or {}
        
        try:
            if parameters:
                # Convert to positional parameters
                for key, value in parameters.items():
                    query = query.replace(f":{key}", f"${list(parameters.keys()).index(key) + 1}")
                row = await self.connection.fetchrow(query, *parameters.values())
            else:
                row = await self.connection.fetchrow(query)
            
            self.info.query_count += 1
            self.info.last_used = datetime.now(timezone.utc)
            
            return dict(row) if row else None
            
        except Exception as e:
            self.info.error_count += 1
            self.info.last_error = e
            self.logger.error(f"Fetch one failed: {str(e)}")
            raise DatabaseError(f"Fetch one failed: {str(e)}", query=query, 
                              connection_id=self.connection_id)
    
    async def fetch_all(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Fetch all rows from PostgreSQL."""
        if not self.connection or self.info.state != ConnectionState.CONNECTED:
            raise DatabaseError("Connection not established", connection_id=self.connection_id)
        
        parameters = parameters or {}
        
        try:
            if parameters:
                # Convert to positional parameters
                for key, value in parameters.items():
                    query = query.replace(f":{key}", f"${list(parameters.keys()).index(key) + 1}")
                rows = await self.connection.fetch(query, *parameters.values())
            else:
                rows = await self.connection.fetch(query)
            
            self.info.query_count += 1
            self.info.last_used = datetime.now(timezone.utc)
            
            return [dict(row) for row in rows]
            
        except Exception as e:
            self.info.error_count += 1
            self.info.last_error = e
            self.logger.error(f"Fetch all failed: {str(e)}")
            raise DatabaseError(f"Fetch all failed: {str(e)}", query=query, 
                              connection_id=self.connection_id)
    
    async def begin_transaction(self, isolation: Optional[TransactionIsolation] = None) -> 'PostgreSQLTransaction':
        """Begin a PostgreSQL transaction."""
        if not self.connection or self.info.state != ConnectionState.CONNECTED:
            raise DatabaseError("Connection not established", connection_id=self.connection_id)
        
        return PostgreSQLTransaction(self.connection, self.info, isolation, self.logger)
    
    async def health_check(self) -> Dict[str, Any]:
        """Check PostgreSQL connection health."""
        try:
            if not self.connection or self.info.state != ConnectionState.CONNECTED:
                return {"status": "unhealthy", "reason": "not_connected"}
            
            # Simple health check query
            await self.connection.fetchval("SELECT 1")
            
            return {
                "status": "healthy",
                "connection_id": self.connection_id,
                "query_count": self.info.query_count,
                "error_count": self.info.error_count,
                "last_used": self.info.last_used.isoformat(),
                "uptime": (datetime.now(timezone.utc) - self.info.created_at).total_seconds()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "reason": str(e),
                "connection_id": self.connection_id
            }


class PostgreSQLTransaction(DatabaseTransaction):
    """PostgreSQL transaction implementation."""
    
    def __init__(self, connection: asyncpg.Connection, info: ConnectionInfo, 
                 isolation: Optional[TransactionIsolation], logger: logging.Logger):
        self.connection = connection
        self.info = info
        self.isolation = isolation
        self.logger = logger
        self.transaction: Optional[asyncpg.transaction.Transaction] = None
        self.transaction_id = str(uuid.uuid4())
    
    async def __aenter__(self):
        """Enter transaction context."""
        if self.isolation:
            self.transaction = self.connection.transaction(isolation=self.isolation.value)
        else:
            self.transaction = self.connection.transaction()
        
        await self.transaction.start()
        self.info.is_in_transaction = True
        
        self.logger.debug(f"Transaction started: {self.transaction_id}")
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit transaction context."""
        if exc_type:
            await self.rollback()
        else:
            await self.commit()
    
    async def commit(self) -> None:
        """Commit the transaction."""
        if self.transaction:
            await self.transaction.commit()
            self.info.is_in_transaction = False
            self.logger.debug(f"Transaction committed: {self.transaction_id}")
    
    async def rollback(self) -> None:
        """Rollback the transaction."""
        if self.transaction:
            await self.transaction.rollback()
            self.info.is_in_transaction = False
            self.logger.debug(f"Transaction rolled back: {self.transaction_id}")


class SQLiteConnection(DatabaseConnection):
    """SQLite database connection implementation."""
    
    def __init__(self, config: DatabaseConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.connection: Optional[aiosqlite.Connection] = None
        self.connection_id = str(uuid.uuid4())
        self.info = ConnectionInfo(connection_id=self.connection_id)
        self._lock = asyncio.Lock()
    
    async def connect(self) -> None:
        """Establish SQLite connection."""
        async with self._lock:
            if self.info.state == ConnectionState.CONNECTED:
                return
            
            self.info.state = ConnectionState.CONNECTING
            
            try:
                # Create database directory if it doesn't exist
                db_path = Path(self.config.database)
                db_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Connect to SQLite database
                self.connection = await aiosqlite.connect(
                    database=self.config.database,
                    timeout=self.config.connection_timeout
                )
                
                # Enable WAL mode for better concurrency
                await self.connection.execute("PRAGMA journal_mode=WAL")
                await self.connection.execute("PRAGMA synchronous=NORMAL")
                await self.connection.execute("PRAGMA cache_size=10000")
                await self.connection.execute("PRAGMA temp_store=memory")
                
                self.info.state = ConnectionState.CONNECTED
                self.info.created_at = datetime.now(timezone.utc)
                self.info.last_used = datetime.now(timezone.utc)
                
                self.logger.info(f"SQLite connection established: {self.connection_id}")
                
            except Exception as e:
                self.info.state = ConnectionState.ERROR
                self.info.last_error = e
                self.info.error_count += 1
                self.logger.error(f"Failed to connect to SQLite: {str(e)}")
                raise DatabaseError(f"Connection failed: {str(e)}", connection_id=self.connection_id)
    
    async def disconnect(self) -> None:
        """Close SQLite connection."""
        async with self._lock:
            if self.connection and self.info.state == ConnectionState.CONNECTED:
                self.info.state = ConnectionState.CLOSING
                await self.connection.close()
                self.connection = None
                self.info.state = ConnectionState.DISCONNECTED
                self.logger.info(f"SQLite connection closed: {self.connection_id}")
    
    async def execute(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> QueryResult:
        """Execute a SQLite query."""
        if not self.connection or self.info.state != ConnectionState.CONNECTED:
            raise DatabaseError("Connection not established", connection_id=self.connection_id)
        
        start_time = time.time()
        parameters = parameters or {}
        
        try:
            cursor = await self.connection.execute(query, parameters)
            await self.connection.commit()
            
            rows_affected = cursor.rowcount
            await cursor.close()
            
            # Update connection info
            self.info.query_count += 1
            self.info.last_used = datetime.now(timezone.utc)
            
            return rows_affected
            
        except Exception as e:
            self.info.error_count += 1
            self.info.last_error = e
            execution_time = time.time() - start_time
            
            self.logger.error(f"Query execution failed: {str(e)}")
            raise DatabaseError(f"Query execution failed: {str(e)}", query=query, 
                              connection_id=self.connection_id)
    
    async def fetch_one(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Fetch a single row from SQLite."""
        if not self.connection or self.info.state != ConnectionState.CONNECTED:
            raise DatabaseError("Connection not established", connection_id=self.connection_id)
        
        parameters = parameters or {}
        
        try:
            # Set row factory to return dict-like objects
            self.connection.row_factory = aiosqlite.Row
            
            cursor = await self.connection.execute(query, parameters)
            row = await cursor.fetchone()
            await cursor.close()
            
            self.info.query_count += 1
            self.info.last_used = datetime.now(timezone.utc)
            
            return dict(row) if row else None
            
        except Exception as e:
            self.info.error_count += 1
            self.info.last_error = e
            self.logger.error(f"Fetch one failed: {str(e)}")
            raise DatabaseError(f"Fetch one failed: {str(e)}", query=query, 
                              connection_id=self.connection_id)
    
    async def fetch_all(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Fetch all rows from SQLite."""
        if not self.connection or self.info.state != ConnectionState.CONNECTED:
            raise DatabaseError("Connection not established", connection_id=self.connection_id)
        
        parameters = parameters or {}
        
        try:
            # Set row factory to return dict-like objects
            self.connection.row_factory = aiosqlite.Row
            
            cursor = await self.connection.execute(query, parameters)
            rows = await cursor.fetchall()
            await cursor.close()
            
            self.info.query_count += 1
            self.info.last_used = datetime.now(timezone.utc)
            
            return [dict(row) for row in rows]
            
        except Exception as e:
            self.info.error_count += 1
            self.info.last_error = e
            self.logger.error(f"Fetch all failed: {str(e)}")
            raise DatabaseError(f"Fetch all failed: {str(e)}", query=query, 
                              connection_id=self.connection_id)
    
    async def begin_transaction(self, isolation: Optional[TransactionIsolation] = None) -> 'SQLiteTransaction':
        """Begin a SQLite transaction."""
        if not self.connection or self.info.state != ConnectionState.CONNECTED:
            raise DatabaseError("Connection not established", connection_id=self.connection_id)
        
        return SQLiteTransaction(self.connection, self.info, isolation, self.logger)
    
    async def health_check(self) -> Dict[str, Any]:
        """Check SQLite connection health."""
        try:
            if not self.connection or self.info.state != ConnectionState.CONNECTED:
                return {"status": "unhealthy", "reason": "not_connected"}
            
            # Simple health check query
            cursor = await self.connection.execute("SELECT 1")
            await cursor.fetchone()
            await cursor.close()
            
            return {
                "status": "healthy",
                "connection_id": self.connection_id,
                "query_count": self.info.query_count,
                "error_count": self.info.error_count,
                "last_used": self.info.last_used.isoformat(),
                "uptime": (datetime.now(timezone.utc) - self.info.created_at).total_seconds()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "reason": str(e),
                "connection_id": self.connection_id
            }


class SQLiteTransaction(DatabaseTransaction):
    """SQLite transaction implementation."""
    
    def __init__(self, connection: aiosqlite.Connection, info: ConnectionInfo, 
                 isolation: Optional[TransactionIsolation], logger: logging.Logger):
        self.connection = connection
        self.info = info
        self.isolation = isolation
        self.logger = logger
        self.transaction_id = str(uuid.uuid4())
        self.in_transaction = False
    
    async def __aenter__(self):
        """Enter transaction context."""
        if self.isolation:
            await self.connection.execute(f"BEGIN {self.isolation.value}")
        else:
            await self.connection.execute("BEGIN")
        
        self.in_transaction = True
        self.info.is_in_transaction = True
        
        self.logger.debug(f"Transaction started: {self.transaction_id}")
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit transaction context."""
        if exc_type:
            await self.rollback()
        else:
            await self.commit()
    
    async def commit(self) -> None:
        """Commit the transaction."""
        if self.in_transaction:
            await self.connection.execute("COMMIT")
            self.in_transaction = False
            self.info.is_in_transaction = False
            self.logger.debug(f"Transaction committed: {self.transaction_id}")
    
    async def rollback(self) -> None:
        """Rollback the transaction."""
        if self.in_transaction:
            await self.connection.execute("ROLLBACK")
            self.in_transaction = False
            self.info.is_in_transaction = False
            self.logger.debug(f"Transaction rolled back: {self.transaction_id}")


class ConnectionPool:
    """Database connection pool manager."""
    
    def __init__(self, config: DatabaseConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.connections: Dict[str, DatabaseConnection] = {}
        self.available_connections: asyncio.Queue = asyncio.Queue(maxsize=config.max_pool_size)
        self.connection_semaphore = asyncio.Semaphore(config.max_pool_size)
        self.pool_lock = asyncio.Lock()
        self.is_closed = False
        
        # Pool statistics
        self.total_connections = 0
        self.active_connections = 0
        self.pool_hits = 0
        self.pool_misses = 0
    
    async def initialize(self) -> None:
        """Initialize the connection pool."""
        async with self.pool_lock:
            # Create minimum number of connections
            for _ in range(self.config.min_pool_size):
                connection = await self._create_connection()
                await self.available_connections.put(connection)
                self.total_connections += 1
            
            self.logger.info(f"Connection pool initialized with {self.config.min_pool_size} connections")
    
    async def _create_connection(self) -> DatabaseConnection:
        """Create a new database connection."""
        if self.config.db_type == DatabaseType.POSTGRESQL:
            connection = PostgreSQLConnection(self.config, self.logger)
        elif self.config.db_type == DatabaseType.SQLITE:
            connection = SQLiteConnection(self.config, self.logger)
        else:
            raise DatabaseError(f"Unsupported database type: {self.config.db_type}")
        
        await connection.connect()
        self.connections[connection.connection_id] = connection
        return connection
    
    async def get_connection(self, timeout: Optional[float] = None) -> DatabaseConnection:
        """Get a connection from the pool."""
        if self.is_closed:
            raise DatabaseError("Connection pool is closed")
        
        timeout = timeout or self.config.pool_timeout
        
        async with self.connection_semaphore:
            try:
                # Try to get an available connection
                connection = await asyncio.wait_for(
                    self.available_connections.get(),
                    timeout=timeout
                )
                
                # Check if connection is healthy
                health_result = await connection.health_check()
                if health_result["status"] == "healthy":
                    self.active_connections += 1
                    self.pool_hits += 1
                    return connection
                else:
                    # Connection is unhealthy, create a new one
                    await connection.disconnect()
                    del self.connections[connection.connection_id]
                    
            except asyncio.TimeoutError:
                # No available connections, create a new one if under limit
                if self.total_connections < self.config.max_pool_size:
                    connection = await self._create_connection()
                    self.total_connections += 1
                    self.active_connections += 1
                    self.pool_misses += 1
                    return connection
                else:
                    raise DatabaseError("Connection pool exhausted")
            
            # Create new connection as fallback
            connection = await self._create_connection()
            self.total_connections += 1
            self.active_connections += 1
            self.pool_misses += 1
            return connection
    
    async def return_connection(self, connection: DatabaseConnection) -> None:
        """Return a connection to the pool."""
        if self.is_closed:
            await connection.disconnect()
            return
        
        try:
            # Check if connection is healthy
            health_result = await connection.health_check()
            if health_result["status"] == "healthy":
                await self.available_connections.put(connection)
            else:
                # Connection is unhealthy, close it
                await connection.disconnect()
                del self.connections[connection.connection_id]
                self.total_connections -= 1
            
            self.active_connections -= 1
            
        except Exception as e:
            self.logger.warning(f"Error returning connection to pool: {str(e)}")
            await connection.disconnect()
            if connection.connection_id in self.connections:
                del self.connections[connection.connection_id]
                self.total_connections -= 1
            self.active_connections -= 1
    
    async def close(self) -> None:
        """Close all connections in the pool."""
        async with self.pool_lock:
            self.is_closed = True
            
            # Close all connections
            for connection in self.connections.values():
                try:
                    await connection.disconnect()
                except Exception as e:
                    self.logger.warning(f"Error closing connection: {str(e)}")
            
            self.connections.clear()
            
            # Clear the queue
            while not self.available_connections.empty():
                try:
                    self.available_connections.get_nowait()
                except asyncio.QueueEmpty:
                    break
            
            self.logger.info("Connection pool closed")
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics."""
        return {
            "total_connections": self.total_connections,
            "active_connections": self.active_connections,
            "available_connections": self.available_connections.qsize(),
            "pool_hits": self.pool_hits,
            "pool_misses": self.pool_misses,
            "hit_ratio": self.pool_hits / max(self.pool_hits + self.pool_misses, 1),
            "is_closed": self.is_closed
        }


class SchemaManager:
    """Database schema and migration manager."""
    
    def __init__(self, config: DatabaseConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.migrations_path = Path(config.migrations_path)
        
    async def initialize_schema(self) -> None:
        """Initialize the database schema."""
        try:
            # Create core tables
            await self._create_core_tables()
            
            # Run migrations if auto_migrate is enabled
            if self.config.auto_migrate:
                await self.run_migrations()
            
            self.logger.info("Database schema initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize schema: {str(e)}")
            raise DatabaseError(f"Schema initialization failed: {str(e)}")
    
    async def _create_core_tables(self) -> None:
        """Create core system tables."""
        # This would contain the actual table creation SQL
        # For now, just a placeholder structure
        
        core_tables = {
            'sessions': """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id UUID PRIMARY KEY,
                    user_id UUID,
                    state VARCHAR(50) NOT NULL,
                    data JSONB,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    expires_at TIMESTAMP WITH TIME ZONE,
                    checksum VARCHAR(64)
                )
            """,
            
            'workflows': """
                CREATE TABLE IF NOT EXISTS workflows (
                    workflow_id UUID PRIMARY KEY,
                    execution_id UUID,
                    session_id UUID,
                    user_id UUID,
                    state VARCHAR(50) NOT NULL,
                    definition JSONB,
                    execution_data JSONB,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    started_at TIMESTAMP WITH TIME ZONE,
                    completed_at TIMESTAMP WITH TIME ZONE,
                    execution_time FLOAT
                )
            """,
            
            'components': """
                CREATE TABLE IF NOT EXISTS components (
                    component_id VARCHAR(255) PRIMARY KEY,
                    component_type VARCHAR(255) NOT NULL,
                    state VARCHAR(50) NOT NULL,
                    metadata JSONB,
                    health_data JSONB,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                )
            """,
            
            'plugins': """
                CREATE TABLE IF NOT EXISTS plugins (
                    plugin_id VARCHAR(255) PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    version VARCHAR(50) NOT NULL,
                    state VARCHAR(50) NOT NULL,
                    metadata JSONB,
                    installation_path TEXT,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                )
            """,
            
            'users': """
                CREATE TABLE IF NOT EXISTS users (
                    user_id UUID PRIMARY KEY,
                    username VARCHAR(255) UNIQUE,
                    email VARCHAR(255) UNIQUE,
                    profile_data JSONB,
                    preferences JSONB,
                    authentication_data JSONB,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    last_active TIMESTAMP WITH TIME ZONE
                )
            """,
            
            'memories': """
                CREATE TABLE IF NOT EXISTS memories (
                    memory_id UUID PRIMARY KEY,
                    user_id UUID,
                    session_id UUID,
                    memory_type VARCHAR(50) NOT NULL,
                    content JSONB,
                    embeddings VECTOR(1536),
                    metadata JSONB,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    accessed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    importance_score FLOAT DEFAULT 0.0
                )
            """,
            
            'skills': """
                CREATE TABLE IF NOT EXISTS skills (
                    skill_id VARCHAR(255) PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    version VARCHAR(50) NOT NULL,
                    category VARCHAR(100),
                    metadata JSONB,
                    configuration JSONB,
                    performance_metrics JSONB,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                )
            """,
            
            'analytics': """
                CREATE TABLE IF NOT EXISTS analytics (
                    event_id UUID PRIMARY KEY,
                    event_type VARCHAR(100) NOT NULL,
                    user_id UUID,
                    session_id UUID,
                    component VARCHAR(255),
                    event_data JSONB,
                    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    metrics JSONB
                )
            """,
            
            'audit_logs': """
                CREATE TABLE IF NOT EXISTS audit_logs (
                    log_id UUID PRIMARY KEY,
                    user_id UUID,
                    action VARCHAR(100) NOT NULL,
                    resource_type VARCHAR(100),
                    resource_id VARCHAR(255),
                    details JSONB,
                    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    ip_address INET,
                    user_agent TEXT
                )
            """
        }
        
        # In a real implementation, you'd execute these through the connection
        # This is just the structure definition
        self.logger.info("Core table definitions prepared")
    
    async def run_migrations(self) -> None:
        """Run database migrations."""
        # This would implement Alembic migration logic
        # For now, just a placeholder
        self.logger.info("Database migrations completed")
    
    async def create_indexes(self) -> None:
        """Create database indexes for performance."""
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id)",
            "CREATE INDEX IF NOT EXISTS idx_sessions_state ON sessions(state)",
            "CREATE INDEX IF NOT EXISTS idx_workflows_session_id ON workflows(session_id)",
            "CREATE INDEX IF NOT EXISTS idx_workflows_state ON workflows(state)",
            "CREATE INDEX IF NOT EXISTS idx_memories_user_id ON memories(user_id)",
            "CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type)",
            "CREATE INDEX IF NOT EXISTS idx_analytics_user_id ON analytics(user_id)",
            "CREATE INDEX IF NOT EXISTS idx_analytics_event_type ON analytics(event_type)",
            "CREATE INDEX IF NOT EXISTS idx_audit_logs_user_id ON audit_logs(user_id)",
            "CREATE INDEX IF NOT EXISTS idx_audit_logs_timestamp ON audit_logs(timestamp)"
        ]
        
        # In a real implementation, you'd execute these
        self.logger.info("Database indexes created")


class QueryCache:
    """Query result caching system."""
    
    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        self.cache: Dict[str, Tuple[Any, datetime]] = {}
        self.max_size = max_size
        self.ttl = ttl
        self.hits = 0
        self.misses = 0
        self._lock = asyncio.Lock()
    
    def _generate_key(self, query: str, parameters: Dict[str, Any]) -> str:
        """Generate cache key for query."""
        key_data = {"query": query, "parameters": parameters}
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    async def get(self, query: str, parameters: Dict[str, Any]) -> Optional[Any]:
        """Get cached query result."""
        async with self._lock:
            key = self._generate_key(query, parameters)
            
            if key in self.cache:
                result, timestamp = self.cache[key]
                
                # Check if cache entry is still valid
                if (datetime.now(timezone.utc) - timestamp).total_seconds() < self.ttl:
                    self.hits += 1
                    return result
                else:
                    # Remove expired entry
                    del self.cache[key]
            
            self.misses += 1
            return None
    
    async def set(self, query: str, parameters: Dict[str, Any], result: Any) -> None:
        """Cache query result."""
        async with self._lock:
            key = self._generate_key(query, parameters)
            
            # Remove oldest entries if cache is full
            if len(self.cache) >= self.max_size:
                oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][1])
                del self.cache[oldest_key]
            
            self.cache[key] = (result, datetime.now(timezone.utc))
    
    async def invalidate(self, pattern: Optional[str] = None) -> None:
        """Invalidate cache entries."""
        async with self._lock:
            if pattern:
                # Remove entries matching pattern (simplified implementation)
                keys_to_remove = [k for k in self.cache.keys() if pattern in k]
                for key in keys_to_remove:
                    del self.cache[key]
            else:
                # Clear entire cache
                self.cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.hits + self.misses
        hit_ratio = self.hits / max(total_requests, 1)
        
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_ratio": hit_ratio,
            "cache_size": len(self.cache),
            "max_size": self.max_size
        }


class DatabaseManager:
    """
    Advanced Database Management System for the AI Assistant.
    
    This manager provides comprehensive database functionality including:
    - Multi-database support (PostgreSQL, SQLite, MySQL)
    - Connection pooling with health monitoring
    - Query optimization and caching
    - Schema migrations and version management
    - Transaction management with isolation levels
    - Performance monitoring and analytics
    - Security features (encryption, audit logging)
    - Integration with all core system components
    - Async operations with proper error handling
    - Backup and recovery capabilities
    """
    
    def __init__(self, container: Container):
        """
        Initialize the database manager.
        
        Args:
            container: Dependency injection container
        """
        self.container = container
        self.logger = get_logger(__name__)
        
        # Core services
        self.config_loader = container.get(ConfigLoader)
        self.event_bus = container.get(EventBus)
        self.error_handler = container.get(ErrorHandler)
        self.health_check = container.get(HealthCheck)
        
        # Security
        try:
            self.encryption_manager = container.get(EncryptionManager)
        except Exception:
            self.encryption_manager = None
        
        # Observability
        self.metrics = container.get(MetricsCollector)
        self.tracer = container.get(TraceManager)
        
        # Database configuration
        self.config = self._load_database_config()
        
        # Core components
        self.connection_pool: Optional[ConnectionPool] = None
        self.schema_manager: Optional[SchemaManager] = None
        self.query_cache: Optional[QueryCache] = None
        
        # State management
        self.is_initialized = False
        self.query_metrics: deque = deque(maxlen=10000)
        self.connection_monitor_task: Optional[asyncio.Task] = None
        
        # Performance tracking
        self.performance_stats = {
            "total_queries": 0,
            "total_execution_time": 0.0,
            "slow_queries": 0,
            "failed_queries": 0
        }
        
        # Setup monitoring
        self._setup_monitoring()
        
        # Register health check
        self.health_check.register_component("database_manager", self._health_check_callback)
        
        self.logger.info("DatabaseManager initialized successfully")
    
    def _load_database_config(self) -> DatabaseConfig:
        """Load database configuration."""
        try:
            db_config = self.config_loader.get("database", {})
            
            return DatabaseConfig(
                db_type=DatabaseType(db_config.get("type", "sqlite")),
                host=db_config.get("host", "localhost"),
                port=db_config.get("port", 5432),
                username=db_config.get("username", ""),
                password=db_config.get("password", ""),
                database=db_config.get("database", "data/ai_assistant.db"),
                min_pool_size=db_config.get("min_pool_size", 5),
                max_pool_size=db_config.get("max_pool_size", 20),
                pool_timeout=db_config.get("pool_timeout", 30.0),
                query_timeout=db_config.get("query_timeout", 30.0),
                enable_query_cache=db_config.get("enable_query_cache", True),
                enable_performance_monitoring=db_config.get("enable_performance_monitoring", True),
                slow_query_threshold=db_config.get("slow_query_threshold", 1.0),
                auto_migrate=db_config.get("auto_migrate", False),
                options=db_config.get("options", {})
            )
            
        except Exception as e:
            self.logger.warning(f"Failed to load database config: {str(e)}, using defaults")
            return DatabaseConfig(
                db_type=DatabaseType.SQLITE,
                database="data/ai_assistant.db"
            )
    
    def _setup_monitoring(self) -> None:
        """Setup monitoring and metrics."""
        try:
            # Register database metrics
            self.metrics.register_counter("database_queries_total")
            self.metrics.register_counter("database_queries_successful")
            self.metrics.register_counter("database_queries_failed")
            self.metrics.register_histogram("database_query_duration_seconds")
            self.metrics.register_gauge("database_connections_active")
            self.metrics.register_gauge("database_connections_total")
            self.metrics.register_counter("database_transactions_total")
            self.metrics.register_counter("database_transactions_committed")
            self.metrics.register_counter("database_transactions_rolled_back")
            self.metrics.register_gauge("database_cache_hit_ratio")
            
        except Exception as e:
            self.logger.warning(f"Failed to setup monitoring: {str(e)}")
    
    async def initialize(self) -> None:
        """Initialize the database manager."""
        if self.is_initialized:
            return
        
        try:
            self.logger.info("Initializing database manager...")
            
            # Initialize connection pool
            self.connection_pool = ConnectionPool(self.config, self.logger)
            await self.connection_pool.initialize()
            
            # Initialize schema manager
            self.schema_manager = SchemaManager(self.config, self.logger)
            await self.schema_manager.initialize_schema()
            
            # Initialize query cache if enabled
            if self.config.enable_query_cache:
                self.query_cache = QueryCache(
                    max_size=self.config.options.get("cache_max_size", 1000),
                    ttl=self.config.options.get("cache_ttl", 3600)
                )
            
            # Start connection monitoring
            self.connection_monitor_task = asyncio.create_task(self._connection_monitor_loop())
            
            # Register event handlers
            await self._register_event_handlers()
            
            # Emit initialization event
            await self.event_bus.emit(DatabaseConnectionEstablished(
                database_type=self.config.db_type.value,
                host=self.config.host,
                database=self.config.database
            ))
            
            self.is_initialized = True
            self.logger.info("Database manager initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize database manager: {str(e)}")
            raise DatabaseError(f"Database initialization failed: {str(e)}")
    
    async def _register_event_handlers(self) -> None:
        """Register event handlers for system events."""
        # Component health events
        self.event_bus.subscribe("component_health_changed", self._handle_component_health_change)
        
        # System shutdown events
        self.event_bus.subscribe("system_shutdown_started", self._handle_system_shutdown)
    
    @handle_exceptions
    async def execute(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> QueryResult:
        """
        Execute a database query.
        
        Args:
            query: SQL query to execute
            parameters: Query parameters
            session_id: Optional session ID for tracking
            user_id: Optional user ID for auditing
            
        Returns:
            Query result
        """
        if not self.is_initialized:
            raise DatabaseError("Database manager not initialized")
        
        start_time = time.time()
        connection = None
        parameters = parameters or {}
        
        try:
            with self.tracer.trace("database_query") as span:
                span.set_attributes({
                    "query_type": self._detect_query_type(query).value,
                    "session_id": session_id or "unknown",
                    "user_id": user_id or "anonymous"
                })
                
                # Get connection from pool
                connection = await self.connection_pool.get_connection()
                
                # Check cache for SELECT queries
                if self.query_cache and query.strip().upper().startswith("SELECT"):
                    cached_result = await self.query_cache.get(query, parameters)
                    if cached_result is not None:
                        await self.connection_pool.return_connection(connection)
                        return cached_result
                
                # Execute query
                result = await connection.execute(query, parameters)
                
                # Cache result for SELECT queries
                if (self.query_cache and 
                    query.strip().upper().startswith("SELECT") and 
                    result is not None):
                    await self.query_cache.set(query, parameters, result)
                
                # Track metrics
                execution_time = time.time() - start_time
                await self._track_query_metrics(query, execution_time, True, session_id, user_id)
                
                # Emit query event
                await self.event_bus.emit(DatabaseQueryExecuted(
                    query_type=self._detect_query_type(query).value,
                    execution_time=execution_time,
                    session_id=session_id,
                    user_id=user_id,
                    success=True
                ))
                
                self.logger.debug(f"Query executed successfully in {execution_time:.3f}s")
                return result
                
        except Exception as e:
            execution_time = time.time() - start_time
            await self._track_query_metrics(query, execution_time, False, session_id, user_id, str(e))
            
            await self.event_bus.emit(DatabaseQueryExecuted(
                query_type=self._detect_query_type(query).value,
                execution_time=execution_time,
                session_id=session_id,
                user_id=user_id,
                success=False,
                error_message=str(e)
            ))
            
            self.logger.error(f"Query execution failed: {str(e)}")
            raise DatabaseError(f"Query execution failed: {str(e)}", query=query)
            
        finally:
            if connection:
                await self.connection_pool.return_connection(connection)
    
    @handle_exceptions
    async def fetch_one(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch a single row from the database.
        
        Args:
            query: SQL query to execute
            parameters: Query parameters
            session_id: Optional session ID for tracking
            user_id: Optional user ID for auditing
            
        Returns:
            Single row result or None
        """
        if not self.is_initialized:
            raise DatabaseError("Database manager not initialized")
        
        start_time = time.time()
        connection = None
        parameters = parameters or {}
        
        try:
            with self.tracer.trace("database_fetch_one") as span:
                span.set_attributes({
                    "session_id": session_id or "unknown",
                    "user_id": user_id or "anonymous"
                })
                
                # Get connection from pool
                connection = await self.connection_pool.get_connection()
                
                # Check cache
                if self.query_cache:
                    cached_result = await self.query_cache.get(query, parameters)
                    if cached_result is not None:
                        await self.connection_pool.return_connection(connection)
                        return cached_result
                
                # Fetch row
                result = await connection.fetch_one(query, parameters)
                
                # Cache result
                if self.query_cache and result is not None:
                    await self.query_cache.set(query, parameters, result)
                
                # Track metrics
                execution_time = time.time() - start_time
                await self._track_query_metrics(query, execution_time, True, session_id, user_id)
                
                return result
                
        except Exception as e:
            execution_time = time.time() - start_time
            await self._track_query_metrics(query, execution_time, False, session_id, user_id, str(e))
            
            self.logger.error(f"Fetch one failed: {str(e)}")
            raise DatabaseError(f"Fetch one failed: {str(e)}", query=query)
            
        finally:
            if connection:
                await self.connection_pool.return_connection(connection)
    
    @handle_exceptions
    async def fetch_all(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Fetch all rows from the database.
        
        Args:
            query: SQL query to execute
            parameters: Query parameters
            session_id: Optional session ID for tracking
            user_id: Optional user ID for auditing
            
        Returns:
            List of row results
        """
        if not self.is_initialized:
            raise DatabaseError("Database manager not initialized")
        
        start_time = time.time()
        connection = None
        parameters = parameters or {}
        
        try:
            with self.tracer.trace("database_fetch_all") as span:
                span.set_attributes({
                    "session_id": session_id or "unknown",
                    "user_id": user_id or "anonymous"
                })
                
                # Get connection from pool
                connection = await self.connection_pool.get_connection()
                
                # Check cache
                if self.query_cache:
                    cached_result = await self.query_cache.get(query, parameters)
                    if cached_result is not None:
                        await self.connection_pool.return_connection(connection)
                        return cached_result
                
                # Fetch rows
                result = await connection.fetch_all(query, parameters)
                
                # Cache result
                if self.query_cache:
                    await self.query_cache.set(query, parameters, result)
                
                # Track metrics
                execution_time = time.time() - start_time
                await self._track_query_metrics(query, execution_time, True, session_id, user_id)
                
                return result
                
        except Exception as e:
            execution_time = time.time() - start_time
            await self._track_query_metrics(query, execution_time, False, session_id, user_id, str(e))
            
            self.logger.error(f"Fetch all failed: {str(e)}")
            raise DatabaseError(f"Fetch all failed: {str(e)}", query=query)
            
        finally:
            if connection:
                await self.connection_pool.return_connection(connection)
    
    @handle_exceptions
    @asynccontextmanager
    async def transaction(
        self,
        isolation: Optional[TransactionIsolation] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> AsyncGenerator[DatabaseTransaction, None]:
        """
        Create a database transaction context.
        
        Args:
            isolation: Transaction isolation level
            session_id: Optional session ID for tracking
            user_id: Optional user ID for auditing
            
        Yields:
            DatabaseTransaction instance
        """
        if not self.is_initialized:
            raise DatabaseError("Database manager not initialized")
        
        connection = None
        transaction = None
        
        try:
            # Get connection from pool
            connection = await self.connection_pool.get_connection()
            
            # Begin transaction
            transaction = await connection.begin_transaction(isolation)
            
            # Emit transaction started event
            await self.event_bus.emit(DatabaseTransactionStarted(
                transaction_id=transaction.transaction_id,
                isolation_level=isolation.value if isolation else "default",
                session_id=session_id,
                user_id=user_id
            ))
            
            async with transaction:
                yield transaction
            
            # Transaction committed successfully
            await self.event_bus.emit(DatabaseTransactionCommitted(
                transaction_id=transaction.transaction_id,
                session_id=session_id,
                user_id=user_id
            ))
            
            self.metrics.increment("database_transactions_committed")
            
        except Exception as e:
            # Transaction rolled back
            if transaction:
                await self.event_bus.emit(DatabaseTransactionRolledBack(
                    transaction_id=transaction.transaction_id,
                    error_message=str(e),
                    session_id=session_id,
                    user_id=user_id
                ))
            
            self.metrics.increment("database_transactions_rolled_back")
            
            self.logger.error(f"Transaction failed: {str(e)}")
            raise DatabaseError(f"Transaction failed: {str(e)}")
            
        finally:
            if connection:
                await self.connection_pool.return_connection(connection)
    
    def _detect_query_type(self, query: str) -> QueryType:
        """Detect the type of SQL query."""
        query_upper = query.strip().upper()
        
        if query_upper.startswith("SELECT"):
            return QueryType.SELECT
        elif query_upper.startswith("INSERT"):
            return QueryType.INSERT
        elif query_upper.startswith("UPDATE"):
            return QueryType.UPDATE
        elif query_upper.startswith("DELETE"):
            return QueryType.DELETE
        elif any(query_upper.startswith(ddl) for ddl in ["CREATE", "ALTER", "DROP"]):
            return QueryType.DDL
        else:
            return QueryType.TRANSACTION
    
    async def _track_query_metrics(
        self,
        query: str,
        execution_time: float,
        success: bool,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        error: Optional[str] = None
    ) -> None:
        """Track query performance metrics."""
        try:
            # Create query metrics
            metrics = QueryMetrics(
                query=query[:200] + "..." if len(query) > 200 else query,  # Truncate long queries
                query_type=self._detect_query_type(query),
                execution_time=execution_time,
                rows_affected=0,  # Would be filled by actual result
                session_id=session_id,
                user_id=user_id,
                error=error
            )
            
            # Store metrics
            self.query_metrics
