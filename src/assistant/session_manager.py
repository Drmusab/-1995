"""
Advanced Session Management System for AI Assistant
Author: Drmusab
Last Modified: 2025-05-26 15:59:42 UTC

This module provides comprehensive session management for the AI assistant,
handling user sessions, state persistence, context management, and seamless
integration with all core system components.
"""

from pathlib import Path
from typing import Optional, Dict, Any, List, Set, Callable, Type, Union, AsyncGenerator, TypeVar
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
from collections import defaultdict, deque
import weakref
from abc import ABC, abstractmethod
import logging
import pickle
import base64
from concurrent.futures import ThreadPoolExecutor

# Core imports
from src.core.config.loader import ConfigLoader
from src.core.events.event_bus import EventBus
from src.core.events.event_types import (
    SessionStarted, SessionEnded, SessionExpired, SessionRestored,
    SessionContextUpdated, SessionStateChanged, SessionCleanupStarted,
    SessionCleanupCompleted, UserJoinedSession, UserLeftSession,
    SessionMigrated, SessionClusteringStarted, SessionHealthCheckFailed,
    ErrorOccurred, SystemStateChanged, ComponentHealthChanged
)
from src.core.error_handling import ErrorHandler, handle_exceptions
from src.core.dependency_injection import Container
from src.core.health_check import HealthCheck
from src.core.security.authentication import AuthenticationManager
from src.core.security.authorization import AuthorizationManager
from src.core.security.encryption import EncryptionManager

# Memory and storage
from src.memory.memory_manager import MemoryManager
from src.memory.context_manager import ContextManager
from src.memory.working_memory import WorkingMemory
from src.memory.episodic_memory import EpisodicMemory
from src.integrations.storage.database import DatabaseManager
from src.integrations.cache.redis_cache import RedisCache
from src.integrations.storage.backup_manager import BackupManager

# Learning and adaptation
from src.learning.continual_learning import ContinualLearner
from src.learning.preference_learning import PreferenceLearner
from src.learning.feedback_processor import FeedbackProcessor

# Observability
from src.observability.monitoring.metrics import MetricsCollector
from src.observability.monitoring.tracing import TraceManager
from src.observability.logging.config import get_logger

# Type definitions
T = TypeVar('T')


class SessionState(Enum):
    """Session lifecycle states."""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    IDLE = "idle"
    PAUSED = "paused"
    SUSPENDED = "suspended"
    EXPIRING = "expiring"
    EXPIRED = "expired"
    TERMINATED = "terminated"
    ERROR = "error"
    MIGRATING = "migrating"


class SessionType(Enum):
    """Types of sessions."""
    INTERACTIVE = "interactive"      # Real-time user interaction
    BATCH = "batch"                 # Batch processing session
    API = "api"                     # API-based session
    BACKGROUND = "background"       # Background processing
    SYSTEM = "system"               # System maintenance session
    GUEST = "guest"                 # Anonymous guest session
    AUTHENTICATED = "authenticated" # Authenticated user session


class SessionPriority(Enum):
    """Session priority levels."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3
    SYSTEM = 4


class ClusterNode(Enum):
    """Session clustering node types."""
    PRIMARY = "primary"
    SECONDARY = "secondary"
    REPLICA = "replica"
    BACKUP = "backup"


@dataclass
class SessionConfiguration:
    """Configuration settings for a session."""
    session_type: SessionType = SessionType.INTERACTIVE
    priority: SessionPriority = SessionPriority.NORMAL
    max_idle_time: float = 1800.0  # 30 minutes
    max_session_time: float = 86400.0  # 24 hours
    cleanup_on_expire: bool = True
    persist_context: bool = True
    enable_clustering: bool = False
    enable_backup: bool = True
    auto_save_interval: float = 300.0  # 5 minutes
    context_window_size: int = 4096
    memory_limit_mb: float = 512.0
    cpu_limit_percent: float = 50.0
    network_timeout: float = 30.0
    encryption_enabled: bool = True
    compression_enabled: bool = True
    audit_logging: bool = True
    analytics_enabled: bool = True


@dataclass
class SessionContext:
    """Comprehensive session context data."""
    session_id: str
    user_id: Optional[str] = None
    
    # Session metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_activity: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_heartbeat: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # User information
    user_profile: Dict[str, Any] = field(default_factory=dict)
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    authentication_data: Dict[str, Any] = field(default_factory=dict)
    
    # Conversation state
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    current_topic: Optional[str] = None
    conversation_flow: List[str] = field(default_factory=list)
    
    # Processing context
    active_workflows: Set[str] = field(default_factory=set)
    interaction_history: List[Dict[str, Any]] = field(default_factory=list)
    processing_queue: List[Dict[str, Any]] = field(default_factory=list)
    
    # Memory context
    working_memory_data: Dict[str, Any] = field(default_factory=dict)
    episodic_memories: List[str] = field(default_factory=list)
    semantic_context: Dict[str, Any] = field(default_factory=dict)
    
    # Environment context
    device_info: Dict[str, Any] = field(default_factory=dict)
    network_info: Dict[str, Any] = field(default_factory=dict)
    location_info: Dict[str, Any] = field(default_factory=dict)
    timezone_info: str = "UTC"
    
    # Technical context
    api_keys: Dict[str, str] = field(default_factory=dict)
    feature_flags: Dict[str, bool] = field(default_factory=dict)
    experiments: Dict[str, Any] = field(default_factory=dict)
    
    # Performance context
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    resource_usage: Dict[str, float] = field(default_factory=dict)
    quality_settings: Dict[str, str] = field(default_factory=dict)
    
    # Custom data
    custom_data: Dict[str, Any] = field(default_factory=dict)
    tags: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SessionInfo:
    """Runtime session information."""
    session_id: str
    state: SessionState = SessionState.INITIALIZING
    config: SessionConfiguration = field(default_factory=SessionConfiguration)
    context: SessionContext = field(default_factory=lambda: SessionContext(""))
    
    # Lifecycle tracking
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    last_activity: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = None
    
    # Resource tracking
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    network_bytes_sent: int = 0
    network_bytes_received: int = 0
    
    # Health and performance
    health_score: float = 1.0
    response_time_avg: float = 0.0
    error_count: int = 0
    warning_count: int = 0
    
    # Clustering information
    cluster_node: Optional[str] = None
    primary_node: Optional[str] = None
    replica_nodes: Set[str] = field(default_factory=set)
    
    # Version and consistency
    version: int = 1
    checksum: Optional[str] = None
    last_backup: Optional[datetime] = None
    
    # Statistics
    interaction_count: int = 0
    workflow_count: int = 0
    message_count: int = 0
    total_processing_time: float = 0.0


class SessionError(Exception):
    """Custom exception for session management operations."""
    
    def __init__(self, message: str, session_id: Optional[str] = None, 
                 error_code: Optional[str] = None, user_id: Optional[str] = None):
        super().__init__(message)
        self.session_id = session_id
        self.error_code = error_code
        self.user_id = user_id
        self.timestamp = datetime.now(timezone.utc)


class SessionStore(ABC):
    """Abstract interface for session storage backends."""
    
    @abstractmethod
    async def store_session(self, session_info: SessionInfo) -> None:
        """Store session information."""
        pass
    
    @abstractmethod
    async def load_session(self, session_id: str) -> Optional[SessionInfo]:
        """Load session information."""
        pass
    
    @abstractmethod
    async def delete_session(self, session_id: str) -> None:
        """Delete session information."""
        pass
    
    @abstractmethod
    async def list_sessions(self, user_id: Optional[str] = None) -> List[str]:
        """List session IDs."""
        pass
    
    @abstractmethod
    async def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions and return count."""
        pass


class MemorySessionStore(SessionStore):
    """In-memory session store for development and testing."""
    
    def __init__(self):
        self.sessions: Dict[str, SessionInfo] = {}
        self.user_sessions: Dict[str, Set[str]] = defaultdict(set)
        self.lock = threading.Lock()
    
    async def store_session(self, session_info: SessionInfo) -> None:
        """Store session in memory."""
        with self.lock:
            self.sessions[session_info.session_id] = session_info
            if session_info.context.user_id:
                self.user_sessions[session_info.context.user_id].add(session_info.session_id)
    
    async def load_session(self, session_id: str) -> Optional[SessionInfo]:
        """Load session from memory."""
        with self.lock:
            return self.sessions.get(session_id)
    
    async def delete_session(self, session_id: str) -> None:
        """Delete session from memory."""
        with self.lock:
            session_info = self.sessions.pop(session_id, None)
            if session_info and session_info.context.user_id:
                self.user_sessions[session_info.context.user_id].discard(session_id)
    
    async def list_sessions(self, user_id: Optional[str] = None) -> List[str]:
        """List sessions in memory."""
        with self.lock:
            if user_id:
                return list(self.user_sessions.get(user_id, set()))
            return list(self.sessions.keys())
    
    async def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions."""
        current_time = datetime.now(timezone.utc)
        expired_sessions = []
        
        with self.lock:
            for session_id, session_info in self.sessions.items():
                if session_info.expires_at and current_time > session_info.expires_at:
                    expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            await self.delete_session(session_id)
        
        return len(expired_sessions)


class DatabaseSessionStore(SessionStore):
    """Database-backed session store for production."""
    
    def __init__(self, database: DatabaseManager, encryption: Optional[EncryptionManager] = None):
        self.database = database
        self.encryption = encryption
        self.logger = get_logger(__name__)
    
    async def store_session(self, session_info: SessionInfo) -> None:
        """Store session in database."""
        try:
            # Serialize session data
            session_data = self._serialize_session(session_info)
            
            # Encrypt if enabled
            if self.encryption:
                session_data = await self.encryption.encrypt(session_data)
            
            # Store in database
            await self.database.execute(
                """
                INSERT INTO sessions (session_id, user_id, state, data, created_at, expires_at, checksum)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (session_id) DO UPDATE SET
                    state = excluded.state,
                    data = excluded.data,
                    expires_at = excluded.expires_at,
                    checksum = excluded.checksum,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (
                    session_info.session_id,
                    session_info.context.user_id,
                    session_info.state.value,
                    session_data,
                    session_info.created_at,
                    session_info.expires_at,
                    session_info.checksum
                )
            )
            
        except Exception as e:
            self.logger.error(f"Failed to store session {session_info.session_id}: {str(e)}")
            raise SessionError(f"Failed to store session: {str(e)}", session_info.session_id)
    
    async def load_session(self, session_id: str) -> Optional[SessionInfo]:
        """Load session from database."""
        try:
            result = await self.database.fetch_one(
                "SELECT data, checksum FROM sessions WHERE session_id = ? AND state != 'expired'",
                (session_id,)
            )
            
            if not result:
                return None
            
            session_data, checksum = result
            
            # Decrypt if enabled
            if self.encryption:
                session_data = await self.encryption.decrypt(session_data)
            
            # Deserialize session
            session_info = self._deserialize_session(session_data)
            session_info.checksum = checksum
            
            return session_info
            
        except Exception as e:
            self.logger.error(f"Failed to load session {session_id}: {str(e)}")
            return None
    
    async def delete_session(self, session_id: str) -> None:
        """Delete session from database."""
        try:
            await self.database.execute(
                "UPDATE sessions SET state = 'expired', updated_at = CURRENT_TIMESTAMP WHERE session_id = ?",
                (session_id,)
            )
            
        except Exception as e:
            self.logger.error(f"Failed to delete session {session_id}: {str(e)}")
            raise SessionError(f"Failed to delete session: {str(e)}", session_id)
    
    async def list_sessions(self, user_id: Optional[str] = None) -> List[str]:
        """List sessions from database."""
        try:
            if user_id:
                results = await self.database.fetch_all(
                    "SELECT session_id FROM sessions WHERE user_id = ? AND state != 'expired'",
                    (user_id,)
                )
            else:
                results = await self.database.fetch_all(
                    "SELECT session_id FROM sessions WHERE state != 'expired'"
                )
            
            return [row[0] for row in results]
            
        except Exception as e:
            self.logger.error(f"Failed to list sessions: {str(e)}")
            return []
    
    async def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions from database."""
        try:
            result = await self.database.execute(
                """
                UPDATE sessions 
                SET state = 'expired', updated_at = CURRENT_TIMESTAMP 
                WHERE expires_at < CURRENT_TIMESTAMP AND state != 'expired'
                """
            )
            
            # Delete old expired sessions
            await self.database.execute(
                """
                DELETE FROM sessions 
                WHERE state = 'expired' AND updated_at < datetime('now', '-7 days')
                """
            )
            
            return result.rowcount if result else 0
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup expired sessions: {str(e)}")
            return 0
    
    def _serialize_session(self, session_info: SessionInfo) -> str:
        """Serialize session info to string."""
        try:
            # Convert to dictionary
            data = asdict(session_info)
            
            # Handle datetime objects
            for key, value in data.items():
                if isinstance(value, datetime):
                    data[key] = value.isoformat()
                elif isinstance(value, dict):
                    for k, v in value.items():
                        if isinstance(v, datetime):
                            data[key][k] = v.isoformat()
            
            # Convert to JSON and encode
            json_data = json.dumps(data, default=str)
            return base64.b64encode(json_data.encode()).decode()
            
        except Exception as e:
            raise SessionError(f"Failed to serialize session: {str(e)}")
    
    def _deserialize_session(self, session_data: str) -> SessionInfo:
        """Deserialize session info from string."""
        try:
            # Decode and parse JSON
            json_data = base64.b64decode(session_data.encode()).decode()
            data = json.loads(json_data)
            
            # Convert datetime strings back
            datetime_fields = ['created_at', 'started_at', 'last_activity', 'expires_at', 'last_backup']
            for field in datetime_fields:
                if data.get(field):
                    data[field] = datetime.fromisoformat(data[field])
            
            # Handle context datetime fields
            context_data = data.get('context', {})
            context_datetime_fields = ['created_at', 'last_activity', 'last_heartbeat']
            for field in context_datetime_fields:
                if context_data.get(field):
                    context_data[field] = datetime.fromisoformat(context_data[field])
            
            # Convert enums
            data['state'] = SessionState(data['state'])
            if 'config' in data:
                config_data = data['config']
                config_data['session_type'] = SessionType(config_data['session_type'])
                config_data['priority'] = SessionPriority(config_data['priority'])
                data['config'] = SessionConfiguration(**config_data)
            
            # Reconstruct context
            if 'context' in data:
                data['context'] = SessionContext(**context_data)
            
            return SessionInfo(**data)
            
        except Exception as e:
            raise SessionError(f"Failed to deserialize session: {str(e)}")


class SessionCluster:
    """Manages session clustering across multiple nodes."""
    
    def __init__(self, node_id: str, redis_cache: Optional[RedisCache] = None):
        self.node_id = node_id
        self.redis_cache = redis_cache
        self.cluster_nodes: Dict[str, Dict[str, Any]] = {}
        self.session_assignments: Dict[str, str] = {}  # session_id -> node_id
        self.logger = get_logger(__name__)
    
    async def register_node(self, node_id: str, node_info: Dict[str, Any]) -> None:
        """Register a cluster node."""
        self.cluster_nodes[node_id] = {
            **node_info,
            'last_heartbeat': datetime.now(timezone.utc),
            'status': 'active'
        }
        
        if self.redis_cache:
            await self.redis_cache.set(
                f"cluster:node:{node_id}",
                json.dumps(node_info, default=str),
                ttl=60
            )
    
    async def assign_session(self, session_id: str) -> str:
        """Assign session to the best available node."""
        # Simple round-robin assignment for now
        active_nodes = [
            node_id for node_id, info in self.cluster_nodes.items()
            if info['status'] == 'active'
        ]
        
        if not active_nodes:
            return self.node_id  # Fallback to current node
        
        # Select node with least load
        best_node = min(active_nodes, key=lambda n: self.cluster_nodes[n].get('load', 0))
        self.session_assignments[session_id] = best_node
        
        if self.redis_cache:
            await self.redis_cache.set(f"session:assignment:{session_id}", best_node)
        
        return best_node
    
    async def migrate_session(self, session_id: str, target_node: str) -> bool:
        """Migrate session to target node."""
        try:
            # Implementation would depend on inter-node communication
            # For now, just update assignment
            self.session_assignments[session_id] = target_node
            
            if self.redis_cache:
                await self.redis_cache.set(f"session:assignment:{session_id}", target_node)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to migrate session {session_id}: {str(e)}")
            return False
    
    async def handle_node_failure(self, failed_node: str) -> None:
        """Handle node failure by reassigning sessions."""
        if failed_node in self.cluster_nodes:
            self.cluster_nodes[failed_node]['status'] = 'failed'
        
        # Reassign sessions from failed node
        failed_sessions = [
            session_id for session_id, node_id in self.session_assignments.items()
            if node_id == failed_node
        ]
        
        for session_id in failed_sessions:
            new_node = await self.assign_session(session_id)
            self.logger.info(f"Reassigned session {session_id} from {failed_node} to {new_node}")


class EnhancedSessionManager:
    """
    Advanced Session Management System for the AI Assistant.
    
    This manager provides comprehensive session handling including:
    - Multi-user session support with isolation
    - Session state persistence and recovery
    - Context-aware session management
    - Integration with all core system components
    - Session clustering for scalability
    - Real-time monitoring and analytics
    - Security and authentication integration
    - Automatic cleanup and optimization
    - Event-driven session lifecycle
    - Memory and resource management
    """
    
    def __init__(self, container: Container):
        """
        Initialize the enhanced session manager.
        
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
        
        # Memory and storage
        self.memory_manager = container.get(MemoryManager)
        self.context_manager = container.get(ContextManager)
        self.working_memory = container.get(WorkingMemory)
        self.episodic_memory = container.get(EpisodicMemory)
        
        # Security components
        try:
            self.auth_manager = container.get(AuthenticationManager)
            self.authz_manager = container.get(AuthorizationManager)
            self.encryption_manager = container.get(EncryptionManager)
        except Exception:
            self.auth_manager = None
            self.authz_manager = None
            self.encryption_manager = None
        
        # Storage and caching
        try:
            self.database = container.get(DatabaseManager)
            self.redis_cache = container.get(RedisCache)
            self.backup_manager = container.get(BackupManager)
        except Exception:
            self.database = None
            self.redis_cache = None
            self.backup_manager = None
        
        # Learning systems
        try:
            self.continual_learner = container.get(ContinualLearner)
            self.preference_learner = container.get(PreferenceLearner)
            self.feedback_processor = container.get(FeedbackProcessor)
        except Exception:
            self.continual_learner = None
            self.preference_learner = None
            self.feedback_processor = None
        
        # Observability
        self.metrics = container.get(MetricsCollector)
        self.tracer = container.get(TraceManager)
        
        # Session management
        self.active_sessions: Dict[str, SessionInfo] = {}
        self.session_locks: Dict[str, asyncio.Lock] = {}
        self.user_sessions: Dict[str, Set[str]] = defaultdict(set)
        
        # Storage backend
        self._setup_session_store()
        
        # Clustering support
        self.node_id = self.config.get("sessions.node_id", f"node_{uuid.uuid4().hex[:8]}")
        self.enable_clustering = self.config.get("sessions.enable_clustering", False)
        if self.enable_clustering:
            self.cluster = SessionCluster(self.node_id, self.redis_cache)
        else:
            self.cluster = None
        
        # Configuration
        self.default_config = SessionConfiguration(
            max_idle_time=self.config.get("sessions.max_idle_time", 1800.0),
            max_session_time=self.config.get("sessions.max_session_time", 86400.0),
            cleanup_on_expire=self.config.get("sessions.cleanup_on_expire", True),
            persist_context=self.config.get("sessions.persist_context", True),
            enable_clustering=self.enable_clustering,
            auto_save_interval=self.config.get("sessions.auto_save_interval", 300.0),
            encryption_enabled=self.config.get("sessions.encryption_enabled", True),
            audit_logging=self.config.get("sessions.audit_logging", True)
        )
        
        # Performance tracking
        self.session_stats: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.cleanup_stats: Dict[str, int] = defaultdict(int)
        
        # Background tasks
        self.cleanup_task: Optional[asyncio.Task] = None
        self.heartbeat_task: Optional[asyncio.Task] = None
        self.backup_task: Optional[asyncio.Task] = None
        
        # Setup monitoring and health checks
        self._setup_monitoring()
        self.health_check.register_component("session_manager", self._health_check_callback)
        
        self.logger.info("EnhancedSessionManager initialized successfully")

    def _setup_session_store(self) -> None:
        """Setup appropriate session storage backend."""
        storage_type = self.config.get("sessions.storage_type", "memory")
        
        if storage_type == "database" and self.database:
            self.session_store = DatabaseSessionStore(self.database, self.encryption_manager)
        else:
            self.session_store = MemorySessionStore()
        
        self.logger.info(f"Using {type(self.session_store).__name__} for session storage")

    def _setup_monitoring(self) -> None:
        """Setup monitoring and metrics."""
        try:
            # Register session metrics
            self.metrics.register_counter("sessions_created_total")
            self.metrics.register_counter("sessions_ended_total")
            self.metrics.register_counter("sessions_expired_total")
            self.metrics.register_gauge("active_sessions")
            self.metrics.register_gauge("sessions_by_state")
            self.metrics.register_histogram("session_duration_seconds")
            self.metrics.register_histogram("session_memory_usage_mb")
            self.metrics.register_counter("session_errors_total")
            self.metrics.register_counter("session_migrations_total")
            
        except Exception as e:
            self.logger.warning(f"Failed to setup monitoring: {str(e)}")

    async def initialize(self) -> None:
        """Initialize the session manager."""
        try:
            # Initialize session store
            if hasattr(self.session_store, 'initialize'):
                await self.session_store.initialize()
            
            # Initialize cluster if enabled
            if self.cluster:
                await self.cluster.register_node(self.node_id, {
                    'hostname': self.config.get("hostname", "localhost"),
                    'capacity': self.config.get("sessions.max_sessions_per_node", 1000),
                    'load': 0
                })
            
            # Start background tasks
            self.cleanup_task = asyncio.create_task(self._cleanup_loop())
            self.heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            
            if self.backup_manager:
                self.backup_task = asyncio.create_task(self._backup_loop())
            
            # Register event handlers
            await self._register_event_handlers()
            
            # Load existing sessions if recovering
            await self._recover_sessions()
            
            self.logger.info("SessionManager initialization completed")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize SessionManager: {str(e)}")
            raise SessionError(f"Initialization failed: {str(e)}")

    async def _register_event_handlers(self) -> None:
        """Register event handlers for system events."""
        # Component health events
        self.event_bus.subscribe("component_health_changed", self._handle_component_health_change)
        
        # System shutdown events
        self.event_bus.subscribe("system_shutdown_started", self._handle_system_shutdown)
        
        # User events
        self.event_bus.subscribe("user_authenticated", self._handle_user_authentication)
        self.event_bus.subscribe("user_logged_out", self._handle_user_logout)

    async def _recover_sessions(self) -> None:
        """Recover sessions after restart."""
        try:
            session_ids = await self.session_store.list_sessions()
            recovered_count = 0
            
            for session_id in session_ids:
                try:
                    session_info = await self.session_store.load_session(session_id)
                    if session_info and session_info.state in [SessionState.ACTIVE, SessionState.IDLE]:
                        # Check if session should be expired
                        if session_info.expires_at and datetime.now(timezone.utc) > session_info.expires_at:
                            await self._expire_session(session_id)
                        else:
                            # Restore session
                            self.active_sessions[session_id] = session_info
                            self.session_locks[session_id] = asyncio.Lock()
                            
                            if session_info.context.user_id:
                                self.user_sessions[session_info.context.user_id].add(session_id)
                            
                            recovered_count += 1
                            
                except Exception as e:
                    self.logger.warning(f"Failed to recover session {session_id}: {str(e)}")
            
            self.logger.info(f"Recovered {recovered_count} active sessions")
            
        except Exception as e:
            self.logger.error(f"Failed to recover sessions: {str(e)}")

    @handle_exceptions
    async def create_session(
        self,
        user_id: Optional[str] = None,
        session_config: Optional[SessionConfiguration] = None,
        context_data: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a new session.
        
        Args:
            user_id: Optional user identifier
            session_config: Session configuration
            context_data: Initial context data
            
        Returns:
            Session ID
        """
        session_id = str(uuid.uuid4())
        config = session_config or self.default_config
        
        # Create session context
        context = SessionContext(
            session_id=session_id,
            user_id=user_id
        )
        
        if context_data:
            context.custom_data.update(context_data)
        
        # Load user profile and preferences
        if user_id:
            await self._load_user_context(context)
        
        # Calculate expiration time
        expires_at = datetime.now(timezone.utc) + timedelta(seconds=config.max_session_time)
        
        # Create session info
        session_info = SessionInfo(
            session_id=session_id,
            state=SessionState.INITIALIZING,
            config=config,
            context=context,
            expires_at=expires_at
        )
        
        # Assign to cluster node if clustering enabled
        if self.cluster:
            target_node = await self.cluster.assign_session(session_id)
            session_info.cluster_node = target_node
            
            if target_node != self.node_id:
                # Session assigned to different node
                await self.cluster.migrate_session(session_id, target_node)
                return session_id
        
        # Store session
        async with self._get_session_lock(session_id):
            try:
                # Initialize session state
                session_info.state = SessionState.ACTIVE
                session_info.started_at = datetime.now(timezone.utc)
                
                # Store in memory and persistence
                self.active_sessions[session_id] = session_info
                await self.session_store.store_session(session_info)
                
                # Track user sessions
                if user_id:
                    self.user_sessions[user_id].add(session_id)
                
                # Initialize working memory for session
                await self.working_memory.initialize_session(session_id)
                
                # Emit session started event
                await self.event_bus.emit(SessionStarted(
                    session_id=session_id,
                    user_id=user_id,
                    session_type=config.session_type.value,
                    created_at=session_info.created_at
                ))
                
                # Update metrics
                self.metrics.increment("sessions_created_total")
                self.metrics.set("active_sessions", len(self.active_sessions))
                
                self.logger.info(f"Created session: {session_id} for user: {user_id}")
                return session_id
                
            except Exception as e:
                # Cleanup on failure
                self.active_sessions.pop(session_id, None)
                if user_id:
                    self.user_sessions[user_id].discard(session_id)
                
                raise SessionError(f"Failed to create session: {str(e)}", session_id)

    async def _load_user_context(self, context: SessionContext) -> None:
        """Load user profile and preferences into session context."""
        try:
            user_id = context.user_id
            if not user_id:
                return
            
            # Load user preferences
            if self.preference_learner:
                user_prefs = await self.preference_learner.get_user_preferences(user_id)
                context.user_preferences.update(user_prefs or {})
            
            # Load authentication data
            if self.auth_manager:
                auth_data = await self.auth_manager.get_user_info(user_id)
                context.authentication_data.update(auth_data or {})
            
            # Load user profile from memory
            user_memories = await self.episodic_memory.get_user_memories(user_id, limit=10)
            context.episodic_memories.extend([m.get('id') for m in user_memories if m.get('id')])
            
        except Exception as e:
            self.logger.warning(f"Failed to load user context for {context.user_id}: {str(e)}")

    def _get_session_lock(self, session_id: str) -> asyncio.Lock:
        """Get or create a lock for session operations."""
        if session_id not in self.session_locks:
            self.session_locks[session_id] = asyncio.Lock()
        return self.session_locks[session_id]

    @handle_exceptions
    async def get_session(self, session_id: str) -> Optional[SessionInfo]:
        """
        Get session information.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session information or None if not found
        """
        # Check active sessions first
        if session_id in self.active_sessions:
            session_info = self.active_sessions[session_id]
            await self._update_last_activity(session_info)
            return session_info
        
        # Try to load from storage
        session_info = await self.session_store.load_session(session_id)
        if session_info:
            # Check if session is still valid
            if session_info.expires_at and datetime.now(timezone.utc) > session_info.expires_at:
                await self._expire_session(session_id)
                return None
            
            # Restore to active sessions
            self.active_sessions[session_id] = session_info
            self.session_locks[session_id] = asyncio.Lock()
            
            if session_info.context.user_id:
                self.user_sessions[session_info.context.user_id].add(session_id)
            
            await self._update_last_activity(session_info)
            return session_info
        
        return None

    async def _update_last_activity(self, session_info: SessionInfo) -> None:
        """Update session last activity timestamp."""
        current_time = datetime.now(timezone.utc)
        session_info.last_activity = current_time
        session_info.context.last_activity = current_time
        session_info.context.last_heartbeat = current_time
        
        # Save periodically to avoid too frequent writes
        if hasattr(session_info, '_last_save'):
            time_since_save = (current_time - session_info._last_save).total_seconds()
            if time_since_save < 30:  # Don't save more than once per 30 seconds
                return
        
        session_info._last_save = current_time
        await self.session_store.store_session(session_info)

    @handle_exceptions
    async def update_session_context(
        self,
        session_id: str,
        context_updates: Dict[str, Any]
    ) -> None:
        """
        Update session context.
        
        Args:
            session_id: Session identifier
            context_updates: Context data to update
        """
        session_info = await self.get_session(session_id)
        if not session_info:
            raise SessionError(f"Session {session_id} not found")
        
        async with self._get_session_lock(session_id):
            try:
                # Update context
                for key, value in context_updates.items():
                    if hasattr(session_info.context, key):
                        setattr(session_info.context, key, value)
                    else:
                        session_info.context.custom_data[key] = value
                
                # Update version and checksum
                session_info.version += 1
                session_info.checksum = self._calculate_checksum(session_info)
                
                # Save updated session
                await self.session_store.store_session(session_info)
                
                # Emit context updated event
                await self.event_bus.emit(SessionContextUpdated(
                    session_id=session_id,
                    user_id=session_info.context.user_id,
                    updates=list(context_updates.keys())
                ))
                
                self.logger.debug(f"Updated context for session: {session_id}")
                
            except Exception as e:
                raise SessionError(f"Failed to update session context: {str(e)}", session_id)

    def _calculate_checksum(self, session_info: SessionInfo) -> str:
        """Calculate checksum for session data integrity."""
        try:
            # Create a simplified representation for checksum
            data = {
                'session_id': session_info.session_id,
                'version': session_info.version,
                'state': session_info.state.value,
                'context_size': len(str(session_info.context.custom_data))
            }
            return hashlib.md5(json.dumps(data, sort_keys=True).encode()).hexdigest()
        except Exception:
            return "unknown"

    @handle_exceptions
    async def add_interaction(
        self,
        session_id: str,
        interaction_data: Dict[str, Any]
    ) -> None:
        """
        Add interaction data to session.
        
        Args:
            session_id: Session identifier
            interaction_data: Interaction data
        """
        session_info = await self.get_session(session_id)
        if not session_info:
            raise SessionError(f"Session {session_id} not found")
        
        async with self._get_session_lock(session_id):
            # Add to interaction history
            interaction_entry = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'data': interaction_data
            }
            
            session_info.context.interaction_history.append(interaction_entry)
            session_info.interaction_count += 1
            
            # Limit history size
            max_history = session_info.config.context_window_size
            if len(session_info.context.interaction_history) > max_history:
                session_info.context.interaction_history = session_info.context.interaction_history[-max_history:]
            
            # Update performance metrics
            if 'processing_time' in interaction_data:
                session_info.total_processing_time += interaction_data['processing_time']
                
                # Update average response time
                session_info.response_time_avg = (
                    session_info.total_processing_time / session_info.interaction_count
                )
            
            await self._update_last_activity(session_info)

    @handle_exceptions
    async def add_workflow(
        self,
        session_id: str,
        workflow_id: str,
        workflow_data: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add workflow to session.
        
        Args:
            session_id: Session identifier
            workflow_id: Workflow identifier
            workflow_data: Optional workflow data
        """
        session_info = await self.get_session(session_id)
        if not session_info:
            raise SessionError(f"Session {session_id} not found")
        
        async with self._get_session_lock(session_id):
            session_info.context.active_workflows.add(workflow_id)
            session_info.workflow_count += 1
            
            if workflow_data:
                session_info.context.custom_data[f"workflow_{workflow_id}"] = workflow_data
            
            await self._update_last_activity(session_info)

    @handle_exceptions
    async def remove_workflow(
        self,
        session_id: str,
        workflow_id: str
    ) -> None:
        """
        Remove workflow from session.
        
        Args:
            session_id: Session identifier
            workflow_id: Workflow identifier
        """
        session_info = await self.get_session(session_id)
        if not session_info:
            return
        
        async with self._get_session_lock(session_id):
            session_info.context.active_workflows.discard(workflow_id)
            session_info.context.custom_data.pop(f"workflow_{workflow_id}", None)
            
            await self._update_last_activity(session_info)

    @handle_exceptions
    async def end_session(
        self,
        session_id: str,
        reason: str = "user_ended"
    ) -> None:
        """
        End a session.
        
        Args:
            session_id: Session identifier
            reason: Reason for ending the session
        """
        session_info = await self.get_session(session_id)
        if not session_info:
            return
        
        async with self._get_session_lock(session_id):
            try:
                # Calculate session duration
                duration = (
                    datetime.now(timezone.utc) - session_info.created_at
                ).total_seconds()
                
                # Update session state
                session_info.state = SessionState.TERMINATED
                
                # Store final session state
                await self.session_store.store_session(session_info)
                
                # Cleanup working memory
                await self.working_memory.cleanup_session(session_id)
                
                # Store session in episodic memory for learning
                if self.episodic_memory and session_info.context.user_id:
                    session_memory = {
                        'session_id': session_id,
                        'user_id': session_info.context.user_id,
                        'duration': duration,
                        'interaction_count': session_info.interaction_count,
                        'workflow_count': session_info.workflow_count,
                        'total_processing_time': session_info.total_processing_time,
                        'reason': reason,
                        'context_summary': self._summarize_session_context(session_info.context)
                    }
                    
                    await self.episodic_memory.store(session_memory)
                
                # Remove from active sessions
                self.active_sessions.pop(session_id, None)
                self.session_locks.pop(session_id, None)
                
                if session_info.context.user_id:
                    self.user_sessions[session_info.context.user_id].discard(session_id)
                
                # Emit session ended event
                await self.event_bus.emit(SessionEnded(
                    session_id=session_id,
                    user_id=session_info.context.user_id,
                    duration=duration,
                    interaction_count=session_info.interaction_count,
                    reason=reason
                ))
                
                # Update metrics
                self.metrics.increment("sessions_ended_total")
                self.metrics.record("session_duration_seconds", duration)
                self.metrics.set("active_sessions", len(self.active_sessions))
                
                self.logger.info(f"Ended session: {session_id} (reason: {reason}, duration: {duration:.2f}s)")
                
            except Exception as e:
                self.logger.error(f"Error ending session {session_id}: {str(e)}")
                raise SessionError(f"Failed to end session: {str(e)}", session_id)

    def _summarize_session_context(self, context: SessionContext) -> Dict[str, Any]:
        """Create a summary of session context for memory storage."""
        return {
            'user_id': context.user_id,
            'duration': (context.last_activity - context.created_at).total_seconds(),
            'topics_discussed': list(set([
                interaction.get('data', {}).get('topic', 'unknown')
                for interaction in context.interaction_history
                if isinstance(interaction, dict)
            ])),
            'interaction_count': len(context.interaction_history),
            'active_workflows': len(context.active_workflows),
            'device_type': context.device_info.get('type', 'unknown'),
            'primary_language': context.user_preferences.get('language', 'en'),
            'session_quality': 'high' if len(context.interaction_history) > 5 else 'low'
        }

    async def _expire_session(self, session_id: str) -> None:
        """Expire a session due to timeout."""
        try:
            session_info = self.active_sessions.get(session_id)
            if session_info:
                session_info.state = SessionState.EXPIRED
                
                # Store expired state
                await self.session_store.store_session(session_info)
                
                # Emit expiration event
                await self.event_bus.emit(SessionExpired(
                    session_id=session_id,
                    user_id=session_info.context.user_id,
                    duration=(datetime.now(timezone.utc) - session_info.created_at).total_seconds()
                ))
                
                # Update metrics
                self.metrics.increment("sessions_expired_total")
            
            # Clean up if configured
            if session_info and session_info.config.cleanup_on_expire:
                await self.end_session(session_id, "expired")
            
        except Exception as e:
            self.logger.error(f"Error expiring session {session_id}: {str(e)}")

    @handle_exceptions
    async def pause_session(self, session_id: str) -> None:
        """
        Pause a session.
        
        Args:
            session_id: Session identifier
        """
        session_info = await self.get_session(session_id)
        if not session_info:
            raise SessionError(f"Session {session_id} not found")
        
        async with self._get_session_lock(session_id):
            if session_info.state == SessionState.ACTIVE:
                session_info.state = SessionState.PAUSED
                await self.session_store.store_session(session_info)
                
                await self.event_bus.emit(SessionStateChanged(
                    session_id=session_id,
                    old_state=SessionState.ACTIVE.value,
                    new_state=SessionState.PAUSED.value
                ))

    @handle_exceptions
    async def resume_session(self, session_id: str) -> None:
        """
        Resume a paused session.
        
        Args:
            session_id: Session identifier
        """
        session_info = await self.get_session(session_id)
        if not session_info:
            raise SessionError(f"Session {session_id} not found")
        
        async with self._get_session_lock(session_id):
            if session_info.state == SessionState.PAUSED:
                session_info.state = SessionState.ACTIVE
                await self.session_store.store_session(session_info)
                
                await self.event_bus.emit(SessionStateChanged(
                    session_id=session_id,
                    old_state=SessionState.PAUSED.value,
                    new_state=SessionState.ACTIVE.value
                ))

    def list_user_sessions(self, user_id: str) -> List[str]:
        """
        List all sessions for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            List of session IDs
        """
        return list(self.user_sessions.get(user_id, set()))

    def get_active_sessions(self) -> List[Dict[str, Any]]:
        """Get information about all active sessions."""
        sessions = []
        
        for session_id, session_info in self.active_sessions.items():
            sessions.append({
                'session_id': session_id,
                'user_id': session_info.context.user_id,
                'state': session_info.state.value,
                'created_at': session_info.created_at.isoformat(),
                'last_activity': session_info.last_activity.isoformat(),
                'interaction_count': session_info.interaction_count,
                'workflow_count': session_info.workflow_count,
                'memory_usage_mb': session_info.memory_usage_mb,
                'response_time_avg': session_info.response_time_avg,
                'cluster_node': session_info.cluster_node
            })
        
        return sessions

    def get_session_statistics(self) -> Dict[str, Any]:
        """Get comprehensive session statistics."""
        total_sessions = len(self.active_sessions)
        
        states_count = defaultdict(int)
        total_interactions = 0
        total_workflows = 0
        total_memory = 0.0
        
        for session_info in self.active_sessions.values():
            states_count[session_info.state.value] += 1
            total_interactions += session_info.interaction_count
            total_workflows += session_info.workflow_count
            total_memory += session_info.memory_usage_mb
        
        return {
            'total_active_sessions': total_sessions,
            'sessions_by_state': dict(states_count),
            'total_interactions': total_interactions,
            'total_workflows': total_workflows,
            'total_memory_usage_mb': total_memory,
            'average_memory_per_session_mb': total_memory / max(total_sessions, 1),
            'cleanup_stats': dict(self.cleanup_stats),
            'node_id': self.node_id,
            'clustering_enabled': self.enable_clustering
        }

    async def _cleanup_loop(self) -> None:
        """Background task for session cleanup."""
        while True:
            try:
                current_time = datetime.now(timezone.utc)
                expired_sessions = []
                idle_sessions = []
                
                # Check for expired and idle sessions
                for session_id, session_info in list(self.active_sessions.items()):
                    # Check expiration
                    if session_info.expires_at and current_time > session_info.expires_at:
                        expired_sessions.append(session_id)
                        continue
                    
                    # Check idle timeout
                    idle_time = (current_time - session_info.last_activity).total_seconds()
                    if idle_time > session_info.config.max_idle_time:
                        if session_info.state == SessionState.ACTIVE:
                            session_info.state = SessionState.IDLE
                        elif session_info.state == SessionState.IDLE and idle_time > session_info.config.max_idle_time * 2:
                            idle_sessions.append(session_id)
                
                # Clean up expired sessions
                for session_id in expired_sessions:
                    try:
                        await self._expire_session(session_id)
                        self.cleanup_stats['expired'] += 1
                    except Exception as e:
                        self.logger.error(f"Failed to expire session {session_id}: {str(e)}")
                
                # Clean up idle sessions
                for session_id in idle_sessions:
                    try:
                        await self.end_session(session_id, "idle_timeout")
                        self.cleanup_stats['idle_timeout'] += 1
                    except Exception as e:
                        self.logger.error(f"Failed to clean up idle session {session_id}: {str(e)}")
                
                # Clean up storage
                storage_cleaned = await self.session_store.cleanup_expired_sessions()
                self.cleanup_stats['storage_cleaned'] += storage_cleaned
                
                # Update metrics
                self.metrics.set("active_sessions", len(self.active_sessions))
                
                if expired_sessions or idle_sessions or storage_cleaned:
                    self.logger.info(
                        f"Session cleanup: {len(expired_sessions)} expired, "
                        f"{len(idle_sessions)} idle, {storage_cleaned} from storage"
                    )
                
                await asyncio.sleep(60)  # Run cleanup every minute
                
            except Exception as e:
                self.logger.error(f"Error in cleanup loop: {str(e)}")
                await asyncio.sleep(60)

    async def _heartbeat_loop(self) -> None:
        """Background task for session heartbeat and health monitoring."""
        while True:
            try:
                current_time = datetime.now(timezone.utc)
                
                # Update session health scores
                for session_id, session_info in self.active_sessions.items():
                    try:
                        # Calculate health score based on various factors
                        health_factors = {
                            'activity': min(1.0, 300.0 / max(1.0, (current_time - session_info.last_activity).total_seconds())),
                            'memory': max(0.0, 1.0 - (session_info.memory_usage_mb / session_info.config.memory_limit_mb)),
                            'errors': max(0.0, 1.0 - (session_info.error_count / 10.0)),
                            'performance': min(1.0, 5.0 / max(0.1, session_info.response_time_avg))
                        }
                        
                        session_info.health_score = sum(health_factors.values()) / len(health_factors)
                        
                        # Update cluster if enabled
                        if self.cluster and session_info.cluster_node == self.node_id:
                            await self.cluster.register_node(self.node_id, {
                                'load': len(self.active_sessions),
                                'memory_usage': sum(s.memory_usage_mb for s in self.active_sessions.values()),
                                'health_score': session_info.health_score
                            })
                    
                    except Exception as e:
                        self.logger.warning(f"Error updating health for session {session_id}: {str(e)}")
                
                await asyncio.sleep(30)  # Run heartbeat every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Error in heartbeat loop: {str(e)}")
                await asyncio.sleep(30)

    async def _backup_loop(self) -> None:
        """Background task for session backup."""
        if not self.backup_manager:
            return
        
        while True:
            try:
                backup_interval = self.default_config.auto_save_interval
                
                # Backup active sessions
                for session_id, session_info in list(self.active_sessions.items()):
                    try:
                        if session_info.config.enable_backup:
                            last_backup = session_info.last_backup
                            current_time = datetime.now(timezone.utc)
                            
                            if not last_backup or (current_time - last_backup).total_seconds() > backup_interval:
                                # Create backup
                                backup_data = {
                                    'session_info': asdict(session_info),
                                    'timestamp': current_time.isoformat(),
                                    'checksum': self._calculate_checksum(session_info)
                                }
                                
                                await self.backup_manager.backup_data(
                                    f"session_{session_id}",
                                    backup_data,
                                    metadata={'type': 'session', 'user_id': session_info.context.user_id}
                                )
                                
                                session_info.last_backup = current_time
                    
                    except Exception as e:
                        self.logger.warning(f"Failed to backup session {session_id}: {str(e)}")
                
                await asyncio.sleep(backup_interval)
                
            except Exception as e:
                self.logger.error(f"Error in backup loop: {str(e)}")
                await asyncio.sleep(backup_interval)

    async def _handle_component_health_change(self, event) -> None:
        """Handle component health change events."""
        if not event.healthy:
            # Component is unhealthy, might need to adapt session handling
            self.logger.warning(f"Component {event.component} is unhealthy, monitoring sessions")

    async def _handle_system_shutdown(self, event) -> None:
        """Handle system shutdown by cleaning up sessions."""
        try:
            # Save all active sessions
            save_tasks = []
            for session_info in self.active_sessions.values():
                save_tasks.append(self.session_store.store_session(session_info))
            
            if save_tasks:
                await asyncio.gather(*save_tasks, return_exceptions=True)
            
            self.logger.info(f"Saved {len(save_tasks)} sessions before shutdown")
            
        except Exception as e:
            self.logger.error(f"Error saving sessions during shutdown: {str(e)}")

    async def _handle_user_authentication(self, event) -> None:
        """Handle user authentication events."""
        # Update any guest sessions for this user
        user_id = event.user_id
        for session_id, session_info in list(self.active_sessions.items()):
            if (session_info.context.user_id is None and 
                session_info.config.session_type == SessionType.GUEST):
                # Convert guest session to authenticated
                session_info.context.user_id = user_id
                session_info.config.session_type = SessionType.AUTHENTICATED
                
                # Load user context
                await self._load_user_context(session_info.context)
                
                # Update storage
                await self.session_store.store_session(session_info)
                
                # Update tracking
                self.user_sessions[user_id].add(session_id)

    async def _handle_user_logout(self, event) -> None:
        """Handle user logout events."""
        user_id = event.user_id
        
        # End all sessions for this user
        user_session_ids = list(self.user_sessions.get(user_id, set()))
        for session_id in user_session_ids:
            try:
                await self.end_session(session_id, "user_logout")
            except Exception as e:
                self.logger.error(f"Error ending session {session_id} on logout: {str(e)}")

    async def _health_check_callback(self) -> Dict[str, Any]:
        """Health check callback for the session manager."""
        
