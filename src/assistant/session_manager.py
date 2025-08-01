"""
Unified Session Manager with Memory Integration
Authors: AI Assistant Contributors
Last Modified: 2025-07-20

Enhanced session management system with integrated memory capabilities,
providing conversational context persistence and intelligent memory recall.
"""

import hashlib
import json
import pickle
import threading
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Type, TypeVar, Union

import asyncio

from src.assistant.session_memory_integrator import SessionMemoryIntegrator
from src.core.config.loader import ConfigLoader
from src.core.dependency_injection import Container
from src.core.events.event_bus import EventBus
from src.core.events.event_types import (
    ComponentHealthChanged,
    ErrorOccurred,
    MemoryItemStored,
    MemoryRetrievalRequested,
    MessageProcessed,
    MessageReceived,
    SessionEnded,
    SessionStarted,
    SystemShutdown,
    UserAuthentication,
    UserLogout,
)
from src.core.security.encryption import EncryptionManager
from src.integrations.cache.redis_cache import RedisCache
from src.integrations.storage.database import DatabaseManager
from src.memory.core_memory.base_memory import MemoryItem, MemoryType
from src.memory.operations.retrieval import RetrievalRequest, RetrievalStrategy
from src.observability.logging.config import get_logger
from src.observability.monitoring.metrics import MetricsCollector
from src.observability.monitoring.tracing import TraceManager
from src.utils.health import HealthComponent

# TypeVar for generic typing
T = TypeVar("T")


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

    INTERACTIVE = "interactive"  # Real-time user interaction
    BATCH = "batch"  # Batch processing session
    API = "api"  # API-based session
    BACKGROUND = "background"  # Background processing
    SYSTEM = "system"  # System maintenance session
    GUEST = "guest"  # Anonymous guest session
    AUTHENTICATED = "authenticated"  # Authenticated user session


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
    # Memory integration settings
    enable_memory: bool = True
    memory_retention_policy: str = "adaptive"
    max_memory_items: int = 1000
    memory_consolidation_enabled: bool = True


@dataclass
class SessionContext:
    """Comprehensive session context data with memory integration."""

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
    memory_ids: List[str] = field(default_factory=list)
    important_facts: List[Dict[str, Any]] = field(default_factory=list)

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
    """Runtime session information with memory support."""

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

    # Memory statistics
    memory_item_count: int = 0
    memory_retrieval_count: int = 0
    memory_consolidation_count: int = 0

    def touch(self) -> None:
        """Update last activity time."""
        self.last_activity = datetime.now(timezone.utc)
        self.interaction_count += 1


class SessionError(Exception):
    """Custom exception for session management operations."""

    def __init__(
        self, message: str, session_id: Optional[str] = None, error_code: Optional[str] = None
    ):
        super().__init__(message)
        self.session_id = session_id
        self.error_code = error_code


class SessionStore(ABC):
    """Abstract base class for session storage backends."""

    @abstractmethod
    async def store_session(self, session_info: SessionInfo) -> None:
        """Store or update a session."""
        pass

    @abstractmethod
    async def load_session(self, session_id: str) -> Optional[SessionInfo]:
        """Load a session by ID."""
        pass

    @abstractmethod
    async def delete_session(self, session_id: str) -> None:
        """Delete a session."""
        pass

    @abstractmethod
    async def list_sessions(self, user_id: Optional[str] = None) -> List[str]:
        """List session IDs, optionally filtered by user."""
        pass

    @abstractmethod
    async def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions and return count."""
        pass


class MemorySessionStore(SessionStore):
    """In-memory session store for development and testing."""

    def __init__(self):
        self._sessions: Dict[str, SessionInfo] = {}
        self._lock = asyncio.Lock()

    async def store_session(self, session_info: SessionInfo) -> None:
        async with self._lock:
            self._sessions[session_info.session_id] = session_info

    async def load_session(self, session_id: str) -> Optional[SessionInfo]:
        async with self._lock:
            return self._sessions.get(session_id)

    async def delete_session(self, session_id: str) -> None:
        async with self._lock:
            self._sessions.pop(session_id, None)

    async def list_sessions(self, user_id: Optional[str] = None) -> List[str]:
        async with self._lock:
            if user_id:
                return [
                    sid for sid, info in self._sessions.items() if info.context.user_id == user_id
                ]
            return list(self._sessions.keys())

    async def cleanup_expired_sessions(self) -> int:
        async with self._lock:
            now = datetime.now(timezone.utc)
            expired = [
                sid
                for sid, info in self._sessions.items()
                if info.expires_at and info.expires_at < now
            ]
            for sid in expired:
                del self._sessions[sid]
            return len(expired)


class DatabaseSessionStore(SessionStore):
    """Database-backed session store for production with encryption support."""

    def __init__(self, database: DatabaseManager, encryption: Optional[EncryptionManager] = None):
        self.database = database
        self.encryption = encryption
        self.logger = get_logger(__name__)

    async def store_session(self, session_info: SessionInfo) -> None:
        try:
            session_data = self._serialize_session(session_info)

            # Encrypt if enabled
            if self.encryption and session_info.config.encryption_enabled:
                session_data = await self.encryption.encrypt(session_data)

            await self.database.execute(
                """
                INSERT INTO sessions (session_id, data, user_id, created_at, 
                                    expires_at, last_activity)
                VALUES ($1, $2, $3, $4, $5, $6)
                ON CONFLICT (session_id) 
                DO UPDATE SET data = EXCLUDED.data, 
                            last_activity = EXCLUDED.last_activity
                """,
                session_info.session_id,
                session_data,
                session_info.context.user_id,
                session_info.created_at,
                session_info.expires_at,
                session_info.last_activity,
            )
        except Exception as e:
            self.logger.error(f"Failed to store session: {str(e)}")
            raise SessionError(
                f"Failed to store session: {str(e)}", session_id=session_info.session_id
            )

    async def load_session(self, session_id: str) -> Optional[SessionInfo]:
        try:
            result = await self.database.fetch_one(
                "SELECT data FROM sessions WHERE session_id = $1", session_id
            )

            if result:
                session_data = result["data"]

                # Decrypt if encrypted
                if self.encryption and isinstance(session_data, bytes):
                    session_data = await self.encryption.decrypt(session_data)

                return self._deserialize_session(session_data)

            return None
        except Exception as e:
            self.logger.error(f"Failed to load session: {str(e)}")
            return None

    async def delete_session(self, session_id: str) -> None:
        try:
            await self.database.execute("DELETE FROM sessions WHERE session_id = $1", session_id)
        except Exception as e:
            self.logger.error(f"Failed to delete session: {str(e)}")

    async def list_sessions(self, user_id: Optional[str] = None) -> List[str]:
        try:
            if user_id:
                results = await self.database.fetch_all(
                    "SELECT session_id FROM sessions WHERE user_id = $1", user_id
                )
            else:
                results = await self.database.fetch_all("SELECT session_id FROM sessions")

            return [row["session_id"] for row in results]
        except Exception as e:
            self.logger.error(f"Failed to list sessions: {str(e)}")
            return []

    async def cleanup_expired_sessions(self) -> int:
        try:
            result = await self.database.execute(
                """
                DELETE FROM sessions 
                WHERE expires_at IS NOT NULL AND expires_at < $1
                RETURNING session_id
                """,
                datetime.now(timezone.utc),
            )
            return len(result) if result else 0
        except Exception as e:
            self.logger.error(f"Failed to cleanup sessions: {str(e)}")
            return 0

    def _serialize_session(self, session_info: SessionInfo) -> str:
        """Serialize session to JSON string."""
        return json.dumps(
            {
                "session_id": session_info.session_id,
                "state": session_info.state.value,
                "config": {
                    "session_type": session_info.config.session_type.value,
                    "priority": session_info.config.priority.value,
                    "max_idle_time": session_info.config.max_idle_time,
                    "max_session_time": session_info.config.max_session_time,
                    "enable_memory": session_info.config.enable_memory,
                    "memory_retention_policy": session_info.config.memory_retention_policy,
                },
                "context": {
                    "user_id": session_info.context.user_id,
                    "conversation_history": session_info.context.conversation_history,
                    "current_topic": session_info.context.current_topic,
                    "memory_ids": session_info.context.memory_ids,
                    "important_facts": session_info.context.important_facts,
                    "custom_data": session_info.context.custom_data,
                },
                "stats": {
                    "interaction_count": session_info.interaction_count,
                    "memory_item_count": session_info.memory_item_count,
                    "memory_retrieval_count": session_info.memory_retrieval_count,
                },
                "version": session_info.version,
            }
        )

    def _deserialize_session(self, session_data: str) -> SessionInfo:
        """Deserialize session from JSON string."""
        data = json.loads(session_data)

        # Create config
        config = SessionConfiguration()
        config.session_type = SessionType(data["config"]["session_type"])
        config.priority = SessionPriority(data["config"]["priority"])
        config.max_idle_time = data["config"]["max_idle_time"]
        config.max_session_time = data["config"]["max_session_time"]
        config.enable_memory = data["config"].get("enable_memory", True)
        config.memory_retention_policy = data["config"].get("memory_retention_policy", "adaptive")

        # Create context
        context = SessionContext(data["session_id"])
        context.user_id = data["context"]["user_id"]
        context.conversation_history = data["context"]["conversation_history"]
        context.current_topic = data["context"]["current_topic"]
        context.memory_ids = data["context"].get("memory_ids", [])
        context.important_facts = data["context"].get("important_facts", [])
        context.custom_data = data["context"]["custom_data"]

        # Create session info
        session_info = SessionInfo(data["session_id"])
        session_info.state = SessionState(data["state"])
        session_info.config = config
        session_info.context = context
        session_info.interaction_count = data["stats"]["interaction_count"]
        session_info.memory_item_count = data["stats"].get("memory_item_count", 0)
        session_info.memory_retrieval_count = data["stats"].get("memory_retrieval_count", 0)
        session_info.version = data["version"]

        return session_info


class SessionCluster:
    """Manages session clustering across multiple nodes."""

    def __init__(self, node_id: str, redis_cache: Optional[RedisCache] = None):
        self.node_id = node_id
        self.redis_cache = redis_cache
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.session_assignments: Dict[str, str] = {}
        self.logger = get_logger(__name__)

    async def register_node(self, node_id: str, node_info: Dict[str, Any]) -> None:
        """Register a cluster node."""
        self.nodes[node_id] = {**node_info, "last_heartbeat": datetime.now(timezone.utc)}

        if self.redis_cache:
            await self.redis_cache.set(
                f"cluster:node:{node_id}", json.dumps(self.nodes[node_id]), ttl=60
            )

    async def assign_session(self, session_id: str) -> str:
        """Assign a session to a node using consistent hashing."""
        if not self.nodes:
            return self.node_id

        # Simple consistent hashing
        hash_value = int(hashlib.md5(session_id.encode()).hexdigest(), 16)
        nodes = sorted(self.nodes.keys())
        node_index = hash_value % len(nodes)
        assigned_node = nodes[node_index]

        self.session_assignments[session_id] = assigned_node

        if self.redis_cache:
            await self.redis_cache.set(f"cluster:session:{session_id}", assigned_node, ttl=3600)

        return assigned_node

    async def migrate_session(self, session_id: str, target_node: str) -> bool:
        """Migrate a session to another node."""
        try:
            current_node = self.session_assignments.get(session_id)
            if current_node == target_node:
                return True

            # Implement actual session migration logic
            # 1. Get session data
            session = await self.get_session(session_id)
            if not session:
                return False

            # 2. Serialize session data
            session_data = {
                "session_id": session.session_id,
                "user_id": session.context.user_id,
                "session_type": session.session_type.value,
                "state": session.state.value,
                "context": session.context.__dict__,
                "metadata": session.metadata,
                "created_at": session.created_at.isoformat(),
                "last_activity": (
                    session.last_activity.isoformat() if session.last_activity else None
                ),
            }

            # 3. Update assignment (simplified - in real system would transfer to target node)
            self.session_assignments[session_id] = target_node

            # 4. Update in cache
            if self.redis_cache:
                await self.redis_cache.set(f"cluster:session:{session_id}", target_node, ttl=3600)

                # Store migration metadata
                await self.redis_cache.set(
                    f"migration:session:{session_id}",
                    json.dumps(
                        {
                            "from_node": current_node,
                            "to_node": target_node,
                            "migrated_at": datetime.now(timezone.utc).isoformat(),
                        }
                    ),
                    ttl=86400,  # Keep migration history for 24 hours
                )

            return True

        except Exception as e:
            self.logger.error(f"Failed to migrate session: {str(e)}")
            return False

    async def handle_node_failure(self, failed_node: str) -> None:
        """Handle node failure by reassigning sessions."""
        if failed_node not in self.nodes:
            return

        # Remove failed node
        del self.nodes[failed_node]

        # Reassign sessions from failed node
        sessions_to_reassign = [
            sid for sid, node in self.session_assignments.items() if node == failed_node
        ]

        for session_id in sessions_to_reassign:
            await self.assign_session(session_id)

        self.logger.info(
            f"Reassigned {len(sessions_to_reassign)} sessions from failed node {failed_node}"
        )


class EnhancedSessionManager:
    """
    Enhanced session manager with memory integration, clustering support,
    and comprehensive session lifecycle management.
    """

    def __init__(self, container: Container):
        """Initialize the enhanced session manager."""
        self.container = container
        self.logger = get_logger(__name__)

        # Core dependencies
        self.config_loader = container.get(ConfigLoader)
        self.event_bus = container.get(EventBus)

        # Configuration
        self.config = self.config_loader.get("session_manager", {})
        self.enable_clustering = self.config.get("enable_clustering", False)
        self.enable_memory = self.config.get("enable_memory", True)
        self.node_id = self.config.get("node_id", str(uuid.uuid4()))

        # Session storage
        self._setup_session_store()

        # Memory integration
        if self.enable_memory:
            try:
                self.memory_integrator = container.get(SessionMemoryIntegrator)
                self.logger.info("Memory integration enabled for sessions")
            except Exception as e:
                self.logger.warning(f"Memory integration not available: {str(e)}")
                self.memory_integrator = None
        else:
            self.memory_integrator = None

        # Active sessions cache
        self._sessions: Dict[str, SessionInfo] = {}
        self._session_locks: Dict[str, asyncio.Lock] = {}
        self._global_lock = asyncio.Lock()

        # Optional components
        try:
            self.database = container.get(DatabaseManager)
        except Exception:
            self.database = None

        try:
            self.redis_cache = container.get(RedisCache)
        except Exception:
            self.redis_cache = None

        try:
            self.encryption = container.get(EncryptionManager)
        except Exception:
            self.encryption = None

        # Clustering support
        if self.enable_clustering and self.redis_cache:
            self.cluster = SessionCluster(self.node_id, self.redis_cache)
        else:
            self.cluster = None

        # Health component
        self.health_component = HealthComponent(
            name="session_manager", check_callback=self._health_check_callback
        )

        # Monitoring
        self._setup_monitoring()

        # Background tasks
        self._cleanup_task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._backup_task: Optional[asyncio.Task] = None

        self.logger.info(f"Session manager initialized with node ID: {self.node_id}")

    def _setup_session_store(self) -> None:
        """Set up session storage backend."""
        storage_type = self.config.get("storage_type", "memory")

        if storage_type == "database" and self.container.has(DatabaseManager):
            self.session_store = DatabaseSessionStore(
                self.container.get(DatabaseManager),
                (
                    self.container.get(EncryptionManager)
                    if self.container.has(EncryptionManager)
                    else None
                ),
            )
            self.logger.info("Using database session store")
        else:
            self.session_store = MemorySessionStore()
            self.logger.info("Using memory session store")

    def _setup_monitoring(self) -> None:
        """Set up monitoring and metrics."""
        try:
            self.metrics = self.container.get(MetricsCollector)

            # Register metrics
            self.metrics.register_counter("session_created_total", "Total sessions created")
            self.metrics.register_counter("session_ended_total", "Total sessions ended")
            self.metrics.register_gauge("session_active_count", "Number of active sessions")
            self.metrics.register_histogram(
                "session_duration_seconds", "Session duration in seconds"
            )
            self.metrics.register_counter(
                "session_memory_operations_total", "Total memory operations in sessions"
            )

        except Exception:
            self.logger.warning("Metrics collector not available")
            self.metrics = None

        try:
            self.tracer = self.container.get(TraceManager)
        except Exception:
            self.tracer = None

    async def initialize(self) -> None:
        """Initialize the session manager."""
        self.logger.info("Initializing session manager...")

        # Register event handlers
        await self._register_event_handlers()

        # Recover persisted sessions
        await self._recover_sessions()

        # Start background tasks
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

        if self.config.get("enable_backup", True):
            self._backup_task = asyncio.create_task(self._backup_loop())

        # Register with cluster if enabled
        if self.cluster:
            await self.cluster.register_node(
                self.node_id,
                {
                    "type": "session_manager",
                    "capacity": self.config.get("max_sessions", 10000),
                    "current_load": len(self._sessions),
                },
            )

        # Register health component
        await self.event_bus.emit(
            ComponentHealthChanged(
                component_name="session_manager",
                health_status="healthy",
                details={"active_sessions": len(self._sessions)},
            )
        )

        self.logger.info("Session manager initialized successfully")

    async def _register_event_handlers(self) -> None:
        """Register event handlers."""
        self.event_bus.subscribe(ComponentHealthChanged, self._handle_component_health_change)
        self.event_bus.subscribe(SystemShutdown, self._handle_system_shutdown)
        self.event_bus.subscribe(UserAuthentication, self._handle_user_authentication)
        self.event_bus.subscribe(UserLogout, self._handle_user_logout)
        self.event_bus.subscribe(MessageReceived, self._handle_message_received)

    async def _recover_sessions(self) -> None:
        """Recover sessions from persistent storage."""
        try:
            session_ids = await self.session_store.list_sessions()
            recovered = 0

            for session_id in session_ids:
                session_info = await self.session_store.load_session(session_id)
                if session_info and session_info.state == SessionState.ACTIVE:
                    self._sessions[session_id] = session_info
                    recovered += 1

            if recovered > 0:
                self.logger.info(f"Recovered {recovered} active sessions")

        except Exception as e:
            self.logger.error(f"Failed to recover sessions: {str(e)}")

    async def create_session(
        self,
        user_id: Optional[str] = None,
        session_type: SessionType = SessionType.INTERACTIVE,
        config: Optional[SessionConfiguration] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Create a new session with optional memory support."""
        # Generate session ID
        session_id = str(uuid.uuid4())

        # Use provided config or create default
        if not config:
            config = SessionConfiguration(session_type=session_type)

        # Create context
        context = SessionContext(session_id=session_id, user_id=user_id)

        # Add metadata
        if metadata:
            context.metadata.update(metadata)

        # Load user profile and preferences if available
        if user_id:
            await self._load_user_context(context)

        # Create session info
        session_info = SessionInfo(
            session_id=session_id, state=SessionState.ACTIVE, config=config, context=context
        )

        # Set expiration
        if config.max_session_time > 0:
            session_info.expires_at = datetime.now(timezone.utc) + timedelta(
                seconds=config.max_session_time
            )

        # Store session
        async with self._global_lock:
            self._sessions[session_id] = session_info
            self._session_locks[session_id] = asyncio.Lock()

        # Persist to storage
        await self.session_store.store_session(session_info)

        # Initialize memory if enabled
        if self.memory_integrator and config.enable_memory:
            try:
                # Initialize memory context for session
                await self.memory_integrator._handle_session_started(
                    SessionStarted(session_id=session_id, user_id=user_id)
                )

                # Retrieve relevant past memories if user is authenticated
                if user_id:
                    memories = await self.memory_integrator.retrieve_session_memories(
                        session_id=session_id, limit=5
                    )

                    # Store memory references in context
                    for memory in memories:
                        context.memory_ids.append(memory["memory_id"])

            except Exception as e:
                self.logger.error(f"Failed to initialize memory for session: {str(e)}")

        # Emit event
        await self.event_bus.emit(
            SessionStarted(
                session_id=session_id,
                user_id=user_id,
                session_type=session_type.value,
                metadata=metadata,
            )
        )

        # Update metrics
        if self.metrics:
            self.metrics.increment("session_created_total")
            self.metrics.gauge("session_active_count", len(self._sessions))

        self.logger.info(f"Created session {session_id} for user {user_id}")

        return session_id

    async def _load_user_context(self, context: SessionContext) -> None:
        """Load user context and preferences."""
        try:
            # Load from user service or database
            if context.user_id:
                # Try to get user preferences from database or cache
                user_data = None

                # First try cache
                if self.redis_cache:
                    cache_key = f"user:preferences:{context.user_id}"
                    try:
                        cached_data = await self.redis_cache.get(cache_key)
                        if cached_data:
                            user_data = json.loads(cached_data)
                    except Exception:
                        pass

                # If not in cache, load from database
                if not user_data and self.database:
                    try:
                        user_record = await self.database.fetch_one(
                            "SELECT preferences, language, timezone FROM users WHERE user_id = ?",
                            (context.user_id,),
                        )
                        if user_record:
                            user_data = {
                                "language": user_record.get("language", "en"),
                                "timezone": user_record.get("timezone", "UTC"),
                                "preferences": json.loads(user_record.get("preferences", "{}")),
                            }

                            # Cache for future use
                            if self.redis_cache:
                                await self.redis_cache.set(
                                    cache_key, json.dumps(user_data), ttl=3600
                                )

                    except Exception as e:
                        self.logger.warning(f"Failed to load user data from database: {e}")

                # Set preferences from loaded data or defaults
                if user_data:
                    context.user_preferences = {
                        "language": user_data.get("language", "en"),
                        "timezone": user_data.get("timezone", "UTC"),
                        **user_data.get("preferences", {}),
                    }
                else:
                    # Fallback to defaults
                    context.user_preferences = {
                        "language": "en",
                        "timezone": "UTC",
                        "theme": "light",
                        "notifications": True,
                    }
            else:
                # Guest user defaults
                context.user_preferences = {
                    "language": "en",
                    "timezone": "UTC",
                    "theme": "light",
                    "notifications": False,
                }
        except Exception as e:
            self.logger.error(f"Failed to load user context: {str(e)}")

    def _get_session_lock(self, session_id: str) -> asyncio.Lock:
        """Get or create a lock for a session."""
        if session_id not in self._session_locks:
            self._session_locks[session_id] = asyncio.Lock()
        return self._session_locks[session_id]

    async def get_session(self, session_id: str) -> Optional[SessionInfo]:
        """Get session by ID with memory context."""
        # Check cache first
        if session_id in self._sessions:
            session = self._sessions[session_id]
            await self._update_last_activity(session)

            # Enhance with memory context if available
            if self.memory_integrator and session.config.enable_memory:
                try:
                    memory_context = await self.memory_integrator.get_session_context(session_id)
                    session.context.working_memory_data = memory_context.get("context", {})
                except Exception as e:
                    self.logger.error(f"Failed to get memory context: {str(e)}")

            return session

        # Try loading from storage
        session = await self.session_store.load_session(session_id)
        if session:
            # Cache it
            async with self._global_lock:
                self._sessions[session_id] = session
                if session_id not in self._session_locks:
                    self._session_locks[session_id] = asyncio.Lock()

            await self._update_last_activity(session)
            return session

        return None

    async def _update_last_activity(self, session_info: SessionInfo) -> None:
        """Update session last activity time."""
        session_info.last_activity = datetime.now(timezone.utc)
        session_info.version += 1

    async def update_session_context(
        self, session_id: str, updates: Dict[str, Any], merge: bool = True
    ) -> bool:
        """Update session context with optional memory storage."""
        session = await self.get_session(session_id)
        if not session:
            return False

        lock = self._get_session_lock(session_id)
        async with lock:
            if merge:
                # Merge updates into existing context
                for key, value in updates.items():
                    if hasattr(session.context, key):
                        if isinstance(getattr(session.context, key), dict):
                            getattr(session.context, key).update(value)
                        elif isinstance(getattr(session.context, key), list):
                            getattr(session.context, key).extend(value)
                        else:
                            setattr(session.context, key, value)
                    else:
                        session.context.custom_data[key] = value
            else:
                # Replace context values
                for key, value in updates.items():
                    if hasattr(session.context, key):
                        setattr(session.context, key, value)
                    else:
                        session.context.custom_data[key] = value

            # Store important facts in memory if identified
            if self.memory_integrator and session.config.enable_memory:
                important_facts = updates.get("important_facts", [])
                for fact in important_facts:
                    if isinstance(fact, dict):
                        memory_id = await self.memory_integrator.store_session_fact(
                            session_id=session_id,
                            user_id=session.context.user_id,
                            fact=fact.get("content", ""),
                            importance=fact.get("importance", 0.7),
                            tags=set(fact.get("tags", [])),
                        )
                        if memory_id:
                            session.context.memory_ids.append(memory_id)
                            session.context.important_facts.append(fact)

            # Update version and checksum
            session.version += 1
            session.checksum = self._calculate_checksum(session)

            # Persist changes
            await self.session_store.store_session(session)

            return True

    def _calculate_checksum(self, session_info: SessionInfo) -> str:
        """Calculate session checksum for integrity verification."""
        data = f"{session_info.session_id}{session_info.version}{session_info.state.value}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    async def add_interaction(
        self, session_id: str, interaction_type: str, data: Dict[str, Any]
    ) -> bool:
        """Add an interaction to session history with memory storage."""
        session = await self.get_session(session_id)
        if not session:
            return False

        lock = self._get_session_lock(session_id)
        async with lock:
            interaction = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "type": interaction_type,
                "data": data,
            }

            session.context.interaction_history.append(interaction)
            session.touch()

            # Store in memory if it's a message interaction
            if self.memory_integrator and interaction_type in [
                "user_message",
                "assistant_response",
            ]:
                await self.memory_integrator._handle_message_processed(
                    MessageProcessed(
                        session_id=session_id,
                        user_id=session.context.user_id,
                        message=data.get("message", ""),
                        response=data.get("response", ""),
                        metadata=data,
                    )
                )
                session.memory_item_count += 1

            # Keep interaction history size manageable
            max_history = session.config.context_window_size // 10
            if len(session.context.interaction_history) > max_history:
                session.context.interaction_history = session.context.interaction_history[
                    -max_history:
                ]

            await self.session_store.store_session(session)

            return True

    async def add_workflow(
        self, session_id: str, workflow_id: str, workflow_data: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Add a workflow to session tracking."""
        session = await self.get_session(session_id)
        if not session:
            return False

        lock = self._get_session_lock(session_id)
        async with lock:
            session.context.active_workflows.add(workflow_id)
            session.workflow_count += 1

            if workflow_data:
                session.context.custom_data[f"workflow_{workflow_id}"] = workflow_data

            await self.session_store.store_session(session)

            return True

    async def remove_workflow(self, session_id: str, workflow_id: str) -> bool:
        """Remove a workflow from session tracking."""
        session = await self.get_session(session_id)
        if not session:
            return False

        lock = self._get_session_lock(session_id)
        async with lock:
            session.context.active_workflows.discard(workflow_id)
            session.context.custom_data.pop(f"workflow_{workflow_id}", None)

            await self.session_store.store_session(session)

            return True

    async def process_message(
        self,
        session_id: str,
        message: str,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Process a message with memory-enhanced context."""
        # Ensure session exists
        session = await self.get_session(session_id)
        if not session:
            session_id = await self.create_session(user_id=user_id)
            session = await self.get_session(session_id)

        session.touch()
        session.message_count += 1

        # Get memory context if available
        memory_context = {}
        if self.memory_integrator and session.config.enable_memory:
            try:
                memory_context = await self.memory_integrator.get_session_context(session_id)
                session.memory_retrieval_count += 1
            except Exception as e:
                self.logger.error(f"Failed to get memory context: {str(e)}")

        # Process message with core engine
        try:
            if self.core_engine:
                # Create processing context from session
                processing_context = ProcessingContext(
                    user_id=session.context.user_id,
                    session_id=session_id,
                    conversation_history=session.context.conversation_history[
                        -10:
                    ],  # Last 10 messages
                    user_preferences=session.context.user_preferences,
                    memory_context=memory_context,
                )

                # Create multimodal input
                multimodal_input = MultimodalInput(text=message, context=processing_context)

                # Process through core engine
                result = await self.core_engine.process_multimodal_input(
                    input_data=multimodal_input, session_context=session.context
                )

                response = {
                    "text": result.get("response", "I'm processing your message..."),
                    "session_id": session_id,
                    "context_used": bool(memory_context),
                    "memory_context_size": len(json.dumps(memory_context)),
                    "confidence": result.get("confidence", 0.8),
                    "processing_time": result.get("processing_time", 0.0),
                    "session_stats": {
                        "interaction_count": session.interaction_count,
                        "memory_item_count": session.memory_item_count,
                    },
                }
            else:
                # Fallback if core engine not available
                response = {
                    "text": f"Received your message: {message[:100]}{'...' if len(message) > 100 else ''}",
                    "session_id": session_id,
                    "context_used": bool(memory_context),
                    "memory_context_size": len(json.dumps(memory_context)),
                    "confidence": 0.5,
                    "processing_time": 0.0,
                    "session_stats": {
                        "interaction_count": session.interaction_count,
                        "memory_item_count": session.memory_item_count,
                    },
                }

        except Exception as e:
            self.logger.error(f"Failed to process message with core engine: {e}")
            # Fallback response
            response = {
                "text": "I'm having trouble processing your message right now. Please try again.",
                "session_id": session_id,
                "context_used": False,
                "memory_context_size": 0,
                "confidence": 0.0,
                "processing_time": 0.0,
                "error": str(e),
                "session_stats": {
                    "interaction_count": session.interaction_count,
                    "memory_item_count": session.memory_item_count,
                },
            }

        # Store interaction
        await self.add_interaction(
            session_id=session_id,
            interaction_type="conversation",
            data={
                "message": message,
                "response": response["text"],
                "memory_context_used": bool(memory_context),
            },
        )

        return response

    async def store_session_fact(
        self, session_id: str, fact: str, importance: float = 0.7, tags: Optional[Set[str]] = None
    ) -> str:
        """Store an important fact learned in the session."""
        session = await self.get_session(session_id)
        if not session or not self.memory_integrator:
            return ""

        memory_id = await self.memory_integrator.store_session_fact(
            session_id=session_id,
            user_id=session.context.user_id,
            fact=fact,
            importance=importance,
            tags=tags,
        )

        if memory_id:
            # Update session context
            lock = self._get_session_lock(session_id)
            async with lock:
                session.context.memory_ids.append(memory_id)
                session.context.important_facts.append(
                    {
                        "memory_id": memory_id,
                        "fact": fact,
                        "importance": importance,
                        "tags": list(tags or set()),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                )
                session.memory_item_count += 1

                await self.session_store.store_session(session)

        return memory_id

    async def get_session_memories(
        self, session_id: str, query: Optional[str] = None, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get memories for a specific session."""
        if not self.memory_integrator:
            return []

        session = await self.get_session(session_id)
        if session:
            session.memory_retrieval_count += 1

        return await self.memory_integrator.retrieve_session_memories(
            session_id=session_id, query=query, limit=limit
        )

    async def end_session(self, session_id: str, reason: str = "user_request") -> bool:
        """End a session and consolidate memories."""
        session = await self.get_session(session_id)
        if not session:
            return False

        lock = self._get_session_lock(session_id)
        async with lock:
            # Update session state
            session.state = SessionState.TERMINATED

            # Calculate session duration
            duration = (datetime.now(timezone.utc) - session.created_at).total_seconds()

            # Create session summary
            summary = self._summarize_session_context(session.context)

            # Consolidate memories if enabled
            if self.memory_integrator and session.config.enable_memory:
                try:
                    await self.memory_integrator._handle_session_ended(
                        SessionEnded(
                            session_id=session_id,
                            user_id=session.context.user_id,
                            duration=duration,
                            reason=reason,
                            summary=summary,
                        )
                    )
                    session.memory_consolidation_count += 1
                except Exception as e:
                    self.logger.error(f"Failed to consolidate session memories: {str(e)}")

            # Remove from active sessions
            del self._sessions[session_id]
            del self._session_locks[session_id]

            # Persist final state
            await self.session_store.store_session(session)

            # Clean up if configured
            if session.config.cleanup_on_expire:
                await self.session_store.delete_session(session_id)

            # Emit event
            await self.event_bus.emit(
                SessionEnded(
                    session_id=session_id,
                    user_id=session.context.user_id,
                    duration=duration,
                    reason=reason,
                    summary=summary,
                )
            )

            # Update metrics
            if self.metrics:
                self.metrics.increment("session_ended_total")
                self.metrics.gauge("session_active_count", len(self._sessions))
                self.metrics.record("session_duration_seconds", duration)
                if session.memory_item_count > 0:
                    self.metrics.increment(
                        "session_memory_operations_total", value=session.memory_item_count
                    )

            self.logger.info(f"Ended session {session_id} after {duration:.1f} seconds")

            return True

    def _summarize_session_context(self, context: SessionContext) -> Dict[str, Any]:
        """Create a summary of session context."""
        return {
            "user_id": context.user_id,
            "topics_discussed": list(set(context.conversation_flow)),
            "interaction_count": len(context.interaction_history),
            "workflows_executed": len(context.active_workflows),
            "memory_items_created": len(context.memory_ids),
            "important_facts": len(context.important_facts),
            "custom_data_keys": list(context.custom_data.keys()),
        }

    async def _expire_session(self, session_id: str) -> None:
        """Expire a session due to timeout."""
        await self.end_session(session_id, reason="timeout")

    async def pause_session(self, session_id: str) -> None:
        """Pause a session."""
        session = await self.get_session(session_id)
        if session and session.state == SessionState.ACTIVE:
            lock = self._get_session_lock(session_id)
            async with lock:
                session.state = SessionState.PAUSED
                await self.session_store.store_session(session)

    async def resume_session(self, session_id: str) -> None:
        """Resume a paused session."""
        session = await self.get_session(session_id)
        if session and session.state == SessionState.PAUSED:
            lock = self._get_session_lock(session_id)
            async with lock:
                session.state = SessionState.ACTIVE
                session.last_activity = datetime.now(timezone.utc)
                await self.session_store.store_session(session)

    def list_user_sessions(self, user_id: str) -> List[str]:
        """List all active sessions for a user."""
        return [
            sid
            for sid, session in self._sessions.items()
            if session.context.user_id == user_id and session.state == SessionState.ACTIVE
        ]

    def get_active_sessions(self) -> List[Dict[str, Any]]:
        """Get information about all active sessions."""
        sessions = []
        for session in self._sessions.values():
            if session.state == SessionState.ACTIVE:
                sessions.append(
                    {
                        "session_id": session.session_id,
                        "user_id": session.context.user_id,
                        "type": session.config.session_type.value,
                        "duration": (
                            datetime.now(timezone.utc) - session.created_at
                        ).total_seconds(),
                        "interaction_count": session.interaction_count,
                        "memory_item_count": session.memory_item_count,
                        "last_activity": session.last_activity.isoformat(),
                    }
                )
        return sessions

    def get_session_statistics(self) -> Dict[str, Any]:
        """Get comprehensive session statistics."""
        now = datetime.now(timezone.utc)
        active_sessions = [s for s in self._sessions.values() if s.state == SessionState.ACTIVE]

        total_memory_items = sum(s.memory_item_count for s in self._sessions.values())
        total_memory_retrievals = sum(s.memory_retrieval_count for s in self._sessions.values())

        return {
            "total_active": len(active_sessions),
            "by_type": {
                session_type.value: len(
                    [s for s in active_sessions if s.config.session_type == session_type]
                )
                for session_type in SessionType
            },
            "average_duration": (
                sum((now - s.created_at).total_seconds() for s in active_sessions)
                / len(active_sessions)
                if active_sessions
                else 0
            ),
            "total_interactions": sum(s.interaction_count for s in active_sessions),
            "memory_stats": {
                "total_items": total_memory_items,
                "total_retrievals": total_memory_retrievals,
                "sessions_with_memory": len(
                    [s for s in active_sessions if s.memory_item_count > 0]
                ),
            },
            "node_id": self.node_id,
            "clustering_enabled": self.enable_clustering,
            "memory_enabled": self.enable_memory,
        }

    async def _cleanup_loop(self) -> None:
        """Background task to clean up expired sessions."""
        while True:
            try:
                # Wait for cleanup interval
                await asyncio.sleep(self.config.get("cleanup_interval", 300))

                # Find expired sessions
                now = datetime.now(timezone.utc)
                expired = []

                for session_id, session in list(self._sessions.items()):
                    # Check for timeout
                    idle_time = (now - session.last_activity).total_seconds()
                    if idle_time > session.config.max_idle_time:
                        expired.append((session_id, "idle_timeout"))
                    # Check for max session time
                    elif session.expires_at and session.expires_at < now:
                        expired.append((session_id, "max_time_exceeded"))

                # End expired sessions
                for session_id, reason in expired:
                    await self._expire_session(session_id)

                # Clean up from storage
                cleaned = await self.session_store.cleanup_expired_sessions()

                if expired or cleaned > 0:
                    self.logger.info(
                        f"Cleaned up {len(expired)} active and {cleaned} stored sessions"
                    )

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in cleanup loop: {str(e)}")

    async def _heartbeat_loop(self) -> None:
        """Background task for cluster heartbeat."""
        if not self.cluster:
            return

        while True:
            try:
                await asyncio.sleep(30)  # Heartbeat every 30 seconds

                # Update cluster with current state
                await self.cluster.register_node(
                    self.node_id,
                    {
                        "type": "session_manager",
                        "capacity": self.config.get("max_sessions", 10000),
                        "current_load": len(self._sessions),
                        "memory_enabled": self.enable_memory,
                    },
                )

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in heartbeat loop: {str(e)}")

    async def _backup_loop(self) -> None:
        """Background task for session backup."""
        while True:
            try:
                # Wait for backup interval
                await asyncio.sleep(self.config.get("backup_interval", 3600))

                # Backup active sessions
                backup_count = 0
                for session in self._sessions.values():
                    if session.config.enable_backup:
                        await self.session_store.store_session(session)
                        session.last_backup = datetime.now(timezone.utc)
                        backup_count += 1

                if backup_count > 0:
                    self.logger.info(f"Backed up {backup_count} sessions")

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in backup loop: {str(e)}")

    async def _handle_component_health_change(self, event) -> None:
        """Handle component health changes."""
        # If memory integrator fails, disable memory features
        if event.component_name == "memory_integrator" and event.health_status != "healthy":
            self.logger.warning("Memory integrator unhealthy, disabling memory features")
            self.memory_integrator = None

    async def _handle_system_shutdown(self, event) -> None:
        """Handle system shutdown."""
        self.logger.info("System shutdown initiated, ending all sessions...")

        # End all active sessions
        for session_id in list(self._sessions.keys()):
            await self.end_session(session_id, reason="system_shutdown")

    async def _handle_user_authentication(self, event) -> None:
        """Handle user authentication by creating or updating session."""
        user_id = event.user_id

        # Check if user already has an active session
        existing_sessions = self.list_user_sessions(user_id)

        if not existing_sessions:
            # Create new authenticated session
            await self.create_session(
                user_id=user_id,
                session_type=SessionType.AUTHENTICATED,
                metadata={"auth_method": event.auth_method},
            )

    async def _handle_user_logout(self, event) -> None:
        """Handle user logout by ending their sessions."""
        user_id = event.user_id

        # End all user sessions
        for session_id in self.list_user_sessions(user_id):
            await self.end_session(session_id, reason="user_logout")

    async def _handle_message_received(self, event: MessageReceived) -> None:
        """Handle incoming message by ensuring session exists."""
        session_id = event.session_id
        user_id = event.user_id

        # Create or get session
        if session_id not in self._sessions:
            await self.create_session(session_id=session_id, user_id=user_id)
        else:
            # Update last activity
            session = self._sessions.get(session_id)
            if session:
                session.touch()

    async def _health_check_callback(self) -> Dict[str, Any]:
        """Health check callback."""
        return {
            "status": "healthy",
            "active_sessions": len(self._sessions),
            "session_store_type": type(self.session_store).__name__,
            "memory_enabled": self.enable_memory and self.memory_integrator is not None,
            "clustering_enabled": self.enable_clustering,
            "node_id": self.node_id,
        }

    async def shutdown(self) -> None:
        """Shutdown the session manager."""
        self.logger.info("Shutting down session manager...")

        # Cancel background tasks
        for task in [self._cleanup_task, self._heartbeat_task, self._backup_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Save all active sessions
        for session in self._sessions.values():
            try:
                await self.session_store.store_session(session)
            except Exception as e:
                self.logger.error(f"Failed to save session {session.session_id}: {str(e)}")

        self.logger.info("Session manager shutdown complete")
