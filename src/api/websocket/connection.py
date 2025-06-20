"""
Advanced WebSocket Connection Management System
Author: Drmusab
Last Modified: 2025-05-26 16:30:00 UTC

This module provides comprehensive WebSocket connection management for the AI assistant,
enabling real-time bidirectional communication, authentication, message routing,
broadcasting, and seamless integration with all core system components.
"""

from pathlib import Path
from typing import Optional, Dict, Any, List, Set, Callable, Type, Union, AsyncGenerator, TypeVar
import asyncio
import threading
import time
import json
import uuid
import weakref
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
from contextlib import asynccontextmanager
import hashlib
import logging
from collections import defaultdict, deque
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
import ssl
import websockets
from websockets.server import WebSocketServerProtocol
from websockets.exceptions import ConnectionClosed, WebSocketException

# Core imports
from src.core.config.loader import ConfigLoader
from src.core.events.event_bus import EventBus
from src.core.events.event_types import (
    ConnectionEstablished, ConnectionClosed as ConnectionClosedEvent, ConnectionFailed,
    MessageReceived, MessageSent, MessageBroadcast, UserAuthenticated, UserAuthorized,
    SessionStarted, SessionEnded, WorkflowStarted, WorkflowCompleted, WorkflowFailed,
    ProcessingStarted, ProcessingCompleted, ErrorOccurred, SystemStateChanged,
    ComponentHealthChanged, RealTimeNotification, PresenceUpdated
)
from src.core.error_handling import ErrorHandler, handle_exceptions
from src.core.dependency_injection import Container
from src.core.health_check import HealthCheck
from src.core.security.authentication import AuthenticationManager
from src.core.security.authorization import AuthorizationManager
from src.core.security.encryption import EncryptionManager

# Assistant components
from src.assistant.core_engine import (
    EnhancedCoreEngine, MultimodalInput, ProcessingContext, ProcessingResult,
    EngineState, ProcessingMode, ModalityType, PriorityLevel
)
from src.assistant.component_manager import ComponentManager
from src.assistant.workflow_orchestrator import WorkflowOrchestrator, WorkflowPriority
from src.assistant.session_manager import SessionManager
from src.assistant.interaction_handler import (
    InteractionHandler, InteractionContext, UserMessage, AssistantResponse,
    InteractionMode, InteractionPriority, InputModality, OutputModality
)
from src.assistant.plugin_manager import PluginManager

# Memory and storage
from src.memory.core_memory.memory_manager import MemoryManager
from src.memory.operations.context_manager import ContextManager

# Integrations
from src.integrations.cache.redis_cache import RedisCache

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


class ConnectionState(Enum):
    """WebSocket connection states."""
    CONNECTING = "connecting"
    AUTHENTICATING = "authenticating"
    CONNECTED = "connected"
    AUTHENTICATED = "authenticated"
    ACTIVE = "active"
    IDLE = "idle"
    SUSPENDED = "suspended"
    DISCONNECTING = "disconnecting"
    DISCONNECTED = "disconnected"
    ERROR = "error"


class MessageType(Enum):
    """Types of WebSocket messages."""
    # Authentication
    AUTH_REQUEST = "auth_request"
    AUTH_RESPONSE = "auth_response"
    AUTH_TOKEN_REFRESH = "auth_token_refresh"
    
    # Session management
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    SESSION_STATUS = "session_status"
    
    # Interactions
    USER_MESSAGE = "user_message"
    ASSISTANT_RESPONSE = "assistant_response"
    INTERACTION_START = "interaction_start"
    INTERACTION_END = "interaction_end"
    
    # Workflows
    WORKFLOW_EXECUTE = "workflow_execute"
    WORKFLOW_STATUS = "workflow_status"
    WORKFLOW_PROGRESS = "workflow_progress"
    WORKFLOW_RESULT = "workflow_result"
    
    # Real-time updates
    PROCESSING_UPDATE = "processing_update"
    SYSTEM_NOTIFICATION = "system_notification"
    PRESENCE_UPDATE = "presence_update"
    
    # System
    HEARTBEAT = "heartbeat"
    ERROR = "error"
    DISCONNECT = "disconnect"
    
    # Broadcasting
    BROADCAST = "broadcast"
    MULTICAST = "multicast"
    
    # Plugin system
    PLUGIN_MESSAGE = "plugin_message"
    PLUGIN_NOTIFICATION = "plugin_notification"


class BroadcastScope(Enum):
    """Scopes for message broadcasting."""
    ALL = "all"                          # All connected clients
    AUTHENTICATED = "authenticated"      # All authenticated users
    SESSION = "session"                  # All connections in a session
    USER = "user"                        # All connections for a user
    ROLE = "role"                        # All users with specific role
    CUSTOM = "custom"                    # Custom filter criteria


@dataclass
class WebSocketMessage:
    """Structured WebSocket message."""
    message_id: str
    message_type: MessageType
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Message routing
    source_connection_id: Optional[str] = None
    target_connection_id: Optional[str] = None
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    
    # Message properties
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = None
    priority: int = 1
    require_ack: bool = False
    
    # Encryption and security
    encrypted: bool = False
    signature: Optional[str] = None
    
    def to_json(self) -> str:
        """Convert message to JSON string."""
        data = asdict(self)
        # Convert datetime objects to ISO strings
        data['timestamp'] = self.timestamp.isoformat()
        if self.expires_at:
            data['expires_at'] = self.expires_at.isoformat()
        data['message_type'] = self.message_type.value
        return json.dumps(data)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'WebSocketMessage':
        """Create message from JSON string."""
        data = json.loads(json_str)
        # Convert ISO strings back to datetime objects
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        if data.get('expires_at'):
            data['expires_at'] = datetime.fromisoformat(data['expires_at'])
        data['message_type'] = MessageType(data['message_type'])
        return cls(**data)


@dataclass
class ConnectionInfo:
    """Information about a WebSocket connection."""
    connection_id: str
    websocket: WebSocketServerProtocol
    state: ConnectionState = ConnectionState.CONNECTING
    
    # User information
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    authentication_token: Optional[str] = None
    roles: Set[str] = field(default_factory=set)
    
    # Connection metadata
    client_ip: Optional[str] = None
    user_agent: Optional[str] = None
    connection_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_activity: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_heartbeat: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Interaction context
    current_interaction_id: Optional[str] = None
    current_workflow_id: Optional[str] = None
    
    # Performance metrics
    messages_sent: int = 0
    messages_received: int = 0
    bytes_sent: int = 0
    bytes_received: int = 0
    latency_samples: deque = field(default_factory=lambda: deque(maxlen=100))
    
    # Message queue
    message_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    pending_acks: Dict[str, datetime] = field(default_factory=dict)
    
    # Subscriptions and preferences
    subscriptions: Set[str] = field(default_factory=set)
    notification_preferences: Dict[str, bool] = field(default_factory=dict)
    
    # Error tracking
    error_count: int = 0
    last_error: Optional[Exception] = None
    
    @property
    def is_authenticated(self) -> bool:
        """Check if connection is authenticated."""
        return self.state in [ConnectionState.AUTHENTICATED, ConnectionState.ACTIVE]
    
    @property
    def uptime(self) -> float:
        """Get connection uptime in seconds."""
        return (datetime.now(timezone.utc) - self.connection_time).total_seconds()


class ConnectionError(Exception):
    """Custom exception for connection management."""
    
    def __init__(self, message: str, connection_id: Optional[str] = None,
                 error_code: Optional[str] = None):
        super().__init__(message)
        self.connection_id = connection_id
        self.error_code = error_code
        self.timestamp = datetime.now(timezone.utc)


class MessageRouter:
    """Routes messages between connections and system components."""
    
    def __init__(self, logger):
        self.logger = logger
        self.routing_rules: Dict[str, Callable] = {}
        self.message_handlers: Dict[MessageType, List[Callable]] = defaultdict(list)
    
    def register_handler(self, message_type: MessageType, handler: Callable) -> None:
        """Register a message handler for a specific message type."""
        self.message_handlers[message_type].append(handler)
    
    def register_routing_rule(self, rule_name: str, rule_func: Callable) -> None:
        """Register a custom routing rule."""
        self.routing_rules[rule_name] = rule_func
    
    async def route_message(self, message: WebSocketMessage, 
                          source_connection: ConnectionInfo) -> List[str]:
        """Route message and return target connection IDs."""
        target_connections = []
        
        # Direct targeting
        if message.target_connection_id:
            target_connections.append(message.target_connection_id)
        
        # Session-based routing
        elif message.session_id:
            # Would get all connections for the session
            pass
        
        # User-based routing
        elif message.user_id:
            # Would get all connections for the user
            pass
        
        # Apply custom routing rules
        for rule_name, rule_func in self.routing_rules.items():
            try:
                rule_targets = await rule_func(message, source_connection)
                if rule_targets:
                    target_connections.extend(rule_targets)
            except Exception as e:
                self.logger.warning(f"Routing rule {rule_name} failed: {str(e)}")
        
        return list(set(target_connections))  # Remove duplicates
    
    async def handle_message(self, message: WebSocketMessage, 
                           source_connection: ConnectionInfo) -> None:
        """Handle message using registered handlers."""
        handlers = self.message_handlers.get(message.message_type, [])
        
        for handler in handlers:
            try:
                await handler(message, source_connection)
            except Exception as e:
                self.logger.error(f"Message handler failed: {str(e)}")


class BroadcastManager:
    """Manages message broadcasting to multiple connections."""
    
    def __init__(self, logger, redis_cache: Optional[RedisCache] = None):
        self.logger = logger
        self.redis_cache = redis_cache
        self.broadcast_filters: Dict[str, Callable] = {}
        
        # Redis pub/sub for distributed broadcasting
        if self.redis_cache:
            self.redis_subscriber = None
            self.redis_channel = "websocket_broadcast"
    
    async def initialize(self) -> None:
        """Initialize broadcast manager."""
        if self.redis_cache:
            try:
                self.redis_subscriber = self.redis_cache.pubsub()
                await self.redis_subscriber.subscribe(self.redis_channel)
                asyncio.create_task(self._redis_message_handler())
            except Exception as e:
                self.logger.warning(f"Failed to initialize Redis broadcasting: {str(e)}")
    
    def register_filter(self, filter_name: str, filter_func: Callable) -> None:
        """Register a custom broadcast filter."""
        self.broadcast_filters[filter_name] = filter_func
    
    async def broadcast(self, message: WebSocketMessage, scope: BroadcastScope,
                      connections: Dict[str, ConnectionInfo],
                      filter_criteria: Optional[Dict[str, Any]] = None) -> int:
        """Broadcast message to connections based on scope."""
        target_connections = await self._filter_connections(
            scope, connections, filter_criteria
        )
        
        broadcast_count = 0
        broadcast_message = message.to_json()
        
        # Broadcast to local connections
        for conn_id in target_connections:
            connection = connections.get(conn_id)
            if connection and connection.websocket:
                try:
                    await connection.websocket.send(broadcast_message)
                    connection.messages_sent += 1
                    connection.bytes_sent += len(broadcast_message)
                    broadcast_count += 1
                except Exception as e:
                    self.logger.warning(f"Failed to send to connection {conn_id}: {str(e)}")
        
        # Distribute via Redis for multi-node deployments
        if self.redis_cache and broadcast_count > 0:
            try:
                distributed_message = {
                    'message': message.to_json(),
                    'scope': scope.value,
                    'filter_criteria': filter_criteria,
                    'source_node': 'current_node'  # Would be actual node ID
                }
                await self.redis_cache.publish(
                    self.redis_channel, 
                    json.dumps(distributed_message)
                )
            except Exception as e:
                self.logger.warning(f"Failed to distribute broadcast via Redis: {str(e)}")
        
        return broadcast_count
    
    async def _filter_connections(self, scope: BroadcastScope,
                                connections: Dict[str, ConnectionInfo],
                                filter_criteria: Optional[Dict[str, Any]] = None) -> List[str]:
        """Filter connections based on broadcast scope."""
        filtered_connections = []
        
        for conn_id, connection in connections.items():
            include = False
            
            if scope == BroadcastScope.ALL:
                include = True
            elif scope == BroadcastScope.AUTHENTICATED:
                include = connection.is_authenticated
            elif scope == BroadcastScope.SESSION and filter_criteria:
                include = connection.session_id == filter_criteria.get('session_id')
            elif scope == BroadcastScope.USER and filter_criteria:
                include = connection.user_id == filter_criteria.get('user_id')
            elif scope == BroadcastScope.ROLE and filter_criteria:
                required_role = filter_criteria.get('role')
                include = required_role in connection.roles
            elif scope == BroadcastScope.CUSTOM and filter_criteria:
                # Apply custom filters
                include = await self._apply_custom_filters(connection, filter_criteria)
            
            if include:
                filtered_connections.append(conn_id)
        
        return filtered_connections
    
    async def _apply_custom_filters(self, connection: ConnectionInfo,
                                  filter_criteria: Dict[str, Any]) -> bool:
        """Apply custom broadcast filters."""
        for filter_name, filter_value in filter_criteria.items():
            if filter_name in self.broadcast_filters:
                try:
                    filter_func = self.broadcast_filters[filter_name]
                    if not await filter_func(connection, filter_value):
                        return False
                except Exception as e:
                    self.logger.warning(f"Custom filter {filter_name} failed: {str(e)}")
                    return False
        return True
    
    async def _redis_message_handler(self) -> None:
        """Handle incoming Redis broadcast messages."""
        if not self.redis_subscriber:
            return
        
        try:
            async for message in self.redis_subscriber.listen():
                if message['type'] == 'message':
                    try:
                        data = json.loads(message['data'])
                        # Handle distributed broadcast message
                        # This would be implemented based on the specific setup
                        pass
                    except Exception as e:
                        self.logger.error(f"Failed to handle Redis broadcast: {str(e)}")
        except Exception as e:
            self.logger.error(f"Redis message handler error: {str(e)}")


class ConnectionManager:
    """
    Advanced WebSocket Connection Manager for the AI Assistant.
    
    This manager provides comprehensive WebSocket connection handling including:
    - Connection lifecycle management with authentication
    - Real-time bidirectional messaging
    - Integration with all core system components
    - Broadcasting and message routing
    - Performance monitoring and metrics
    - Security and rate limiting
    - Scalability with Redis pub/sub
    - Event-driven architecture integration
    - Session and interaction management
    - Plugin system integration
    """
    
    def __init__(self, container: Container):
        """
        Initialize the connection manager.
        
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
        
        # Assistant components
        self.core_engine = container.get(EnhancedCoreEngine)
        self.component_manager = container.get(ComponentManager)
        self.workflow_orchestrator = container.get(WorkflowOrchestrator)
        self.session_manager = container.get(SessionManager)
        self.interaction_handler = container.get(InteractionHandler)
        self.plugin_manager = container.get(PluginManager)
        
        # Memory and context
        self.memory_manager = container.get(MemoryManager)
        self.context_manager = container.get(ContextManager)
        
        # Security
        self.auth_manager = container.get(AuthenticationManager)
        self.authz_manager = container.get(AuthorizationManager)
        self.encryption_manager = container.get(EncryptionManager)
        
        # Integrations
        try:
            self.redis_cache = container.get(RedisCache)
        except Exception:
            self.redis_cache = None
        
        # Learning systems
        self.continual_learner = container.get(ContinualLearner)
        self.preference_learner = container.get(PreferenceLearner)
        self.feedback_processor = container.get(FeedbackProcessor)
        
        # Observability
        self.metrics = container.get(MetricsCollector)
        self.tracer = container.get(TraceManager)
        
        # Connection management
        self.connections: Dict[str, ConnectionInfo] = {}
        self.user_connections: Dict[str, Set[str]] = defaultdict(set)
        self.session_connections: Dict[str, Set[str]] = defaultdict(set)
        
        # Messaging infrastructure
        self.message_router = MessageRouter(self.logger)
        self.broadcast_manager = BroadcastManager(self.logger, self.redis_cache)
        
        # Rate limiting
        self.rate_limits: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.connection_limits: Dict[str, int] = defaultdict(int)
        
        # Background tasks
        self.background_tasks: List[asyncio.Task] = []
        self.shutdown_event = asyncio.Event()
        
        # Performance tracking
        self.connection_stats: Dict[str, Any] = defaultdict(dict)
        self.message_stats: Dict[str, int] = defaultdict(int)
        
        # Configuration
        self.host = self.config.get("websocket.host", "localhost")
        self.port = self.config.get("websocket.port", 8765)
        self.max_connections = self.config.get("websocket.max_connections", 1000)
        self.max_connections_per_user = self.config.get("websocket.max_connections_per_user", 5)
        self.heartbeat_interval = self.config.get("websocket.heartbeat_interval", 30.0)
        self.message_rate_limit = self.config.get("websocket.message_rate_limit", 60)
        self.enable_compression = self.config.get("websocket.enable_compression", True)
        self.enable_ssl = self.config.get("websocket.enable_ssl", False)
        
        # Setup message handlers
        self._setup_message_handlers()
        self._setup_monitoring()
        
        # Register health check
        self.health_check.register_component("connection_manager", self._health_check_callback)
        
        self.logger.info("ConnectionManager initialized successfully")

    def _setup_message_handlers(self) -> None:
        """Setup message handlers for different message types."""
        # Authentication handlers
        self.message_router.register_handler(
            MessageType.AUTH_REQUEST, self._handle_auth_request
        )
        self.message_router.register_handler(
            MessageType.AUTH_TOKEN_REFRESH, self._handle_token_refresh
        )
        
        # Session handlers
        self.message_router.register_handler(
            MessageType.SESSION_START, self._handle_session_start
        )
        self.message_router.register_handler(
            MessageType.SESSION_END, self._handle_session_end
        )
        
        # Interaction handlers
        self.message_router.register_handler(
            MessageType.USER_MESSAGE, self._handle_user_message
        )
        self.message_router.register_handler(
            MessageType.INTERACTION_START, self._handle_interaction_start
        )
        self.message_router.register_handler(
            MessageType.INTERACTION_END, self._handle_interaction_end
        )
        
        # Workflow handlers
        self.message_router.register_handler(
            MessageType.WORKFLOW_EXECUTE, self._handle_workflow_execute
        )
        
        # System handlers
        self.message_router.register_handler(
            MessageType.HEARTBEAT, self._handle_heartbeat
        )
        
        # Plugin handlers
        self.message_router.register_handler(
            MessageType.PLUGIN_MESSAGE, self._handle_plugin_message
        )

    def _setup_monitoring(self) -> None:
        """Setup monitoring and metrics."""
        try:
            # Register WebSocket metrics
            self.metrics.register_counter("websocket_connections_total")
            self.metrics.register_gauge("websocket_connections_active")
            self.metrics.register_counter("websocket_messages_sent_total")
            self.metrics.register_counter("websocket_messages_received_total")
            self.metrics.register_histogram("websocket_message_processing_duration_seconds")
            self.metrics.register_counter("websocket_authentication_attempts_total")
            self.metrics.register_counter("websocket_authentication_failures_total")
            self.metrics.register_counter("websocket_rate_limit_exceeded_total")
            self.metrics.register_histogram("websocket_connection_duration_seconds")
            self.metrics.register_gauge("websocket_message_queue_size")
            
        except Exception as e:
            self.logger.warning(f"Failed to setup monitoring: {str(e)}")

    async def initialize(self) -> None:
        """Initialize the connection manager."""
        try:
            # Initialize broadcast manager
            await self.broadcast_manager.initialize()
            
            # Register event handlers
            await self._register_event_handlers()
            
            # Start background tasks
            self.background_tasks.extend([
                asyncio.create_task(self._heartbeat_monitor()),
                asyncio.create_task(self._connection_cleanup()),
                asyncio.create_task(self._performance_monitor()),
                asyncio.create_task(self._rate_limit_cleanup())
            ])
            
            self.logger.info("ConnectionManager initialization completed")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize ConnectionManager: {str(e)}")
            raise ConnectionError(f"Initialization failed: {str(e)}")

    async def _register_event_handlers(self) -> None:
        """Register event handlers for system events."""
        # System events
        self.event_bus.subscribe("workflow_started", self._handle_workflow_started_event)
        self.event_bus.subscribe("workflow_completed", self._handle_workflow_completed_event)
        self.event_bus.subscribe("workflow_failed", self._handle_workflow_failed_event)
        self.event_bus.subscribe("processing_started", self._handle_processing_started_event)
        self.event_bus.subscribe("processing_completed", self._handle_processing_completed_event)
        self.event_bus.subscribe("session_ended", self._handle_session_ended_event)
        self.event_bus.subscribe("component_health_changed", self._handle_component_health_event)
        self.event_bus.subscribe("system_shutdown_started", self._handle_system_shutdown)

    async def start_server(self) -> None:
        """Start the WebSocket server."""
        try:
            # SSL configuration
            ssl_context = None
            if self.enable_ssl:
                ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
                ssl_context.load_cert_chain(
                    self.config.get("websocket.ssl_cert_path"),
                    self.config.get("websocket.ssl_key_path")
                )
            
            # WebSocket server options
            server_options = {
                'compression': 'deflate' if self.enable_compression else None,
                'max_size': self.config.get("websocket.max_message_size", 1048576),  # 1MB
                'max_queue': self.config.get("websocket.max_queue_size", 32),
                'read_limit': self.config.get("websocket.read_limit", 65536),
                'write_limit': self.config.get("websocket.write_limit", 65536),
            }
            
            # Start WebSocket server
            self.server = await websockets.serve(
                self._handle_connection,
                self.host,
                self.port,
                ssl=ssl_context,
                **server_options
            )
            
            self.logger.info(f"WebSocket server started on {self.host}:{self.port}")
            
        except Exception as e:
            self.logger.error(f"Failed to start WebSocket server: {str(e)}")
            raise ConnectionError(f"Server startup failed: {str(e)}")

    async def _handle_connection(self, websocket: WebSocketServerProtocol, path: str) -> None:
        """Handle a new WebSocket connection."""
        connection_id = str(uuid.uuid4())
        connection = None
        
        try:
            # Check connection limits
            if len(self.connections) >= self.max_connections:
                await websocket.close(code=1013, reason="Server overloaded")
                return
            
            # Extract client information
            client_ip = websocket.remote_address[0] if websocket.remote_address else "unknown"
            user_agent = websocket.request_headers.get("User-Agent", "unknown")
            
            # Create connection info
            connection = ConnectionInfo(
                connection_id=connection_id,
                websocket=websocket,
                client_ip=client_ip,
                user_agent=user_agent
            )
            
            # Register connection
            self.connections[connection_id] = connection
            
            # Emit connection established event
            await self.event_bus.emit(ConnectionEstablished(
                connection_id=connection_id,
                client_ip=client_ip,
                user_agent=user_agent
            ))
            
            # Update metrics
            self.metrics.increment("websocket_connections_total")
            self.metrics.set("websocket_connections_active", len(self.connections))
            
            self.logger.info(f"New WebSocket connection: {connection_id} from {client_ip}")
            
            # Handle connection lifecycle
            await self._connection_handler(connection)
            
        except ConnectionClosed:
            self.logger.info(f"Connection {connection_id} closed normally")
        except WebSocketException as e:
            self.logger.warning(f"WebSocket error on connection {connection_id}: {str(e)}")
        except Exception as e:
            self.logger.error(f"Unexpected error on connection {connection_id}: {str(e)}")
        finally:
            # Cleanup connection
            await self._cleanup_connection(connection_id)

    async def _connection_handler(self, connection: ConnectionInfo) -> None:
        """Handle the lifecycle of a WebSocket connection."""
        try:
            # Set connection state
            connection.state = ConnectionState.CONNECTED
            
            # Start message processing
            message_task = asyncio.create_task(self._message_processor(connection))
            queue_task = asyncio.create_task(self._queue_processor(connection))
            
            # Wait for either task to complete
            done, pending = await asyncio.wait(
                [message_task, queue_task],
                return_when=asyncio.FIRST_COMPLETED
            )
            
            # Cancel pending tasks
            for task in pending:
                task.cancel()
            
            # Check for exceptions
            for task in done:
                if task.exception():
                    raise task.exception()
                    
        except Exception as e:
            connection.state = ConnectionState.ERROR
            connection.last_error = e
            self.logger.error(f"Connection handler error for {connection.connection_id}: {str(e)}")
            raise

    async def _message_processor(self, connection: ConnectionInfo) -> None:
        """Process incoming messages from a WebSocket connection."""
        while connection.state not in [ConnectionState.DISCONNECTING, ConnectionState.DISCONNECTED]:
            try:
                # Receive message with timeout
                raw_message = await asyncio.wait_for(
                    connection.websocket.recv(),
                    timeout=self.config.get("websocket.receive_timeout", 60.0)
                )
                
                # Update connection activity
                connection.last_activity = datetime.now(timezone.utc)
                connection.messages_received += 1
                connection.bytes_received += len(raw_message)
                
                # Rate limiting check
                if not await self._check_rate_limit(connection):
                    await self._send_error(connection, "Rate limit exceeded", "RATE_LIMIT")
                    continue
                
                # Parse message
                try:
                    message = WebSocketMessage.from_json(raw_message)
                    message.source_connection_id = connection.connection_id
                except json.JSONDecodeError as e:
                    await self._send_error(connection, f"Invalid JSON: {str(e)}", "INVALID_JSON")
                    continue
                except Exception as e:
                    await self._send_error(connection, f"Message parsing error: {str(e)}", "PARSE_ERROR")
                    continue
                
                # Update metrics
                self.metrics.increment("websocket_messages_received_total")
                self.message_stats[message.message_type.value] += 1
                
                # Emit message received event
                await self.event_bus.emit(MessageReceived(
                    connection_id=connection.connection_id,
                    message_type=message.message_type.value,
                    user_id=connection.user_id,
                    session_id=connection.session_id
                ))
                
                # Route and handle message
                start_time = time.time()
                
                try:
                    # Handle message through router
                    await self.message_router.handle_message(message, connection)
                    
                    # Route to other connections if needed
                    target_connections = await self.message_router.route_message(message, connection)
                    for target_id in target_connections:
                        target_connection = self.connections.get(target_id)
                        if target_connection:
                            await target_connection.message_queue.put(message)
                    
                except Exception as e:
                    await self._send_error(connection, f"Message handling error: {str(e)}", "HANDLER_ERROR")
                    self.logger.error(f"Message handling error: {str(e)}")
                
                # Record processing time
                processing_time = time.time() - start_time
                self.metrics.record("websocket_message_processing_duration_seconds", processing_time)
                
                # Send acknowledgment if required
                if message.require_ack:
                    await self._send_ack(connection, message.message_id)
                
            except asyncio.TimeoutError:
                # No message received within timeout, continue
                continue
            except ConnectionClosed:
                break
            except Exception as e:
                self.logger.error(f"Message processor error for {connection.connection_id}: {str(e)}")
                break

    async def _queue_processor(self, connection: ConnectionInfo) -> None:
        """Process outgoing message queue for a connection."""
        while connection.state not in [ConnectionState.DISCONNECTING, ConnectionState.DISCONNECTED]:
            try:
                # Get message from queue with timeout
                message = await asyncio.wait_for(
                    connection.message_queue.get(),
                    timeout=1.0
                )
                
                # Send message
                await self._send_message(connection, message)
                
            except asyncio.TimeoutError:
                # No message in queue, continue
                continue
            except ConnectionClosed:
                break
            except Exception as e:
                self.logger.error(f"Queue processor error for {connection.connection_id}: {str(e)}")
                break

    async def _check_rate_limit(self, connection: ConnectionInfo) -> bool:
        """Check if connection is within rate limits."""
        current_time = time.time()
        rate_window = 60.0  # 1 minute window
        
        # Clean old entries
        rate_history = self.rate_limits[connection.connection_id]
        while rate_history and current_time - rate_history[0] > rate_window:
            rate_history.popleft()
        
        # Check rate limit
        if len(rate_history) >= self.message_rate_limit:
            self.metrics.increment("websocket_rate_limit_exceeded_total")
            return False
        
        # Add current request
        rate_history.append(current_time)
        return True

    async def _send_message(self, connection: ConnectionInfo, message: WebSocketMessage) -> None:
        """Send a message to a WebSocket connection."""
        try:
            if connection.websocket and connection.state not in [
                ConnectionState.DISCONNECTING, ConnectionState.DISCONNECTED
            ]:
                # Encrypt message if required
                message_json = message.to_json()
                if connection.is_authenticated and self.config.get("websocket.enable_encryption", False):
                    # Would implement encryption here
                    pass
                
                # Send message
                await connection.websocket.send(message_json)
                
                # Update metrics
                connection.messages_sent += 1
                connection.bytes_sent += len(message_json)
                self.metrics.increment("websocket_messages_sent_total")
                
                # Emit message sent event
                await self.event_bus.emit(MessageSent(
                    connection_id=connection.connection_id,
                    message_type=message.message_type.value,
                    user_id=connection.user_id,
                    session_id=connection.session_id
                ))
                
        except ConnectionClosed:
            connection.state = ConnectionState.DISCONNECTED
        except Exception as e:
            self.logger.error(f"Failed to send message to {connection.connection_id}: {str(e)}")
            connection.error_count += 1
            connection.last_error = e

    async def _send_error(self, connection: ConnectionInfo, error_message: str, 
                         error_code: str) -> None:
        """Send an error message to a connection."""
        error_message_obj = WebSocketMessage(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.ERROR,
            data={
                "error_code": error_code,
                "error_message": error_message,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )
        await self._send_message(connection, error_message_obj)

    async def _send_ack(self, connection: ConnectionInfo, message_id: str) -> None:
        """Send an acknowledgment for a received message."""
        ack_message = WebSocketMessage(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.HEARTBEAT,  # Using heartbeat type for acks
            data={
                "ack_for": message_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )
        await self._send_message(connection, ack_message)

    # Message Handlers
    
    async def _handle_auth_request(self, message: WebSocketMessage, 
                                 connection: ConnectionInfo) -> None:
        """Handle authentication request."""
        try:
            token = message.data.get("token")
            if not token:
                await self._send_error(connection, "Authentication token required", "AUTH_REQUIRED")
                return
            
            # Authenticate with auth manager
            auth_result = await self.auth_manager.verify_token(token)
            
            if auth_result.get("valid"):
                user_id = auth_result.get("user_id")
                roles = set(auth_result.get("roles", []))
                
                # Check connection limits per user
                if len(self.user_connections[user_id]) >= self.max_connections_per_user:
                    await self._send_error(connection, "Too many connections for user", "CONNECTION_LIMIT")
                    return
                
                # Update connection info
                connection.user_id = user_id
                connection.authentication_token = token
                connection.roles = roles
                connection.state = ConnectionState.AUTHENTICATED
                
                # Track user connections
                self.user_connections[user_id].add(connection.connection_id)
                
                # Send authentication response
                auth_response = WebSocketMessage(
                    message_id=str(uuid.uuid4()),
                    message_type=MessageType.AUTH_RESPONSE,
                    data={
                        "authenticated": True,
                        "user_id": user_id,
                        "roles": list(roles),
                        "connection_id": connection.connection_id
                    }
                )
                await self._send_message(connection, auth_response)
                
                # Emit authentication event
                await self.event_bus.emit(UserAuthenticated(
                    user_id=user_id,
                    connection_id=connection.connection_id,
                    roles=list(roles)
                ))
                
                self.metrics.increment("websocket_authentication_attempts_total", tags={"status": "success"})
                self.logger.info(f"User {user_id} authenticated on connection {connection.connection_id}")
                
            else:
                await self._send_error(connection, "Invalid authentication token", "AUTH_INVALID")
                self.metrics.increment("websocket_authentication_failures_total")
                
        except Exception as e:
            await self._send_error(connection, f"Authentication error: {str(e)}", "AUTH_ERROR")
            self.logger.error(f"Authentication error for {connection.connection_id}: {str(e)}")

    async def _handle_token_refresh(self, message: WebSocketMessage,
                                  connection: ConnectionInfo) -> None:
        """Handle token refresh request."""
        try:
            refresh_token = message.data.get("refresh_token")
            if not refresh_token:
                await self._send_error(connection, "Refresh token required", "REFRESH_REQUIRED")
                return
            
            # Refresh token with auth manager
            refresh_result = await self.auth_manager.refresh_token(refresh_token)
            
            if refresh_result.get("success"):
                new_token = refresh_result.get("access_token")
                connection.authentication_token = new_token
                
                # Send new token
                response = WebSocketMessage(
                    message_id=str(uuid.uuid4()),
                    message_type=MessageType.AUTH_TOKEN_REFRESH,
                    data={
                        "success": True,
                        "access_token": new_token,
                        "expires_in": refresh_result.get("expires_in")
                    }
                )
                await self._send_message(connection, response)
                
            else:
                await self._send_error(connection, "Token refresh failed", "REFRESH_FAILED")
                
        except Exception as e:
            await self._send_error(connection, f"Token refresh error: {str(e)}", "REFRESH_ERROR")

    async def _handle_session_start(self, message: WebSocketMessage,
                                  connection: ConnectionInfo) -> None:
        """Handle session start request."""
        try:
            if not connection.is_authenticated:
                await self._send_error(connection, "Authentication required", "AUTH_REQUIRED")
                return
            
            # Create session
            session_id = await self.session_manager.create_session(
                user_id=connection.user_id,
                session_config=message.data.get("config")
            )
            
            # Update connection
            connection.session_id = session_id
            self.session_connections[session_id].add(connection.connection_id)
            
            # Send response
            response = WebSocketMessage(
                message_id=str(uuid.uuid4()),
                message_type=MessageType.SESSION_START,
                data={
                    "success": True,
                    "session_id": session_id
                }
            )
            await self._send_message(connection, response)
            
            self.logger.info(f"Session {session_id} started for user {connection.user_id}")
            
        except Exception as e:
            await self._send_error(connection, f"Session start error: {str(e)}", "SESSION_ERROR")

    async def _handle_session_end(self, message: WebSocketMessage,
                                connection: ConnectionInfo) -> None:
        """Handle session end request."""
        try:
            if connection.session_id:
                await self.session_manager.end_session(connection.session_id)
                
                # Cleanup session connections
                self.session_connections[connection.session_id].discard(connection.connection_id)
                connection.session_id = None
                
                # Send response
                response = WebSocketMessage(
                    message_id=str(uuid.uuid4()),
                    message_type=MessageType.SESSION_END,
                    data={"success": True}
                )
                await self._send_message(connection, response)
                
        except Exception as e:
            await self._send_error(connection, f"Session end error: {str(e)}", "SESSION_ERROR")

    async def _handle_user_message(self, message: WebSocketMessage,
                                 connection: ConnectionInfo) -> None:
        """Handle user message for processing."""
        try:
            if not connection.is_authenticated or not connection.session_id:
                await self._send_error(connection, "Session required", "SESSION_REQUIRED")
                return
            
            # Create user message
            user_message = UserMessage(
                message_id=message.message_id,
                user_id=connection.user_id,
                session_id=connection.session_id,
                text=message.data.get("text"),
                modality=InputModality(message.data.get("modality", "text"))
            )
            
            # Start or get interaction
            if not connection.current_interaction_id:
                interaction_id = await self.interaction_handler.start_interaction(
                    user_id=connection.user_id,
                    session_id=connection.session_id,
                    interaction_mode=InteractionMode(message.data.get("interaction_mode", "conversational"))
                )
                connection.current_interaction_id = interaction_id
            
            # Process message
            response = await self.interaction_handler.process_user_message(
                connection.current_interaction_id,
                user_message,
                real_time=message.data.get("real_time", False),
                streaming=message.data.get("streaming", False)
            )
            
            # Send response
            response_message = WebSocketMessage(
                message_id=str(uuid.uuid4()),
                message_type=MessageType.ASSISTANT_RESPONSE,
                data={
                    "response_id": response.response_id,
                    "interaction_id": connection.current_interaction_id,
                    "text": response.text,
                    "audio_data": response.audio_data.tolist() if response.audio_data is not None else None,
                    "confidence": response.confidence,
                    "processing_time": response.processing_time,
                    "modalities": [m.value for m in response.modalities]
                },
                session_id=connection.session_id,
                user_id=connection.user_id
            )
            await self._send_message(connection, response_message)
            
        except Exception as e:
            await self._send_error(connection, f"Message processing error: {str(e)}", "PROCESSING_ERROR")
            self.logger.error(f"User message processing error: {str(e)}")

    async def _handle_interaction_start(self, message: WebSocketMessage,
                                      connection: ConnectionInfo) -> None:
        """Handle interaction start request."""
        try:
            if not connection.is_authenticated or not connection.session_id:
                await self._send_error(connection, "Session required", "SESSION_REQUIRED")
                return
            
            interaction_id = await self.interaction_handler.start_interaction(
                user_id=connection.user_id,
                session_id=connection.session_id,
                interaction_mode=InteractionMode(message.data.get("mode", "conversational")),
                priority=InteractionPriority(message.data.get("priority", 1))
            )
            
            connection.current_interaction_id = interaction_id
            
            response = WebSocketMessage(
                message_id=str(uuid.uuid4()),
                message_type=MessageType.INTERACTION_START,
                data={
                    "success": True,
                    "interaction_id": interaction_id
                }
            )
            await self._send_message(connection, response)
            
        except Exception as e:
            await self._send_error(connection, f"Interaction start error: {str(e)}", "INTERACTION_ERROR")

    async def _handle_interaction_end(self, message: WebSocketMessage,
                                    connection: ConnectionInfo) -> None:
        """Handle interaction end request."""
        try:
            if connection.current_interaction_id:
                await self.interaction_handler.end_interaction(
                    connection.current_interaction_id,
                    reason=message.data.get("reason", "user_ended")
                )
                connection.current_interaction_id = None
                
                response = WebSocketMessage(
                    message_id=str(uuid.uuid4()),
                    message_type=MessageType.INTERACTION_END,
                    data={"success": True}
                )
                await self._send_message(connection, response)
                
        except Exception as e:
            await self._send_error(connection, f"Interaction end error: {str(e)}", "INTERACTION_ERROR")

    async def _handle_workflow_execute(self, message: WebSocketMessage,
                                     connection: ConnectionInfo) -> None:
        """Handle workflow execution request."""
        try:
            if not connection.is_authenticated or not connection.session_id:
                await self._send_error(connection, "Session required", "SESSION_REQUIRED")
                return
            
            workflow_id = message.data.get("workflow_id")
            input_data = message.data.get("input_data", {})
            priority = WorkflowPriority(message.data.get("priority", 1))
            
            execution_id = await self.workflow_orchestrator.execute_workflow(
                workflow_id=workflow_id,
                session_id=connection.session_id,
                input_data=input_data,
                user_id=connection.user_id,
                priority=priority
            )
            
            connection.current_workflow_id = execution_id
            
            response = WebSocketMessage(
                message_id=str(uuid.uuid4()),
                message_type=MessageType.WORKFLOW_EXECUTE,
                data={
                    "success": True,
                    "execution_id": execution_id
                }
            )
            await self._send_message(connection, response)
            
        except Exception as e:
            await self._send_error(connection, f"Workflow execution error: {str(e)}", "WORKFLOW_ERROR")

    async def _handle_heartbeat(self, message: WebSocketMessage,
                              connection: ConnectionInfo) -> None:
        """Handle heartbeat message."""
        connection.last_heartbeat = datetime.now(timezone.utc)
        
        # Calculate latency if timestamp provided
        if "timestamp" in message.data:
            try:
                sent_time = datetime.fromisoformat(message.data["timestamp"])
                latency = (connection.last_heartbeat - sent_time).total_seconds() * 1000
                connection.latency_samples.append(latency)
            except Exception:
                pass
        
        # Send heartbeat response
        response = WebSocketMessage(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.HEARTBEAT,
            data={
                "timestamp": connection.last_heartbeat.isoformat(),
                "status": "alive"
            }
        )
        await self._send_message(connection, response)

    async def _handle_plugin_message(self, message: WebSocketMessage,
                                   connection: ConnectionInfo) -> None:
        """Handle plugin-specific messages."""
        try:
            plugin_id = message.data.get("plugin_id")
            plugin_message = message.data.get("message")
            
            if not plugin_id or not plugin_message:
                await self._send_error(connection, "Plugin ID and message required", "PLUGIN_INVALID")
                return
            
            # Forward to plugin manager
            # This would be implemented based on plugin manager capabilities
            result = await self.plugin_manager.handle_websocket_message(
                plugin_id, plugin_message, connection.user_id
            )
            
            response = WebSocketMessage(
                message_id=str(uuid.uuid4()),
                message_type=MessageType.PLUGIN_MESSAGE,
                data={
                    "plugin_id": plugin_id,
                    "result": result
                }
            )
            await self._send_message(connection, response)
            
        except Exception as e:
            await self._send_error(connection, f"Plugin message error: {str(e)}", "PLUGIN_ERROR")

    # Event Handlers
    
    async def _handle_workflow_started_event(self, event) -> None:
        """Handle workflow started events."""
        # Notify relevant connections
        if event.session_id:
            await self._broadcast_to_session(
                event.session_id,
                WebSocketMessage(
                    message_id=str(uuid.uuid4()),
                    message_type=MessageType.WORKFLOW_STATUS,
                    data={
                        "workflow_id": event.workflow_id,
                        "execution_id": event.execution_id,
                        "status": "started",
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }
                )
            )

    async def _handle_workflow_completed_event(self, event) -> None:
        """Handle workflow completed events."""
        if event.session_id:
            await self._broadcast_to_session(
                event.session_id,
                WebSocketMessage(
                    message_id=str(uuid.uuid4()),
                    message_type=MessageType.WORKFLOW_RESULT,
                    data={
                        "workflow_id": event.workflow_id,
                        "execution_id": event.execution_id,
                        "status": "completed",
                        "execution_time": event.execution_time,
                        "steps_completed": event.steps_completed,
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }
                )
            )

    async def _handle_workflow_failed_event(self, event) -> None:
        """Handle workflow failed events."""
        if event.session_id:
            await self._broadcast_to_session(
                event.session_id,
                WebSocketMessage(
                    message_id=str(uuid.uuid4()),
                    message_type=MessageType.WORKFLOW_STATUS,
                    data={
                        "workflow_id": event.workflow_id,
                        "execution_id": event.execution_id,
                        "status": "failed",
                        "error_message": event.error_message,
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }
                )
            )

    async def _handle_processing_started_event(self, event) -> None:
        """Handle processing started events."""
        if event.session_id:
            await self._broadcast_to_session(
                event.session_id,
                WebSocketMessage(
                    message_id=str(uuid.uuid4()),
                    message_type=MessageType.PROCESSING_UPDATE,
                    data={
                        "status": "started",
                        "session_id": event.session_id,
                        "request_id": event.request_id,
                        "modalities": event.input_modalities,
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }
                )
            )

    async def _handle_processing_completed_event(self, event) -> None:
        """Handle processing completed events."""
        if event.session_id:
            await self._broadcast_to_session(
                event.session_id,
                WebSocketMessage(
                    message_id=str(uuid.uuid4()),
                    message_type=MessageType.PROCESSING_UPDATE,
                    data={
                        "status": "completed",
                        "session_id": event.session_id,
                        "request_id": event.request_id,
                        "processing_time": event.processing_time,
                        "confidence": event.confidence,
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }
                )
            )

    async def _handle_session_ended_event(self, event) -> None:
        """Handle session ended events."""
        # Clean up session connections
        session_connections = self.session_connections.get(event.session_id, set())
        for conn_id in session_connections:
            connection = self.connections.get(conn_id)
            if connection:
                connection.session_id = None
        
        # Remove session tracking
        self.session_connections.pop(event.session_id, None)

    async def _handle_component_health_event(self, event) -> None:
        """Handle component health change events."""
        # Broadcast system notifications to authenticated users
        if not event.healthy:
            await self.broadcast_notification(
                f"System component {event.component} is experiencing issues",
                scope=BroadcastScope.AUTHENTICATED
            )

    async def _handle_system_shutdown(self, event) -> None:
        """Handle system shutdown events."""
        self.shutdown_event.set()
        await self.shutdown()

    # Utility Methods
    
    async def _broadcast_to_session(self, session_id: str, message: WebSocketMessage) -> None:
        """Broadcast message to all connections in a session."""
        session_connections = self.session_connections.get(session_id, set())
        for conn_id in session_connections:
            connection = self.connections.get(conn_id)
            if connection:
                await connection.message_queue.put(message)

    async def _broadcast_to_user(self, user_id: str, message: WebSocketMessage) -> None:
        """Broadcast message to all connections for a user."""
        user_connections = self.user_connections.get(user_id, set())
        for conn_id in user_connections:
            connection = self.connections.get(conn_id)
            if connection:
                await connection.message_queue.put(message)

    async def broadcast_notification(self, notification: str, scope: BroadcastScope,
                                   filter_criteria: Optional[Dict[str, Any]] = None) -> int:
        """Broadcast a notification to connections."""
        message = WebSocketMessage(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.SYSTEM_NOTIFICATION,
            data={
                "notification": notification,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )
        
        return await self.broadcast_manager.broadcast(
            message, scope, self.connections, filter_criteria
        )

    async def send_to_connection(self, connection_id: str, message: WebSocketMessage) -> bool:
        """Send a message to a specific connection."""
        connection = self.connections.get(connection_id)
        if connection:
            await connection.message_queue.put(message)
            return True
        return False

    async def _cleanup_connection(self, connection_id: str) -> None:
        """Clean up a connection and associated resources."""
        try:
            connection = self.connections.pop(connection_id, None)
            if not connection:
                return
            
            # Update connection state
            connection.state = ConnectionState.DISCONNECTED
            
            # Clean up user tracking
            if connection.user_id:
                self.user_connections[connection.user_id].discard(connection_id)
            
            # Clean up session tracking
            if connection.session_id:
                self.session_connections[connection.session_id].discard(connection_id)
            
            # End active interactions
            if connection.current_interaction_id:
                try:
                    await self.interaction_handler.end_interaction(
                        connection.current_interaction_id, "connection_closed"
                    )
                except Exception as e:
                    self.logger.warning(f"Failed to end interaction: {str(e)}")
            
            # Calculate connection duration
            duration = connection.uptime
            
            # Emit connection closed event
            await self.event_bus.emit(ConnectionClosedEvent(
                connection_id=connection_id,
                user_id=connection.user_id,
                duration=duration,
                messages_sent=connection.messages_sent,
                messages_received=connection.messages_received
            ))
            
            # Update metrics
            self.metrics.set("websocket_connections_active", len(self.connections))
            self.metrics.record("websocket_connection_duration_seconds", duration)
            
            # Clean up rate limiting
            self.rate_limits.pop(connection_id, None)
            
            self.logger.info(f"Connection {connection_id} cleaned up (duration: {duration:.2f}s)")
            
        except Exception as e:
            self.logger.error(f"Error cleaning up connection {connection_id}: {str(e)}")

    # Background Tasks
    
    async def _heart
