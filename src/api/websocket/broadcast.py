"""
WebSocket Broadcasting System
Author: Drmusab
Last Modified: 2025-05-26 16:30:00 UTC

This module provides comprehensive broadcasting capabilities for WebSocket connections
in the AI assistant system. It enables efficient message distribution to multiple
clients based on various filtering criteria, with support for Redis-based distribution
for multi-node deployments.
"""

import json
import asyncio
import logging
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass
from enum import Enum

# Core imports - using optional imports to avoid dependency issues
try:
    from src.integrations.cache.redis_cache import RedisCache
except ImportError:
    RedisCache = None

try:
    from src.observability.logging.config import get_logger
except ImportError:
    import logging
    def get_logger(name):
        return logging.getLogger(name)


class BroadcastScope(Enum):
    """Scopes for message broadcasting."""
    ALL = "all"                          # All connected clients
    AUTHENTICATED = "authenticated"      # All authenticated users
    SESSION = "session"                  # All connections in a session
    USER = "user"                        # All connections for a user
    ROLE = "role"                        # All users with specific role
    CUSTOM = "custom"                    # Custom filter criteria


@dataclass
class BroadcastFilter:
    """Configuration for broadcast filtering."""
    name: str
    filter_func: Callable
    description: str = ""
    enabled: bool = True


@dataclass
class BroadcastMetrics:
    """Metrics for broadcast operations."""
    total_broadcasts: int = 0
    successful_deliveries: int = 0
    failed_deliveries: int = 0
    bytes_transmitted: int = 0
    average_latency_ms: float = 0.0
    redis_publishes: int = 0
    redis_failures: int = 0


class BroadcastManager:
    """
    Manages message broadcasting to multiple WebSocket connections.
    
    This manager provides:
    - Scope-based message filtering and distribution
    - Custom filter registration and application
    - Redis pub/sub for distributed broadcasting across nodes
    - Performance metrics and monitoring
    - Error handling and recovery
    """
    
    def __init__(self, redis_cache: Optional[Any] = None,  # Changed from RedisCache to Any
                 redis_channel: str = "websocket_broadcast"):
        """
        Initialize the broadcast manager.
        
        Args:
            redis_cache: Optional Redis cache for distributed broadcasting
            redis_channel: Redis channel name for pub/sub messaging
        """
        self.logger = get_logger(__name__)
        self.redis_cache = redis_cache
        self.redis_channel = redis_channel
        
        # Filter management
        self.broadcast_filters: Dict[str, BroadcastFilter] = {}
        
        # Metrics tracking
        self.metrics = BroadcastMetrics()
        
        # Redis pub/sub components
        self.redis_subscriber = None
        self.redis_handler_task = None
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize the broadcast manager and Redis connections."""
        if self._initialized:
            return
            
        if self.redis_cache:
            try:
                self.redis_subscriber = self.redis_cache.pubsub()
                await self.redis_subscriber.subscribe(self.redis_channel)
                
                # Start Redis message handler task
                self.redis_handler_task = asyncio.create_task(
                    self._redis_message_handler()
                )
                
                self.logger.info(f"Redis broadcasting initialized on channel: {self.redis_channel}")
            except Exception as e:
                self.logger.warning(f"Failed to initialize Redis broadcasting: {str(e)}")
        
        self._initialized = True
        self.logger.info("Broadcast manager initialized successfully")
    
    async def shutdown(self) -> None:
        """Shutdown the broadcast manager and cleanup resources."""
        if self.redis_handler_task:
            self.redis_handler_task.cancel()
            try:
                await self.redis_handler_task
            except asyncio.CancelledError:
                pass
        
        if self.redis_subscriber:
            await self.redis_subscriber.unsubscribe(self.redis_channel)
            await self.redis_subscriber.close()
        
        self._initialized = False
        self.logger.info("Broadcast manager shutdown complete")
    
    def register_filter(self, name: str, filter_func: Callable, 
                       description: str = "", enabled: bool = True) -> None:
        """
        Register a custom broadcast filter.
        
        Args:
            name: Unique name for the filter
            filter_func: Async function that takes (connection, filter_value) and returns bool
            description: Optional description of the filter
            enabled: Whether the filter is initially enabled
        """
        self.broadcast_filters[name] = BroadcastFilter(
            name=name,
            filter_func=filter_func,
            description=description,
            enabled=enabled
        )
        self.logger.info(f"Registered broadcast filter: {name}")
    
    def enable_filter(self, name: str) -> bool:
        """Enable a registered filter."""
        if name in self.broadcast_filters:
            self.broadcast_filters[name].enabled = True
            return True
        return False
    
    def disable_filter(self, name: str) -> bool:
        """Disable a registered filter."""
        if name in self.broadcast_filters:
            self.broadcast_filters[name].enabled = False
            return True
        return False
    
    def remove_filter(self, name: str) -> bool:
        """Remove a registered filter."""
        if name in self.broadcast_filters:
            del self.broadcast_filters[name]
            self.logger.info(f"Removed broadcast filter: {name}")
            return True
        return False
    
    async def broadcast(self, message: Any, scope: BroadcastScope,
                       connections: Dict[str, Any],
                       filter_criteria: Optional[Dict[str, Any]] = None,
                       exclude_connections: Optional[List[str]] = None) -> int:
        """
        Broadcast message to connections based on scope and filters.
        
        Args:
            message: Message to broadcast (should have to_json() method)
            scope: Broadcast scope for filtering
            connections: Dictionary of connection_id -> connection_info
            filter_criteria: Additional filter criteria
            exclude_connections: List of connection IDs to exclude
            
        Returns:
            Number of successful broadcasts
        """
        start_time = asyncio.get_event_loop().time()
        
        # Filter target connections
        target_connections = await self._filter_connections(
            scope, connections, filter_criteria, exclude_connections
        )
        
        if not target_connections:
            return 0
        
        # Prepare message
        if hasattr(message, 'to_json'):
            broadcast_message = message.to_json()
        else:
            broadcast_message = json.dumps(message)
        
        message_size = len(broadcast_message.encode('utf-8'))
        broadcast_count = 0
        
        # Broadcast to local connections
        for conn_id in target_connections:
            connection = connections.get(conn_id)
            if connection and hasattr(connection, 'websocket') and connection.websocket:
                try:
                    await connection.websocket.send(broadcast_message)
                    
                    # Update connection metrics if available
                    if hasattr(connection, 'messages_sent'):
                        connection.messages_sent += 1
                    if hasattr(connection, 'bytes_sent'):
                        connection.bytes_sent += message_size
                    
                    broadcast_count += 1
                    self.metrics.successful_deliveries += 1
                    
                except Exception as e:
                    self.logger.warning(f"Failed to send to connection {conn_id}: {str(e)}")
                    self.metrics.failed_deliveries += 1
        
        # Distribute via Redis for multi-node deployments
        if self.redis_cache and broadcast_count > 0:
            await self._distribute_via_redis(message, scope, filter_criteria, exclude_connections)
        
        # Update metrics
        end_time = asyncio.get_event_loop().time()
        latency_ms = (end_time - start_time) * 1000
        
        self.metrics.total_broadcasts += 1
        self.metrics.bytes_transmitted += message_size * broadcast_count
        self.metrics.average_latency_ms = (
            (self.metrics.average_latency_ms * (self.metrics.total_broadcasts - 1) + latency_ms) 
            / self.metrics.total_broadcasts
        )
        
        self.logger.debug(
            f"Broadcast completed: {broadcast_count} recipients, "
            f"{message_size} bytes, {latency_ms:.2f}ms latency"
        )
        
        return broadcast_count
    
    async def _filter_connections(self, scope: BroadcastScope,
                                 connections: Dict[str, Any],
                                 filter_criteria: Optional[Dict[str, Any]] = None,
                                 exclude_connections: Optional[List[str]] = None) -> List[str]:
        """Filter connections based on broadcast scope and criteria."""
        filtered_connections = []
        exclude_set = set(exclude_connections or [])
        
        for conn_id, connection in connections.items():
            if conn_id in exclude_set:
                continue
                
            include = await self._should_include_connection(scope, connection, filter_criteria)
            
            if include:
                filtered_connections.append(conn_id)
        
        return filtered_connections
    
    async def _should_include_connection(self, scope: BroadcastScope, connection: Any,
                                        filter_criteria: Optional[Dict[str, Any]] = None) -> bool:
        """Determine if a connection should be included in the broadcast."""
        # Basic scope filtering
        if scope == BroadcastScope.ALL:
            include = True
        elif scope == BroadcastScope.AUTHENTICATED:
            include = getattr(connection, 'is_authenticated', False)
        elif scope == BroadcastScope.SESSION and filter_criteria:
            include = getattr(connection, 'session_id', None) == filter_criteria.get('session_id')
        elif scope == BroadcastScope.USER and filter_criteria:
            include = getattr(connection, 'user_id', None) == filter_criteria.get('user_id')
        elif scope == BroadcastScope.ROLE and filter_criteria:
            required_role = filter_criteria.get('role')
            connection_roles = getattr(connection, 'roles', [])
            include = required_role in connection_roles
        elif scope == BroadcastScope.CUSTOM and filter_criteria:
            include = await self._apply_custom_filters(connection, filter_criteria)
        else:
            include = False
        
        return include
    
    async def _apply_custom_filters(self, connection: Any,
                                   filter_criteria: Dict[str, Any]) -> bool:
        """Apply custom broadcast filters."""
        for filter_name, filter_value in filter_criteria.items():
            if filter_name in self.broadcast_filters:
                filter_config = self.broadcast_filters[filter_name]
                
                if not filter_config.enabled:
                    continue
                    
                try:
                    filter_func = filter_config.filter_func
                    if not await filter_func(connection, filter_value):
                        return False
                except Exception as e:
                    self.logger.warning(f"Custom filter {filter_name} failed: {str(e)}")
                    return False
        
        return True
    
    async def _distribute_via_redis(self, message: Any, scope: BroadcastScope,
                                   filter_criteria: Optional[Dict[str, Any]] = None,
                                   exclude_connections: Optional[List[str]] = None) -> None:
        """Distribute broadcast message via Redis pub/sub."""
        try:
            distributed_message = {
                'message': message.to_json() if hasattr(message, 'to_json') else json.dumps(message),
                'scope': scope.value,
                'filter_criteria': filter_criteria,
                'exclude_connections': exclude_connections,
                'source_node': 'current_node',  # Would be actual node ID in production
                'timestamp': asyncio.get_event_loop().time()
            }
            
            await self.redis_cache.publish(
                self.redis_channel,
                json.dumps(distributed_message)
            )
            
            self.metrics.redis_publishes += 1
            
        except Exception as e:
            self.logger.warning(f"Failed to distribute broadcast via Redis: {str(e)}")
            self.metrics.redis_failures += 1
    
    async def _redis_message_handler(self) -> None:
        """Handle incoming Redis broadcast messages from other nodes."""
        if not self.redis_subscriber:
            return
        
        try:
            async for message in self.redis_subscriber.listen():
                if message['type'] == 'message':
                    try:
                        data = json.loads(message['data'])
                        
                        # Avoid processing messages from the same node
                        if data.get('source_node') == 'current_node':
                            continue
                        
                        # Implement distributed broadcast handling
                        message_type = data.get('type', 'unknown')
                        broadcast_data = data.get('data', {})
                        target_criteria = data.get('target_criteria', {})
                        
                        # 1. Parse the distributed message
                        if message_type == 'user_broadcast':
                            # Broadcast to specific user
                            user_id = target_criteria.get('user_id')
                            if user_id:
                                await self._broadcast_to_local_user(user_id, broadcast_data)
                                
                        elif message_type == 'session_broadcast':
                            # Broadcast to specific session
                            session_id = target_criteria.get('session_id')
                            if session_id:
                                await self._broadcast_to_local_session(session_id, broadcast_data)
                                
                        elif message_type == 'channel_broadcast':
                            # Broadcast to channel subscribers
                            channel = target_criteria.get('channel')
                            if channel:
                                await self._broadcast_to_local_channel(channel, broadcast_data)
                                
                        elif message_type == 'role_broadcast':
                            # Broadcast to users with specific role
                            role = target_criteria.get('role')
                            if role:
                                await self._broadcast_to_local_role(role, broadcast_data)
                                
                        else:
                            self.logger.warning(f"Unknown distributed broadcast type: {message_type}")
                        
                        # Update metrics
                        self.metrics.distributed_received += 1
                        
                        self.logger.debug(f"Received distributed broadcast: {data}")
                        
                    except Exception as e:
                        self.logger.error(f"Failed to handle Redis broadcast: {str(e)}")
        except Exception as e:
            self.logger.error(f"Redis message handler error: {str(e)}")
    
    async def _broadcast_to_local_user(self, user_id: str, data: Dict[str, Any]) -> None:
        """Broadcast to local connections for a specific user."""
        try:
            connections = self.connection_manager.get_user_connections(user_id)
            for connection_id in connections:
                await self.connection_manager.send_message(connection_id, data)
                self.metrics.distributed_delivered += 1
        except Exception as e:
            self.logger.warning(f"Failed to broadcast to local user {user_id}: {e}")
    
    async def _broadcast_to_local_session(self, session_id: str, data: Dict[str, Any]) -> None:
        """Broadcast to local connections for a specific session."""
        try:
            connections = self.connection_manager.get_session_connections(session_id)
            for connection_id in connections:
                await self.connection_manager.send_message(connection_id, data)
                self.metrics.distributed_delivered += 1
        except Exception as e:
            self.logger.warning(f"Failed to broadcast to local session {session_id}: {e}")
    
    async def _broadcast_to_local_channel(self, channel: str, data: Dict[str, Any]) -> None:
        """Broadcast to local connections subscribed to a channel."""
        try:
            connections = self.connection_manager.get_channel_connections(channel)
            for connection_id in connections:
                await self.connection_manager.send_message(connection_id, data)
                self.metrics.distributed_delivered += 1
        except Exception as e:
            self.logger.warning(f"Failed to broadcast to local channel {channel}: {e}")
    
    async def _broadcast_to_local_role(self, role: str, data: Dict[str, Any]) -> None:
        """Broadcast to local connections for users with a specific role."""
        try:
            connections = self.connection_manager.get_role_connections(role)
            for connection_id in connections:
                await self.connection_manager.send_message(connection_id, data)
                self.metrics.distributed_delivered += 1
        except Exception as e:
            self.logger.warning(f"Failed to broadcast to local role {role}: {e}")
    
    def get_metrics(self) -> BroadcastMetrics:
        """Get current broadcast metrics."""
        return self.metrics
    
    def reset_metrics(self) -> None:
        """Reset broadcast metrics."""
        self.metrics = BroadcastMetrics()
    
    def get_registered_filters(self) -> Dict[str, BroadcastFilter]:
        """Get all registered filters."""
        return self.broadcast_filters.copy()


# Convenience functions for common broadcast patterns

async def broadcast_to_all(manager: BroadcastManager, message: Any, 
                          connections: Dict[str, Any]) -> int:
    """Broadcast message to all connected clients."""
    return await manager.broadcast(message, BroadcastScope.ALL, connections)


async def broadcast_to_authenticated(manager: BroadcastManager, message: Any,
                                   connections: Dict[str, Any]) -> int:
    """Broadcast message to all authenticated clients."""
    return await manager.broadcast(message, BroadcastScope.AUTHENTICATED, connections)


async def broadcast_to_session(manager: BroadcastManager, message: Any,
                              connections: Dict[str, Any], session_id: str) -> int:
    """Broadcast message to all connections in a specific session."""
    return await manager.broadcast(
        message, BroadcastScope.SESSION, connections,
        filter_criteria={'session_id': session_id}
    )


async def broadcast_to_user(manager: BroadcastManager, message: Any,
                           connections: Dict[str, Any], user_id: str) -> int:
    """Broadcast message to all connections for a specific user."""
    return await manager.broadcast(
        message, BroadcastScope.USER, connections,
        filter_criteria={'user_id': user_id}
    )


async def broadcast_to_role(manager: BroadcastManager, message: Any,
                           connections: Dict[str, Any], role: str) -> int:
    """Broadcast message to all users with a specific role."""
    return await manager.broadcast(
        message, BroadcastScope.ROLE, connections,
        filter_criteria={'role': role}
    )