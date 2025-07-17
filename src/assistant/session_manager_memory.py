"""
Memory-Enabled Session Manager
Author: Drmusab
Last Modified: 2025-07-17 19:35:00 UTC

This module provides a session manager with memory capabilities, allowing
the assistant to maintain context and recall past interactions throughout
a conversation and across multiple sessions.
"""

import asyncio
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from datetime import datetime, timezone, timedelta
import logging
import uuid
import json

from src.core.dependency_injection import Container
from src.core.events.event_bus import EventBus
from src.core.events.event_types import (
    SessionStarted, SessionEnded, MessageReceived, 
    MessageProcessed, ErrorOccurred
)
from src.assistant.session_memory_integrator import SessionMemoryIntegrator
from src.memory.operations.retrieval import RetrievalRequest, RetrievalStrategy
from src.observability.logging.config import get_logger
from src.observability.monitoring.metrics import MetricsCollector


class Session:
    """Represents a user session with memory capabilities."""
    
    def __init__(
        self,
        session_id: str,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a new session.
        
        Args:
            session_id: Unique session identifier
            user_id: Optional user identifier
            metadata: Optional session metadata
        """
        self.session_id = session_id
        self.user_id = user_id
        self.metadata = metadata or {}
        self.created_at = datetime.now(timezone.utc)
        self.last_active = self.created_at
        self.interaction_count = 0
        self.active = True
        
        # Memory-related fields
        self.context: Dict[str, Any] = {}
        self.memory_ids: List[str] = []
        self.important_facts: List[Dict[str, Any]] = []
    
    def touch(self) -> None:
        """Update last activity time."""
        self.last_active = datetime.now(timezone.utc)
        self.interaction_count += 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary."""
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "last_active": self.last_active.isoformat(),
            "interaction_count": self.interaction_count,
            "active": self.active,
            "context": self.context,
            "memory_count": len(self.memory_ids)
        }


class MemoryEnabledSessionManager:
    """
    Session manager with memory integration for maintaining conversational
    context and recalling past interactions.
    """
    
    def __init__(self, container: Container):
        """
        Initialize the memory-enabled session manager.
        
        Args:
            container: The dependency injection container
        """
        self.container = container
        self.logger = get_logger(__name__)
        
        # Core components
        self.event_bus = container.get(EventBus)
        
        # Memory integration
        self.memory_integrator = container.get(SessionMemoryIntegrator)
        
        # Active sessions
        self.sessions: Dict[str, Session] = {}
        
        # Optional components
        try:
            self.metrics = container.get(MetricsCollector)
            self.metrics.register_counter("session_total", 
                                         "Total number of sessions")
            self.metrics.register_gauge("session_active", 
                                       "Number of active sessions")
            self.metrics.register_histogram("session_duration_seconds", 
                                           "Session duration in seconds")
        except Exception:
            self.logger.warning("Metrics collector not available")
            self.metrics = None
        
        # Register event handlers
        self._register_event_handlers()
        
        # Start background tasks
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        self.logger.info("Memory-enabled session manager initialized")
    
    def _register_event_handlers(self) -> None:
        """Register event handlers."""
        self.event_bus.subscribe(MessageReceived, self._handle_message_received)
    
    async def _handle_message_received(self, event: MessageReceived) -> None:
        """
        Handle incoming message by ensuring session exists.
        
        Args:
            event: The message received event
        """
        session_id = event.session_id
        user_id = event.user_id
        
        # Create or get session
        if session_id not in self.sessions:
            await self.create_session(session_id, user_id)
        else:
            # Update last activity
            self.sessions[session_id].touch()
    
    async def create_session(
        self, 
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a new session with memory capabilities.
        
        Args:
            session_id: Optional session ID (generated if not provided)
            user_id: Optional user ID
            metadata: Optional session metadata
            
        Returns:
            Session ID
        """
        # Generate session ID if not provided
        if not session_id:
            session_id = str(uuid.uuid4())
        
        # Create session object
        session = Session(
            session_id=session_id,
            user_id=user_id,
            metadata=metadata
        )
        
        # Store in active sessions
        self.sessions[session_id] = session
        
        # Emit session started event
        await self.event_bus.emit(SessionStarted(
            session_id=session_id,
            user_id=user_id
        ))
        
        # Update metrics
        if self.metrics:
            self.metrics.increment("session_total")
            self.metrics.gauge("session_active", len(self.sessions))
        
        self.logger.info(f"Created session {session_id} for user {user_id}")
        
        return session_id
    
    async def end_session(self, session_id: str) -> bool:
        """
        End a session and consolidate its memories.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if session was ended, False otherwise
        """
        if session_id not in self.sessions:
            return False
        
        session = self.sessions[session_id]
        session.active = False
        
        # Calculate session duration
        duration = (datetime.now(timezone.utc) - session.created_at).total_seconds()
        
        # Emit session ended event
        await self.event_bus.emit(SessionEnded(
            session_id=session_id,
            user_id=session.user_id,
            duration=duration
        ))
        
        # Remove from active sessions
        del self.sessions[session_id]
        
        # Update metrics
        if self.metrics:
            self.metrics.gauge("session_active", len(self.sessions))
            self.metrics.record("session_duration_seconds", duration)
        
        self.logger.info(f"Ended session {session_id} after {duration:.1f} seconds")
        
        return True
    
    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get session information including memory context.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session information or None if not found
        """
        if session_id not in self.sessions:
            return None
        
        session = self.sessions[session_id]
        
        # Get basic session info
        session_dict = session.to_dict()
        
        # Get memory context
        try:
            context = await self.memory_integrator.get_session_context(session_id)
            session_dict["context"] = context
        except Exception as e:
            self.logger.error(f"Error getting session context: {str(e)}")
            session_dict["context_error"] = str(e)
        
        return session_dict
    
    async def process_message(
        self,
        session_id: str,
        message: str,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a message with memory-enhanced context.
        
        Args:
            session_id: Session identifier
            message: User message
            user_id: Optional user identifier
            metadata: Optional message metadata
            
        Returns:
            Response with context information
        """
        # Ensure session exists
        if session_id not in self.sessions:
            await self.create_session(session_id, user_id)
        
        session = self.sessions[session_id]
        session.touch()
        
        # Get memory context
        context = await self.memory_integrator.get_session_context(session_id)
        
        # TODO: Process message with context using the core engine
        # This is a placeholder for demonstration
        response = {
            "text": f"Processed message with memory context: {message}",
            "session_id": session_id,
            "context_used": True,
            "memory_context_size": len(json.dumps(context))
        }
        
        # Emit message processed event
        await self.event_bus.emit(MessageProcessed(
            session_id=session_id,
            user_id=user_id,
            message=message,
            response=response["text"],
            metadata=metadata
        ))
        
        return response
    
    async def store_session_fact(
        self,
        session_id: str,
        fact: str,
        importance: float = 0.7,
        tags: Optional[Set[str]] = None
    ) -> str:
        """
        Store an important fact learned in the session.
        
        Args:
            session_id: Session identifier
            fact: The fact to store
            importance: Importance score (0-1)
            tags: Optional tags
            
        Returns:
            Memory ID of the stored fact
        """
        if session_id not in self.sessions:
            self.logger.warning(f"Attempting to store fact for unknown session {session_id}")
            return ""
        
        session = self.sessions[session_id]
        
        # Store the fact
        memory_id = await self.memory_integrator.store_session_fact(
            session_id=session_id,
            user_id=session.user_id,
            fact=fact,
            importance=importance,
            tags=tags
        )
        
        # Track in session
        if memory_id:
            session.memory_ids.append(memory_id)
            session.important_facts.append({
                "memory_id": memory_id,
                "fact": fact,
                "importance": importance,
                "tags": list(tags or set())
            })
        
        return memory_id
    
    async def get_session_memories(
        self,
        session_id: str,
        query: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get memories for a specific session.
        
        Args:
            session_id: Session identifier
            query: Optional search query
            limit: Maximum number of memories to retrieve
            
        Returns:
            List of memory items
        """
        if session_id not in self.sessions:
            self.logger.warning(f"Attempting to get memories for unknown session {session_id}")
            return []
        
        return await self.memory_integrator.retrieve_session_memories(
            session_id=session_id,
            query=query,
            limit=limit
        )
    
    async def _cleanup_loop(self) -> None:
        """Background task to clean up inactive sessions."""
        try:
            while True:
                # Run every 5 minutes
                await asyncio.sleep(300)
                
                # Find inactive sessions (no activity for 30 minutes)
                now = datetime.now(timezone.utc)
                inactive_sessions = [
                    session_id for session_id, session in self.sessions.items()
                    if (now - session.last_active) > timedelta(minutes=30)
                ]
                
                # End inactive sessions
                for session_id in inactive_sessions:
                    await self.end_session(session_id)
                
                if inactive_sessions:
                    self.logger.info(f"Cleaned up {len(inactive_sessions)} inactive sessions")
                
        except asyncio.CancelledError:
            self.logger.info("Session cleanup task cancelled")
        except Exception as e:
            self.logger.error(f"Error in session cleanup: {str(e)}")
    
    async def shutdown(self) -> None:
        """Shutdown the session manager."""
        # Cancel cleanup task
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # End all active sessions
        for session_id in list(self.sessions.keys()):
            await self.end_session(session_id)
        
        self.logger.info("Memory-enabled session manager shutdown")
