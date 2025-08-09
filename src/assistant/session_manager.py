"""
Enhanced Session Manager for AI Assistant System

This module manages user sessions, conversation state, and context persistence.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Set
import uuid
import logging

from src.core.dependency_injection import Container
from src.core.events.event_bus import EventBus
from src.core.events.event_types import (
    SessionStarted,
    SessionEnded,
    SessionCleanupStarted
)


@dataclass
class SessionInfo:
    """Information about a user session."""
    session_id: str
    user_id: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_activity: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    context: Dict[str, Any] = field(default_factory=dict)
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    preferences: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_active: bool = True


class EnhancedSessionManager:
    """
    Enhanced session manager that provides comprehensive session management
    with context persistence and automatic cleanup.
    """

    def __init__(self, container: Container):
        """Initialize the session manager."""
        self.container = container
        self.sessions: Dict[str, SessionInfo] = {}
        self.user_sessions: Dict[str, Set[str]] = {}  # user_id -> session_ids
        self.event_bus = container.get(EventBus) if container else None
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.max_session_age = timedelta(hours=24)
        self.max_inactive_time = timedelta(hours=2)
        self.cleanup_interval = timedelta(minutes=30)
        
        # Background task
        self._cleanup_task: Optional[asyncio.Task] = None

    async def initialize(self) -> None:
        """Initialize the session manager."""
        self.logger.info("Initializing Enhanced Session Manager")
        
        # Start background cleanup task
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        self.logger.info("Enhanced Session Manager initialized successfully")

    async def create_session(
        self,
        user_id: Optional[str] = None,
        initial_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a new session.
        
        Args:
            user_id: Optional user identifier
            initial_context: Optional initial context data
            
        Returns:
            Session identifier
        """
        session_id = str(uuid.uuid4())
        
        session_info = SessionInfo(
            session_id=session_id,
            user_id=user_id,
            context=initial_context or {},
        )
        
        self.sessions[session_id] = session_info
        
        # Track user sessions
        if user_id:
            if user_id not in self.user_sessions:
                self.user_sessions[user_id] = set()
            self.user_sessions[user_id].add(session_id)
        
        # Emit event
        if self.event_bus:
            await self.event_bus.emit(
                SessionStarted(
                    session_id=session_id,
                    user_id=user_id
                )
            )
        
        self.logger.info(f"Created session {session_id} for user {user_id}")
        return session_id

    async def get_session(self, session_id: str) -> Optional[SessionInfo]:
        """Get session information."""
        session = self.sessions.get(session_id)
        if session and session.is_active:
            # Update last activity
            session.last_activity = datetime.now(timezone.utc)
            return session
        return None

    async def update_session_context(
        self,
        session_id: str,
        context_update: Dict[str, Any]
    ) -> bool:
        """
        Update session context.
        
        Args:
            session_id: Session identifier
            context_update: Context data to update
            
        Returns:
            True if successful, False if session not found
        """
        session = self.sessions.get(session_id)
        if session and session.is_active:
            session.context.update(context_update)
            session.last_activity = datetime.now(timezone.utc)
            return True
        return False

    async def add_conversation_entry(
        self,
        session_id: str,
        entry: Dict[str, Any]
    ) -> bool:
        """
        Add an entry to the conversation history.
        
        Args:
            session_id: Session identifier
            entry: Conversation entry
            
        Returns:
            True if successful, False if session not found
        """
        session = self.sessions.get(session_id)
        if session and session.is_active:
            entry_with_timestamp = {
                **entry,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            session.conversation_history.append(entry_with_timestamp)
            session.last_activity = datetime.now(timezone.utc)
            
            # Limit conversation history size
            max_history = 100
            if len(session.conversation_history) > max_history:
                session.conversation_history = session.conversation_history[-max_history:]
            
            return True
        return False

    async def end_session(self, session_id: str) -> bool:
        """
        End a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if successful, False if session not found
        """
        session = self.sessions.get(session_id)
        if session:
            session.is_active = False
            
            # Remove from user sessions
            if session.user_id and session.user_id in self.user_sessions:
                self.user_sessions[session.user_id].discard(session_id)
                if not self.user_sessions[session.user_id]:
                    del self.user_sessions[session.user_id]
            
            # Emit event
            if self.event_bus:
                await self.event_bus.emit(
                    SessionEnded(
                        session_id=session_id,
                        user_id=session.user_id,
                        duration=(datetime.now(timezone.utc) - session.created_at).total_seconds()
                    )
                )
            
            self.logger.info(f"Ended session {session_id}")
            return True
        return False

    async def get_user_sessions(self, user_id: str) -> List[SessionInfo]:
        """Get all active sessions for a user."""
        session_ids = self.user_sessions.get(user_id, set())
        sessions = []
        
        for session_id in session_ids:
            session = self.sessions.get(session_id)
            if session and session.is_active:
                sessions.append(session)
        
        return sessions

    def get_session_statistics(self) -> Dict[str, Any]:
        """Get session statistics."""
        active_sessions = [s for s in self.sessions.values() if s.is_active]
        
        return {
            "total_sessions": len(self.sessions),
            "active_sessions": len(active_sessions),
            "unique_users": len(self.user_sessions),
            "average_session_duration": self._calculate_average_session_duration(),
        }

    def _calculate_average_session_duration(self) -> float:
        """Calculate average session duration in seconds."""
        if not self.sessions:
            return 0.0
            
        total_duration = 0.0
        count = 0
        now = datetime.now(timezone.utc)
        
        for session in self.sessions.values():
            if session.is_active:
                duration = (now - session.created_at).total_seconds()
            else:
                duration = (session.last_activity - session.created_at).total_seconds()
            
            total_duration += duration
            count += 1
        
        return total_duration / count if count > 0 else 0.0

    async def _cleanup_loop(self) -> None:
        """Background task for cleaning up expired sessions."""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval.total_seconds())
                await self._cleanup_expired_sessions()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in session cleanup: {e}")

    async def _cleanup_expired_sessions(self) -> None:
        """Clean up expired sessions."""
        now = datetime.now(timezone.utc)
        expired_sessions = []
        
        for session_id, session in self.sessions.items():
            if not session.is_active:
                continue
                
            # Check if session is too old
            if now - session.created_at > self.max_session_age:
                expired_sessions.append(session_id)
                continue
                
            # Check if session is inactive
            if now - session.last_activity > self.max_inactive_time:
                expired_sessions.append(session_id)
        
        if expired_sessions:
            if self.event_bus:
                await self.event_bus.emit(
                    SessionCleanupStarted(count=len(expired_sessions))
                )
            
            for session_id in expired_sessions:
                await self.end_session(session_id)
                del self.sessions[session_id]
            
            self.logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")

    async def cleanup(self) -> None:
        """Cleanup the session manager."""
        self.logger.info("Cleaning up Enhanced Session Manager")
        
        # Cancel cleanup task
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # End all active sessions
        active_session_ids = [
            session_id for session_id, session in self.sessions.items()
            if session.is_active
        ]
        
        for session_id in active_session_ids:
            await self.end_session(session_id)
        
        self.logger.info("Enhanced Session Manager cleanup complete")