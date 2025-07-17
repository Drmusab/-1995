"""
Enhanced Session Manager with Memory Integration
Author: Drmusab
Last Modified: 2025-07-17 18:15:00 UTC

This module enhances the session manager with memory integration,
allowing sessions to have persistent memory across interactions.
"""

from typing import Dict, List, Optional, Any, Set, Tuple
import asyncio
import logging
from datetime import datetime, timezone

from src.core.dependency_injection import Container
from src.assistant.session_manager import (
    EnhancedSessionManager, SessionInfo, SessionContext, SessionState, 
    SessionConfiguration, SessionError
)
from src.assistant.session_memory_integrator import SessionMemoryIntegrator
from src.memory.core_memory.base_memory import MemoryType
from src.observability.logging.config import get_logger
from src.core.events.event_bus import EventBus
from src.core.events.event_types import SessionStarted, SessionEnded


class MemoryEnabledSessionManager(EnhancedSessionManager):
    """
    Enhanced session manager with memory system integration.
    
    This class extends the EnhancedSessionManager to integrate with the
    memory system, allowing sessions to store and retrieve memories,
    maintain context across interactions, and persist important information.
    """
    
    def __init__(self, container: Container):
        """
        Initialize the memory-enabled session manager.
        
        Args:
            container: Dependency injection container
        """
        super().__init__(container)
        self.logger = get_logger(__name__)
        
        # Create the session memory integrator
        self.memory_integrator = SessionMemoryIntegrator(container)
        
        self.logger.info("Memory-enabled session manager initialized")
    
    async def create_session(
        self, 
        user_id: Optional[str] = None, 
        config: Optional[SessionConfiguration] = None,
        initial_context: Optional[Dict[str, Any]] = None
    ) -> SessionInfo:
        """
        Create a new session with memory integration.
        
        Args:
            user_id: Optional user identifier
            config: Optional session configuration
            initial_context: Optional initial context
            
        Returns:
            Session information
        """
        # Create the session using the parent method
        session_info = await super().create_session(user_id, config, initial_context)
        
        try:
            # Initialize memory for the session
            await self.memory_integrator.initialize_session_memory(session_info)
            
            # Record session creation in session context
            if not session_info.context.custom_data:
                session_info.context.custom_data = {}
            
            session_info.context.custom_data["memory_initialized"] = True
            session_info.context.custom_data["memory_initialized_at"] = datetime.now(timezone.utc).isoformat()
            
            return session_info
            
        except Exception as e:
            self.logger.error(f"Failed to initialize session memory: {str(e)}")
            # Continue with session creation even if memory initialization fails
            # to avoid disrupting the user experience
            return session_info
    
    async def add_interaction(
        self, 
        session_id: str, 
        interaction_data: Dict[str, Any],
        interaction_type: str = "message"
    ) -> str:
        """
        Add an interaction to a session with memory integration.
        
        Args:
            session_id: Session identifier
            interaction_data: Interaction data
            interaction_type: Type of interaction
            
        Returns:
            Interaction ID
        """
        # Add interaction using parent method
        interaction_id = await super().add_interaction(session_id, interaction_data, interaction_type)
        
        try:
            # Store interaction in memory
            memory_id = await self.memory_integrator.store_interaction(
                session_id, 
                interaction_data, 
                interaction_type
            )
            
            # Update the session's last activity
            session_info = await self.get_session(session_id)
            if session_info:
                await self._update_last_activity(session_info)
            
            return interaction_id
            
        except Exception as e:
            self.logger.error(f"Failed to store interaction in memory: {str(e)}")
            # Return the original interaction ID even if memory storage fails
            return interaction_id
    
    async def update_session_context(
        self, 
        session_id: str, 
        context_updates: Dict[str, Any],
        merge: bool = True
    ) -> bool:
        """
        Update session context with memory integration.
        
        Args:
            session_id: Session identifier
            context_updates: Context updates
            merge: Whether to merge or replace context
            
        Returns:
            True if successful
        """
        # Update context using parent method
        result = await super().update_session_context(session_id, context_updates, merge)
        
        if result:
            try:
                # Get the updated session
                session_info = await self.get_session(session_id)
                if session_info:
                    # Update memory context based on session context
                    context_element_data = {
                        "context_update": context_updates,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "merge_mode": "merge" if merge else "replace"
                    }
                    
                    # Store as a memory item
                    await self.memory_integrator.store_interaction(
                        session_id,
                        context_element_data,
                        "context_update"
                    )
            except Exception as e:
                self.logger.error(f"Failed to update memory context: {str(e)}")
        
        return result
    
    async def end_session(
        self, 
        session_id: str, 
        reason: str = "user_request"
    ) -> bool:
        """
        End a session with memory consolidation.
        
        Args:
            session_id: Session identifier
            reason: Reason for ending session
            
        Returns:
            True if successful
        """
        try:
            # Get session before ending it
            session_info = await self.get_session(session_id)
            
            if not session_info:
                return False
            
            # Calculate session duration
            session_duration = 0.0
            if session_info.started_at:
                duration = datetime.now(timezone.utc) - session_info.started_at
                session_duration = duration.total_seconds()
            
            # End session using parent method
            result = await super().end_session(session_id, reason)
            
            if result:
                # Publish session ended event with duration
                self.event_bus.publish(SessionEnded(
                    session_id=session_id,
                    user_id=session_info.context.user_id,
                    end_reason=reason,
                    session_duration=session_duration
                ))
                
                # Memory consolidation will be triggered by the event handler
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error ending session with memory consolidation: {str(e)}")
            # Try to end the session with the parent method as a fallback
            return await super().end_session(session_id, reason)
    
    async def get_session_memory_context(self, session_id: str) -> Dict[str, Any]:
        """
        Get the memory context for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Memory context data
        """
        try:
            return await self.memory_integrator.retrieve_session_context(session_id)
        except Exception as e:
            self.logger.error(f"Failed to get session memory context: {str(e)}")
            return {}
    
    async def get_session_history(
        self, 
        session_id: str, 
        limit: int = 10,
        interaction_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get interaction history for a session from memory.
        
        Args:
            session_id: Session identifier
            limit: Maximum number of interactions to retrieve
            interaction_type: Optional filter by interaction type
            
        Returns:
            List of interaction data
        """
        try:
            return await self.memory_integrator.retrieve_session_history(
                session_id, 
                limit, 
                interaction_type
            )
        except Exception as e:
            self.logger.error(f"Failed to get session history: {str(e)}")
            return []
    
    async def get_relevant_memories(
        self, 
        session_id: str, 
        query: str,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Get memories relevant to a query for a session.
        
        Args:
            session_id: Session identifier
            query: Query string
            limit: Maximum number of memories to retrieve
            
        Returns:
            List of relevant memory data
        """
        try:
            return await self.memory_integrator.retrieve_relevant_memories(
                session_id,
                query,
                limit
            )
        except Exception as e:
            self.logger.error(f"Failed to get relevant memories: {str(e)}")
            return []
    
    async def pause_session(self, session_id: str, reason: str = "user_request") -> None:
        """
        Pause a session with memory state preservation.
        
        Args:
            session_id: Session identifier
            reason: Reason for pausing
        """
        # Pause session using parent method
        await super().pause_session(session_id)
        
        # Store session state in memory
        session_info = await self.get_session(session_id)
        if session_info:
            pause_data = {
                "session_id": session_id,
                "paused_at": datetime.now(timezone.utc).isoformat(),
                "reason": reason,
                "session_state": {
                    "active_workflows": list(session_info.context.active_workflows),
                    "current_topic": session_info.context.current_topic,
                    "interaction_count": session_info.interaction_count
                }
            }
            
            # The event handler will store this in memory
            self.event_bus.publish(SessionPaused(
                session_id=session_id,
                user_id=session_info.context.user_id,
                reason=reason
            ))
    
    async def resume_session(self, session_id: str) -> None:
        """
        Resume a session with memory state restoration.
        
        Args:
            session_id: Session identifier
        """
        session_info = await self.get_session(session_id)
        if not session_info:
            raise SessionError(f"Cannot resume non-existent session: {session_id}")
        
        # Get pause time from session info
        pause_duration = 0.0
        if session_info.state == SessionState.PAUSED and session_info.last_activity:
            duration = datetime.now(timezone.utc) - session_info.last_activity
            pause_duration = duration.total_seconds()
        
        # Resume session using parent method
        await super().resume_session(session_id)
        
        # The event handler will handle memory updates
        self.event_bus.publish(SessionResumed(
            session_id=session_id,
            user_id=session_info.context.user_id,
            pause_duration=pause_duration
        ))
    
    async def _handle_component_health_change(self, event) -> None:
        """
        Handle component health change events.
        
        Args:
            event: Health change event
        """
        await super()._handle_component_health_change(event)
        
        # Additional handling for memory system health changes
        if event.component.startswith("memory."):
            self.logger.warning(f"Memory system health change: {event.component} is {event.status}")
            
            # If the memory system is down, we can still operate in degraded mode
            if event.status == "down":
                self.logger.warning("Memory system is down, operating in degraded mode")
