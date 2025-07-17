"""
Session Memory Integration
Author: Drmusab
Last Modified: 2025-07-17 18:00:00 UTC

This module provides integration between the session management system
and the memory system, allowing sessions to utilize memory capabilities
for contextual continuity, user history preservation, and session context.
"""

from typing import Dict, List, Optional, Any, Set, Tuple
import asyncio
import logging
from datetime import datetime, timezone

from src.core.dependency_injection import Container
from src.assistant.session_manager import SessionInfo, SessionContext, SessionError
from src.memory.core_memory.memory_manager import MemoryManager
from src.memory.core_memory.base_memory import (
    MemoryItem, MemoryType, MemoryMetadata, MemoryRetentionPolicy,
    MemoryAccess, MemorySensitivity
)
from src.memory.operations.retrieval import (
    RetrievalRequest, RetrievalStrategy, MemoryRetrievalMode, MemoryRetriever
)
from src.memory.operations.context_manager import (
    MemoryContextManager, ContextElement, ContextType, ContextPriority
)
from src.memory.operations.consolidation import (
    MemoryConsolidator, ConsolidationJob, ConsolidationStrategy, ConsolidationLevel
)
from src.observability.logging.config import get_logger
from src.core.events.event_bus import EventBus
from src.core.events.event_types import (
    SessionStarted, SessionEnded, SessionPaused, SessionResumed,
    MemoryItemStored, MemoryConsolidationCompleted
)


class SessionMemoryIntegrator:
    """
    Integrates session management with the memory system.
    
    This class provides the bridge between the session management system
    and the memory system, allowing sessions to store and retrieve memories,
    maintain context across interactions, and persist important information.
    """
    
    def __init__(self, container: Container):
        """
        Initialize the session memory integrator.
        
        Args:
            container: Dependency injection container
        """
        self.container = container
        self.memory_manager = container.resolve(MemoryManager)
        self.memory_retriever = container.resolve(MemoryRetriever)
        self.context_manager = container.resolve(MemoryContextManager)
        self.consolidator = container.resolve(MemoryConsolidator)
        self.event_bus = container.resolve(EventBus)
        self.logger = get_logger(__name__)
        
        # Cache to store session-to-memory mappings for quick lookups
        self._session_memory_cache: Dict[str, Set[str]] = {}
        self._cache_lock = asyncio.Lock()
        
        # Register event handlers
        self._register_event_handlers()
    
    def _register_event_handlers(self) -> None:
        """Register event handlers for memory-related events."""
        self.event_bus.subscribe(SessionStarted, self._handle_session_started)
        self.event_bus.subscribe(SessionEnded, self._handle_session_ended)
        self.event_bus.subscribe(SessionPaused, self._handle_session_paused)
        self.event_bus.subscribe(SessionResumed, self._handle_session_resumed)
        self.event_bus.subscribe(MemoryItemStored, self._handle_memory_stored)
        self.event_bus.subscribe(MemoryConsolidationCompleted, self._handle_consolidation_completed)
    
    async def initialize_session_memory(self, session_info: SessionInfo) -> None:
        """
        Initialize memory structures for a new session.
        
        Args:
            session_info: Session information
        """
        try:
            session_id = session_info.session_id
            user_id = session_info.context.user_id
            
            # Initialize working memory for the session
            await self.memory_manager.get_memory_type(MemoryType.WORKING).initialize_session(session_id)
            
            # Initialize context window for the session
            context_id = await self.context_manager.initialize_context(session_id)
            
            # Store initial session metadata in memory
            session_metadata = {
                "session_id": session_id,
                "user_id": user_id,
                "created_at": session_info.created_at.isoformat(),
                "session_type": session_info.config.session_type.value,
                "device_info": session_info.context.device_info,
                "initial_context": self._extract_initial_context(session_info.context)
            }
            
            # Store session start memory
            memory_id = await self.memory_manager.store_memory(
                data=session_metadata,
                memory_type=MemoryType.EPISODIC,
                owner_id=user_id,
                session_id=session_id,
                context_id=context_id,
                tags={"session_start", "metadata", f"session:{session_id}"},
                retention_policy=MemoryRetentionPolicy.STANDARD
            )
            
            # Initialize session memory cache
            async with self._cache_lock:
                self._session_memory_cache[session_id] = {memory_id}
                
            self.logger.info(f"Initialized memory for session {session_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize session memory: {str(e)}")
            raise SessionError(f"Memory initialization failed: {str(e)}", session_id=session_info.session_id)
    
    async def store_interaction(
        self, 
        session_id: str, 
        interaction_data: Dict[str, Any],
        interaction_type: str = "message"
    ) -> str:
        """
        Store an interaction in session memory.
        
        Args:
            session_id: Session identifier
            interaction_data: Interaction data to store
            interaction_type: Type of interaction
            
        Returns:
            Memory ID of the stored interaction
        """
        try:
            # Add timestamp if not present
            if "timestamp" not in interaction_data:
                interaction_data["timestamp"] = datetime.now(timezone.utc).isoformat()
            
            # Add interaction type
            interaction_data["interaction_type"] = interaction_type
            
            # Store in working memory first (for quick access)
            working_memory_id = await self.memory_manager.store_memory(
                data=interaction_data,
                memory_type=MemoryType.WORKING,
                session_id=session_id,
                tags={interaction_type, f"session:{session_id}"}
            )
            
            # Also store in episodic memory for long-term persistence
            episodic_memory_id = await self.memory_manager.store_memory(
                data=interaction_data,
                memory_type=MemoryType.EPISODIC,
                session_id=session_id,
                tags={interaction_type, f"session:{session_id}"}
            )
            
            # Update context with relevant interaction information
            await self._update_context_from_interaction(session_id, interaction_data, interaction_type)
            
            # Add to session memory cache
            async with self._cache_lock:
                if session_id in self._session_memory_cache:
                    self._session_memory_cache[session_id].add(working_memory_id)
                    self._session_memory_cache[session_id].add(episodic_memory_id)
                else:
                    self._session_memory_cache[session_id] = {working_memory_id, episodic_memory_id}
            
            return episodic_memory_id
        
        except Exception as e:
            self.logger.error(f"Failed to store interaction in memory: {str(e)}")
            # Return empty string instead of raising to avoid disrupting interaction flow
            return ""
    
    async def retrieve_session_context(self, session_id: str) -> Dict[str, Any]:
        """
        Retrieve the current context for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session context data
        """
        try:
            # Get context window
            return await self.context_manager.get_context_dict(session_id)
        except Exception as e:
            self.logger.error(f"Failed to retrieve session context: {str(e)}")
            return {}
    
    async def retrieve_session_history(
        self, 
        session_id: str, 
        limit: int = 10,
        interaction_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve interaction history for a session.
        
        Args:
            session_id: Session identifier
            limit: Maximum number of interactions to retrieve
            interaction_type: Optional filter by interaction type
            
        Returns:
            List of interaction data
        """
        try:
            # Create retrieval request
            request = RetrievalRequest(
                query=f"session:{session_id}",
                session_id=session_id,
                memory_types=[MemoryType.WORKING, MemoryType.EPISODIC],
                strategy=RetrievalStrategy.TEMPORAL,
                mode=MemoryRetrievalMode.CONTEXTUAL,
                max_results=limit,
                tags={f"session:{session_id}"}
            )
            
            if interaction_type:
                request.tags.add(interaction_type)
            
            # Retrieve memories
            result = await self.memory_retriever.retrieve(request)
            
            # Extract content
            return [memory.content for memory in result.memories]
        
        except Exception as e:
            self.logger.error(f"Failed to retrieve session history: {str(e)}")
            return []
    
    async def retrieve_relevant_memories(
        self, 
        session_id: str, 
        query: str,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Retrieve memories relevant to a query within the session context.
        
        Args:
            session_id: Session identifier
            query: Query string
            limit: Maximum number of memories to retrieve
            
        Returns:
            List of relevant memory contents
        """
        try:
            # Create retrieval request
            request = RetrievalRequest(
                query=query,
                session_id=session_id,
                memory_types=[MemoryType.EPISODIC, MemoryType.SEMANTIC],
                strategy=RetrievalStrategy.SEMANTIC,
                mode=MemoryRetrievalMode.BALANCED,
                max_results=limit
            )
            
            # Retrieve memories
            result = await self.memory_retriever.retrieve(request)
            
            # Return memory contents
            return [memory.content for memory in result.memories]
        
        except Exception as e:
            self.logger.error(f"Failed to retrieve relevant memories: {str(e)}")
            return []
    
    async def consolidate_session_memories(self, session_id: str) -> None:
        """
        Consolidate memories from a session when it ends.
        
        This process extracts important information from session interactions
        and stores it in long-term memory structures.
        
        Args:
            session_id: Session identifier
        """
        try:
            self.logger.info(f"Consolidating memories for session {session_id}")
            
            # Get memory IDs associated with this session
            memory_ids = []
            async with self._cache_lock:
                if session_id in self._session_memory_cache:
                    memory_ids = list(self._session_memory_cache[session_id])
            
            if not memory_ids:
                # Try to find memories for this session if cache is empty
                request = RetrievalRequest(
                    query=f"session:{session_id}",
                    session_id=session_id,
                    memory_types=[MemoryType.WORKING, MemoryType.EPISODIC],
                    strategy=RetrievalStrategy.RECENCY,
                    max_results=100,
                    tags={f"session:{session_id}"}
                )
                result = await self.memory_retriever.retrieve(request)
                memory_ids = [memory.memory_id for memory in result.memories]
            
            if not memory_ids:
                self.logger.warning(f"No memories found to consolidate for session {session_id}")
                return
            
            # Create a consolidation job
            job = ConsolidationJob(
                source_type=MemoryType.WORKING,
                memory_ids=memory_ids,
                session_id=session_id,
                target_types=[MemoryType.EPISODIC, MemoryType.SEMANTIC],
                strategy=ConsolidationStrategy.SESSION_BASED,
                level=ConsolidationLevel.STANDARD
            )
            
            # Schedule the consolidation job
            job_id = await self.consolidator.consolidate_memories(job)
            self.logger.info(f"Scheduled consolidation job {job_id} for session {session_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to consolidate session memories: {str(e)}")
    
    async def cleanup_session_memory(self, session_id: str) -> None:
        """
        Clean up temporary memory structures when a session ends.
        
        Args:
            session_id: Session identifier
        """
        try:
            # Clean up working memory for the session
            working_memory = self.memory_manager.get_memory_type(MemoryType.WORKING)
            await working_memory.cleanup_session(session_id)
            
            # Clean up context window
            await self.context_manager.cleanup_session(session_id)
            
            # Remove from cache
            async with self._cache_lock:
                self._session_memory_cache.pop(session_id, None)
                
            self.logger.info(f"Cleaned up memory for session {session_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to clean up session memory: {str(e)}")
    
    async def _update_context_from_interaction(
        self, 
        session_id: str, 
        interaction_data: Dict[str, Any],
        interaction_type: str
    ) -> None:
        """
        Update session context based on an interaction.
        
        Args:
            session_id: Session identifier
            interaction_data: Interaction data
            interaction_type: Type of interaction
        """
        try:
            # Extract text content for context update
            text_content = ""
            if "message" in interaction_data:
                text_content = interaction_data["message"]
            elif "content" in interaction_data:
                text_content = interaction_data["content"]
            elif "text" in interaction_data:
                text_content = interaction_data["text"]
            
            if text_content:
                # Update context with this text
                await self.context_manager.update_context_from_text(
                    session_id=session_id,
                    text=text_content,
                    source=interaction_type
                )
            
            # Add specific context elements based on interaction type
            if interaction_type == "user_message":
                # Add as conversation context with high priority
                element = ContextElement(
                    element_id=f"msg_{datetime.now(timezone.utc).timestamp()}",
                    content=interaction_data,
                    context_type=ContextType.CONVERSATION,
                    priority=ContextPriority.HIGH
                )
                await self.context_manager.add_context_element(session_id, element)
            
            elif interaction_type == "system_message":
                # Add as conversation context with medium priority
                element = ContextElement(
                    element_id=f"sys_{datetime.now(timezone.utc).timestamp()}",
                    content=interaction_data,
                    context_type=ContextType.CONVERSATION,
                    priority=ContextPriority.MEDIUM
                )
                await self.context_manager.add_context_element(session_id, element)
                
        except Exception as e:
            self.logger.error(f"Failed to update context from interaction: {str(e)}")
    
    def _extract_initial_context(self, context: SessionContext) -> Dict[str, Any]:
        """
        Extract relevant initial context from session context.
        
        Args:
            context: Session context
            
        Returns:
            Initial context dictionary
        """
        return {
            "user_profile": context.user_profile,
            "user_preferences": context.user_preferences,
            "device_info": context.device_info,
            "location_info": context.location_info,
            "timezone_info": context.timezone_info,
            "feature_flags": context.feature_flags
        }
    
    async def _handle_session_started(self, event: SessionStarted) -> None:
        """
        Handle session started event.
        
        Args:
            event: Session started event
        """
        # This is handled directly by initialize_session_memory
        pass
    
    async def _handle_session_ended(self, event: SessionEnded) -> None:
        """
        Handle session ended event.
        
        Args:
            event: Session ended event
        """
        session_id = event.session_id
        
        # Store session end memory
        await self.memory_manager.store_memory(
            data={
                "session_id": session_id,
                "end_time": datetime.now(timezone.utc).isoformat(),
                "duration_seconds": event.session_duration,
                "end_reason": event.end_reason
            },
            memory_type=MemoryType.EPISODIC,
            session_id=session_id,
            tags={"session_end", f"session:{session_id}"}
        )
        
        # Consolidate memories
        await self.consolidate_session_memories(session_id)
        
        # Cleanup temporary memory
        await self.cleanup_session_memory(session_id)
    
    async def _handle_session_paused(self, event: SessionPaused) -> None:
        """
        Handle session paused event.
        
        Args:
            event: Session paused event
        """
        session_id = event.session_id
        
        # Store pause state in memory
        await self.memory_manager.store_memory(
            data={
                "session_id": session_id,
                "pause_time": datetime.now(timezone.utc).isoformat(),
                "pause_reason": event.reason
            },
            memory_type=MemoryType.WORKING,
            session_id=session_id,
            tags={"session_pause", f"session:{session_id}"}
        )
    
    async def _handle_session_resumed(self, event: SessionResumed) -> None:
        """
        Handle session resumed event.
        
        Args:
            event: Session resumed event
        """
        session_id = event.session_id
        
        # Store resume state in memory
        await self.memory_manager.store_memory(
            data={
                "session_id": session_id,
                "resume_time": datetime.now(timezone.utc).isoformat(),
                "pause_duration": event.pause_duration
            },
            memory_type=MemoryType.WORKING,
            session_id=session_id,
            tags={"session_resume", f"session:{session_id}"}
        )
    
    async def _handle_memory_stored(self, event: MemoryItemStored) -> None:
        """
        Handle memory stored event.
        
        Args:
            event: Memory stored event
        """
        memory_id = event.memory_id
        session_id = event.session_id
        
        if session_id:
            # Add to session memory cache
            async with self._cache_lock:
                if session_id in self._session_memory_cache:
                    self._session_memory_cache[session_id].add(memory_id)
                else:
                    self._session_memory_cache[session_id] = {memory_id}
    
    async def _handle_consolidation_completed(self, event: MemoryConsolidationCompleted) -> None:
        """
        Handle memory consolidation completed event.
        
        Args:
            event: Memory consolidation completed event
        """
        # No specific action needed here, but could be used for notifications or analytics
        pass
