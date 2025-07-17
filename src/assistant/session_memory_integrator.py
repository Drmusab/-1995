"""
Session Memory Integrator
Author: Drmusab
Last Modified: 2025-07-17 19:30:00 UTC

This module provides the integration layer between the assistant's session management
and the memory system, ensuring proper persistence and retrieval of session data.
"""

import asyncio
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from datetime import datetime, timezone
import logging
import uuid

from src.core.dependency_injection import Container
from src.core.events.event_bus import EventBus
from src.core.events.event_types import (
    SessionStarted, SessionEnded, MessageProcessed, 
    MemoryRetrievalRequested, MemoryItemStored
)
from src.memory.core_memory.base_memory import (
    MemoryItem, MemoryType, MemoryMetadata, MemoryRetentionPolicy,
    MemoryAccess, MemorySensitivity
)
from src.memory.core_memory.memory_manager import MemoryManager
from src.memory.operations.retrieval import (
    MemoryRetriever, RetrievalRequest, RetrievalResult,
    RetrievalStrategy, MemoryRetrievalMode
)
from src.memory.operations.context_manager import (
    MemoryContextManager, ContextType, ContextPriority, ContextElement
)
from src.memory.operations.consolidation import (
    MemoryConsolidator, ConsolidationJob, ConsolidationStrategy, ConsolidationLevel
)
from src.memory.cache_manager import MemoryCacheManager, CacheKey
from src.observability.logging.config import get_logger
from src.observability.monitoring.metrics import MetricsCollector
from src.observability.monitoring.tracing import TraceManager


class SessionMemoryIntegrator:
    """
    Integrates the memory system with session management to provide
    persistent memory across user sessions and interactions.
    """
    
    def __init__(self, container: Container):
        """
        Initialize the session memory integrator.
        
        Args:
            container: The dependency injection container
        """
        self.container = container
        self.logger = get_logger(__name__)
        
        # Core components
        self.memory_manager = container.get(MemoryManager)
        self.memory_retriever = container.get(MemoryRetriever)
        self.context_manager = container.get(MemoryContextManager)
        self.memory_consolidator = container.get(MemoryConsolidator)
        self.cache_manager = container.get(MemoryCacheManager)
        self.event_bus = container.get(EventBus)
        
        # Optional components
        try:
            self.metrics = container.get(MetricsCollector)
            self.metrics.register_counter("session_memory_operations_total", 
                                         "Total number of session memory operations")
            self.metrics.register_histogram("session_memory_retrieval_time", 
                                           "Time to retrieve session memories")
        except Exception:
            self.logger.warning("Metrics collector not available")
            self.metrics = None
            
        try:
            self.tracer = container.get(TraceManager)
        except Exception:
            self.logger.warning("Trace manager not available")
            self.tracer = None
        
        # Register event handlers
        self._register_event_handlers()
        
        self.logger.info("Session memory integrator initialized")
    
    def _register_event_handlers(self) -> None:
        """Register event handlers for session events."""
        self.event_bus.subscribe(SessionStarted, self._handle_session_started)
        self.event_bus.subscribe(SessionEnded, self._handle_session_ended)
        self.event_bus.subscribe(MessageProcessed, self._handle_message_processed)
    
    async def _handle_session_started(self, event: SessionStarted) -> None:
        """
        Handle session started event by initializing session memory.
        
        Args:
            event: The session started event
        """
        session_id = event.session_id
        user_id = event.user_id
        
        self.logger.info(f"Initializing memory for session {session_id}, user {user_id}")
        
        # Initialize working memory for this session
        await self.memory_manager.initialize_working_memory(session_id, user_id)
        
        # Initialize context window for this session
        await self.context_manager.initialize_context(session_id)
        
        # Retrieve user's relevant episodic memories
        if user_id:
            try:
                # Get recent and important memories for this user
                request = RetrievalRequest(
                    query="",  # Empty query for recent memories
                    user_id=user_id,
                    memory_types=[MemoryType.EPISODIC, MemoryType.SEMANTIC],
                    strategy=RetrievalStrategy.RECENCY,
                    mode=MemoryRetrievalMode.BALANCED,
                    max_results=5
                )
                
                result = await self.memory_retriever.retrieve(request)
                
                # Store relevant memories in session cache for quick access
                if result.items:
                    await self.cache_manager.cache_session_data(
                        session_id=session_id,
                        data=result.memories,
                        subtype="recent_memories"
                    )
                    
                    # Add important memories to context
                    for memory, score in result.items[:3]:  # Only add top 3 to context
                        await self.context_manager.add_context_element(
                            session_id=session_id,
                            content=memory.content,
                            context_type=ContextType.REFERENCE,
                            priority=ContextPriority.MEDIUM,
                            source="past_memory",
                            memory_id=memory.memory_id,
                            metadata={"memory_type": memory.memory_type.value}
                        )
                
                self.logger.info(f"Retrieved {len(result.items)} past memories for user {user_id}")
                
            except Exception as e:
                self.logger.error(f"Error retrieving past memories for user {user_id}: {str(e)}")
        
        # Record metric
        if self.metrics:
            self.metrics.increment("session_memory_operations_total", 
                                  tags={"operation": "session_start"})
    
    async def _handle_session_ended(self, event: SessionEnded) -> None:
        """
        Handle session ended event by consolidating session memories.
        
        Args:
            event: The session ended event
        """
        session_id = event.session_id
        user_id = event.user_id
        
        self.logger.info(f"Consolidating memories for ended session {session_id}")
        
        try:
            # Create a consolidation job for this session
            job = ConsolidationJob(
                source_type=MemoryType.WORKING,
                session_id=session_id,
                user_id=user_id,
                target_types=[MemoryType.EPISODIC, MemoryType.SEMANTIC],
                strategy=ConsolidationStrategy.SESSION_BASED,
                level=ConsolidationLevel.STANDARD
            )
            
            # Schedule the job
            job_id = await self.memory_consolidator.consolidate_session(
                session_id=session_id,
                user_id=user_id
            )
            
            self.logger.info(f"Scheduled consolidation job {job_id} for session {session_id}")
            
            # Clean up context
            await self.context_manager.cleanup_session(session_id)
            
            # Clean up session cache
            await self.cache_manager.invalidate_session_data(session_id)
            
            # Record metric
            if self.metrics:
                self.metrics.increment("session_memory_operations_total", 
                                      tags={"operation": "session_end"})
        
        except Exception as e:
            self.logger.error(f"Error consolidating memories for session {session_id}: {str(e)}")
    
    async def _handle_message_processed(self, event: MessageProcessed) -> None:
        """
        Handle message processed event by storing in working memory.
        
        Args:
            event: The message processed event
        """
        session_id = event.session_id
        user_id = event.user_id
        message = event.message
        response = event.response
        
        # Store user message in working memory
        try:
            user_message_id = await self.memory_manager.store_memory(
                data=message,
                memory_type=MemoryType.WORKING,
                owner_id=user_id,
                session_id=session_id,
                metadata={
                    "message_type": "user",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            )
            
            # Store assistant response in working memory
            assistant_message_id = await self.memory_manager.store_memory(
                data=response,
                memory_type=MemoryType.WORKING,
                session_id=session_id,
                metadata={
                    "message_type": "assistant",
                    "in_response_to": user_message_id,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            )
            
            # Update context with the conversation
            await self.context_manager.update_context_from_text(
                session_id=session_id,
                text=f"User: {message}\nAssistant: {response}",
                context_type=ContextType.CONVERSATION,
                priority=ContextPriority.HIGH
            )
            
            self.logger.debug(f"Stored conversation in memory for session {session_id}")
            
            # Record metric
            if self.metrics:
                self.metrics.increment("session_memory_operations_total", 
                                      tags={"operation": "message_store"})
                
        except Exception as e:
            self.logger.error(f"Error storing message in memory: {str(e)}")
    
    async def retrieve_session_memories(
        self, 
        session_id: str,
        query: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Retrieve memories for a specific session.
        
        Args:
            session_id: The session identifier
            query: Optional search query
            limit: Maximum number of memories to retrieve
            
        Returns:
            List of memory items as dictionaries
        """
        start_time = asyncio.get_event_loop().time()
        
        try:
            if query:
                # Search for specific memories
                request = RetrievalRequest(
                    query=query,
                    session_id=session_id,
                    memory_types=[MemoryType.WORKING],
                    strategy=RetrievalStrategy.SEMANTIC,
                    max_results=limit
                )
                
                result = await self.memory_retriever.retrieve(request)
                memories = result.memories
            else:
                # Get recent memories for this session
                memories = await self.memory_manager.get_recent_memories(
                    memory_type=MemoryType.WORKING,
                    session_id=session_id,
                    limit=limit
                )
            
            # Convert to dictionaries
            memory_dicts = []
            for memory in memories:
                memory_dict = {
                    "memory_id": memory.memory_id,
                    "content": memory.content,
                    "memory_type": memory.memory_type.value,
                    "created_at": memory.metadata.created_at.isoformat(),
                    "session_id": memory.session_id
                }
                
                # Add message_type if available
                if "message_type" in memory.metadata.custom_metadata:
                    memory_dict["message_type"] = memory.metadata.custom_metadata["message_type"]
                
                memory_dicts.append(memory_dict)
            
            # Record metrics
            end_time = asyncio.get_event_loop().time()
            if self.metrics:
                self.metrics.increment("session_memory_operations_total", 
                                      tags={"operation": "retrieve"})
                self.metrics.record("session_memory_retrieval_time", 
                                   end_time - start_time)
            
            return memory_dicts
            
        except Exception as e:
            self.logger.error(f"Error retrieving session memories: {str(e)}")
            return []
    
    async def store_session_fact(
        self,
        session_id: str,
        user_id: Optional[str],
        fact: str,
        importance: float = 0.7,
        tags: Optional[Set[str]] = None
    ) -> str:
        """
        Store an important fact learned during the session.
        
        Args:
            session_id: The session identifier
            user_id: The user identifier
            fact: The fact to store
            importance: Importance score (0-1)
            tags: Optional tags for the fact
            
        Returns:
            Memory ID of the stored fact
        """
        try:
            # Store fact in semantic memory for future retrieval
            memory_id = await self.memory_manager.store_memory(
                data=fact,
                memory_type=MemoryType.SEMANTIC,
                owner_id=user_id,
                session_id=session_id,
                context_id=session_id,
                metadata={
                    "importance": importance,
                    "tags": list(tags or set()),
                    "fact_type": "session_fact"
                }
            )
            
            # Add to session context if important enough
            if importance >= 0.7:
                await self.context_manager.add_context_element(
                    session_id=session_id,
                    content=fact,
                    context_type=ContextType.FACTUAL,
                    priority=ContextPriority.HIGH if importance > 0.8 else ContextPriority.MEDIUM,
                    source="session_fact",
                    memory_id=memory_id
                )
            
            # Record metric
            if self.metrics:
                self.metrics.increment("session_memory_operations_total", 
                                      tags={"operation": "store_fact"})
            
            return memory_id
            
        except Exception as e:
            self.logger.error(f"Error storing session fact: {str(e)}")
            return ""
    
    async def get_session_context(self, session_id: str) -> Dict[str, Any]:
        """
        Get the current context for a session.
        
        Args:
            session_id: The session identifier
            
        Returns:
            Dictionary with session context
        """
        try:
            # Get context from context manager
            context_dict = await self.context_manager.get_context_dict(session_id)
            
            # Get related memories
            memories = await self.context_manager.get_memories_for_context(session_id)
            
            # Combine into a comprehensive context object
            result = {
                "session_id": session_id,
                "context": context_dict,
                "related_memories": [
                    {
                        "memory_id": mem.memory_id,
                        "content": mem.content,
                        "memory_type": mem.memory_type.value
                    } 
                    for mem in memories
                ],
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            # Record metric
            if self.metrics:
                self.metrics.increment("session_memory_operations_total", 
                                      tags={"operation": "get_context"})
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error getting session context: {str(e)}")
            return {"session_id": session_id, "error": str(e)}
