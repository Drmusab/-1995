"""
Session Memory Integrator
Author: AI Assistant Contributors  
Last Modified: 2025-07-20

Integrates session management with memory systems to provide context-aware
conversational experiences with persistent memory across sessions.
"""

import asyncio
import uuid
import json
from typing import Any, Dict, List, Optional, Set, Tuple
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from enum import Enum

from src.core.dependency_injection import Container
from src.core.config.loader import ConfigLoader
from src.core.events.event_bus import EventBus
from src.core.events.event_types import (
    SessionStarted, SessionEnded, MessageReceived, MessageProcessed,
    MemoryItemStored, MemoryRetrievalRequested, ErrorOccurred
)
from src.observability.logging.config import get_logger
from src.observability.monitoring.metrics import MetricsCollector


class MemoryIntegrationType(Enum):
    """Types of memory integration."""
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    WORKING = "working"
    CONTEXTUAL = "contextual"


@dataclass
class SessionMemoryContext:
    """Context data for session memory integration."""
    session_id: str
    user_id: Optional[str] = None
    memories: List[Dict[str, Any]] = field(default_factory=list)
    context_elements: List[Dict[str, Any]] = field(default_factory=list)
    facts: List[Dict[str, Any]] = field(default_factory=list)
    preferences: Dict[str, Any] = field(default_factory=dict)
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    memory_retrieval_count: int = 0
    memory_storage_count: int = 0


class SessionMemoryIntegrator:
    """
    Integrates session management with memory systems to provide
    context-aware conversational experiences.
    """
    
    def __init__(self, container: Container):
        """Initialize the session memory integrator."""
        self.container = container
        self.logger = get_logger(__name__)
        
        # Core dependencies
        self.config_loader = container.get(ConfigLoader)
        self.event_bus = container.get(EventBus) 
        
        # Configuration
        self.config = self.config_loader.get('memory.integration', {})
        self.enable_episodic = self.config.get('enable_episodic', True)
        self.enable_semantic = self.config.get('enable_semantic', True)
        self.enable_working = self.config.get('enable_working', True)
        self.max_context_items = self.config.get('max_context_items', 10)
        self.memory_retention_days = self.config.get('memory_retention_days', 30)
        
        # Session contexts cache
        self._session_contexts: Dict[str, SessionMemoryContext] = {}
        
        # Optional components (will be available after initialization)
        self.memory_manager = None
        self.working_memory = None
        self.episodic_memory = None
        self.semantic_memory = None
        
        # Monitoring
        try:
            self.metrics = container.get(MetricsCollector)
            self._setup_metrics()
        except Exception:
            self.metrics = None
            
        self.logger.info("Session Memory Integrator initialized")
    
    def _setup_metrics(self):
        """Set up monitoring metrics."""
        if self.metrics:
            self.metrics.register_counter("session_memory_integrations")
            self.metrics.register_counter("memory_retrievals")
            self.metrics.register_counter("memory_storage_operations")
            self.metrics.register_histogram("memory_retrieval_time")
    
    async def initialize(self):
        """Initialize the session memory integrator."""
        try:
            # Get memory components if available
            try:
                from src.memory.core_memory.memory_manager import MemoryManager
                self.memory_manager = self.container.get(MemoryManager)
                self.logger.info("Memory manager connected")
            except Exception as e:
                self.logger.warning(f"Memory manager not available: {e}")
            
            try:
                from src.memory.core_memory.memory_types import WorkingMemory, EpisodicMemory, SemanticMemory
                self.working_memory = self.container.get(WorkingMemory)
                self.episodic_memory = self.container.get(EpisodicMemory)
                self.semantic_memory = self.container.get(SemanticMemory)
                self.logger.info("Memory components connected")
            except Exception as e:
                self.logger.warning(f"Memory components not available: {e}")
            
            # Register event handlers
            await self._register_event_handlers()
            
            self.logger.info("Session Memory Integrator initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize session memory integrator: {e}")
            raise
    
    async def _register_event_handlers(self):
        """Register event handlers for session and memory events."""
        self.event_bus.subscribe(SessionStarted, self._handle_session_started)
        self.event_bus.subscribe(SessionEnded, self._handle_session_ended)
        self.event_bus.subscribe(MessageReceived, self._handle_message_received)
        self.event_bus.subscribe(MessageProcessed, self._handle_message_processed)
    
    async def _handle_session_started(self, event: SessionStarted):
        """Handle session started event."""
        try:
            session_id = event.session_id
            user_id = getattr(event, 'user_id', None)
            
            # Create session memory context
            context = SessionMemoryContext(
                session_id=session_id,
                user_id=user_id
            )
            
            # Load relevant memories if user is authenticated
            if user_id and self.memory_manager:
                relevant_memories = await self._load_user_memories(user_id)
                context.memories = relevant_memories
            
            # Store context
            self._session_contexts[session_id] = context
            
            if self.metrics:
                self.metrics.increment("session_memory_integrations")
            
            self.logger.info(f"Session memory context created for session {session_id}")
            
        except Exception as e:
            self.logger.error(f"Error handling session started: {e}")
            await self.event_bus.emit(ErrorOccurred(
                error_type="SessionMemoryError",
                error_message=str(e),
                component="session_memory_integrator"
            ))
    
    async def _handle_session_ended(self, event: SessionEnded):
        """Handle session ended event."""
        try:
            session_id = event.session_id
            duration = getattr(event, 'duration', 0)
            summary = getattr(event, 'summary', {})
            
            # Get session context
            context = self._session_contexts.get(session_id)
            if not context:
                return
            
            # Consolidate session memories if enabled
            if self.episodic_memory and context.user_id:
                await self._consolidate_session_memories(context, duration, summary)
            
            # Clean up context
            del self._session_contexts[session_id]
            
            self.logger.info(f"Session memory context cleaned up for session {session_id}")
            
        except Exception as e:
            self.logger.error(f"Error handling session ended: {e}")
    
    async def _handle_message_received(self, event: MessageReceived):
        """Handle message received event."""
        try:
            session_id = event.session_id
            message = getattr(event, 'message', '')
            
            # Update session context with new message
            context = self._session_contexts.get(session_id)
            if context:
                context.context_elements.append({
                    'type': 'user_message',
                    'content': message,
                    'timestamp': datetime.now(timezone.utc).isoformat()
                })
                
                # Keep context size manageable
                if len(context.context_elements) > self.max_context_items:
                    context.context_elements = context.context_elements[-self.max_context_items:]
                
                context.last_updated = datetime.now(timezone.utc)
            
        except Exception as e:
            self.logger.error(f"Error handling message received: {e}")
    
    async def _handle_message_processed(self, event: MessageProcessed):
        """Handle message processed event."""
        try:
            session_id = event.session_id
            user_id = getattr(event, 'user_id', None)
            message = getattr(event, 'message', '')
            response = getattr(event, 'response', '')
            metadata = getattr(event, 'metadata', {})
            
            # Store interaction in episodic memory
            if self.episodic_memory and session_id:
                interaction_data = {
                    'session_id': session_id,
                    'user_id': user_id,
                    'user_message': message,
                    'assistant_response': response,
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'metadata': metadata
                }
                
                await self.episodic_memory.store(interaction_data)
                
                # Update session context
                context = self._session_contexts.get(session_id)
                if context:
                    context.memory_storage_count += 1
                    context.context_elements.append({
                        'type': 'assistant_response',
                        'content': response,
                        'timestamp': datetime.now(timezone.utc).isoformat()
                    })
            
            if self.metrics:
                self.metrics.increment("memory_storage_operations")
            
        except Exception as e:
            self.logger.error(f"Error handling message processed: {e}")
    
    async def get_session_context(self, session_id: str) -> Dict[str, Any]:
        """Get memory context for a session."""
        try:
            context = self._session_contexts.get(session_id)
            if not context:
                return {}
            
            # Update retrieval count
            context.memory_retrieval_count += 1
            context.last_updated = datetime.now(timezone.utc)
            
            if self.metrics:
                self.metrics.increment("memory_retrievals")
            
            return {
                'session_id': context.session_id,
                'user_id': context.user_id,
                'memories': context.memories,
                'context': context.context_elements,
                'facts': context.facts,
                'preferences': context.preferences,
                'stats': {
                    'retrieval_count': context.memory_retrieval_count,
                    'storage_count': context.memory_storage_count,
                    'last_updated': context.last_updated.isoformat()
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error getting session context: {e}")
            return {}
    
    async def retrieve_session_memories(
        self, 
        session_id: str, 
        query: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Retrieve memories for a session."""
        try:
            context = self._session_contexts.get(session_id)
            if not context or not self.memory_manager:
                return []
            
            # Use episodic memory to get session-specific memories
            if self.episodic_memory:
                # Get recent interactions for this session
                memories = await self.episodic_memory.retrieve_by_session(
                    session_id=session_id,
                    limit=limit
                )
                
                if self.metrics:
                    self.metrics.increment("memory_retrievals")
                
                return [
                    {
                        'memory_id': mem.get('id', str(uuid.uuid4())),
                        'content': mem.get('content', ''),
                        'timestamp': mem.get('timestamp', ''),
                        'type': 'episodic',
                        'relevance': 1.0  # Session memories are highly relevant
                    }
                    for mem in memories
                ]
            
            return []
            
        except Exception as e:
            self.logger.error(f"Error retrieving session memories: {e}")
            return []
    
    async def store_session_fact(
        self,
        session_id: str,
        user_id: Optional[str],
        fact: str,
        importance: float = 0.7,
        tags: Optional[Set[str]] = None
    ) -> str:
        """Store an important fact from the session."""
        try:
            # Store in semantic memory if available
            if self.semantic_memory:
                fact_data = {
                    'content': fact,
                    'session_id': session_id,
                    'user_id': user_id,
                    'importance': importance,
                    'tags': list(tags or set()),
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'type': 'session_fact'
                }
                
                memory_id = await self.semantic_memory.store(fact_data)
                
                # Update session context
                context = self._session_contexts.get(session_id)
                if context:
                    context.facts.append({
                        'memory_id': memory_id,
                        'fact': fact,
                        'importance': importance,
                        'timestamp': datetime.now(timezone.utc).isoformat()
                    })
                    context.memory_storage_count += 1
                
                if self.metrics:
                    self.metrics.increment("memory_storage_operations")
                
                return memory_id
            
            return ""
            
        except Exception as e:
            self.logger.error(f"Error storing session fact: {e}")
            return ""
    
    async def _load_user_memories(self, user_id: str) -> List[Dict[str, Any]]:
        """Load relevant memories for a user."""
        try:
            memories = []
            
            # Load recent episodic memories
            if self.episodic_memory:
                recent_episodes = await self.episodic_memory.retrieve_by_user(
                    user_id=user_id,
                    limit=5,
                    days_back=self.memory_retention_days
                )
                
                for episode in recent_episodes:
                    memories.append({
                        'type': 'episodic',
                        'content': episode.get('content', ''),
                        'timestamp': episode.get('timestamp', ''),
                        'memory_id': episode.get('id', str(uuid.uuid4()))
                    })
            
            # Load relevant semantic memories/facts
            if self.semantic_memory:
                user_facts = await self.semantic_memory.retrieve_by_user(
                    user_id=user_id,
                    limit=10
                )
                
                for fact in user_facts:
                    memories.append({
                        'type': 'semantic',
                        'content': fact.get('content', ''),
                        'importance': fact.get('importance', 0.5),
                        'memory_id': fact.get('id', str(uuid.uuid4()))
                    })
            
            return memories
            
        except Exception as e:
            self.logger.error(f"Error loading user memories: {e}")
            return []
    
    async def _consolidate_session_memories(
        self,
        context: SessionMemoryContext,
        duration: float,
        summary: Dict[str, Any]
    ):
        """Consolidate session memories for long-term storage."""
        try:
            if not self.semantic_memory or not context.user_id:
                return
            
            # Create session summary
            session_summary = {
                'session_id': context.session_id,
                'user_id': context.user_id,
                'duration_seconds': duration,
                'interaction_count': len([
                    e for e in context.context_elements 
                    if e.get('type') in ['user_message', 'assistant_response']
                ]),
                'facts_learned': len(context.facts),
                'summary': summary,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'type': 'session_summary'
            }
            
            # Store session summary in semantic memory
            await self.semantic_memory.store(session_summary)
            
            self.logger.info(
                f"Consolidated memories for session {context.session_id}: "
                f"{len(context.facts)} facts, {len(context.context_elements)} interactions"
            )
            
        except Exception as e:
            self.logger.error(f"Error consolidating session memories: {e}")
    
    def get_session_statistics(self) -> Dict[str, Any]:
        """Get statistics about session memory integration."""
        total_sessions = len(self._session_contexts)
        total_memories = sum(len(ctx.memories) for ctx in self._session_contexts.values())
        total_facts = sum(len(ctx.facts) for ctx in self._session_contexts.values())
        total_retrievals = sum(ctx.memory_retrieval_count for ctx in self._session_contexts.values())
        total_storage = sum(ctx.memory_storage_count for ctx in self._session_contexts.values())
        
        return {
            'active_sessions': total_sessions,
            'total_memories': total_memories,
            'total_facts': total_facts,
            'total_retrievals': total_retrievals,
            'total_storage_operations': total_storage,
            'memory_types_enabled': {
                'episodic': self.enable_episodic,
                'semantic': self.enable_semantic,
                'working': self.enable_working
            }
        }
    
    async def cleanup_session(self, session_id: str):
        """Clean up session memory context."""
        if session_id in self._session_contexts:
            del self._session_contexts[session_id]
            self.logger.info(f"Cleaned up memory context for session {session_id}")
    
    async def shutdown(self):
        """Shutdown the session memory integrator."""
        self.logger.info("Shutting down Session Memory Integrator...")
        
        # Save any remaining session contexts if needed
        for session_id, context in self._session_contexts.items():
            try:
                if context.facts:
                    self.logger.info(f"Session {session_id} had {len(context.facts)} unsaved facts")
            except Exception as e:
                self.logger.error(f"Error during shutdown cleanup: {e}")
        
        self._session_contexts.clear()
        self.logger.info("Session Memory Integrator shutdown complete")