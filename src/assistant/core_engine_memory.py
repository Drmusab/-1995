"""
Memory-Enhanced Core Engine
Author: Drmusab
Last Modified: 2025-07-17 19:45:00 UTC

This module provides a memory-enhanced version of the core assistant engine,
integrating with the memory system to provide context-aware responses.
"""

import asyncio
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from datetime import datetime, timezone
import logging
import uuid
import json

from src.core.dependency_injection import Container
from src.core.events.event_bus import EventBus
from src.core.events.event_types import (
    MessageReceived, MessageProcessed, ErrorOccurred,
    MemoryRetrievalRequested, MemoryItemStored
)
from src.assistant.session_memory_integrator import SessionMemoryIntegrator
from src.memory.core_memory.base_memory import MemoryType
from src.memory.operations.retrieval import (
    MemoryRetriever, RetrievalRequest, RetrievalResult,
    RetrievalStrategy, MemoryRetrievalMode
)
from src.memory.operations.context_manager import (
    MemoryContextManager, ContextType, ContextPriority
)
from src.integrations.llm.model_router import ModelRouter
from src.processing.natural_language.intent_manager import IntentManager
from src.processing.natural_language.entity_extractor import EntityExtractor
from src.observability.logging.config import get_logger
from src.observability.monitoring.metrics import MetricsCollector
from src.observability.monitoring.tracing import TraceManager


class MemoryEnhancedCoreEngine:
    """
    Core assistant engine with memory integration for context-aware processing.
    """
    
    def __init__(self, container: Container):
        """
        Initialize the memory-enhanced core engine.
        
        Args:
            container: The dependency injection container
        """
        self.container = container
        self.logger = get_logger(__name__)
        
        # Core components
        self.event_bus = container.get(EventBus)
        self.model_router = container.get(ModelRouter)
        
        # Memory components
        self.memory_integrator = container.get(SessionMemoryIntegrator)
        self.memory_retriever = container.get(MemoryRetriever)
        self.context_manager = container.get(MemoryContextManager)
        
        # NLP components
        self.intent_manager = container.get(IntentManager)
        self.entity_extractor = container.get(EntityExtractor)
        
        # Observability
        try:
            self.metrics = container.get(MetricsCollector)
            self.metrics.register_counter("messages_processed_total", 
                                         "Total messages processed")
            self.metrics.register_histogram("message_processing_time", 
                                           "Message processing time")
            self.metrics.register_counter("memory_enhanced_responses", 
                                         "Responses enhanced with memory")
        except Exception:
            self.logger.warning("Metrics collector not available")
            self.metrics = None
            
        try:
            self.tracer = container.get(TraceManager)
        except Exception:
            self.logger.warning("Trace manager not available")
            self.tracer = None
        
        self.logger.info("Memory-enhanced core engine initialized")
    
    async def process_message(
        self,
        message: str,
        session_id: str,
        user_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a user message with memory-enhanced context.
        
        Args:
            message: The user message
            session_id: Session identifier
            user_id: Optional user identifier
            context: Optional additional context
            
        Returns:
            Response with memory-enhanced context
        """
        start_time = asyncio.get_event_loop().time()
        trace_context = {}
        
        # Start trace if available
        if self.tracer:
            with self.tracer.trace("process_message") as span:
                span.set_attributes({
                    "session_id": session_id,
                    "user_id": user_id or "anonymous",
                    "message_length": len(message)
                })
                trace_context = {"trace_id": span.get_trace_id()}
        
        try:
            # Emit message received event
            await self.event_bus.emit(MessageReceived(
                session_id=session_id,
                user_id=user_id,
                message=message,
                timestamp=datetime.now(timezone.utc)
            ))
            
            # Get memory context
            memory_context = await self._get_memory_context(session_id, message)
            
            # Perform intent detection
            intent = await self.intent_manager.detect_intent(message, context=memory_context)
            
            # Extract entities
            entities = await self.entity_extractor.extract_entities(message)
            
            # Update context with detected entities
            if entities:
                for entity in entities:
                    await self.context_manager.add_entity(
                        session_id=session_id,
                        entity_id=str(uuid.uuid4()),
                        name=entity["text"],
                        entity_type=entity["type"]
                    )
            
            # Prepare prompt with memory context
            prompt = self._build_prompt_with_memory(message, memory_context, intent, entities)
            
            # Get response from LLM
            llm_response = await self.model_router.generate_text(prompt)
            
            # Extract any new facts or knowledge from the response
            await self._extract_and_store_knowledge(llm_response, session_id, user_id)
            
            # Format final response
            response = {
                "text": llm_response,
                "session_id": session_id,
                "intent": intent,
                "entities": entities,
                "memory_enhanced": True,
                "context_used": len(memory_context.get("elements", [])),
                "trace_context": trace_context
            }
            
            # Emit message processed event
            await self.event_bus.emit(MessageProcessed(
                session_id=session_id,
                user_id=user_id,
                message=message,
                response=llm_response,
                metadata={
                    "intent": intent,
                    "memory_enhanced": True,
                    "entities": len(entities)
                }
            ))
            
            # Record metrics
            end_time = asyncio.get_event_loop().time()
            if self.metrics:
                self.metrics.increment("messages_processed_total")
                self.metrics.increment("memory_enhanced_responses")
                self.metrics.record("message_processing_time", end_time - start_time)
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error processing message: {str(e)}")
            
            # Emit error event
            await self.event_bus.emit(ErrorOccurred(
                component="core_engine",
                error_type="message_processing_error",
                error_message=str(e),
                context={"session_id": session_id, "user_id": user_id}
            ))
            
            # Return error response
            return {
                "text": "I'm sorry, I encountered an error processing your message.",
                "session_id": session_id,
                "error": str(e),
                "trace_context": trace_context
            }
    
    async def _get_memory_context(
        self, 
        session_id: str, 
        message: str
    ) -> Dict[str, Any]:
        """
        Get memory context for message processing.
        
        Args:
            session_id: Session identifier
            message: User message
            
        Returns:
            Memory context
        """
        try:
            # Get context from context manager
            context = await self.context_manager.get_context_dict(session_id)
            
            # Emit memory retrieval event
            await self.event_bus.emit(MemoryRetrievalRequested(
                session_id=session_id,
                query=message,
                context_id=context.get("context_id"),
                timestamp=datetime.now(timezone.utc)
            ))
            
            # Get semantic memories related to the message
            request = RetrievalRequest(
                query=message,
                session_id=session_id,
                memory_types=[MemoryType.SEMANTIC],
                strategy=RetrievalStrategy.SEMANTIC,
                mode=MemoryRetrievalMode.CONTEXTUAL,
                max_results=3,
                min_relevance=0.7
            )
            
            semantic_results = await self.memory_retriever.retrieve(request)
            
            # Get recent episodic memories
            episodic_request = RetrievalRequest(
                query=message,
                session_id=session_id,
                memory_types=[MemoryType.EPISODIC],
                strategy=RetrievalStrategy.RECENCY,
                max_results=2
            )
            
            episodic_results = await self.memory_retriever.retrieve(episodic_request)
            
            # Combine all context
            memory_context = {
                "session_id": session_id,
                "context_elements": context.get("elements", []),
                "entities": context.get("entities", []),
                "semantic_memories": [
                    {
                        "content": memory.content,
                        "relevance": score,
                        "memory_id": memory.memory_id
                    }
                    for memory, score in semantic_results.items
                ],
                "episodic_memories": [
                    {
                        "content": memory.content,
                        "memory_id": memory.memory_id
                    }
                    for memory in episodic_results.memories
                ]
            }
            
            return memory_context
            
        except Exception as e:
            self.logger.error(f"Error getting memory context: {str(e)}")
            return {
                "session_id": session_id,
                "error": str(e),
                "context_elements": [],
                "entities": [],
                "semantic_memories": [],
                "episodic_memories": []
            }
    
    def _build_prompt_with_memory(
        self,
        message: str,
        memory_context: Dict[str, Any],
        intent: str,
        entities: List[Dict[str, Any]]
    ) -> str:
        """
        Build a prompt enhanced with memory context.
        
        Args:
            message: User message
            memory_context: Memory context
            intent: Detected intent
            entities: Extracted entities
            
        Returns:
            Enhanced prompt
        """
        # Start with system instructions
        prompt = "You are an AI assistant with memory capabilities. "
        prompt += "Use the following context from your memory to provide a helpful response.\n\n"
        
        # Add context elements
        if memory_context.get("context_elements"):
            prompt += "## Current Context\n"
            for element in memory_context.get("context_elements", [])[:5]:
                prompt += f"- {element.get('content', '')}\n"
            prompt += "\n"
        
        # Add entities
        if memory_context.get("entities"):
            prompt += "## Recognized Entities\n"
            for entity in memory_context.get("entities", [])[:5]:
                prompt += f"- {entity.get('name', '')} ({entity.get('entity_type', '')})\n"
            prompt += "\n"
        
        # Add relevant semantic memories
        if memory_context.get("semantic_memories"):
            prompt += "## Relevant Knowledge\n"
            for memory in memory_context.get("semantic_memories", []):
                content = memory.get("content", "")
                if isinstance(content, dict):
                    # Format dictionary content
                    content_str = json.dumps(content, indent=2)
                else:
                    content_str = str(content)
                prompt += f"- {content_str}\n"
            prompt += "\n"
        
        # Add recent episodic memories if available
        if memory_context.get("episodic_memories"):
            prompt += "## Recent Interactions\n"
            for memory in memory_context.get("episodic_memories", []):
                content = memory.get("content", "")
                if isinstance(content, dict):
                    # Format dictionary content
                    content_str = json.dumps(content, indent=2)
                else:
                    content_str = str(content)
                prompt += f"- {content_str}\n"
            prompt += "\n"
        
        # Add detected intent
        prompt += f"## Detected Intent: {intent}\n\n"
        
        # Add user message
        prompt += f"## User Message\n{message}\n\n"
        
        # Add final instructions
        prompt += "## Instructions\n"
        prompt += "1. Use the provided context and knowledge to formulate your response\n"
        prompt += "2. If you recognize entities or concepts from previous interactions, acknowledge them\n"
        prompt += "3. Provide a helpful, relevant response based on your understanding of the context\n"
        prompt += "4. Be concise but thorough\n\n"
        
        prompt += "## Your Response:"
        
        return prompt
    
    async def _extract_and_store_knowledge(
        self,
        response: str,
        session_id: str,
        user_id: Optional[str]
    ) -> None:
        """
        Extract and store knowledge from a response.
        
        Args:
            response: The assistant's response
            session_id: Session identifier
            user_id: User identifier
        
