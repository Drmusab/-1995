"""
Memory Learning Bridge
Author: Drmusab
Last Modified: 2025-07-17 19:40:00 UTC

This module provides integration between the learning system and memory,
allowing the assistant to learn from experiences and improve over time.
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import asyncio

from src.core.dependency_injection import Container
from src.core.events.event_bus import EventBus
from src.core.events.event_types import FeedbackReceived, LearningEventOccurred, MemoryItemStored
from src.memory.core_memory.base_memory import (
    MemoryAccess,
    MemoryItem,
    MemoryMetadata,
    MemoryRetentionPolicy,
    MemorySensitivity,
    MemoryType,
)
from src.memory.core_memory.memory_manager import MemoryManager
from src.memory.operations.consolidation import MemoryConsolidator
from src.memory.operations.context_manager import MemoryContextManager
from src.memory.operations.retrieval import MemoryRetriever, RetrievalRequest
from src.observability.logging.config import get_logger
from src.observability.monitoring.metrics import MetricsCollector


class MemoryLearningBridge:
    """
    Bridges the learning and memory systems, enabling the assistant to
    learn from experiences stored in memory and adapt over time.
    """

    def __init__(self, container: Container):
        """
        Initialize the memory learning bridge.

        Args:
            container: The dependency injection container
        """
        self.container = container
        self.logger = get_logger(__name__)

        # Core components
        self.memory_manager = container.get(MemoryManager)
        self.memory_retriever = container.get(MemoryRetriever)
        self.memory_consolidator = container.get(MemoryConsolidator)
        self.event_bus = container.get(EventBus)

        # Optional components
        try:
            self.metrics = container.get(MetricsCollector)
            self.metrics.register_counter("learning_events_processed", "Learning events processed")
            self.metrics.register_counter("feedback_memories_created", "Feedback memories created")
        except Exception:
            self.logger.warning("Metrics collector not available")
            self.metrics = None

        # Register event handlers
        self._register_event_handlers()

        self.logger.info("Memory learning bridge initialized")

    def _register_event_handlers(self) -> None:
        """Register event handlers."""
        self.event_bus.subscribe(LearningEventOccurred, self._handle_learning_event)
        self.event_bus.subscribe(FeedbackReceived, self._handle_feedback)

    async def _handle_learning_event(self, event: LearningEventOccurred) -> None:
        """
        Handle learning events by storing them in memory.

        Args:
            event: The learning event
        """
        event_data = event.data
        event_type = event.event_type
        user_id = event.user_id
        session_id = event.session_id

        try:
            # Store learning event in semantic memory
            memory_id = await self.memory_manager.store_memory(
                data={
                    "event_type": event_type,
                    "description": event.description,
                    "data": event_data,
                },
                memory_type=MemoryType.SEMANTIC,
                owner_id=user_id,
                session_id=session_id,
                metadata={
                    "learning_event": True,
                    "event_type": event_type,
                    "importance": event.importance,
                    "tags": ["learning", event_type] + list(event.tags or []),
                },
            )

            self.logger.info(f"Stored learning event {event_type} in memory: {memory_id}")

            # Record metric
            if self.metrics:
                self.metrics.increment("learning_events_processed", tags={"event_type": event_type})

        except Exception as e:
            self.logger.error(f"Error storing learning event in memory: {str(e)}")

    async def _handle_feedback(self, event: FeedbackReceived) -> None:
        """
        Handle feedback events by storing them in memory.

        Args:
            event: The feedback event
        """
        feedback_data = event.feedback
        user_id = event.user_id
        session_id = event.session_id

        try:
            # Store feedback in episodic memory
            memory_id = await self.memory_manager.store_memory(
                data={"feedback": feedback_data, "rating": event.rating, "context": event.context},
                memory_type=MemoryType.EPISODIC,
                owner_id=user_id,
                session_id=session_id,
                metadata={
                    "feedback": True,
                    "rating": event.rating,
                    "importance": 0.8,  # Feedback is important for learning
                    "tags": ["feedback", f"rating_{event.rating}"],
                },
            )

            self.logger.info(f"Stored feedback in memory: {memory_id}")

            # Record metric
            if self.metrics:
                self.metrics.increment(
                    "feedback_memories_created", tags={"rating": str(event.rating)}
                )

        except Exception as e:
            self.logger.error(f"Error storing feedback in memory: {str(e)}")

    async def retrieve_learning_examples(
        self, concept: str, limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Retrieve learning examples for a concept.

        Args:
            concept: The concept to retrieve examples for
            limit: Maximum number of examples

        Returns:
            List of learning examples
        """
        try:
            # Create retrieval request
            request = RetrievalRequest(
                query=concept,
                memory_types=[MemoryType.SEMANTIC],
                max_results=limit,
                tags={"learning"},
            )

            # Retrieve memories
            result = await self.memory_retriever.retrieve(request)

            # Format results
            examples = []
            for memory in result.memories:
                if isinstance(memory.content, dict):
                    example = {
                        "memory_id": memory.memory_id,
                        "concept": concept,
                        "event_type": memory.content.get("event_type", "unknown"),
                        "description": memory.content.get("description", ""),
                        "data": memory.content.get("data", {}),
                    }
                    examples.append(example)

            return examples

        except Exception as e:
            self.logger.error(f"Error retrieving learning examples: {str(e)}")
            return []

    async def retrieve_feedback_patterns(
        self, positive_only: bool = False, limit: int = 10
    ) -> Dict[str, Any]:
        """
        Retrieve patterns from user feedback.

        Args:
            positive_only: Only include positive feedback
            limit: Maximum number of feedback items

        Returns:
            Dictionary with feedback patterns
        """
        try:
            # Create retrieval request
            request = RetrievalRequest(
                query="feedback",
                memory_types=[MemoryType.EPISODIC],
                max_results=limit * 2,  # Get more to filter
                tags={"feedback"},
            )

            # Retrieve memories
            result = await self.memory_retriever.retrieve(request)

            # Filter and analyze feedback
            positive_feedback = []
            negative_feedback = []

            for memory in result.memories:
                if isinstance(memory.content, dict):
                    rating = memory.content.get("rating", 0)
                    feedback = memory.content.get("feedback", "")

                    if rating >= 4:  # Positive feedback
                        positive_feedback.append(
                            {"rating": rating, "feedback": feedback, "memory_id": memory.memory_id}
                        )
                    elif rating <= 2 and not positive_only:  # Negative feedback
                        negative_feedback.append(
                            {"rating": rating, "feedback": feedback, "memory_id": memory.memory_id}
                        )

            # Limit results
            positive_feedback = positive_feedback[:limit]
            negative_feedback = negative_feedback[:limit]

            return {
                "positive_feedback": positive_feedback,
                "negative_feedback": negative_feedback if not positive_only else [],
                "positive_count": len(positive_feedback),
                "negative_count": len(negative_feedback) if not positive_only else 0,
            }

        except Exception as e:
            self.logger.error(f"Error retrieving feedback patterns: {str(e)}")
            return {
                "positive_feedback": [],
                "negative_feedback": [],
                "positive_count": 0,
                "negative_count": 0,
                "error": str(e),
            }

    async def store_learned_concept(
        self,
        concept: str,
        definition: str,
        examples: List[str],
        source: Optional[str] = None,
        user_id: Optional[str] = None,
        confidence: float = 0.8,
        tags: Optional[Set[str]] = None,
    ) -> str:
        """
        Store a newly learned concept in semantic memory.

        Args:
            concept: The concept name
            definition: Definition of the concept
            examples: Examples of the concept
            source: Source of the knowledge
            user_id: Optional user ID
            confidence: Confidence in the learned concept
            tags: Optional tags

        Returns:
            Memory ID of the stored concept
        """
        try:
            # Prepare concept data
            concept_data = {
                "concept": concept,
                "definition": definition,
                "examples": examples,
                "source": source,
                "learned_at": datetime.now(timezone.utc).isoformat(),
            }

            # Store in semantic memory
            memory_id = await self.memory_manager.store_memory(
                data=concept_data,
                memory_type=MemoryType.SEMANTIC,
                owner_id=user_id,
                metadata={
                    "concept": True,
                    "confidence": confidence,
                    "importance": 0.7,
                    "tags": ["concept", concept] + list(tags or []),
                },
            )

            self.logger.info(f"Stored learned concept '{concept}' in memory: {memory_id}")

            # Emit learning event
            await self.event_bus.emit(
                LearningEventOccurred(
                    event_type="concept_learned",
                    description=f"Learned new concept: {concept}",
                    data=concept_data,
                    user_id=user_id,
                    importance=0.7,
                    tags={"concept", "semantic_memory"},
                )
            )

            return memory_id

        except Exception as e:
            self.logger.error(f"Error storing learned concept: {str(e)}")
            return ""

    async def apply_learning_from_memories(
        self, memory_ids: List[str], learning_type: str, metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Apply learning from specific memories.

        Args:
            memory_ids: List of memory IDs to learn from
            learning_type: Type of learning to apply
            metadata: Additional metadata

        Returns:
            Results of the learning process
        """
        result = {
            "success": False,
            "learning_type": learning_type,
            "memories_processed": 0,
            "insights": [],
        }

        try:
            # Retrieve the specified memories
            memories = []
            for memory_id in memory_ids:
                memory = await self.memory_manager.retrieve_memory(memory_id)
                if memory:
                    memories.append(memory)

            result["memories_processed"] = len(memories)

            if not memories:
                result["error"] = "No valid memories found"
                return result

            # Apply different learning strategies based on type
            if learning_type == "pattern_recognition":
                # Example implementation for pattern recognition
                patterns = await self._extract_patterns_from_memories(memories)
                result["insights"] = patterns
                result["success"] = True

            elif learning_type == "feedback_analysis":
                # Example implementation for feedback analysis
                feedback_insights = await self._analyze_feedback_memories(memories)
                result["insights"] = feedback_insights
                result["success"] = True

            elif learning_type == "knowledge_extraction":
                # Example implementation for knowledge extraction
                knowledge = await self._extract_knowledge_from_memories(memories)
                result["insights"] = knowledge
                result["success"] = True

            else:
                result["error"] = f"Unsupported learning type: {learning_type}"

            # Record the learning event
            if result["success"]:
                await self.event_bus.emit(
                    LearningEventOccurred(
                        event_type=f"learning_applied_{learning_type}",
                        description=f"Applied {learning_type} learning to {len(memories)} memories",
                        data={
                            "memory_ids": memory_ids,
                            "insights_count": len(result["insights"]),
                            "metadata": metadata or {},
                        },
                        importance=0.6,
                        tags={learning_type, "memory_based_learning"},
                    )
                )

            return result

        except Exception as e:
            self.logger.error(f"Error applying learning from memories: {str(e)}")
            result["error"] = str(e)
            return result

    async def _extract_patterns_from_memories(
        self, memories: List[MemoryItem]
    ) -> List[Dict[str, Any]]:
        """
        Extract patterns from memories.

        Args:
            memories: List of memory items

        Returns:
            List of identified patterns
        """
        # This is a simplified implementation
        # In a real system, this would use more sophisticated pattern recognition

        patterns = []

        # Example: Look for repeated content patterns
        content_frequency = {}

        for memory in memories:
            if isinstance(memory.content, dict):
                # For dictionary content, convert to string for comparison
                content_str = str(memory.content)
            else:
                content_str = str(memory.content)

            if content_str in content_frequency:
                content_frequency[content_str] += 1
            else:
                content_frequency[content_str] = 1

        # Find patterns with frequency > 1
        for content, frequency in content_frequency.items():
            if frequency > 1:
                patterns.append(
                    {
                        "pattern_type": "repeated_content",
                        "content": content[:100] + "..." if len(content) > 100 else content,
                        "frequency": frequency,
                        "confidence": min(0.5 + (frequency * 0.1), 0.9),
                    }
                )

        return patterns

    async def _analyze_feedback_memories(self, memories: List[MemoryItem]) -> List[Dict[str, Any]]:
        """
        Analyze feedback memories.

        Args:
            memories: List of memory items

        Returns:
            List of feedback insights
        """
        # This is a simplified implementation

        positive_count = 0
        negative_count = 0
        neutral_count = 0
        feedback_texts = []

        for memory in memories:
            if isinstance(memory.content, dict) and "rating" in memory.content:
                rating = memory.content.get("rating", 0)
                feedback = memory.content.get("feedback", "")

                if feedback:
                    feedback_texts.append(feedback)

                if rating >= 4:
                    positive_count += 1
                elif rating <= 2:
                    negative_count += 1
                else:
                    neutral_count += 1

        # Generate insights
        insights = [
            {
                "insight_type": "feedback_distribution",
                "positive_count": positive_count,
                "negative_count": negative_count,
                "neutral_count": neutral_count,
                "total_count": len(memories),
                "positive_ratio": positive_count / len(memories) if memories else 0,
            }
        ]

        # Add common feedback themes if we have enough data
        if feedback_texts:
            # In a real system, this would use NLP to extract themes
            # Here we just use a simplified approach
            insights.append(
                {
                    "insight_type": "feedback_themes",
                    "sample_count": len(feedback_texts),
                    "common_words": self._get_common_words(feedback_texts),
                }
            )

        return insights

    def _get_common_words(self, texts: List[str], limit: int = 5) -> List[str]:
        """Extract common words from texts."""
        word_count = {}

        for text in texts:
            # Simple word extraction
            words = text.lower().split()
            for word in words:
                if len(word) > 3:  # Skip short words
                    if word in word_count:
                        word_count[word] += 1
                    else:
                        word_count[word] = 1

        # Sort by frequency
        sorted_words = sorted(word_count.items(), key=lambda x: x[1], reverse=True)

        # Return top words
        return [word for word, count in sorted_words[:limit]]

    async def _extract_knowledge_from_memories(
        self, memories: List[MemoryItem]
    ) -> List[Dict[str, Any]]:
        """
        Extract knowledge from memories.

        Args:
            memories: List of memory items

        Returns:
            List of knowledge items
        """
        # This is a simplified implementation

        knowledge_items = []

        for memory in memories:
            if memory.memory_type == MemoryType.SEMANTIC:
                # Extract from semantic memories
                if isinstance(memory.content, dict):
                    if "concept" in memory.content:
                        # This is a concept
                        knowledge_items.append(
                            {
                                "knowledge_type": "concept",
                                "concept": memory.content.get("concept", ""),
                                "definition": memory.content.get("definition", ""),
                                "source": memory.content.get("source", "memory"),
                                "confidence": memory.metadata.confidence,
                            }
                        )
                    elif "fact" in memory.content:
                        # This is a fact
                        knowledge_items.append(
                            {
                                "knowledge_type": "fact",
                                "fact": memory.content.get("fact", ""),
                                "source": memory.content.get("source", "memory"),
                                "confidence": memory.metadata.confidence,
                            }
                        )

        return knowledge_items
