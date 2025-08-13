"""
Memory Context Manager
Author: Drmusab
Last Modified: 2025-07-05 10:30:05 UTC

This module provides context management functionality for the AI assistant's memory,
maintaining and tracking conversational context, relevant entities, and contextual
information. It enables context-aware memory retrieval and maintains appropriate
context windows for optimal memory utilization.
"""

import heapq
import json
import logging
import math
import time
import traceback
import uuid
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import (
    Any,
    Callable,
    DefaultDict,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)

import asyncio

# Core imports
from src.core.config.loader import ConfigLoader
from src.core.dependency_injection import Container
from src.core.error_handling import ErrorHandler, handle_exceptions
from src.core.events.event_bus import EventBus
from src.core.events.event_types import (
    ContextCleared,
    ContextEntityDetected,
    ContextEntityRemoved,
    ContextRestored,
    ContextUpdated,
    ContextWindowChanged,
    MemoryItemRetrieved,
    MemoryItemStored,
    MemoryItemUpdated,
)

# Integration components
from src.integrations.llm.model_router import ModelRouter

# Memory system imports
from src.memory.core_memory.base_memory import (
    BaseMemory,
    MemoryError,
    MemoryItem,
    MemoryType,
    SimpleMemoryQuery,
)
from src.memory.core_memory.memory_types import EpisodicMemory, SemanticMemory, WorkingMemory
from src.memory.storage.memory_graph import (
    GraphNodeType,
    GraphQuery,
    MemoryGraphStore,
    RelationshipType,
)
from src.memory.storage.vector_store import VectorMemoryStore
from src.observability.logging.config import get_logger

# Observability
from src.observability.monitoring.metrics import MetricsCollector
from src.observability.monitoring.tracing import TraceManager

# Processing components
from src.processing.natural_language.entity_extractor import EntityExtractor
from src.processing.natural_language.intent_manager import IntentManager
from src.processing.natural_language.sentiment_analyzer import SentimentAnalyzer


class ContextType(Enum):
    """Types of context elements."""

    CONVERSATION = "conversation"  # Conversational context
    TASK = "task"  # Task/goal context
    USER = "user"  # User information context
    ENVIRONMENT = "environment"  # Environmental context
    TEMPORAL = "temporal"  # Time-based context
    EMOTIONAL = "emotional"  # Emotional context
    FACTUAL = "factual"  # Factual/knowledge context
    REFERENCE = "reference"  # Reference to external entities
    CUSTOM = "custom"  # Custom context type


class ContextPriority(Enum):
    """Priority levels for context elements."""

    CRITICAL = 3  # Critical context (always retained)
    HIGH = 2  # High priority context
    MEDIUM = 1  # Medium priority context
    LOW = 0  # Low priority context (first to be pruned)


@dataclass
class ContextEntity:
    """
    Represents an entity detected in context.

    Entities are important elements in the conversation context that
    can be referenced and tracked across the interaction.
    """

    entity_id: str  # Unique entity identifier
    name: str  # Entity name
    entity_type: str  # Entity type (person, place, organization, etc.)
    first_mentioned_at: datetime  # When first mentioned
    last_mentioned_at: datetime  # When last mentioned
    mention_count: int = 1  # Number of mentions
    attributes: Dict[str, Any] = field(default_factory=dict)  # Entity attributes
    linked_memory_ids: List[str] = field(default_factory=list)  # Associated memory IDs
    confidence: float = 1.0  # Confidence in entity detection

    def update_mention(self, attributes: Optional[Dict[str, Any]] = None) -> None:
        """
        Update entity with a new mention.

        Args:
            attributes: New attributes to merge
        """
        self.last_mentioned_at = datetime.now(timezone.utc)
        self.mention_count += 1

        if attributes:
            self.attributes.update(attributes)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = asdict(self)
        # Convert datetime objects to ISO format strings
        result["first_mentioned_at"] = self.first_mentioned_at.isoformat()
        result["last_mentioned_at"] = self.last_mentioned_at.isoformat()
        return result


@dataclass
class ContextElement:
    """
    A single element in the context.

    Context elements represent pieces of information that collectively
    form the overall context for the conversation or interaction.
    """

    element_id: str  # Unique element identifier
    content: Any  # Element content
    context_type: ContextType  # Type of context
    priority: ContextPriority = ContextPriority.MEDIUM  # Element priority
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    source: Optional[str] = None  # Source of the context element
    memory_id: Optional[str] = None  # Associated memory ID
    relevance: float = 1.0  # Current relevance score
    ttl: Optional[timedelta] = None  # Time-to-live
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional metadata

    def update(self, content: Any) -> None:
        """
        Update element content.

        Args:
            content: New content
        """
        self.content = content
        self.updated_at = datetime.now(timezone.utc)
        self.relevance = 1.0  # Reset relevance on update

    def update_relevance(self, score: float) -> None:
        """
        Update element relevance.

        Args:
            score: New relevance score
        """
        self.relevance = max(0.0, min(1.0, score))

    def is_expired(self) -> bool:
        """Check if the element has expired based on TTL."""
        if not self.ttl:
            return False

        return datetime.now(timezone.utc) - self.updated_at > self.ttl

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = {
            "element_id": self.element_id,
            "content": self.content,
            "context_type": self.context_type.value,
            "priority": self.priority.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "source": self.source,
            "memory_id": self.memory_id,
            "relevance": self.relevance,
            "ttl": str(self.ttl) if self.ttl else None,
            "metadata": self.metadata,
        }
        return result


@dataclass
class ContextWindow:
    """
    A window of context elements.

    The context window represents the current active context for the
    conversation, containing a limited number of context elements based
    on recency, relevance, and priority.
    """

    window_id: str  # Unique window identifier
    session_id: str  # Associated session ID
    elements: List[ContextElement] = field(default_factory=list)  # Context elements
    max_size: int = 20  # Maximum number of elements
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def add_element(self, element: ContextElement) -> bool:
        """
        Add an element to the window.

        Args:
            element: Context element

        Returns:
            True if added, False if window is full
        """
        # Check if element already exists (by ID)
        for i, existing in enumerate(self.elements):
            if existing.element_id == element.element_id:
                # Update existing element
                self.elements[i] = element
                self.updated_at = datetime.now(timezone.utc)
                return True

        # If window is full, we need to make room
        if len(self.elements) >= self.max_size:
            # Try to remove expired elements first
            expired = [e for e in self.elements if e.is_expired()]
            if expired:
                # Remove the oldest expired element
                expired.sort(key=lambda e: e.updated_at)
                self.elements.remove(expired[0])
            else:
                # Remove lowest priority, least relevant element
                candidates = [(e, e.priority.value, e.relevance) for e in self.elements]
                candidates.sort(key=lambda x: (x[1], x[2]))  # Sort by priority, then relevance

                # Don't remove if new element has lower priority and relevance
                if element.priority.value < candidates[0][1] or (
                    element.priority.value == candidates[0][1]
                    and element.relevance < candidates[0][2]
                ):
                    return False

                self.elements.remove(candidates[0][0])

        # Add new element
        self.elements.append(element)
        self.updated_at = datetime.now(timezone.utc)
        return True

    def remove_element(self, element_id: str) -> bool:
        """
        Remove an element from the window.

        Args:
            element_id: Element identifier

        Returns:
            True if removed, False if not found
        """
        for element in self.elements:
            if element.element_id == element_id:
                self.elements.remove(element)
                self.updated_at = datetime.now(timezone.utc)
                return True
        return False

    def get_element(self, element_id: str) -> Optional[ContextElement]:
        """
        Get an element by ID.

        Args:
            element_id: Element identifier

        Returns:
            Context element or None if not found
        """
        for element in self.elements:
            if element.element_id == element_id:
                return element
        return None

    def clear(self) -> None:
        """Clear all elements from the window."""
        self.elements.clear()
        self.updated_at = datetime.now(timezone.utc)

    def prune(self, min_relevance: float = 0.3) -> int:
        """
        Prune low-relevance elements.

        Args:
            min_relevance: Minimum relevance to keep

        Returns:
            Number of elements removed
        """
        original_size = len(self.elements)

        # Keep critical elements regardless of relevance
        self.elements = [
            e
            for e in self.elements
            if e.priority == ContextPriority.CRITICAL or e.relevance >= min_relevance
        ]

        if len(self.elements) < original_size:
            self.updated_at = datetime.now(timezone.utc)

        return original_size - len(self.elements)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "window_id": self.window_id,
            "session_id": self.session_id,
            "elements": [e.to_dict() for e in self.elements],
            "max_size": self.max_size,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "current_size": len(self.elements),
        }


@dataclass
class ConversationContext:
    """
    Complete context for a conversation session.

    This class represents the full context for a conversation,
    including the context window, entities, and metadata.
    """

    session_id: str  # Associated session ID
    window: ContextWindow  # Active context window
    entities: Dict[str, ContextEntity] = field(default_factory=dict)  # Tracked entities
    metadata: Dict[str, Any] = field(default_factory=dict)  # Context metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def update(self) -> None:
        """Update timestamp."""
        self.updated_at = datetime.now(timezone.utc)

    def add_entity(self, entity: ContextEntity) -> None:
        """
        Add or update an entity.

        Args:
            entity: Context entity
        """
        if entity.entity_id in self.entities:
            # Update existing entity
            existing = self.entities[entity.entity_id]
            existing.update_mention(entity.attributes)
            # Update linked memories
            for memory_id in entity.linked_memory_ids:
                if memory_id not in existing.linked_memory_ids:
                    existing.linked_memory_ids.append(memory_id)
        else:
            # Add new entity
            self.entities[entity.entity_id] = entity

        self.update()

    def remove_entity(self, entity_id: str) -> bool:
        """
        Remove an entity.

        Args:
            entity_id: Entity identifier

        Returns:
            True if removed, False if not found
        """
        if entity_id in self.entities:
            del self.entities[entity_id]
            self.update()
            return True
        return False

    def get_entity(self, entity_id: str) -> Optional[ContextEntity]:
        """
        Get an entity by ID.

        Args:
            entity_id: Entity identifier

        Returns:
            Context entity or None if not found
        """
        return self.entities.get(entity_id)

    def find_entity_by_name(self, name: str) -> Optional[ContextEntity]:
        """
        Find an entity by name.

        Args:
            name: Entity name

        Returns:
            Context entity or None if not found
        """
        for entity in self.entities.values():
            if entity.name.lower() == name.lower():
                return entity
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "session_id": self.session_id,
            "window": self.window.to_dict(),
            "entities": {eid: e.to_dict() for eid, e in self.entities.items()},
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


class MemoryContextManager:
    """
    Memory context management system for the AI assistant.

    This class manages the context for conversations and interactions,
    providing methods for updating, retrieving, and maintaining context
    to support memory operations.
    """

    def __init__(self, container: Container):
        """
        Initialize the context manager.

        Args:
            container: Dependency injection container
        """
        self.container = container
        self.logger = get_logger(__name__)

        # Load configuration
        self.config_loader = container.get(ConfigLoader)
        self.context_config = self.config_loader.get("memory.context", {})

        # Event system
        self.event_bus = container.get(EventBus)

        # Get memory systems
        self.working_memory = container.get(WorkingMemory)
        self.episodic_memory = container.get(EpisodicMemory)
        self.semantic_memory = container.get(SemanticMemory)

        # Get storage systems
        try:
            self.vector_store = container.get(VectorMemoryStore)
        except Exception:
            self.logger.warning(
                "VectorMemoryStore not available, vector-based context retrieval will be limited"
            )
            self.vector_store = None

        try:
            self.graph_store = container.get(MemoryGraphStore)
        except Exception:
            self.logger.warning(
                "MemoryGraphStore not available, graph-based context retrieval will be limited"
            )
            self.graph_store = None

        # Get processing components
        try:
            self.entity_extractor = container.get(EntityExtractor)
        except Exception:
            self.logger.warning("EntityExtractor not available, entity detection will be limited")
            self.entity_extractor = None

        try:
            self.intent_manager = container.get(IntentManager)
        except Exception:
            self.logger.warning("IntentManager not available, intent-based context will be limited")
            self.intent_manager = None

        try:
            self.sentiment_analyzer = container.get(SentimentAnalyzer)
        except Exception:
            self.logger.warning(
                "SentimentAnalyzer not available, emotional context will be limited"
            )
            self.sentiment_analyzer = None

        # Get model router for embeddings
        try:
            self.model_router = container.get(ModelRouter)
        except Exception:
            self.logger.warning(
                "ModelRouter not available, semantic context operations will be limited"
            )
            self.model_router = None

        # Monitoring components
        try:
            self.metrics = container.get(MetricsCollector)
            self.tracer = container.get(TraceManager)
        except Exception:
            self.logger.warning("Monitoring components not available")
            self.metrics = None
            self.tracer = None

        # Active contexts by session
        self._contexts: Dict[str, ConversationContext] = {}

        # Context element caches
        self._element_cache: Dict[str, ContextElement] = {}
        self._entity_cache: Dict[str, ContextEntity] = {}

        # Configure settings
        self.default_window_size = self.context_config.get("default_window_size", 20)
        self.min_relevance_threshold = self.context_config.get("min_relevance_threshold", 0.3)
        self.auto_entity_detection = self.context_config.get("auto_entity_detection", True)
        self.auto_prune_interval = self.context_config.get(
            "auto_prune_interval", 10
        )  # Number of updates
        self._update_counter: Dict[str, int] = defaultdict(int)  # Track updates per session

        # Register metrics
        if self.metrics:
            self.metrics.register_counter("context_elements_added")
            self.metrics.register_counter("context_elements_removed")
            self.metrics.register_counter("context_entities_detected")
            self.metrics.register_counter("context_updates")
            self.metrics.register_gauge("context_window_size")
            self.metrics.register_gauge("context_entity_count")

        self.logger.info("MemoryContextManager initialized")

    async def initialize_context(self, session_id: str) -> str:
        """
        Initialize context for a session.

        Args:
            session_id: Session identifier

        Returns:
            Context window identifier
        """
        # Create new context window
        window_id = str(uuid.uuid4())
        window = ContextWindow(
            window_id=window_id, session_id=session_id, max_size=self.default_window_size
        )

        # Create conversation context
        context = ConversationContext(session_id=session_id, window=window)

        # Store context
        self._contexts[session_id] = context

        # Add initial temporal context
        now = datetime.now(timezone.utc)
        self.add_context_element(
            session_id=session_id,
            content={
                "timestamp": now.isoformat(),
                "date": now.strftime("%Y-%m-%d"),
                "time": now.strftime("%H:%M:%S"),
                "timezone": "UTC",
            },
            context_type=ContextType.TEMPORAL,
            priority=ContextPriority.MEDIUM,
            ttl=timedelta(hours=1),
        )

        # Emit event
        await self.event_bus.emit(
            ContextWindowChanged(
                session_id=session_id, window_id=window_id, operation="initialize", element_count=0
            )
        )

        # Update metrics
        if self.metrics:
            self.metrics.gauge("context_window_size", 1, {"session_id": session_id})
            self.metrics.gauge("context_entity_count", 0, {"session_id": session_id})

        self.logger.info(f"Initialized context for session {session_id}")
        return window_id

    async def add_context_element(
        self,
        session_id: str,
        content: Any,
        context_type: ContextType,
        priority: ContextPriority = ContextPriority.MEDIUM,
        source: Optional[str] = None,
        memory_id: Optional[str] = None,
        ttl: Optional[timedelta] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Add a context element to a session.

        Args:
            session_id: Session identifier
            content: Element content
            context_type: Type of context
            priority: Element priority
            source: Source of the context element
            memory_id: Associated memory ID
            ttl: Time-to-live
            metadata: Additional metadata

        Returns:
            Element identifier if added, None if failed
        """
        # Check if session exists
        if session_id not in self._contexts:
            self.logger.warning(f"Cannot add context element: session {session_id} not found")
            return None

        # Create element
        element_id = str(uuid.uuid4())
        element = ContextElement(
            element_id=element_id,
            content=content,
            context_type=context_type,
            priority=priority,
            source=source,
            memory_id=memory_id,
            ttl=ttl,
            metadata=metadata or {},
        )

        # Add to context window
        context = self._contexts[session_id]
        if not context.window.add_element(element):
            self.logger.debug("Failed to add element to context window: window full")
            return None

        # Update context
        context.update()

        # Add to cache
        self._element_cache[element_id] = element

        # Extract entities if enabled
        if self.auto_entity_detection and self.entity_extractor:
            await self._extract_entities_from_element(element, session_id)

        # Increment update counter
        self._update_counter[session_id] += 1

        # Auto-prune if needed
        if self._update_counter[session_id] >= self.auto_prune_interval:
            await self.prune_context(session_id)
            self._update_counter[session_id] = 0

        # Emit event
        await self.event_bus.emit(
            ContextUpdated(
                session_id=session_id,
                context_type=context_type.value,
                element_id=element_id,
                operation="add",
            )
        )

        # Update metrics
        if self.metrics:
            self.metrics.increment("context_elements_added")
            self.metrics.increment("context_updates")
            self.metrics.gauge(
                "context_window_size", len(context.window.elements), {"session_id": session_id}
            )

        return element_id

    async def update_context_element(
        self,
        session_id: str,
        element_id: str,
        content: Any,
        update_metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Update a context element.

        Args:
            session_id: Session identifier
            element_id: Element identifier
            content: New content
            update_metadata: Metadata to update

        Returns:
            True if updated, False if not found
        """
        # Check if session exists
        if session_id not in self._contexts:
            self.logger.warning(f"Cannot update context element: session {session_id} not found")
            return False

        # Get context
        context = self._contexts[session_id]

        # Find element
        element = context.window.get_element(element_id)
        if not element:
            return False

        # Update element
        element.update(content)

        # Update metadata if provided
        if update_metadata:
            element.metadata.update(update_metadata)

        # Update context
        context.update()

        # Update cache
        self._element_cache[element_id] = element

        # Emit event
        await self.event_bus.emit(
            ContextUpdated(
                session_id=session_id,
                context_type=element.context_type.value,
                element_id=element_id,
                operation="update",
            )
        )

        # Update metrics
        if self.metrics:
            self.metrics.increment("context_updates")

        return True

    async def remove_context_element(self, session_id: str, element_id: str) -> bool:
        """
        Remove a context element.

        Args:
            session_id: Session identifier
            element_id: Element identifier

        Returns:
            True if removed, False if not found
        """
        # Check if session exists
        if session_id not in self._contexts:
            self.logger.warning(f"Cannot remove context element: session {session_id} not found")
            return False

        # Get context
        context = self._contexts[session_id]

        # Remove element
        if not context.window.remove_element(element_id):
            return False

        # Update context
        context.update()

        # Remove from cache
        if element_id in self._element_cache:
            del self._element_cache[element_id]

        # Emit event
        await self.event_bus.emit(
            ContextUpdated(
                session_id=session_id, context_type="any", element_id=element_id, operation="remove"
            )
        )

        # Update metrics
        if self.metrics:
            self.metrics.increment("context_elements_removed")
            self.metrics.gauge(
                "context_window_size", len(context.window.elements), {"session_id": session_id}
            )

        return True

    async def get_context_element(
        self, session_id: str, element_id: str
    ) -> Optional[ContextElement]:
        """
        Get a context element.

        Args:
            session_id: Session identifier
            element_id: Element identifier

        Returns:
            Context element or None if not found
        """
        # Check cache first
        if element_id in self._element_cache:
            element = self._element_cache[element_id]
            # Verify session
            if (
                session_id
                == self._contexts.get(
                    session_id, ConversationContext(session_id="", window=None)
                ).session_id
            ):
                return element

        # Check if session exists
        if session_id not in self._contexts:
            return None

        # Get context
        context = self._contexts[session_id]

        # Get element
        return context.window.get_element(element_id)

    async def get_context_window(self, session_id: str) -> Optional[ContextWindow]:
        """
        Get the context window for a session.

        Args:
            session_id: Session identifier

        Returns:
            Context window or None if session not found
        """
        if session_id not in self._contexts:
            return None

        return self._contexts[session_id].window

    async def get_context(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the full context for a session.

        Args:
            session_id: Session identifier

        Returns:
            Context dictionary or None if session not found
        """
        if session_id not in self._contexts:
            return None

        context = self._contexts[session_id]

        # Convert to dictionary
        return context.to_dict()

    async def get_context_dict(
        self, session_id: str, include_entities: bool = True, include_metadata: bool = True
    ) -> Dict[str, Any]:
        """
        Get a simplified context dictionary for a session.

        Args:
            session_id: Session identifier
            include_entities: Whether to include entities
            include_metadata: Whether to include metadata

        Returns:
            Context dictionary (empty if session not found)
        """
        if session_id not in self._contexts:
            return {}

        context = self._contexts[session_id]

        # Build dictionary of current context elements
        result = {}

        # Group elements by type
        grouped_elements = defaultdict(list)
        for element in context.window.elements:
            grouped_elements[element.context_type.value].append(element)

        # Add each context type
        for context_type, elements in grouped_elements.items():
            # Sort by priority and recency
            elements.sort(
                key=lambda e: (
                    -e.priority.value,  # Highest priority first
                    e.updated_at,  # Most recent first
                )
            )

            # For single elements, use direct mapping
            if len(elements) == 1 and isinstance(elements[0].content, (str, dict)):
                result[context_type] = elements[0].content
            else:
                # For multiple elements, create a list
                result[context_type] = [e.content for e in elements]

        # Add entities if requested
        if include_entities and context.entities:
            entity_dict = {}
            for entity_id, entity in context.entities.items():
                entity_dict[entity.name] = {
                    "type": entity.entity_type,
                    "mentions": entity.mention_count,
                    "attributes": entity.attributes,
                    "last_mentioned": entity.last_mentioned_at.isoformat(),
                }
            result["entities"] = entity_dict

        # Add metadata if requested
        if include_metadata and context.metadata:
            result["metadata"] = context.metadata

        return result

    async def update_context_from_text(
        self,
        session_id: str,
        text: str,
        source: str = "user",
        memory_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Update context based on text input.

        This method processes text to extract relevant context information,
        including entities, sentiment, and conversation content.

        Args:
            session_id: Session identifier
            text: Input text
            source: Source of the text
            memory_id: Associated memory ID
            metadata: Additional metadata

        Returns:
            Dictionary of updated context elements
        """
        start_time = time.time()

        # Check if session exists
        if session_id not in self._contexts:
            await self.initialize_context(session_id)

        # Track updated elements
        updated_elements = {}

        try:
            # Add conversation element
            conversation_id = await self.add_context_element(
                session_id=session_id,
                content={
                    "text": text,
                    "source": source,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
                context_type=ContextType.CONVERSATION,
                priority=ContextPriority.HIGH,
                source=source,
                memory_id=memory_id,
                metadata=metadata,
            )

            if conversation_id:
                updated_elements["conversation"] = conversation_id

            # Extract entities if available
            if self.entity_extractor:
                entities = await self.entity_extractor.extract_entities(text)

                for entity in entities:
                    entity_id = await self.add_entity(
                        session_id=session_id,
                        name=entity["text"],
                        entity_type=entity["type"],
                        attributes=entity.get("attributes", {}),
                        linked_memory_ids=[memory_id] if memory_id else [],
                    )

                    if entity_id:
                        if "entities" not in updated_elements:
                            updated_elements["entities"] = []
                        updated_elements["entities"].append(entity_id)

            # Extract sentiment if available
            if self.sentiment_analyzer:
                sentiment = await self.sentiment_analyzer.analyze(text)

                if sentiment:
                    emotion_id = await self.add_context_element(
                        session_id=session_id,
                        content={
                            "sentiment": sentiment["sentiment"],
                            "score": sentiment["score"],
                            "emotions": sentiment.get("emotions", {}),
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        },
                        context_type=ContextType.EMOTIONAL,
                        priority=ContextPriority.MEDIUM,
                        source=source,
                        ttl=timedelta(minutes=30),  # Emotional context expires faster
                    )

                    if emotion_id:
                        updated_elements["emotional"] = emotion_id

            # Extract intent if available
            if self.intent_manager:
                intent = await self.intent_manager.detect_intent(text)

                if intent and intent.get("intent"):
                    task_id = await self.add_context_element(
                        session_id=session_id,
                        content={
                            "intent": intent["intent"],
                            "confidence": intent["confidence"],
                            "parameters": intent.get("parameters", {}),
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        },
                        context_type=ContextType.TASK,
                        priority=ContextPriority.HIGH,
                        source=source,
                    )

                    if task_id:
                        updated_elements["task"] = task_id

            # Update metrics
            if self.metrics:
                self.metrics.record("context_update_time_seconds", time.time() - start_time)

            return updated_elements

        except Exception as e:
            self.logger.error(f"Error updating context from text: {str(e)}")
            traceback.print_exc()
            return updated_elements

    async def update_context_from_memory(
        self, session_id: str, memory_item: MemoryItem
    ) -> Dict[str, Any]:
        """
        Update context based on a memory item.

        Args:
            session_id: Session identifier
            memory_item: Memory item

        Returns:
            Dictionary of updated context elements
        """
        # Check if session exists
        if session_id not in self._contexts:
            await self.initialize_context(session_id)

        # Track updated elements
        updated_elements = {}

        try:
            # Determine context type based on memory type
            context_type = ContextType.CONVERSATION
            if memory_item.memory_type == MemoryType.SEMANTIC:
                context_type = ContextType.FACTUAL

            # Set priority based on importance
            priority = ContextPriority.MEDIUM
            if memory_item.metadata and memory_item.metadata.importance >= 0.8:
                priority = ContextPriority.HIGH
            elif memory_item.metadata and memory_item.metadata.importance <= 0.3:
                priority = ContextPriority.LOW

            # Extract content
            content = memory_item.content

            # Add context element
            element_id = await self.add_context_element(
                session_id=session_id,
                content=content,
                context_type=context_type,
                priority=priority,
                source=f"memory:{memory_item.memory_type.value}",
                memory_id=memory_item.memory_id,
            )

            if element_id:
                updated_elements[context_type.value] = element_id

            # Extract entities if it's a text or has a text field
            text = None
            if isinstance(content, str):
                text = content
            elif isinstance(content, dict) and "text" in content:
                text = content["text"]
            elif isinstance(content, dict) and "content" in content:
                text = content["content"] if isinstance(content["content"], str) else None

            if text and self.entity_extractor:
                entities = await self.entity_extractor.extract_entities(text)

                for entity in entities:
                    entity_id = await self.add_entity(
                        session_id=session_id,
                        name=entity["text"],
                        entity_type=entity["type"],
                        attributes=entity.get("attributes", {}),
                        linked_memory_ids=[memory_item.memory_id],
                    )

                    if entity_id:
                        if "entities" not in updated_elements:
                            updated_elements["entities"] = []
                        updated_elements["entities"].append(entity_id)

            return updated_elements

        except Exception as e:
            self.logger.error(f"Error updating context from memory: {str(e)}")
            traceback.print_exc()
            return updated_elements

    async def add_entity(
        self,
        session_id: str,
        name: str,
        entity_type: str,
        attributes: Optional[Dict[str, Any]] = None,
        confidence: float = 1.0,
        linked_memory_ids: Optional[List[str]] = None,
    ) -> Optional[str]:
        """
        Add an entity to the context.

        Args:
            session_id: Session identifier
            name: Entity name
            entity_type: Entity type
            attributes: Entity attributes
            confidence: Detection confidence
            linked_memory_ids: Associated memory IDs

        Returns:
            Entity identifier if added, None if failed
        """
        # Check if session exists
        if session_id not in self._contexts:
            self.logger.warning(f"Cannot add entity: session {session_id} not found")
            return None

        # Get context
        context = self._contexts[session_id]

        # Check if entity already exists
        existing = context.find_entity_by_name(name)
        if existing:
            # Update existing entity
            existing.update_mention(attributes or {})

            # Add linked memories
            if linked_memory_ids:
                for memory_id in linked_memory_ids:
                    if memory_id not in existing.linked_memory_ids:
                        existing.linked_memory_ids.append(memory_id)

            # Update entity in context
            context.add_entity(existing)

            # Update cache
            self._entity_cache[existing.entity_id] = existing

            # Emit event
            await self.event_bus.emit(
                ContextEntityDetected(
                    session_id=session_id,
                    entity_id=existing.entity_id,
                    entity_type=entity_type,
                    entity_name=name,
                    operation="update",
                )
            )

            return existing.entity_id

        # Create new entity
        entity_id = str(uuid.uuid4())
        entity = ContextEntity(
            entity_id=entity_id,
            name=name,
            entity_type=entity_type,
            first_mentioned_at=datetime.now(timezone.utc),
            last_mentioned_at=datetime.now(timezone.utc),
            attributes=attributes or {},
            linked_memory_ids=linked_memory_ids or [],
            confidence=confidence,
        )

        # Add to context
        context.add_entity(entity)

        # Add to cache
        self._entity_cache[entity_id] = entity

        # Emit event
        await self.event_bus.emit(
            ContextEntityDetected(
                session_id=session_id,
                entity_id=entity_id,
                entity_type=entity_type,
                entity_name=name,
                operation="add",
            )
        )

        # Update metrics
        if self.metrics:
            self.metrics.increment("context_entities_detected")
            self.metrics.gauge(
                "context_entity_count", len(context.entities), {"session_id": session_id}
            )

        return entity_id

    async def remove_entity(self, session_id: str, entity_id: str) -> bool:
        """
        Remove an entity from the context.

        Args:
            session_id: Session identifier
            entity_id: Entity identifier

        Returns:
            True if removed, False if not found
        """
        # Check if session exists
        if session_id not in self._contexts:
            self.logger.warning(f"Cannot remove entity: session {session_id} not found")
            return False

        # Get context
        context = self._contexts[session_id]

        # Get entity before removal
        entity = context.get_entity(entity_id)

        # Remove entity
        if not context.remove_entity(entity_id):
            return False

        # Remove from cache
        if entity_id in self._entity_cache:
            del self._entity_cache[entity_id]

        # Emit event
        if entity:
            await self.event_bus.emit(
                ContextEntityRemoved(
                    session_id=session_id,
                    entity_id=entity_id,
                    entity_type=entity.entity_type,
                    entity_name=entity.name,
                )
            )

        # Update metrics
        if self.metrics:
            self.metrics.gauge(
                "context_entity_count", len(context.entities), {"session_id": session_id}
            )

        return True

    async def get_entity(self, session_id: str, entity_id: str) -> Optional[ContextEntity]:
        """
        Get an entity from the context.

        Args:
            session_id: Session identifier
            entity_id: Entity identifier

        Returns:
            Context entity or None if not found
        """
        # Check cache first
        if entity_id in self._entity_cache:
            entity = self._entity_cache[entity_id]
            # Verify session
            if (
                session_id
                == self._contexts.get(
                    session_id, ConversationContext(session_id="", window=None)
                ).session_id
            ):
                return entity

        # Check if session exists
        if session_id not in self._contexts:
            return None

        # Get context
        context = self._contexts[session_id]

        # Get entity
        return context.get_entity(entity_id)

    async def find_entity_by_name(self, session_id: str, name: str) -> Optional[ContextEntity]:
        """
        Find an entity by name.

        Args:
            session_id: Session identifier
            name: Entity name

        Returns:
            Context entity or None if not found
        """
        # Check if session exists
        if session_id not in self._contexts:
            return None

        # Get context
        context = self._contexts[session_id]

        # Find entity
        return context.find_entity_by_name(name)

    async def prune_context(self, session_id: str, min_relevance: Optional[float] = None) -> int:
        """
        Prune low-relevance elements from the context.

        Args:
            session_id: Session identifier
            min_relevance: Minimum relevance to keep

        Returns:
            Number of elements removed
        """
        # Check if session exists
        if session_id not in self._contexts:
            return 0

        # Get context
        context = self._contexts[session_id]

        # Prune context window
        removed = context.window.prune(min_relevance or self.min_relevance_threshold)

        if removed > 0:
            # Update context
            context.update()

            # Emit event
            await self.event_bus.emit(
                ContextWindowChanged(
                    session_id=session_id,
                    window_id=context.window.window_id,
                    operation="prune",
                    element_count=len(context.window.elements),
                )
            )

            # Update metrics
            if self.metrics:
                self.metrics.gauge(
                    "context_window_size", len(context.window.elements), {"session_id": session_id}
                )

        return removed

    async def clear_context(self, session_id: str) -> bool:
        """
        Clear the context for a session.

        Args:
            session_id: Session identifier

        Returns:
            True if cleared, False if session not found
        """
        # Check if session exists
        if session_id not in self._contexts:
            return False

        # Get context
        context = self._contexts[session_id]

        # Clear context window
        context.window.clear()

        # Clear entities
        context.entities.clear()

        # Update context
        context.update()

        # Emit event
        await self.event_bus.emit(
            ContextCleared(session_id=session_id, window_id=context.window.window_id)
        )

        # Update metrics
        if self.metrics:
            self.metrics.gauge("context_window_size", 0, {"session_id": session_id})
            self.metrics.gauge("context_entity_count", 0, {"session_id": session_id})

        return True

    async def save_context(self, session_id: str) -> Optional[str]:
        """
        Save the current context to memory.

        Args:
            session_id: Session identifier

        Returns:
            Memory ID if saved, None if failed
        """
        # Check if session exists
        if session_id not in self._contexts:
            return None

        # Get context
        context = self._contexts[session_id]

        try:
            # Convert context to dictionary
            context_dict = context.to_dict()

            # Store in working memory
            memory_id = await self.working_memory.store(
                data=context_dict,
                session_id=session_id,
                tags={"context", "saved_context"},
                priority=0.7,
            )

            self.logger.info(f"Saved context for session {session_id} to memory {memory_id}")
            return memory_id

        except Exception as e:
            self.logger.error(f"Failed to save context: {str(e)}")
            return None

    async def restore_context(self, session_id: str, memory_id: str) -> bool:
        """
        Restore context from a saved memory.

        Args:
            session_id: Session identifier
            memory_id: Memory identifier

        Returns:
            True if restored, False if failed
        """
        try:
            # Retrieve memory
            memory = await self.working_memory.retrieve(memory_id)
            if not memory:
                # Try episodic memory
                memory = await self.episodic_memory.retrieve(memory_id)

            if not memory or not isinstance(memory.content, dict):
                self.logger.warning(
                    f"Cannot restore context: memory {memory_id} not found or invalid format"
                )
                return False

            # Clear existing context
            await self.clear_context(session_id)

            # Initialize new context if needed
            if session_id not in self._contexts:
                await self.initialize_context(session_id)

            # Get context
            context = self._contexts[session_id]

            # Restore window elements
            if "window" in memory.content and "elements" in memory.content["window"]:
                for element_data in memory.content["window"]["elements"]:
                    # Create element
                    element = ContextElement(
                        element_id=element_data.get("element_id", str(uuid.uuid4())),
                        content=element_data.get("content"),
                        context_type=ContextType(element_data.get("context_type", "conversation")),
                        priority=ContextPriority(element_data.get("priority", 1)),
                        created_at=datetime.fromisoformat(
                            element_data.get("created_at", datetime.now(timezone.utc).isoformat())
                        ),
                        updated_at=datetime.fromisoformat(
                            element_data.get("updated_at", datetime.now(timezone.utc).isoformat())
                        ),
                        source=element_data.get("source"),
                        memory_id=element_data.get("memory_id"),
                        relevance=element_data.get("relevance", 1.0),
                        ttl=(
                            timedelta(seconds=int(element_data.get("ttl", 0)))
                            if element_data.get("ttl")
                            else None
                        ),
                        metadata=element_data.get("metadata", {}),
                    )

                    # Add to window
                    context.window.add_element(element)

                    # Add to cache
                    self._element_cache[element.element_id] = element

            # Restore entities
            if "entities" in memory.content:
                for entity_id, entity_data in memory.content["entities"].items():
                    # Create entity
                    entity = ContextEntity(
                        entity_id=entity_id,
                        name=entity_data.get("name", ""),
                        entity_type=entity_data.get("entity_type", "unknown"),
                        first_mentioned_at=datetime.fromisoformat(
                            entity_data.get(
                                "first_mentioned_at", datetime.now(timezone.utc).isoformat()
                            )
                        ),
                        last_mentioned_at=datetime.fromisoformat(
                            entity_data.get(
                                "last_mentioned_at", datetime.now(timezone.utc).isoformat()
                            )
                        ),
                        mention_count=entity_data.get("mention_count", 1),
                        attributes=entity_data.get("attributes", {}),
                        linked_memory_ids=entity_data.get("linked_memory_ids", []),
                        confidence=entity_data.get("confidence", 1.0),
                    )

                    # Add to context
                    context.add_entity(entity)

                    # Add to cache
                    self._entity_cache[entity.entity_id] = entity

            # Update metadata
            if "metadata" in memory.content:
                context.metadata = memory.content["metadata"]

            # Update context
            context.update()

            # Emit event
            await self.event_bus.emit(
                ContextRestored(
                    session_id=session_id,
                    window_id=context.window.window_id,
                    memory_id=memory_id,
                    element_count=len(context.window.elements),
                    entity_count=len(context.entities),
                )
            )

            # Update metrics
            if self.metrics:
                self.metrics.gauge(
                    "context_window_size", len(context.window.elements), {"session_id": session_id}
                )
                self.metrics.gauge(
                    "context_entity_count", len(context.entities), {"session_id": session_id}
                )

            self.logger.info(f"Restored context for session {session_id} from memory {memory_id}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to restore context: {str(e)}")
            traceback.print_exc()
            return False

    async def get_memories_for_context(
        self, context: Dict[str, Any], session_id: Optional[str] = None, limit: int = 5
    ) -> List[MemoryItem]:
        """
        Get memories relevant to a specific context.

        Args:
            context: Context information
            session_id: Optional session identifier
            limit: Maximum number of memories to retrieve

        Returns:
            List of relevant memory items
        """
        # Start with an empty result
        result = []

        try:
            # Extract key entities from context
            entities = []
            if "entities" in context:
                entities = (
                    [entity["name"] for entity in context["entities"].values()]
                    if isinstance(context["entities"], dict)
                    else context["entities"]
                )

            # Extract conversation content
            conversation = None
            if "conversation" in context:
                if isinstance(context["conversation"], list):
                    conversation = " ".join(
                        [
                            c["text"] if isinstance(c, dict) and "text" in c else str(c)
                            for c in context["conversation"][-3:]  # Last 3 conversation items
                        ]
                    )
                elif (
                    isinstance(context["conversation"], dict) and "text" in context["conversation"]
                ):
                    conversation = context["conversation"]["text"]
                else:
                    conversation = str(context["conversation"])

            # Get task/intent if available
            task = None
            if "task" in context:
                if isinstance(context["task"], dict) and "intent" in context["task"]:
                    task = context["task"]["intent"]
                else:
                    task = str(context["task"])

            # Compose query from context
            query_parts = []
            if conversation:
                query_parts.append(conversation)
            if task:
                query_parts.append(f"task: {task}")
            if entities:
                query_parts.append(f"entities: {', '.join(entities)}")

            query = " ".join(query_parts)

            # If we have a vector store, use it for semantic search
            if self.vector_store and query:
                # Generate query embedding
                if self.model_router:
                    query_embedding = await self.model_router.get_embeddings(query)

                    # Define filters
                    filters = {}
                    if session_id:
                        filters["session_id"] = session_id

                    # Perform vector search
                    vector_results = await self.vector_store.similarity_search(
                        query_vector=query_embedding,
                        similarity_threshold=0.7,
                        top_k=limit,
                        filters=filters,
                    )

                    # Add to results
                    result.extend(vector_results)

            # If we have a graph store, use it for relationship-based retrieval
            if self.graph_store and entities and len(result) < limit:
                for entity_name in entities:
                    # Find entity in graph
                    entity_node = None

                    # Get current session context
                    if session_id and session_id in self._contexts:
                        # Check if entity exists in current context
                        entity = self._contexts[session_id].find_entity_by_name(entity_name)
                        if entity and entity.linked_memory_ids:
                            # Get memories linked to this entity
                            for memory_id in entity.linked_memory_ids:
                                memory = None

                                # Try working memory first
                                memory = await self.working_memory.retrieve(memory_id)
                                if not memory:
                                    # Try episodic memory
                                    memory = await self.episodic_memory.retrieve(memory_id)

                                if memory and memory not in result:
                                    result.append(memory)

                                    # Stop if we have enough results
                                    if len(result) >= limit:
                                        break

            # If we still need more results, query working memory directly
            if session_id and len(result) < limit:
                # Get recent items from working memory
                working_items = await self.working_memory.get_recent_items(
                    session_id, limit=limit - len(result)
                )

                # Add unique items
                for item in working_items:
                    if item not in result:
                        result.append(item)

            # If we have a task/intent, try to get semantic memories
            if task and len(result) < limit:
                # Get semantic memories related to the task
                semantic_facts = await self.semantic_memory.retrieve_relevant(
                    task, context=context, limit=limit - len(result)
                )

                # Process semantic facts
                for fact_data in semantic_facts:
                    if "id" in fact_data:
                        memory = await self.semantic_memory.retrieve(fact_data["id"])
                        if memory and memory not in result:
                            result.append(memory)

            # Return all relevant memories
            return result[:limit]

        except Exception as e:
            self.logger.error(f"Error retrieving memories for context: {str(e)}")
            traceback.print_exc()
            return result

    async def _extract_entities_from_element(
        self, element: ContextElement, session_id: str
    ) -> List[str]:
        """
        Extract entities from a context element.

        Args:
            element: Context element
            session_id: Session identifier

        Returns:
            List of entity IDs
        """
        if not self.entity_extractor:
            return []

        entity_ids = []

        try:
            # Extract text from element
            text = None
            if isinstance(element.content, str):
                text = element.content
            elif isinstance(element.content, dict):
                if "text" in element.content:
                    text = element.content["text"]
                elif "content" in element.content:
                    text = (
                        element.content["content"]
                        if isinstance(element.content["content"], str)
                        else None
                    )
                elif "summary" in element.content:
                    text = element.content["summary"]

            if not text:
                return []

            # Extract entities
            entities = await self.entity_extractor.extract_entities(text)

            # Add entities to context
            for entity in entities:
                entity_id = await self.add_entity(
                    session_id=session_id,
                    name=entity["text"],
                    entity_type=entity["type"],
                    attributes=entity.get("attributes", {}),
                    linked_memory_ids=[element.memory_id] if element.memory_id else [],
                )

                if entity_id:
                    entity_ids.append(entity_id)

            return entity_ids

        except Exception as e:
            self.logger.error(f"Error extracting entities from element: {str(e)}")
            return []

    async def cleanup_session(self, session_id: str) -> bool:
        """
        Clean up context for a session.

        Args:
            session_id: Session identifier

        Returns:
            True if cleaned up, False if session not found
        """
        # Check if session exists
        if session_id not in self._contexts:
            return False

        # Save context first
        await self.save_context(session_id)

        # Get context
        context = self._contexts[session_id]

        # Clear caches
        for element in context.window.elements:
            if element.element_id in self._element_cache:
                del self._element_cache[element.element_id]

        for entity_id in context.entities:
            if entity_id in self._entity_cache:
                del self._entity_cache[entity_id]

        # Remove context
        del self._contexts[session_id]

        # Clear update counter
        if session_id in self._update_counter:
            del self._update_counter[session_id]

        # Update metrics
        if self.metrics:
            self.metrics.gauge("context_window_size", 0, {"session_id": session_id})
            self.metrics.gauge("context_entity_count", 0, {"session_id": session_id})

        self.logger.info(f"Cleaned up context for session {session_id}")
        return True

    def get_session_ids(self) -> List[str]:
        """
        Get IDs of all active sessions.

        Returns:
            List of session IDs
        """
        return list(self._contexts.keys())

    def get_stats(self) -> Dict[str, Any]:
        """
        Get context manager statistics.

        Returns:
            Dictionary of statistics
        """
        stats = {
            "active_sessions": len(self._contexts),
            "element_cache_size": len(self._element_cache),
            "entity_cache_size": len(self._entity_cache),
            "sessions": {},
        }

        # Add per-session stats
        for session_id, context in self._contexts.items():
            stats["sessions"][session_id] = {
                "window_size": len(context.window.elements),
                "entity_count": len(context.entities),
                "created_at": context.created_at.isoformat(),
                "updated_at": context.updated_at.isoformat(),
            }

        return stats
