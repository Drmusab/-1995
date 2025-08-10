"""
Working Memory Implementation
Author: Drmusab
Created: 2025-08-09

This module provides the working memory implementation for the AI assistant.
Working memory stores current, active information that is immediately relevant
to ongoing tasks and interactions. It has limited capacity and implements
recency-based forgetting mechanisms.

Key Features:
- Fast access to recent and frequently used items
- Limited capacity with automatic forgetting
- Prioritization based on importance and recency
- Integration with attention mechanisms
- Automatic consolidation into long-term memory
"""

import asyncio
import json
import logging
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

# Core imports
from src.core.config.loader import ConfigLoader
from src.core.dependency_injection import Container
from src.core.error_handling import ErrorHandler, handle_exceptions
from src.core.events.event_bus import EventBus
from src.core.events.event_types import (
    MemoryConsolidationCompleted,
    MemoryConsolidationStarted,
    MemoryItemDeleted,
    MemoryItemRetrieved,
    MemoryItemStored,
    MemoryItemUpdated,
)
from src.core.health_check import HealthCheck
from src.core.security.encryption import EncryptionManager

# Integration components
from src.integrations.llm.model_router import ModelRouter
from src.integrations.storage.database import DatabaseManager
from src.integrations.cache.cache_strategy import CacheStrategy

# Memory system imports
from src.memory.core_memory.base_memory import (
    BaseMemory,
    BaseMemoryStore,
    MemoryAccess,
    MemoryAccessError,
    MemoryError,
    MemoryItem,
    MemoryMetadata,
    MemoryNotFoundError,
    MemoryRetentionPolicy,
    MemorySensitivity,
    MemoryStorageType,
    MemoryType,
    MemoryUtils,
    SimpleMemoryQuery,
    memory_operation_span,
)
from src.memory.operations.consolidation import ConsolidationEngine
from src.memory.operations.context_manager import MemoryContextManager
from src.memory.storage.memory_graph import MemoryGraphStore
from src.memory.storage.vector_store import VectorMemoryStore

# Observability
from src.observability.logging.config import get_logger
from src.observability.monitoring.metrics import MetricsCollector
from src.observability.monitoring.tracing import TraceManager

# Learning integration
from src.learning.memory_learning_bridge import MemoryLearningBridge


@dataclass
class WorkingMemoryConfig:
    """Configuration for working memory."""
    max_capacity: int = 100
    priority_levels: int = 3
    cache_size: int = 100
    decay_rate: float = 0.1
    consolidation_threshold: float = 0.8
    attention_window: int = 10
    recency_weight: float = 0.3
    importance_weight: float = 0.5
    access_frequency_weight: float = 0.2


@dataclass
class AttentionFocus:
    """Represents current attention focus in working memory."""
    primary_focus: Optional[str] = None
    secondary_focuses: List[str] = field(default_factory=list)
    context_items: Dict[str, float] = field(default_factory=dict)  # memory_id -> relevance
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class WorkingMemory(BaseMemory):
    """
    Working memory implementation - stores current, active information.
    
    This class provides the core working memory functionality for the AI assistant,
    managing short-term storage of immediately relevant information with automatic
    forgetting and consolidation mechanisms.
    """

    def __init__(
        self,
        container: Container,
        memory_store: BaseMemoryStore,
        config: Optional[WorkingMemoryConfig] = None,
    ):
        """
        Initialize working memory.

        Args:
            container: Dependency injection container
            memory_store: Memory storage backend
            config: Working memory configuration
        """
        super().__init__()
        self.container = container
        self.memory_store = memory_store
        self.config = config or WorkingMemoryConfig()
        self.logger = get_logger(__name__)

        # Get core services
        self.event_bus = container.get(EventBus)
        self.error_handler = container.get(ErrorHandler)
        
        # Get optional services
        try:
            self.metrics = container.get(MetricsCollector)
            self.tracer = container.get(TraceManager)
        except Exception:
            self.metrics = None
            self.tracer = None
            
        try:
            self.cache_strategy = container.get(CacheStrategy)
        except Exception:
            self.cache_strategy = None
            
        try:
            self.consolidation_engine = container.get(ConsolidationEngine)
        except Exception:
            self.consolidation_engine = None
            
        try:
            self.context_manager = container.get(MemoryContextManager)
        except Exception:
            self.context_manager = None
            
        try:
            self.learning_bridge = container.get(MemoryLearningBridge)
        except Exception:
            self.learning_bridge = None

        # Initialize internal data structures
        self._session_memory: Dict[str, Dict[int, List[str]]] = defaultdict(
            lambda: defaultdict(list)
        )
        self._access_counts: Dict[str, int] = defaultdict(int)
        self._consolidation_candidates: Set[str] = set()
        self._attention_focus: Dict[str, AttentionFocus] = {}  # session_id -> focus
        
        # Cache management
        self._recent_access_cache: Dict[str, MemoryItem] = {}
        self._cache_order: deque = deque(maxlen=self.config.cache_size)
        
        # Performance optimization
        self._batch_update_queue: List[Tuple[str, Dict[str, Any]]] = []
        self._last_cleanup_time: Dict[str, datetime] = {}
        
        # Register health check
        self._register_health_check()
        
        self.logger.info(
            f"WorkingMemory initialized with capacity={self.config.max_capacity}, "
            f"priority_levels={self.config.priority_levels}"
        )

    def _register_health_check(self) -> None:
        """Register health check for working memory."""
        try:
            health_check = self.container.get(HealthCheck)
            health_check.register_check(
                "working_memory",
                self._health_check,
                critical=True
            )
        except Exception:
            pass

    async def _health_check(self) -> Tuple[bool, str]:
        """
        Perform health check on working memory.
        
        Returns:
            Tuple of (is_healthy, status_message)
        """
        try:
            # Check memory store connectivity
            test_id = f"health_check_{uuid.uuid4().hex[:8]}"
            test_item = MemoryItem(
                memory_id=test_id,
                content={"test": "health_check"},
                memory_type=MemoryType.WORKING,
                metadata=MemoryMetadata()
            )
            
            await self.memory_store.store_item(test_item)
            retrieved = await self.memory_store.get_item(test_id)
            await self.memory_store.delete_item(test_id)
            
            if not retrieved:
                return False, "Failed to retrieve test item"
                
            # Check memory capacity
            total_items = sum(
                sum(len(items) for items in session.values())
                for session in self._session_memory.values()
            )
            
            if total_items > self.config.max_capacity * 1.5:
                return False, f"Memory overloaded: {total_items} items"
                
            return True, f"Healthy - {total_items} items in memory"
            
        except Exception as e:
            return False, f"Health check failed: {str(e)}"

    async def initialize_session(self, session_id: str) -> None:
        """
        Initialize working memory for a session.
        
        Args:
            session_id: Session identifier
        """
        # Clear existing session memory
        if session_id in self._session_memory:
            await self.cleanup_session(session_id)
            
        # Initialize priority queues
        self._session_memory[session_id] = defaultdict(list)
        
        # Initialize attention focus
        self._attention_focus[session_id] = AttentionFocus()
        
        # Set cleanup time
        self._last_cleanup_time[session_id] = datetime.now(timezone.utc)
        
        # Emit event
        await self.event_bus.emit(
            MemoryItemStored(
                memory_id=f"session_init_{session_id}",
                memory_type=MemoryType.WORKING.value,
                context_id=session_id,
            )
        )
        
        self.logger.debug(f"Initialized working memory for session {session_id}")

    @handle_exceptions
    async def store(self, data: Any, **kwargs) -> str:
        """
        Store data in working memory.

        Args:
            data: Data to store
            **kwargs: Additional parameters including:
                session_id: Session identifier (required)
                priority: Item priority (0-1, higher is more important)
                owner_id: Owner identifier
                context_id: Context identifier
                tags: Memory tags
                attention_boost: Whether to boost attention focus

        Returns:
            Memory ID

        Raises:
            MemoryError: If session_id is not provided
        """
        session_id = kwargs.get("session_id")
        if not session_id:
            raise MemoryError("Session ID required for working memory storage")

        # Generate memory ID
        memory_id = MemoryUtils.generate_memory_id()

        # Calculate priority level
        priority = kwargs.get("priority", 0.5)
        priority_level = min(
            self.config.priority_levels - 1,
            int(priority * self.config.priority_levels)
        )

        # Create memory item
        metadata = MemoryMetadata(
            retention_policy=MemoryRetentionPolicy.TRANSIENT,
            tags=kwargs.get("tags", set()),
            importance=priority,
            custom_metadata={
                "session_id": session_id,
                "priority_level": priority_level,
                "attention_boost": kwargs.get("attention_boost", False),
            }
        )

        memory_item = MemoryItem(
            memory_id=memory_id,
            content=data,
            memory_type=MemoryType.WORKING,
            owner_id=kwargs.get("owner_id"),
            session_id=session_id,
            context_id=kwargs.get("context_id"),
            metadata=metadata,
        )

        # Check capacity and apply forgetting if needed
        await self._check_capacity(session_id)

        # Store in memory store with tracing
        async with memory_operation_span(self.tracer, "store_working_memory", memory_id):
            await self.memory_store.store_item(memory_item)

        # Add to priority queue
        self._session_memory[session_id][priority_level].append(memory_id)

        # Update attention focus if requested
        if kwargs.get("attention_boost"):
            await self._update_attention_focus(session_id, memory_id, priority)

        # Add to cache
        self._add_to_cache(memory_item)

        # Track for learning if bridge available
        if self.learning_bridge:
            await self.learning_bridge.track_memory_usage(
                memory_id=memory_id,
                memory_type=MemoryType.WORKING,
                operation="store",
                context={"session_id": session_id, "priority": priority}
            )

        # Emit event
        await self.event_bus.emit(
            MemoryItemStored(
                memory_id=memory_id,
                memory_type=MemoryType.WORKING.value,
                owner_id=kwargs.get("owner_id"),
                context_id=kwargs.get("context_id"),
            )
        )

        # Update metrics
        if self.metrics:
            self.metrics.increment("working_memory_items_stored")
            session_items = sum(
                len(items) for items in self._session_memory[session_id].values()
            )
            self.metrics.gauge(
                "working_memory_items_per_session",
                session_items,
                {"session_id": session_id}
            )

        return memory_id

    async def retrieve(self, memory_id: str) -> Optional[MemoryItem]:
        """
        Retrieve an item from working memory.

        Args:
            memory_id: Memory identifier

        Returns:
            Memory item or None if not found
        """
        # Check cache first
        if memory_id in self._recent_access_cache:
            item = self._recent_access_cache[memory_id]
            self._access_counts[memory_id] += 1
            item.metadata.update_access()
            
            # Track for learning
            if self.learning_bridge:
                await self.learning_bridge.track_memory_usage(
                    memory_id=memory_id,
                    memory_type=MemoryType.WORKING,
                    operation="retrieve",
                    context={"cache_hit": True}
                )
            
            return item

        # Retrieve from store with tracing
        async with memory_operation_span(self.tracer, "retrieve_working_memory", memory_id):
            item = await self.memory_store.get_item(memory_id)

        if item:
            # Update access statistics
            self._access_counts[memory_id] += 1
            item.metadata.update_access()
            
            # Update store with new metadata
            await self.memory_store.update_item(memory_id, {"metadata": item.metadata})
            
            # Add to cache
            self._add_to_cache(item)
            
            # Update attention focus based on access
            if item.session_id:
                await self._update_attention_on_access(item.session_id, memory_id)
            
            # Track for learning
            if self.learning_bridge:
                await self.learning_bridge.track_memory_usage(
                    memory_id=memory_id,
                    memory_type=MemoryType.WORKING,
                    operation="retrieve",
                    context={"cache_hit": False}
                )
            
            # Emit event
            await self.event_bus.emit(
                MemoryItemRetrieved(
                    memory_id=memory_id,
                    memory_type=MemoryType.WORKING.value,
                    owner_id=item.owner_id,
                )
            )
            
            # Update metrics
            if self.metrics:
                self.metrics.increment("working_memory_items_retrieved")

        return item

    async def update(self, session_id: str, data: Dict[str, Any]) -> None:
        """
        Update working memory with new data.

        Args:
            session_id: Session identifier
            data: Data to update
        """
        if isinstance(data, dict) and "last_interaction" in data:
            # Store latest interaction data
            interaction_data = data.get("last_interaction", {})
            await self.store(
                interaction_data,
                session_id=session_id,
                priority=0.8,  # High priority for recent interactions
                context_id=interaction_data.get("context_id"),
                tags={"interaction", "recent"},
                attention_boost=True,
            )

            # Update context information
            if "context" in data:
                context_data = data.get("context", {})
                await self.store(
                    context_data,
                    session_id=session_id,
                    priority=0.7,
                    tags={"context", "session_state"},
                )

        elif isinstance(data, MemoryItem):
            # Store the memory item directly
            await self.memory_store.store_item(data)

            # Update priority queues if it's a working memory item
            if data.memory_type == MemoryType.WORKING and data.session_id:
                priority = data.metadata.importance
                priority_level = min(
                    self.config.priority_levels - 1,
                    int(priority * self.config.priority_levels)
                )

                # Remove from existing priority level
                for level in range(self.config.priority_levels):
                    if data.memory_id in self._session_memory[data.session_id][level]:
                        self._session_memory[data.session_id][level].remove(data.memory_id)

                # Add to new priority level
                self._session_memory[data.session_id][priority_level].append(data.memory_id)

        else:
            # Store generic data
            await self.store(
                data,
                session_id=session_id,
                priority=0.5,
                tags={"update"}
            )

        # Emit update event
        await self.event_bus.emit(
            MemoryItemUpdated(
                memory_id=f"update_{session_id}_{int(time.time())}",
                memory_type=MemoryType.WORKING.value,
                context_id=session_id,
            )
        )

    async def search(self, query: Any) -> List[MemoryItem]:
        """
        Search working memory.

        Args:
            query: Search query

        Returns:
            List of matching memory items
        """
        if isinstance(query, SimpleMemoryQuery):
            # Use memory store query capabilities
            return await self.memory_store.query(query)

        elif isinstance(query, str):
            # Parse query string
            if query.startswith("session:"):
                # Get memories for specific session
                session_id = query.split(":", 1)[1].strip()
                return await self.get_session_memories(session_id)
            elif query.startswith("priority:"):
                # Get memories by priority
                priority = query.split(":", 1)[1].strip()
                return await self.get_by_priority(priority)
            else:
                # Text search across all working memory
                return await self._text_search(query)

        elif isinstance(query, dict):
            # Complex query with filters
            return await self._complex_search(query)

        else:
            raise MemoryError(f"Unsupported query type for working memory: {type(query)}")

    async def get_recent_items(
        self,
        session_id: str,
        limit: int = 10
    ) -> List[MemoryItem]:
        """
        Get most recent working memory items for a session.

        Args:
            session_id: Session identifier
            limit: Maximum number of items to return

        Returns:
            List of recent memory items
        """
        if session_id not in self._session_memory:
            return []

        # Collect items from all priority levels (highest priority first)
        memory_ids = []
        for priority_level in range(self.config.priority_levels - 1, -1, -1):
            memory_ids.extend(self._session_memory[session_id][priority_level])

        # Retrieve items
        items = []
        for memory_id in memory_ids[:limit * 2]:  # Get extra to account for failures
            item = await self.retrieve(memory_id)
            if item:
                items.append(item)
                if len(items) >= limit:
                    break

        # Sort by recency
        items.sort(
            key=lambda x: x.metadata.last_accessed or x.metadata.created_at,
            reverse=True
        )

        return items[:limit]

    async def get_most_relevant(
        self,
        session_id: str,
        context: Dict[str, Any],
        limit: int = 5
    ) -> List[MemoryItem]:
        """
        Get most relevant working memory items for a context.

        Args:
            session_id: Session identifier
            context: Context information
            limit: Maximum number of items to return

        Returns:
            List of relevant memory items
        """
        if session_id not in self._session_memory:
            return []

        # Get all memory items for session
        all_items = []
        for priority_level in range(self.config.priority_levels - 1, -1, -1):
            for memory_id in self._session_memory[session_id][priority_level]:
                item = await self.retrieve(memory_id)
                if item:
                    all_items.append(item)

        # Calculate relevance scores
        scored_items = []
        for item in all_items:
            score = self._calculate_relevance_score(item, context, session_id)
            scored_items.append((item, score))

        # Sort by relevance and return top items
        scored_items.sort(key=lambda x: x[1], reverse=True)
        return [item for item, _ in scored_items[:limit]]

    def _calculate_relevance_score(
        self,
        item: MemoryItem,
        context: Dict[str, Any],
        session_id: str
    ) -> float:
        """
        Calculate relevance score for a memory item.

        Args:
            item: Memory item
            context: Context information
            session_id: Session identifier

        Returns:
            Relevance score (0-1)
        """
        # Base score from importance
        score = item.metadata.importance * self.config.importance_weight

        # Recency factor
        if item.metadata.last_accessed:
            time_diff = (
                datetime.now(timezone.utc) - item.metadata.last_accessed
            ).total_seconds()
            recency_score = max(0, 1.0 - (time_diff / 3600))  # Decay over an hour
            score += recency_score * self.config.recency_weight

        # Access frequency factor
        access_count = self._access_counts.get(item.memory_id, 0)
        frequency_score = min(1.0, access_count / 10)  # Normalize to 0-1
        score += frequency_score * self.config.access_frequency_weight

        # Context match factor
        if context.get("context_id") and item.context_id == context.get("context_id"):
            score += 0.2

        # Attention focus factor
        if session_id in self._attention_focus:
            focus = self._attention_focus[session_id]
            if item.memory_id == focus.primary_focus:
                score += 0.3
            elif item.memory_id in focus.secondary_focuses:
                score += 0.2
            elif item.memory_id in focus.context_items:
                score += focus.context_items[item.memory_id] * 0.1

        return min(1.0, score)  # Cap at 1.0

    async def get_attention_context(
        self,
        session_id: str,
        limit: int = 10
    ) -> List[MemoryItem]:
        """
        Get items currently in attention focus.

        Args:
            session_id: Session identifier
            limit: Maximum number of items to return

        Returns:
            List of memory items in attention context
        """
        if session_id not in self._attention_focus:
            return []

        focus = self._attention_focus[session_id]
        memory_ids = []

        # Add primary focus
        if focus.primary_focus:
            memory_ids.append(focus.primary_focus)

        # Add secondary focuses
        memory_ids.extend(focus.secondary_focuses)

        # Add context items sorted by relevance
        sorted_context = sorted(
            focus.context_items.items(),
            key=lambda x: x[1],
            reverse=True
        )
        memory_ids.extend([mid for mid, _ in sorted_context])

        # Retrieve items
        items = []
        for memory_id in memory_ids[:limit]:
            item = await self.retrieve(memory_id)
            if item:
                items.append(item)

        return items

    async def clear(self) -> None:
        """Clear all working memory."""
        # Clear all session data
        session_ids = list(self._session_memory.keys())
        for session_id in session_ids:
            await self.cleanup_session(session_id)

        # Clear global data structures
        self._access_counts.clear()
        self._consolidation_candidates.clear()
        self._recent_access_cache.clear()
        self._cache_order.clear()
        self._attention_focus.clear()
        self._last_cleanup_time.clear()

        # Clear underlying store (only working memory items)
        query = SimpleMemoryQuery(memory_type=MemoryType.WORKING)
        items = await self.memory_store.query(query)

        for item in items:
            await self.memory_store.delete_item(item.memory_id)

        self.logger.info(f"Cleared all working memory ({len(items)} items)")

    async def cleanup_session(self, session_id: str) -> None:
        """
        Clean up working memory for a session.

        Args:
            session_id: Session identifier
        """
        if session_id not in self._session_memory:
            return

        # Collect all memory IDs for this session
        memory_ids = []
        for priority_level in range(self.config.priority_levels):
            memory_ids.extend(self._session_memory[session_id][priority_level])

        # Mark for consolidation if engine available
        if self.consolidation_engine:
            self._consolidation_candidates.update(memory_ids)
            # Trigger consolidation asynchronously
            asyncio.create_task(self._consolidate_memories(memory_ids))
        else:
            # Just mark for potential consolidation
            self._consolidation_candidates.update(memory_ids)

        # Remove from session memory
        del self._session_memory[session_id]

        # Remove from attention focus
        if session_id in self._attention_focus:
            del self._attention_focus[session_id]

        # Remove from cleanup time
        if session_id in self._last_cleanup_time:
            del self._last_cleanup_time[session_id]

        # Remove from cache
        for memory_id in memory_ids:
            if memory_id in self._recent_access_cache:
                del self._recent_access_cache[memory_id]
                try:
                    self._cache_order.remove(memory_id)
                except ValueError:
                    pass

        self.logger.info(
            f"Cleaned up working memory for session {session_id} "
            f"({len(memory_ids)} items marked for consolidation)"
        )

    async def _check_capacity(self, session_id: str) -> None:
        """
        Check working memory capacity and apply forgetting if needed.

        Args:
            session_id: Session identifier
        """
        if session_id not in self._session_memory:
            return

        # Count items for this session
        item_count = sum(
            len(items) for items in self._session_memory[session_id].values()
        )

        # Issue warning at threshold
        warning_threshold = int(self.config.max_capacity * self.config.consolidation_threshold)
        if item_count >= warning_threshold:
            await self.event_bus.emit(
#                 MemoryCapacityWarning(
#                     memory_type=MemoryType.WORKING.value,
#                     current_count=item_count,
#                     capacity=self.config.max_capacity,
#                     session_id=session_id,
#                 )
            )

        # Apply forgetting if over capacity
        if item_count >= self.config.max_capacity:
            await self._apply_forgetting(session_id)

    async def _apply_forgetting(self, session_id: str) -> None:
        """
        Apply forgetting mechanism to free up working memory.

        Args:
            session_id: Session identifier
        """
        forgotten_count = 0

        # Start forgetting from lowest priority
        for priority_level in range(self.config.priority_levels):
            if not self._session_memory[session_id][priority_level]:
                continue

            # Calculate how many items to forget
            level_items = self._session_memory[session_id][priority_level]
            forget_count = max(1, int(len(level_items) * self.config.decay_rate))

            # Get least recently accessed items
            items_with_access = []
            for memory_id in level_items:
                item = await self.retrieve(memory_id)
                if item:
                    last_access = item.metadata.last_accessed or item.metadata.created_at
                    items_with_access.append((memory_id, last_access))

            # Sort by last access time (oldest first)
            items_with_access.sort(key=lambda x: x[1])

            # Forget oldest items
            items_to_forget = [mid for mid, _ in items_with_access[:forget_count]]

            for memory_id in items_to_forget:
                # Add to consolidation candidates
                self._consolidation_candidates.add(memory_id)

                # Remove from session memory
                self._session_memory[session_id][priority_level].remove(memory_id)

                # Remove from cache
                if memory_id in self._recent_access_cache:
                    del self._recent_access_cache[memory_id]
                    try:
                        self._cache_order.remove(memory_id)
                    except ValueError:
                        pass

                forgotten_count += 1

            # Check if we've freed enough space
            current_count = sum(
                len(items) for items in self._session_memory[session_id].values()
            )
            if current_count < self.config.max_capacity:
                break

        # Emit decay event
        await self.event_bus.emit(
#             MemoryDecayApplied(
#                 memory_type=MemoryType.WORKING.value,
#                 count=forgotten_count,
#                 session_id=session_id,
#             )
        )

        self.logger.debug(
            f"Applied forgetting to {forgotten_count} items in session {session_id}"
        )

    async def _consolidate_memories(self, memory_ids: List[str]) -> None:
        """
        Consolidate memories to long-term storage.

        Args:
            memory_ids: List of memory IDs to consolidate
        """
        if not self.consolidation_engine:
            return

        try:
            # Emit start event
            await self.event_bus.emit(
                MemoryConsolidationStarted(
                    memory_type=MemoryType.WORKING.value,
                    count=len(memory_ids),
                )
            )

            # Retrieve memory items
            items = []
            for memory_id in memory_ids:
                item = await self.memory_store.get_item(memory_id)
                if item:
                    items.append(item)

            # Perform consolidation
            consolidated_count = await self.consolidation_engine.consolidate(
                items,
                source_type=MemoryType.WORKING,
                target_type=MemoryType.LONG_TERM
            )

            # Remove from consolidation candidates
            self._consolidation_candidates.difference_update(memory_ids)

            # Emit completion event
            await self.event_bus.emit(
                MemoryConsolidationCompleted(
                    memory_type=MemoryType.WORKING.value,
                    count=consolidated_count,
                )
            )

            self.logger.info(f"Consolidated {consolidated_count} memories to long-term storage")

        except Exception as e:
            self.logger.error(f"Error during memory consolidation: {str(e)}")
            if self.error_handler:
                await self.error_handler.handle_error(e, context={
                    "operation": "memory_consolidation",
                    "memory_ids": memory_ids
                })

    async def _update_attention_focus(
        self,
        session_id: str,
        memory_id: str,
        relevance: float
    ) -> None:
        """
        Update attention focus for a session.

        Args:
            session_id: Session identifier
            memory_id: Memory ID to focus on
            relevance: Relevance score
        """
        if session_id not in self._attention_focus:
            self._attention_focus[session_id] = AttentionFocus()

        focus = self._attention_focus[session_id]

        # Update primary focus if relevance is high
        if relevance > 0.8:
            # Move current primary to secondary
            if focus.primary_focus and focus.primary_focus != memory_id:
                focus.secondary_focuses.append(focus.primary_focus)
                # Keep only recent secondary focuses
                if len(focus.secondary_focuses) > 5:
                    focus.secondary_focuses.pop(0)

            focus.primary_focus = memory_id

        # Update context items
        focus.context_items[memory_id] = relevance

        # Maintain attention window size
        if len(focus.context_items) > self.config.attention_window:
            # Remove least relevant items
            sorted_items = sorted(
                focus.context_items.items(),
                key=lambda x: x[1]
            )
            for mid, _ in sorted_items[:-self.config.attention_window]:
                del focus.context_items[mid]

        focus.timestamp = datetime.now(timezone.utc)

    async def _update_attention_on_access(
        self,
        session_id: str,
        memory_id: str
    ) -> None:
        """
        Update attention focus when an item is accessed.

        Args:
            session_id: Session identifier
            memory_id: Accessed memory ID
        """
        if session_id not in self._attention_focus:
            self._attention_focus[session_id] = AttentionFocus()

        focus = self._attention_focus[session_id]

        # Boost relevance for accessed item
        current_relevance = focus.context_items.get(memory_id, 0.5)
        new_relevance = min(1.0, current_relevance + 0.1)

        await self._update_attention_focus(session_id, memory_id, new_relevance)

    def _add_to_cache(self, item: MemoryItem) -> None:
        """
        Add item to recent access cache with LRU eviction.

        Args:
            item: Memory item to cache
        """
        # Remove if already in cache
        if item.memory_id in self._recent_access_cache:
            try:
                self._cache_order.remove(item.memory_id)
            except ValueError:
                pass

        # Add to cache
        self._recent_access_cache[item.memory_id] = item
        self._cache_order.append(item.memory_id)

        # Evict oldest if cache is full
        while len(self._recent_access_cache) > self.config.cache_size:
            if self._cache_order:
                oldest_id = self._cache_order.popleft()
                if oldest_id in self._recent_access_cache:
                    del self._recent_access_cache[oldest_id]

    async def _text_search(self, query: str, limit: int = 10) -> List[MemoryItem]:
        """
        Perform text search across working memory.

        Args:
            query: Search text
            limit: Maximum results

        Returns:
            List of matching items
        """
        # Get all working memory items
        all_query = SimpleMemoryQuery(
            memory_type=MemoryType.WORKING,
            limit=1000
        )
        all_items = await self.memory_store.query(all_query)

        # Filter by query text
        matching_items = []
        query_lower = query.lower()

        for item in all_items:
            content = item.content
            if isinstance(content, dict):
                content = json.dumps(content)
            elif not isinstance(content, str):
                content = str(content)

            if query_lower in content.lower():
                matching_items.append(item)

        # Sort by relevance (access count and recency)
        matching_items.sort(
            key=lambda x: (
                self._access_counts.get(x.memory_id, 0),
                x.metadata.last_accessed or x.metadata.created_at
            ),
            reverse=True
        )

        return matching_items[:limit]

    async def _complex_search(self, query: Dict[str, Any]) -> List[MemoryItem]:
        """
        Perform complex search with multiple filters.

        Args:
            query: Query dictionary with filters

        Returns:
            List of matching items
        """
        # Build SimpleMemoryQuery from complex query
        simple_query = SimpleMemoryQuery(
            memory_type=MemoryType.WORKING,
            owner_id=query.get("owner_id"),
            session_id=query.get("session_id"),
            context_id=query.get("context_id"),
            tags=set(query.get("tags", [])),
            limit=query.get("limit", 100),
        )

        # Get base results
        items = await self.memory_store.query(simple_query)

        # Apply additional filters
        if "min_importance" in query:
            items = [
                item for item in items
                if item.metadata.importance >= query["min_importance"]
            ]

        if "min_access_count" in query:
            items = [
                item for item in items
                if self._access_counts.get(item.memory_id, 0) >= query["min_access_count"]
            ]

        if "created_after" in query:
            cutoff = query["created_after"]
            if isinstance(cutoff, str):
                cutoff = datetime.fromisoformat(cutoff.replace("Z", "+00:00"))
            items = [
                item for item in items
                if item.metadata.created_at >= cutoff
            ]

        return items

    async def get_session_memories(self, session_id: str) -> List[MemoryItem]:
        """
        Get all memories for a session.

        Args:
            session_id: Session identifier

        Returns:
            List of memory items
        """
        if session_id not in self._session_memory:
            return []

        memory_ids = []
        for priority_level in range(self.config.priority_levels - 1, -1, -1):
            memory_ids.extend(self._session_memory[session_id][priority_level])

        items = []
        for memory_id in memory_ids:
            item = await self.retrieve(memory_id)
            if item:
                items.append(item)

        return items

    async def get_by_priority(self, priority: str) -> List[MemoryItem]:
        """
        Get memories by priority level.

        Args:
            priority: Priority level (high/medium/low or numeric)

        Returns:
            List of memory items
        """
        # Convert priority string to level
        if priority.lower() == "high":
            priority_level = self.config.priority_levels - 1
        elif priority.lower() == "medium":
            priority_level = self.config.priority_levels // 2
        elif priority.lower() == "low":
            priority_level = 0
        else:
            try:
                priority_level = int(priority)
            except ValueError:
                return []

        # Validate priority level
        if not 0 <= priority_level < self.config.priority_levels:
            return []

        # Collect items from all sessions at this priority level
        memory_ids = []
        for session_memory in self._session_memory.values():
            memory_ids.extend(session_memory[priority_level])

        # Retrieve items
        items = []
        for memory_id in memory_ids:
            item = await self.retrieve(memory_id)
            if item:
                items.append(item)

        return items

    async def get_stats(self) -> Dict[str, Any]:
        """Get working memory statistics."""
        stats = {
            "total_items": 0,
            "items_by_priority": {},
            "items_by_session": {},
            "consolidation_candidates": len(self._consolidation_candidates),
            "cache_size": len(self._recent_access_cache),
            "cache_hit_rate": 0.0,
            "attention_sessions": len(self._attention_focus),
            "memory_type": MemoryType.WORKING.value,
        }

        # Count items by priority and session
        for session_id, priority_levels in self._session_memory.items():
            session_count = 0
            for priority_level, items in priority_levels.items():
                level_count = len(items)
                if priority_level not in stats["items_by_priority"]:
                    stats["items_by_priority"][priority_level] = 0
                stats["items_by_priority"][priority_level] += level_count
                session_count += level_count

            stats["items_by_session"][session_id] = session_count
            stats["total_items"] += session_count

        # Calculate cache hit rate
        total_accesses = sum(self._access_counts.values())
        if total_accesses > 0:
            # Estimate based on cache size and total items
            stats["cache_hit_rate"] = min(
                1.0,
                len(self._recent_access_cache) / max(1, stats["total_items"])
            )

        # Add attention focus stats
        stats["attention_stats"] = {
            session_id: {
                "primary_focus": focus.primary_focus,
                "secondary_count": len(focus.secondary_focuses),
                "context_size": len(focus.context_items),
            }
            for session_id, focus in self._attention_focus.items()
        }

        return stats

    async def export_session_state(self, session_id: str) -> Dict[str, Any]:
        """
        Export the complete state of a session's working memory.

        Args:
            session_id: Session identifier

        Returns:
            Serializable dictionary of session state
        """
        if session_id not in self._session_memory:
            return {}

        # Get all memory items
        items = await self.get_session_memories(session_id)

        # Get attention focus
        attention = None
        if session_id in self._attention_focus:
            focus = self._attention_focus[session_id]
            attention = {
                "primary_focus": focus.primary_focus,
                "secondary_focuses": focus.secondary_focuses,
                "context_items": dict(focus.context_items),
                "timestamp": focus.timestamp.isoformat(),
            }

        return {
            "session_id": session_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "item_count": len(items),
            "attention_focus": attention,
            "memory_items": [
                {
                    "memory_id": item.memory_id,
                    "content": item.content,
                    "metadata": {
                        "importance": item.metadata.importance,
                        "created_at": item.metadata.created_at.isoformat(),
                        "last_accessed": (
                            item.metadata.last_accessed.isoformat()
                            if item.metadata.last_accessed
                            else None
                        ),
                        "access_count": item.metadata.access_count,
                        "tags": list(item.metadata.tags),
                    },
                    "priority_level": item.metadata.custom_metadata.get("priority_level", 0),
                }
                for item in items
            ],
        }

    async def import_session_state(
        self,
        session_id: str,
        state: Dict[str, Any]
    ) -> None:
        """
        Import a session state into working memory.

        Args:
            session_id: Session identifier
            state: Session state to import
        """
        # Initialize session
        await self.initialize_session(session_id)

        # Import memory items
        for item_data in state.get("memory_items", []):
            await self.store(
                item_data["content"],
                session_id=session_id,
                priority=item_data["metadata"]["importance"],
                tags=set(item_data["metadata"]["tags"]),
            )

        # Import attention focus
        if "attention_focus" in state and state["attention_focus"]:
            focus_data = state["attention_focus"]
            focus = AttentionFocus(
                primary_focus=focus_data.get("primary_focus"),
                secondary_focuses=focus_data.get("secondary_focuses", []),
                context_items=focus_data.get("context_items", {}),
                timestamp=datetime.fromisoformat(focus_data["timestamp"]),
            )
            self._attention_focus[session_id] = focus

        self.logger.info(
            f"Imported session state for {session_id} with "
            f"{len(state.get('memory_items', []))} items"
        )
