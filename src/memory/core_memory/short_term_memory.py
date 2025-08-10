"""
Short-Term Memory Implementation
Author: Drmusab
Last Modified: 2025-08-09 21:30:00 UTC

This module implements short-term memory - a temporary storage system that bridges
working memory and long-term memory. It provides automatic consolidation, importance
scoring, and intelligent memory management for recent experiences and information.
"""

import asyncio
import heapq
import json
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

# Core imports
from src.core.config.loader import ConfigLoader
from src.core.dependency_injection import Container
from src.core.error_handling import ErrorHandler, handle_exceptions
from src.core.events.event_bus import EventBus
from src.core.events.event_types import (
    MemoryCapacityWarning,
    MemoryConsolidationCompleted,
    MemoryConsolidationStarted,
    MemoryDecayApplied,
    MemoryItemDeleted,
    MemoryItemRetrieved,
    MemoryItemStored,
)

# Memory system imports
from src.memory.core_memory.base_memory import (
    BaseMemory,
    BaseMemoryStore,
    MemoryError,
    MemoryItem,
    MemoryMetadata,
    MemoryRetentionPolicy,
    MemoryType,
    MemoryUtils,
    SimpleMemoryQuery,
    memory_operation_span,
)
from src.memory.operations.consolidation import MemoryConsolidation

# Integration components
from src.integrations.llm.model_router import ModelRouter

# Learning integration
from src.learning.continual_learning import ContinualLearner
from src.learning.preference_learning import PreferenceLearner

# Observability
from src.observability.logging.config import get_logger
from src.observability.monitoring.metrics import MetricsCollector
from src.observability.monitoring.tracing import TraceManager


class ConsolidationStatus(Enum):
    """Status of memory consolidation."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class MemoryPriority:
    """Priority information for memory items."""
    base_importance: float
    access_count: int
    recency_score: float
    context_relevance: float
    emotional_weight: float
    consolidation_score: float = field(init=False)
    
    def __post_init__(self):
        """Calculate consolidation score after initialization."""
        self.consolidation_score = (
            self.base_importance * 0.3 +
            min(self.access_count / 10, 1.0) * 0.2 +
            self.recency_score * 0.2 +
            self.context_relevance * 0.2 +
            self.emotional_weight * 0.1
        )


class ShortTermMemoryConfig:
    """Configuration for short-term memory."""
    
    def __init__(self, config_loader: ConfigLoader):
        """Initialize configuration from config loader."""
        self.capacity = config_loader.get("memory.short_term.capacity", 10000)
        self.consolidation_threshold = config_loader.get("memory.short_term.consolidation_threshold", 0.6)
        self.retention_hours = config_loader.get("memory.short_term.retention_hours", 24)
        self.decay_rate = config_loader.get("memory.short_term.decay_rate", 0.1)
        self.consolidation_batch_size = config_loader.get("memory.short_term.consolidation_batch_size", 100)
        self.auto_consolidation = config_loader.get("memory.short_term.auto_consolidation", True)
        self.importance_threshold = config_loader.get("memory.short_term.importance_threshold", 0.3)
        self.cleanup_interval = config_loader.get("memory.short_term.cleanup_interval", 3600)
        self.priority_queue_size = config_loader.get("memory.short_term.priority_queue_size", 1000)
        self.enable_embeddings = config_loader.get("memory.short_term.enable_embeddings", True)


class ShortTermMemory(BaseMemory):
    """
    Short-term memory implementation - temporary storage with consolidation.
    
    This implementation provides:
    - Temporary storage with automatic expiration
    - Importance-based memory prioritization
    - Automatic consolidation to long-term memory
    - Intelligent forgetting mechanisms
    - Context-aware memory management
    - Integration with learning systems
    - Performance optimization through batching
    """

    def __init__(
        self,
        container: Container,
        memory_store: BaseMemoryStore,
        model_router: Optional[ModelRouter] = None,
    ):
        """
        Initialize short-term memory.

        Args:
            container: Dependency injection container
            memory_store: Memory storage backend
            model_router: Model router for embeddings
        """
        self.container = container
        self.memory_store = memory_store
        self.model_router = model_router
        self.logger = get_logger(__name__)

        # Get core dependencies
        self.event_bus = container.get(EventBus)
        self.error_handler = container.get(ErrorHandler)
        
        # Load configuration
        config_loader = container.get(ConfigLoader)
        self.config = ShortTermMemoryConfig(config_loader)

        # Get optional dependencies
        try:
            self.consolidation = container.get(MemoryConsolidation)
        except Exception:
            self.logger.warning("Memory consolidation service not available")
            self.consolidation = None

        try:
            self.continual_learner = container.get(ContinualLearner)
            self.preference_learner = container.get(PreferenceLearner)
        except Exception:
            self.logger.warning("Learning components not available")
            self.continual_learner = None
            self.preference_learner = None

        # Get monitoring components
        try:
            self.metrics = container.get(MetricsCollector)
            self.tracer = container.get(TraceManager)
        except Exception:
            self.metrics = None
            self.tracer = None
            self.logger.warning("Monitoring components not available")

        # Initialize storage structures
        self._initialize_storage()

        # Cache
        self._memory_cache: Dict[str, MemoryItem] = {}
        self._cache_lock = asyncio.Lock()

        # Background task management
        self._background_tasks: List[asyncio.Task] = []
        self._shutdown_event = asyncio.Event()

        self.logger.info("ShortTermMemory initialized successfully")

    def _initialize_storage(self) -> None:
        """Initialize storage structures for short-term memory."""
        # Time-based organization
        self._time_buckets: Dict[str, List[str]] = defaultdict(list)  # hour -> memory_ids
        self._expiration_queue: List[Tuple[datetime, str]] = []  # (expiration_time, memory_id)
        
        # Priority management
        self._priority_queue: List[Tuple[float, str]] = []  # (-priority, memory_id) for max heap
        self._memory_priorities: Dict[str, MemoryPriority] = {}
        
        # Context tracking
        self._context_index: Dict[str, Set[str]] = defaultdict(set)  # context_id -> memory_ids
        self._session_index: Dict[str, Set[str]] = defaultdict(set)  # session_id -> memory_ids
        self._user_index: Dict[str, Set[str]] = defaultdict(set)  # user_id -> memory_ids
        
        # Consolidation tracking
        self._consolidation_status: Dict[str, ConsolidationStatus] = {}
        self._consolidation_queue: deque = deque()
        self._consolidation_history: Dict[str, datetime] = {}
        
        # Memory relationships
        self._related_memories: Dict[str, Set[str]] = defaultdict(set)
        self._memory_clusters: Dict[str, Set[str]] = defaultdict(set)

    async def initialize(self) -> None:
        """Initialize short-term memory and start background tasks."""
        try:
            self.logger.info("Initializing short-term memory...")

            # Initialize storage backend if needed
            if hasattr(self.memory_store, 'initialize'):
                await self.memory_store.initialize()

            # Start background tasks
            self._background_tasks.append(
                asyncio.create_task(self._cleanup_loop())
            )

            if self.config.auto_consolidation:
                self._background_tasks.append(
                    asyncio.create_task(self._consolidation_loop())
                )

            self._background_tasks.append(
                asyncio.create_task(self._decay_loop())
            )

            # Rebuild indices from existing data
            await self._rebuild_indices()

            self.logger.info("Short-term memory initialization complete")

        except Exception as e:
            self.logger.error(f"Failed to initialize short-term memory: {str(e)}")
            raise MemoryError(f"Short-term memory initialization failed: {str(e)}")

    async def shutdown(self) -> None:
        """Shutdown short-term memory and cleanup resources."""
        self.logger.info("Shutting down short-term memory...")
        
        # Signal shutdown to background tasks
        self._shutdown_event.set()

        # Cancel and wait for background tasks
        for task in self._background_tasks:
            task.cancel()

        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)

        # Process any pending consolidations
        await self._process_pending_consolidations()

        # Clear caches and indices
        self._memory_cache.clear()
        self._clear_indices()

        self.logger.info("Short-term memory shutdown complete")

    @handle_exceptions
    async def store(self, data: Any, **kwargs) -> str:
        """
        Store data in short-term memory.

        Args:
            data: Data to store
            **kwargs: Additional parameters including:
                user_id: User identifier
                session_id: Session identifier
                context_id: Context identifier
                tags: Memory tags
                importance: Importance score (0-1)
                related_to: List of related memory IDs

        Returns:
            Memory ID
        """
        async with memory_operation_span(self.tracer, "store_short_term"):
            # Check capacity
            current_count = await self._get_current_count()
            if current_count >= self.config.capacity:
                await self._apply_capacity_management()
                
                # Emit warning
                await self.event_bus.emit(
                    MemoryCapacityWarning(
                        memory_type=MemoryType.SHORT_TERM.value,
                        current_count=current_count,
                        capacity=self.config.capacity,
                    )
                )

            # Generate memory ID
            memory_id = MemoryUtils.generate_memory_id()

            # Create metadata
            metadata = await self._create_metadata(data, **kwargs)

            # Create memory item
            memory_item = MemoryItem(
                memory_id=memory_id,
                content=data,
                memory_type=MemoryType.SHORT_TERM,
                owner_id=kwargs.get("user_id"),
                session_id=kwargs.get("session_id"),
                context_id=kwargs.get("context_id"),
                metadata=metadata,
            )

            # Generate embeddings if enabled
            if self.config.enable_embeddings and self.model_router:
                memory_item.embeddings = await self._generate_embeddings(data)

            # Store in backend
            await self.memory_store.store_item(memory_item)

            # Update indices and tracking
            await self._update_indices_on_store(memory_item, **kwargs)

            # Calculate and store priority
            priority = await self._calculate_priority(memory_item, **kwargs)
            self._memory_priorities[memory_id] = priority
            heapq.heappush(self._priority_queue, (-priority.consolidation_score, memory_id))

            # Handle relationships
            related_to = kwargs.get("related_to", [])
            if related_to:
                await self._update_relationships(memory_id, related_to)

            # Add to cache
            await self._add_to_cache(memory_item)

            # Schedule for consolidation if high importance
            if priority.consolidation_score >= self.config.consolidation_threshold:
                self._consolidation_queue.append(memory_id)

            # Emit event
            await self.event_bus.emit(
                MemoryItemStored(
                    memory_id=memory_id,
                    memory_type=MemoryType.SHORT_TERM.value,
                    owner_id=kwargs.get("user_id"),
                    context_id=kwargs.get("context_id"),
                )
            )

            # Update metrics
            if self.metrics:
                self.metrics.increment("short_term_memory_items_stored")
                self.metrics.gauge("short_term_memory_count", await self._get_current_count())

            self.logger.debug(f"Stored short-term memory {memory_id}")
            return memory_id

    @handle_exceptions
    async def retrieve(self, memory_id: str) -> Optional[MemoryItem]:
        """
        Retrieve an item from short-term memory.

        Args:
            memory_id: Memory identifier

        Returns:
            Memory item or None if not found
        """
        async with memory_operation_span(self.tracer, "retrieve_short_term", memory_id):
            # Check cache first
            async with self._cache_lock:
                if memory_id in self._memory_cache:
                    item = self._memory_cache[memory_id]
                    await self._update_on_access(item)
                    return item

            # Retrieve from store
            item = await self.memory_store.get_item(memory_id)
            
            if item:
                # Update on access
                await self._update_on_access(item)
                
                # Add to cache
                await self._add_to_cache(item)
                
                # Emit event
                await self.event_bus.emit(
                    MemoryItemRetrieved(
                        memory_id=memory_id,
                        memory_type=MemoryType.SHORT_TERM.value,
                        owner_id=item.owner_id,
                    )
                )
                
                # Update metrics
                if self.metrics:
                    self.metrics.increment("short_term_memory_items_retrieved")

            return item

    async def update(self, memory_id: str, data: Any) -> bool:
        """
        Update existing memory.

        Args:
            memory_id: Memory identifier
            data: New data

        Returns:
            True if successful
        """
        async with memory_operation_span(self.tracer, "update_short_term", memory_id):
            # Retrieve existing memory
            memory_item = await self.retrieve(memory_id)
            if not memory_item:
                return False

            # Update content
            memory_item.content = data
            memory_item.metadata.update_modification()

            # Regenerate embeddings if needed
            if self.config.enable_embeddings and self.model_router:
                memory_item.embeddings = await self._generate_embeddings(data)

            # Recalculate priority
            priority = await self._calculate_priority(memory_item)
            self._memory_priorities[memory_id] = priority

            # Store updated item
            await self.memory_store.store_item(memory_item)

            # Update cache
            await self._add_to_cache(memory_item)

            return True

    async def delete(self, memory_id: str) -> bool:
        """
        Delete memory.

        Args:
            memory_id: Memory identifier

        Returns:
            True if successful
        """
        async with memory_operation_span(self.tracer, "delete_short_term", memory_id):
            # Remove from indices
            await self._remove_from_indices(memory_id)

            # Remove from cache
            async with self._cache_lock:
                self._memory_cache.pop(memory_id, None)

            # Remove from tracking structures
            self._memory_priorities.pop(memory_id, None)
            self._consolidation_status.pop(memory_id, None)
            self._consolidation_history.pop(memory_id, None)

            # Remove relationships
            self._related_memories.pop(memory_id, None)
            for related_set in self._related_memories.values():
                related_set.discard(memory_id)

            # Emit event
            await self.event_bus.emit(
                MemoryItemDeleted(
                    memory_id=memory_id,
                    memory_type=MemoryType.SHORT_TERM.value,
                )
            )

            # Delete from store
            return await self.memory_store.delete_item(memory_id)

    async def search(self, query: Any) -> List[MemoryItem]:
        """
        Search short-term memory.

        Args:
            query: Search query

        Returns:
            List of matching memory items
        """
        async with memory_operation_span(self.tracer, "search_short_term"):
            if isinstance(query, SimpleMemoryQuery):
                return await self.memory_store.query(query)

            elif isinstance(query, str):
                return await self._handle_string_query(query)

            elif isinstance(query, dict):
                return await self._handle_dict_query(query)

            else:
                raise MemoryError(f"Unsupported query type for short-term memory: {type(query)}")

    async def clear(self) -> None:
        """Clear all short-term memory."""
        # Clear indices and tracking
        self._clear_indices()
        
        # Clear priority structures
        self._priority_queue.clear()
        self._memory_priorities.clear()
        
        # Clear consolidation tracking
        self._consolidation_status.clear()
        self._consolidation_queue.clear()
        self._consolidation_history.clear()
        
        # Clear relationships
        self._related_memories.clear()
        self._memory_clusters.clear()

        # Clear cache
        async with self._cache_lock:
            self._memory_cache.clear()

        # Clear underlying store
        query = SimpleMemoryQuery(memory_type=MemoryType.SHORT_TERM)
        items = await self.memory_store.query(query)

        for item in items:
            await self.memory_store.delete_item(item.memory_id)

        self.logger.info(f"Cleared all short-term memory ({len(items)} items)")

    async def get_stats(self) -> Dict[str, Any]:
        """Get short-term memory statistics."""
        current_count = await self._get_current_count()
        
        stats = {
            "total_items": current_count,
            "capacity": self.config.capacity,
            "utilization": current_count / self.config.capacity if self.config.capacity > 0 else 0,
            "cache_size": len(self._memory_cache),
            "priority_queue_size": len(self._priority_queue),
            "consolidation_queue_size": len(self._consolidation_queue),
            "memory_type": MemoryType.SHORT_TERM.value,
            "time_distribution": self._get_time_distribution(),
            "consolidation_stats": self._get_consolidation_stats(),
        }

        # Add user distribution
        stats["user_distribution"] = {
            user_id: len(memory_ids) 
            for user_id, memory_ids in self._user_index.items()
        }

        return stats

    # Additional public methods specific to short-term memory

    async def get_memories_for_consolidation(
        self, batch_size: Optional[int] = None, importance_threshold: Optional[float] = None
    ) -> List[MemoryItem]:
        """
        Get memories eligible for consolidation.

        Args:
            batch_size: Number of memories to return
            importance_threshold: Minimum importance for consolidation

        Returns:
            List of memory items ready for consolidation
        """
        if batch_size is None:
            batch_size = self.config.consolidation_batch_size
        if importance_threshold is None:
            importance_threshold = self.config.importance_threshold

        eligible_memories = []
        
        # Get high-priority memories from the priority queue
        temp_queue = []
        
        while self._priority_queue and len(eligible_memories) < batch_size:
            priority, memory_id = heapq.heappop(self._priority_queue)
            temp_queue.append((priority, memory_id))
            
            # Check if already consolidated
            if self._consolidation_status.get(memory_id) == ConsolidationStatus.COMPLETED:
                continue
            
            # Check importance threshold
            if memory_id in self._memory_priorities:
                mem_priority = self._memory_priorities[memory_id]
                if mem_priority.consolidation_score >= importance_threshold:
                    memory = await self.retrieve(memory_id)
                    if memory:
                        eligible_memories.append(memory)
                        self._consolidation_status[memory_id] = ConsolidationStatus.PENDING
        
        # Restore priority queue
        for item in temp_queue:
            heapq.heappush(self._priority_queue, item)
        
        return eligible_memories

    async def mark_consolidated(self, memory_id: str) -> None:
        """
        Mark a memory as consolidated.

        Args:
            memory_id: Memory identifier
        """
        self._consolidation_status[memory_id] = ConsolidationStatus.COMPLETED
        self._consolidation_history[memory_id] = datetime.now(timezone.utc)
        
        # Update memory metadata
        memory = await self.retrieve(memory_id)
        if memory:
            memory.metadata.custom_metadata["consolidated"] = True
            memory.metadata.custom_metadata["consolidation_time"] = datetime.now(timezone.utc).isoformat()
            await self.memory_store.store_item(memory)

    async def get_recent_memories(
        self, hours: int = 1, limit: int = 100, user_id: Optional[str] = None
    ) -> List[MemoryItem]:
        """
        Get recent memories within specified hours.

        Args:
            hours: Number of hours to look back
            limit: Maximum number of memories
            user_id: Optional user filter

        Returns:
            List of recent memory items
        """
        current_time = datetime.now(timezone.utc)
        cutoff_time = current_time - timedelta(hours=hours)
        
        # Get memories from time buckets
        memory_ids = []
        for h in range(hours):
            bucket_time = current_time - timedelta(hours=h)
            bucket_key = bucket_time.strftime("%Y-%m-%d-%H")
            
            if bucket_key in self._time_buckets:
                bucket_ids = self._time_buckets[bucket_key]
                
                # Filter by user if specified
                if user_id:
                    bucket_ids = [
                        mid for mid in bucket_ids 
                        if mid in self._user_index.get(user_id, set())
                    ]
                
                memory_ids.extend(bucket_ids)
        
        # Retrieve memories
        memories = []
        for memory_id in memory_ids[:limit]:
            memory = await self.retrieve(memory_id)
            if memory and memory.metadata.created_at >= cutoff_time:
                memories.append(memory)
        
        # Sort by recency
        memories.sort(key=lambda x: x.metadata.created_at, reverse=True)
        
        return memories[:limit]

    async def get_related_memories(
        self, memory_id: str, max_depth: int = 2, limit: int = 10
    ) -> List[MemoryItem]:
        """
        Get memories related to a specific memory.

        Args:
            memory_id: Memory identifier
            max_depth: Maximum relationship depth
            limit: Maximum number of memories

        Returns:
            List of related memory items
        """
        visited = set()
        related_items = []
        queue = deque([(memory_id, 0)])
        
        while queue and len(related_items) < limit:
            current_id, depth = queue.popleft()
            
            if current_id in visited or depth > max_depth:
                continue
            
            visited.add(current_id)
            
            # Get directly related memories
            if current_id in self._related_memories:
                for related_id in self._related_memories[current_id]:
                    if related_id not in visited:
                        memory = await self.retrieve(related_id)
                        if memory:
                            related_items.append(memory)
                            if depth < max_depth:
                                queue.append((related_id, depth + 1))
        
        return related_items[:limit]

    async def apply_decay(self, decay_rate: Optional[float] = None) -> int:
        """
        Apply decay to memory importance scores.

        Args:
            decay_rate: Rate of decay (0-1)

        Returns:
            Number of memories affected
        """
        if decay_rate is None:
            decay_rate = self.config.decay_rate

        affected_count = 0
        current_time = datetime.now(timezone.utc)
        
        # Get all short-term memories
        query = SimpleMemoryQuery(memory_type=MemoryType.SHORT_TERM)
        items = await self.memory_store.query(query)
        
        for item in items:
            # Skip recently accessed items
            last_access = item.metadata.last_accessed or item.metadata.created_at
            hours_since_access = (current_time - last_access).total_seconds() / 3600
            
            if hours_since_access < 1:  # Don't decay very recent memories
                continue
            
            # Apply decay to importance
            if item.memory_id in self._memory_priorities:
                priority = self._memory_priorities[item.memory_id]
                
                # Exponential decay based on time
                decay_factor = decay_rate * (hours_since_access / 24)
                new_importance = priority.base_importance * (1 - decay_factor)
                
                if new_importance < self.config.importance_threshold:
                    # Mark for deletion
                    await self.delete(item.memory_id)
                else:
                    # Update importance
                    priority.base_importance = new_importance
                    priority.recency_score *= (1 - decay_factor)
                    # Recalculate consolidation score
                    priority.__post_init__()
                    
                    affected_count += 1
        
        # Emit event
        if affected_count > 0:
            await self.event_bus.emit(
                MemoryDecayApplied(
                    memory_type=MemoryType.SHORT_TERM.value,
                    count=affected_count
                )
            )
        
        return affected_count

    # Private helper methods

    async def _create_metadata(self, data: Any, **kwargs) -> MemoryMetadata:
        """Create metadata for a short-term memory item."""
        # Calculate expiration
        retention_hours = kwargs.get("retention_hours", self.config.retention_hours)
        expiration = datetime.now(timezone.utc) + timedelta(hours=retention_hours)
        
        # Create base metadata
        metadata = MemoryMetadata(
            retention_policy=MemoryRetentionPolicy.TEMPORARY,
            expiration=expiration,
            tags=kwargs.get("tags", set()),
            importance=kwargs.get("importance", 0.5),
            custom_metadata={
                "retention_hours": retention_hours,
                "consolidated": False,
            }
        )
        
        return metadata

    async def _generate_embeddings(self, data: Any) -> Optional[List[float]]:
        """Generate embeddings for memory content."""
        if not self.model_router:
            return None

        try:
            # Convert to text for embedding
            if isinstance(data, dict):
                embed_text = json.dumps(data)
            elif not isinstance(data, str):
                embed_text = str(data)
            else:
                embed_text = data

            return await self.model_router.get_embeddings(embed_text)

        except Exception as e:
            self.logger.warning(f"Failed to generate embeddings: {str(e)}")
            return None

    async def _calculate_priority(
        self, memory_item: MemoryItem, **kwargs
    ) -> MemoryPriority:
        """Calculate priority for a memory item."""
        # Base importance from metadata
        base_importance = memory_item.metadata.importance
        
        # Access count
        access_count = memory_item.metadata.access_count
        
        # Recency score (1.0 for new, decreasing over time)
        age = (datetime.now(timezone.utc) - memory_item.metadata.created_at).total_seconds()
        recency_score = max(0, 1.0 - (age / (self.config.retention_hours * 3600)))
        
        # Context relevance (simplified - would be more sophisticated in practice)
        context_relevance = 0.5
        if memory_item.context_id:
            # Boost if part of active context
            context_relevance = 0.8
        
        # Emotional weight (if emotional content detected)
        emotional_weight = 0.5
        if isinstance(memory_item.content, dict):
            if "emotion" in memory_item.content or "sentiment" in memory_item.content:
                emotional_weight = 0.8
        
        # Learn importance if preference learner available
        if self.preference_learner:
            try:
                learned_importance = await self.preference_learner.predict_importance(
                    memory_item.content,
                    memory_item.metadata
                )
                base_importance = (base_importance + learned_importance) / 2
            except Exception as e:
                self.logger.debug(f"Failed to learn importance: {str(e)}")
        
        return MemoryPriority(
            base_importance=base_importance,
            access_count=access_count,
            recency_score=recency_score,
            context_relevance=context_relevance,
            emotional_weight=emotional_weight,
        )

    async def _update_indices_on_store(self, memory_item: MemoryItem, **kwargs) -> None:
        """Update indices when storing a memory."""
        memory_id = memory_item.memory_id
        
        # Time bucket
        time_bucket = memory_item.metadata.created_at.strftime("%Y-%m-%d-%H")
        self._time_buckets[time_bucket].append(memory_id)
        
        # Expiration queue
        if memory_item.metadata.expiration:
            heapq.heappush(self._expiration_queue, (memory_item.metadata.expiration, memory_id))
        
        # Context index
        if memory_item.context_id:
            self._context_index[memory_item.context_id].add(memory_id)
        
        # Session index
        if memory_item.session_id:
            self._session_index[memory_item.session_id].add(memory_id)
        
        # User index
        if memory_item.owner_id:
            self._user_index[memory_item.owner_id].add(memory_id)

    async def _remove_from_indices(self, memory_id: str) -> None:
        """Remove a memory from all indices."""
        # Get memory to access metadata
        memory = await self.memory_store.get_item(memory_id)
        if not memory:
            return
        
        # Remove from time buckets
        time_bucket = memory.metadata.created_at.strftime("%Y-%m-%d-%H")
        if time_bucket in self._time_buckets:
            self._time_buckets[time_bucket] = [
                mid for mid in self._time_buckets[time_bucket] if mid != memory_id
            ]
        
        # Remove from context index
        if memory.context_id:
            self._context_index[memory.context_id].discard(memory_id)
        
        # Remove from session index
        if memory.session_id:
            self._session_index[memory.session_id].discard(memory_id)
        
        # Remove from user index
        if memory.owner_id:
            self._user_index[memory.owner_id].discard(memory_id)

    def _clear_indices(self) -> None:
        """Clear all short-term memory indices."""
        self._time_buckets.clear()
        self._expiration_queue.clear()
        self._context_index.clear()
        self._session_index.clear()
        self._user_index.clear()

    async def _rebuild_indices(self) -> None:
        """Rebuild indices from stored memories."""
        self.logger.info("Rebuilding short-term memory indices...")
        
        # Clear existing indices
        self._clear_indices()
        
        # Query all short-term memories
        query = SimpleMemoryQuery(memory_type=MemoryType.SHORT_TERM, limit=10000)
        items = await self.memory_store.query(query)
        
        # Rebuild indices and priority information
        for item in items:
            await self._update_indices_on_store(item)
            
            # Recalculate priority
            priority = await self._calculate_priority(item)
            self._memory_priorities[item.memory_id] = priority
            heapq.heappush(self._priority_queue, (-priority.consolidation_score, item.memory_id))
        
        self.logger.info(f"Rebuilt indices for {len(items)} short-term memories")

    async def _add_to_cache(self, item: MemoryItem) -> None:
        """Add item to memory cache."""
        async with self._cache_lock:
            # Enforce cache size limit
            if len(self._memory_cache) >= self.config.priority_queue_size:
                # Remove least recently used
                if self._memory_cache:
                    self._memory_cache.pop(next(iter(self._memory_cache)))
            
            # Add to cache
            self._memory_cache[item.memory_id] = item

    async def _update_on_access(self, item: MemoryItem) -> None:
        """Update memory item on access."""
        # Update metadata
        item.metadata.update_access()
        
        # Update priority
        if item.memory_id in self._memory_priorities:
            priority = self._memory_priorities[item.memory_id]
            priority.access_count = item.metadata.access_count
            
            # Boost importance slightly on access
            priority.base_importance = min(1.0, priority.base_importance * 1.1)
            
            # Recalculate consolidation score
            priority.__post_init__()
        
        # Store updated metadata
        await self.memory_store.update_item(item.memory_id, {"metadata": item.metadata})

    async def _update_relationships(self, memory_id: str, related_ids: List[str]) -> None:
        """Update memory relationships."""
        # Add bidirectional relationships
        for related_id in related_ids:
            self._related_memories[memory_id].add(related_id)
            self._related_memories[related_id].add(memory_id)

    async def _handle_string_query(self, query: str) -> List[MemoryItem]:
        """Handle string-based queries."""
        query_lower = query.lower()
        
        if query.startswith("user:"):
            user_id = query.split(":", 1)[1].strip()
            memory_ids = list(self._user_index.get(user_id, set()))[:100]
            return await self._retrieve_multiple(memory_ids)
        
        elif query.startswith("session:"):
            session_id = query.split(":", 1)[1].strip()
            memory_ids = list(self._session_index.get(session_id, set()))[:100]
            return await self._retrieve_multiple(memory_ids)
        
        elif query.startswith("context:"):
            context_id = query.split(":", 1)[1].strip()
            memory_ids = list(self._context_index.get(context_id, set()))[:100]
            return await self._retrieve_multiple(memory_ids)
        
        elif query.startswith("recent:"):
            hours = int(query.split(":", 1)[1].strip())
            return await self.get_recent_memories(hours=hours)
        
        else:
            # Text search across all short-term memories
            all_query = SimpleMemoryQuery(memory_type=MemoryType.SHORT_TERM, limit=1000)
            all_items = await self.memory_store.query(all_query)
            
            # Filter by query text
            matching_items = []
            for item in all_items:
                content = item.content
                if isinstance(content, dict):
                    content = json.dumps(content)
                elif not isinstance(content, str):
                    content = str(content)
                
                if query_lower in content.lower():
                    matching_items.append(item)
            
            # Sort by priority
            matching_items.sort(
                key=lambda x: self._memory_priorities.get(
                    x.memory_id, 
                    MemoryPriority(0.5, 0, 0.5, 0.5, 0.5)
                ).consolidation_score,
                reverse=True
            )
            
            return matching_items[:10]

    async def _handle_dict_query(self, query: dict) -> List[MemoryItem]:
        """Handle dictionary-based queries."""
        # Convert to SimpleMemoryQuery
        simple_query = SimpleMemoryQuery(
            memory_type=MemoryType.SHORT_TERM,
            owner_id=query.get("user_id"),
            session_id=query.get("session_id"),
            context_id=query.get("context_id"),
            tags=set(query.get("tags", [])),
            limit=query.get("limit", 100),
        )
        
        # Add time range if specified
        if "hours_back" in query:
            hours = query["hours_back"]
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(hours=hours)
            simple_query.time_range = (start_time, end_time)
        
        results = await self.memory_store.query(simple_query)
        
        # Apply importance filter if specified
        if "min_importance" in query:
            min_importance = query["min_importance"]
            results = [
                item for item in results
                if self._memory_priorities.get(
                    item.memory_id, 
                    MemoryPriority(0.5, 0, 0.5, 0.5, 0.5)
                ).consolidation_score >= min_importance
            ]
        
        return results

    async def _retrieve_multiple(self, memory_ids: List[str]) -> List[MemoryItem]:
        """Retrieve multiple memory items efficiently."""
        items = []
        
        for memory_id in memory_ids:
            item = await self.retrieve(memory_id)
            if item:
                items.append(item)
        
        return items

    async def _get_current_count(self) -> int:
        """Get current number of items in short-term memory."""
        query = SimpleMemoryQuery(memory_type=MemoryType.SHORT_TERM, limit=1)
        # This is a simplified count - in practice would use a more efficient method
        items = await self.memory_store.query(query)
        return len(self._user_index) * 100  # Rough estimate

    async def _apply_capacity_management(self) -> None:
        """Apply capacity management when memory is full."""
        # Remove expired memories first
        await self._cleanup_expired_memories()
        
        # If still over capacity, remove low-priority memories
        current_count = await self._get_current_count()
        if current_count >= self.config.capacity:
            # Remove bottom 10% by priority
            to_remove = int(self.config.capacity * 0.1)
            
            # Get low-priority items
            low_priority_items = []
            temp_queue = []
            
            while self._priority_queue and len(low_priority_items) < to_remove:
                priority, memory_id = heapq.heappop(self._priority_queue)
                temp_queue.append((priority, memory_id))
                
                # Skip if already consolidated
                if self._consolidation_status.get(memory_id) != ConsolidationStatus.COMPLETED:
                    low_priority_items.append(memory_id)
            
            # Restore priority queue
            for item in temp_queue:
                heapq.heappush(self._priority_queue, item)
            
            # Delete low-priority items
            for memory_id in low_priority_items:
                await self.delete(memory_id)

    async def _cleanup_expired_memories(self) -> None:
        """Clean up expired memories."""
        current_time = datetime.now(timezone.utc)
        expired_count = 0
        
        while self._expiration_queue:
            expiration_time, memory_id = self._expiration_queue[0]
            
            if expiration_time > current_time:
                break
            
            # Remove expired memory
            heapq.heappop(self._expiration_queue)
            
            # Delete if not consolidated
            if self._consolidation_status.get(memory_id) != ConsolidationStatus.COMPLETED:
                await self.delete(memory_id)
                expired_count += 1
        
        if expired_count > 0:
            self.logger.info(f"Cleaned up {expired_count} expired memories")

    async def _process_pending_consolidations(self) -> None:
        """Process any pending consolidations before shutdown."""
        if not self._consolidation_queue:
            return
        
        self.logger.info(f"Processing {len(self._consolidation_queue)} pending consolidations")
        
        # Get memories for consolidation
        memories = await self.get_memories_for_consolidation()
        
        if memories and self.consolidation:
            # Emit consolidation start event
            await self.event_bus.emit(MemoryConsolidationStarted())
            
            try:
                # Perform consolidation
                await self.consolidation.consolidate_memories(memories)
                
                # Mark as consolidated
                for memory in memories:
                    await self.mark_consolidated(memory.memory_id)
                
                # Emit completion event
                await self.event_bus.emit(
                    MemoryConsolidationCompleted(
                        duration=0,  # Would calculate actual duration
                    )
                )
                
            except Exception as e:
                self.logger.error(f"Error during final consolidation: {str(e)}")

    def _get_time_distribution(self) -> Dict[str, int]:
        """Get time distribution of memories."""
        distribution = {}
        
        for bucket_key, memory_ids in self._time_buckets.items():
            distribution[bucket_key] = len(memory_ids)
        
        return distribution

    def _get_consolidation_stats(self) -> Dict[str, int]:
        """Get consolidation statistics."""
        stats = defaultdict(int)
        
        for status in self._consolidation_status.values():
            stats[status.value] += 1
        
        return dict(stats)

    async def _cleanup_loop(self) -> None:
        """Background task for periodic cleanup."""
        while not self._shutdown_event.is_set():
            try:
                # Wait for cleanup interval
                await asyncio.sleep(self.config.cleanup_interval)
                
                if not self._shutdown_event.is_set():
                    self.logger.debug("Running short-term memory cleanup")
                    await self._cleanup_expired_memories()
                    
                    # Update metrics
                    if self.metrics:
                        current_count = await self._get_current_count()
                        self.metrics.gauge("short_term_memory_count", current_count)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in cleanup loop: {str(e)}")
                await asyncio.sleep(300)

    async def _consolidation_loop(self) -> None:
        """Background task for automatic consolidation."""
        while not self._shutdown_event.is_set():
            try:
                # Wait for consolidation interval (30 minutes)
                await asyncio.sleep(1800)
                
                if not self._shutdown_event.is_set() and self._consolidation_queue:
                    self.logger.info("Running automatic memory consolidation")
                    
                    # Get memories for consolidation
                    memories = await self.get_memories_for_consolidation()
                    
                    if memories and self.consolidation:
                        # Emit consolidation start event
                        await self.event_bus.emit(MemoryConsolidationStarted())
                        
                        start_time = time.time()
                        
                        try:
                            # Perform consolidation
                            await self.consolidation.consolidate_memories(memories)
                            
                            # Mark as consolidated
                            for memory in memories:
                                await self.mark_consolidated(memory.memory_id)
                            
                            # Emit completion event
                            await self.event_bus.emit(
                                MemoryConsolidationCompleted(
                                    duration=time.time() - start_time,
                                )
                            )
                            
                        except Exception as e:
                            self.logger.error(f"Error during consolidation: {str(e)}")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in consolidation loop: {str(e)}")
                await asyncio.sleep(600)

    async def _decay_loop(self) -> None:
        """Background task for memory decay."""
        while not self._shutdown_event.is_set():
            try:
                # Wait for decay interval (6 hours)
                await asyncio.sleep(21600)
                
                if not self._shutdown_event.is_set():
                    self.logger.info("Applying memory decay")
                    await self.apply_decay()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in decay loop: {str(e)}")
                await asyncio.sleep(3600)
