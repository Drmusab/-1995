"""
Episodic Memory Implementation
Author: Drmusab
Last Modified: 2025-01-10 21:00:00 UTC

This module implements episodic memory - a system for storing experience-based memories
including events, interactions, and contextual information. It provides chronological
organization, emotional tagging, and sophisticated retrieval mechanisms.
"""

import asyncio
import json
import logging
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

# Core imports
from src.core.config.loader import ConfigLoader
from src.core.dependency_injection import Container
from src.core.error_handling import ErrorHandler, handle_exceptions
from src.core.events.event_bus import EventBus
from src.core.events.event_types import (
    MemoryItemRetrieved,
    MemoryItemStored,
)

# Integration components
from src.integrations.llm.model_router import ModelRouter

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

# Observability
from src.observability.logging.config import get_logger
from src.observability.monitoring.metrics import MetricsCollector
from src.observability.monitoring.tracing import TraceManager


class EpisodicMemoryConfig:
    """Configuration for episodic memory."""
    
    def __init__(self, config_loader: ConfigLoader):
        """Initialize configuration from config loader."""
        self.max_cache_size = config_loader.get("memory.episodic.cache_size", 1000)
        self.decay_rate = config_loader.get("memory.episodic.decay_rate", 0.05)
        self.consolidation_threshold = config_loader.get("memory.episodic.consolidation_threshold", 0.7)
        self.embedding_enabled = config_loader.get("memory.episodic.embedding_enabled", True)
        self.auto_tagging = config_loader.get("memory.episodic.auto_tagging", True)
        self.emotion_detection = config_loader.get("memory.episodic.emotion_detection", True)
        self.index_batch_size = config_loader.get("memory.episodic.index_batch_size", 100)


class EpisodicMemory(BaseMemory):
    """
    Episodic memory implementation - stores experience-based memories.
    
    This implementation provides:
    - Chronological storage of experiences and events
    - Contextual organization by session, user, and interaction
    - Emotional tagging and sentiment association
    - Time-based and context-based retrieval
    - Memory strength decay and reinforcement mechanisms
    - Integration with vector embeddings for semantic search
    - Automatic memory consolidation and forgetting
    """

    def __init__(
        self,
        container: Container,
        memory_store: BaseMemoryStore,
        model_router: Optional[ModelRouter] = None,
    ):
        """
        Initialize episodic memory.

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
        self.config = EpisodicMemoryConfig(config_loader)

        # Get monitoring components
        try:
            self.metrics = container.get(MetricsCollector)
            self.tracer = container.get(TraceManager)
        except Exception:
            self.metrics = None
            self.tracer = None
            self.logger.warning("Monitoring components not available")

        # Initialize indexes for fast lookup
        self._initialize_indexes()

        # Initialize caching
        self._recent_access_cache: Dict[str, MemoryItem] = {}
        self._cache_lock = asyncio.Lock()

        # Background task management
        self._background_tasks: List[asyncio.Task] = []
        self._shutdown_event = asyncio.Event()

        self.logger.info("EpisodicMemory initialized successfully")

    def _initialize_indexes(self) -> None:
        """Initialize memory indexes for efficient retrieval."""
        self._user_index: Dict[str, List[str]] = defaultdict(list)
        self._session_index: Dict[str, List[str]] = defaultdict(list)
        self._time_index: Dict[str, List[Tuple[datetime, str]]] = defaultdict(list)
        self._emotional_index: Dict[str, List[str]] = defaultdict(list)
        self._context_index: Dict[str, List[str]] = defaultdict(list)
        self._tag_index: Dict[str, Set[str]] = defaultdict(set)

        # Embedding index for semantic search
        self._embedding_index: Optional[Dict[str, List[float]]] = None
        if self.model_router and self.config.embedding_enabled:
            self._embedding_index = {}

    async def initialize(self) -> None:
        """Initialize episodic memory and start background tasks."""
        try:
            self.logger.info("Initializing episodic memory...")

            # Initialize storage backend if needed
            if hasattr(self.memory_store, 'initialize'):
                await self.memory_store.initialize()

            # Start background tasks
            if self.config.decay_rate > 0:
                self._background_tasks.append(
                    asyncio.create_task(self._memory_decay_loop())
                )

            # Load existing memories into indexes
            await self._rebuild_indexes()

            self.logger.info("Episodic memory initialization complete")

        except Exception as e:
            self.logger.error(f"Failed to initialize episodic memory: {str(e)}")
            raise MemoryError(f"Episodic memory initialization failed: {str(e)}")

    async def shutdown(self) -> None:
        """Shutdown episodic memory and cleanup resources."""
        self.logger.info("Shutting down episodic memory...")
        
        # Signal shutdown to background tasks
        self._shutdown_event.set()

        # Cancel and wait for background tasks
        for task in self._background_tasks:
            task.cancel()

        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)

        # Clear caches and indexes
        self._recent_access_cache.clear()
        self._clear_indexes()

        self.logger.info("Episodic memory shutdown complete")

    @handle_exceptions
    async def store(self, data: Any, **kwargs) -> str:
        """
        Store data in episodic memory.

        Args:
            data: Data to store
            **kwargs: Additional parameters including:
                user_id: User identifier
                session_id: Session identifier
                context_id: Context identifier
                tags: Memory tags
                emotion: Emotional association
                importance: Importance score (0-1)
                timestamp: Event timestamp

        Returns:
            Memory ID
        """
        async with memory_operation_span(self.tracer, "store_episodic"):
            # Generate memory ID
            memory_id = MemoryUtils.generate_memory_id()

            # Extract metadata
            metadata = await self._create_metadata(data, **kwargs)

            # Create memory item
            memory_item = MemoryItem(
                memory_id=memory_id,
                content=data,
                memory_type=MemoryType.EPISODIC,
                owner_id=kwargs.get("user_id"),
                session_id=kwargs.get("session_id"),
                context_id=kwargs.get("context_id"),
                metadata=metadata,
            )

            # Generate embeddings if enabled
            if self.config.embedding_enabled and self.model_router:
                memory_item.embeddings = await self._generate_embeddings(data)
                if memory_item.embeddings and self._embedding_index is not None:
                    self._embedding_index[memory_id] = memory_item.embeddings

            # Store in backend
            await self.memory_store.store_item(memory_item)

            # Update indexes
            await self._update_indexes_on_store(memory_item, **kwargs)

            # Add to cache
            await self._add_to_cache(memory_item)

            # Emit event
            await self.event_bus.emit(
                MemoryItemStored(
                    memory_id=memory_id,
                    memory_type=MemoryType.EPISODIC.value,
                    owner_id=kwargs.get("user_id"),
                    context_id=kwargs.get("context_id"),
                )
            )

            # Update metrics
            if self.metrics:
                self.metrics.increment("episodic_memory_items_stored")
                self._update_user_metrics(kwargs.get("user_id"))

            self.logger.debug(f"Stored episodic memory {memory_id}")
            return memory_id

    @handle_exceptions
    async def retrieve(self, memory_id: str) -> Optional[MemoryItem]:
        """
        Retrieve an item from episodic memory.

        Args:
            memory_id: Memory identifier

        Returns:
            Memory item or None if not found
        """
        async with memory_operation_span(self.tracer, "retrieve_episodic", memory_id):
            # Check cache first
            async with self._cache_lock:
                if memory_id in self._recent_access_cache:
                    item = self._recent_access_cache[memory_id]
                    await self._strengthen_and_update_memory(item)
                    return item

            # Retrieve from store
            item = await self.memory_store.get_item(memory_id)
            
            if item:
                # Strengthen and update memory
                await self._strengthen_and_update_memory(item)
                
                # Add to cache
                await self._add_to_cache(item)
                
                # Emit event
                await self.event_bus.emit(
                    MemoryItemRetrieved(
                        memory_id=memory_id,
                        memory_type=MemoryType.EPISODIC.value,
                        owner_id=item.owner_id,
                    )
                )
                
                # Update metrics
                if self.metrics:
                    self.metrics.increment("episodic_memory_items_retrieved")

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
        async with memory_operation_span(self.tracer, "update_episodic", memory_id):
            # Retrieve existing memory
            memory_item = await self.retrieve(memory_id)
            if not memory_item:
                return False

            # Update content
            memory_item.content = data
            memory_item.metadata.update_modification()

            # Regenerate embeddings if needed
            if self.config.embedding_enabled and self.model_router:
                memory_item.embeddings = await self._generate_embeddings(data)
                if memory_item.embeddings and self._embedding_index is not None:
                    self._embedding_index[memory_id] = memory_item.embeddings

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
        async with memory_operation_span(self.tracer, "delete_episodic", memory_id):
            # Remove from indexes
            await self._remove_from_indexes(memory_id)

            # Remove from cache
            async with self._cache_lock:
                self._recent_access_cache.pop(memory_id, None)

            # Remove from embedding index
            if self._embedding_index is not None:
                self._embedding_index.pop(memory_id, None)

            # Delete from store
            return await self.memory_store.delete_item(memory_id)

    async def search(self, query: Any) -> List[MemoryItem]:
        """
        Search episodic memory.

        Args:
            query: Search query (can be SimpleMemoryQuery, text, or other query types)

        Returns:
            List of matching memory items
        """
        async with memory_operation_span(self.tracer, "search_episodic"):
            if isinstance(query, SimpleMemoryQuery):
                return await self.memory_store.query(query)

            elif isinstance(query, str):
                return await self._handle_string_query(query)

            elif isinstance(query, dict):
                return await self._handle_dict_query(query)

            else:
                raise MemoryError(f"Unsupported query type for episodic memory: {type(query)}")

    async def clear(self) -> None:
        """Clear all episodic memory."""
        # Clear indexes
        self._clear_indexes()

        # Clear embedding index
        if self._embedding_index is not None:
            self._embedding_index.clear()

        # Clear cache
        async with self._cache_lock:
            self._recent_access_cache.clear()

        # Clear underlying store
        query = SimpleMemoryQuery(memory_type=MemoryType.EPISODIC)
        items = await self.memory_store.query(query)

        for item in items:
            await self.memory_store.delete_item(item.memory_id)

        self.logger.info(f"Cleared all episodic memory ({len(items)} items)")

    async def get_stats(self) -> Dict[str, Any]:
        """Get episodic memory statistics."""
        total_items = sum(len(memories) for memories in self._user_index.values())
        
        stats = {
            "total_items": total_items,
            "users_count": len(self._user_index),
            "sessions_count": len(self._session_index),
            "date_entries": len(self._time_index),
            "emotions_tracked": len(self._emotional_index),
            "contexts_count": len(self._context_index),
            "tags_count": len(self._tag_index),
            "cache_size": len(self._recent_access_cache),
            "embedding_count": (
                len(self._embedding_index) if self._embedding_index is not None else 0
            ),
            "memory_type": MemoryType.EPISODIC.value,
        }

        # Add emotional breakdown
        stats["emotion_breakdown"] = {
            emotion: len(memories) for emotion, memories in self._emotional_index.items()
        }

        # Add temporal distribution
        stats["temporal_distribution"] = self._get_temporal_distribution()

        return stats

    # Additional public methods specific to episodic memory

    async def get_user_memories(self, user_id: str, limit: int = 100) -> List[MemoryItem]:
        """Get memories for a specific user."""
        if user_id not in self._user_index:
            return []

        memory_ids = self._user_index[user_id][-limit:]
        return await self._retrieve_multiple(memory_ids)

    async def get_session_memories(self, session_id: str, limit: int = 100) -> List[MemoryItem]:
        """Get memories for a specific session."""
        if session_id not in self._session_index:
            return []

        memory_ids = self._session_index[session_id][-limit:]
        items = await self._retrieve_multiple(memory_ids)
        
        # Sort chronologically
        items.sort(key=lambda x: x.metadata.created_at)
        return items

    async def get_memories_by_date(self, date_str: str, limit: int = 100) -> List[MemoryItem]:
        """Get memories for a specific date."""
        if date_str not in self._time_index:
            return []

        time_entries = sorted(self._time_index[date_str])[-limit:]
        memory_ids = [memory_id for _, memory_id in time_entries]
        return await self._retrieve_multiple(memory_ids)

    async def get_memories_by_timerange(
        self, start_str: str, end_str: str, limit: int = 100
    ) -> List[MemoryItem]:
        """Get memories within a time range."""
        try:
            start_date = datetime.strptime(start_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            end_date = datetime.strptime(end_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            end_date = end_date + timedelta(days=1) - timedelta(microseconds=1)

            memory_ids = []
            current_date = start_date
            
            while current_date <= end_date:
                date_key = current_date.strftime("%Y-%m-%d")
                if date_key in self._time_index:
                    for timestamp, memory_id in self._time_index[date_key]:
                        if start_date <= timestamp <= end_date:
                            memory_ids.append((timestamp, memory_id))
                current_date += timedelta(days=1)

            # Sort by timestamp and limit
            memory_ids.sort()
            memory_ids = [mid for _, mid in memory_ids[-limit:]]
            
            return await self._retrieve_multiple(memory_ids)

        except Exception as e:
            self.logger.error(f"Error retrieving memories by timerange: {str(e)}")
            return []

    async def get_memories_by_emotion(self, emotion: str, limit: int = 100) -> List[MemoryItem]:
        """Get memories associated with a specific emotion."""
        if emotion not in self._emotional_index:
            return []

        memory_ids = self._emotional_index[emotion][-limit:]
        items = await self._retrieve_multiple(memory_ids)
        
        # Sort by recency
        items.sort(key=lambda x: x.metadata.created_at, reverse=True)
        return items

    async def get_memories_by_context(self, context_id: str, limit: int = 100) -> List[MemoryItem]:
        """Get memories for a specific context."""
        if context_id not in self._context_index:
            return []

        memory_ids = self._context_index[context_id][-limit:]
        return await self._retrieve_multiple(memory_ids)

    async def apply_memory_decay(self, decay_rate: Optional[float] = None) -> int:
        """
        Apply memory decay to episodic memories.

        Args:
            decay_rate: Rate at which memories decay (0-1)

        Returns:
            Number of memories affected
        """
        if decay_rate is None:
            decay_rate = self.config.decay_rate

        query = SimpleMemoryQuery(memory_type=MemoryType.EPISODIC)
        items = await self.memory_store.query(query)

        affected_count = 0
        current_time = datetime.now(timezone.utc)

        for item in items:
            # Skip recently accessed items
            last_access = item.metadata.last_accessed or item.metadata.created_at
            days_since_access = (current_time - last_access).days

            if days_since_access < 7:  # Don't decay recent memories
                continue

            # Calculate decay
            if await self._apply_decay_to_memory(item, decay_rate, days_since_access):
                affected_count += 1

        # Emit event
        await self.event_bus.emit(
#             MemoryDecayApplied(
#                 memory_type=MemoryType.EPISODIC.value,
#                 count=affected_count
#             )
        )

        self.logger.info(f"Applied memory decay to {affected_count} episodic memories")
        return affected_count

    # Private helper methods

    async def _create_metadata(self, data: Any, **kwargs) -> MemoryMetadata:
        """Create metadata for a memory item."""
        # Extract timestamp
        timestamp = kwargs.get("timestamp", datetime.now(timezone.utc))
        if isinstance(data, dict) and "timestamp" in data:
            try:
                if isinstance(data["timestamp"], str):
                    timestamp = datetime.fromisoformat(data["timestamp"].replace("Z", "+00:00"))
                elif isinstance(data["timestamp"], datetime):
                    timestamp = data["timestamp"]
            except Exception:
                pass

        # Create base metadata
        metadata = MemoryMetadata(
            created_at=timestamp,
            retention_policy=MemoryRetentionPolicy.STANDARD,
            tags=kwargs.get("tags", set()),
            importance=kwargs.get("importance", 0.5),
            custom_metadata={
                "memory_strength": 1.0,  # Initial strength
            }
        )

        # Add emotion if provided
        emotion = kwargs.get("emotion")
        if not emotion and self.config.emotion_detection and isinstance(data, dict):
            emotion = data.get("emotional_state")
        
        if emotion:
            metadata.custom_metadata["emotion"] = emotion

        # Auto-tagging if enabled
        if self.config.auto_tagging:
            extracted_tags = MemoryUtils.extract_tags_from_content(data)
            metadata.tags.update(extracted_tags)

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

    async def _update_indexes_on_store(self, memory_item: MemoryItem, **kwargs) -> None:
        """Update all indexes when storing a new memory."""
        memory_id = memory_item.memory_id

        # User index
        if memory_item.owner_id:
            self._user_index[memory_item.owner_id].append(memory_id)

        # Session index
        if memory_item.session_id:
            self._session_index[memory_item.session_id].append(memory_id)

        # Time index
        timestamp = memory_item.metadata.created_at
        date_key = timestamp.strftime("%Y-%m-%d")
        self._time_index[date_key].append((timestamp, memory_id))

        # Emotional index
        emotion = memory_item.metadata.custom_metadata.get("emotion")
        if emotion:
            self._emotional_index[emotion].append(memory_id)

        # Context index
        if memory_item.context_id:
            self._context_index[memory_item.context_id].append(memory_id)

        # Tag index
        for tag in memory_item.metadata.tags:
            self._tag_index[tag].add(memory_id)

    async def _remove_from_indexes(self, memory_id: str) -> None:
        """Remove a memory from all indexes."""
        # Retrieve memory to get metadata
        memory = await self.memory_store.get_item(memory_id)
        if not memory:
            return

        # Remove from user index
        if memory.owner_id and memory.owner_id in self._user_index:
            self._user_index[memory.owner_id] = [
                mid for mid in self._user_index[memory.owner_id] if mid != memory_id
            ]

        # Remove from session index
        if memory.session_id and memory.session_id in self._session_index:
            self._session_index[memory.session_id] = [
                mid for mid in self._session_index[memory.session_id] if mid != memory_id
            ]

        # Remove from time index
        date_key = memory.metadata.created_at.strftime("%Y-%m-%d")
        if date_key in self._time_index:
            self._time_index[date_key] = [
                (ts, mid) for ts, mid in self._time_index[date_key] if mid != memory_id
            ]

        # Remove from emotional index
        emotion = memory.metadata.custom_metadata.get("emotion")
        if emotion and emotion in self._emotional_index:
            self._emotional_index[emotion] = [
                mid for mid in self._emotional_index[emotion] if mid != memory_id
            ]

        # Remove from context index
        if memory.context_id and memory.context_id in self._context_index:
            self._context_index[memory.context_id] = [
                mid for mid in self._context_index[memory.context_id] if mid != memory_id
            ]

        # Remove from tag index
        for tag in memory.metadata.tags:
            if tag in self._tag_index:
                self._tag_index[tag].discard(memory_id)

    def _clear_indexes(self) -> None:
        """Clear all memory indexes."""
        self._user_index.clear()
        self._session_index.clear()
        self._time_index.clear()
        self._emotional_index.clear()
        self._context_index.clear()
        self._tag_index.clear()

    async def _rebuild_indexes(self) -> None:
        """Rebuild all indexes from stored memories."""
        self.logger.info("Rebuilding episodic memory indexes...")
        
        # Clear existing indexes
        self._clear_indexes()

        # Query all episodic memories
        query = SimpleMemoryQuery(memory_type=MemoryType.EPISODIC, limit=10000)
        items = await self.memory_store.query(query)

        # Process in batches
        for i in range(0, len(items), self.config.index_batch_size):
            batch = items[i:i + self.config.index_batch_size]
            
            for item in batch:
                await self._update_indexes_on_store(item)
                
                # Update embedding index if needed
                if (self.config.embedding_enabled and 
                    self._embedding_index is not None and 
                    item.embeddings):
                    self._embedding_index[item.memory_id] = item.embeddings

        self.logger.info(f"Rebuilt indexes for {len(items)} episodic memories")

    async def _add_to_cache(self, item: MemoryItem) -> None:
        """Add item to recent access cache."""
        async with self._cache_lock:
            # Enforce cache size limit
            if len(self._recent_access_cache) >= self.config.max_cache_size:
                # Remove oldest item (simple FIFO for now)
                if self._recent_access_cache:
                    self._recent_access_cache.pop(next(iter(self._recent_access_cache)))

            # Add to cache
            self._recent_access_cache[item.memory_id] = item

    async def _strengthen_and_update_memory(self, item: MemoryItem) -> None:
        """Strengthen a memory based on access and update metadata."""
        # Update access metadata
        item.metadata.update_access()

        # Strengthen memory
        custom_metadata = item.metadata.custom_metadata
        current_strength = custom_metadata.get("memory_strength", 1.0)

        # Increase strength with diminishing returns
        new_strength = current_strength + (1.0 - current_strength) * 0.1
        new_strength = min(new_strength, 2.0)  # Cap at 2.0

        custom_metadata["memory_strength"] = new_strength

        # Adjust importance based on strength
        item.metadata.importance = min(1.0, item.metadata.importance * (new_strength / 1.0))

        # Store updated metadata
        await self.memory_store.update_item(
            item.memory_id,
            {"metadata": item.metadata}
        )

    async def _apply_decay_to_memory(
        self, item: MemoryItem, decay_rate: float, days_since_access: int
    ) -> bool:
        """Apply decay to a single memory item."""
        # Calculate decay factor
        age_factor = min(days_since_access / 365, 1.0)
        access_factor = 1.0 / (item.metadata.access_count + 1)
        decay_factor = decay_rate * age_factor * access_factor

        # Get current strength
        custom_metadata = item.metadata.custom_metadata
        current_strength = custom_metadata.get("memory_strength", 1.0)

        # Apply decay
        new_strength = max(0.1, current_strength * (1.0 - decay_factor))

        # Update if changed significantly
        if abs(new_strength - current_strength) > 0.01:
            custom_metadata["memory_strength"] = new_strength

            # Update importance based on strength
            item.metadata.importance = item.metadata.importance * (new_strength / current_strength)

            # Store updated item
            await self.memory_store.update_item(
                item.memory_id,
                {"metadata": item.metadata}
            )

            # Update cache if present
            async with self._cache_lock:
                if item.memory_id in self._recent_access_cache:
                    self._recent_access_cache[item.memory_id] = item

            return True

        return False

    async def _retrieve_multiple(self, memory_ids: List[str]) -> List[MemoryItem]:
        """Retrieve multiple memory items efficiently."""
        items = []
        
        # Batch retrieve if supported by store
        if hasattr(self.memory_store, 'get_items'):
            items = await self.memory_store.get_items(memory_ids)
        else:
            # Fall back to individual retrieval
            for memory_id in memory_ids:
                item = await self.retrieve(memory_id)
                if item:
                    items.append(item)

        return items

    async def _handle_string_query(self, query: str) -> List[MemoryItem]:
        """Handle string-based queries with special prefixes."""
        if query.startswith("user:"):
            user_id = query.split(":", 1)[1].strip()
            return await self.get_user_memories(user_id)

        elif query.startswith("session:"):
            session_id = query.split(":", 1)[1].strip()
            return await self.get_session_memories(session_id)

        elif query.startswith("date:"):
            date_str = query.split(":", 1)[1].strip()
            return await self.get_memories_by_date(date_str)

        elif query.startswith("emotion:"):
            emotion = query.split(":", 1)[1].strip()
            return await self.get_memories_by_emotion(emotion)

        elif query.startswith("context:"):
            context_id = query.split(":", 1)[1].strip()
            return await self.get_memories_by_context(context_id)

        elif query.startswith("tag:"):
            tag = query.split(":", 1)[1].strip()
            return await self._get_memories_by_tag(tag)

        elif query.startswith("timerange:"):
            timerange = query.split(":", 1)[1].strip()
            start_str, end_str = timerange.split("-")
            return await self.get_memories_by_timerange(start_str.strip(), end_str.strip())

        else:
            # Semantic search if embeddings available
            if self.model_router and self._embedding_index:
                return await self._semantic_search(query)
            else:
                # Fall back to text search
                return await self._text_search(query)

    async def _handle_dict_query(self, query: dict) -> List[MemoryItem]:
        """Handle dictionary-based queries."""
        # Convert to SimpleMemoryQuery
        simple_query = SimpleMemoryQuery(
            memory_type=MemoryType.EPISODIC,
            owner_id=query.get("user_id"),
            session_id=query.get("session_id"),
            context_id=query.get("context_id"),
            tags=set(query.get("tags", [])),
            limit=query.get("limit", 100),
            offset=query.get("offset", 0),
        )

        # Add time range if provided
        if "start_date" in query and "end_date" in query:
            try:
                start = datetime.fromisoformat(query["start_date"])
                end = datetime.fromisoformat(query["end_date"])
                simple_query.time_range = (start, end)
            except Exception:
                pass

        return await self.memory_store.query(simple_query)

    async def _get_memories_by_tag(self, tag: str, limit: int = 100) -> List[MemoryItem]:
        """Get memories with a specific tag."""
        if tag not in self._tag_index:
            return []

        memory_ids = list(self._tag_index[tag])[:limit]
        return await self._retrieve_multiple(memory_ids)

    async def _semantic_search(self, query: str, limit: int = 10) -> List[MemoryItem]:
        """Perform semantic search on episodic memories."""
        if not self.model_router or not self._embedding_index:
            return []

        try:
            # Generate query embedding
            query_embedding = await self.model_router.get_embeddings(query)
            if not query_embedding:
                return []

            # Calculate similarities
            similarities = []
            for memory_id, embedding in self._embedding_index.items():
                similarity = self._cosine_similarity(query_embedding, embedding)
                similarities.append((memory_id, similarity))

            # Sort by similarity and get top results
            similarities.sort(key=lambda x: x[1], reverse=True)
            top_memory_ids = [memory_id for memory_id, _ in similarities[:limit]]

            # Retrieve items
            items = await self._retrieve_multiple(top_memory_ids)

            # Strengthen matching memories
            for item in items:
                self._strengthen_memory(item)

            return items

        except Exception as e:
            self.logger.error(f"Semantic search error: {str(e)}")
            return []

    async def _text_search(self, query: str, limit: int = 10) -> List[MemoryItem]:
        """Perform simple text search on episodic memories."""
        # Get all episodic memories
        all_query = SimpleMemoryQuery(memory_type=MemoryType.EPISODIC, limit=1000)
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

        # Sort by recency and limit
        matching_items.sort(key=lambda x: x.metadata.created_at, reverse=True)
        return matching_items[:limit]

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between vectors."""
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0

        # Convert to numpy arrays for efficiency
        arr1 = np.array(vec1)
        arr2 = np.array(vec2)

        # Calculate cosine similarity
        dot_product = np.dot(arr1, arr2)
        magnitude1 = np.linalg.norm(arr1)
        magnitude2 = np.linalg.norm(arr2)

        if magnitude1 * magnitude2 == 0:
            return 0.0

        return float(dot_product / (magnitude1 * magnitude2))

    def _strengthen_memory(self, item: MemoryItem) -> None:
        """Strengthen a memory based on search match (synchronous version)."""
        custom_metadata = item.metadata.custom_metadata
        current_strength = custom_metadata.get("memory_strength", 1.0)

        # Increase strength
        new_strength = current_strength + (1.0 - current_strength) * 0.05
        new_strength = min(new_strength, 2.0)

        custom_metadata["memory_strength"] = new_strength

    def _get_temporal_distribution(self) -> Dict[str, int]:
        """Get temporal distribution of memories."""
        distribution = {}
        
        for date_key, entries in self._time_index.items():
            distribution[date_key] = len(entries)

        return distribution

    def _update_user_metrics(self, user_id: Optional[str]) -> None:
        """Update user-specific metrics."""
        if not self.metrics or not user_id:
            return

        user_items = len(self._user_index.get(user_id, []))
        self.metrics.gauge(
            "episodic_memory_items_per_user",
            user_items,
            {"user_id": user_id}
        )

    async def _memory_decay_loop(self) -> None:
        """Background task for periodic memory decay."""
        while not self._shutdown_event.is_set():
            try:
                # Wait for decay interval (default: 24 hours)
                await asyncio.sleep(86400)

                if not self._shutdown_event.is_set():
                    self.logger.info("Running periodic memory decay...")
                    await self.apply_memory_decay()

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in memory decay loop: {str(e)}")
                await asyncio.sleep(3600)  # Retry after 1 hour on error
