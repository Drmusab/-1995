"""
Advanced Memory Types Implementation
Author: Drmusab
Last Modified: 2025-07-05 09:40:16 UTC

This module provides implementations of various memory types for the AI assistant,
each with specialized behaviors, characteristics, and optimizations.
It defines working memory, episodic memory, semantic memory, procedural memory,
short-term memory, and long-term memory components that collectively form
the assistant's comprehensive memory system.
"""

from pathlib import Path
from typing import Optional, Dict, Any, List, Set, Callable, Type, Union, AsyncGenerator, TypeVar, Tuple, Generic
import asyncio
import time
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
import json
import uuid
import numpy as np
import logging
import traceback
from collections import defaultdict, deque
import heapq
import random
import hashlib

# Core imports
from src.core.config.loader import ConfigLoader
from src.core.events.event_bus import EventBus
from src.core.events.event_types import (
    MemoryItemStored, MemoryItemRetrieved, MemoryItemUpdated, MemoryItemDeleted,
    MemoryCapacityWarning, MemoryCapacityExceeded, MemoryDecayApplied,
    MemoryConsolidationStarted, MemoryConsolidationCompleted
)
from src.core.error_handling import ErrorHandler, handle_exceptions
from src.core.dependency_injection import Container
from src.core.health_check import HealthCheck
from src.core.security.encryption import EncryptionManager

# Memory system imports
from src.memory.core_memory.base_memory import (
    BaseMemory, BaseMemoryStore, MemoryItem, MemoryType, MemoryStorageType,
    MemoryAccess, MemorySensitivity, MemoryRetentionPolicy, MemoryMetadata,
    MemoryError, MemoryNotFoundError, MemoryAccessError, MemoryUtils,
    SimpleMemoryQuery, memory_operation_span
)
from src.memory.storage.vector_store import VectorMemoryStore
from src.memory.storage.memory_graph import MemoryGraphStore

# Integration components
from src.integrations.llm.model_router import ModelRouter
from src.integrations.storage.database import DatabaseManager

# Observability
from src.observability.monitoring.metrics import MetricsCollector
from src.observability.monitoring.tracing import TraceManager
from src.observability.logging.config import get_logger

# Type definitions
T = TypeVar('T')


class WorkingMemory(BaseMemory):
    """
    Working memory implementation - stores current, active information.
    
    Working memory serves as a temporary storage for information that is currently
    being processed or is immediately relevant to ongoing tasks and interactions.
    It has limited capacity and implements recency-based forgetting mechanisms.
    
    Features:
    - Fast access to recent and frequently used items
    - Limited capacity with automatic forgetting
    - Prioritization of items based on importance and recency
    - Integration with attention mechanisms
    - Automatic consolidation into long-term memory
    """
    
    def __init__(self, 
                 container: Container,
                 memory_store: BaseMemoryStore,
                 max_capacity: int = 100,
                 priority_levels: int = 3):
        """
        Initialize working memory.
        
        Args:
            container: Dependency injection container
            memory_store: Memory storage backend
            max_capacity: Maximum number of items to store
            priority_levels: Number of priority levels for memory items
        """
        self.container = container
        self.memory_store = memory_store
        self.max_capacity = max_capacity
        self.priority_levels = priority_levels
        self.logger = get_logger(__name__)
        
        # Get event bus
        self.event_bus = container.get(EventBus)
        
        # Monitoring
        try:
            self.metrics = container.get(MetricsCollector)
            self.tracer = container.get(TraceManager)
        except Exception:
            self.metrics = None
            self.tracer = None
        
        # Priority queues for different sessions
        self._session_memory: Dict[str, Dict[int, List[str]]] = defaultdict(
            lambda: defaultdict(list)
        )
        
        # Access frequency tracking
        self._access_counts: Dict[str, int] = defaultdict(int)
        
        # Items scheduled for consolidation
        self._consolidation_candidates: Set[str] = set()
        
        # Cache of recently accessed items for quick lookup
        self._recent_access_cache: Dict[str, MemoryItem] = {}
        self._max_cache_size = 100
        
        self.logger.info("WorkingMemory initialized")

    async def initialize_session(self, session_id: str) -> None:
        """Initialize working memory for a session."""
        # Clear any existing memory for this session
        if session_id in self._session_memory:
            del self._session_memory[session_id]
        
        # Initialize priority queues
        self._session_memory[session_id] = defaultdict(list)
        
        self.logger.debug(f"Initialized working memory for session {session_id}")

    async def store(self, data: Any, **kwargs) -> str:
        """
        Store data in working memory.
        
        Args:
            data: Data to store
            **kwargs: Additional parameters including:
                session_id: Session identifier
                priority: Item priority (0-1, higher is more important)
                owner_id: Owner identifier
                context_id: Context identifier
                tags: Memory tags
                
        Returns:
            Memory ID
        """
        session_id = kwargs.get('session_id')
        if not session_id:
            raise MemoryError("Session ID required for working memory storage")
        
        # Generate memory ID
        memory_id = MemoryUtils.generate_memory_id()
        
        # Calculate priority level
        priority = kwargs.get('priority', 0.5)
        priority_level = min(
            self.priority_levels - 1,
            int(priority * self.priority_levels)
        )
        
        # Create memory item
        metadata = MemoryMetadata(
            retention_policy=MemoryRetentionPolicy.TRANSIENT,
            tags=kwargs.get('tags', set()),
            importance=priority
        )
        
        memory_item = MemoryItem(
            memory_id=memory_id,
            content=data,
            memory_type=MemoryType.WORKING,
            owner_id=kwargs.get('owner_id'),
            session_id=session_id,
            context_id=kwargs.get('context_id'),
            metadata=metadata
        )
        
        # Check capacity and apply forgetting if needed
        await self._check_capacity(session_id)
        
        # Store in memory store
        async with memory_operation_span(self.tracer, "store_working_memory", memory_id):
            await self.memory_store.store_item(memory_item)
        
        # Add to priority queue
        self._session_memory[session_id][priority_level].append(memory_id)
        
        # Add to recent access cache
        self._add_to_cache(memory_item)
        
        # Emit event
        await self.event_bus.emit(MemoryItemStored(
            memory_id=memory_id,
            memory_type=MemoryType.WORKING.value,
            owner_id=kwargs.get('owner_id'),
            context_id=kwargs.get('context_id')
        ))
        
        # Update metrics
        if self.metrics:
            self.metrics.increment("working_memory_items_stored")
            session_items = sum(len(items) for items in self._session_memory[session_id].values())
            self.metrics.gauge("working_memory_items_per_session", session_items, 
                             {"session_id": session_id})
        
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
            # Update access statistics
            self._access_counts[memory_id] += 1
            # Update metadata
            item.metadata.update_access()
            return item
        
        # Retrieve from store
        async with memory_operation_span(self.tracer, "retrieve_working_memory", memory_id):
            item = await self.memory_store.get_item(memory_id)
        
        if item:
            # Update access statistics
            self._access_counts[memory_id] += 1
            # Update metadata
            item.metadata.update_access()
            # Add to cache
            self._add_to_cache(item)
            # Emit event
            await self.event_bus.emit(MemoryItemRetrieved(
                memory_id=memory_id,
                memory_type=MemoryType.WORKING.value,
                owner_id=item.owner_id
            ))
            # Update metrics
            if self.metrics:
                self.metrics.increment("working_memory_items_retrieved")
        
        return item

    async def update(self, session_id: str, data: Dict[str, Any]) -> None:
        """
        Update working memory with new data.
        
        Args:
            session_id: Session identifier
            data: Data to update (can be a dictionary or memory item)
        """
        if isinstance(data, dict) and 'last_interaction' in data:
            # Store latest interaction data
            interaction_data = data.get('last_interaction', {})
            await self.store(
                interaction_data,
                session_id=session_id,
                priority=0.8,  # High priority for recent interactions
                context_id=interaction_data.get('context_id'),
                tags={'interaction', 'recent'}
            )
            
            # Update any context information
            if 'context' in data:
                context_data = data.get('context', {})
                await self.store(
                    context_data,
                    session_id=session_id,
                    priority=0.7,
                    tags={'context', 'session_state'}
                )
        
        elif isinstance(data, MemoryItem):
            # Store the memory item directly
            await self.memory_store.store_item(data)
            
            # Update priority queues if it's a working memory item
            if data.memory_type == MemoryType.WORKING and data.session_id:
                priority = data.metadata.importance
                priority_level = min(
                    self.priority_levels - 1,
                    int(priority * self.priority_levels)
                )
                
                # Remove existing entry if present
                for level in range(self.priority_levels):
                    if data.memory_id in self._session_memory[data.session_id][level]:
                        self._session_memory[data.session_id][level].remove(data.memory_id)
                
                # Add to appropriate priority level
                self._session_memory[data.session_id][priority_level].append(data.memory_id)
        
        else:
            # Store generic data
            await self.store(
                data,
                session_id=session_id,
                priority=0.5,
                tags={'update'}
            )

    async def clear(self) -> None:
        """Clear all working memory."""
        # Clear priority queues
        self._session_memory.clear()
        
        # Clear access counts
        self._access_counts.clear()
        
        # Clear consolidation candidates
        self._consolidation_candidates.clear()
        
        # Clear cache
        self._recent_access_cache.clear()
        
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
        
        # Get all memory IDs for this session
        memory_ids = []
        for priority_level in range(self.priority_levels):
            memory_ids.extend(self._session_memory[session_id][priority_level])
        
        # Add all to consolidation candidates
        self._consolidation_candidates.update(memory_ids)
        
        # Remove from session memory
        del self._session_memory[session_id]
        
        # Remove from cache
        for memory_id in memory_ids:
            if memory_id in self._recent_access_cache:
                del self._recent_access_cache[memory_id]
        
        self.logger.info(f"Cleaned up working memory for session {session_id} "
                       f"({len(memory_ids)} items marked for consolidation)")

    async def search(self, query: Any) -> List[MemoryItem]:
        """
        Search working memory.
        
        Args:
            query: Search query (can be SimpleMemoryQuery or other query types)
            
        Returns:
            List of matching memory items
        """
        if isinstance(query, SimpleMemoryQuery):
            # Use memory store query capabilities
            return await self.memory_store.query(query)
        
        elif isinstance(query, str):
            # Simple text search
            session_id = None
            if ':' in query and query.split(':', 1)[0] == 'session':
                # Handle session:SESSION_ID format
                parts = query.split(':', 1)
                if len(parts) == 2:
                    session_id = parts[1]
                    query = ''
            
            # Get all items for the session if specified
            all_items = []
            
            if session_id:
                # Get all memory IDs for this session
                memory_ids = []
                if session_id in self._session_memory:
                    for priority_level in range(self.priority_levels):
                        memory_ids.extend(self._session_memory[session_id][priority_level])
                
                # Retrieve each item
                for memory_id in memory_ids:
                    item = await self.retrieve(memory_id)
                    if item:
                        all_items.append(item)
            else:
                # Get all working memory items
                all_query = SimpleMemoryQuery(memory_type=MemoryType.WORKING)
                all_items = await self.memory_store.query(all_query)
            
            # Filter by query text if provided
            if query:
                filtered_items = []
                for item in all_items:
                    content = item.content
                    if isinstance(content, dict):
                        content = json.dumps(content)
                    elif not isinstance(content, str):
                        content = str(content)
                    
                    if query.lower() in content.lower():
                        filtered_items.append(item)
                return filtered_items
            
            return all_items
        
        else:
            # Unsupported query type
            raise MemoryError(f"Unsupported query type for working memory: {type(query)}")

    async def get_recent_items(self, session_id: str, limit: int = 10) -> List[MemoryItem]:
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
        
        # Get all memory IDs for this session
        memory_ids = []
        for priority_level in range(self.priority_levels - 1, -1, -1):
            # Start with highest priority
            memory_ids.extend(self._session_memory[session_id][priority_level])
        
        # Retrieve items in batches
        items = []
        for memory_id in memory_ids[:limit]:
            item = await self.retrieve(memory_id)
            if item:
                items.append(item)
        
        # Sort by recency
        items.sort(
            key=lambda x: x.metadata.last_accessed or x.metadata.created_at,
            reverse=True
        )
        
        return items[:limit]

    async def get_most_relevant(self, session_id: str, context: Dict[str, Any], limit: int = 5) -> List[MemoryItem]:
        """
        Get most relevant working memory items for a context.
        
        Args:
            session_id: Session identifier
            context: Context information
            limit: Maximum number of items to return
            
        Returns:
            List of relevant memory items
        """
        # This is a simplified implementation - a real system would use more 
        # sophisticated relevance measures
        
        # Get all memory IDs for this session
        memory_ids = []
        if session_id in self._session_memory:
            for priority_level in range(self.priority_levels - 1, -1, -1):
                # Start with highest priority
                memory_ids.extend(self._session_memory[session_id][priority_level])
        
        # Calculate relevance scores
        items_with_scores = []
        for memory_id in memory_ids:
            item = await self.retrieve(memory_id)
            if not item:
                continue
            
            # Calculate relevance score based on:
            # 1. Priority (importance)
            # 2. Recency of access
            # 3. Context match (simplified)
            
            # Start with base score from importance
            score = item.metadata.importance * 0.5
            
            # Add recency factor
            if item.metadata.last_accessed:
                time_diff = (datetime.now(timezone.utc) - item.metadata.last_accessed).total_seconds()
                recency_score = max(0, 1.0 - (time_diff / 3600))  # Decay over an hour
                score += recency_score * 0.3
            
            # Add context match factor
            if context.get('context_id') and item.context_id == context.get('context_id'):
                score += 0.2
            
            items_with_scores.append((item, score))
        
        # Sort by relevance score
        items_with_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top items
        return [item for item, _ in items_with_scores[:limit]]

    async def _check_capacity(self, session_id: str) -> None:
        """
        Check working memory capacity and apply forgetting if needed.
        
        Args:
            session_id: Session identifier
        """
        if session_id not in self._session_memory:
            return
        
        # Count items for this session
        item_count = sum(len(items) for items in self._session_memory[session_id].values())
        
        # Issue warning at 80% capacity
        if item_count >= int(self.max_capacity * 0.8):
            await self.event_bus.emit(MemoryCapacityWarning(
                memory_type=MemoryType.WORKING.value,
                current_count=item_count,
                capacity=self.max_capacity,
                session_id=session_id
            ))
        
        # Apply forgetting if over capacity
        if item_count >= self.max_capacity:
            await self._apply_forgetting(session_id)

    async def _apply_forgetting(self, session_id: str) -> None:
        """
        Apply forgetting mechanism to free up working memory.
        
        Args:
            session_id: Session identifier
        """
        # Start forgetting from lowest priority
        for priority_level in range(self.priority_levels):
            if not self._session_memory[session_id][priority_level]:
                continue
            
            # Get items to forget (25% of this priority level)
            items_to_forget = self._session_memory[session_id][priority_level][:max(1, len(self._session_memory[session_id][priority_level]) // 4)]
            
            # Add to consolidation candidates
            self._consolidation_candidates.update(items_to_forget)
            
            # Remove from session memory
            for memory_id in items_to_forget:
                self._session_memory[session_id][priority_level].remove(memory_id)
                # Remove from cache
                if memory_id in self._recent_access_cache:
                    del self._recent_access_cache[memory_id]
            
            self.logger.debug(f"Applied forgetting to {len(items_to_forget)} items in session {session_id}")
            
            # Emit event
            await self.event_bus.emit(MemoryDecayApplied(
                memory_type=MemoryType.WORKING.value,
                count=len(items_to_forget),
                session_id=session_id
            ))
            
            # If we've forgotten enough items, stop
            item_count = sum(len(items) for items in self._session_memory[session_id].values())
            if item_count < self.max_capacity:
                break

    def _add_to_cache(self, item: MemoryItem) -> None:
        """Add item to recent access cache."""
        # If cache is full, remove least recently used item
        if len(self._recent_access_cache) >= self._max_cache_size:
            # Simple implementation - just remove a random item
            # In a real system, would use LRU or other cache eviction policy
            if self._recent_access_cache:
                self._recent_access_cache.pop(next(iter(self._recent_access_cache)))
        
        # Add to cache
        self._recent_access_cache[item.memory_id] = item

    async def get_stats(self) -> Dict[str, Any]:
        """Get working memory statistics."""
        stats = {
            "total_items": 0,
            "items_by_priority": {},
            "items_by_session": {},
            "consolidation_candidates": len(self._consolidation_candidates),
            "cache_size": len(self._recent_access_cache),
            "memory_type": MemoryType.WORKING.value
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
        
        return stats


class EpisodicMemory(BaseMemory):
    """
    Episodic memory implementation - stores experience-based memories.
    
    Episodic memory stores events, experiences, and interactions that the
    assistant has encountered, organized chronologically and by context.
    It supports time-based retrieval, associative recall, and emotional tagging.
    
    Features:
    - Chronological storage of experiences and events
    - Contextual organization by session, user, and interaction
    - Emotional tagging and sentiment association
    - Time-based and context-based retrieval
    - Memory strength decay and reinforcement mechanisms
    """
    
    def __init__(self, 
                 container: Container,
                 memory_store: BaseMemoryStore,
                 model_router: Optional[ModelRouter] = None):
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
        
        # Get event bus
        self.event_bus = container.get(EventBus)
        
        # Monitoring
        try:
            self.metrics = container.get(MetricsCollector)
            self.tracer = container.get(TraceManager)
        except Exception:
            self.metrics = None
            self.tracer = None
        
        # Indexes for fast lookup
        self._user_index: Dict[str, List[str]] = defaultdict(list)  # user_id -> memory_ids
        self._session_index: Dict[str, List[str]] = defaultdict(list)  # session_id -> memory_ids
        self._time_index: Dict[str, List[Tuple[datetime, str]]] = defaultdict(list)  # date_key -> [(timestamp, memory_id)]
        self._emotional_index: Dict[str, List[str]] = defaultdict(list)  # emotion -> memory_ids
        
        # Cache of recently accessed items
        self._recent_access_cache: Dict[str, MemoryItem] = {}
        self._max_cache_size = 1000
        
        # Initialize embedding index if model router available
        self._embedding_index = None
        if self.model_router:
            self._embedding_index = {}  # memory_id -> embedding
        
        self.logger.info("EpisodicMemory initialized")

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
                
        Returns:
            Memory ID
        """
        # Generate memory ID
        memory_id = MemoryUtils.generate_memory_id()
        
        # Extract emotional content if present
        emotion = kwargs.get('emotion', None)
        if not emotion and isinstance(data, dict) and 'emotional_state' in data:
            emotion = data['emotional_state']
        
        # Determine importance
        importance = kwargs.get('importance', 0.5)
        if not importance and isinstance(data, dict) and 'importance' in data:
            importance = data['importance']
        
        # Extract timestamp
        timestamp = kwargs.get('timestamp', datetime.now(timezone.utc))
        if isinstance(data, dict) and 'timestamp' in data:
            try:
                if isinstance(data['timestamp'], str):
                    timestamp = datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00'))
                elif isinstance(data['timestamp'], datetime):
                    timestamp = data['timestamp']
            except Exception:
                pass
        
        # Create memory item
        metadata = MemoryMetadata(
            created_at=timestamp,
            retention_policy=MemoryRetentionPolicy.STANDARD,
            tags=kwargs.get('tags', set()),
            importance=importance,
            custom_metadata={
                'emotion': emotion,
                'memory_strength': 1.0  # Initial strength
            }
        )
        
        memory_item = MemoryItem(
            memory_id=memory_id,
            content=data,
            memory_type=MemoryType.EPISODIC,
            owner_id=kwargs.get('user_id'),
            session_id=kwargs.get('session_id'),
            context_id=kwargs.get('context_id'),
            metadata=metadata
        )
        
        # Generate embeddings if model router available
        if self.model_router:
            try:
                # Convert to text for embedding
                if isinstance(data, dict):
                    embed_text = json.dumps(data)
                elif not isinstance(data, str):
                    embed_text = str(data)
                else:
                    embed_text = data
                
                memory_item.embeddings = await self.model_router.get_embeddings(embed_text)
                
                # Store in embedding index
                if self._embedding_index is not None and memory_item.embeddings:
                    self._embedding_index[memory_id] = memory_item.embeddings
            except Exception as e:
                self.logger.warning(f"Failed to generate embeddings: {str(e)}")
        
        # Store in memory store
        async with memory_operation_span(self.tracer, "store_episodic_memory", memory_id):
            await self.memory_store.store_item(memory_item)
        
        # Update indexes
        user_id = kwargs.get('user_id')
        if user_id:
            self._user_index[user_id].append(memory_id)
        
        session_id = kwargs.get('session_id')
        if session_id:
            self._session_index[session_id].append(memory_id)
        
        # Update time index
        date_key = timestamp.strftime('%Y-%m-%d')
        self._time_index[date_key].append((timestamp, memory_id))
        
        # Update emotional index
        if emotion:
            self._emotional_index[emotion].append(memory_id)
        
        # Add to cache
        self._add_to_cache(memory_item)
        
        # Emit event
        await self.event_bus.emit(MemoryItemStored(
            memory_id=memory_id,
            memory_type=MemoryType.EPISODIC.value,
            owner_id=kwargs.get('user_id'),
            context_id=kwargs.get('context_id')
        ))
        
        # Update metrics
        if self.metrics:
            self.metrics.increment("episodic_memory_items_stored")
            if user_id:
                user_items = len(self._user_index[user_id])
                self.metrics.gauge("episodic_memory_items_per_user", user_items, 
                                 {"user_id": user_id})
        
        return memory_id

    async def retrieve(self, memory_id: str) -> Optional[MemoryItem]:
        """
        Retrieve an item from episodic memory.
        
        Args:
            memory_id: Memory identifier
            
        Returns:
            Memory item or None if not found
        """
        # Check cache first
        if memory_id in self._recent_access_cache:
            item = self._recent_access_cache[memory_id]
            # Update metadata
            item.metadata.update_access()
            # Strengthen memory
            self._strengthen_memory(item)
            return item
        
        # Retrieve from store
        async with memory_operation_span(self.tracer, "retrieve_episodic_memory", memory_id):
            item = await self.memory_store.get_item(memory_id)
        
        if item:
            # Update metadata
            item.metadata.update_access()
            # Strengthen memory
            self._strengthen_memory(item)
            # Add to cache
            self._add_to_cache(item)
            # Store updated item
            await self.memory_store.update_item(memory_id, {"metadata": item.metadata})
            # Emit event
            await self.event_bus.emit(MemoryItemRetrieved(
                memory_id=memory_id,
                memory_type=MemoryType.EPISODIC.value,
                owner_id=item.owner_id
            ))
            # Update metrics
            if self.metrics:
                self.metrics.increment("episodic_memory_items_retrieved")
        
        return item

    async def search(self, query: Any) -> List[MemoryItem]:
        """
        Search episodic memory.
        
        Args:
            query: Search query (can be SimpleMemoryQuery, text, or other query types)
            
        Returns:
            List of matching memory items
        """
        if isinstance(query, SimpleMemoryQuery):
            # Use memory store query capabilities
            return await self.memory_store.query(query)
        
        elif isinstance(query, str):
            # Text-based search
            if query.startswith("user:"):
                # Search by user
                user_id = query.split(":", 1)[1].strip()
                return await self.get_user_memories(user_id)
            
            elif query.startswith("session:"):
                # Search by session
                session_id = query.split(":", 1)[1].strip()
                return await self.get_session_memories(session_id)
            
            elif query.startswith("date:"):
                # Search by date
                date_str = query.split(":", 1)[1].strip()
                return await self.get_memories_by_date(date_str)
            
            elif query.startswith("emotion:"):
                # Search by emotion
                emotion = query.split(":", 1)[1].strip()
                return await self.get_memories_by_emotion(emotion)
            
            elif query.startswith("timerange:"):
                # Search by time range
                timerange = query.split(":", 1)[1].strip()
                start_str, end_str = timerange.split("-")
                return await self.get_memories_by_timerange(start_str, end_str)
            
            else:
                # Semantic search if embeddings available
                if self.model_router and self._embedding_index:
                    return await self._semantic_search(query)
                else:
                    # Fall back to simple text search
                    return await self._text_search(query)
        
        else:
            # Unsupported query type
            raise MemoryError(f"Unsupported query type for episodic memory: {type(query)}")

    async def get_user_memories(self, user_id: str, limit: int = 100) -> List[MemoryItem]:
        """
        Get memories for a specific user.
        
        Args:
            user_id: User identifier
            limit: Maximum number of memories to return
            
        Returns:
            List of memory items
        """
        if user_id not in self._user_index:
            return []
        
        # Get memory IDs for user
        memory_ids = self._user_index[user_id][-limit:]
        
        # Retrieve items
        items = []
        for memory_id in memory_ids:
            item = await self.retrieve(memory_id)
            if item:
                items.append(item)
        
        # Sort by timestamp
        items.sort(key=lambda x: x.metadata.created_at, reverse=True)
        
        return items

    async def get_session_memories(self, session_id: str, limit: int = 100) -> List[MemoryItem]:
        """
        Get memories for a specific session.
        
        Args:
            session_id: Session identifier
            limit: Maximum number of memories to return
            
        Returns:
            List of memory items
        """
        if session_id not in self._session_index:
            return []
        
        # Get memory IDs for session
        memory_ids = self._session_index[session_id][-limit:]
        
        # Retrieve items
        items = []
        for memory_id in memory_ids:
            item = await self.retrieve(memory_id)
            if item:
                items.append(item)
        
        # Sort by timestamp
        items.sort(key=lambda x: x.metadata.created_at)
        
        return items

    async def get_memories_by_date(self, date_str: str, limit: int = 100) -> List[MemoryItem]:
        """
        Get memories for a specific date.
        
        Args:
            date_str: Date string (YYYY-MM-DD)
            limit: Maximum number of memories to return
            
        Returns:
            List of memory items
        """
        if date_str not in self._time_index:
            return []
        
        # Get memory IDs for date
        time_entries = sorted(self._time_index[date_str])[-limit:]
        memory_ids = [memory_id for _, memory_id in time_entries]
        
        # Retrieve items
        items = []
        for memory_id in memory_ids:
            item = await self.retrieve(memory_id)
            if item:
                items.append(item)
        
        return items

    async def get_memories_by_timerange(self, start_str: str, end_str: str, limit: int = 100) -> List[MemoryItem]:
        """
        Get memories within a time range.
        
        Args:
            start_str: Start date string (YYYY-MM-DD)
            end_str: End date string (YYYY-MM-DD)
            limit: Maximum number of memories to return
            
        Returns:
            List of memory items
        """
        try:
            # Parse dates
            start_date = datetime.strptime(start_str, '%Y-%m-%d').replace(tzinfo=timezone.utc)
            end_date = datetime.strptime(end_str, '%Y-%m-%d').replace(tzinfo=timezone.utc)
            end_date = end_date + timedelta(days=1) - timedelta(microseconds=1)  # End of day
            
            # Get all memory IDs in range
            memory_ids = []
            current_date = start_date
            while current_date <= end_date:
                date_key = current_date.strftime('%Y-%m-%d')
                if date_key in self._time_index:
                    for timestamp, memory_id in self._time_index[date_key]:
                        if start_date <= timestamp <= end_date:
                            memory_ids.append((timestamp, memory_id))
                current_date += timedelta(days=1)
            
            # Sort by timestamp and limit
            memory_ids.sort()
            memory_ids = memory_ids[-limit:]
            
            # Retrieve items
            items = []
            for _, memory_id in memory_ids:
                item = await self.retrieve(memory_id)
                if item:
                    items.append(item)
            
            return items
            
        except Exception as e:
            self.logger.error(f"Error retrieving memories by timerange: {str(e)}")
            return []

    async def get_memories_by_emotion(self, emotion: str, limit: int = 100) -> List[MemoryItem]:
        """
        Get memories associated with a specific emotion.
        
        Args:
            emotion: Emotion label
            limit: Maximum number of memories to return
            
        Returns:
            List of memory items
        """
        if emotion not in self._emotional_index:
            return []
        
        # Get memory IDs for emotion
        memory_ids = self._emotional_index[emotion][-limit:]
        
        # Retrieve items
        items = []
        for memory_id in memory_ids:
            item = await self.retrieve(memory_id)
            if item:
                items.append(item)
        
        # Sort by timestamp
        items.sort(key=lambda x: x.metadata.created_at, reverse=True)
        
        return items

    async def _semantic_search(self, query: str, limit: int = 10) -> List[MemoryItem]:
        """
        Perform semantic search on episodic memories.
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of memory items
        """
        if not self.model_router or not self._embedding_index:
            return []
        
        try:
            # Generate query embedding
            query_embedding = await self.model_router.get_embeddings(query)
            
            # Calculate similarities
            similarities = []
            for memory_id, embedding in self._embedding_index.items():
                # Calculate cosine similarity
                similarity = self._cosine_similarity(query_embedding, embedding)
                similarities.append((memory_id, similarity))
            
            # Sort by similarity and get top results
            similarities.sort(key=lambda x: x[1], reverse=True)
            top_memory_ids = [memory_id for memory_id, _ in similarities[:limit]]
            
            # Retrieve items
            items = []
            for memory_id in top_memory_ids:
                item = await self.retrieve(memory_id)
                if item:
                    items.append(item)
            
            return items
            
        except Exception as e:
            self.logger.error(f"Semantic search error: {str(e)}")
            return []

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between vectors."""
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = sum(a * a for a in vec1) ** 0.5
        magnitude2 = sum(b * b for b in vec2) ** 0.5
        
        if magnitude1 * magnitude2 == 0:
            return 0.0
            
        return dot_product / (magnitude1 * magnitude2)

    async def _text_search(self, query: str, limit: int = 10) -> List[MemoryItem]:
        """
        Perform simple text search on episodic memories.
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of memory items
        """
        # Get all episodic memories
        all_query = SimpleMemoryQuery(memory_type=MemoryType.EPISODIC, limit=1000)
        all_items = await self.memory_store.query(all_query)
        
        # Filter by query text
        matching_items = []
        for item in all_items:
            content = item.content
            if isinstance(content, dict):
                content = json.dumps(content)
            elif not isinstance(content, str):
                content = str(content)
            
            if query.lower() in content.lower():
                matching_items.append(item)
                
                # Strengthen matching memories
                self._strengthen_memory(item)
        
        # Sort by recency and limit
        matching_items.sort(key=lambda x: x.metadata.created_at, reverse=True)
        
        return matching_items[:limit]

    def _strengthen_memory(self, item: MemoryItem) -> None:
        """
        Strengthen a memory based on access.
        
        Args:
            item: Memory item to strengthen
        """
        # Get current strength
        custom_metadata = item.metadata.custom_metadata
        current_strength = custom_metadata.get('memory_strength', 1.0)
        
        # Increase strength (with diminishing returns)
        new_strength = current_strength + (1.0 - current_strength) * 0.1
        new_strength = min(new_strength, 2.0)  # Cap at 2.0
        
        # Update strength
        custom_metadata['memory_strength'] = new_strength
        
        # Adjust importance based on strength
        item.metadata.importance = min(1.0, item.metadata.importance * (new_strength / 1.0))

    def _add_to_cache(self, item: MemoryItem) -> None:
        """Add item to recent access cache."""
        # If cache is full, remove oldest item
        if len(self._recent_access_cache) >= self._max_cache_size:
            # Simple implementation - just remove a random item
            if self._recent_access_cache:
                self._recent_access_cache.pop(next(iter(self._recent_access_cache)))
        
        # Add to cache
        self._recent_access_cache[item.memory_id] = item

    async def clear(self) -> None:
        """Clear all episodic memory."""
        # Clear indexes
        self._user_index.clear()
        self._session_index.clear()
        self._time_index.clear()
        self._emotional_index.clear()
        
        # Clear embedding index
        if self._embedding_index is not None:
            self._embedding_index.clear()
        
        # Clear cache
        self._recent_access_cache.clear()
        
        # Clear underlying store (only episodic memory items)
        query = SimpleMemoryQuery(memory_type=MemoryType.EPISODIC)
        items = await self.memory_store.query(query)
        
        for item in items:
            await self.memory_store.delete_item(item.memory_id)
        
        self.logger.info(f"Cleared all episodic memory ({len(items)} items)")

    async def apply_memory_decay(self, decay_rate: float = 0.05) -> int:
        """
        Apply memory decay to episodic memories.
        
        Args:
            decay_rate: Rate at which memories decay (0-1)
            
        Returns:
            Number of memories affected
        """
        # Get all episodic memories
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
            
            # Calculate decay factor based on age and access
            age_factor = min(days_since_access / 365, 1.0)
            access_factor = 1.0 / (item.metadata.access_count + 1)
            
            decay_factor = decay_rate * age_factor * access_factor
            
            # Get current strength
            custom_metadata = item.metadata.custom_metadata
            current_strength = custom_metadata.get('memory_strength', 1.0)
            
            # Apply decay
            new_strength = max(0.1, current_strength * (1.0 - decay_factor))
            
            # Update if changed significantly
            if abs(new_strength - current_strength) > 0.01:
                custom_metadata['memory_strength'] = new_strength
                
                # Update importance based on strength
                item.metadata.importance = item.metadata.importance * (new_strength / current_strength)
                
                # Store updated item
                await self.memory_store.update_item(item.memory_id, {
                    "metadata": item.metadata
                })
                
                affected_count += 1
                
                # Update cache if present
                if item.memory_id in self._recent_access_cache:
                    self._recent_access_cache[item.memory_id] = item
        
        # Emit event
        await self.event_bus.emit(MemoryDecayApplied(
            memory_type=MemoryType.EPISODIC.value,
            count=affected_count
        ))
        
        self.logger.info(f"Applied memory decay to {affected_count} episodic memories")
        
        return affected_count

    async def get_stats(self) -> Dict[str, Any]:
        """Get episodic memory statistics."""
        stats = {
            "total_items": sum(len(memories) for memories in self._user_index.values()),
            "users_count": len(self._user_index),
            "sessions_count": len(self._session_index),
            "date_entries": len(self._time_index),
            "emotions_tracked": len(self._emotional_index),
            "cache_size": len(self._recent_access_cache),
            "embedding_count": len(self._embedding_index) if self._embedding_index is not None else 0,
            "memory_type": MemoryType.EPISODIC.value
        }
        
        # Add emotional breakdown
        stats["emotion_breakdown"] = {
            emotion: len(memories) for emotion, memories in self._emotional_index.items()
        }
        
        return stats


class SemanticMemory(BaseMemory):
    """
    Semantic memory implementation - stores factual and conceptual knowledge.
    
    Semantic memory stores general knowledge, facts, concepts, and information
    that is not tied to specific experiences or episodes. It serves as the
    assistant's knowledge base for answering questions and reasoning.
    
    Features:
    - Vector-based knowledge representation
    - Concept hierarchies and relationships
    - Fast semantic search and retrieval
    - Knowledge graph integration
    - Fact verification and consistency checking
    """
    
    def __init__(self, 
                 container: Container,
                 vector_store: VectorMemoryStore,
                 model_router: ModelRouter):
        """
        Initialize semantic memory.
        
        Args:
            container: Dependency injection container
            vector_store: Vector store for embeddings
            model_router: Model router for embeddings
        """
        self.container = container
        self.vector_store = vector_store
        self.model_router = model_router
        self.logger = get_logger(__name__)
        
        # Get event bus
        self.event_bus = container.get(EventBus)
        
        # Get graph store if available
        try:
            self.graph_store = container.get(MemoryGraphStore)
        except Exception:
            self.logger.warning("Graph store not available, semantic relationships limited")
            self.graph_store = None
        
        # Monitoring
        try:
            self.metrics = container.get(MetricsCollector)
            self.tracer = container.get(TraceManager)
        except Exception:
            self.metrics = None
            self.tracer = None
        
        # Semantic indices
        self._concept_index: Dict[str, List[str]] = defaultdict(list)  # concept -> memory_ids
        self._domain_index: Dict[str, List[str]] = defaultdict(list)  # domain -> memory_ids
        self._entity_index: Dict[str, List[str]] = defaultdict(list)  # entity -> memory_ids
        
        # Validation tracking
        self._fact_confidence: Dict[str, float] = {}  # memory_id -> confidence
        self._fact_validation_status: Dict[str, str] = {}  # memory_id -> status
        
        # Cache
        self._semantic_cache: Dict[str, MemoryItem] = {}
        self._max_cache_size = 1000
        
        self.logger.info("SemanticMemory initialized")

    async def store(self, data: Any, **kwargs) -> str:
        """
        Store data in semantic memory.
        
        Args:
            data: Data to store (fact, concept, knowledge)
            **kwargs: Additional parameters including:
                concepts: List of related concepts
                domain: Knowledge domain
                entities: List of related entities
                confidence: Fact confidence (0-1)
                relationships: Dict of relationships to other concepts
                
        Returns:
            Memory ID
        """
        # Generate memory ID
        memory_id = MemoryUtils.generate_memory_id()
        
        # Extract metadata
        concepts = kwargs.get('concepts', [])
        domain = kwargs.get('domain', 'general')
        entities = kwargs.get('entities', [])
        confidence = kwargs.get('confidence', 0.9)
        relationships = kwargs.get('relationships', {})
        
        # Generate embeddings
        try:
            # Convert to text for embedding
            if isinstance(data, dict):
                embed_text = json.dumps(data)
            elif not isinstance(data, str):
                embed_text = str(data)
            else:
                embed_text = data
            
            embeddings = await self.model_router.get_embeddings(embed_text)
        except Exception as e:
            self.logger.error(f"Failed to generate embeddings: {str(e)}")
            embeddings = None
        
        # Create memory item
        metadata = MemoryMetadata(
            retention_policy=MemoryRetentionPolicy.EXTENDED,
            tags=set(concepts) | {domain} | set(entities),
            importance=kwargs.get('importance', 0.7),
            custom_metadata={
                'domain': domain,
                'concepts': concepts,
                'entities': entities,
                'confidence': confidence,
                'validation_status': 'unverified'
            }
        )
        
        memory_item = MemoryItem(
            memory_id=memory_id,
            content=data,
            memory_type=MemoryType.SEMANTIC,
            owner_id=kwargs.get('owner_id'),
            context_id=kwargs.get('context_id'),
            metadata=metadata,
            embeddings=embeddings,
            relationships=relationships
        )
        
        # Store in vector store
        async with memory_operation_span(self.tracer, "store_semantic_memory", memory_id):
            await self.vector_store.store_item(memory_item)
        
        # Update semantic indices
        for concept in concepts:
            self._concept_index[concept].append(memory_id)
        
        self._domain_index[domain].append(memory_id)
        
        for entity in entities:
            self._entity_index[entity].append(memory_id)
        
        # Store confidence
        self._fact_confidence[memory_id] = confidence
        self._fact_validation_status[memory_id] = 'unverified'
        
        # Store in graph if available
        if self.graph_store and relationships:
            await self._store_semantic_relationships(memory_id, data, relationships)
        
        # Add to cache
        self._add_to_cache(memory_item)
        
        # Emit event
        await self.event_bus.emit(MemoryItemStored(
            memory_id=memory_id,
            memory_type=MemoryType.SEMANTIC.value,
            owner_id=kwargs.get('owner_id'),
            context_id=kwargs.get('context_id')
        ))
        
        # Update metrics
        if self.metrics:
            self.metrics.increment("semantic_memory_items_stored")
            self.metrics.gauge("semantic_concepts_count", len(self._concept_index))
        
        return memory_id

    async def retrieve(self, memory_id: str) -> Optional[MemoryItem]:
        """
        Retrieve an item from semantic memory.
        
        Args:
            memory_id: Memory identifier
            
        Returns:
            Memory item or None if not found
        """
        # Check cache first
        if memory_id in self._semantic_cache:
            item = self._semantic_cache[memory_id]
            # Update metadata
            item.metadata.update_access()
            return item
        
        # Retrieve from store
        async with memory_operation_span(self.tracer, "retrieve_semantic_memory", memory_id):
            item = await self.vector_store.get_item(memory_id)
        
        if item:
            # Update metadata
            item.metadata.update_access()
            # Add to cache
            self._add_to_cache(item)
            # Store updated item
            await self.vector_store.update_item(memory_id, {"metadata": item.metadata})
            # Emit event
            await self.event_bus.emit(MemoryItemRetrieved(
                memory_id=memory_id,
                memory_type=MemoryType.SEMANTIC.value,
                owner_id=item.owner_id
            ))
            # Update metrics
            if self.metrics:
                self.metrics.increment("semantic_memory_items_retrieved")
        
        return item

    async def search(self, query: Any) -> List[MemoryItem]:
        """
        Search semantic memory.
        
        Args:
            query: Search query (text, embedding, or query object)
            
        Returns:
            List of matching memory items
        """
        if isinstance(query, str):
            # Text query - convert to embedding for semantic search
            try:
                query_embedding = await self.model_router.get_embeddings(query)
                return await self.vector_store.similarity_search(
                    query_embedding,
                    similarity_threshold=0.7,
                    top_k=10
                )
            except Exception as e:
                self.logger.error(f"Semantic search error: {str(e)}")
                # Fall back to concept/entity search
                return await self._concept_entity_search(query)
        
        elif isinstance(query, list) and all(isinstance(x, (int, float)) for x in query):
            # Vector query
            return await self.vector_store.similarity_search(
                query,
                similarity_threshold=0.7,
                top_k=10
            )
        
        elif isinstance(query, SimpleMemoryQuery):
            # Use memory store query capabilities
            return await self.vector_store.query(query)
        
        else:
            # Unsupported query type
            raise MemoryError(f"Unsupported query type for semantic memory: {type(query)}")

    async def _concept_entity_search(self, query: str) -> List[MemoryItem]:
        """
        Search by concepts and entities in the query.
        
        Args:
            query: Search query
            
        Returns:
            List of matching memory items
        """
        # Extract potential concepts and entities from query
        query_terms = query.lower().split()
        
        # Check for direct concept/entity matches
        matching_memory_ids = set()
        
        for concept in self._concept_index:
            if concept.lower() in query.lower():
                matching_memory_ids.update(self._concept_index[concept])
        
        for entity in self._entity_index:
            if entity.lower() in query.lower():
                matching_memory_ids.update(self._entity_index[entity])
        
        # Check for domain matches
        for domain in self._domain_index:
            if domain.lower() in query.lower():
                matching_memory_ids.update(self._domain_index[domain])
        
        # Retrieve matching items
        items = []
        for memory_id in matching_memory_ids:
            item = await self.retrieve(memory_id)
            if item:
                items.append(item)
        
        # Sort by confidence
        items.sort(key=lambda x: x.metadata.custom_metadata.get('confidence', 0.0), reverse=True)
        
        return items[:10]  # Limit to top 10

    async def retrieve_relevant(
        self, 
        query: str, 
        context: Optional[Dict[str, Any]] = None, 
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant semantic memories for a query and context.
        
        Args:
            query: Query text
            context: Optional context information
            limit: Maximum number of items to return
            
        Returns:
            List of relevant semantic knowledge items
        """
        # Start with semantic search
        try:
            items = await self.search(query)
            
            # If we have context, refine the results
            if context:
                # Boost items that match context concepts/entities
                scored_items = []
                
                for item in items:
                    # Base score from semantic similarity
                    score = 1.0
                    
                    # Check for context matches
                    if 'concepts' in context and 'concepts' in item.metadata.custom_metadata:
                        item_concepts = item.metadata.custom_metadata['concepts']
                        context_concepts = context['concepts']
                        shared_concepts = set(item_concepts) & set(context_concepts)
                        if shared_concepts:
                            score += 0.5 * (len(shared_concepts) / len(item_concepts))
                    
                    if 'domain' in context and 'domain' in item.metadata.custom_metadata:
                        if context['domain'] == item.metadata.custom_metadata['domain']:
                            score += 0.3
                    
                    # Boost based on confidence
                    confidence = item.metadata.custom_metadata.get('confidence', 0.5)
                    score *= confidence
                    
                    scored_items.append((item, score))
                
                # Sort by score
                scored_items.sort(key=lambda x: x[1], reverse=True)
                items = [item for item, _ in scored_items[:limit]]
            else:
                # Just take top items
                items = items[:limit]
            
            # Format response
            result = []
            for item in items:
                fact_data = {
                    'content': item.content,
                    'confidence': item.metadata.custom_metadata.get('confidence', 0.5),
                    'domain': item.metadata.custom_metadata.get('domain', 'general'),
                    'id': item.memory_id
                }
                result.append(fact_data)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error retrieving relevant semantic memories: {str(e)}")
            return []

    async def get_by_concept(self, concept: str, limit: int = 10) -> List[MemoryItem]:
        """
        Get memories related to a concept.
        
        Args:
            concept: Concept name
            limit: Maximum number of items to return
            
        Returns:
            List of memory items
        """
        if concept not in self._concept_index:
            return []
        
        # Get memory IDs for concept
        memory_ids = self._concept_index[concept][:limit]
        
        # Retrieve items
        items = []
        for memory_id in memory_ids:
            item = await self.retrieve(memory_id)
            if item:
                items.append(item)
        
        return items

    async def get_by_entity(self, entity: str, limit: int = 10) -> List[MemoryItem]:
        """
        Get memories related to an entity.
        
        Args:
            entity: Entity name
            limit: Maximum number of items to return
            
        Returns:
            List of memory items
        """
        if entity not in self._entity_index:
            return []
        
        # Get memory IDs for entity
        memory_ids = self._entity_index[entity][:limit]
        
        # Retrieve items
        items = []
        for memory_id in memory_ids:
            item = await self.retrieve(memory_id)
            if item:
                items.append(item)
        
        return items

    async def get_by_domain(self, domain: str, limit: int = 10) -> List[MemoryItem]:
        """
        Get memories in a specific knowledge domain.
        
        Args:
            domain: Knowledge domain
            limit: Maximum number of items to return
            
        Returns:
            List of memory items
        """
        if domain not in self._domain_index:
            return []
        
        # Get memory IDs for domain
        memory_ids = self._domain_index[domain][:limit]
        
        # Retrieve items
        items = []
        for memory_id in memory_ids:
            item = await self.retrieve(memory_id)
            if item:
                items.append(item)
        
        return items

    async def validate_fact(self, memory_id: str, validation_result: bool) -> None:
        """Validate a fact in semantic memory."""
        try:
            memory_item = await self.retrieve(memory_id)
            if memory_item:
                memory_item.metadata['validated'] = validation_result
                memory_item.metadata['validation_timestamp'] = datetime.now(timezone.utc)
                await self.store(memory_item)
        except Exception as e:
            self.logger.error(f"Error validating fact {memory_id}: {str(e)}")
