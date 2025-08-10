"""
Simple Memory Manager
Author: Drmusab

Simplified memory manager providing essential memory management functionality
without complex dependencies.
"""

import asyncio
from typing import Any, Dict, List, Optional

from .simple_memory import (
    BaseMemory,
    MemoryItem,
    MemoryQuery,
    MemoryType,
    SimpleMemory,
    MemoryMetadata
)


class MemoryManager:
    """Simplified memory manager for the AI assistant."""

    def __init__(self):
        """Initialize the memory manager."""
        self._memories: Dict[MemoryType, BaseMemory] = {
            MemoryType.WORKING: SimpleMemory(MemoryType.WORKING, max_items=100),
            MemoryType.LONG_TERM: SimpleMemory(MemoryType.LONG_TERM, max_items=10000),
        }

    async def store_memory(
        self,
        content: Any,
        memory_type: MemoryType = MemoryType.WORKING,
        owner_id: Optional[str] = None,
        importance: float = 0.5,
        tags: Optional[List[str]] = None,
    ) -> Optional[str]:
        """Store a memory item."""
        try:
            metadata = MemoryMetadata(
                importance=importance,
                tags=tags or []
            )
            
            item = MemoryItem(
                content=content,
                memory_type=memory_type,
                owner_id=owner_id,
                metadata=metadata
            )

            memory = self._memories.get(memory_type)
            if memory and await memory.store(item):
                return item.memory_id
            
            return None
        except Exception:
            return None

    async def retrieve_memory(self, memory_id: str, memory_type: Optional[MemoryType] = None) -> Optional[MemoryItem]:
        """Retrieve a memory item by ID."""
        try:
            if memory_type:
                # Search specific memory type
                memory = self._memories.get(memory_type)
                if memory:
                    return await memory.retrieve(memory_id)
            else:
                # Search all memory types
                for memory in self._memories.values():
                    item = await memory.retrieve(memory_id)
                    if item:
                        return item
            return None
        except Exception:
            return None

    async def query_memories(self, query: MemoryQuery) -> List[MemoryItem]:
        """Query for memory items."""
        try:
            all_results = []
            
            if query.memory_type:
                # Query specific memory type
                memory = self._memories.get(query.memory_type)
                if memory:
                    results = await memory.query(query)
                    all_results.extend(results)
            else:
                # Query all memory types
                for memory in self._memories.values():
                    results = await memory.query(query)
                    all_results.extend(results)
            
            # Sort combined results by importance
            all_results.sort(
                key=lambda x: x.metadata.importance,
                reverse=True
            )
            
            return all_results[:query.limit]
        except Exception:
            return []

    async def delete_memory(self, memory_id: str, memory_type: Optional[MemoryType] = None) -> bool:
        """Delete a memory item."""
        try:
            if memory_type:
                memory = self._memories.get(memory_type)
                if memory:
                    return await memory.delete(memory_id)
            else:
                # Try deleting from all memory types
                for memory in self._memories.values():
                    if await memory.delete(memory_id):
                        return True
            return False
        except Exception:
            return False

    async def update_memory(
        self,
        memory_id: str,
        content: Any = None,
        importance: Optional[float] = None,
        tags: Optional[List[str]] = None,
        memory_type: Optional[MemoryType] = None
    ) -> bool:
        """Update a memory item."""
        try:
            item = await self.retrieve_memory(memory_id, memory_type)
            if not item:
                return False

            if content is not None:
                item.content = content
            if importance is not None:
                item.metadata.importance = importance
            if tags is not None:
                item.metadata.tags = tags

            memory = self._memories.get(item.memory_type)
            if memory:
                return await memory.store(item)
            
            return False
        except Exception:
            return False

    async def get_recent_memories(
        self,
        memory_type: Optional[MemoryType] = None,
        limit: int = 10,
        owner_id: Optional[str] = None
    ) -> List[MemoryItem]:
        """Get recent memories."""
        query = MemoryQuery(
            memory_type=memory_type,
            owner_id=owner_id,
            limit=limit
        )
        return await self.query_memories(query)

    async def get_important_memories(
        self,
        memory_type: Optional[MemoryType] = None,
        min_importance: float = 0.7,
        limit: int = 10,
        owner_id: Optional[str] = None
    ) -> List[MemoryItem]:
        """Get important memories."""
        query = MemoryQuery(
            memory_type=memory_type,
            owner_id=owner_id,
            min_importance=min_importance,
            limit=limit
        )
        return await self.query_memories(query)

    async def search_memories(
        self,
        content: str,
        memory_type: Optional[MemoryType] = None,
        limit: int = 10,
        owner_id: Optional[str] = None
    ) -> List[MemoryItem]:
        """Search memories by content."""
        query = MemoryQuery(
            content=content,
            memory_type=memory_type,
            owner_id=owner_id,
            limit=limit
        )
        return await self.query_memories(query)

    async def get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """Get user preferences from memory."""
        try:
            query = MemoryQuery(
                owner_id=user_id,
                tags=["preference", "settings"],
                memory_type=MemoryType.LONG_TERM,
                limit=50
            )
            
            preference_items = await self.query_memories(query)
            preferences = {}
            
            for item in preference_items:
                if isinstance(item.content, dict):
                    preferences.update(item.content)
                    
            return preferences
        except Exception:
            return {}

    async def store_user_preference(self, user_id: str, key: str, value: Any) -> bool:
        """Store a user preference."""
        try:
            content = {key: value}
            return await self.store_memory(
                content=content,
                memory_type=MemoryType.LONG_TERM,
                owner_id=user_id,
                importance=0.8,
                tags=["preference", "settings", key]
            ) is not None
        except Exception:
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        stats = {}
        for memory_type, memory in self._memories.items():
            stats[memory_type.value] = memory.get_stats()
        return stats

    async def clear_working_memory(self) -> bool:
        """Clear working memory."""
        try:
            memory = self._memories.get(MemoryType.WORKING)
            if memory and hasattr(memory, 'clear'):
                return await memory.clear()
            return False
        except Exception:
            return False

    async def cleanup_old_memories(self, max_age_hours: int = 168) -> int:
        """Clean up old memories (default: older than 1 week)."""
        try:
            cleaned_count = 0
            
            for memory_type, memory in self._memories.items():
                if memory_type == MemoryType.LONG_TERM:
                    # Don't auto-clean long term memories
                    continue
                    
                # Get all items for this memory type
                query = MemoryQuery(memory_type=memory_type, limit=10000)
                items = await memory.query(query)
                
                # Delete old items
                from datetime import datetime, timezone
                current_time = datetime.now(timezone.utc)
                
                for item in items:
                    age_hours = (current_time - item.metadata.created_at).total_seconds() / 3600
                    if age_hours > max_age_hours:
                        if await memory.delete(item.memory_id):
                            cleaned_count += 1
                            
            return cleaned_count
        except Exception:
            return 0