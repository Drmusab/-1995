"""
Simple Memory Base Classes
Author: Drmusab

Simplified memory base classes providing only essential functionality
without complex dependencies.
"""

import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union


class MemoryType(Enum):
    """Types of memory in the system."""
    WORKING = "working"
    LONG_TERM = "long_term"
    SEMANTIC = "semantic"
    EPISODIC = "episodic"
    PROCEDURAL = "procedural"


class MemoryAccess(Enum):
    """Memory access levels."""
    PUBLIC = "public"
    PRIVATE = "private"
    SHARED = "shared"


class MemorySensitivity(Enum):
    """Memory sensitivity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class MemoryMetadata:
    """Metadata for memory items."""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_accessed: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    access_count: int = 0
    importance: float = 0.5
    tags: List[str] = field(default_factory=list)
    source: Optional[str] = None
    confidence: float = 1.0


@dataclass
class MemoryItem:
    """A single memory item."""
    memory_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: Any = None
    memory_type: MemoryType = MemoryType.WORKING
    owner_id: Optional[str] = None
    metadata: MemoryMetadata = field(default_factory=MemoryMetadata)
    access_level: MemoryAccess = MemoryAccess.PRIVATE
    sensitivity: MemorySensitivity = MemorySensitivity.LOW

    def touch(self):
        """Update access timestamp and count."""
        self.metadata.last_accessed = datetime.now(timezone.utc)
        self.metadata.access_count += 1


@dataclass
class MemoryQuery:
    """Query for retrieving memories."""
    content: Optional[str] = None
    memory_type: Optional[MemoryType] = None
    owner_id: Optional[str] = None
    tags: Optional[List[str]] = None
    limit: int = 10
    min_importance: float = 0.0
    max_age_hours: Optional[int] = None


class BaseMemory(ABC):
    """Abstract base class for memory implementations."""

    def __init__(self, memory_type: MemoryType):
        self.memory_type = memory_type
        self._items: Dict[str, MemoryItem] = {}

    @abstractmethod
    async def store(self, item: MemoryItem) -> bool:
        """Store a memory item."""
        pass

    @abstractmethod
    async def retrieve(self, memory_id: str) -> Optional[MemoryItem]:
        """Retrieve a memory item by ID."""
        pass

    @abstractmethod
    async def query(self, query: MemoryQuery) -> List[MemoryItem]:
        """Query for memory items."""
        pass

    @abstractmethod
    async def delete(self, memory_id: str) -> bool:
        """Delete a memory item."""
        pass

    async def update(self, memory_id: str, content: Any = None, metadata: MemoryMetadata = None) -> bool:
        """Update a memory item."""
        item = await self.retrieve(memory_id)
        if not item:
            return False

        if content is not None:
            item.content = content
        if metadata is not None:
            item.metadata = metadata

        return await self.store(item)

    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        return {
            "total_items": len(self._items),
            "memory_type": self.memory_type.value,
            "total_accesses": sum(item.metadata.access_count for item in self._items.values()),
        }


class SimpleMemory(BaseMemory):
    """Simple in-memory implementation of BaseMemory."""

    def __init__(self, memory_type: MemoryType, max_items: int = 1000):
        super().__init__(memory_type)
        self.max_items = max_items

    async def store(self, item: MemoryItem) -> bool:
        """Store a memory item."""
        try:
            # Enforce memory type
            item.memory_type = self.memory_type
            
            # Evict oldest items if at capacity
            if len(self._items) >= self.max_items:
                oldest_id = min(
                    self._items.keys(),
                    key=lambda k: self._items[k].metadata.last_accessed
                )
                del self._items[oldest_id]

            self._items[item.memory_id] = item
            return True
        except Exception:
            return False

    async def retrieve(self, memory_id: str) -> Optional[MemoryItem]:
        """Retrieve a memory item by ID."""
        item = self._items.get(memory_id)
        if item:
            item.touch()
        return item

    async def query(self, query: MemoryQuery) -> List[MemoryItem]:
        """Query for memory items."""
        results = []
        
        for item in self._items.values():
            # Filter by memory type
            if query.memory_type and item.memory_type != query.memory_type:
                continue
                
            # Filter by owner
            if query.owner_id and item.owner_id != query.owner_id:
                continue
                
            # Filter by importance
            if item.metadata.importance < query.min_importance:
                continue
                
            # Filter by age
            if query.max_age_hours:
                age_hours = (datetime.now(timezone.utc) - item.metadata.created_at).total_seconds() / 3600
                if age_hours > query.max_age_hours:
                    continue
                    
            # Filter by tags
            if query.tags:
                if not any(tag in item.metadata.tags for tag in query.tags):
                    continue
                    
            # Simple content matching
            if query.content:
                content_str = str(item.content).lower()
                if query.content.lower() not in content_str:
                    continue
                    
            results.append(item)
            
        # Sort by importance and access time
        results.sort(
            key=lambda x: (x.metadata.importance, x.metadata.last_accessed),
            reverse=True
        )
        
        return results[:query.limit]

    async def delete(self, memory_id: str) -> bool:
        """Delete a memory item."""
        if memory_id in self._items:
            del self._items[memory_id]
            return True
        return False

    async def clear(self) -> bool:
        """Clear all memory items."""
        self._items.clear()
        return True