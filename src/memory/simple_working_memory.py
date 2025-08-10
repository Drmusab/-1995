"""
Simple Working Memory Implementation
Author: Drmusab

Simplified working memory for immediate processing and short-term storage.
"""

from typing import Any, Dict, List, Optional

from .simple_memory import BaseMemory, MemoryItem, MemoryQuery, MemoryType, SimpleMemory


class WorkingMemory(SimpleMemory):
    """Working memory for short-term storage and immediate processing."""

    def __init__(self, max_items: int = 50):
        """Initialize working memory with limited capacity."""
        super().__init__(MemoryType.WORKING, max_items)
        self._context: Dict[str, Any] = {}

    async def set_context(self, key: str, value: Any) -> None:
        """Set a context variable."""
        self._context[key] = value

    async def get_context(self, key: str, default: Any = None) -> Any:
        """Get a context variable."""
        return self._context.get(key, default)

    async def clear_context(self) -> None:
        """Clear all context variables."""
        self._context.clear()

    async def get_context_summary(self) -> Dict[str, Any]:
        """Get a summary of current context."""
        return {
            "context_variables": len(self._context),
            "memory_items": len(self._items),
            "context_keys": list(self._context.keys())
        }

    async def store_temporary(
        self,
        content: Any,
        owner_id: Optional[str] = None,
        ttl_minutes: int = 60
    ) -> Optional[str]:
        """Store temporary memory item with automatic expiration."""
        from datetime import datetime, timezone, timedelta
        
        # Create memory item with expiration metadata
        item = MemoryItem(
            content=content,
            memory_type=MemoryType.WORKING,
            owner_id=owner_id
        )
        
        # Add expiration info to metadata
        item.metadata.tags.append("temporary")
        item.metadata.source = f"ttl_{ttl_minutes}min"
        
        if await self.store(item):
            return item.memory_id
        return None

    async def cleanup_expired(self) -> int:
        """Clean up expired temporary items."""
        # This is a simplified version - in a real implementation,
        # you might want to track expiration times properly
        expired_count = 0
        expired_items = []
        
        for memory_id, item in self._items.items():
            if "temporary" in item.metadata.tags:
                # Simple check - remove items older than 1 hour
                from datetime import datetime, timezone, timedelta
                age = datetime.now(timezone.utc) - item.metadata.created_at
                if age > timedelta(hours=1):
                    expired_items.append(memory_id)
        
        for memory_id in expired_items:
            if await self.delete(memory_id):
                expired_count += 1
                
        return expired_count

    def get_stats(self) -> Dict[str, Any]:
        """Get working memory statistics."""
        stats = super().get_stats()
        stats.update({
            "context_variables": len(self._context),
            "temporary_items": sum(1 for item in self._items.values() 
                                 if "temporary" in item.metadata.tags),
            "max_capacity": self.max_items,
            "utilization": len(self._items) / self.max_items if self.max_items > 0 else 0
        })
        return stats