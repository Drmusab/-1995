"""
Simple Memory Operations
Author: Drmusab

Minimal memory operations for basic functionality.
"""

from typing import Any, Dict, List, Optional

from .simple_memory import MemoryItem, MemoryQuery, MemoryType
from .simple_manager import MemoryManager


class MemoryConsolidator:
    """Simple memory consolidation operations."""

    def __init__(self, memory_manager: Optional[MemoryManager] = None):
        """Initialize consolidator."""
        self.memory_manager = memory_manager or MemoryManager()

    async def consolidate_memories(
        self,
        owner_id: Optional[str] = None,
        min_importance: float = 0.5
    ) -> int:
        """Consolidate memories by promoting important working memories to long-term."""
        try:
            # Get important working memories
            working_memories = await self.memory_manager.get_important_memories(
                memory_type=MemoryType.WORKING,
                min_importance=min_importance,
                owner_id=owner_id,
                limit=100
            )
            
            consolidated_count = 0
            
            for memory in working_memories:
                # Create long-term copy
                long_term_id = await self.memory_manager.store_memory(
                    content=memory.content,
                    memory_type=MemoryType.LONG_TERM,
                    owner_id=memory.owner_id,
                    importance=memory.metadata.importance,
                    tags=memory.metadata.tags + ["consolidated"]
                )
                
                if long_term_id:
                    # Remove from working memory
                    await self.memory_manager.delete_memory(
                        memory.memory_id,
                        MemoryType.WORKING
                    )
                    consolidated_count += 1
                    
            return consolidated_count
        except Exception:
            return 0


class MemoryContextManager:
    """Simple memory context management."""

    def __init__(self, memory_manager: Optional[MemoryManager] = None):
        """Initialize context manager."""
        self.memory_manager = memory_manager or MemoryManager()
        self._current_context: Dict[str, Any] = {}

    async def set_context(self, key: str, value: Any) -> None:
        """Set context variable."""
        self._current_context[key] = value

    async def get_context(self, key: str, default: Any = None) -> Any:
        """Get context variable."""
        return self._current_context.get(key, default)

    async def get_relevant_memories(
        self,
        context_query: str,
        owner_id: Optional[str] = None,
        limit: int = 10
    ) -> List[MemoryItem]:
        """Get memories relevant to current context."""
        try:
            return await self.memory_manager.search_memories(
                content=context_query,
                owner_id=owner_id,
                limit=limit
            )
        except Exception:
            return []

    async def clear_context(self) -> None:
        """Clear current context."""
        self._current_context.clear()

    def get_context_summary(self) -> Dict[str, Any]:
        """Get context summary."""
        return {
            "context_variables": len(self._current_context),
            "keys": list(self._current_context.keys())
        }


# Compatibility aliases for external imports
class EnhancedRetrieval:
    """Simple retrieval operations."""

    def __init__(self, memory_manager: Optional[MemoryManager] = None):
        """Initialize retrieval system."""
        self.memory_manager = memory_manager or MemoryManager()

    async def retrieve_similar(
        self,
        query: str,
        memory_type: Optional[MemoryType] = None,
        limit: int = 10
    ) -> List[MemoryItem]:
        """Retrieve similar memories."""
        return await self.memory_manager.search_memories(
            content=query,
            memory_type=memory_type,
            limit=limit
        )

    async def retrieve_by_importance(
        self,
        min_importance: float = 0.7,
        memory_type: Optional[MemoryType] = None,
        limit: int = 10
    ) -> List[MemoryItem]:
        """Retrieve memories by importance."""
        return await self.memory_manager.get_important_memories(
            memory_type=memory_type,
            min_importance=min_importance,
            limit=limit
        )


# For backwards compatibility
ContextManager = MemoryContextManager