"""
Memory Module
Author: Drmusab

Simplified memory management for the AI assistant with essential functionality
and minimal dependencies.
"""

# Import simplified memory components
from .simple_memory import (
    BaseMemory,
    MemoryItem,
    MemoryType,
    MemoryQuery,
    MemoryAccess,
    MemorySensitivity,
    MemoryMetadata,
    SimpleMemory,
)

from .simple_manager import MemoryManager
from .simple_working_memory import WorkingMemory
from .simple_vector_store import VectorStore, VectorMemoryStore
from .simple_operations import (
    MemoryConsolidator,
    MemoryContextManager,
    EnhancedRetrieval,
    ContextManager,
)

# Create aliases for backward compatibility
LongTermMemory = SimpleMemory
ShortTermMemory = SimpleMemory
EpisodicMemory = SimpleMemory
SemanticMemory = SimpleMemory
ProceduralMemory = SimpleMemory

# Legacy aliases
MemoryConsolidation = MemoryConsolidator

__all__ = [
    # Core Memory Components
    'BaseMemory',
    'MemoryItem',
    'MemoryType',
    'MemoryQuery',
    'MemoryAccess',
    'MemorySensitivity',
    'MemoryMetadata',
    'SimpleMemory',
    'MemoryManager',
    'WorkingMemory',
    'VectorStore',
    'VectorMemoryStore',
    # Operations
    'MemoryConsolidator',
    'MemoryContextManager',
    'EnhancedRetrieval',
    'ContextManager',
    # Backward compatibility aliases
    'LongTermMemory',
    'ShortTermMemory',
    'EpisodicMemory',
    'SemanticMemory',
    'ProceduralMemory',
    'MemoryConsolidation',
]