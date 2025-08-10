"""
Memory Module
Author: Drmusab

This module provides comprehensive memory management for the AI assistant,
including different memory types, storage, operations, and caching.
"""

# Import only core memory classes to avoid import issues in other modules
from .core_memory import (
    BaseMemory,
    MemoryItem,
    MemoryType,
    MemoryQuery,
    EpisodicMemory,
    SemanticMemory,
    WorkingMemory,
)

# Conditional imports to avoid breaking dependencies
try:
    from .cache_manager import MemoryCacheManager
except ImportError:
    # MemoryCacheManager has dependency issues, skip for now
    MemoryCacheManager = None

try:
    from .core_memory import MemoryManager
except ImportError:
    # MemoryManager has dependency issues, skip for now  
    MemoryManager = None

try:
    from .operations import (
        MemoryConsolidation,
        ContextManager,
        EnhancedRetrieval,
    )
except ImportError:
    # Operations have dependency issues, skip for now
    MemoryConsolidation = None
    ContextManager = None
    EnhancedRetrieval = None

try:
    from .storage import (
        MemoryGraph,
        MemoryGraphStore,
        VectorMemoryStore,
    )
except ImportError:
    # Storage has dependency issues, skip for now
    MemoryGraph = None
    MemoryGraphStore = None
    VectorMemoryStore = None

__all__ = [
    # Core Memory (always available)
    'BaseMemory',
    'MemoryItem',
    'MemoryType', 
    'MemoryQuery',
    'EpisodicMemory',
    'SemanticMemory',
    'WorkingMemory',
    # Optional components (may be None if dependencies missing)
    'MemoryCacheManager',
    'MemoryManager',
    'MemoryConsolidation',
    'ContextManager',
    'EnhancedRetrieval',
    'MemoryGraph',
    'MemoryGraphStore',
    'VectorMemoryStore',
]