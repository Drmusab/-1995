
# Compatibility shim for operations module
import warnings
warnings.warn("Importing from memory.operations is deprecated. Use memory directly.", DeprecationWarning)

from ..simple_operations import MemoryConsolidator, MemoryContextManager, EnhancedRetrieval, ContextManager

# Legacy aliases
MemoryConsolidation = MemoryConsolidator

__all__ = ['MemoryConsolidator', 'MemoryContextManager', 'EnhancedRetrieval', 'ContextManager', 'MemoryConsolidation']
