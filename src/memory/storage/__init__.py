
# Compatibility shim for storage module
import warnings
warnings.warn("Importing from memory.storage is deprecated. Use memory directly.", DeprecationWarning)

from ..simple_vector_store import VectorStore, VectorMemoryStore

__all__ = ['VectorStore', 'VectorMemoryStore']
