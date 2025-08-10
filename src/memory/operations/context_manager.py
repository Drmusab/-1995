
# Compatibility shim for context_manager
import warnings
warnings.warn("Importing from memory.operations.context_manager is deprecated. Use memory.simple_operations.", DeprecationWarning)

from ..simple_operations import MemoryContextManager
