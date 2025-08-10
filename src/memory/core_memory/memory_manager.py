
# Compatibility shim for memory_manager
import warnings
warnings.warn("Importing from memory.core_memory.memory_manager is deprecated. Use memory.simple_manager.", DeprecationWarning)

from ..simple_manager import MemoryManager
