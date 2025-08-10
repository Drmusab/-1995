
# Compatibility shim for consolidation
import warnings
warnings.warn("Importing from memory.operations.consolidation is deprecated. Use memory.simple_operations.", DeprecationWarning)

from ..simple_operations import MemoryConsolidator
