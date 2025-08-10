
# Compatibility shim for retrieval
import warnings
warnings.warn("Importing from memory.operations.retrieval is deprecated. Use memory.simple_operations.", DeprecationWarning)

from ..simple_operations import EnhancedRetrieval
