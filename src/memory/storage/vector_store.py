
# Compatibility shim for vector_store
import warnings
warnings.warn("Importing from memory.storage.vector_store is deprecated. Use memory.simple_vector_store.", DeprecationWarning)

from ..simple_vector_store import *
