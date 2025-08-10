
# Compatibility shim for base_memory
import warnings
warnings.warn("Importing from memory.core_memory.base_memory is deprecated. Use memory.simple_memory.", DeprecationWarning)

from ..simple_memory import *
