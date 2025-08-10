
# Compatibility shim for working_memory
import warnings
warnings.warn("Importing from memory.core_memory.working_memory is deprecated. Use memory.simple_working_memory.", DeprecationWarning)

from ..simple_working_memory import WorkingMemory
