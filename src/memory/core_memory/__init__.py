
# Compatibility shim for core_memory module
import warnings
warnings.warn("Importing from memory.core_memory is deprecated. Use memory directly.", DeprecationWarning)

from ..simple_memory import BaseMemory, MemoryItem, MemoryType, MemoryQuery
from ..simple_manager import MemoryManager  
from ..simple_working_memory import WorkingMemory

# Legacy memory types - all use SimpleMemory
from ..simple_memory import SimpleMemory as EpisodicMemory
from ..simple_memory import SimpleMemory as LongTermMemory
from ..simple_memory import SimpleMemory as ProceduralMemory
from ..simple_memory import SimpleMemory as SemanticMemory
from ..simple_memory import SimpleMemory as ShortTermMemory

__all__ = [
    'BaseMemory', 'MemoryItem', 'MemoryType', 'MemoryQuery', 'MemoryManager',
    'WorkingMemory', 'EpisodicMemory', 'LongTermMemory', 'ProceduralMemory',
    'SemanticMemory', 'ShortTermMemory'
]
