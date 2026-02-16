# Memory Management
from .hierarchical import HierarchicalMemory, MemoryTier

try:
    from .vector_store import LessonStore, VectorMemoryStore

    VECTOR_MEMORY_AVAILABLE = True
except ImportError:
    VECTOR_MEMORY_AVAILABLE = False
    VectorMemoryStore = None  # type: ignore
    LessonStore = None  # type: ignore

try:
    from .dream_processor import DreamProcessor, DreamResult

    DREAM_PROCESSOR_AVAILABLE = True
except ImportError:
    DREAM_PROCESSOR_AVAILABLE = False
    DreamProcessor = None  # type: ignore
    DreamResult = None  # type: ignore

__all__ = [
    "HierarchicalMemory",
    "MemoryTier",
    "VectorMemoryStore",
    "LessonStore",
    "VECTOR_MEMORY_AVAILABLE",
    "DreamProcessor",
    "DreamResult",
    "DREAM_PROCESSOR_AVAILABLE",
]
