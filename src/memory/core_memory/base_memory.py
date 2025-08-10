"""
Advanced Memory System Foundation
Author: Drmusab
Last Modified: 2025-07-04 19:55:00 UTC

This module provides the abstract base classes and foundational components for
the AI assistant's memory system, including memory interface definitions,
memory item representations, and core memory operations.
"""

import hashlib
import json
import logging
import time
import uuid
import weakref
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    AsyncGenerator,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Protocol,
    Set,
    Type,
    TypeVar,
    Union,
)

import asyncio

# Core imports
from src.core.config.loader import ConfigLoader
from src.core.dependency_injection import Container
from src.core.error_handling import ErrorHandler, handle_exceptions
from src.core.events.event_bus import EventBus
from src.core.events.event_types import (
    MemoryConsolidationCompleted,
    MemoryConsolidationStarted,
    MemoryItemDeleted,
    MemoryItemRetrieved,
    MemoryItemStored,
    MemoryItemUpdated,
)

# Optional imports to avoid breaking dependencies
try:
    from src.core.security.encryption import EncryptionManager
except ImportError:
    EncryptionManager = None

from src.observability.logging.config import get_logger

# Observability (optional imports to avoid breaking dependencies)
try:
    from src.observability.monitoring.metrics import MetricsCollector
except ImportError:
    MetricsCollector = None

try:
    from src.observability.monitoring.tracing import TraceManager
except ImportError:
    TraceManager = None

# Type definitions
T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")


class MemoryType(Enum):
    """Types of memory in the system."""

    WORKING = "working"  # Short-term, active information
    EPISODIC = "episodic"  # Event/experience-based memories
    SEMANTIC = "semantic"  # Factual, conceptual knowledge
    PROCEDURAL = "procedural"  # Skills, procedures, how-to knowledge
    SHORT_TERM = "short_term"  # Temporary, volatile memory
    LONG_TERM = "long_term"  # Persistent, stable memory
    SENSORY = "sensory"  # Raw perceptual data
    META = "meta"  # Memory about memory


class MemoryStorageType(Enum):
    """Storage types for memory items."""

    IN_MEMORY = "in_memory"  # Volatile, fast access
    PERSISTENT = "persistent"  # Durable storage
    DISTRIBUTED = "distributed"  # Shared across nodes
    VECTOR = "vector"  # Vector embeddings storage
    GRAPH = "graph"  # Graph-based storage
    HYBRID = "hybrid"  # Combination of storage types


class MemoryAccess(Enum):
    """Access levels for memory items."""

    PUBLIC = "public"  # Accessible to all
    PRIVATE = "private"  # Accessible only to owner
    SHARED = "shared"  # Accessible to specific entities
    SYSTEM = "system"  # Accessible only to system
    SENSITIVE = "sensitive"  # Requires extra authorization


class MemorySensitivity(Enum):
    """Sensitivity levels for memory items."""

    LOW = "low"  # Non-sensitive information
    MEDIUM = "medium"  # Moderately sensitive
    HIGH = "high"  # Highly sensitive, PII
    CRITICAL = "critical"  # Critical sensitive data


class MemoryRetentionPolicy(Enum):
    """Retention policies for memory items."""

    TRANSIENT = "transient"  # Very short-lived (minutes to hours)
    TEMPORARY = "temporary"  # Short-lived (hours to days)
    STANDARD = "standard"  # Normal retention (weeks to months)
    EXTENDED = "extended"  # Long retention (months to years)
    PERMANENT = "permanent"  # Never automatically deleted
    CUSTOM = "custom"  # Custom retention period


@dataclass
class MemoryMetadata:
    """Metadata for memory items."""

    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_accessed: Optional[datetime] = None
    access_count: int = 0
    last_modified: Optional[datetime] = None
    modification_count: int = 0
    expiration: Optional[datetime] = None
    access_level: MemoryAccess = MemoryAccess.PRIVATE
    sensitivity: MemorySensitivity = MemorySensitivity.LOW
    retention_policy: MemoryRetentionPolicy = MemoryRetentionPolicy.STANDARD
    retention_period: Optional[timedelta] = None
    tags: Set[str] = field(default_factory=set)
    source: Optional[str] = None
    confidence: float = 1.0
    importance: float = 0.5
    checksum: Optional[str] = None
    version: int = 1
    encryption_status: bool = False
    custom_metadata: Dict[str, Any] = field(default_factory=dict)

    def update_access(self) -> None:
        """Update access metadata."""
        self.last_accessed = datetime.now(timezone.utc)
        self.access_count += 1

    def update_modification(self) -> None:
        """Update modification metadata."""
        self.last_modified = datetime.now(timezone.utc)
        self.modification_count += 1
        self.version += 1

    def is_expired(self) -> bool:
        """Check if memory has expired."""
        if not self.expiration:
            return False
        return datetime.now(timezone.utc) > self.expiration

    def calculate_checksum(self, content: Any) -> str:
        """Calculate checksum for memory content."""
        if isinstance(content, dict):
            content_str = json.dumps(content, sort_keys=True)
        elif not isinstance(content, str):
            content_str = str(content)
        else:
            content_str = content

        return hashlib.sha256(content_str.encode()).hexdigest()

    def verify_checksum(self, content: Any) -> bool:
        """Verify checksum against content."""
        if not self.checksum:
            return True
        current_checksum = self.calculate_checksum(content)
        return current_checksum == self.checksum


@dataclass
class MemoryItem(Generic[T]):
    """Generic memory item container."""

    memory_id: str
    content: T
    memory_type: MemoryType
    owner_id: Optional[str] = None
    session_id: Optional[str] = None
    context_id: Optional[str] = None
    metadata: MemoryMetadata = field(default_factory=MemoryMetadata)
    relationships: Dict[str, List[str]] = field(default_factory=dict)
    embeddings: Optional[List[float]] = None

    def __post_init__(self):
        """Post initialization processing."""
        if not self.memory_id:
            self.memory_id = str(uuid.uuid4())

        # Calculate checksum if not present
        if not self.metadata.checksum:
            self.metadata.checksum = self.metadata.calculate_checksum(self.content)


class MemoryError(Exception):
    """Base exception for memory operations."""

    def __init__(
        self,
        message: str,
        memory_id: Optional[str] = None,
        error_code: Optional[str] = None,
        owner_id: Optional[str] = None,
    ):
        super().__init__(message)
        self.memory_id = memory_id
        self.error_code = error_code
        self.owner_id = owner_id
        self.timestamp = datetime.now(timezone.utc)


class MemoryAccessError(MemoryError):
    """Exception for memory access violations."""

    pass


class MemoryNotFoundError(MemoryError):
    """Exception for memory not found."""

    pass


class MemoryCorruptionError(MemoryError):
    """Exception for corrupted memory."""

    pass


class MemoryStorageError(MemoryError):
    """Exception for memory storage issues."""

    pass


class MemoryOperationError(MemoryError):
    """Exception for memory operation failures."""

    pass


class MemoryQuery(Protocol):
    """Protocol for memory query interface."""

    async def execute(self, memory_store: "BaseMemoryStore") -> List[MemoryItem]:
        """Execute query against memory store."""
        ...


@dataclass
class SimpleMemoryQuery:
    """Simple query implementation."""

    memory_type: Optional[MemoryType] = None
    owner_id: Optional[str] = None
    session_id: Optional[str] = None
    context_id: Optional[str] = None
    tags: Optional[Set[str]] = None
    time_range: Optional[tuple[datetime, datetime]] = None
    limit: int = 100
    offset: int = 0

    async def execute(self, memory_store: "BaseMemoryStore") -> List[MemoryItem]:
        """Execute query against memory store."""
        return await memory_store.query(self)


@dataclass
class VectorMemoryQuery:
    """Vector-based similarity query."""

    query_vector: List[float]
    memory_type: Optional[MemoryType] = None
    similarity_threshold: float = 0.7
    top_k: int = 10

    async def execute(self, memory_store: "BaseMemoryStore") -> List[MemoryItem]:
        """Execute vector query against memory store."""
        if hasattr(memory_store, "similarity_search"):
            return await memory_store.similarity_search(
                self.query_vector,
                memory_type=self.memory_type,
                similarity_threshold=self.similarity_threshold,
                top_k=self.top_k,
            )
        return []


@dataclass
class SemanticMemoryQuery:
    """Semantic query implementation."""

    query_text: str
    memory_type: Optional[MemoryType] = None
    relevance_threshold: float = 0.7
    top_k: int = 10

    async def execute(self, memory_store: "BaseMemoryStore") -> List[MemoryItem]:
        """Execute semantic query against memory store."""
        if hasattr(memory_store, "semantic_search"):
            return await memory_store.semantic_search(
                self.query_text,
                memory_type=self.memory_type,
                relevance_threshold=self.relevance_threshold,
                top_k=self.top_k,
            )
        return []


class MemorySearchResult:
    """Container for memory search results."""

    def __init__(
        self,
        items: List[MemoryItem],
        total_count: int,
        query_time: float,
        relevance_scores: Optional[Dict[str, float]] = None,
    ):
        self.items = items
        self.total_count = total_count
        self.query_time = query_time
        self.relevance_scores = relevance_scores or {}

    def get_most_relevant(self, top_k: int = 5) -> List[MemoryItem]:
        """Get most relevant items based on scores."""
        if not self.relevance_scores:
            return self.items[:top_k]

        sorted_items = sorted(
            self.items,
            key=lambda item: self.relevance_scores.get(item.memory_id, 0.0),
            reverse=True,
        )
        return sorted_items[:top_k]

    def get_item_by_id(self, memory_id: str) -> Optional[MemoryItem]:
        """Get specific item by ID."""
        for item in self.items:
            if item.memory_id == memory_id:
                return item
        return None

    def filter_by_type(self, memory_type: MemoryType) -> "MemorySearchResult":
        """Filter results by memory type."""
        filtered_items = [item for item in self.items if item.memory_type == memory_type]
        return MemorySearchResult(
            filtered_items,
            len(filtered_items),
            self.query_time,
            {
                k: v
                for k, v in self.relevance_scores.items()
                if k in [item.memory_id for item in filtered_items]
            },
        )

    def filter_by_tags(self, tags: Set[str]) -> "MemorySearchResult":
        """Filter results by tags."""
        filtered_items = [
            item for item in self.items if any(tag in item.metadata.tags for tag in tags)
        ]
        return MemorySearchResult(
            filtered_items,
            len(filtered_items),
            self.query_time,
            {
                k: v
                for k, v in self.relevance_scores.items()
                if k in [item.memory_id for item in filtered_items]
            },
        )


class BaseMemory(ABC):
    """Abstract base class for memory implementations."""

    @abstractmethod
    async def store(self, data: Any, **kwargs) -> str:
        """
        Store data in memory.

        Args:
            data: Data to store
            **kwargs: Additional parameters

        Returns:
            Memory ID
        """
        pass

    @abstractmethod
    async def retrieve(self, memory_id: str) -> Optional[MemoryItem]:
        """
        Retrieve memory by ID.

        Args:
            memory_id: Memory identifier

        Returns:
            Memory item or None if not found
        """
        pass

    @abstractmethod
    async def update(self, memory_id: str, data: Any) -> bool:
        """
        Update existing memory.

        Args:
            memory_id: Memory identifier
            data: New data

        Returns:
            True if successful
        """
        pass

    @abstractmethod
    async def delete(self, memory_id: str) -> bool:
        """
        Delete memory.

        Args:
            memory_id: Memory identifier

        Returns:
            True if successful
        """
        pass

    @abstractmethod
    async def search(self, query: Any) -> MemorySearchResult:
        """
        Search memory.

        Args:
            query: Search query

        Returns:
            Search results
        """
        pass

    @abstractmethod
    async def clear(self) -> None:
        """Clear all memory."""
        pass

    @abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get memory statistics.

        Returns:
            Memory statistics
        """
        pass


class BaseMemoryStore(ABC):
    """Abstract base class for memory storage implementations."""

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the memory store."""
        pass

    @abstractmethod
    async def store_item(self, item: MemoryItem) -> None:
        """
        Store a memory item.

        Args:
            item: Memory item to store
        """
        pass

    @abstractmethod
    async def get_item(self, memory_id: str) -> Optional[MemoryItem]:
        """
        Get a memory item by ID.

        Args:
            memory_id: Memory identifier

        Returns:
            Memory item or None if not found
        """
        pass

    @abstractmethod
    async def update_item(self, memory_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update a memory item.

        Args:
            memory_id: Memory identifier
            updates: Fields to update

        Returns:
            True if successful
        """
        pass

    @abstractmethod
    async def delete_item(self, memory_id: str) -> bool:
        """
        Delete a memory item.

        Args:
            memory_id: Memory identifier

        Returns:
            True if successful
        """
        pass

    @abstractmethod
    async def query(self, query: SimpleMemoryQuery) -> List[MemoryItem]:
        """
        Query memory items.

        Args:
            query: Query parameters

        Returns:
            List of matching memory items
        """
        pass

    @abstractmethod
    async def count(self, memory_type: Optional[MemoryType] = None) -> int:
        """
        Count memory items.

        Args:
            memory_type: Optional filter by memory type

        Returns:
            Number of items
        """
        pass

    @abstractmethod
    async def clear_all(self) -> None:
        """Clear all memory items."""
        pass

    @abstractmethod
    async def backup(self, backup_path: Path) -> bool:
        """
        Backup memory store.

        Args:
            backup_path: Path to backup file

        Returns:
            True if successful
        """
        pass

    @abstractmethod
    async def restore(self, backup_path: Path) -> bool:
        """
        Restore from backup.

        Args:
            backup_path: Path to backup file

        Returns:
            True if successful
        """
        pass

    @abstractmethod
    async def get_statistics(self) -> Dict[str, Any]:
        """
        Get store statistics.

        Returns:
            Store statistics
        """
        pass


class MemoryIndexer(ABC):
    """Abstract base class for memory indexing."""

    @abstractmethod
    async def index_item(self, item: MemoryItem) -> None:
        """
        Index a memory item.

        Args:
            item: Memory item to index
        """
        pass

    @abstractmethod
    async def remove_from_index(self, memory_id: str) -> None:
        """
        Remove item from index.

        Args:
            memory_id: Memory identifier
        """
        pass

    @abstractmethod
    async def search_index(self, query: str, **kwargs) -> List[str]:
        """
        Search the index.

        Args:
            query: Search query
            **kwargs: Additional parameters

        Returns:
            List of matching memory IDs
        """
        pass

    @abstractmethod
    async def rebuild_index(self) -> None:
        """Rebuild the entire index."""
        pass


class MemoryCache:
    """Memory caching implementation."""

    def __init__(self, max_size: int = 1000, ttl_seconds: float = 300.0):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._access_times: Dict[str, float] = {}
        self._expiration_times: Dict[str, float] = {}
        self._cache_lock = asyncio.Lock()
        self.logger = get_logger(__name__)

        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    async def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        async with self._cache_lock:
            current_time = time.time()

            # Check if key exists and not expired
            if key in self._cache:
                if current_time < self._expiration_times.get(key, 0):
                    self._access_times[key] = current_time
                    self.hits += 1
                    return self._cache[key]
                else:
                    # Expired
                    self._remove_item(key)

            self.misses += 1
            return None

    async def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Set item in cache."""
        async with self._cache_lock:
            current_time = time.time()

            # Enforce size limit with LRU eviction
            if len(self._cache) >= self.max_size and key not in self._cache:
                await self._evict_lru_item()

            self._cache[key] = value
            self._access_times[key] = current_time
            self._expiration_times[key] = current_time + (ttl or self.ttl_seconds)

    async def remove(self, key: str) -> None:
        """Remove item from cache."""
        async with self._cache_lock:
            self._remove_item(key)

    async def clear(self) -> None:
        """Clear entire cache."""
        async with self._cache_lock:
            self._cache.clear()
            self._access_times.clear()
            self._expiration_times.clear()

    async def _evict_lru_item(self) -> None:
        """Evict least recently used item."""
        if not self._access_times:
            return

        lru_key = min(self._access_times.items(), key=lambda x: x[1])[0]
        self._remove_item(lru_key)
        self.evictions += 1

    def _remove_item(self, key: str) -> None:
        """Remove item from all internal structures."""
        self._cache.pop(key, None)
        self._access_times.pop(key, None)
        self._expiration_times.pop(key, None)

    async def cleanup_expired(self) -> int:
        """Clean up expired items."""
        async with self._cache_lock:
            current_time = time.time()
            expired_keys = [
                key for key, expiry in self._expiration_times.items() if current_time > expiry
            ]

            for key in expired_keys:
                self._remove_item(key)

            return len(expired_keys)

    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        async with self._cache_lock:
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "ttl_seconds": self.ttl_seconds,
                "hits": self.hits,
                "misses": self.misses,
                "hit_ratio": (
                    self.hits / (self.hits + self.misses) if (self.hits + self.misses) > 0 else 0
                ),
                "evictions": self.evictions,
            }


class AbstractMemoryManager(ABC):
    """Abstract interface for memory management."""

    @abstractmethod
    async def store_memory(self, data: Any, memory_type: MemoryType, **kwargs) -> str:
        """Store data in appropriate memory system."""
        pass

    @abstractmethod
    async def retrieve_memory(self, memory_id: str) -> Optional[MemoryItem]:
        """Retrieve memory by ID."""
        pass

    @abstractmethod
    async def search_memories(
        self, query: Any, memory_type: Optional[MemoryType] = None
    ) -> MemorySearchResult:
        """Search memories."""
        pass

    @abstractmethod
    async def get_recent_memories(
        self, memory_type: Optional[MemoryType] = None, limit: int = 10
    ) -> List[MemoryItem]:
        """Get recent memories."""
        pass

    @abstractmethod
    async def get_memory_by_context(self, context_id: str) -> List[MemoryItem]:
        """Get memories associated with a context."""
        pass

    @abstractmethod
    async def consolidate_memories(self) -> None:
        """Consolidate memories from short-term to long-term storage."""
        pass

    @abstractmethod
    async def forget_memory(self, memory_id: str) -> bool:
        """Explicitly forget a memory."""
        pass

    @abstractmethod
    async def get_memory_statistics(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics."""
        pass


class MemoryAccessController:
    """Controls access to memory items based on security policies."""

    def __init__(self, encryption_manager: Optional[EncryptionManager] = None):
        self.encryption_manager = encryption_manager
        self.logger = get_logger(__name__)

    async def check_access(self, memory_item: MemoryItem, user_id: Optional[str] = None) -> bool:
        """
        Check if access is allowed to memory item.

        Args:
            memory_item: Memory item to check
            user_id: User requesting access

        Returns:
            True if access is allowed
        """
        # Public items are accessible to all
        if memory_item.metadata.access_level == MemoryAccess.PUBLIC:
            return True

        # System items require system access
        if memory_item.metadata.access_level == MemoryAccess.SYSTEM:
            # Would implement system-level authorization check
            return False

        # Private items are only accessible to owner
        if memory_item.metadata.access_level == MemoryAccess.PRIVATE:
            return memory_item.owner_id == user_id

        # Shared items have specific access control
        if memory_item.metadata.access_level == MemoryAccess.SHARED:
            # Would check if user_id is in shared list
            return memory_item.owner_id == user_id

        # Sensitive items require extra verification
        if memory_item.metadata.access_level == MemoryAccess.SENSITIVE:
            # Would implement additional authorization checks
            return memory_item.owner_id == user_id

        return False

    async def encrypt_memory_content(self, item: MemoryItem) -> MemoryItem:
        """
        Encrypt memory content if required.

        Args:
            item: Memory item to encrypt

        Returns:
            Memory item with encrypted content if needed
        """
        # Skip if encryption not available or not sensitive
        if not self.encryption_manager or item.metadata.sensitivity == MemorySensitivity.LOW:
            return item

        # Only encrypt sensitive content
        if item.metadata.sensitivity in [MemorySensitivity.HIGH, MemorySensitivity.CRITICAL]:
            try:
                content_str = (
                    json.dumps(item.content)
                    if isinstance(item.content, dict)
                    else str(item.content)
                )
                encrypted_content = await self.encryption_manager.encrypt(content_str)

                # Create new item with encrypted content
                encrypted_item = MemoryItem(
                    memory_id=item.memory_id,
                    content=encrypted_content,
                    memory_type=item.memory_type,
                    owner_id=item.owner_id,
                    session_id=item.session_id,
                    context_id=item.context_id,
                    metadata=item.metadata,
                    relationships=item.relationships,
                    embeddings=item.embeddings,
                )

                encrypted_item.metadata.encryption_status = True
                return encrypted_item

            except Exception as e:
                self.logger.error(f"Failed to encrypt memory {item.memory_id}: {str(e)}")

        return item

    async def decrypt_memory_content(self, item: MemoryItem) -> MemoryItem:
        """
        Decrypt memory content if encrypted.

        Args:
            item: Memory item to decrypt

        Returns:
            Memory item with decrypted content
        """
        if not self.encryption_manager or not item.metadata.encryption_status:
            return item

        try:
            decrypted_content = await self.encryption_manager.decrypt(item.content)

            # Try to parse JSON if possible
            try:
                parsed_content = json.loads(decrypted_content)
            except json.JSONDecodeError:
                parsed_content = decrypted_content

            # Create new item with decrypted content
            decrypted_item = MemoryItem(
                memory_id=item.memory_id,
                content=parsed_content,
                memory_type=item.memory_type,
                owner_id=item.owner_id,
                session_id=item.session_id,
                context_id=item.context_id,
                metadata=item.metadata,
                relationships=item.relationships,
                embeddings=item.embeddings,
            )

            decrypted_item.metadata.encryption_status = False
            return decrypted_item

        except Exception as e:
            self.logger.error(f"Failed to decrypt memory {item.memory_id}: {str(e)}")
            return item


class MemoryUtils:
    """Utility functions for memory operations."""

    @staticmethod
    def generate_memory_id() -> str:
        """Generate a unique memory ID."""
        return str(uuid.uuid4())

    @staticmethod
    def calculate_expiration(retention_policy: MemoryRetentionPolicy) -> Optional[datetime]:
        """Calculate expiration time based on retention policy."""
        now = datetime.now(timezone.utc)

        if retention_policy == MemoryRetentionPolicy.TRANSIENT:
            return now + timedelta(hours=1)
        elif retention_policy == MemoryRetentionPolicy.TEMPORARY:
            return now + timedelta(days=1)
        elif retention_policy == MemoryRetentionPolicy.STANDARD:
            return now + timedelta(days=30)
        elif retention_policy == MemoryRetentionPolicy.EXTENDED:
            return now + timedelta(days=365)
        elif retention_policy == MemoryRetentionPolicy.PERMANENT:
            return None

        # Default
        return now + timedelta(days=30)

    @staticmethod
    def extract_entities(text: str) -> List[str]:
        """Extract entity mentions from text."""
        # Simplified entity extraction
        # In a real implementation, this would use NLP techniques
        words = text.split()
        entities = []

        for word in words:
            # Look for capitalized words as potential entities
            if word and word[0].isupper() and len(word) > 1:
                entities.append(word)

        return entities

    @staticmethod
    def extract_tags_from_content(content: Any) -> Set[str]:
        """Extract potential tags from content."""
        if isinstance(content, str):
            text = content
        elif isinstance(content, dict):
            text = " ".join(str(v) for v in content.values() if isinstance(v, (str, int, float)))
        else:
            text = str(content)

        # Extract hashtags
        hashtags = {word.strip("#") for word in text.split() if word.startswith("#")}

        # Extract key terms (simplified)
        words = text.lower().split()
        key_terms = {word for word in words if len(word) > 5 and words.count(word) > 1}

        return hashtags.union(key_terms)

    @staticmethod
    def serialize_memory_item(item: MemoryItem) -> Dict[str, Any]:
        """Serialize memory item to dictionary."""
        result = {
            "memory_id": item.memory_id,
            "content": item.content,
            "memory_type": item.memory_type.value,
            "owner_id": item.owner_id,
            "session_id": item.session_id,
            "context_id": item.context_id,
            "relationships": item.relationships,
        }

        # Serialize metadata
        metadata = asdict(item.metadata)
        # Convert datetime objects to ISO format
        for key, value in metadata.items():
            if isinstance(value, datetime):
                metadata[key] = value.isoformat()
            elif isinstance(value, timedelta):
                metadata[key] = value.total_seconds()
            elif isinstance(value, Enum):
                metadata[key] = value.value

        result["metadata"] = metadata

        # Add embeddings if present
        if item.embeddings:
            result["embeddings"] = item.embeddings

        return result

    @staticmethod
    def deserialize_memory_item(data: Dict[str, Any]) -> MemoryItem:
        """Deserialize memory item from dictionary."""
        # Extract and convert metadata
        metadata_dict = data.get("metadata", {})

        # Convert string dates back to datetime
        date_fields = ["created_at", "last_accessed", "last_modified", "expiration"]
        for field in date_fields:
            if field in metadata_dict and metadata_dict[field]:
                try:
                    metadata_dict[field] = datetime.fromisoformat(metadata_dict[field])
                except (ValueError, TypeError):
                    metadata_dict[field] = None

        # Convert enum strings back to enums
        if "access_level" in metadata_dict:
            metadata_dict["access_level"] = MemoryAccess(metadata_dict["access_level"])
        if "sensitivity" in metadata_dict:
            metadata_dict["sensitivity"] = MemorySensitivity(metadata_dict["sensitivity"])
        if "retention_policy" in metadata_dict:
            metadata_dict["retention_policy"] = MemoryRetentionPolicy(
                metadata_dict["retention_policy"]
            )

        # Convert retention period back to timedelta
        if "retention_period" in metadata_dict and metadata_dict["retention_period"]:
            metadata_dict["retention_period"] = timedelta(
                seconds=float(metadata_dict["retention_period"])
            )

        # Create metadata object
        metadata = MemoryMetadata(**metadata_dict)

        # Create memory item
        return MemoryItem(
            memory_id=data["memory_id"],
            content=data["content"],
            memory_type=MemoryType(data["memory_type"]),
            owner_id=data.get("owner_id"),
            session_id=data.get("session_id"),
            context_id=data.get("context_id"),
            metadata=metadata,
            relationships=data.get("relationships", {}),
            embeddings=data.get("embeddings"),
        )


# Define memory-related observability functions
def register_memory_metrics(metrics) -> None:
    """Register memory-related metrics."""
    if metrics is None:
        return
    metrics.register_counter("memory_operations_total")
    metrics.register_counter("memory_store_operations")
    metrics.register_counter("memory_retrieve_operations")
    metrics.register_counter("memory_update_operations")
    metrics.register_counter("memory_delete_operations")
    metrics.register_counter("memory_search_operations")

    metrics.register_histogram("memory_operation_duration_seconds")
    metrics.register_histogram("memory_item_size_bytes")

    metrics.register_gauge("memory_items_total")
    metrics.register_gauge("memory_working_items")
    metrics.register_gauge("memory_episodic_items")
    metrics.register_gauge("memory_semantic_items")

    metrics.register_counter("memory_cache_hits")
    metrics.register_counter("memory_cache_misses")
    metrics.register_counter("memory_errors_total")


@asynccontextmanager
async def memory_operation_span(
    tracer: Optional[TraceManager], operation: str, memory_id: Optional[str] = None
):
    """Context manager for tracing memory operations."""
    if tracer:
        with tracer.trace(f"memory_{operation}") as span:
            if memory_id:
                span.set_attributes({"memory_id": memory_id})
            try:
                yield
            except Exception as e:
                span.set_attributes({"error": True, "error.message": str(e)})
                raise
    else:
        yield
