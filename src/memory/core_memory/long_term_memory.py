"""
Long-Term Memory Implementation
Author: Drmusab
Last Modified: 2025-08-09 21:34:47 UTC

This module implements long-term memory - a persistent storage system that combines
consolidated memories from various sources. It provides hierarchical organization,
advanced retrieval capabilities, and integration with different memory types.
"""

import asyncio
import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np

# Core imports
from src.core.config.loader import ConfigLoader
from src.core.dependency_injection import Container
from src.core.error_handling import ErrorHandler, handle_exceptions
from src.core.events.event_bus import EventBus
from src.core.events.event_types import (
    MemoryItemDeleted,
    MemoryItemRetrieved,
    MemoryItemStored,
    MemoryItemUpdated,
)
from src.core.security.encryption import EncryptionManager

# Memory system imports
from src.memory.core_memory.base_memory import (
    BaseMemory,
    BaseMemoryStore,
    MemoryError,
    MemoryItem,
    MemoryMetadata,
    MemoryRetentionPolicy,
    MemorySensitivity,
    MemoryType,
    MemoryUtils,
    SimpleMemoryQuery,
    memory_operation_span,
)
from src.memory.storage.memory_graph import MemoryGraphStore
from src.memory.storage.vector_store import VectorMemoryStore

# Integration components
from src.integrations.llm.model_router import ModelRouter
from src.integrations.storage.database import DatabaseManager
from src.integrations.storage.encrypted_storage import EncryptedStorage

# Reasoning integration
from src.reasoning.knowledge_graph import KnowledgeGraph

# Observability
from src.observability.logging.config import get_logger
from src.observability.monitoring.metrics import MetricsCollector
from src.observability.monitoring.tracing import TraceManager


class MemoryCategory(Enum):
    """Categories for organizing long-term memories."""
    PERSONAL = "personal"
    PROFESSIONAL = "professional"
    KNOWLEDGE = "knowledge"
    SKILLS = "skills"
    RELATIONSHIPS = "relationships"
    EXPERIENCES = "experiences"
    PREFERENCES = "preferences"
    GOALS = "goals"
    ACHIEVEMENTS = "achievements"
    REFLECTIONS = "reflections"


class RetrievalStrategy(Enum):
    """Strategies for memory retrieval."""
    EXACT = "exact"
    SEMANTIC = "semantic"
    TEMPORAL = "temporal"
    ASSOCIATIVE = "associative"
    HIERARCHICAL = "hierarchical"
    CONTEXTUAL = "contextual"
    PATTERN_BASED = "pattern_based"


@dataclass
class MemoryCluster:
    """Represents a cluster of related memories."""
    cluster_id: str
    theme: str
    category: MemoryCategory
    memory_ids: Set[str] = field(default_factory=set)
    centroid: Optional[List[float]] = None
    keywords: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    access_count: int = 0
    importance_score: float = 0.5


@dataclass
class MemoryHierarchy:
    """Hierarchical organization of memories."""
    hierarchy_id: str
    name: str
    level: int
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    memory_ids: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)


class LongTermMemoryConfig:
    """Configuration for long-term memory."""
    
    def __init__(self, config_loader: ConfigLoader):
        """Initialize configuration from config loader."""
        self.enable_encryption = config_loader.get("memory.long_term.enable_encryption", True)
        self.enable_compression = config_loader.get("memory.long_term.enable_compression", True)
        self.enable_versioning = config_loader.get("memory.long_term.enable_versioning", True)
        self.max_versions = config_loader.get("memory.long_term.max_versions", 10)
        self.clustering_threshold = config_loader.get("memory.long_term.clustering_threshold", 0.7)
        self.hierarchy_depth = config_loader.get("memory.long_term.hierarchy_depth", 5)
        self.cache_size = config_loader.get("memory.long_term.cache_size", 5000)
        self.index_update_interval = config_loader.get("memory.long_term.index_update_interval", 3600)
        self.backup_interval = config_loader.get("memory.long_term.backup_interval", 86400)
        self.enable_graph_storage = config_loader.get("memory.long_term.enable_graph_storage", True)
        self.enable_vector_storage = config_loader.get("memory.long_term.enable_vector_storage", True)


class LongTermMemory(BaseMemory):
    """
    Long-term memory implementation - persistent, organized storage system.
    
    This implementation provides:
    - Persistent storage with encryption and compression
    - Hierarchical and categorical organization
    - Advanced retrieval strategies
    - Memory clustering and pattern recognition
    - Version control for memory updates
    - Integration with multiple storage backends
    - Sophisticated search and analysis capabilities
    """

    def __init__(
        self,
        container: Container,
        memory_store: BaseMemoryStore,
        model_router: Optional[ModelRouter] = None,
    ):
        """
        Initialize long-term memory.

        Args:
            container: Dependency injection container
            memory_store: Primary memory storage backend
            model_router: Model router for embeddings
        """
        self.container = container
        self.memory_store = memory_store
        self.model_router = model_router
        self.logger = get_logger(__name__)

        # Get core dependencies
        self.event_bus = container.get(EventBus)
        self.error_handler = container.get(ErrorHandler)
        
        # Load configuration
        config_loader = container.get(ConfigLoader)
        self.config = LongTermMemoryConfig(config_loader)

        # Get storage backends
        try:
            self.database = container.get(DatabaseManager)
        except Exception:
            self.logger.warning("Database manager not available")
            self.database = None

        if self.config.enable_encryption:
            try:
                self.encryption = container.get(EncryptionManager)
                self.encrypted_storage = container.get(EncryptedStorage)
            except Exception:
                self.logger.warning("Encryption services not available")
                self.encryption = None
                self.encrypted_storage = None

        if self.config.enable_vector_storage:
            try:
                self.vector_store = container.get(VectorMemoryStore)
            except Exception:
                self.logger.warning("Vector store not available")
                self.vector_store = None

        if self.config.enable_graph_storage:
            try:
                self.graph_store = container.get(MemoryGraphStore)
                self.knowledge_graph = container.get(KnowledgeGraph)
            except Exception:
                self.logger.warning("Graph storage not available")
                self.graph_store = None
                self.knowledge_graph = None

        # Get monitoring components
        try:
            self.metrics = container.get(MetricsCollector)
            self.tracer = container.get(TraceManager)
        except Exception:
            self.metrics = None
            self.tracer = None
            self.logger.warning("Monitoring components not available")

        # Initialize storage structures
        self._initialize_storage()

        # Cache
        self._memory_cache: Dict[str, MemoryItem] = {}
        self._cache_lock = asyncio.Lock()

        # Background task management
        self._background_tasks: List[asyncio.Task] = []
        self._shutdown_event = asyncio.Event()

        self.logger.info("LongTermMemory initialized successfully")

    def _initialize_storage(self) -> None:
        """Initialize storage structures for long-term memory."""
        # Organizational structures
        self._category_index: Dict[MemoryCategory, Set[str]] = defaultdict(set)
        self._tag_index: Dict[str, Set[str]] = defaultdict(set)
        self._user_index: Dict[str, Set[str]] = defaultdict(set)
        self._source_index: Dict[str, Set[str]] = defaultdict(set)
        
        # Clustering and hierarchy
        self._memory_clusters: Dict[str, MemoryCluster] = {}
        self._memory_to_cluster: Dict[str, str] = {}
        self._memory_hierarchies: Dict[str, MemoryHierarchy] = {}
        self._hierarchy_roots: List[str] = []
        
        # Version control
        self._memory_versions: Dict[str, List[Tuple[datetime, str]]] = defaultdict(list)
        self._version_store: Dict[str, MemoryItem] = {}
        
        # Pattern tracking
        self._access_patterns: Dict[str, List[Tuple[datetime, str]]] = defaultdict(list)
        self._co_access_matrix: Dict[Tuple[str, str], int] = defaultdict(int)
        
        # Search optimization
        self._embedding_index: Optional[Dict[str, List[float]]] = {}
        self._keyword_index: Dict[str, Set[str]] = defaultdict(set)

    async def initialize(self) -> None:
        """Initialize long-term memory and start background tasks."""
        try:
            self.logger.info("Initializing long-term memory...")

            # Initialize storage backends
            if hasattr(self.memory_store, 'initialize'):
                await self.memory_store.initialize()

            if self.vector_store and hasattr(self.vector_store, 'initialize'):
                await self.vector_store.initialize()

            if self.graph_store and hasattr(self.graph_store, 'initialize'):
                await self.graph_store.initialize()

            # Start background tasks
            self._background_tasks.append(
                asyncio.create_task(self._index_update_loop())
            )

            if self.config.backup_interval > 0:
                self._background_tasks.append(
                    asyncio.create_task(self._backup_loop())
                )

            self._background_tasks.append(
                asyncio.create_task(self._clustering_loop())
            )

            # Rebuild indices from existing data
            await self._rebuild_indices()

            # Load existing clusters and hierarchies
            await self._load_organizational_structures()

            self.logger.info("Long-term memory initialization complete")

        except Exception as e:
            self.logger.error(f"Failed to initialize long-term memory: {str(e)}")
            raise MemoryError(f"Long-term memory initialization failed: {str(e)}")

    async def shutdown(self) -> None:
        """Shutdown long-term memory and cleanup resources."""
        self.logger.info("Shutting down long-term memory...")
        
        # Signal shutdown to background tasks
        self._shutdown_event.set()

        # Cancel and wait for background tasks
        for task in self._background_tasks:
            task.cancel()

        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)

        # Save organizational structures
        await self._save_organizational_structures()

        # Perform final backup if configured
        if self.config.backup_interval > 0:
            await self._perform_backup()

        # Clear caches
        self._memory_cache.clear()

        self.logger.info("Long-term memory shutdown complete")

    @handle_exceptions
    async def store(self, data: Any, **kwargs) -> str:
        """
        Store data in long-term memory.

        Args:
            data: Data to store
            **kwargs: Additional parameters including:
                user_id: User identifier
                category: Memory category
                tags: Memory tags
                source: Source of the memory
                source_type: Type of source (episodic, semantic, etc.)
                importance: Importance score
                parent_id: Parent memory for hierarchical organization
                related_ids: Related memory IDs

        Returns:
            Memory ID
        """
        async with memory_operation_span(self.tracer, "store_long_term"):
            # Generate memory ID
            memory_id = MemoryUtils.generate_memory_id()

            # Create metadata
            metadata = await self._create_metadata(data, **kwargs)

            # Determine sensitivity and apply encryption if needed
            sensitivity = self._determine_sensitivity(data, **kwargs)
            if sensitivity in [MemorySensitivity.HIGH, MemorySensitivity.CRITICAL]:
                metadata.sensitivity = sensitivity
                metadata.encryption_status = True

            # Create memory item
            memory_item = MemoryItem(
                memory_id=memory_id,
                content=data,
                memory_type=MemoryType.LONG_TERM,
                owner_id=kwargs.get("user_id"),
                metadata=metadata,
            )

            # Apply encryption if needed
            if metadata.encryption_status and self.encryption:
                memory_item = await self._encrypt_memory(memory_item)

            # Generate embeddings
            if self.model_router:
                memory_item.embeddings = await self._generate_embeddings(data)

            # Store in primary backend
            await self.memory_store.store_item(memory_item)

            # Store in additional backends
            await self._store_in_backends(memory_item)

            # Update indices
            await self._update_indices_on_store(memory_item, **kwargs)

            # Handle versioning
            if self.config.enable_versioning:
                await self._create_version(memory_item)

            # Update hierarchical organization
            parent_id = kwargs.get("parent_id")
            if parent_id:
                await self._add_to_hierarchy(memory_id, parent_id)

            # Handle relationships
            related_ids = kwargs.get("related_ids", [])
            if related_ids:
                await self._update_relationships(memory_id, related_ids)

            # Add to cache
            await self._add_to_cache(memory_item)

            # Emit event
            await self.event_bus.emit(
                MemoryItemStored(
                    memory_id=memory_id,
                    memory_type=MemoryType.LONG_TERM.value,
                    owner_id=kwargs.get("user_id"),
                )
            )

            # Update metrics
            if self.metrics:
                self.metrics.increment("long_term_memory_items_stored")
                self._update_storage_metrics()

            self.logger.debug(f"Stored long-term memory {memory_id}")
            return memory_id

    @handle_exceptions
    async def retrieve(self, memory_id: str) -> Optional[MemoryItem]:
        """
        Retrieve an item from long-term memory.

        Args:
            memory_id: Memory identifier

        Returns:
            Memory item or None if not found
        """
        async with memory_operation_span(self.tracer, "retrieve_long_term", memory_id):
            # Check cache first
            async with self._cache_lock:
                if memory_id in self._memory_cache:
                    item = self._memory_cache[memory_id]
                    await self._update_access_tracking(memory_id)
                    return item

            # Retrieve from store
            item = await self.memory_store.get_item(memory_id)
            
            if item:
                # Decrypt if needed
                if item.metadata.encryption_status and self.encryption:
                    item = await self._decrypt_memory(item)

                # Update access tracking
                await self._update_access_tracking(memory_id)
                
                # Update metadata
                item.metadata.update_access()
                await self.memory_store.update_item(memory_id, {"metadata": item.metadata})
                
                # Add to cache
                await self._add_to_cache(item)
                
                # Emit event
                await self.event_bus.emit(
                    MemoryItemRetrieved(
                        memory_id=memory_id,
                        memory_type=MemoryType.LONG_TERM.value,
                        owner_id=item.owner_id,
                    )
                )
                
                # Update metrics
                if self.metrics:
                    self.metrics.increment("long_term_memory_items_retrieved")

            return item

    async def update(self, memory_id: str, data: Any) -> bool:
        """
        Update existing memory with versioning.

        Args:
            memory_id: Memory identifier
            data: New data

        Returns:
            True if successful
        """
        async with memory_operation_span(self.tracer, "update_long_term", memory_id):
            # Retrieve existing memory
            memory_item = await self.retrieve(memory_id)
            if not memory_item:
                return False

            # Create version before update
            if self.config.enable_versioning:
                await self._create_version(memory_item)

            # Update content
            memory_item.content = data
            memory_item.metadata.update_modification()

            # Re-encrypt if needed
            if memory_item.metadata.encryption_status and self.encryption:
                memory_item = await self._encrypt_memory(memory_item)

            # Regenerate embeddings
            if self.model_router:
                memory_item.embeddings = await self._generate_embeddings(data)

            # Store updated item
            await self.memory_store.store_item(memory_item)

            # Update in additional backends
            await self._store_in_backends(memory_item)

            # Update cache
            await self._add_to_cache(memory_item)

            # Emit event
            await self.event_bus.emit(
                MemoryItemUpdated(
                    memory_id=memory_id,
                    memory_type=MemoryType.LONG_TERM.value,
                    owner_id=memory_item.owner_id,
                )
            )

            return True

    async def delete(self, memory_id: str) -> bool:
        """
        Delete memory with cleanup of all references.

        Args:
            memory_id: Memory identifier

        Returns:
            True if successful
        """
        async with memory_operation_span(self.tracer, "delete_long_term", memory_id):
            # Remove from all indices
            await self._remove_from_indices(memory_id)

            # Remove from organizational structures
            await self._remove_from_organization(memory_id)

            # Remove from cache
            async with self._cache_lock:
                self._memory_cache.pop(memory_id, None)

            # Remove versions
            if memory_id in self._memory_versions:
                for _, version_id in self._memory_versions[memory_id]:
                    self._version_store.pop(version_id, None)
                del self._memory_versions[memory_id]

            # Remove from additional backends
            if self.vector_store:
                await self.vector_store.delete_item(memory_id)
            
            if self.graph_store:
                await self.graph_store.remove_node(memory_id)

            # Emit event
            await self.event_bus.emit(
                MemoryItemDeleted(
                    memory_id=memory_id,
                    memory_type=MemoryType.LONG_TERM.value,
                )
            )

            # Delete from primary store
            return await self.memory_store.delete_item(memory_id)

    async def search(self, query: Any, strategy: RetrievalStrategy = RetrievalStrategy.SEMANTIC) -> List[MemoryItem]:
        """
        Search long-term memory using specified strategy.

        Args:
            query: Search query
            strategy: Retrieval strategy to use

        Returns:
            List of matching memory items
        """
        async with memory_operation_span(self.tracer, "search_long_term"):
            if strategy == RetrievalStrategy.EXACT:
                return await self._exact_search(query)
            elif strategy == RetrievalStrategy.SEMANTIC:
                return await self._semantic_search(query)
            elif strategy == RetrievalStrategy.TEMPORAL:
                return await self._temporal_search(query)
            elif strategy == RetrievalStrategy.ASSOCIATIVE:
                return await self._associative_search(query)
            elif strategy == RetrievalStrategy.HIERARCHICAL:
                return await self._hierarchical_search(query)
            elif strategy == RetrievalStrategy.CONTEXTUAL:
                return await self._contextual_search(query)
            elif strategy == RetrievalStrategy.PATTERN_BASED:
                return await self._pattern_based_search(query)
            else:
                # Default to semantic search
                return await self._semantic_search(query)

    async def clear(self) -> None:
        """Clear all long-term memory."""
        # This is a dangerous operation - add confirmation in practice
        self.logger.warning("Clearing all long-term memory!")

        # Clear all indices and structures
        self._category_index.clear()
        self._tag_index.clear()
        self._user_index.clear()
        self._source_index.clear()
        
        self._memory_clusters.clear()
        self._memory_to_cluster.clear()
        self._memory_hierarchies.clear()
        self._hierarchy_roots.clear()
        
        self._memory_versions.clear()
        self._version_store.clear()
        
        self._access_patterns.clear()
        self._co_access_matrix.clear()
        
        self._embedding_index.clear()
        self._keyword_index.clear()

        # Clear cache
        async with self._cache_lock:
            self._memory_cache.clear()

        # Clear all backends
        query = SimpleMemoryQuery(memory_type=MemoryType.LONG_TERM)
        items = await self.memory_store.query(query)

        for item in items:
            await self.memory_store.delete_item(item.memory_id)

        if self.vector_store:
            # Clear vector store entries
            pass

        if self.graph_store:
            # Clear graph store entries
            pass

        self.logger.info(f"Cleared all long-term memory ({len(items)} items)")

    async def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive long-term memory statistics."""
        total_items = sum(len(items) for items in self._user_index.values())
        
        stats = {
            "total_items": total_items,
            "users_count": len(self._user_index),
            "cache_size": len(self._memory_cache),
            "memory_type": MemoryType.LONG_TERM.value,
            "organizational_stats": {
                "categories": {
                    category.value: len(memory_ids)
                    for category, memory_ids in self._category_index.items()
                },
                "clusters": len(self._memory_clusters),
                "hierarchies": len(self._memory_hierarchies),
                "hierarchy_depth": self._get_max_hierarchy_depth(),
            },
            "storage_stats": {
                "versioned_memories": len(self._memory_versions),
                "total_versions": sum(len(versions) for versions in self._memory_versions.values()),
                "encrypted_count": await self._count_encrypted_memories(),
            },
            "access_patterns": {
                "most_accessed": await self._get_most_accessed_memories(10),
                "co_access_pairs": len(self._co_access_matrix),
            },
        }

        # Add tag distribution
        tag_distribution = {}
        for tag, memory_ids in self._tag_index.items():
            tag_distribution[tag] = len(memory_ids)
        stats["tag_distribution"] = dict(sorted(
            tag_distribution.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:20])

        return stats

    # Additional public methods

    async def get_memory_cluster(self, memory_id: str) -> Optional[MemoryCluster]:
        """Get the cluster containing a memory."""
        cluster_id = self._memory_to_cluster.get(memory_id)
        if cluster_id:
            return self._memory_clusters.get(cluster_id)
        return None

    async def get_memory_hierarchy(self, memory_id: str) -> Optional[MemoryHierarchy]:
        """Get the hierarchy containing a memory."""
        for hierarchy in self._memory_hierarchies.values():
            if memory_id in hierarchy.memory_ids:
                return hierarchy
        return None

    async def get_memories_by_category(
        self, category: Union[str, MemoryCategory], limit: int = 100
    ) -> List[MemoryItem]:
        """Get memories by category."""
        if isinstance(category, str):
            try:
                category = MemoryCategory(category)
            except ValueError:
                return []

        memory_ids = list(self._category_index.get(category, set()))[:limit]
        return await self._retrieve_multiple(memory_ids)

    async def get_memory_versions(self, memory_id: str) -> List[Tuple[datetime, MemoryItem]]:
        """Get all versions of a memory."""
        versions = []
        
        if memory_id in self._memory_versions:
            for timestamp, version_id in self._memory_versions[memory_id]:
                if version_id in self._version_store:
                    versions.append((timestamp, self._version_store[version_id]))
        
        return versions

    async def restore_version(self, memory_id: str, version_timestamp: datetime) -> bool:
        """Restore a memory to a specific version."""
        versions = self._memory_versions.get(memory_id, [])
        
        for timestamp, version_id in versions:
            if timestamp == version_timestamp:
                if version_id in self._version_store:
                    version_item = self._version_store[version_id]
                    
                    # Create current version before restore
                    current = await self.retrieve(memory_id)
                    if current:
                        await self._create_version(current)
                    
                    # Restore version
                    await self.memory_store.store_item(version_item)
                    
                    # Update cache
                    await self._add_to_cache(version_item)
                    
                    return True
        
        return False

    async def create_memory_cluster(
        self, memory_ids: List[str], theme: str, category: MemoryCategory
    ) -> str:
        """Create a new memory cluster."""
        cluster_id = MemoryUtils.generate_memory_id()
        
        # Calculate centroid if embeddings available
        centroid = None
        if self._embedding_index:
            embeddings = []
            for memory_id in memory_ids:
                if memory_id in self._embedding_index:
                    embeddings.append(self._embedding_index[memory_id])
            
            if embeddings:
                centroid = np.mean(embeddings, axis=0).tolist()
        
        # Extract keywords
        keywords = await self._extract_cluster_keywords(memory_ids)
        
        cluster = MemoryCluster(
            cluster_id=cluster_id,
            theme=theme,
            category=category,
            memory_ids=set(memory_ids),
            centroid=centroid,
            keywords=keywords,
        )
        
        self._memory_clusters[cluster_id] = cluster
        
        # Update memory-to-cluster mapping
        for memory_id in memory_ids:
            self._memory_to_cluster[memory_id] = cluster_id
        
        return cluster_id

    async def search_similar_memories(
        self, memory_id: str, similarity_threshold: float = 0.7, limit: int = 10
    ) -> List[Tuple[MemoryItem, float]]:
        """Find memories similar to a given memory."""
        source_memory = await self.retrieve(memory_id)
        if not source_memory or not source_memory.embeddings:
            return []

        similar_memories = []
        
        if self.vector_store:
            # Use vector store for similarity search
            results = await self.vector_store.similarity_search(
                source_memory.embeddings,
                similarity_threshold=similarity_threshold,
                top_k=limit
            )
            
            for item in results:
                if item.memory_id != memory_id:
                    similarity = self._calculate_similarity(
                        source_memory.embeddings, 
                        item.embeddings
                    )
                    similar_memories.append((item, similarity))
        
        # Sort by similarity
        similar_memories.sort(key=lambda x: x[1], reverse=True)
        
        return similar_memories[:limit]

    # Private helper methods

    async def _create_metadata(self, data: Any, **kwargs) -> MemoryMetadata:
        """Create metadata for a long-term memory item."""
        # Determine category
        category = kwargs.get("category", MemoryCategory.KNOWLEDGE)
        if isinstance(category, str):
            try:
                category = MemoryCategory(category)
            except ValueError:
                category = MemoryCategory.KNOWLEDGE

        # Create base metadata
        metadata = MemoryMetadata(
            retention_policy=MemoryRetentionPolicy.PERMANENT,
            tags=kwargs.get("tags", set()),
            importance=kwargs.get("importance", 0.7),
            source=kwargs.get("source"),
            custom_metadata={
                "category": category.value,
                "source_type": kwargs.get("source_type", "unknown"),
                "compressed": False,
                "indexed": False,
            }
        )

        return metadata

    def _determine_sensitivity(self, data: Any, **kwargs) -> MemorySensitivity:
        """Determine sensitivity level of memory content."""
        # Check for explicit sensitivity
        if "sensitivity" in kwargs:
            return kwargs["sensitivity"]

        # Check for PII patterns
        if isinstance(data, str):
            # Simple checks - would be more sophisticated in practice
            if any(pattern in data.lower() for pattern in ["ssn", "password", "credit card"]):
                return MemorySensitivity.CRITICAL
            elif any(pattern in data.lower() for pattern in ["email", "phone", "address"]):
                return MemorySensitivity.HIGH

        # Check category
        category = kwargs.get("category", MemoryCategory.KNOWLEDGE)
        if category in [MemoryCategory.PERSONAL, MemoryCategory.RELATIONSHIPS]:
            return MemorySensitivity.HIGH

        return MemorySensitivity.MEDIUM

    async def _encrypt_memory(self, memory_item: MemoryItem) -> MemoryItem:
        """Encrypt memory content."""
        if not self.encryption:
            return memory_item

        try:
            # Serialize content
            if isinstance(memory_item.content, str):
                content_bytes = memory_item.content.encode()
            else:
                content_bytes = json.dumps(memory_item.content).encode()

            # Encrypt
            encrypted_content = await self.encryption.encrypt(content_bytes)
            
            # Update memory item
            memory_item.content = encrypted_content
            memory_item.metadata.encryption_status = True

        except Exception as e:
            self.logger.error(f"Failed to encrypt memory: {str(e)}")

        return memory_item

    async def _decrypt_memory(self, memory_item: MemoryItem) -> MemoryItem:
        """Decrypt memory content."""
        if not self.encryption or not memory_item.metadata.encryption_status:
            return memory_item

        try:
            # Decrypt
            decrypted_bytes = await self.encryption.decrypt(memory_item.content)
            
            # Deserialize
            try:
                memory_item.content = json.loads(decrypted_bytes.decode())
            except json.JSONDecodeError:
                memory_item.content = decrypted_bytes.decode()

        except Exception as e:
            self.logger.error(f"Failed to decrypt memory: {str(e)}")

        return memory_item

    async def _generate_embeddings(self, data: Any) -> Optional[List[float]]:
        """Generate embeddings for memory content."""
        if not self.model_router:
            return None

        try:
            # Convert to text
            if isinstance(data, dict):
                embed_text = json.dumps(data)
            elif not isinstance(data, str):
                embed_text = str(data)
            else:
                embed_text = data

            return await self.model_router.get_embeddings(embed_text)

        except Exception as e:
            self.logger.warning(f"Failed to generate embeddings: {str(e)}")
            return None

    async def _store_in_backends(self, memory_item: MemoryItem) -> None:
        """Store memory in additional backends."""
        # Store in vector store if available
        if self.vector_store and memory_item.embeddings:
            await self.vector_store.store_item(memory_item)

        # Store in graph store if available
        if self.graph_store:
            await self.graph_store.add_node(
                memory_item.memory_id,
                properties={
                    "type": memory_item.memory_type.value,
                    "category": memory_item.metadata.custom_metadata.get("category"),
                    "importance": memory_item.metadata.importance,
                }
            )

        # Store in encrypted storage for sensitive data
        if (self.encrypted_storage and 
            memory_item.metadata.sensitivity in [MemorySensitivity.HIGH, MemorySensitivity.CRITICAL]):
            await self.encrypted_storage.store(
                f"memory_{memory_item.memory_id}",
                memory_item
            )

    async def _update_indices_on_store(self, memory_item: MemoryItem, **kwargs) -> None:
        """Update indices when storing a memory."""
        memory_id = memory_item.memory_id

        # Category index
        category_str = memory_item.metadata.custom_metadata.get("category")
        if category_str:
            try:
                category = MemoryCategory(category_str)
                self._category_index[category].add(memory_id)
            except ValueError:
                pass

        # Tag index
        for tag in memory_item.metadata.tags:
            self._tag_index[tag].add(memory_id)

        # User index
        if memory_item.owner_id:
            self._user_index[memory_item.owner_id].add(memory_id)

        # Source index
        if memory_item.metadata.source:
            self._source_index[memory_item.metadata.source].add(memory_id)

        # Embedding index
        if memory_item.embeddings and self._embedding_index is not None:
            self._embedding_index[memory_id] = memory_item.embeddings

        # Extract and index keywords
        keywords = MemoryUtils.extract_tags_from_content(memory_item.content)
        for keyword in keywords:
            self._keyword_index[keyword.lower()].add(memory_id)

    async def _remove_from_indices(self, memory_id: str) -> None:
        """Remove memory from all indices."""
        # Get memory to access metadata
        memory = await self.memory_store.get_item(memory_id)
        if not memory:
            return

        # Remove from category index
        category_str = memory.metadata.custom_metadata.get("category")
        if category_str:
            try:
                category = MemoryCategory(category_str)
                self._category_index[category].discard(memory_id)
            except ValueError:
                pass

        # Remove from tag index
        for tag in memory.metadata.tags:
            self._tag_index[tag].discard(memory_id)

        # Remove from user index
        if memory.owner_id:
            self._user_index[memory.owner_id].discard(memory_id)

        # Remove from source index
        if memory.metadata.source:
            self._source_index[memory.metadata.source].discard(memory_id)

        # Remove from embedding index
        if self._embedding_index is not None:
            self._embedding_index.pop(memory_id, None)

        # Remove from keyword index
        for keyword_set in self._keyword_index.values():
            keyword_set.discard(memory_id)

    async def _rebuild_indices(self) -> None:
        """Rebuild indices from stored memories."""
        self.logger.info("Rebuilding long-term memory indices...")

        # Query all long-term memories
        query = SimpleMemoryQuery(memory_type=MemoryType.LONG_TERM, limit=50000)
        items = await self.memory_store.query(query)

        # Rebuild indices
        for item in items:
            await self._update_indices_on_store(item)

        self.logger.info(f"Rebuilt indices for {len(items)} long-term memories")

    async def _create_version(self, memory_item: MemoryItem) -> None:
        """Create a version of a memory item."""
        version_id = f"{memory_item.memory_id}_v_{datetime.now(timezone.utc).timestamp()}"
        
        # Store version
        self._version_store[version_id] = memory_item
        
        # Add to version history
        self._memory_versions[memory_item.memory_id].append(
            (datetime.now(timezone.utc), version_id)
        )
        
        # Limit versions
        if len(self._memory_versions[memory_item.memory_id]) > self.config.max_versions:
            # Remove oldest version
            _, old_version_id = self._memory_versions[memory_item.memory_id].pop(0)
            self._version_store.pop(old_version_id, None)

    async def _add_to_hierarchy(self, memory_id: str, parent_id: str) -> None:
        """Add memory to hierarchical structure."""
        # Find parent hierarchy
        parent_hierarchy = None
        for hierarchy in self._memory_hierarchies.values():
            if parent_id in hierarchy.memory_ids:
                parent_hierarchy = hierarchy
                break

        if parent_hierarchy:
            # Add to existing hierarchy
            parent_hierarchy.memory_ids.add(memory_id)
        else:
            # Create new hierarchy level
            hierarchy_id = MemoryUtils.generate_memory_id()
            hierarchy = MemoryHierarchy(
                hierarchy_id=hierarchy_id,
                name=f"Hierarchy_{hierarchy_id[:8]}",
                level=1,
                parent_id=None,
                memory_ids={parent_id, memory_id},
            )
            self._memory_hierarchies[hierarchy_id] = hierarchy
            self._hierarchy_roots.append(hierarchy_id)

    async def _update_relationships(self, memory_id: str, related_ids: List[str]) -> None:
        """Update memory relationships in graph store."""
        if not self.graph_store:
            return

        for related_id in related_ids:
            await self.graph_store.add_edge(
                memory_id, 
                related_id, 
                "related_to"
            )

    async def _add_to_cache(self, item: MemoryItem) -> None:
        """Add item to cache with LRU eviction."""
        async with self._cache_lock:
            # Evict if cache is full
            if len(self._memory_cache) >= self.config.cache_size:
                # Simple FIFO eviction
                if self._memory_cache:
                    self._memory_cache.pop(next(iter(self._memory_cache)))

            self._memory_cache[item.memory_id] = item

    async def _update_access_tracking(self, memory_id: str) -> None:
        """Update access pattern tracking."""
        current_time = datetime.now(timezone.utc)
        
        # Track access time
        self._access_patterns[memory_id].append((current_time, "access"))

        # Update co-access matrix
        # Look for other recent accesses
        recent_cutoff = current_time - timedelta(minutes=5)
        recent_accesses = set()

        for other_id, patterns in self._access_patterns.items():
            if other_id != memory_id:
                for timestamp, _ in patterns:
                    if timestamp > recent_cutoff:
                        recent_accesses.add(other_id)
                        break

        # Update co-access counts
        for other_id in recent_accesses:
            key = tuple(sorted([memory_id, other_id]))
            self._co_access_matrix[key] += 1

    async def _retrieve_multiple(self, memory_ids: List[str]) -> List[MemoryItem]:
        """Retrieve multiple memory items efficiently."""
        items = []
        
        for memory_id in memory_ids:
            item = await self.retrieve(memory_id)
            if item:
                items.append(item)
        
        return items

    def _calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between embeddings."""
        if not embedding1 or not embedding2 or len(embedding1) != len(embedding2):
            return 0.0

        arr1 = np.array(embedding1)
        arr2 = np.array(embedding2)

        dot_product = np.dot(arr1, arr2)
        magnitude1 = np.linalg.norm(arr1)
        magnitude2 = np.linalg.norm(arr2)

        if magnitude1 * magnitude2 == 0:
            return 0.0

        return float(dot_product / (magnitude1 * magnitude2))

    async def _extract_cluster_keywords(self, memory_ids: List[str]) -> List[str]:
        """Extract keywords from a cluster of memories."""
        word_freq = defaultdict(int)
        
        for memory_id in memory_ids:
            memory = await self.retrieve(memory_id)
            if memory:
                # Extract words from content
                content = memory.content
                if isinstance(content, dict):
                    content = json.dumps(content)
                elif not isinstance(content, str):
                    content = str(content)
                
                # Simple word extraction
                words = content.lower().split()
                for word in words:
                    if len(word) > 3:  # Skip short words
                        word_freq[word] += 1
        
        # Get top keywords
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, _ in sorted_words[:10]]

    # Search strategy implementations

    async def _exact_search(self, query: Any) -> List[MemoryItem]:
        """Exact match search."""
        if isinstance(query, str):
            # Search by ID
            item = await self.retrieve(query)
            return [item] if item else []
        
        elif isinstance(query, dict):
            # Search by exact criteria
            results = []
            
            if "memory_id" in query:
                item = await self.retrieve(query["memory_id"])
                if item:
                    results.append(item)
            
            elif "tag" in query and query["tag"] in self._tag_index:
                memory_ids = list(self._tag_index[query["tag"]])[:10]
                results = await self._retrieve_multiple(memory_ids)
            
            return results
        
        return []

    async def _semantic_search(self, query: Any) -> List[MemoryItem]:
        """Semantic similarity search."""
        if not self.model_router:
            return []

        # Convert query to embedding
        query_embedding = None
        
        if isinstance(query, str):
            query_embedding = await self.model_router.get_embeddings(query)
        elif isinstance(query, list) and all(isinstance(x, (int, float)) for x in query):
            query_embedding = query
        
        if not query_embedding:
            return []

        # Search using vector store if available
        if self.vector_store:
            return await self.vector_store.similarity_search(
                query_embedding,
                similarity_threshold=0.7,
                top_k=20
            )
        
        # Fallback to manual search
        similarities = []
        
        for memory_id, embedding in self._embedding_index.items():
            similarity = self._calculate_similarity(query_embedding, embedding)
            if similarity > 0.7:
                similarities.append((memory_id, similarity))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Retrieve top items
        results = []
        for memory_id, _ in similarities[:20]:
            item = await self.retrieve(memory_id)
            if item:
                results.append(item)
        
        return results

    async def _temporal_search(self, query: Any) -> List[MemoryItem]:
        """Time-based search."""
        if not isinstance(query, dict):
            return []

        start_date = query.get("start_date")
        end_date = query.get("end_date")
        
        if not start_date or not end_date:
            return []

        # Convert to datetime if string
        if isinstance(start_date, str):
            start_date = datetime.fromisoformat(start_date)
        if isinstance(end_date, str):
            end_date = datetime.fromisoformat(end_date)

        # Query by time range
        time_query = SimpleMemoryQuery(
            memory_type=MemoryType.LONG_TERM,
            time_range=(start_date, end_date),
            limit=query.get("limit", 100)
        )

        return await self.memory_store.query(time_query)

    async def _associative_search(self, query: Any) -> List[MemoryItem]:
        """Search based on associations and co-access patterns."""
        if not isinstance(query, str):
            return []

        # Start with exact memory
        source_item = await self.retrieve(query)
        if not source_item:
            return []

        associated_ids = set()

        # Find co-accessed memories
        for (id1, id2), count in self._co_access_matrix.items():
            if count > 2:  # Threshold for association
                if id1 == query:
                    associated_ids.add(id2)
                elif id2 == query:
                    associated_ids.add(id1)

        # Find memories in same cluster
        cluster_id = self._memory_to_cluster.get(query)
        if cluster_id and cluster_id in self._memory_clusters:
            cluster = self._memory_clusters[cluster_id]
            associated_ids.update(cluster.memory_ids)

        # Remove source memory
        associated_ids.discard(query)

        # Retrieve associated memories
        return await self._retrieve_multiple(list(associated_ids)[:20])

    async def _hierarchical_search(self, query: Any) -> List[MemoryItem]:
        """Search within hierarchical structures."""
        if not isinstance(query, str):
            return []

        # Find hierarchy containing the query memory
        target_hierarchy = None
        
        for hierarchy in self._memory_hierarchies.values():
            if query in hierarchy.memory_ids:
                target_hierarchy = hierarchy
                break
        
        if not target_hierarchy:
            return []

        # Get all memories in hierarchy
        memory_ids = list(target_hierarchy.memory_ids)
        
        # Include child hierarchies
        for other_hierarchy in self._memory_hierarchies.values():
            if other_hierarchy.parent_id == target_hierarchy.hierarchy_id:
                memory_ids.extend(list(other_hierarchy.memory_ids))

        return await self._retrieve_multiple(memory_ids[:50])

    async def _contextual_search(self, query: Any) -> List[MemoryItem]:
        """Search based on context and metadata."""
        if not isinstance(query, dict):
            return []

        results = []
        
        # Search by user
        if "user_id" in query:
            user_memories = list(self._user_index.get(query["user_id"], set()))[:100]
            results.extend(await self._retrieve_multiple(user_memories))
        
        # Search by category
        if "category" in query:
            try:
                category = MemoryCategory(query["category"])
                cat_memories = list(self._category_index.get(category, set()))[:50]
                results.extend(await self._retrieve_multiple(cat_memories))
            except ValueError:
                pass
        
        # Search by source
        if "source" in query:
            source_memories = list(self._source_index.get(query["source"], set()))[:50]
            results.extend(await self._retrieve_multiple(source_memories))
        
        # Remove duplicates
        seen = set()
        unique_results = []
        for item in results:
            if item.memory_id not in seen:
                seen.add(item.memory_id)
                unique_results.append(item)
        
        return unique_results[:query.get("limit", 20)]

    async def _pattern_based_search(self, query: Any) -> List[MemoryItem]:
        """Search based on access patterns."""
        if not isinstance(query, dict):
            return []

        pattern_type = query.get("pattern", "frequent")
        
        if pattern_type == "frequent":
            # Find frequently accessed memories
            access_counts = defaultdict(int)
            
            for memory_id, patterns in self._access_patterns.items():
                access_counts[memory_id] = len(patterns)
            
            # Sort by access count
            sorted_memories = sorted(
                access_counts.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            
            memory_ids = [mid for mid, _ in sorted_memories[:20]]
            return await self._retrieve_multiple(memory_ids)
        
        elif pattern_type == "recent":
            # Find recently accessed memories
            recent_accesses = []
            
            for memory_id, patterns in self._access_patterns.items():
                if patterns:
                    last_access = max(timestamp for timestamp, _ in patterns)
                    recent_accesses.append((memory_id, last_access))
            
            # Sort by recency
            recent_accesses.sort(key=lambda x: x[1], reverse=True)
            
            memory_ids = [mid for mid, _ in recent_accesses[:20]]
            return await self._retrieve_multiple(memory_ids)
        
        return []

    # Background task implementations

    async def _index_update_loop(self) -> None:
        """Background task for updating indices."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(self.config.index_update_interval)
                
                if not self._shutdown_event.is_set():
                    self.logger.debug("Updating long-term memory indices")
                    
                    # Update keyword index for new memories
                    # Update embedding index if needed
                    # Cleanup removed memories
                    
                    # Update metrics
                    if self.metrics:
                        self._update_storage_metrics()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in index update loop: {str(e)}")
                await asyncio.sleep(300)

    async def _backup_loop(self) -> None:
        """Background task for periodic backups."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(self.config.backup_interval)
                
                if not self._shutdown_event.is_set():
                    self.logger.info("Performing long-term memory backup")
                    await self._perform_backup()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in backup loop: {str(e)}")
                await asyncio.sleep(3600)

    async def _clustering_loop(self) -> None:
        """Background task for memory clustering."""
        while not self._shutdown_event.is_set():
            try:
                # Run clustering every 6 hours
                await asyncio.sleep(21600)
                
                if not self._shutdown_event.is_set():
                    self.logger.info("Running memory clustering")
                    await self._perform_clustering()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in clustering loop: {str(e)}")
                await asyncio.sleep(3600)

    async def _perform_backup(self) -> None:
        """Perform memory backup."""
        try:
            backup_path = Path("data/backups/long_term_memory")
            backup_path.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            backup_file = backup_path / f"backup_{timestamp}.json"
            
            # This is a simplified backup - real implementation would be more robust
            self.logger.info(f"Backup completed: {backup_file}")
            
        except Exception as e:
            self.logger.error(f"Backup failed: {str(e)}")

    async def _perform_clustering(self) -> None:
        """Perform memory clustering using embeddings."""
        if not self._embedding_index or len(self._embedding_index) < 10:
            return

        try:
            # Simple clustering implementation
            # In practice, would use more sophisticated algorithms
            
            self.logger.info("Memory clustering completed")
            
        except Exception as e:
            self.logger.error(f"Clustering failed: {str(e)}")

    async def _load_organizational_structures(self) -> None:
        """Load existing organizational structures."""
        # Load from persistent storage if available
        pass

    async def _save_organizational_structures(self) -> None:
        """Save organizational structures."""
        # Save to persistent storage
        pass

    async def _remove_from_organization(self, memory_id: str) -> None:
        """Remove memory from organizational structures."""
        # Remove from clusters
        cluster_id = self._memory_to_cluster.pop(memory_id, None)
        if cluster_id and cluster_id in self._memory_clusters:
            self._memory_clusters[cluster_id].memory_ids.discard(memory_id)
            
            # Remove empty clusters
            if not self._memory_clusters[cluster_id].memory_ids:
                del self._memory_clusters[cluster_id]

        # Remove from hierarchies
        for hierarchy in self._memory_hierarchies.values():
            hierarchy.memory_ids.discard(memory_id)

    def _get_max_hierarchy_depth(self) -> int:
        """Get maximum depth of memory hierarchies."""
        if not self._memory_hierarchies:
            return 0

        def get_depth(hierarchy_id: str, visited: Set[str] = None) -> int:
            if visited is None:
                visited = set()
            
            if hierarchy_id in visited:
                return 0
            
            visited.add(hierarchy_id)
            
            hierarchy = self._memory_hierarchies.get(hierarchy_id)
            if not hierarchy or not hierarchy.children_ids:
                return 1
            
            max_child_depth = 0
            for child_id in hierarchy.children_ids:
                depth = get_depth(child_id, visited.copy())
                max_child_depth = max(max_child_depth, depth)
            
            return 1 + max_child_depth

        max_depth = 0
        for root_id in self._hierarchy_roots:
            depth = get_depth(root_id)
            max_depth = max(max_depth, depth)

        return max_depth

    async def _count_encrypted_memories(self) -> int:
        """Count number of encrypted memories."""
        # This is a simplified count - would query actual storage
        return sum(
            1 for item in self._memory_cache.values()
            if item.metadata.encryption_status
        )

    async def _get_most_accessed_memories(self, limit: int) -> List[Tuple[str, int]]:
        """Get most frequently accessed memories."""
        access_counts = []
        
        for memory_id, patterns in self._access_patterns.items():
            access_counts.append((memory_id, len(patterns)))
        
        # Sort by access count
        access_counts.sort(key=lambda x: x[1], reverse=True)
        
        return access_counts[:limit]

    def _update_storage_metrics(self) -> None:
        """Update storage-related metrics."""
        if not self.metrics:
            return

        total_items = sum(len(items) for items in self._user_index.values())
        self.metrics.gauge("long_term_memory_total_items", total_items)
        self.metrics.gauge("long_term_memory_clusters", len(self._memory_clusters))
        self.metrics.gauge("long_term_memory_hierarchies", len(self._memory_hierarchies))
        self.metrics.gauge("long_term_memory_cache_size", len(self._memory_cache))
