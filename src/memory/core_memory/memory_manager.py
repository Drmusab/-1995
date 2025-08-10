"""
Advanced Memory Management System
Author: Drmusab
Last Modified: 2025-07-04 20:10:32 UTC

This module provides the comprehensive memory management system for the AI assistant,
orchestrating different memory types, storage, retrieval, consolidation, and
seamless integration with all core system components.
"""

import hashlib
import json
import logging
import time
import uuid
import weakref
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
from src.core.health_check import HealthCheck
from src.core.security.encryption import EncryptionManager


# Storage integration
from src.integrations.storage.database import DatabaseManager

# Learning and optimization
from src.learning.continual_learning import ContinualLearner
from src.learning.feedback_processor import FeedbackProcessor

# Memory foundation
from src.memory.core_memory.base_memory import (
    AbstractMemoryManager,
    BaseMemory,
    BaseMemoryStore,
    MemoryAccess,
    MemoryAccessController,
    MemoryAccessError,
    MemoryCache,
    MemoryCorruptionError,
    MemoryError,
    MemoryIndexer,
    MemoryItem,
    MemoryMetadata,
    MemoryNotFoundError,
    MemoryOperationError,
    MemoryQuery,
    MemoryRetentionPolicy,
    MemorySearchResult,
    MemorySensitivity,
    MemoryStorageError,
    MemoryStorageType,
    MemoryType,
    MemoryUtils,
    SemanticMemoryQuery,
    SimpleMemoryQuery,
    VectorMemoryQuery,
    memory_operation_span,
    register_memory_metrics,
)

# Memory types
from src.memory.core_memory.memory_types import (
    EpisodicMemory,
    LongTermMemory,
    ProceduralMemory,
    SemanticMemory,
    ShortTermMemory,
    WorkingMemory,
)
from src.memory.operations.consolidation import MemoryConsolidation
from src.memory.operations.context_manager import ContextManager

# Memory operations
from src.memory.operations.retrieval import EnhancedRetrieval
from src.memory.storage.memory_graph import MemoryGraph
from src.memory.storage.relational_store import RelationalMemoryStore

# Storage backends
from src.memory.storage.vector_store import VectorStore
from src.observability.logging.config import get_logger

# Observability
from src.observability.monitoring.metrics import MetricsCollector
from src.observability.monitoring.tracing import TraceManager

# Type definitions
T = TypeVar("T")


class MemoryManagerConfig:
    """Configuration for the memory manager."""

    def __init__(self, config_loader: ConfigLoader):
        # Memory capacity limits
        self.working_memory_capacity = config_loader.get("memory.working_memory_capacity", 1000)
        self.short_term_memory_capacity = config_loader.get(
            "memory.short_term_memory_capacity", 10000
        )
        self.long_term_memory_capacity = config_loader.get(
            "memory.long_term_memory_capacity", 1000000
        )

        # Memory retention times (in seconds)
        self.working_memory_retention = config_loader.get(
            "memory.working_memory_retention", 3600
        )  # 1 hour
        self.short_term_retention = config_loader.get("memory.short_term_retention", 86400)  # 1 day
        self.long_term_retention = config_loader.get(
            "memory.long_term_retention", 31536000
        )  # 1 year

        # Consolidation settings
        self.consolidation_interval = config_loader.get(
            "memory.consolidation_interval", 1800
        )  # 30 minutes
        self.consolidation_batch_size = config_loader.get("memory.consolidation_batch_size", 100)
        self.importance_threshold = config_loader.get("memory.importance_threshold", 0.3)

        # Caching settings
        self.cache_enabled = config_loader.get("memory.cache_enabled", True)
        self.cache_size = config_loader.get("memory.cache_size", 10000)
        self.cache_ttl = config_loader.get("memory.cache_ttl", 300)  # 5 minutes

        # Retrieval settings
        self.default_retrieval_limit = config_loader.get("memory.default_retrieval_limit", 50)
        self.similarity_threshold = config_loader.get("memory.similarity_threshold", 0.7)

        # Security settings
        self.encryption_enabled = config_loader.get("memory.encryption_enabled", True)
        self.security_validation = config_loader.get("memory.security_validation", True)

        # Backup settings
        self.backup_enabled = config_loader.get("memory.backup_enabled", True)
        self.backup_interval = config_loader.get("memory.backup_interval", 86400)  # 1 day
        self.backup_retention = config_loader.get("memory.backup_retention", 10)  # Keep 10 backups

        # Performance settings
        self.parallelism_enabled = config_loader.get("memory.parallelism_enabled", True)
        self.max_concurrent_operations = config_loader.get("memory.max_concurrent_operations", 10)

        # Advanced features
        self.adaptive_retention = config_loader.get("memory.adaptive_retention", True)
        self.forgetting_enabled = config_loader.get("memory.forgetting_enabled", True)
        self.associative_recall = config_loader.get("memory.associative_recall", True)


class MemoryManager(AbstractMemoryManager):
    """
    Advanced Memory Management System for the AI Assistant.

    This system orchestrates different memory types, providing:
    - Comprehensive memory lifecycle management
    - Multi-tier memory architecture (working, short-term, long-term)
    - Intelligent memory consolidation and forgetting
    - Context-aware retrieval and storage
    - Semantic and associative recall
    - Security and access control
    - Performance optimization via caching and parallelism
    - Integration with all core system components
    - Memory backup and restoration
    - Observability and monitoring
    """

    def __init__(self, container: Container):
        """
        Initialize the memory manager.

        Args:
            container: Dependency injection container
        """
        self.container = container
        self.logger = get_logger(__name__)

        # Core services
        self.config_loader = container.get(ConfigLoader)
        self.event_bus = container.get(EventBus)
        self.error_handler = container.get(ErrorHandler)
        self.health_check = container.get(HealthCheck)

        # Configuration
        self.config = MemoryManagerConfig(self.config_loader)

        # Memory stores and types
        self._setup_memory_stores()
        self._setup_memory_types()

        # Memory operations
        self.retrieval = container.get(EnhancedRetrieval)
        self.consolidation = container.get(MemoryConsolidation)
        self.context_manager = container.get(ContextManager)

        # Security
        try:
            self.encryption_manager = container.get(EncryptionManager)
            self.access_controller = MemoryAccessController(self.encryption_manager)
        except Exception:
            self.logger.warning("Encryption manager not available, running without encryption")
            self.encryption_manager = None
            self.access_controller = MemoryAccessController()

        # Storage integration
        try:
            self.database = container.get(DatabaseManager)
        except Exception:
            self.logger.warning("Database manager not available")
            self.database = None

        # Learning integration
        try:
            self.continual_learner = container.get(ContinualLearner)
            self.feedback_processor = container.get(FeedbackProcessor)
        except Exception:
            self.logger.warning("Learning components not available")
            self.continual_learner = None
            self.feedback_processor = None

        # Observability
        self._setup_monitoring()

        # State management
        self.memory_semaphore = asyncio.Semaphore(self.config.max_concurrent_operations)
        self._shutdown_event = asyncio.Event()

        # Setup caching
        if self.config.cache_enabled:
            self.memory_cache = MemoryCache(
                max_size=self.config.cache_size, ttl_seconds=self.config.cache_ttl
            )
        else:
            self.memory_cache = None

        # Performance tracking
        self.operation_stats = defaultdict(
            lambda: {
                "count": 0,
                "success_count": 0,
                "error_count": 0,
                "total_time": 0.0,
                "avg_time": 0.0,
            }
        )

        # Register health check
        self.health_check.register_component("memory_manager", self._health_check_callback)

        self.logger.info("MemoryManager initialized successfully")

    def _setup_memory_stores(self) -> None:
        """Setup memory storage backends."""
        try:
            # Vector store for semantic retrieval
            self.vector_store = self.container.get(VectorStore)

            # Graph store for relational memory
            self.memory_graph = self.container.get(MemoryGraph)

            # Relational store for structured memory
            self.relational_store = self.container.get(RelationalMemoryStore)

            # Map memory types to appropriate stores
            self.memory_stores = {
                MemoryType.WORKING: self.relational_store,
                MemoryType.SHORT_TERM: self.relational_store,
                MemoryType.EPISODIC: self.vector_store,
                MemoryType.SEMANTIC: self.vector_store,
                MemoryType.PROCEDURAL: self.relational_store,
                MemoryType.SENSORY: self.vector_store,
                MemoryType.LONG_TERM: self.memory_graph,
                MemoryType.META: self.relational_store,
            }

            self.logger.info("Memory stores initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize memory stores: {str(e)}")
            # Fallback to default stores
            self.memory_stores = {}

    def _setup_memory_types(self) -> None:
        """Setup specialized memory type handlers."""
        try:
            # Specialized memory handlers
            self.working_memory = self.container.get(WorkingMemory)
            self.episodic_memory = self.container.get(EpisodicMemory)
            self.semantic_memory = self.container.get(SemanticMemory)
            self.procedural_memory = self.container.get(ProceduralMemory)
            # self.sensory_memory = self.container.get(SensoryMemory)  # SensoryMemory class not implemented
            self.short_term_memory = self.container.get(ShortTermMemory)
            self.long_term_memory = self.container.get(LongTermMemory)

            # Map memory types to handlers
            self.memory_handlers = {
                MemoryType.WORKING: self.working_memory,
                MemoryType.EPISODIC: self.episodic_memory,
                MemoryType.SEMANTIC: self.semantic_memory,
                MemoryType.PROCEDURAL: self.procedural_memory,
                # MemoryType.SENSORY: self.sensory_memory,  # SensoryMemory class not implemented
                MemoryType.SHORT_TERM: self.short_term_memory,
                MemoryType.LONG_TERM: self.long_term_memory,
            }

            self.logger.info("Memory type handlers initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize memory type handlers: {str(e)}")
            # Fallback to empty handlers
            self.memory_handlers = {}

    def _setup_monitoring(self) -> None:
        """Setup monitoring and metrics collection."""
        try:
            self.metrics = self.container.get(MetricsCollector)
            self.tracer = self.container.get(TraceManager)

            # Register memory-specific metrics
            register_memory_metrics(self.metrics)

        except Exception as e:
            self.logger.warning(f"Failed to setup monitoring: {str(e)}")
            self.metrics = None
            self.tracer = None

    async def initialize(self) -> None:
        """Initialize the memory manager and its components."""
        try:
            self.logger.info("Initializing memory manager...")

            # Initialize memory stores
            init_tasks = []
            for store in set(self.memory_stores.values()):
                if hasattr(store, "initialize"):
                    init_tasks.append(store.initialize())

            if init_tasks:
                await asyncio.gather(*init_tasks)

            # Initialize specialized memory handlers
            for handler in set(self.memory_handlers.values()):
                if hasattr(handler, "initialize"):
                    await handler.initialize()

            # Initialize memory operations
            for component in [self.retrieval, self.consolidation, self.context_manager]:
                if hasattr(component, "initialize"):
                    await component.initialize()

            # Start background tasks
            asyncio.create_task(self._consolidation_loop())
            asyncio.create_task(self._cleanup_loop())
            asyncio.create_task(self._backup_loop())

            # Register event handlers
            await self._register_event_handlers()

            self.logger.info("Memory manager initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize memory manager: {str(e)}")
            raise MemoryError(f"Memory manager initialization failed: {str(e)}")

    async def _register_event_handlers(self) -> None:
        """Register event handlers for system events."""
        # Register for system shutdown
        self.event_bus.subscribe("system_shutdown_started", self._handle_system_shutdown)

        # Register for session events
        self.event_bus.subscribe("session_ended", self._handle_session_ended)

        # Register for learning events
        self.event_bus.subscribe("learning_event_occurred", self._handle_learning_event)

    @handle_exceptions
    async def store_memory(
        self,
        data: Any,
        memory_type: MemoryType,
        owner_id: Optional[str] = None,
        session_id: Optional[str] = None,
        context_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[Set[str]] = None,
        **kwargs,
    ) -> str:
        """
        Store data in the appropriate memory system.

        Args:
            data: Data to store in memory
            memory_type: Type of memory to store in
            owner_id: Owner of the memory
            session_id: Associated session ID
            context_id: Associated context ID
            metadata: Additional metadata
            tags: Memory tags
            **kwargs: Additional parameters

        Returns:
            Memory ID of the stored item
        """
        start_time = time.time()
        memory_id = MemoryUtils.generate_memory_id()

        try:
            async with memory_operation_span(self.tracer, "store", memory_id):
                # Create memory metadata
                mem_metadata = MemoryMetadata()

                # Apply custom metadata if provided
                if metadata:
                    for key, value in metadata.items():
                        if hasattr(mem_metadata, key):
                            setattr(mem_metadata, key, value)
                        else:
                            mem_metadata.custom_metadata[key] = value

                # Add tags
                if tags:
                    mem_metadata.tags.update(tags)

                # Extract additional tags from content
                content_tags = MemoryUtils.extract_tags_from_content(data)
                mem_metadata.tags.update(content_tags)

                # Set expiration based on retention policy
                mem_metadata.expiration = MemoryUtils.calculate_expiration(
                    mem_metadata.retention_policy
                )

                # Calculate checksum
                mem_metadata.checksum = mem_metadata.calculate_checksum(data)

                # Create memory item
                memory_item = MemoryItem(
                    memory_id=memory_id,
                    content=data,
                    memory_type=memory_type,
                    owner_id=owner_id,
                    session_id=session_id,
                    context_id=context_id,
                    metadata=mem_metadata,
                )

                # Apply encryption if needed
                if self.config.encryption_enabled and self.encryption_manager:
                    memory_item = await self.access_controller.encrypt_memory_content(memory_item)

                # Store in appropriate memory system
                await self._store_in_appropriate_system(memory_item)

                # Cache the item
                if self.memory_cache:
                    await self.memory_cache.set(memory_id, memory_item)

                # Emit event
                await self.event_bus.emit(
                    MemoryItemStored(
                        memory_id=memory_id,
                        memory_type=memory_type.value,
                        owner_id=owner_id,
                        session_id=session_id,
                    )
                )

                # Update metrics
                if self.metrics:
                    self.metrics.increment("memory_operations_total")
                    self.metrics.increment("memory_store_operations")
                    self.metrics.record(
                        "memory_operation_duration_seconds", time.time() - start_time
                    )
                    self.metrics.record("memory_item_size_bytes", len(str(data)))

                # Update stats
                self._update_operation_stats("store", start_time, success=True)

                self.logger.debug(f"Stored memory {memory_id} of type {memory_type.value}")
                return memory_id

        except Exception as e:
            self._update_operation_stats("store", start_time, success=False)
            self.logger.error(f"Failed to store memory: {str(e)}")
            raise MemoryOperationError(f"Failed to store memory: {str(e)}")

    async def _store_in_appropriate_system(self, memory_item: MemoryItem) -> None:
        """Store memory item in the appropriate memory system."""
        # Check if we have a specialized handler for this memory type
        if memory_item.memory_type in self.memory_handlers:
            handler = self.memory_handlers[memory_item.memory_type]
            if hasattr(handler, "store"):
                await handler.store(memory_item)
                return

        # Fall back to generic storage
        if memory_item.memory_type in self.memory_stores:
            store = self.memory_stores[memory_item.memory_type]
            await store.store_item(memory_item)
        else:
            # Default to first available store
            if self.memory_stores:
                store = next(iter(self.memory_stores.values()))
                await store.store_item(memory_item)
            else:
                raise MemoryStorageError("No suitable memory store available")

    @handle_exceptions
    async def retrieve_memory(
        self, memory_id: str, user_id: Optional[str] = None
    ) -> Optional[MemoryItem]:
        """
        Retrieve memory by ID.

        Args:
            memory_id: Memory identifier
            user_id: User requesting the memory (for access control)

        Returns:
            Memory item or None if not found
        """
        start_time = time.time()

        try:
            async with memory_operation_span(self.tracer, "retrieve", memory_id):
                # Check cache first
                if self.memory_cache:
                    cached_item = await self.memory_cache.get(memory_id)
                    if cached_item:
                        # Update access time
                        cached_item.metadata.update_access()

                        # Emit cache hit event
                        # await self.event_bus.emit(MemoryCacheHit(memory_id=memory_id))

                        if self.metrics:
                            self.metrics.increment("memory_cache_hits")

                        # Check access control
                        if not await self.access_controller.check_access(cached_item, user_id):
                            raise MemoryAccessError(
                                f"Access denied to memory {memory_id}", memory_id, owner_id=user_id
                            )

                        # Decrypt if needed
                        if cached_item.metadata.encryption_status and self.encryption_manager:
                            cached_item = await self.access_controller.decrypt_memory_content(
                                cached_item
                            )

                        return cached_item

                # Cache miss, search in all memory stores
                memory_item = None

                # Try specialized handler first
                for handler in self.memory_handlers.values():
                    if hasattr(handler, "retrieve"):
                        try:
                            item = await handler.retrieve(memory_id)
                            if item:
                                memory_item = item
                                break
                        except Exception:
                            pass

                # Fall back to generic stores
                if not memory_item:
                    for store in set(self.memory_stores.values()):
                        try:
                            item = await store.get_item(memory_id)
                            if item:
                                memory_item = item
                                break
                        except Exception:
                            pass

                if not memory_item:
                    if self.memory_cache:
                        # await self.event_bus.emit(MemoryCacheMiss(memory_id=memory_id))
                        if self.metrics:
                            self.metrics.increment("memory_cache_misses")

                    self._update_operation_stats("retrieve", start_time, success=False)
                    return None

                # Update access time
                memory_item.metadata.update_access()

                # Check access control
                if not await self.access_controller.check_access(memory_item, user_id):
                    raise MemoryAccessError(
                        f"Access denied to memory {memory_id}", memory_id, owner_id=user_id
                    )

                # Decrypt if needed
                if memory_item.metadata.encryption_status and self.encryption_manager:
                    memory_item = await self.access_controller.decrypt_memory_content(memory_item)

                # Update cache
                if self.memory_cache:
                    await self.memory_cache.set(memory_id, memory_item)

                # Emit event
                await self.event_bus.emit(
                    MemoryItemRetrieved(
                        memory_id=memory_id,
                        memory_type=memory_item.memory_type.value,
                        owner_id=memory_item.owner_id,
                    )
                )

                # Update metrics
                if self.metrics:
                    self.metrics.increment("memory_operations_total")
                    self.metrics.increment("memory_retrieve_operations")
                    self.metrics.record(
                        "memory_operation_duration_seconds", time.time() - start_time
                    )

                # Update stats
                self._update_operation_stats("retrieve", start_time, success=True)

                return memory_item

        except MemoryAccessError:
            # Re-raise access errors
            raise
        except Exception as e:
            self._update_operation_stats("retrieve", start_time, success=False)
            self.logger.error(f"Failed to retrieve memory {memory_id}: {str(e)}")
            raise MemoryOperationError(f"Failed to retrieve memory: {str(e)}", memory_id)

    @handle_exceptions
    async def update_memory(
        self,
        memory_id: str,
        data: Any,
        user_id: Optional[str] = None,
        update_metadata: Optional[Dict[str, Any]] = None,
        add_tags: Optional[Set[str]] = None,
        remove_tags: Optional[Set[str]] = None,
    ) -> bool:
        """
        Update existing memory.

        Args:
            memory_id: Memory identifier
            data: New content data
            user_id: User updating the memory (for access control)
            update_metadata: Metadata to update
            add_tags: Tags to add
            remove_tags: Tags to remove

        Returns:
            True if successful
        """
        start_time = time.time()

        try:
            async with memory_operation_span(self.tracer, "update", memory_id):
                # Retrieve the memory first
                memory_item = await self.retrieve_memory(memory_id, user_id)
                if not memory_item:
                    raise MemoryNotFoundError(f"Memory {memory_id} not found", memory_id)

                # Update memory content
                memory_item.content = data

                # Update metadata
                memory_item.metadata.update_modification()
                memory_item.metadata.checksum = memory_item.metadata.calculate_checksum(data)

                if update_metadata:
                    for key, value in update_metadata.items():
                        if hasattr(memory_item.metadata, key):
                            setattr(memory_item.metadata, key, value)
                        else:
                            memory_item.metadata.custom_metadata[key] = value

                # Update tags
                if add_tags:
                    memory_item.metadata.tags.update(add_tags)
                if remove_tags:
                    memory_item.metadata.tags = memory_item.metadata.tags - remove_tags

                # Apply encryption if needed
                if self.config.encryption_enabled and self.encryption_manager:
                    memory_item = await self.access_controller.encrypt_memory_content(memory_item)

                # Store updated item
                await self._store_in_appropriate_system(memory_item)

                # Update cache
                if self.memory_cache:
                    await self.memory_cache.set(memory_id, memory_item)

                # Emit event
                await self.event_bus.emit(
                    MemoryItemUpdated(
                        memory_id=memory_id,
                        memory_type=memory_item.memory_type.value,
                        owner_id=memory_item.owner_id,
                    )
                )

                # Update metrics
                if self.metrics:
                    self.metrics.increment("memory_operations_total")
                    self.metrics.increment("memory_update_operations")
                    self.metrics.record(
                        "memory_operation_duration_seconds", time.time() - start_time
                    )

                # Update stats
                self._update_operation_stats("update", start_time, success=True)

                self.logger.debug(f"Updated memory {memory_id}")
                return True

        except Exception as e:
            self._update_operation_stats("update", start_time, success=False)
            self.logger.error(f"Failed to update memory {memory_id}: {str(e)}")
            raise MemoryOperationError(f"Failed to update memory: {str(e)}", memory_id)

    @handle_exceptions
    async def search_memories(
        self,
        query: Any,
        memory_type: Optional[MemoryType] = None,
        user_id: Optional[str] = None,
        max_results: int = 100,
        **kwargs,
    ) -> MemorySearchResult:
        """
        Search memories based on query.

        Args:
            query: Search query (string, vector, or structured query)
            memory_type: Optional filter by memory type
            user_id: User performing the search (for access control)
            max_results: Maximum number of results to return
            **kwargs: Additional search parameters

        Returns:
            Memory search results
        """
        start_time = time.time()

        await self.event_bus.emit(
            MemorySearchStarted(
                query_type=type(query).__name__,
                memory_type=memory_type.value if memory_type else "all",
            )
        )

        try:
            async with memory_operation_span(self.tracer, "search"):
                # Choose search strategy based on query type
                if isinstance(query, str):
                    # Text query - convert to semantic search
                    semantic_query = SemanticMemoryQuery(
                        query_text=query, memory_type=memory_type, top_k=max_results
                    )
                    results = await self._perform_semantic_search(semantic_query, user_id)

                elif isinstance(query, list) and len(query) > 0 and isinstance(query[0], float):
                    # Vector query
                    vector_query = VectorMemoryQuery(
                        query_vector=query, memory_type=memory_type, top_k=max_results
                    )
                    results = await self._perform_vector_search(vector_query, user_id)

                elif isinstance(query, dict):
                    # Dictionary-based query - convert to SimpleMemoryQuery
                    simple_query = SimpleMemoryQuery(
                        memory_type=memory_type,
                        **{k: v for k, v in query.items() if hasattr(SimpleMemoryQuery, k)},
                    )
                    results = await self._perform_structured_search(simple_query, user_id)

                elif isinstance(query, MemoryQuery):
                    # Direct query object
                    if hasattr(self.retrieval, "execute_query"):
                        results = await self.retrieval.execute_query(query, user_id=user_id)
                    else:
                        # Delegate to appropriate store
                        stores = []
                        if memory_type and memory_type in self.memory_stores:
                            stores = [self.memory_stores[memory_type]]
                        else:
                            stores = list(set(self.memory_stores.values()))

                        all_results = []
                        for store in stores:
                            try:
                                items = await query.execute(store)
                                all_results.extend(items)
                            except Exception as e:
                                self.logger.warning(f"Error searching in store {store}: {str(e)}")

                        # Filter by access control
                        filtered_results = []
                        for item in all_results:
                            if await self.access_controller.check_access(item, user_id):
                                # Decrypt if needed
                                if item.metadata.encryption_status and self.encryption_manager:
                                    item = await self.access_controller.decrypt_memory_content(item)
                                filtered_results.append(item)

                        results = MemorySearchResult(
                            items=filtered_results[:max_results],
                            total_count=len(filtered_results),
                            query_time=time.time() - start_time,
                        )
                else:
                    raise ValueError(f"Unsupported query type: {type(query)}")

                # Emit completion event
                await self.event_bus.emit(
                    # MemorySearchCompleted(
                        query_type=type(query).__name__,
                        memory_type=memory_type.value if memory_type else "all",
                        result_count=results.total_count,
                        query_time=results.query_time,
                    )
                )

                # Update metrics
                if self.metrics:
                    self.metrics.increment("memory_operations_total")
                    self.metrics.increment("memory_search_operations")
                    self.metrics.record(
                        "memory_operation_duration_seconds", time.time() - start_time
                    )

                # Update stats
                self._update_operation_stats("search", start_time, success=True)

                return results

        except Exception as e:
            self._update_operation_stats("search", start_time, success=False)
            self.logger.error(f"Failed to search memories: {str(e)}")
            raise MemoryOperationError(f"Failed to search memories: {str(e)}")

    async def _perform_semantic_search(
        self, query: SemanticMemoryQuery, user_id: Optional[str] = None
    ) -> MemorySearchResult:
        """Perform semantic search across memory stores."""
        start_time = time.time()
        all_results = []
        relevance_scores = {}

        # Use dedicated retrieval if available
        if hasattr(self.retrieval, "semantic_search"):
            results = await self.retrieval.semantic_search(
                query.query_text, memory_type=query.memory_type, user_id=user_id, top_k=query.top_k
            )
            return results

        # Otherwise search each compatible store
        stores = []
        if query.memory_type and query.memory_type in self.memory_stores:
            stores = [self.memory_stores[query.memory_type]]
        else:
            # Filter to stores that support semantic search
            stores = [
                store
                for store in set(self.memory_stores.values())
                if hasattr(store, "semantic_search")
            ]

        # Perform search on each store
        for store in stores:
            try:
                if hasattr(store, "semantic_search"):
                    items, scores = await store.semantic_search(
                        query.query_text, similarity_threshold=query.relevance_threshold
                    )

                    # Filter by access control
                    for item in items:
                        if await self.access_controller.check_access(item, user_id):
                            # Decrypt if needed
                            if item.metadata.encryption_status and self.encryption_manager:
                                item = await self.access_controller.decrypt_memory_content(item)
                            all_results.append(item)
                            if scores and item.memory_id in scores:
                                relevance_scores[item.memory_id] = scores[item.memory_id]
            except Exception as e:
                self.logger.warning(f"Error in semantic search for store {store}: {str(e)}")

        # Sort by relevance and limit results
        if relevance_scores:
            all_results.sort(
                key=lambda item: relevance_scores.get(item.memory_id, 0.0), reverse=True
            )

        return MemorySearchResult(
            items=all_results[: query.top_k],
            total_count=len(all_results),
            query_time=time.time() - start_time,
            relevance_scores=relevance_scores,
        )

    async def _perform_vector_search(
        self, query: VectorMemoryQuery, user_id: Optional[str] = None
    ) -> MemorySearchResult:
        """Perform vector similarity search across memory stores."""
        start_time = time.time()
        all_results = []
        similarity_scores = {}

        # Use dedicated retrieval if available
        if hasattr(self.retrieval, "vector_search"):
            results = await self.retrieval.vector_search(
                query.query_vector,
                memory_type=query.memory_type,
                user_id=user_id,
                top_k=query.top_k,
            )
            return results

        # Otherwise search each compatible store
        stores = []
        if query.memory_type and query.memory_type in self.memory_stores:
            stores = [self.memory_stores[query.memory_type]]
        else:
            # Filter to stores that support vector search
            stores = [
                store
                for store in set(self.memory_stores.values())
                if hasattr(store, "similarity_search")
            ]

        # Perform search on each store
        for store in stores:
            try:
                if hasattr(store, "similarity_search"):
                    items, scores = await store.similarity_search(
                        query.query_vector, similarity_threshold=query.similarity_threshold
                    )

                    # Filter by access control
                    for item in items:
                        if await self.access_controller.check_access(item, user_id):
                            # Decrypt if needed
                            if item.metadata.encryption_status and self.encryption_manager:
                                item = await self.access_controller.decrypt_memory_content(item)
                            all_results.append(item)
                            if scores and item.memory_id in scores:
                                similarity_scores[item.memory_id] = scores[item.memory_id]
            except Exception as e:
                self.logger.warning(f"Error in vector search for store {store}: {str(e)}")

        # Sort by similarity and limit results
        if similarity_scores:
            all_results.sort(
                key=lambda item: similarity_scores.get(item.memory_id, 0.0), reverse=True
            )

        return MemorySearchResult(
            items=all_results[: query.top_k],
            total_count=len(all_results),
            query_time=time.time() - start_time,
            relevance_scores=similarity_scores,
        )

    async def _perform_structured_search(
        self, query: SimpleMemoryQuery, user_id: Optional[str] = None
    ) -> MemorySearchResult:
        """Perform structured search across memory stores."""
        start_time = time.time()
        all_results = []

        # Use dedicated retrieval if available
        if hasattr(self.retrieval, "structured_search"):
            results = await self.retrieval.structured_search(query, user_id=user_id)
            return results

        # Otherwise search each store
        stores = []
        if query.memory_type and query.memory_type in self.memory_stores:
            stores = [self.memory_stores[query.memory_type]]
        else:
            stores = list(set(self.memory_stores.values()))

        # Perform query on each store
        for store in stores:
            try:
                items = await store.query(query)

                # Filter by access control
                for item in items:
                    if await self.access_controller.check_access(item, user_id):
                        # Decrypt if needed
                        if item.metadata.encryption_status and self.encryption_manager:
                            item = await self.access_controller.decrypt_memory_content(item)
                        all_results.append(item)
            except Exception as e:
                self.logger.warning(f"Error in structured search for store {store}: {str(e)}")

        # Apply limit and offset
        paginated_results = all_results[query.offset : query.offset + query.limit]

        return MemorySearchResult(
            items=paginated_results,
            total_count=len(all_results),
            query_time=time.time() - start_time,
        )

    @handle_exceptions
    async def get_recent_memories(
        self,
        memory_type: Optional[MemoryType] = None,
        limit: int = 10,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> List[MemoryItem]:
        """
        Get recent memories.

        Args:
            memory_type: Optional filter by memory type
            limit: Maximum number of memories to return
            user_id: User requesting memories (for access control)
            session_id: Optional session filter

        Returns:
            List of recent memory items
        """
        # Create time range query for recent memories (last 24 hours)
        now = datetime.now(timezone.utc)
        yesterday = now - timedelta(days=1)

        query = SimpleMemoryQuery(
            memory_type=memory_type,
            owner_id=user_id,
            session_id=session_id,
            time_range=(yesterday, now),
            limit=limit,
        )

        results = await self.search_memories(query, memory_type, user_id)
        return results.items

    @handle_exceptions
    async def get_memory_by_context(
        self,
        context_id: str,
        memory_type: Optional[MemoryType] = None,
        user_id: Optional[str] = None,
    ) -> List[MemoryItem]:
        """
        Get memories associated with a context.

        Args:
            context_id: Context identifier
            memory_type: Optional filter by memory type
            user_id: User requesting memories (for access control)

        Returns:
            List of memory items for the context
        """
        query = SimpleMemoryQuery(memory_type=memory_type, context_id=context_id, limit=100)

        results = await self.search_memories(query, memory_type, user_id)
        return results.items

    @handle_exceptions
    async def consolidate_memories(self) -> None:
        """
        Consolidate memories from short-term to long-term storage.

        This process:
        1. Identifies important short-term memories
        2. Processes them for long-term storage
        3. Creates relationships between related memories
        4. Updates retrieval indices
        5. Applies forgetting to less important memories
        """
        start_time = time.time()

        # Emit consolidation started event
        await self.event_bus.emit(MemoryConsolidationStarted())

        try:
            # Use dedicated consolidation handler if available
            if hasattr(self.consolidation, "consolidate"):
                await self.consolidation.consolidate()

                # Update metrics
                if self.metrics:
                    self.metrics.record(
                        "memory_operation_duration_seconds", time.time() - start_time
                    )

                # Emit completion event
                await self.event_bus.emit(
                    MemoryConsolidationCompleted(duration=time.time() - start_time)
                )

                return

            # Default consolidation implementation
            self.logger.info("Starting memory consolidation process")

            # 1. Get memories eligible for consolidation (short-term memories)
            recent_memories = await self.short_term_memory.get_memories_for_consolidation(
                self.config.consolidation_batch_size, self.config.importance_threshold
            )

            if not recent_memories:
                self.logger.info("No memories eligible for consolidation")
                return

            self.logger.info(f"Consolidating {len(recent_memories)} memories")

            # 2. Process memories for long-term storage
            consolidated_count = 0
            for memory in recent_memories:
                try:
                    # Skip if already consolidated
                    if memory.metadata.custom_metadata.get("consolidated", False):
                        continue

                    # Prepare for long-term storage
                    # This might involve summarization, extraction of key information, etc.
                    await self.long_term_memory.prepare_for_storage(memory)

                    # Store in long-term memory
                    memory.metadata.custom_metadata["consolidated"] = True
                    memory.metadata.retention_policy = MemoryRetentionPolicy.EXTENDED

                    await self.long_term_memory.store(memory)
                    consolidated_count += 1

                except Exception as e:
                    self.logger.error(f"Error consolidating memory {memory.memory_id}: {str(e)}")

            self.logger.info(f"Successfully consolidated {consolidated_count} memories")

            # 3. Apply forgetting to less important memories if enabled
            if self.config.forgetting_enabled:
                forgotten_count = await self._apply_forgetting()
                self.logger.info(f"Forgotten {forgotten_count} low-importance memories")

            # Update metrics
            if self.metrics:
                self.metrics.record("memory_operation_duration_seconds", time.time() - start_time)

            # Emit completion event
            await self.event_bus.emit(
                MemoryConsolidationCompleted(duration=time.time() - start_time)
            )

        except Exception as e:
            self.logger.error(f"Memory consolidation failed: {str(e)}")
            raise MemoryOperationError(f"Memory consolidation failed: {str(e)}")

    async def _apply_forgetting(self) -> int:
        """Apply forgetting to low-importance memories."""
        # Find memories eligible for forgetting
        try:
            # Find low-importance memories that have been consolidated
            criteria = SimpleMemoryQuery(
                memory_type=MemoryType.SHORT_TERM, limit=self.config.consolidation_batch_size
            )

            search_result = await self.search_memories(criteria)

            forgotten_count = 0
            for memory in search_result.items:
                # Check if consolidated and low importance
                if (
                    memory.metadata.custom_metadata.get("consolidated", False)
                    and memory.metadata.importance < self.config.importance_threshold
                ):
                    # Delete or mark as forgotten
                    memory.metadata.retention_policy = MemoryRetentionPolicy.TRANSIENT
                    await self.forget_memory(memory.memory_id)
                    forgotten_count += 1

            return forgotten_count

        except Exception as e:
            self.logger.error(f"Error applying forgetting: {str(e)}")
            return 0

    @handle_exceptions
    async def forget_memory(self, memory_id: str) -> bool:
        """
        Explicitly forget a memory.

        Args:
            memory_id: Memory identifier

        Returns:
            True if successfully forgotten
        """
        start_time = time.time()

        try:
            async with memory_operation_span(self.tracer, "delete", memory_id):
                # Retrieve memory first to get its type
                memory_item = await self.retrieve_memory(memory_id)
                if not memory_item:
                    return False

                # Remove from cache if present
                if self.memory_cache:
                    await self.memory_cache.remove(memory_id)

                # Try specialized handler first
                if memory_item.memory_type in self.memory_handlers:
                    handler = self.memory_handlers[memory_item.memory_type]
                    if hasattr(handler, "delete"):
                        success = await handler.delete(memory_id)
                        if success:
                            # Emit event
                            await self.event_bus.emit(
                                MemoryItemDeleted(
                                    memory_id=memory_id,
                                    memory_type=memory_item.memory_type.value,
                                    owner_id=memory_item.owner_id,
                                )
                            )

                            # Update metrics
                            if self.metrics:
                                self.metrics.increment("memory_operations_total")
                                self.metrics.increment("memory_delete_operations")
                                self.metrics.record(
                                    "memory_operation_duration_seconds", time.time() - start_time
                                )

                            # Update stats
                            self._update_operation_stats("delete", start_time, success=True)

                            return True

                # Fall back to generic storage
                if memory_item.memory_type in self.memory_stores:
                    store = self.memory_stores[memory_item.memory_type]
                    success = await store.delete_item(memory_id)
                else:
                    # Try all stores
                    success = False
                    for store in set(self.memory_stores.values()):
                        try:
                            if await store.delete_item(memory_id):
                                success = True
                                break
                        except Exception:
                            pass

                if success:
                    # Emit event
                    await self.event_bus.emit(
                        MemoryItemDeleted(
                            memory_id=memory_id,
                            memory_type=memory_item.memory_type.value,
                            owner_id=memory_item.owner_id,
                        )
                    )

                    # Update metrics
                    if self.metrics:
                        self.metrics.increment("memory_operations_total")
                        self.metrics.increment("memory_delete_operations")
                        self.metrics.record(
                            "memory_operation_duration_seconds", time.time() - start_time
                        )

                    # Update stats
                    self._update_operation_stats("delete", start_time, success=True)

                return success

        except Exception as e:
            self._update_operation_stats("delete", start_time, success=False)
            self.logger.error(f"Failed to forget memory {memory_id}: {str(e)}")
            raise MemoryOperationError(f"Failed to forget memory: {str(e)}", memory_id)

    @handle_exceptions
    async def get_memory_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive memory statistics.

        Returns:
            Dictionary of memory statistics
        """
        stats = {
            "total_memories": 0,
            "memory_types": {},
            "storage_usage": {},
            "cache_stats": {},
            "operation_stats": self.operation_stats,
            "consolidation_stats": {},
        }

        # Collect stats from memory types
        for memory_type, handler in self.memory_handlers.items():
            if hasattr(handler, "get_stats"):
                try:
                    type_stats = await handler.get_stats()
                    stats["memory_types"][memory_type.value] = type_stats
                    stats["total_memories"] += type_stats.get("count", 0)
                except Exception as e:
                    self.logger.warning(f"Failed to get stats for {memory_type.value}: {str(e)}")
                    stats["memory_types"][memory_type.value] = {"error": str(e)}

        # Collect stats from stores
        for store_type, store in set(self.memory_stores.items()):
            if hasattr(store, "get_statistics"):
                try:
                    store_stats = await store.get_statistics()
                    stats["storage_usage"][store_type.value] = store_stats
                except Exception as e:
                    self.logger.warning(
                        f"Failed to get stats for store {store_type.value}: {str(e)}"
                    )
                    stats["storage_usage"][store_type.value] = {"error": str(e)}

        # Get cache stats
        if self.memory_cache:
            try:
                stats["cache_stats"] = await self.memory_cache.get_stats()
            except Exception as e:
                self.logger.warning(f"Failed to get cache stats: {str(e)}")
                stats["cache_stats"] = {"error": str(e)}

        # Get consolidation stats
        if hasattr(self.consolidation, "get_stats"):
            try:
                stats["consolidation_stats"] = await self.consolidation.get_stats()
            except Exception as e:
                self.logger.warning(f"Failed to get consolidation stats: {str(e)}")
                stats["consolidation_stats"] = {"error": str(e)}

        return stats

    def _update_operation_stats(self, operation: str, start_time: float, success: bool) -> None:
        """Update operation statistics."""
        duration = time.time() - start_time

        self.operation_stats[operation]["count"] += 1
        if success:
            self.operation_stats[operation]["success_count"] += 1
        else:
            self.operation_stats[operation]["error_count"] += 1

        self.operation_stats[operation]["total_time"] += duration

        # Update average time
        count = self.operation_stats[operation]["count"]
        total_time = self.operation_stats[operation]["total_time"]
        self.operation_stats[operation]["avg_time"] = total_time / count if count > 0 else 0

    async def _consolidation_loop(self) -> None:
        """Background task for periodic memory consolidation."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(self.config.consolidation_interval)

                # Run consolidation
                await self.consolidate_memories()

            except Exception as e:
                self.logger.error(f"Error in consolidation loop: {str(e)}")
                await asyncio.sleep(60)  # Shorter delay after error

    async def _cleanup_loop(self) -> None:
        """Background task for memory cleanup."""
        while not self._shutdown_event.is_set():
            try:
                # Cleanup expired memories
                expired_count = 0
                for handler in self.memory_handlers.values():
                    if hasattr(handler, "cleanup_expired"):
                        expired_count += await handler.cleanup_expired()

                if expired_count > 0:
                    self.logger.info(f"Cleaned up {expired_count} expired memories")

                # Cleanup cache
                if self.memory_cache:
                    cache_expired = await self.memory_cache.cleanup_expired()
                    if cache_expired > 0:
                        self.logger.debug(f"Cleaned up {cache_expired} expired cache items")

                await asyncio.sleep(300)  # Run every 5 minutes

            except Exception as e:
                self.logger.error(f"Error in cleanup loop: {str(e)}")
                await asyncio.sleep(60)

    async def _backup_loop(self) -> None:
        """Background task for memory backup."""
        if not self.config.backup_enabled:
            return

        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(self.config.backup_interval)

                # Emit backup started event
                # await self.event_bus.emit(MemoryBackupStarted())

                backup_start = time.time()
                self.logger.info("Starting memory backup")

                # Create backup directory
                timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
                backup_dir = Path(f"data/backups/memory_{timestamp}")
                backup_dir.parent.mkdir(parents=True, exist_ok=True)

                # Backup each memory store
                success = True
                for store_type, store in self.memory_stores.items():
                    try:
                        store_path = backup_dir / f"{store_type.value}_store"
                        if hasattr(store, "backup"):
                            if await store.backup(store_path):
                                self.logger.info(f"Backed up {store_type.value} store")
                            else:
                                self.logger.warning(f"Failed to backup {store_type.value} store")
                                success = False
                    except Exception as e:
                        self.logger.error(f"Error backing up {store_type.value} store: {str(e)}")
                        success = False

                # Create backup metadata
                metadata = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "stores": list(self.memory_stores.keys()),
                    "stats": await self.get_memory_statistics(),
                }

                with open(backup_dir / "metadata.json", "w") as f:
                    json.dump(metadata, f)

                # Backup completed successfully
                if success:
                    self.logger.info(f"Memory backup completed: {backup_dir}")

                # Emit backup completed event
                await self.event_bus.emit(
                    # MemoryBackupCompleted(success=success, duration=time.time() - backup_start)
                )

                self.logger.info(f"Memory backup completed in {time.time() - backup_start:.2f}s")

            except Exception as e:
                self.logger.error(f"Error in backup loop: {str(e)}")
                await asyncio.sleep(300)  # Shorter delay after error

    async def restore_from_backup(self, backup_path: Union[str, Path]) -> bool:
        """
        Restore memory from backup.

        Args:
            backup_path: Path to backup directory

        Returns:
            True if successful
        """
        if isinstance(backup_path, str):
            backup_path = Path(backup_path)

        if not backup_path.exists():
            raise MemoryOperationError(f"Backup path {backup_path} does not exist")

        # Emit restore started event
        # await self.event_bus.emit(MemoryRestoreStarted(backup_path=str(backup_path)))

        try:
            restore_start = time.time()
            self.logger.info(f"Starting memory restore from {backup_path}")

            # Load metadata
            metadata_path = backup_path / "metadata.json"
            if not metadata_path.exists():
                raise MemoryOperationError(f"Backup metadata not found at {metadata_path}")

            with open(metadata_path, "r") as f:
                metadata = json.load(f)

            # Restore each memory store
            success = True
            for store_type, store in self.memory_stores.items():
                try:
                    store_path = backup_path / f"{store_type.value}_store"
                    if store_path.exists() and hasattr(store, "restore"):
                        if await store.restore(store_path):
                            self.logger.info(f"Restored {store_type.value} store")
                        else:
                            self.logger.warning(f"Failed to restore {store_type.value} store")
                            success = False
                except Exception as e:
                    self.logger.error(f"Error restoring {store_type.value} store: {str(e)}")
                    success = False

            # Clear cache after restore
            if self.memory_cache:
                await self.memory_cache.clear()

            # Emit restore completed event
            await self.event_bus.emit(
                # MemoryRestoreCompleted(success=success, duration=time.time() - restore_start)
            )

            self.logger.info(f"Memory restore completed in {time.time() - restore_start:.2f}s")
            return success

        except Exception as e:
            self.logger.error(f"Failed to restore from backup: {str(e)}")
            # Emit restore completed event with failure
            await self.event_bus.emit(# MemoryRestoreCompleted(success=False, error=str(e)))
            raise MemoryOperationError(f"Failed to restore from backup: {str(e)}")

    async def _handle_system_shutdown(self, event) -> None:
        """Handle system shutdown events."""
        self._shutdown_event.set()
        self.logger.info("Memory manager received shutdown signal")

        # Run one final backup if enabled
        if self.config.backup_enabled:
            try:
                self.logger.info("Running final memory backup before shutdown")
                # Create simple backup with shutdown marker
                timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
                backup_dir = Path(f"data/backups/memory_shutdown_{timestamp}")
                backup_dir.parent.mkdir(parents=True, exist_ok=True)
                self.logger.info(f"Final backup directory created: {backup_dir}")

                # Backup each store
                for store_type, store in self.memory_stores.items():
                    try:
                        if hasattr(store, "backup"):
                            store_path = backup_dir / f"{store_type.value}_store"
                            await store.backup(store_path)
                    except Exception as e:
                        self.logger.error(f"Error in final backup of {store_type.value}: {str(e)}")

                # Create metadata
                metadata = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "shutdown": True,
                    "stores": list(str(k.value) for k in self.memory_stores.keys()),
                }

                with open(backup_dir / "metadata.json", "w") as f:
                    json.dump(metadata, f)

                self.logger.info("Final memory backup completed")

            except Exception as e:
                self.logger.error(f"Failed to create final backup: {str(e)}")

    async def _handle_session_ended(self, event) -> None:
        """Handle session ended events."""
        session_id = event.session_id

        try:
            # Archive session memories
            for memory_type, handler in self.memory_handlers.items():
                if hasattr(handler, "archive_session"):
                    await handler.archive_session(session_id)

            self.logger.info(f"Archived memories for session {session_id}")

        except Exception as e:
            self.logger.error(f"Error archiving session memories: {str(e)}")

    async def _handle_learning_event(self, event) -> None:
        """Handle learning events."""
        if not self.continual_learner:
            return

        try:
            # Update memory importance based on learning events
            if event.event_type == "memory_importance_update":
                memory_id = event.data.get("memory_id")
                importance = event.data.get("importance")

                if memory_id and importance is not None:
                    memory = await self.retrieve_memory(memory_id)
                    if memory:
                        memory.metadata.importance = importance
                        await self.update_memory(memory)

        except Exception as e:
            self.logger.error(f"Error handling importance event: {str(e)}")
