"""
Semantic Memory Implementation
Author: Drmusab
Last Modified: 2025-01-10 21:30:00 UTC

This module implements semantic memory - a system for storing factual and conceptual
knowledge. It provides vector-based knowledge representation, concept hierarchies,
fast semantic search, knowledge graph integration, and fact verification capabilities.
"""

import asyncio
import json
import logging
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np

# Core imports
from src.core.config.loader import ConfigLoader
from src.core.dependency_injection import Container
from src.core.error_handling import ErrorHandler, handle_exceptions
from src.core.events.event_bus import EventBus
from src.core.events.event_types import (
    MemoryItemRetrieved,
    MemoryItemStored,
    MemoryItemUpdated,
)

# Integration components
from src.integrations.llm.model_router import ModelRouter

# Memory system imports
from src.memory.core_memory.base_memory import (
    BaseMemory,
    MemoryError,
    MemoryItem,
    MemoryMetadata,
    MemoryRetentionPolicy,
    MemoryType,
    MemoryUtils,
    SimpleMemoryQuery,
    memory_operation_span,
)
from src.memory.storage.memory_graph import MemoryGraphStore
from src.memory.storage.vector_store import VectorMemoryStore

# Knowledge and reasoning integration
from src.reasoning.knowledge_graph import KnowledgeGraph
from src.reasoning.inference_engine import InferenceEngine

# Observability
from src.observability.logging.config import get_logger
from src.observability.monitoring.metrics import MetricsCollector
from src.observability.monitoring.tracing import TraceManager


class SemanticMemoryConfig:
    """Configuration for semantic memory."""
    
    def __init__(self, config_loader: ConfigLoader):
        """Initialize configuration from config loader."""
        self.similarity_threshold = config_loader.get("memory.semantic.similarity_threshold", 0.7)
        self.max_cache_size = config_loader.get("memory.semantic.cache_size", 1000)
        self.auto_validation = config_loader.get("memory.semantic.auto_validation", True)
        self.confidence_threshold = config_loader.get("memory.semantic.confidence_threshold", 0.8)
        self.enable_relationships = config_loader.get("memory.semantic.enable_relationships", True)
        self.batch_size = config_loader.get("memory.semantic.batch_size", 100)
        self.validation_interval = config_loader.get("memory.semantic.validation_interval", 86400)  # 24 hours
        self.concept_hierarchy_depth = config_loader.get("memory.semantic.concept_hierarchy_depth", 5)
        self.enable_inference = config_loader.get("memory.semantic.enable_inference", True)


class SemanticMemory(BaseMemory):
    """
    Semantic memory implementation - stores factual and conceptual knowledge.
    
    This implementation provides:
    - Vector-based knowledge representation with embeddings
    - Concept hierarchies and relationships
    - Fast semantic search and retrieval
    - Knowledge graph integration
    - Fact verification and consistency checking
    - Automatic knowledge validation
    - Inference capabilities for deriving new knowledge
    """

    def __init__(
        self,
        container: Container,
        vector_store: VectorMemoryStore,
        model_router: ModelRouter,
    ):
        """
        Initialize semantic memory.

        Args:
            container: Dependency injection container
            vector_store: Vector store for embeddings
            model_router: Model router for embeddings
        """
        self.container = container
        self.vector_store = vector_store
        self.model_router = model_router
        self.logger = get_logger(__name__)

        # Get core dependencies
        self.event_bus = container.get(EventBus)
        self.error_handler = container.get(ErrorHandler)
        
        # Load configuration
        config_loader = container.get(ConfigLoader)
        self.config = SemanticMemoryConfig(config_loader)

        # Get optional dependencies
        try:
            self.graph_store = container.get(MemoryGraphStore)
        except Exception:
            self.logger.warning("Graph store not available, semantic relationships limited")
            self.graph_store = None

        try:
            self.knowledge_graph = container.get(KnowledgeGraph)
        except Exception:
            self.logger.warning("Knowledge graph not available")
            self.knowledge_graph = None

        try:
            self.inference_engine = container.get(InferenceEngine)
        except Exception:
            self.logger.warning("Inference engine not available")
            self.inference_engine = None

        # Get monitoring components
        try:
            self.metrics = container.get(MetricsCollector)
            self.tracer = container.get(TraceManager)
        except Exception:
            self.metrics = None
            self.tracer = None
            self.logger.warning("Monitoring components not available")

        # Initialize semantic indices
        self._initialize_indices()

        # Cache
        self._semantic_cache: Dict[str, MemoryItem] = {}
        self._cache_lock = asyncio.Lock()

        # Background task management
        self._background_tasks: List[asyncio.Task] = []
        self._shutdown_event = asyncio.Event()

        self.logger.info("SemanticMemory initialized successfully")

    def _initialize_indices(self) -> None:
        """Initialize semantic indices for efficient retrieval."""
        self._concept_index: Dict[str, List[str]] = defaultdict(list)
        self._domain_index: Dict[str, List[str]] = defaultdict(list)
        self._entity_index: Dict[str, List[str]] = defaultdict(list)
        self._relation_index: Dict[str, List[Tuple[str, str, str]]] = defaultdict(list)
        
        # Validation tracking
        self._fact_confidence: Dict[str, float] = {}
        self._fact_validation_status: Dict[str, str] = {}
        self._validation_timestamps: Dict[str, datetime] = {}
        
        # Concept hierarchy
        self._concept_hierarchy: Dict[str, Set[str]] = defaultdict(set)
        self._concept_children: Dict[str, Set[str]] = defaultdict(set)

    async def initialize(self) -> None:
        """Initialize semantic memory and start background tasks."""
        try:
            self.logger.info("Initializing semantic memory...")

            # Initialize storage backends
            if hasattr(self.vector_store, 'initialize'):
                await self.vector_store.initialize()
            
            if self.graph_store and hasattr(self.graph_store, 'initialize'):
                await self.graph_store.initialize()

            # Start background tasks
            if self.config.auto_validation:
                self._background_tasks.append(
                    asyncio.create_task(self._validation_loop())
                )

            # Rebuild indices from existing data
            await self._rebuild_indices()

            self.logger.info("Semantic memory initialization complete")

        except Exception as e:
            self.logger.error(f"Failed to initialize semantic memory: {str(e)}")
            raise MemoryError(f"Semantic memory initialization failed: {str(e)}")

    async def shutdown(self) -> None:
        """Shutdown semantic memory and cleanup resources."""
        self.logger.info("Shutting down semantic memory...")
        
        # Signal shutdown to background tasks
        self._shutdown_event.set()

        # Cancel and wait for background tasks
        for task in self._background_tasks:
            task.cancel()

        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)

        # Clear caches and indices
        self._semantic_cache.clear()
        self._clear_indices()

        self.logger.info("Semantic memory shutdown complete")

    @handle_exceptions
    async def store(self, data: Any, **kwargs) -> str:
        """
        Store data in semantic memory.

        Args:
            data: Data to store (fact, concept, knowledge)
            **kwargs: Additional parameters including:
                concepts: List of related concepts
                domain: Knowledge domain
                entities: List of related entities
                confidence: Fact confidence (0-1)
                relationships: Dict of relationships to other concepts
                source: Source of the knowledge
                validation_required: Whether validation is required

        Returns:
            Memory ID
        """
        async with memory_operation_span(self.tracer, "store_semantic"):
            # Generate memory ID
            memory_id = MemoryUtils.generate_memory_id()

            # Create metadata
            metadata = await self._create_metadata(data, **kwargs)

            # Generate embeddings
            embeddings = await self._generate_embeddings(data)
            if not embeddings:
                raise MemoryError("Failed to generate embeddings for semantic memory")

            # Create memory item
            memory_item = MemoryItem(
                memory_id=memory_id,
                content=data,
                memory_type=MemoryType.SEMANTIC,
                owner_id=kwargs.get("owner_id"),
                context_id=kwargs.get("context_id"),
                metadata=metadata,
                embeddings=embeddings,
                relationships=kwargs.get("relationships", {}),
            )

            # Store in vector store
            await self.vector_store.store_item(memory_item)

            # Update semantic indices
            await self._update_indices_on_store(memory_item, **kwargs)

            # Store relationships in graph if available
            if self.config.enable_relationships and self.graph_store and memory_item.relationships:
                await self._store_semantic_relationships(memory_id, data, memory_item.relationships)

            # Add to knowledge graph if available
            if self.knowledge_graph:
                await self._add_to_knowledge_graph(memory_item, **kwargs)

            # Add to cache
            await self._add_to_cache(memory_item)

            # Schedule validation if required
            if kwargs.get("validation_required", self.config.auto_validation):
                self._fact_validation_status[memory_id] = "pending"

            # Emit event
            await self.event_bus.emit(
                MemoryItemStored(
                    memory_id=memory_id,
                    memory_type=MemoryType.SEMANTIC.value,
                    owner_id=kwargs.get("owner_id"),
                    context_id=kwargs.get("context_id"),
                )
            )

            # Update metrics
            if self.metrics:
                self.metrics.increment("semantic_memory_items_stored")
                self.metrics.gauge("semantic_concepts_count", len(self._concept_index))
                self.metrics.gauge("semantic_domains_count", len(self._domain_index))

            self.logger.debug(f"Stored semantic memory {memory_id}")
            return memory_id

    @handle_exceptions
    async def retrieve(self, memory_id: str) -> Optional[MemoryItem]:
        """
        Retrieve an item from semantic memory.

        Args:
            memory_id: Memory identifier

        Returns:
            Memory item or None if not found
        """
        async with memory_operation_span(self.tracer, "retrieve_semantic", memory_id):
            # Check cache first
            async with self._cache_lock:
                if memory_id in self._semantic_cache:
                    item = self._semantic_cache[memory_id]
                    item.metadata.update_access()
                    return item

            # Retrieve from store
            item = await self.vector_store.get_item(memory_id)
            
            if item:
                # Update metadata
                item.metadata.update_access()
                
                # Add to cache
                await self._add_to_cache(item)
                
                # Update access in store
                await self.vector_store.update_item(memory_id, {"metadata": item.metadata})
                
                # Emit event
                await self.event_bus.emit(
                    MemoryItemRetrieved(
                        memory_id=memory_id,
                        memory_type=MemoryType.SEMANTIC.value,
                        owner_id=item.owner_id,
                    )
                )
                
                # Update metrics
                if self.metrics:
                    self.metrics.increment("semantic_memory_items_retrieved")

            return item

    async def update(self, memory_id: str, data: Any) -> bool:
        """
        Update existing memory.

        Args:
            memory_id: Memory identifier
            data: New data

        Returns:
            True if successful
        """
        async with memory_operation_span(self.tracer, "update_semantic", memory_id):
            # Retrieve existing memory
            memory_item = await self.retrieve(memory_id)
            if not memory_item:
                return False

            # Update content
            memory_item.content = data
            memory_item.metadata.update_modification()

            # Regenerate embeddings
            memory_item.embeddings = await self._generate_embeddings(data)
            if not memory_item.embeddings:
                self.logger.warning(f"Failed to regenerate embeddings for {memory_id}")
                return False

            # Reset validation status
            self._fact_validation_status[memory_id] = "pending"

            # Store updated item
            await self.vector_store.store_item(memory_item)

            # Update cache
            await self._add_to_cache(memory_item)

            # Emit event
            await self.event_bus.emit(
                MemoryItemUpdated(
                    memory_id=memory_id,
                    memory_type=MemoryType.SEMANTIC.value,
                    owner_id=memory_item.owner_id,
                )
            )

            return True

    async def delete(self, memory_id: str) -> bool:
        """
        Delete memory.

        Args:
            memory_id: Memory identifier

        Returns:
            True if successful
        """
        async with memory_operation_span(self.tracer, "delete_semantic", memory_id):
            # Remove from indices
            await self._remove_from_indices(memory_id)

            # Remove from cache
            async with self._cache_lock:
                self._semantic_cache.pop(memory_id, None)

            # Remove from validation tracking
            self._fact_confidence.pop(memory_id, None)
            self._fact_validation_status.pop(memory_id, None)
            self._validation_timestamps.pop(memory_id, None)

            # Remove from graph store if available
            if self.graph_store:
                await self._remove_from_graph(memory_id)

            # Delete from vector store
            return await self.vector_store.delete_item(memory_id)

    async def search(self, query: Any) -> List[MemoryItem]:
        """
        Search semantic memory.

        Args:
            query: Search query (text, embedding, or query object)

        Returns:
            List of matching memory items
        """
        async with memory_operation_span(self.tracer, "search_semantic"):
            if isinstance(query, str):
                # Text query - convert to embedding for semantic search
                return await self._semantic_search(query)

            elif isinstance(query, list) and all(isinstance(x, (int, float)) for x in query):
                # Vector query
                return await self.vector_store.similarity_search(
                    query, 
                    similarity_threshold=self.config.similarity_threshold, 
                    top_k=10
                )

            elif isinstance(query, SimpleMemoryQuery):
                # Use memory store query capabilities
                return await self.vector_store.query(query)

            elif isinstance(query, dict):
                # Handle dictionary queries
                return await self._handle_dict_query(query)

            else:
                raise MemoryError(f"Unsupported query type for semantic memory: {type(query)}")

    async def clear(self) -> None:
        """Clear all semantic memory."""
        # Clear indices
        self._clear_indices()

        # Clear cache
        async with self._cache_lock:
            self._semantic_cache.clear()

        # Clear validation tracking
        self._fact_confidence.clear()
        self._fact_validation_status.clear()
        self._validation_timestamps.clear()

        # Clear underlying store
        query = SimpleMemoryQuery(memory_type=MemoryType.SEMANTIC)
        items = await self.vector_store.query(query)

        for item in items:
            await self.vector_store.delete_item(item.memory_id)

        self.logger.info(f"Cleared all semantic memory ({len(items)} items)")

    async def get_stats(self) -> Dict[str, Any]:
        """Get semantic memory statistics."""
        # Count total items by summing all concept entries
        total_items = len(set(
            memory_id 
            for concept_memories in self._concept_index.values() 
            for memory_id in concept_memories
        ))

        stats = {
            "total_items": total_items,
            "concepts_count": len(self._concept_index),
            "domains_count": len(self._domain_index),
            "entities_count": len(self._entity_index),
            "relations_count": len(self._relation_index),
            "cache_size": len(self._semantic_cache),
            "memory_type": MemoryType.SEMANTIC.value,
            "validation_stats": {
                "pending": sum(1 for s in self._fact_validation_status.values() if s == "pending"),
                "validated": sum(1 for s in self._fact_validation_status.values() if s == "validated"),
                "failed": sum(1 for s in self._fact_validation_status.values() if s == "failed"),
            },
            "confidence_distribution": self._get_confidence_distribution(),
        }

        # Add domain breakdown
        stats["domain_breakdown"] = {
            domain: len(memories) for domain, memories in self._domain_index.items()
        }

        # Add concept hierarchy stats
        stats["concept_hierarchy"] = {
            "total_hierarchies": len(self._concept_hierarchy),
            "max_depth": self._get_max_hierarchy_depth(),
        }

        return stats

    # Additional public methods specific to semantic memory

    async def retrieve_relevant(
        self, query: str, context: Optional[Dict[str, Any]] = None, limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant semantic memories for a query and context.

        Args:
            query: Query text
            context: Optional context information
            limit: Maximum number of items to return

        Returns:
            List of relevant semantic knowledge items
        """
        # Start with semantic search
        items = await self._semantic_search(query, limit=limit * 2)  # Get more for filtering

        # If we have context, refine the results
        if context:
            items = await self._refine_by_context(items, context)

        # Perform inference if available
        if self.config.enable_inference and self.inference_engine:
            items = await self._apply_inference(items, query, context)

        # Format and limit results
        result = []
        for item in items[:limit]:
            fact_data = {
                "content": item.content,
                "confidence": item.metadata.custom_metadata.get("confidence", 0.5),
                "domain": item.metadata.custom_metadata.get("domain", "general"),
                "validation_status": self._fact_validation_status.get(item.memory_id, "unknown"),
                "id": item.memory_id,
                "relationships": item.relationships,
            }
            result.append(fact_data)

        return result

    async def get_by_concept(self, concept: str, limit: int = 10) -> List[MemoryItem]:
        """Get memories related to a concept."""
        if concept not in self._concept_index:
            # Check for similar concepts
            similar_concepts = await self._find_similar_concepts(concept)
            if similar_concepts:
                concept = similar_concepts[0]  # Use most similar
            else:
                return []

        memory_ids = self._concept_index[concept][:limit]
        return await self._retrieve_multiple(memory_ids)

    async def get_by_entity(self, entity: str, limit: int = 10) -> List[MemoryItem]:
        """Get memories related to an entity."""
        if entity not in self._entity_index:
            return []

        memory_ids = self._entity_index[entity][:limit]
        return await self._retrieve_multiple(memory_ids)

    async def get_by_domain(self, domain: str, limit: int = 10) -> List[MemoryItem]:
        """Get memories in a specific knowledge domain."""
        if domain not in self._domain_index:
            return []

        memory_ids = self._domain_index[domain][:limit]
        return await self._retrieve_multiple(memory_ids)

    async def get_related_concepts(self, concept: str, depth: int = 1) -> Dict[str, List[str]]:
        """Get concepts related to a given concept."""
        related = {
            "parents": list(self._concept_hierarchy.get(concept, set())),
            "children": list(self._concept_children.get(concept, set())),
            "siblings": [],
            "related": [],
        }

        # Find siblings (concepts with same parent)
        for parent in related["parents"]:
            siblings = self._concept_children.get(parent, set())
            related["siblings"].extend([s for s in siblings if s != concept])

        # Find related concepts through shared memories
        if concept in self._concept_index:
            memory_ids = self._concept_index[concept]
            for memory_id in memory_ids[:10]:  # Limit to prevent explosion
                memory = await self.retrieve(memory_id)
                if memory and "concepts" in memory.metadata.custom_metadata:
                    for other_concept in memory.metadata.custom_metadata["concepts"]:
                        if other_concept != concept and other_concept not in related["related"]:
                            related["related"].append(other_concept)

        # Recursively get related concepts if depth > 1
        if depth > 1:
            for category in ["parents", "children"]:
                expanded = []
                for related_concept in related[category]:
                    sub_related = await self.get_related_concepts(related_concept, depth - 1)
                    expanded.extend(sub_related.get(category, []))
                related[f"extended_{category}"] = expanded

        return related

    async def validate_fact(self, memory_id: str, validation_result: bool, confidence: float = None) -> bool:
        """
        Validate a fact in semantic memory.

        Args:
            memory_id: Memory identifier
            validation_result: Whether the fact is valid
            confidence: Optional confidence score

        Returns:
            True if successful
        """
        try:
            memory_item = await self.retrieve(memory_id)
            if not memory_item:
                return False

            # Update validation status
            self._fact_validation_status[memory_id] = "validated" if validation_result else "failed"
            self._validation_timestamps[memory_id] = datetime.now(timezone.utc)

            # Update confidence if provided
            if confidence is not None:
                self._fact_confidence[memory_id] = confidence
                memory_item.metadata.custom_metadata["confidence"] = confidence

            # Update validation metadata
            memory_item.metadata.custom_metadata["validated"] = validation_result
            memory_item.metadata.custom_metadata["validation_timestamp"] = datetime.now(timezone.utc).isoformat()

            # Store updated item
            await self.vector_store.store_item(memory_item)

            # Update cache
            await self._add_to_cache(memory_item)

            # If validation failed and we have inference engine, mark for review
            if not validation_result and self.inference_engine:
                await self._mark_for_review(memory_id, "validation_failed")

            return True

        except Exception as e:
            self.logger.error(f"Error validating fact {memory_id}: {str(e)}")
            return False

    async def check_consistency(self, memory_id: str) -> Dict[str, Any]:
        """
        Check consistency of a fact with existing knowledge.

        Args:
            memory_id: Memory identifier

        Returns:
            Consistency check results
        """
        memory_item = await self.retrieve(memory_id)
        if not memory_item:
            return {"status": "error", "message": "Memory not found"}

        # Find related facts
        if memory_item.embeddings:
            similar_items = await self.vector_store.similarity_search(
                memory_item.embeddings,
                similarity_threshold=0.8,
                top_k=20
            )
        else:
            similar_items = []

        # Check for contradictions
        contradictions = []
        supporting_facts = []

        for item in similar_items:
            if item.memory_id == memory_id:
                continue

            # Simple contradiction detection (would be more sophisticated in practice)
            if await self._check_contradiction(memory_item, item):
                contradictions.append({
                    "memory_id": item.memory_id,
                    "content": item.content,
                    "confidence": item.metadata.custom_metadata.get("confidence", 0.5)
                })
            else:
                supporting_facts.append({
                    "memory_id": item.memory_id,
                    "content": item.content,
                    "confidence": item.metadata.custom_metadata.get("confidence", 0.5)
                })

        consistency_score = len(supporting_facts) / (len(supporting_facts) + len(contradictions)) if (supporting_facts or contradictions) else 1.0

        return {
            "status": "complete",
            "consistency_score": consistency_score,
            "contradictions": contradictions,
            "supporting_facts": supporting_facts,
            "is_consistent": len(contradictions) == 0,
        }

    # Private helper methods

    async def _create_metadata(self, data: Any, **kwargs) -> MemoryMetadata:
        """Create metadata for a semantic memory item."""
        # Extract metadata from kwargs
        concepts = kwargs.get("concepts", [])
        domain = kwargs.get("domain", "general")
        entities = kwargs.get("entities", [])
        confidence = kwargs.get("confidence", 0.9)
        source = kwargs.get("source")

        # Create base metadata
        metadata = MemoryMetadata(
            retention_policy=MemoryRetentionPolicy.EXTENDED,
            tags=set(concepts) | {domain} | set(entities),
            importance=kwargs.get("importance", 0.7),
            custom_metadata={
                "domain": domain,
                "concepts": concepts,
                "entities": entities,
                "confidence": confidence,
                "validation_status": "unverified",
            }
        )

        # Add source if provided
        if source:
            metadata.source = source
            metadata.custom_metadata["source"] = source

        # Extract additional entities from content
        if isinstance(data, str):
            extracted_entities = MemoryUtils.extract_entities(data)
            entities.extend(extracted_entities)
            metadata.custom_metadata["entities"] = list(set(entities))

        return metadata

    async def _generate_embeddings(self, data: Any) -> Optional[List[float]]:
        """Generate embeddings for memory content."""
        try:
            # Convert to text for embedding
            if isinstance(data, dict):
                # Create a structured representation for better embeddings
                parts = []
                if "subject" in data:
                    parts.append(f"Subject: {data['subject']}")
                if "predicate" in data:
                    parts.append(f"Predicate: {data['predicate']}")
                if "object" in data:
                    parts.append(f"Object: {data['object']}")
                if "description" in data:
                    parts.append(f"Description: {data['description']}")
                
                embed_text = " ".join(parts) if parts else json.dumps(data)
            elif not isinstance(data, str):
                embed_text = str(data)
            else:
                embed_text = data

            return await self.model_router.get_embeddings(embed_text)

        except Exception as e:
            self.logger.error(f"Failed to generate embeddings: {str(e)}")
            return None

    async def _update_indices_on_store(self, memory_item: MemoryItem, **kwargs) -> None:
        """Update all indices when storing a new memory."""
        memory_id = memory_item.memory_id
        metadata = memory_item.metadata.custom_metadata

        # Update concept index
        concepts = metadata.get("concepts", [])
        for concept in concepts:
            self._concept_index[concept].append(memory_id)

        # Update domain index
        domain = metadata.get("domain", "general")
        self._domain_index[domain].append(memory_id)

        # Update entity index
        entities = metadata.get("entities", [])
        for entity in entities:
            self._entity_index[entity].append(memory_id)

        # Update confidence tracking
        confidence = metadata.get("confidence", 0.9)
        self._fact_confidence[memory_id] = confidence

        # Update concept hierarchy if parent concepts provided
        parent_concepts = kwargs.get("parent_concepts", [])
        for parent in parent_concepts:
            for concept in concepts:
                self._concept_hierarchy[concept].add(parent)
                self._concept_children[parent].add(concept)

    async def _remove_from_indices(self, memory_id: str) -> None:
        """Remove a memory from all indices."""
        # Retrieve memory to get metadata
        memory = await self.vector_store.get_item(memory_id)
        if not memory:
            return

        metadata = memory.metadata.custom_metadata

        # Remove from concept index
        concepts = metadata.get("concepts", [])
        for concept in concepts:
            if concept in self._concept_index:
                self._concept_index[concept] = [
                    mid for mid in self._concept_index[concept] if mid != memory_id
                ]

        # Remove from domain index
        domain = metadata.get("domain", "general")
        if domain in self._domain_index:
            self._domain_index[domain] = [
                mid for mid in self._domain_index[domain] if mid != memory_id
            ]

        # Remove from entity index
        entities = metadata.get("entities", [])
        for entity in entities:
            if entity in self._entity_index:
                self._entity_index[entity] = [
                    mid for mid in self._entity_index[entity] if mid != memory_id
                ]

    def _clear_indices(self) -> None:
        """Clear all semantic indices."""
        self._concept_index.clear()
        self._domain_index.clear()
        self._entity_index.clear()
        self._relation_index.clear()
        self._concept_hierarchy.clear()
        self._concept_children.clear()

    async def _rebuild_indices(self) -> None:
        """Rebuild all indices from stored memories."""
        self.logger.info("Rebuilding semantic memory indices...")
        
        # Clear existing indices
        self._clear_indices()

        # Query all semantic memories
        query = SimpleMemoryQuery(memory_type=MemoryType.SEMANTIC, limit=10000)
        items = await self.vector_store.query(query)

        # Process in batches
        for i in range(0, len(items), self.config.batch_size):
            batch = items[i:i + self.config.batch_size]
            
            for item in batch:
                await self._update_indices_on_store(item)

        self.logger.info(f"Rebuilt indices for {len(items)} semantic memories")

    async def _add_to_cache(self, item: MemoryItem) -> None:
        """Add item to semantic cache."""
        async with self._cache_lock:
            # Enforce cache size limit
            if len(self._semantic_cache) >= self.config.max_cache_size:
                # Remove oldest item
                if self._semantic_cache:
                    self._semantic_cache.pop(next(iter(self._semantic_cache)))

            # Add to cache
            self._semantic_cache[item.memory_id] = item

    async def _semantic_search(self, query: str, limit: int = 10) -> List[MemoryItem]:
        """Perform semantic search on memories."""
        try:
            # Generate query embedding
            query_embedding = await self.model_router.get_embeddings(query)
            if not query_embedding:
                # Fall back to concept/entity search
                return await self._concept_entity_search(query, limit)

            # Perform similarity search
            items = await self.vector_store.similarity_search(
                query_embedding,
                similarity_threshold=self.config.similarity_threshold,
                top_k=limit
            )

            return items

        except Exception as e:
            self.logger.error(f"Semantic search error: {str(e)}")
            # Fall back to concept/entity search
            return await self._concept_entity_search(query, limit)

    async def _concept_entity_search(self, query: str, limit: int = 10) -> List[MemoryItem]:
        """Search by concepts and entities in the query."""
        query_lower = query.lower()
        matching_memory_ids = set()

        # Check for concept matches
        for concept in self._concept_index:
            if concept.lower() in query_lower:
                matching_memory_ids.update(self._concept_index[concept])

        # Check for entity matches
        for entity in self._entity_index:
            if entity.lower() in query_lower:
                matching_memory_ids.update(self._entity_index[entity])

        # Check for domain matches
        for domain in self._domain_index:
            if domain.lower() in query_lower:
                matching_memory_ids.update(self._domain_index[domain])

        # Retrieve matching items
        items = await self._retrieve_multiple(list(matching_memory_ids)[:limit * 2])

        # Sort by confidence and limit
        items.sort(
            key=lambda x: x.metadata.custom_metadata.get("confidence", 0.0), 
            reverse=True
        )

        return items[:limit]

    async def _handle_dict_query(self, query: dict) -> List[MemoryItem]:
        """Handle dictionary-based queries."""
        # Extract query parameters
        concepts = query.get("concepts", [])
        domain = query.get("domain")
        entities = query.get("entities", [])
        min_confidence = query.get("min_confidence", 0.0)

        # Find matching memory IDs
        matching_ids = None

        # Filter by concepts
        if concepts:
            concept_ids = set()
            for concept in concepts:
                if concept in self._concept_index:
                    concept_ids.update(self._concept_index[concept])
            matching_ids = concept_ids

        # Filter by domain
        if domain and domain in self._domain_index:
            domain_ids = set(self._domain_index[domain])
            if matching_ids is None:
                matching_ids = domain_ids
            else:
                matching_ids = matching_ids.intersection(domain_ids)

        # Filter by entities
        if entities:
            entity_ids = set()
            for entity in entities:
                if entity in self._entity_index:
                    entity_ids.update(self._entity_index[entity])
            if matching_ids is None:
                matching_ids = entity_ids
            else:
                matching_ids = matching_ids.intersection(entity_ids)

        if matching_ids is None:
            return []

        # Retrieve items
        items = await self._retrieve_multiple(list(matching_ids))

        # Filter by confidence
        if min_confidence > 0:
            items = [
                item for item in items
                if item.metadata.custom_metadata.get("confidence", 0.0) >= min_confidence
            ]

        # Apply limit
        limit = query.get("limit", 100)
        return items[:limit]

    async def _retrieve_multiple(self, memory_ids: List[str]) -> List[MemoryItem]:
        """Retrieve multiple memory items efficiently."""
        items = []
        
        # Check cache first
        uncached_ids = []
        async with self._cache_lock:
            for memory_id in memory_ids:
                if memory_id in self._semantic_cache:
                    items.append(self._semantic_cache[memory_id])
                else:
                    uncached_ids.append(memory_id)

        # Batch retrieve uncached items if supported
        if uncached_ids:
            if hasattr(self.vector_store, 'get_items'):
                retrieved_items = await self.vector_store.get_items(uncached_ids)
                items.extend(retrieved_items)
            else:
                # Fall back to individual retrieval
                for memory_id in uncached_ids:
                    item = await self.retrieve(memory_id)
                    if item:
                        items.append(item)

        return items

    async def _store_semantic_relationships(
        self, memory_id: str, data: Any, relationships: Dict[str, Any]
    ) -> None:
        """Store semantic relationships in graph store."""
        if not self.graph_store:
            return

        try:
            # Convert relationships to graph edges
            for relation_type, related_items in relationships.items():
                if isinstance(related_items, list):
                    for related_id in related_items:
                        await self.graph_store.add_edge(
                            memory_id, 
                            related_id, 
                            relation_type
                        )
                        # Update relation index
                        self._relation_index[relation_type].append(
                            (memory_id, relation_type, related_id)
                        )
                elif isinstance(related_items, str):
                    await self.graph_store.add_edge(
                        memory_id, 
                        related_items, 
                        relation_type
                    )
                    self._relation_index[relation_type].append(
                        (memory_id, relation_type, related_items)
                    )

        except Exception as e:
            self.logger.warning(f"Failed to store semantic relationships: {str(e)}")

    async def _remove_from_graph(self, memory_id: str) -> None:
        """Remove memory from graph store."""
        if not self.graph_store:
            return

        try:
            await self.graph_store.remove_node(memory_id)
            
            # Update relation index
            for relation_type in list(self._relation_index.keys()):
                self._relation_index[relation_type] = [
                    (src, rel, tgt) 
                    for src, rel, tgt in self._relation_index[relation_type]
                    if src != memory_id and tgt != memory_id
                ]

        except Exception as e:
            self.logger.warning(f"Failed to remove from graph: {str(e)}")

    async def _add_to_knowledge_graph(self, memory_item: MemoryItem, **kwargs) -> None:
        """Add memory to knowledge graph if available."""
        if not self.knowledge_graph:
            return

        try:
            # Extract triple if possible
            content = memory_item.content
            if isinstance(content, dict) and all(k in content for k in ["subject", "predicate", "object"]):
                await self.knowledge_graph.add_triple(
                    content["subject"],
                    content["predicate"],
                    content["object"],
                    metadata={
                        "memory_id": memory_item.memory_id,
                        "confidence": memory_item.metadata.custom_metadata.get("confidence", 0.9),
                        "source": memory_item.metadata.source,
                    }
                )

        except Exception as e:
            self.logger.warning(f"Failed to add to knowledge graph: {str(e)}")

    async def _find_similar_concepts(self, concept: str) -> List[str]:
        """Find concepts similar to the given one."""
        # Simple implementation - in practice would use embeddings
        similar = []
        concept_lower = concept.lower()

        for known_concept in self._concept_index:
            if concept_lower in known_concept.lower() or known_concept.lower() in concept_lower:
                similar.append(known_concept)

        return similar[:5]  # Return top 5

    async def _refine_by_context(
        self, items: List[MemoryItem], context: Dict[str, Any]
    ) -> List[MemoryItem]:
        """Refine search results based on context."""
        scored_items = []

        for item in items:
            # Base score
            score = 1.0

            # Check for context matches
            metadata = item.metadata.custom_metadata

            # Domain match
            if "domain" in context and context["domain"] == metadata.get("domain"):
                score += 0.3

            # Concept overlap
            if "concepts" in context and "concepts" in metadata:
                context_concepts = set(context["concepts"])
                item_concepts = set(metadata["concepts"])
                overlap = len(context_concepts & item_concepts)
                if overlap > 0:
                    score += 0.5 * (overlap / len(item_concepts))

            # Entity overlap
            if "entities" in context and "entities" in metadata:
                context_entities = set(context["entities"])
                item_entities = set(metadata["entities"])
                overlap = len(context_entities & item_entities)
                if overlap > 0:
                    score += 0.3 * (overlap / len(item_entities))

            # Boost by confidence
            confidence = metadata.get("confidence", 0.5)
            score *= confidence

            # Boost by validation status
            if self._fact_validation_status.get(item.memory_id) == "validated":
                score *= 1.2

            scored_items.append((item, score))

        # Sort by score
        scored_items.sort(key=lambda x: x[1], reverse=True)
        return [item for item, _ in scored_items]

    async def _apply_inference(
        self, items: List[MemoryItem], query: str, context: Optional[Dict[str, Any]]
    ) -> List[MemoryItem]:
        """Apply inference to derive additional facts."""
        if not self.inference_engine:
            return items

        try:
            # Extract facts from items
            facts = []
            for item in items:
                if isinstance(item.content, dict):
                    facts.append(item.content)
                else:
                    facts.append({"content": item.content})

            # Run inference
            inferred_facts = await self.inference_engine.infer(facts, query, context)

            # Convert inferred facts to memory items (temporary, not stored)
            for fact in inferred_facts:
                # Create temporary memory item
                temp_id = f"inferred_{MemoryUtils.generate_memory_id()}"
                metadata = MemoryMetadata(
                    retention_policy=MemoryRetentionPolicy.TRANSIENT,
                    custom_metadata={
                        "inferred": True,
                        "confidence": fact.get("confidence", 0.7),
                        "source": "inference",
                    }
                )
                
                inferred_item = MemoryItem(
                    memory_id=temp_id,
                    content=fact,
                    memory_type=MemoryType.SEMANTIC,
                    metadata=metadata,
                )
                
                items.append(inferred_item)

        except Exception as e:
            self.logger.warning(f"Inference failed: {str(e)}")

        return items

    async def _check_contradiction(self, item1: MemoryItem, item2: MemoryItem) -> bool:
        """Check if two memory items contradict each other."""
        # Simple implementation - in practice would be more sophisticated
        content1 = item1.content
        content2 = item2.content

        # If both are dictionaries with subject-predicate-object structure
        if (isinstance(content1, dict) and isinstance(content2, dict) and
            all(k in content1 for k in ["subject", "predicate", "object"]) and
            all(k in content2 for k in ["subject", "predicate", "object"])):
            
            # Same subject and predicate but different object could be contradiction
            if (content1["subject"] == content2["subject"] and 
                content1["predicate"] == content2["predicate"] and
                content1["object"] != content2["object"]):
                
                # Check if predicate implies uniqueness
                unique_predicates = ["is", "equals", "has_value", "has_age", "located_at"]
                if any(pred in content1["predicate"].lower() for pred in unique_predicates):
                    return True

        return False

    async def _mark_for_review(self, memory_id: str, reason: str) -> None:
        """Mark a memory for manual review."""
        memory = await self.retrieve(memory_id)
        if memory:
            memory.metadata.custom_metadata["review_required"] = True
            memory.metadata.custom_metadata["review_reason"] = reason
            await self.vector_store.store_item(memory)

    def _get_confidence_distribution(self) -> Dict[str, int]:
        """Get distribution of confidence scores."""
        distribution = {
            "high": 0,    # >= 0.8
            "medium": 0,  # 0.5 - 0.8
            "low": 0,     # < 0.5
        }

        for confidence in self._fact_confidence.values():
            if confidence >= 0.8:
                distribution["high"] += 1
            elif confidence >= 0.5:
                distribution["medium"] += 1
            else:
                distribution["low"] += 1

        return distribution

    def _get_max_hierarchy_depth(self) -> int:
        """Get maximum depth of concept hierarchies."""
        if not self._concept_hierarchy:
            return 0

        def get_depth(concept: str, visited: Set[str] = None) -> int:
            if visited is None:
                visited = set()
            
            if concept in visited:
                return 0
            
            visited.add(concept)
            
            parents = self._concept_hierarchy.get(concept, set())
            if not parents:
                return 1
            
            max_parent_depth = 0
            for parent in parents:
                depth = get_depth(parent, visited.copy())
                max_parent_depth = max(max_parent_depth, depth)
            
            return 1 + max_parent_depth

        max_depth = 0
        for concept in self._concept_hierarchy:
            depth = get_depth(concept)
            max_depth = max(max_depth, depth)

        return max_depth

    async def _validation_loop(self) -> None:
        """Background task for periodic fact validation."""
        while not self._shutdown_event.is_set():
            try:
                # Wait for validation interval
                await asyncio.sleep(self.config.validation_interval)

                if not self._shutdown_event.is_set():
                    self.logger.info("Running periodic fact validation...")
                    await self._validate_pending_facts()

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in validation loop: {str(e)}")
                await asyncio.sleep(3600)  # Retry after 1 hour on error

    async def _validate_pending_facts(self) -> None:
        """Validate facts that are pending validation."""
        pending_ids = [
            memory_id for memory_id, status in self._fact_validation_status.items()
            if status == "pending"
        ]

        if not pending_ids:
            return

        self.logger.info(f"Validating {len(pending_ids)} pending facts...")

        for memory_id in pending_ids[:10]:  # Limit batch size
            try:
                # Check consistency
                consistency = await self.check_consistency(memory_id)
                
                # Simple validation based on consistency
                is_valid = consistency["is_consistent"] and consistency["consistency_score"] > 0.7
                
                await self.validate_fact(memory_id, is_valid, consistency["consistency_score"])

            except Exception as e:
                self.logger.error(f"Error validating fact {memory_id}: {str(e)}")
