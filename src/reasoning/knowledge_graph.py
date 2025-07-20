"""
Advanced Knowledge Graph Engine for AI Assistant
Author: Drmusab
Last Modified: 2025-06-13 04:15:00 UTC

This module provides comprehensive knowledge graph capabilities for the AI assistant,
including entity-relationship modeling, graph reasoning, semantic search, knowledge
inference, graph neural networks, and seamless integration with core system components.
"""

from pathlib import Path
from typing import Optional, Dict, Any, List, Set, Callable, Type, Union, AsyncGenerator, TypeVar, Tuple
import asyncio
import threading
import time
import re
import json
import hashlib
import math
import pickle
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
from contextlib import asynccontextmanager
import uuid
import logging
import inspect
from collections import defaultdict, deque, Counter
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
import weakref
import networkx as nx
import numpy as np

# Core imports
from src.core.config.loader import ConfigLoader
from src.core.events.event_bus import EventBus
from src.core.events.event_types import (
    KnowledgeGraphUpdated, EntityAdded, EntityUpdated, EntityRemoved,
    RelationshipAdded, RelationshipUpdated, RelationshipRemoved,
    GraphQueryExecuted, GraphReasoningStarted, GraphReasoningCompleted,
    KnowledgeInferred, GraphIndexUpdated, GraphCompacted, GraphBackupCreated,
    SimilaritySearchPerformed, GraphTraversalStarted, GraphTraversalCompleted,
    ErrorOccurred, SystemStateChanged, ComponentHealthChanged
)
from src.core.error_handling import ErrorHandler, handle_exceptions
from src.core.dependency_injection import Container
from src.core.health_check import HealthCheck

# Memory integration
from src.memory.core_memory.memory_manager import MemoryManager
from src.memory.operations.context_manager import ContextManager
from src.memory.core_memory.memory_types import SemanticMemory, WorkingMemory
from src.memory.storage.vector_store import VectorStore

# Learning integration
from src.learning.continual_learning import ContinualLearner
from src.learning.feedback_processor import FeedbackProcessor
from src.learning.preference_learning import PreferenceLearner

# Processing integration
from src.processing.natural_language.language_chain import LanguageChain
from src.processing.natural_language.entity_extractor import EntityExtractor
from src.processing.natural_language.sentiment_analyzer import SentimentAnalyzer

# Assistant components
from src.assistant.session_manager import SessionManager
from src.assistant.component_manager import ComponentManager

# Integrations
from src.integrations.storage.database import DatabaseManager
from src.integrations.cache.redis_cache import RedisCache
from src.integrations.llm.model_router import ModelRouter

# Observability
from src.observability.monitoring.metrics import MetricsCollector
from src.observability.monitoring.tracing import TraceManager
from src.observability.logging.config import get_logger

# Type definitions
T = TypeVar('T')


class EntityType(Enum):
    """Types of entities in the knowledge graph."""
    PERSON = "person"
    ORGANIZATION = "organization"
    LOCATION = "location"
    EVENT = "event"
    CONCEPT = "concept"
    OBJECT = "object"
    DOCUMENT = "document"
    TOPIC = "topic"
    SKILL = "skill"
    TASK = "task"
    RESOURCE = "resource"
    SYSTEM = "system"
    ABSTRACT = "abstract"


class RelationshipType(Enum):
    """Types of relationships in the knowledge graph."""
    IS_A = "is_a"
    PART_OF = "part_of"
    CONTAINS = "contains"
    RELATED_TO = "related_to"
    CAUSES = "causes"
    ENABLES = "enables"
    REQUIRES = "requires"
    SIMILAR_TO = "similar_to"
    OPPOSITE_OF = "opposite_of"
    TEMPORAL_BEFORE = "temporal_before"
    TEMPORAL_AFTER = "temporal_after"
    SPATIAL_NEAR = "spatial_near"
    OWNS = "owns"
    CREATED_BY = "created_by"
    INFLUENCED_BY = "influenced_by"
    DEPENDS_ON = "depends_on"
    IMPLEMENTS = "implements"
    EXTENDS = "extends"


class GraphQueryType(Enum):
    """Types of graph queries."""
    ENTITY_LOOKUP = "entity_lookup"
    RELATIONSHIP_SEARCH = "relationship_search"
    PATH_FINDING = "path_finding"
    SUBGRAPH_EXTRACTION = "subgraph_extraction"
    SIMILARITY_SEARCH = "similarity_search"
    NEIGHBOR_DISCOVERY = "neighbor_discovery"
    COMMUNITY_DETECTION = "community_detection"
    CENTRALITY_ANALYSIS = "centrality_analysis"
    PATTERN_MATCHING = "pattern_matching"
    REASONING_CHAIN = "reasoning_chain"


class GraphReasoningMode(Enum):
    """Graph reasoning modes."""
    TRAVERSAL = "traversal"
    INFERENCE = "inference"
    ANALOGY = "analogy"
    COMPOSITION = "composition"
    PATTERN_COMPLETION = "pattern_completion"
    CAUSAL_REASONING = "causal_reasoning"
    TEMPORAL_REASONING = "temporal_reasoning"
    SPATIAL_REASONING = "spatial_reasoning"


class ConfidenceLevel(Enum):
    """Confidence levels for graph elements."""
    VERY_LOW = 0.1
    LOW = 0.3
    MEDIUM = 0.5
    HIGH = 0.7
    VERY_HIGH = 0.9
    CERTAIN = 1.0


@dataclass
class GraphEmbedding:
    """Vector embedding for graph elements."""
    vector: np.ndarray
    dimension: int
    model: str = "default"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def similarity(self, other: 'GraphEmbedding') -> float:
        """Calculate cosine similarity with another embedding."""
        if self.dimension != other.dimension:
            return 0.0
        
        dot_product = np.dot(self.vector, other.vector)
        magnitude_self = np.linalg.norm(self.vector)
        magnitude_other = np.linalg.norm(other.vector)
        
        if magnitude_self == 0 or magnitude_other == 0:
            return 0.0
        
        return dot_product / (magnitude_self * magnitude_other)


@dataclass
class GraphEntity:
    """Represents an entity in the knowledge graph."""
    entity_id: str
    entity_type: EntityType
    name: str
    description: Optional[str] = None
    
    # Properties and attributes
    properties: Dict[str, Any] = field(default_factory=dict)
    aliases: Set[str] = field(default_factory=set)
    tags: Set[str] = field(default_factory=set)
    
    # Temporal information
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    valid_from: Optional[datetime] = None
    valid_until: Optional[datetime] = None
    
    # Quality and confidence
    confidence: float = 1.0
    importance: float = 0.5
    quality_score: float = 1.0
    
    # Embedding and similarity
    embedding: Optional[GraphEmbedding] = None
    
    # Provenance and sources
    sources: List[str] = field(default_factory=list)
    created_by: Optional[str] = None
    
    # Statistics
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __str__(self) -> str:
        return f"{self.entity_type.value}:{self.name}"
    
    def is_valid_at(self, timestamp: datetime) -> bool:
        """Check if entity is valid at given timestamp."""
        if self.valid_from and timestamp < self.valid_from:
            return False
        if self.valid_until and timestamp > self.valid_until:
            return False
        return True
    
    def get_property(self, key: str, default: Any = None) -> Any:
        """Get entity property with default value."""
        return self.properties.get(key, default)
    
    def set_property(self, key: str, value: Any) -> None:
        """Set entity property and update timestamp."""
        self.properties[key] = value
        self.updated_at = datetime.now(timezone.utc)
    
    def add_alias(self, alias: str) -> None:
        """Add an alias for the entity."""
        self.aliases.add(alias)
        self.updated_at = datetime.now(timezone.utc)
    
    def add_tag(self, tag: str) -> None:
        """Add a tag to the entity."""
        self.tags.add(tag)
        self.updated_at = datetime.now(timezone.utc)


@dataclass
class GraphRelationship:
    """Represents a relationship in the knowledge graph."""
    relationship_id: str
    source_entity_id: str
    target_entity_id: str
    relationship_type: RelationshipType
    
    # Relationship properties
    properties: Dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0
    confidence: float = 1.0
    
    # Temporal information
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    valid_from: Optional[datetime] = None
    valid_until: Optional[datetime] = None
    
    # Quality and provenance
    quality_score: float = 1.0
    sources: List[str] = field(default_factory=list)
    created_by: Optional[str] = None
    
    # Statistics
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __str__(self) -> str:
        return f"{self.source_entity_id} --{self.relationship_type.value}--> {self.target_entity_id}"
    
    def is_valid_at(self, timestamp: datetime) -> bool:
        """Check if relationship is valid at given timestamp."""
        if self.valid_from and timestamp < self.valid_from:
            return False
        if self.valid_until and timestamp > self.valid_until:
            return False
        return True
    
    def is_bidirectional(self) -> bool:
        """Check if relationship type is bidirectional."""
        bidirectional_types = {
            RelationshipType.SIMILAR_TO,
            RelationshipType.RELATED_TO,
            RelationshipType.SPATIAL_NEAR
        }
        return self.relationship_type in bidirectional_types


@dataclass
class GraphPath:
    """Represents a path in the knowledge graph."""
    path_id: str
    entities: List[str]  # entity IDs
    relationships: List[str]  # relationship IDs
    path_type: str = "directed"
    total_weight: float = 0.0
    confidence: float = 1.0
    hops: int = 0
    
    def __post_init__(self):
        self.hops = len(self.relationships)
    
    def __str__(self) -> str:
        return f"Path({len(self.entities)} entities, {self.hops} hops)"


@dataclass
class GraphQuery:
    """Represents a query to the knowledge graph."""
    query_id: str
    query_type: GraphQueryType
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Constraints and filters
    entity_types: Optional[Set[EntityType]] = None
    relationship_types: Optional[Set[RelationshipType]] = None
    time_range: Optional[Tuple[datetime, datetime]] = None
    confidence_threshold: float = 0.0
    max_results: int = 100
    max_depth: int = 5
    
    # Context
    context: Dict[str, Any] = field(default_factory=dict)
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    
    # Execution
    timeout_seconds: float = 30.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class GraphQueryResult:
    """Result of a graph query."""
    query_id: str
    success: bool
    
    # Results
    entities: List[GraphEntity] = field(default_factory=list)
    relationships: List[GraphRelationship] = field(default_factory=list)
    paths: List[GraphPath] = field(default_factory=list)
    subgraphs: List[Dict[str, Any]] = field(default_factory=list)
    
    # Metadata
    execution_time: float = 0.0
    total_results: int = 0
    confidence: float = 0.0
    explanation: Optional[str] = None
    
    # Statistics
    entities_searched: int = 0
    relationships_traversed: int = 0
    cache_hits: int = 0
    
    # Context
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReasoningResult:
    """Result of graph reasoning operation."""
    reasoning_id: str
    success: bool
    mode: GraphReasoningMode
    
    # Inferred knowledge
    inferred_entities: List[GraphEntity] = field(default_factory=list)
    inferred_relationships: List[GraphRelationship] = field(default_factory=list)
    reasoning_paths: List[GraphPath] = field(default_factory=list)
    
    # Confidence and quality
    overall_confidence: float = 0.0
    reasoning_steps: List[Dict[str, Any]] = field(default_factory=list)
    
    # Performance
    execution_time: float = 0.0
    nodes_processed: int = 0
    
    # Explanation
    explanation: Optional[str] = None
    evidence: List[str] = field(default_factory=list)
    
    # Metadata
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)


class KnowledgeGraphError(Exception):
    """Custom exception for knowledge graph operations."""
    
    def __init__(self, message: str, error_code: Optional[str] = None, 
                 entity_id: Optional[str] = None, relationship_id: Optional[str] = None):
        super().__init__(message)
        self.error_code = error_code
        self.entity_id = entity_id
        self.relationship_id = relationship_id
        self.timestamp = datetime.now(timezone.utc)


class GraphIndex:
    """Provides indexing capabilities for fast graph operations."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        
        # Entity indices
        self.entity_by_type: Dict[EntityType, Set[str]] = defaultdict(set)
        self.entity_by_name: Dict[str, str] = {}  # name -> entity_id
        self.entity_by_alias: Dict[str, str] = {}  # alias -> entity_id
        self.entity_by_tag: Dict[str, Set[str]] = defaultdict(set)
        
        # Relationship indices
        self.relationships_by_type: Dict[RelationshipType, Set[str]] = defaultdict(set)
        self.outgoing_relationships: Dict[str, Set[str]] = defaultdict(set)  # entity_id -> relationship_ids
        self.incoming_relationships: Dict[str, Set[str]] = defaultdict(set)  # entity_id -> relationship_ids
        
        # Temporal indices
        self.entities_by_timeframe: Dict[str, Set[str]] = defaultdict(set)
        self.relationships_by_timeframe: Dict[str, Set[str]] = defaultdict(set)
        
        # Full-text search index
        self.text_index: Dict[str, Set[str]] = defaultdict(set)  # term -> entity_ids
        
        # Performance tracking
        self.index_stats = {
            'last_updated': datetime.now(timezone.utc),
            'total_updates': 0,
            'entities_indexed': 0,
            'relationships_indexed': 0
        }
    
    def index_entity(self, entity: GraphEntity) -> None:
        """Add entity to indices."""
        try:
            # Type index
            self.entity_by_type[entity.entity_type].add(entity.entity_id)
            
            # Name index
            self.entity_by_name[entity.name.lower()] = entity.entity_id
            
            # Alias index
            for alias in entity.aliases:
                self.entity_by_alias[alias.lower()] = entity.entity_id
            
            # Tag index
            for tag in entity.tags:
                self.entity_by_tag[tag].add(entity.entity_id)
            
            # Temporal index
            if entity.valid_from or entity.valid_until:
                timeframe_key = self._get_timeframe_key(entity.valid_from, entity.valid_until)
                self.entities_by_timeframe[timeframe_key].add(entity.entity_id)
            
            # Text index
            self._index_entity_text(entity)
            
            self.index_stats['entities_indexed'] += 1
            self.index_stats['total_updates'] += 1
            self.index_stats['last_updated'] = datetime.now(timezone.utc)
            
        except Exception as e:
            self.logger.error(f"Failed to index entity {entity.entity_id}: {str(e)}")
    
    def index_relationship(self, relationship: GraphRelationship) -> None:
        """Add relationship to indices."""
        try:
            # Type index
            self.relationships_by_type[relationship.relationship_type].add(relationship.relationship_id)
            
            # Entity-relationship index
            self.outgoing_relationships[relationship.source_entity_id].add(relationship.relationship_id)
            self.incoming_relationships[relationship.target_entity_id].add(relationship.relationship_id)
            
            # Temporal index
            if relationship.valid_from or relationship.valid_until:
                timeframe_key = self._get_timeframe_key(relationship.valid_from, relationship.valid_until)
                self.relationships_by_timeframe[timeframe_key].add(relationship.relationship_id)
            
            self.index_stats['relationships_indexed'] += 1
            self.index_stats['total_updates'] += 1
            self.index_stats['last_updated'] = datetime.now(timezone.utc)
            
        except Exception as e:
            self.logger.error(f"Failed to index relationship {relationship.relationship_id}: {str(e)}")
    
    def _index_entity_text(self, entity: GraphEntity) -> None:
        """Index entity text for full-text search."""
        # Index name
        for term in self._tokenize_text(entity.name):
            self.text_index[term].add(entity.entity_id)
        
        # Index description
        if entity.description:
            for term in self._tokenize_text(entity.description):
                self.text_index[term].add(entity.entity_id)
        
        # Index aliases
        for alias in entity.aliases:
            for term in self._tokenize_text(alias):
                self.text_index[term].add(entity.entity_id)
        
        # Index property values
        for value in entity.properties.values():
            if isinstance(value, str):
                for term in self._tokenize_text(value):
                    self.text_index[term].add(entity.entity_id)
    
    def _tokenize_text(self, text: str) -> List[str]:
        """Tokenize text for indexing."""
        # Simple tokenization - in practice would use more sophisticated methods
        return [term.lower().strip() for term in re.findall(r'\b\w+\b', text)]
    
    def _get_timeframe_key(self, valid_from: Optional[datetime], valid_until: Optional[datetime]) -> str:
        """Generate timeframe key for temporal indexing."""
        from_str = valid_from.strftime('%Y-%m') if valid_from else "earliest"
        until_str = valid_until.strftime('%Y-%m') if valid_until else "latest"
        return f"{from_str}:{until_str}"
    
    def search_entities_by_text(self, query: str, max_results: int = 100) -> Set[str]:
        """Search entities by text query."""
        query_terms = self._tokenize_text(query)
        if not query_terms:
            return set()
        
        # Start with entities containing the first term
        result_entities = self.text_index.get(query_terms[0], set()).copy()
        
        # Intersect with entities containing other terms
        for term in query_terms[1:]:
            result_entities.intersection_update(self.text_index.get(term, set()))
        
        # Limit results
        if len(result_entities) > max_results:
            return set(list(result_entities)[:max_results])
        
        return result_entities
    
    def find_entity_by_name(self, name: str) -> Optional[str]:
        """Find entity ID by exact name match."""
        return self.entity_by_name.get(name.lower())
    
    def find_entity_by_alias(self, alias: str) -> Optional[str]:
        """Find entity ID by alias."""
        return self.entity_by_alias.get(alias.lower())
    
    def get_neighbors(self, entity_id: str, relationship_types: Optional[Set[RelationshipType]] = None) -> Set[str]:
        """Get neighboring entity IDs."""
        neighbors = set()
        
        # Outgoing relationships
        for rel_id in self.outgoing_relationships.get(entity_id, set()):
            neighbors.add(rel_id)  # Would need to resolve to target entity
        
        # Incoming relationships
        for rel_id in self.incoming_relationships.get(entity_id, set()):
            neighbors.add(rel_id)  # Would need to resolve to source entity
        
        return neighbors
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get index statistics."""
        return {
            **self.index_stats,
            'entity_types': len(self.entity_by_type),
            'relationship_types': len(self.relationships_by_type),
            'text_terms': len(self.text_index),
            'timeframes': len(self.entities_by_timeframe)
        }


class GraphStorage:
    """Handles persistence of graph data."""
    
    def __init__(self, database: Optional[DatabaseManager] = None, 
                 vector_store: Optional[VectorStore] = None):
        self.database = database
        self.vector_store = vector_store
        self.logger = get_logger(__name__)
        
        # In-memory storage
        self.entities: Dict[str, GraphEntity] = {}
        self.relationships: Dict[str, GraphRelationship] = {}
        
        # Thread safety
        self.lock = threading.RLock()
    
    async def store_entity(self, entity: GraphEntity, persist: bool = True) -> None:
        """Store an entity."""
        with self.lock:
            self.entities[entity.entity_id] = entity
            
            if persist and self.database:
                try:
                    await self._persist_entity(entity)
                except Exception as e:
                    self.logger.error(f"Failed to persist entity {entity.entity_id}: {str(e)}")
            
            # Store embedding if available
            if entity.embedding and self.vector_store:
                try:
                    await self.vector_store.store(
                        entity.entity_id,
                        entity.embedding.vector,
                        metadata={
                            'entity_type': entity.entity_type.value,
                            'name': entity.name,
                            'created_at': entity.created_at.isoformat()
                        }
                    )
                except Exception as e:
                    self.logger.warning(f"Failed to store entity embedding: {str(e)}")
    
    async def store_relationship(self, relationship: GraphRelationship, persist: bool = True) -> None:
        """Store a relationship."""
        with self.lock:
            self.relationships[relationship.relationship_id] = relationship
            
            if persist and self.database:
                try:
                    await self._persist_relationship(relationship)
                except Exception as e:
                    self.logger.error(f"Failed to persist relationship {relationship.relationship_id}: {str(e)}")
    
    async def get_entity(self, entity_id: str) -> Optional[GraphEntity]:
        """Get an entity by ID."""
        with self.lock:
            entity = self.entities.get(entity_id)
            if entity:
                entity.access_count += 1
                entity.last_accessed = datetime.now(timezone.utc)
                return entity
            
            # Try to load from database
            if self.database:
                try:
                    entity = await self._load_entity(entity_id)
                    if entity:
                        self.entities[entity_id] = entity
                        return entity
                except Exception as e:
                    self.logger.warning(f"Failed to load entity {entity_id}: {str(e)}")
            
            return None
    
    async def get_relationship(self, relationship_id: str) -> Optional[GraphRelationship]:
        """Get a relationship by ID."""
        with self.lock:
            relationship = self.relationships.get(relationship_id)
            if relationship:
                relationship.access_count += 1
                relationship.last_accessed = datetime.now(timezone.utc)
                return relationship
            
            # Try to load from database
            if self.database:
                try:
                    relationship = await self._load_relationship(relationship_id)
                    if relationship:
                        self.relationships[relationship_id] = relationship
                        return relationship
                except Exception as e:
                    self.logger.warning(f"Failed to load relationship {relationship_id}: {str(e)}")
            
            return None
    
    async def delete_entity(self, entity_id: str) -> bool:
        """Delete an entity."""
        with self.lock:
            if entity_id in self.entities:
                del self.entities[entity_id]
                
                if self.database:
                    try:
                        await self._delete_entity_from_db(entity_id)
                    except Exception as e:
                        self.logger.error(f"Failed to delete entity from database: {str(e)}")
                
                if self.vector_store:
                    try:
                        await self.vector_store.delete(entity_id)
                    except Exception as e:
                        self.logger.warning(f"Failed to delete entity embedding: {str(e)}")
                
                return True
            
            return False
    
    async def delete_relationship(self, relationship_id: str) -> bool:
        """Delete a relationship."""
        with self.lock:
            if relationship_id in self.relationships:
                del self.relationships[relationship_id]
                
                if self.database:
                    try:
                        await self._delete_relationship_from_db(relationship_id)
                    except Exception as e:
                        self.logger.error(f"Failed to delete relationship from database: {str(e)}")
                
                return True
            
            return False
    
    async def get_all_entities(self) -> List[GraphEntity]:
        """Get all entities."""
        with self.lock:
            return list(self.entities.values())
    
    async def get_all_relationships(self) -> List[GraphRelationship]:
        """Get all relationships."""
        with self.lock:
            return list(self.relationships.values())
    
    async def _persist_entity(self, entity: GraphEntity) -> None:
        """Persist entity to database."""
        if not self.database:
            return
        
        entity_data = {
            'entity_id': entity.entity_id,
            'entity_type': entity.entity_type.value,
            'name': entity.name,
            'description': entity.description,
            'properties': json.dumps(entity.properties),
            'aliases': json.dumps(list(entity.aliases)),
            'tags': json.dumps(list(entity.tags)),
            'created_at': entity.created_at,
            'updated_at': entity.updated_at,
            'valid_from': entity.valid_from,
            'valid_until': entity.valid_until,
            'confidence': entity.confidence,
            'importance': entity.importance,
            'quality_score': entity.quality_score,
            'sources': json.dumps(entity.sources),
            'created_by': entity.created_by,
            'access_count': entity.access_count,
            'last_accessed': entity.last_accessed,
            'metadata': json.dumps(entity.metadata)
        }
        
        await self.database.execute(
            """
            INSERT OR REPLACE INTO graph_entities 
            (entity_id, entity_type, name, description, properties, aliases, tags,
             created_at, updated_at, valid_from, valid_until, confidence, importance,
             quality_score, sources, created_by, access_count, last_accessed, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            tuple(entity_data.values())
        )
    
    async def _persist_relationship(self, relationship: GraphRelationship) -> None:
        """Persist relationship to database."""
        if not self.database:
            return
        
        relationship_data = {
            'relationship_id': relationship.relationship_id,
            'source_entity_id': relationship.source_entity_id,
            'target_entity_id': relationship.target_entity_id,
            'relationship_type': relationship.relationship_type.value,
            'properties': json.dumps(relationship.properties),
            'weight': relationship.weight,
            'confidence': relationship.confidence,
            'created_at': relationship.created_at,
            'updated_at': relationship.updated_at,
            'valid_from': relationship.valid_from,
            'valid_until': relationship.valid_until,
            'quality_score': relationship.quality_score,
            'sources': json.dumps(relationship.sources),
            'created_by': relationship.created_by,
            'access_count': relationship.access_count,
            'last_accessed': relationship.last_accessed,
            'metadata': json.dumps(relationship.metadata)
        }
        
        await self.database.execute(
            """
            INSERT OR REPLACE INTO graph_relationships 
            (relationship_id, source_entity_id, target_entity_id, relationship_type,
             properties, weight, confidence, created_at, updated_at, valid_from,
             valid_until, quality_score, sources, created_by, access_count,
             last_accessed, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            tuple(relationship_data.values())
        )
    
    async def _load_entity(self, entity_id: str) -> Optional[GraphEntity]:
        """Load entity from database."""
        if not self.database:
            return None
        
        result = await self.database.fetch_one(
            "SELECT * FROM graph_entities WHERE entity_id = ?",
            (entity_id,)
        )
        
        if not result:
            return None
        
        # Convert database row to GraphEntity
        return GraphEntity(
            entity_id=result['entity_id'],
            entity_type=EntityType(result['entity_type']),
            name=result['name'],
            description=result['description'],
            properties=json.loads(result['properties'] or '{}'),
            aliases=set(json.loads(result['aliases'] or '[]')),
            tags=set(json.loads(result['tags'] or '[]')),
            created_at=result['created_at'],
            updated_at=result['updated_at'],
            valid_from=result['valid_from'],
            valid_until=result['valid_until'],
            confidence=result['confidence'],
            importance=result['importance'],
            quality_score=result['quality_score'],
            sources=json.loads(result['sources'] or '[]'),
            created_by=result['created_by'],
            access_count=result['access_count'],
            last_accessed=result['last_accessed'],
            metadata=json.loads(result['metadata'] or '{}')
        )
    
    async def _load_relationship(self, relationship_id: str) -> Optional[GraphRelationship]:
        """Load relationship from database."""
        if not self.database:
            return None
        
        result = await self.database.fetch_one(
            "SELECT * FROM graph_relationships WHERE relationship_id = ?",
            (relationship_id,)
        )
        
        if not result:
            return None
        
        # Convert database row to GraphRelationship
        return GraphRelationship(
            relationship_id=result['relationship_id'],
            source_entity_id=result['source_entity_id'],
            target_entity_id=result['target_entity_id'],
            relationship_type=RelationshipType(result['relationship_type']),
            properties=json.loads(result['properties'] or '{}'),
            weight=result['weight'],
            confidence=result['confidence'],
            created_at=result['created_at'],
            updated_at=result['updated_at'],
            valid_from=result['valid_from'],
            valid_until=result['valid_until'],
            quality_score=result['quality_score'],
            sources=json.loads(result['sources'] or '[]'),
            created_by=result['created_by'],
            access_count=result['access_count'],
            last_accessed=result['last_accessed'],
            metadata=json.loads(result['metadata'] or '{}')
        )
    
    async def _delete_entity_from_db(self, entity_id: str) -> None:
        """Delete entity from database."""
        if self.database:
            await self.database.execute(
                "DELETE FROM graph_entities WHERE entity_id = ?",
                (entity_id,)
            )
    
    async def _delete_relationship_from_db(self, relationship_id: str) -> None:
        """Delete relationship from database."""
        if self.database:
            await self.database.execute(
                "DELETE FROM graph_relationships WHERE relationship_id = ?",
                (relationship_id,)
            )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get storage statistics."""
        with self.lock:
            return {
                'entities_in_memory': len(self.entities),
                'relationships_in_memory': len(self.relationships),
                'memory_usage_estimate': self._estimate_memory_usage()
            }
    
    def _estimate_memory_usage(self) -> Dict[str, int]:
        """Estimate memory usage."""
        entity_size = sum(len(pickle.dumps(entity)) for entity in self.entities.values())
        relationship_size = sum(len(pickle.dumps(rel)) for rel in self.relationships.values())
        
        return {
            'entities_bytes': entity_size,
            'relationships_bytes': relationship_size,
            'total_bytes': entity_size + relationship_size
        }


class GraphQueryEngine:
    """Executes queries against the knowledge graph."""
    
    def __init__(self, storage: GraphStorage, index: GraphIndex):
        self.storage = storage
        self.index = index
        self.logger = get_logger(__name__)
        
        # Query optimization
        self.query_cache: Dict[str, GraphQueryResult] = {}
        self.cache_ttl = 3600  # 1 hour
        
        # Performance tracking
        self.query_stats = defaultdict(int)
    
    async def execute_query(self, query: GraphQuery) -> GraphQueryResult:
        """Execute a graph query."""
        start_time = time.time()
        
        # Check cache
        cache_key = self._generate_cache_key(query)
        if cache_key in self.query_cache:
            cached_result = self.query_cache[cache_key]
            cached_time = cached_result.metadata.get('cache_time', 0)
            if time.time() - cached_time < self.cache_ttl:
                cached_result.cache_hits += 1
                return cached_result
        
        # Execute query based on type
        result = GraphQueryResult(query_id=query.query_id, success=False)
        
        try:
            if query.query_type == GraphQueryType.ENTITY_LOOKUP:
                result = await self._execute_entity_lookup(query)
            elif query.query_type == GraphQueryType.RELATIONSHIP_SEARCH:
                result = await self._execute_relationship_search(query)
            elif query.query_type == GraphQueryType.PATH_FINDING:
                result = await self._execute_path_finding(query)
            elif query.query_type == GraphQueryType.SUBGRAPH_EXTRACTION:
                result = await self._execute_subgraph_extraction(query)
            elif query.query_type == GraphQueryType.SIMILARITY_SEARCH:
                result = await self._execute_similarity_search(query)
            elif query.query_type == GraphQueryType.NEIGHBOR_DISCOVERY:
                result = await self._execute_neighbor_discovery(query)
            else:
                raise KnowledgeGraphError(f"Unsupported query type: {query.query_type}")
            
            result.execution_time = time.time() - start_time
            result.success = True
            
            # Cache result
            result.metadata['cache_time'] = time.time()
            self.query_cache[cache_key] = result
            
            # Update statistics
            self.query_stats[query.query_type.value] += 1
            
        except Exception as e:
            result.execution_time = time.time() - start_time
            result.metadata['error'] = str(e)
            self.logger.error(f"Query execution failed: {str(e)}")
        
        return result
    
    async def _execute_entity_lookup(self, query: GraphQuery) -> GraphQueryResult:
        """Execute entity lookup query."""
        result = GraphQueryResult(query_id=query.query_id, success=True)
        
        search_term = query.parameters.get('search_term', '')
        exact_match = query.parameters.get('exact_match', False)
        
        if exact_match:
            # Exact name or alias match
            entity_id = self.index.find_entity_by_name(search_term)
            if not entity_id:
                entity_id = self.index.find_entity_by_alias(search_term)
            
            if entity_id:
                entity = await self.storage.get_entity(entity_id)
                if entity:
                    result.entities.append(entity)
        else:
            # Text search
            entity_ids = self.index.search_entities_by_text(search_term, query.max_results)
            
            for entity_id in entity_ids:
                entity = await self.storage.get_entity(entity_id)
                if entity and self._passes_filters(entity, query):
                    result.entities.append(entity)
                    if len(result.entities) >= query.max_results:
                        break
        
        result.total_results = len(result.entities)
        result.entities_searched = len(result.entities)
        
        return result
    
    async def _execute_relationship_search(self, query: GraphQuery) -> GraphQueryResult:
        """Execute relationship search query."""
        result = GraphQueryResult(query_id=query.query_id, success=True)
        
        source_entity_id = query.parameters.get('source_entity_id')
        target_entity_id = query.parameters.get('target_entity_id')
        relationship_types = query.relationship_types
        
        # Get all relationships
        all_relationships = await self.storage.get_all_relationships()
        
        for relationship in all_relationships:
            # Apply filters
            if source_entity_id and relationship.source_entity_id != source_entity_id:
                continue
            if target_entity_id and relationship.target_entity_id != target_entity_id:
                continue
            if relationship_types and relationship.relationship_type not in relationship_types:
                continue
            if relationship.confidence < query.confidence_threshold:
                continue
            
            result.relationships.append(relationship)
            if len(result.relationships) >= query.max_results:
                break
        
        result.total_results = len(result.relationships)
        result.relationships_traversed = len(result.relationships)
        
        return result
    
    async def _execute_path_finding(self, query: GraphQuery) -> GraphQueryResult:
        """Execute path finding query."""
        result = GraphQueryResult(query_id=query.query_id, success=True)
        
        source_id = query.parameters.get('source_entity_id')
        target_id = query.parameters.get('target_entity_id')
        max_hops = min(query.max_depth, query.parameters.get('max_hops', 5))
        
        if not source_id or not target_id:
            return result
        
        # Use breadth-first search to find paths
        paths = await self._find_paths_bfs(source_id, target_id, max_hops, query)
        
        for path_entities, path_relationships in paths:
            path = GraphPath(
                path_id=str(uuid.uuid4()),
                entities=path_entities,
                relationships=path_relationships
            )
            result.paths.append(path)
            
            if len(result.paths) >= query.max_results:
                break
        
        result.total_results = len(result.paths)
        
        return result
    
    async def _find_paths_bfs(self, source_id: str, target_id: str, max_hops: int, 
                             query: GraphQuery) -> List[Tuple[List[str], List[str]]]:
        """Find paths using breadth-first search."""
        if source_id == target_id:
            return [([source_id], [])]
        
        queue = deque([(source_id, [source_id], [])])  # (current_entity, path_entities, path_relationships)
        visited = set()
        paths = []
        
        while queue and len(paths) < query.max_results:
            current_entity, path_entities, path_relationships = queue.popleft()
            
            if len(path_entities) > max_hops + 1:
                continue
            
            if current_entity in visited:
                continue
            visited.add(current_entity)
            
            # Get outgoing relationships
            outgoing_rel_ids = self.index.outgoing_relationships.get(current_entity, set())
            
            for rel_id in outgoing_rel_ids:
                relationship = await self.storage.get_relationship(rel_id)
                if not relationship:
                    continue
                
                # Apply relationship type filter
                if query.relationship_types and relationship.relationship_type not in query.relationship_types:
                    continue
                
                next_entity = relationship.target_entity_id
                
                if next_entity == target_id:
                    # Found target
                    final_path_entities = path_entities + [next_entity]
                    final_path_relationships = path_relationships + [rel_id]
                    paths.append((final_path_entities, final_path_relationships))
                elif next_entity not in visited and len(path_entities) < max_hops + 1:
                    # Continue search
                    new_path_entities = path_entities + [next_entity]
                    new_path_relationships = path_relationships + [rel_id]
                    queue.append((next_entity, new_path_entities, new_path_relationships))
        
        return paths
    
    async def _execute_subgraph_extraction(self, query: GraphQuery) -> GraphQueryResult:
        """Execute subgraph extraction query."""
        result = GraphQueryResult(query_id=query.query_id, success=True)
        
        center_entity_id = query.parameters.get('center_entity_id')
        radius = min(query.max_depth, query.parameters.get('radius', 2))
        
        if not center_entity_id:
            return result
        
        # Extract subgraph around center entity
        visited_entities = set()
        visited_relationships = set()
        
        await self._extract_subgraph_recursive(
            center_entity_id, radius, visited_entities, visited_relationships, query
        )
        
        # Collect entities and relationships
        for entity_id in visited_entities:
            entity = await self.storage.get_entity(entity_id)
            if entity:
                result.entities.append(entity)
        
        for rel_id in visited_relationships:
            relationship = await self.storage.get_relationship(rel_id)
            if relationship:
                result.relationships.append(relationship)
        
        result.total_results = len(result.entities) + len(result.relationships)
        
        return result
    
    async def _extract_subgraph_recursive(self, entity_id: str, remaining_radius: int,
                                         visited_entities: Set[str], visited_relationships: Set[str],
                                         query: GraphQuery) -> None:
        """Recursively extract subgraph."""
        if remaining_radius < 0 or entity_id in visited_entities:
            return
        
        visited_entities.add(entity_id)
        
        if remaining_radius == 0:
            return
        
        # Get all relationships for this entity
        outgoing_rels = self.index.outgoing_relationships.get(entity_id, set())
        incoming_rels = self.index.incoming_relationships.get(entity_id, set())
        
        for rel_id in outgoing_rels.union(incoming_rels):
            if rel_id in visited_relationships:
                continue
            
            relationship = await self.storage.get_relationship(rel_id)
            if not relationship:
                continue
            
            # Apply filters
            if query.relationship_types and relationship.relationship_type not in query.relationship_types:
                continue
            
            visited_relationships.add(rel_id)
            
            # Recurse to connected entities
            if relationship.source_entity_id == entity_id:
                await self._extract_subgraph_recursive(
                    relationship.target_entity_id, remaining_radius - 1,
                    visited_entities, visited_relationships, query
                )
            else:
                await self._extract_subgraph_recursive(
                    relationship.source_entity_id, remaining_radius - 1,
                    visited_entities, visited_relationships, query
                )
    
    async def _execute_similarity_search(self, query: GraphQuery) -> GraphQueryResult:
        """Execute similarity search query."""
        result = GraphQueryResult(query_id=query.query_id, success=True)
        
        reference_entity_id = query.parameters.get('reference_entity_id')
        similarity_threshold = query.parameters.get('similarity_threshold', 0.7)
        
        if not reference_entity_id:
            return result
        
        reference_entity = await self.storage.get_entity(reference_entity_id)
        if not reference_entity or not reference_entity.embedding:
            return result
        
        # Compare with all other entities
        all_entities = await self.storage.get_all_entities()
        similarities = []
        
        for entity in all_entities:
            if entity.entity_id == reference_entity_id or not entity.embedding:
                continue
            
            similarity = reference_entity.embedding.similarity(entity.embedding)
            if similarity >= similarity_threshold:
                similarities.append((entity, similarity))
        
        # Sort by similarity and take top results
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        for entity, similarity in similarities[:query.max_results]:
            result.entities.append(entity)
            result.metadata[f'similarity_{entity.entity_id}'] = similarity
        
        result.total_results = len(result.entities)
        result.confidence = sum(similarities[i][1] for i in range(min(len(similarities), query.max_results))) / max(len(result.entities), 1)
        
        return result
    
    async def _execute_neighbor_discovery(self, query: GraphQuery) -> GraphQueryResult:
        """Execute neighbor discovery query."""
        result = GraphQueryResult(query_id=query.query_id, success=True)
        
        entity_id = query.parameters.get('entity_id')
        include_incoming = query.parameters.get('include_incoming', True)
        include_outgoing = query.parameters.get('include_outgoing', True)
        
        if not entity_id:
            return result
        
        neighbor_entities = set()
        relationships_found = []
        
        # Outgoing relationships
        if include_outgoing:
            outgoing_rels = self.index.outgoing_relationships.get(entity_id, set())
            for rel_id in outgoing_rels:
                relationship = await self.storage.get_relationship(rel_id)
                if relationship and self._passes_relationship_filters(relationship, query):
                    relationships_found.append(relationship)
                    neighbor_entities.add(relationship.target_entity_id)
        
        # Incoming relationships
        if include_incoming:
            incoming_rels = self.index.incoming_relationships.get(entity_id, set())
            for rel_id in incoming_rels:
                relationship = await self.storage.get_relationship(rel_id)
                if relationship and self._passes_relationship_filters(relationship, query):
                    relationships_found.append(relationship)
                    neighbor_entities.add(relationship.source_entity_id)
        
        # Get neighbor entities
        for neighbor_id in neighbor_entities:
            if len(result.entities) >= query.max_results:
                break
            
            entity = await self.storage.get_entity(neighbor_id)
            if entity and self._passes_filters(entity, query):
                result.entities.append(entity)
        
        result.relationships = relationships_found[:query.max_results]
        result.total_results = len(result.entities)
        
        return result
    
    def _passes_filters(self, entity: GraphEntity, query: GraphQuery) -> bool:
        """Check if entity passes query filters."""
        # Entity type filter
        if query.entity_types and entity.entity_type not in query.entity_types:
            return False
        
        # Confidence filter
        if entity.confidence < query.confidence_threshold:
            return False
        
        # Time range filter
        if query.time_range:
            start_time, end_time = query.time_range
            if not entity.is_valid_at(start_time) and not entity.is_valid_at(end_time):
                return False
        
        return True
    
    def _passes_relationship_filters(self, relationship: GraphRelationship, query: GraphQuery) -> bool:
        """Check if relationship passes query filters."""
        # Relationship type filter
        if query.relationship_types and relationship.relationship_type not in query.relationship_types:
            return False
        
        # Confidence filter
        if relationship.confidence < query.confidence_threshold:
            return False
        
        # Time range filter
        if query.time_range:
            start_time, end_time = query.time_range
            if not relationship.is_valid_at(start_time) and not relationship.is_valid_at(end_time):
                return False
        
        return True
    
    def _generate_cache_key(self, query: GraphQuery) -> str:
        """Generate cache key for query."""
        key_parts = [
            query.query_type.value,
            str(sorted(query.parameters.items())),
            str(sorted(query.entity_types)) if query.entity_types else "",
            str(sorted(query.relationship_types)) if query.relationship_types else "",
            str(query.confidence_threshold),
            str(query.max_results),
            str(query.max_depth)
        ]
        
        key_string = "|".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()


class GraphReasoningEngine:
    """Performs reasoning operations on the knowledge graph."""
    
    def __init__(self, storage: GraphStorage, query_engine: GraphQueryEngine):
        self.storage = storage
        self.query_engine = query_engine
        self.logger = get_logger(__name__)
    
    async def perform_reasoning(self, mode: GraphReasoningMode, parameters: Dict[str, Any],
                               context: Optional[Dict[str, Any]] = None) -> ReasoningResult:
        """Perform graph reasoning operation."""
        start_time = time.time()
        context = context or {}
        
        result = ReasoningResult(
            reasoning_id=str(uuid.uuid4()),
            success=False,
            mode=mode
        )
        
        try:
            if mode == GraphReasoningMode.TRAVERSAL:
                await self._perform_traversal_reasoning(parameters, result, context)
            elif mode == GraphReasoningMode.INFERENCE:
                await self._perform_inference_reasoning(parameters, result, context)
            elif mode == GraphReasoningMode.ANALOGY:
                await self._perform_analogy_reasoning(parameters, result, context)
            elif mode == GraphReasoningMode.PATTERN_COMPLETION:
                await self._perform_pattern_completion(parameters, result, context)
            elif mode == GraphReasoningMode.CAUSAL_REASONING:
                await self._perform_causal_reasoning(parameters, result, context)
            else:
                raise KnowledgeGraphError(f"Unsupported reasoning mode: {mode}")
            
            result.success = True
            result.execution_time = time.time() - start_time
            
        except Exception as e:
            result.execution_time = time.time() - start_time
            result.metadata['error'] = str(e)
            self.logger.error(f"Reasoning failed: {str(e)}")
        
        return result
    
    async def _perform_traversal_reasoning(self, parameters: Dict[str, Any], 
                                          result: ReasoningResult, context: Dict[str, Any]) -> None:
        """Perform traversal-based reasoning."""
        start_entity_id = parameters.get('start_entity_id')
        reasoning_pattern = parameters.get('pattern', 'chain')  # chain, tree, cycle
        max_depth = parameters.get('max_depth', 5)
        
        if not start_entity_id:
            return
        
        # Perform traversal to find reasoning chains
        visited = set()
        reasoning_paths = []
        
        await self._traverse_for_reasoning(
            start_entity_id, [], [], 0, max_depth, visited, reasoning_paths, context
        )
        
        # Convert paths to reasoning results
        for path_entities, path_relationships in reasoning_paths:
            path = GraphPath(
                path_id=str(uuid.uuid4()),
                entities=path_entities,
                relationships=path_relationships
            )
            result.reasoning_paths.append(path)
        
        result.nodes_processed = len(visited)
    
    async def _traverse_for_reasoning(self, current_entity: str, path_entities: List[str],
                                     path_relationships: List[str], depth: int, max_depth: int,
                                     visited: Set[str], reasoning_paths: List[Tuple[List[str], List[str]]],
                                     context: Dict[str, Any]) -> None:
        """Recursively traverse graph for reasoning."""
        if depth >= max_depth or current_entity in visited:
            return
        
        visited.add(current_entity)
        current_path_entities = path_entities + [current_entity]
        
        # Check if current path represents interesting reasoning
        if depth > 1 and self._is_interesting_reasoning_path(current_path_entities, path_relationships):
            reasoning_paths.append((current_path_entities, path_relationships))
        
        # Continue traversal
        outgoing_rels = self.query_engine.index.outgoing_relationships.get(current_entity, set())
        
        for rel_id in outgoing_rels:
            relationship = await self.storage.get_relationship(rel_id)
            if relationship:
                next_entity = relationship.target_entity_id
                new_path_relationships = path_relationships + [rel_id]
                
                await self._traverse_for_reasoning(
                    next_entity, current_path_entities, new_path_relationships,
                    depth + 1, max_depth, visited, reasoning_paths, context
                )
    
    def _is_interesting_reasoning_path(self, entities: List[str], relationships: List[str]) -> bool:
        """Determine if a path represents interesting reasoning."""
        # Simple heuristic - longer paths with varied relationship types
        if len(entities) < 3:
            return False
        
        # Check for relationship type diversity
        rel_types = set()
        for rel_id in relationships:
            # Would need to look up relationship type
            rel_types.add(rel_id[:5])  # Simple approximation
        
        return len(rel_types) > 1
    
    async def _perform_inference_reasoning(self, parameters: Dict[str, Any],
                                          result: ReasoningResult, context: Dict[str, Any]) -> None:
        """Perform inference-based reasoning."""
        premises = parameters.get('premises', [])
        inference_rules = parameters.get('rules', ['transitivity', 'symmetry'])
        
        # Apply inference rules to derive new knowledge
        for rule in inference_rules:
            if rule == 'transitivity':
                await self._apply_transitivity_inference(result, context)
            elif rule == 'symmetry':
                await self._apply_symmetry_inference(result, context)
            elif rule == 'inverse':
                await self._apply_inverse_inference(result, context)
    
    async def _apply_transitivity_inference(self, result: ReasoningResult, context: Dict[str, Any]) -> None:
        """Apply transitivity inference rule."""
        # Find chains A -> B -> C to infer A -> C
        all_relationships = await self.storage.get_all_relationships()
        
        # Group relationships by type
        rel_by_type = defaultdict(list)
        for rel in all_relationships:
            rel_by_type[rel.relationship_type].append(rel)
        
        # Apply transitivity for specific relationship types
        transitive_types = {RelationshipType.IS_A, RelationshipType.PART_OF, RelationshipType.CONTAINS}
        
        for rel_type in transitive_types:
            relationships = rel_by_type[rel_type]
            
            # Build adjacency map
            adjacency = defaultdict(set)
            for rel in relationships:
                adjacency[rel.source_entity_id].add(rel.target_entity_id)
            
            # Find transitive relationships
            for a in adjacency:
                for b in adjacency[a]:
                    for c in adjacency[b]:
                        # Check if A -> C already exists
                        if c not in adjacency[a]:
                            # Infer A -> C
                            inferred_rel = GraphRelationship(
                                relationship_id=str(uuid.uuid4()),
                                source_entity_id=a,
                                target_entity_id=c,
                                relationship_type=rel_type,
                                confidence=0.7,  # Lower confidence for inferred relationships
                                metadata={'inferred': True, 'rule': 'transitivity'}
                            )
                            result.inferred_relationships.append(inferred_rel)
                            
                            # Add reasoning step
                            result.reasoning_steps.append({
                                'rule': 'transitivity',
                                'premise1': f"{a} -> {b}",
                                'premise2': f"{b} -> {c}",
                                'conclusion': f"{a} -> {c}",
                                'confidence': inferred_rel.confidence
                            })
    
    async def _apply_symmetry_inference(self, result: ReasoningResult, context: Dict[str, Any]) -> None:
        """Apply symmetry inference rule."""
        all_relationships = await self.storage.get_all_relationships()
        
        # Symmetric relationship types
        symmetric_types = {RelationshipType.SIMILAR_TO, RelationshipType.SPATIAL_NEAR}
        
        for rel in all_relationships:
            if rel.relationship_type in symmetric_types:
                # Check if reverse relationship exists
                reverse_exists = any(
                    r.source_entity_id == rel.target_entity_id and 
                    r.target_entity_id == rel.source_entity_id and
                    r.relationship_type == rel.relationship_type
                    for r in all_relationships
                )
                
                if not reverse_exists:
                    # Infer reverse relationship
                    inferred_rel = GraphRelationship(
                        relationship_id=str(uuid.uuid4()),
                        source_entity_id=rel.target_entity_id,
                        target_entity_id=rel.source_entity_id,
                        relationship_type=rel.relationship_type,
                        confidence=rel.confidence * 0.9,  # Slightly lower confidence
                        metadata={'inferred': True, 'rule': 'symmetry'}
                    )
                    result.inferred_relationships.append(inferred_rel)
                    
                    result.reasoning_steps.append({
                        'rule': 'symmetry',
                        'premise': str(rel),
                        'conclusion': str(inferred_rel),
                        'confidence': inferred_rel.confidence
                    })
    
    async def _apply_inverse_inference(self, result: ReasoningResult, context: Dict[str, Any]) -> None:
        """Apply inverse relationship inference rules to enhance reasoning results."""
