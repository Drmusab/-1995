"""
Graph-based Memory Storage System
Author: Drmusab
Last Modified: 2025-07-05 09:56:54 UTC

This module provides a graph-based storage system for the AI assistant's memory,
enabling the representation and traversal of relationships between memory items.
It supports multiple storage backends, complex relationship queries, and efficient
retrieval of connected memories for contextual understanding.
"""

from pathlib import Path
from typing import Optional, Dict, Any, List, Set, Tuple, Union, Iterator, Callable, TypeVar, Generic
import asyncio
import time
import json
import pickle
import uuid
import os
from datetime import datetime, timezone
from enum import Enum
from collections import defaultdict, deque
import logging
import traceback
import heapq
import math
import functools
import hashlib

# Core imports
from src.core.config.loader import ConfigLoader
from src.core.events.event_bus import EventBus
from src.core.events.event_types import (
    MemoryItemStored, MemoryItemRetrieved, MemoryItemUpdated, MemoryItemDeleted,
    GraphNodeAdded, GraphNodeUpdated, GraphNodeDeleted,
    GraphEdgeAdded, GraphEdgeUpdated, GraphEdgeDeleted,
    GraphTraversalStarted, GraphTraversalCompleted,
    GraphStoreBackupStarted, GraphStoreBackupCompleted,
    GraphStoreRestoreStarted, GraphStoreRestoreCompleted,
    ErrorOccurred
)
from src.core.error_handling import ErrorHandler, handle_exceptions
from src.core.dependency_injection import Container
from src.core.health_check import HealthCheck

# Memory system imports
from src.memory.core_memory.base_memory import (
    BaseMemoryStore, MemoryItem, MemoryType, MemoryStorageType,
    MemoryError, MemoryNotFoundError, SimpleMemoryQuery, memory_operation_span
)

# Integration imports
from src.integrations.storage.database import DatabaseManager
from src.integrations.cache.redis_cache import RedisCache

# Observability
from src.observability.monitoring.metrics import MetricsCollector
from src.observability.monitoring.tracing import TraceManager
from src.observability.logging.config import get_logger

# Optional dependencies - graph databases and visualization
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

try:
    import neo4j
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False


class RelationshipType(Enum):
    """Types of relationships between memory nodes."""
    IS_A = "is_a"                    # Taxonomy/hierarchy relationship
    HAS_A = "has_a"                  # Composition relationship
    PART_OF = "part_of"              # Membership relationship
    OCCURRED_BEFORE = "before"       # Temporal relationship (before)
    OCCURRED_AFTER = "after"         # Temporal relationship (after)
    OCCURRED_DURING = "during"       # Temporal relationship (during)
    LOCATED_AT = "located_at"        # Spatial relationship
    CAUSES = "causes"                # Causal relationship
    RELATED_TO = "related_to"        # Generic relationship
    SIMILAR_TO = "similar_to"        # Similarity relationship
    OPPOSITE_OF = "opposite_of"      # Contrast relationship
    INSTANCE_OF = "instance_of"      # Type-instance relationship
    REFERENCED_BY = "referenced_by"  # Reference relationship
    DEPENDS_ON = "depends_on"        # Dependency relationship
    INTERACTS_WITH = "interacts_with" # Interaction relationship
    CUSTOM = "custom"                # Custom relationship type


class GraphNodeType(Enum):
    """Types of nodes in the memory graph."""
    MEMORY = "memory"                # Memory item node
    CONCEPT = "concept"              # Concept node
    ENTITY = "entity"                # Entity node
    EVENT = "event"                  # Event node
    LOCATION = "location"            # Location node
    TIME = "time"                    # Time node
    USER = "user"                    # User node
    SESSION = "session"              # Session node
    CONTEXT = "context"              # Context node
    CUSTOM = "custom"                # Custom node type


class GraphNode:
    """
    Represents a node in the memory graph.
    
    A node can represent a memory item, a concept, an entity, or any other
    element that can participate in relationships.
    """
    
    def __init__(
        self,
        node_id: str,
        node_type: GraphNodeType,
        properties: Dict[str, Any],
        labels: Optional[Set[str]] = None,
        memory_id: Optional[str] = None
    ):
        """
        Initialize a graph node.
        
        Args:
            node_id: Node identifier
            node_type: Node type
            properties: Node properties
            labels: Node labels (for categorization)
            memory_id: Associated memory ID (if node represents a memory item)
        """
        self.node_id = node_id
        self.node_type = node_type
        self.properties = properties
        self.labels = labels or set()
        self.memory_id = memory_id
        self.created_at = datetime.now(timezone.utc)
        self.updated_at = self.created_at
    
    def update(self, properties: Dict[str, Any], labels: Optional[Set[str]] = None) -> None:
        """
        Update node properties and labels.
        
        Args:
            properties: Updated properties
            labels: Updated labels
        """
        self.properties.update(properties)
        if labels:
            self.labels.update(labels)
        self.updated_at = datetime.now(timezone.utc)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary representation."""
        return {
            "node_id": self.node_id,
            "node_type": self.node_type.value,
            "properties": self.properties,
            "labels": list(self.labels),
            "memory_id": self.memory_id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GraphNode':
        """Create node from dictionary representation."""
        node = cls(
            node_id=data["node_id"],
            node_type=GraphNodeType(data["node_type"]),
            properties=data["properties"],
            labels=set(data.get("labels", [])),
            memory_id=data.get("memory_id")
        )
        node.created_at = datetime.fromisoformat(data["created_at"])
        node.updated_at = datetime.fromisoformat(data["updated_at"])
        return node
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GraphNode):
            return False
        return self.node_id == other.node_id
    
    def __hash__(self) -> int:
        return hash(self.node_id)


class GraphEdge:
    """
    Represents an edge (relationship) in the memory graph.
    
    An edge connects two nodes and represents a specific type of relationship
    between them, with optional properties for additional context.
    """
    
    def __init__(
        self,
        edge_id: str,
        source_id: str,
        target_id: str,
        relationship: RelationshipType,
        properties: Dict[str, Any],
        weight: float = 1.0,
        bidirectional: bool = False
    ):
        """
        Initialize a graph edge.
        
        Args:
            edge_id: Edge identifier
            source_id: Source node ID
            target_id: Target node ID
            relationship: Relationship type
            properties: Edge properties
            weight: Edge weight (strength of relationship)
            bidirectional: Whether the relationship is bidirectional
        """
        self.edge_id = edge_id
        self.source_id = source_id
        self.target_id = target_id
        self.relationship = relationship
        self.properties = properties
        self.weight = weight
        self.bidirectional = bidirectional
        self.created_at = datetime.now(timezone.utc)
        self.updated_at = self.created_at
    
    def update(self, properties: Dict[str, Any], weight: Optional[float] = None) -> None:
        """
        Update edge properties and weight.
        
        Args:
            properties: Updated properties
            weight: Updated weight
        """
        self.properties.update(properties)
        if weight is not None:
            self.weight = weight
        self.updated_at = datetime.now(timezone.utc)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert edge to dictionary representation."""
        return {
            "edge_id": self.edge_id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relationship": self.relationship.value,
            "properties": self.properties,
            "weight": self.weight,
            "bidirectional": self.bidirectional,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GraphEdge':
        """Create edge from dictionary representation."""
        edge = cls(
            edge_id=data["edge_id"],
            source_id=data["source_id"],
            target_id=data["target_id"],
            relationship=RelationshipType(data["relationship"]),
            properties=data["properties"],
            weight=data["weight"],
            bidirectional=data["bidirectional"]
        )
        edge.created_at = datetime.fromisoformat(data["created_at"])
        edge.updated_at = datetime.fromisoformat(data["updated_at"])
        return edge
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GraphEdge):
            return False
        return self.edge_id == other.edge_id
    
    def __hash__(self) -> int:
        return hash(self.edge_id)


class GraphPathNode:
    """
    Represents a node in a graph traversal path.
    
    This class is used for path representation in graph traversal results,
    including the node and the edge that led to it.
    """
    
    def __init__(
        self,
        node: GraphNode,
        incoming_edge: Optional[GraphEdge] = None,
        distance: int = 0
    ):
        """
        Initialize a path node.
        
        Args:
            node: The graph node
            incoming_edge: The edge that led to this node
            distance: Distance from the start node
        """
        self.node = node
        self.incoming_edge = incoming_edge
        self.distance = distance
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert path node to dictionary representation."""
        return {
            "node": self.node.to_dict(),
            "incoming_edge": self.incoming_edge.to_dict() if self.incoming_edge else None,
            "distance": self.distance
        }


class GraphPath:
    """
    Represents a path in the memory graph.
    
    A path is a sequence of nodes connected by edges, representing a traversal
    through the graph from a start node to an end node.
    """
    
    def __init__(self, nodes: List[GraphPathNode]):
        """
        Initialize a graph path.
        
        Args:
            nodes: Sequence of path nodes
        """
        self.nodes = nodes
    
    @property
    def start_node(self) -> Optional[GraphNode]:
        """Get the start node of the path."""
        if not self.nodes:
            return None
        return self.nodes[0].node
    
    @property
    def end_node(self) -> Optional[GraphNode]:
        """Get the end node of the path."""
        if not self.nodes:
            return None
        return self.nodes[-1].node
    
    @property
    def length(self) -> int:
        """Get the length of the path (number of edges)."""
        return len(self.nodes) - 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert path to dictionary representation."""
        return {
            "nodes": [node.to_dict() for node in self.nodes],
            "length": self.length
        }


class GraphQuery:
    """
    Represents a query for the memory graph.
    
    This class defines parameters for graph traversal and filtering operations,
    enabling complex queries across the memory graph.
    """
    
    def __init__(
        self,
        start_node_id: Optional[str] = None,
        node_types: Optional[List[GraphNodeType]] = None,
        node_labels: Optional[List[str]] = None,
        relationships: Optional[List[RelationshipType]] = None,
        max_depth: int = 3,
        limit: int = 10,
        property_filters: Optional[Dict[str, Any]] = None,
        include_properties: bool = True,
        sort_by: Optional[str] = None,
        sort_order: str = "desc",
        time_range: Optional[Tuple[datetime, datetime]] = None
    ):
        """
        Initialize a graph query.
        
        Args:
            start_node_id: ID of the starting node for traversal
            node_types: Types of nodes to include in results
            node_labels: Labels of nodes to include in results
            relationships: Types of relationships to traverse
            max_depth: Maximum traversal depth
            limit: Maximum number of results
            property_filters: Filters to apply to node properties
            include_properties: Whether to include properties in results
            sort_by: Property to sort results by
            sort_order: Sort order ("asc" or "desc")
            time_range: Time range filter for node creation time
        """
        self.start_node_id = start_node_id
        self.node_types = node_types
        self.node_labels = node_labels
        self.relationships = relationships
        self.max_depth = max_depth
        self.limit = limit
        self.property_filters = property_filters or {}
        self.include_properties = include_properties
        self.sort_by = sort_by
        self.sort_order = sort_order
        self.time_range = time_range


class MemoryGraphConfig:
    """Configuration settings for memory graph store."""
    
    def __init__(self, config_loader: ConfigLoader):
        """
        Initialize memory graph configuration.
        
        Args:
            config_loader: Configuration loader
        """
        graph_config = config_loader.get("memory.graph_store", {})
        
        # Storage settings
        self.storage_type = graph_config.get("storage_type", "memory")  # memory, file, neo4j
        self.file_path = graph_config.get("file_path", "data/cache/memory_graph")
        self.neo4j_uri = graph_config.get("neo4j_uri", "bolt://localhost:7687")
        self.neo4j_user = graph_config.get("neo4j_user", "neo4j")
        self.neo4j_password = graph_config.get("neo4j_password", "password")
        self.database_name = graph_config.get("database_name", "memory_graph")
        
        # Cache settings
        self.cache_enabled = graph_config.get("cache_enabled", True)
        self.cache_size = graph_config.get("cache_size", 1000)
        self.cache_ttl = graph_config.get("cache_ttl", 3600)  # seconds
        
        # Performance settings
        self.batch_size = graph_config.get("batch_size", 100)
        self.use_index = graph_config.get("use_index", True)
        self.auto_cleanup = graph_config.get("auto_cleanup", True)
        self.cleanup_threshold = graph_config.get("cleanup_threshold", 10000)  # nodes
        
        # Traversal settings
        self.default_max_depth = graph_config.get("default_max_depth", 3)
        self.default_weight_threshold = graph_config.get("default_weight_threshold", 0.1)
        self.default_limit = graph_config.get("default_limit", 50)
        
        # Backup settings
        self.backup_enabled = graph_config.get("backup_enabled", True)
        self.backup_interval = graph_config.get("backup_interval", 86400)  # seconds (daily)
        self.backup_path = graph_config.get("backup_path", "data/backups/memory_graph")
        self.max_backups = graph_config.get("max_backups", 5)


class MemoryGraphStore(BaseMemoryStore):
    """
    Graph-based memory storage system.
    
    This class provides an implementation of BaseMemoryStore that uses a graph
    structure to represent relationships between memory items. It supports
    relationship-based traversal, pattern matching, and contextual memory retrieval.
    
    Features:
    - Storage and traversal of semantic relationships between memories
    - Support for different relationship types (causal, temporal, hierarchical, etc.)
    - Complex pattern matching and pathfinding
    - Context retrieval for better understanding of memory items
    - Graph visualization and export capabilities
    - Integration with vector store for hybrid memory retrieval
    """
    
    def __init__(self, container: Container):
        """
        Initialize memory graph store.
        
        Args:
            container: Dependency injection container
        """
        self.container = container
        self.logger = get_logger(__name__)
        
        # Load configuration
        self.config_loader = container.get(ConfigLoader)
        self.config = MemoryGraphConfig(self.config_loader)
        
        # Core integrations
        self.event_bus = container.get(EventBus)
        self.error_handler = container.get(ErrorHandler)
        self.health_check = container.get(HealthCheck)
        
        # Optional integrations
        try:
            self.database = container.get(DatabaseManager)
        except Exception:
            self.logger.warning("DatabaseManager not available, using file-based storage")
            self.database = None
        
        try:
            self.redis_cache = container.get(RedisCache)
        except Exception:
            self.logger.info("RedisCache not available, using in-memory cache")
            self.redis_cache = None
        
        # Monitoring
        try:
            self.metrics = container.get(MetricsCollector)
            self.tracer = container.get(TraceManager)
        except Exception:
            self.logger.warning("Monitoring components not available")
            self.metrics = None
            self.tracer = None
        
        # Initialize storage
        self._nodes: Dict[str, GraphNode] = {}
        self._edges: Dict[str, GraphEdge] = {}
        self._memory_to_node: Dict[str, str] = {}  # memory_id -> node_id
        
        # Indexes for fast access
        self._outgoing_edges: Dict[str, List[str]] = defaultdict(list)  # node_id -> list of outgoing edge_ids
        self._incoming_edges: Dict[str, List[str]] = defaultdict(list)  # node_id -> list of incoming edge_ids
        self._node_type_index: Dict[GraphNodeType, Set[str]] = defaultdict(set)  # node_type -> set of node_ids
        self._node_label_index: Dict[str, Set[str]] = defaultdict(set)  # label -> set of node_ids
        self._relationship_index: Dict[RelationshipType, Set[str]] = defaultdict(set)  # relationship -> set of edge_ids
        
        # Cache
        self._path_cache: Dict[str, GraphPath] = {}
        self._query_cache: Dict[str, List[GraphNode]] = {}
        self._max_cache_size = self.config.cache_size
        
        # Neo4j client (if available)
        self._neo4j_driver = None
        if self.config.storage_type == 'neo4j' and NEO4J_AVAILABLE:
            try:
                self._neo4j_driver = neo4j.GraphDatabase.driver(
                    self.config.neo4j_uri,
                    auth=(self.config.neo4j_user, self.config.neo4j_password)
                )
                self.logger.info("Connected to Neo4j database")
            except Exception as e:
                self.logger.error(f"Failed to connect to Neo4j: {str(e)}")
                self.logger.info("Falling back to in-memory storage")
                self.config.storage_type = 'memory'
        
        # NetworkX graph (if available)
        self._nx_graph = None
        if NETWORKX_AVAILABLE:
            self._nx_graph = nx.MultiDiGraph()
            self.logger.info("NetworkX graph initialized")
        
        # Register health check
        self.health_check.register_component("memory_graph", self._health_check_callback)
        
        self.logger.info("MemoryGraphStore initialized")

    async def initialize(self) -> None:
        """Initialize the graph store."""
        try:
            # Load data from storage
            if self.config.storage_type == 'file':
                await self._load_from_file()
            elif self.config.storage_type == 'neo4j' and self._neo4j_driver:
                await self._load_from_neo4j()
            
            # Register metrics
            if self.metrics:
                self.metrics.register_counter("memory_graph_operations_total")
                self.metrics.register_counter("memory_graph_nodes_added")
                self.metrics.register_counter("memory_graph_edges_added")
                self.metrics.register_counter("memory_graph_traversals")
                self.metrics.register_histogram("memory_graph_operation_duration_seconds")
                self.metrics.register_histogram("memory_graph_traversal_time_seconds")
                self.metrics.register_gauge("memory_graph_node_count")
                self.metrics.register_gauge("memory_graph_edge_count")
            
            # Start background tasks
            if self.config.backup_enabled:
                asyncio.create_task(self._backup_loop())
            
            # Rebuild NetworkX graph
            if NETWORKX_AVAILABLE and self._nx_graph:
                self._rebuild_networkx_graph()
            
            self.logger.info(f"MemoryGraphStore initialized with {len(self._nodes)} nodes and {len(self._edges)} edges")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize memory graph: {str(e)}")
            traceback.print_exc()
            await self.event_bus.emit(ErrorOccurred(
                component="memory_graph",
                error_type=type(e).__name__,
                error_message=str(e),
                severity="critical"
            ))

    async def store_item(self, item: MemoryItem) -> None:
        """
        Store a memory item as a node in the graph.
        
        Args:
            item: Memory item to store
        """
        start_time = time.time()
        
        try:
            # Check if memory already exists
            if item.memory_id in self._memory_to_node:
                # Update existing node
                node_id = self._memory_to_node[item.memory_id]
                node = self._nodes[node_id]
                
                # Update properties
                properties = {
                    "content_type": type(item.content).__name__,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "memory_type": item.memory_type.value,
                    "importance": item.metadata.importance if item.metadata else 0.5
                }
                
                # Add tags as labels
                labels = set()
                if item.metadata and item.metadata.tags:
                    labels.update(item.metadata.tags)
                
                # Update node
                node.update(properties, labels)
                
                # If using Neo4j, update in database
                if self.config.storage_type == 'neo4j' and self._neo4j_driver:
                    await self._update_node_in_neo4j(node)
                
                # Update in NetworkX graph
                if NETWORKX_AVAILABLE and self._nx_graph:
                    self._nx_graph.nodes[node_id].update(node.properties)
                
                # Emit event
                await self.event_bus.emit(GraphNodeUpdated(
                    node_id=node_id,
                    node_type=node.node_type.value,
                    associated_memory=item.memory_id
                ))
                
                # Update relationships if present
                if item.relationships:
                    await self._update_relationships(node_id, item.relationships)
                
                return
            
            # Create new node
            node_id = str(uuid.uuid4())
            
            # Prepare properties
            properties = {
                "content_summary": self._get_content_summary(item.content),
                "content_type": type(item.content).__name__,
                "owner_id": item.owner_id,
                "session_id": item.session_id,
                "context_id": item.context_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "memory_type": item.memory_type.value,
                "importance": item.metadata.importance if item.metadata else 0.5
            }
            
            # Add tags as labels
            labels = set()
            if item.metadata and item.metadata.tags:
                labels.update(item.metadata.tags)
            
            # Create node
            node = GraphNode(
                node_id=node_id,
                node_type=GraphNodeType.MEMORY,
                properties=properties,
                labels=labels,
                memory_id=item.memory_id
            )
            
            # Store node
            self._nodes[node_id] = node
            self._memory_to_node[item.memory_id] = node_id
            self._node_type_index[GraphNodeType.MEMORY].add(node_id)
            
            # Update label index
            for label in labels:
                self._node_label_index[label].add(node_id)
            
            # If using Neo4j, store in database
            if self.config.storage_type == 'neo4j' and self._neo4j_driver:
                await self._store_node_in_neo4j(node)
            
            # Add to NetworkX graph
            if NETWORKX_AVAILABLE and self._nx_graph:
                self._nx_graph.add_node(
                    node_id,
                    **node.properties,
                    labels=list(labels),
                    memory_id=item.memory_id,
                    node_type=node.node_type.value
                )
            
            # Create relationships if present
            if item.relationships:
                await self._create_relationships(node_id, item.relationships)
            
            # If file storage, persist changes
            if self.config.storage_type == 'file':
                await self._persist_to_file()
            
            # Emit event
            await self.event_bus.emit(GraphNodeAdded(
                node_id=node_id,
                node_type=GraphNodeType.MEMORY.value,
                associated_memory=item.memory_id
            ))
            
            # Update metrics
            if self.metrics:
                self.metrics.increment("memory_graph_operations_total")
                self.metrics.increment("memory_graph_nodes_added")
                self.metrics.record("memory_graph_operation_duration_seconds", time.time() - start_time)
                self.metrics.gauge("memory_graph_node_count", len(self._nodes))
            
        except Exception as e:
            self.logger.error(f"Failed to store memory in graph: {str(e)}")
            traceback.print_exc()
            await self.event_bus.emit(ErrorOccurred(
                component="memory_graph",
                error_type=type(e).__name__,
                error_message=str(e),
                severity="error"
            ))
            raise MemoryError(f"Failed to store memory in graph: {str(e)}")

    async def get_item(self, memory_id: str) -> Optional[MemoryItem]:
        """
        This method is not directly supported by the graph store.
        Use get_node_by_memory_id() instead.
        
        Args:
            memory_id: Memory identifier
            
        Returns:
            None (MemoryGraphStore doesn't store full memory items)
        """
        self.logger.warning(
            "MemoryGraphStore.get_item() is not supported. "
            "Use get_node_by_memory_id() to get the graph node."
        )
        return None

    async def get_node_by_memory_id(self, memory_id: str) -> Optional[GraphNode]:
        """
        Get a graph node by memory ID.
        
        Args:
            memory_id: Memory identifier
            
        Returns:
            Graph node or None if not found
        """
        if memory_id not in self._memory_to_node:
            return None
        
        node_id = self._memory_to_node[memory_id]
        return self._nodes.get(node_id)

    async def get_node(self, node_id: str) -> Optional[GraphNode]:
        """
        Get a graph node by node ID.
        
        Args:
            node_id: Node identifier
            
        Returns:
            Graph node or None if not found
        """
        return self._nodes.get(node_id)

    async def get_edge(self, edge_id: str) -> Optional[GraphEdge]:
        """
        Get a graph edge by edge ID.
        
        Args:
            edge_id: Edge identifier
            
        Returns:
            Graph edge or None if not found
        """
        return self._edges.get(edge_id)

    async def add_node(
        self,
        node_type: GraphNodeType,
        properties: Dict[str, Any],
        labels: Optional[Set[str]] = None,
        memory_id: Optional[str] = None
    ) -> str:
        """
        Add a new node to the graph.
        
        Args:
            node_type: Node type
            properties: Node properties
            labels: Node labels
            memory_id: Associated memory ID
            
        Returns:
            Node ID
        """
        start_time = time.time()
        
        try:
            # Create node ID
            node_id = str(uuid.uuid4())
            
            # Create node
            node = GraphNode(
                node_id=node_id,
                node_type=node_type,
                properties=properties,
                labels=labels,
                memory_id=memory_id
            )
            
            # Store node
            self._nodes[node_id] = node
            self._node_type_index[node_type].add(node_id)
            
            # Update memory mapping if applicable
            if memory_id:
                self._memory_to_node[memory_id] = node_id
            
            # Update label index
            if labels:
                for label in labels:
                    self._node_label_index[label].add(node_id)
            
            # If using Neo4j, store in database
            if self.config.storage_type == 'neo4j' and self._neo4j_driver:
                await self._store_node_in_neo4j(node)
            
            # Add to NetworkX graph
            if NETWORKX_AVAILABLE and self._nx_graph:
                self._nx_graph.add_node(
                    node_id,
                    **node.properties,
                    labels=list(labels) if labels else [],
                    memory_id=memory_id,
                    node_type=node_type.value
                )
            
            # If file storage, persist changes
            if self.config.storage_type == 'file':
                await self._persist_to_file()
            
            # Emit event
            await self.event_bus.emit(GraphNodeAdded(
                node_id=node_id,
                node_type=node_type.value,
                associated_memory=memory_id
            ))
            
            # Update metrics
            if self.metrics:
                self.metrics.increment("memory_graph_operations_total")
                self.metrics.increment("memory_graph_nodes_added")
                self.metrics.record("memory_graph_operation_duration_seconds", time.time() - start_time)
                self.metrics.gauge("memory_graph_node_count", len(self._nodes))
            
            return node_id
            
        except Exception as e:
            self.logger.error(f"Failed to add node to graph: {str(e)}")
            traceback.print_exc()
            await self.event_bus.emit(ErrorOccurred(
                component="memory_graph",
                error_type=type(e).__name__,
                error_message=str(e),
                severity="error"
            ))
            raise MemoryError(f"Failed to add node to graph: {str(e)}")

    async def update_node(
        self,
        node_id: str,
        properties: Dict[str, Any],
        labels: Optional[Set[str]] = None
    ) -> bool:
        """
        Update an existing node.
        
        Args:
            node_id: Node identifier
            properties: Updated properties
            labels: Updated labels
            
        Returns:
            True if successful
        """
        start_time = time.time()
        
        try:
            # Check if node exists
            if node_id not in self._nodes:
                return False
            
            node = self._nodes[node_id]
            
            # Update node
            node.update(properties, labels)
            
            # Update label index if labels provided
            if labels:
                for label in labels:
                    self._node_label_index[label].add(node_id)
            
            # If using Neo4j, update in database
            if self.config.storage_type == 'neo4j' and self._neo4j_driver:
                await self._update_node_in_neo4j(node)
            
            # Update in NetworkX graph
            if NETWORKX_AVAILABLE and self._nx_graph:
                self._nx_graph.nodes[node_id].update(node.properties)
                if labels:
                    self._nx_graph.nodes[node_id]['labels'] = list(
                        set(self._nx_graph.nodes[node_id].get('labels', [])) | labels
                    )
            
            # If file storage, persist changes
            if self.config.storage_type == 'file':
                await self._persist_to_file()
            
            # Emit event
            await self.event_bus.emit(GraphNodeUpdated(
                node_id=node_id,
                node_type=node.node_type.value,
                associated_memory=node.memory_id
            ))
            
            # Update metrics
            if self.metrics:
                self.metrics.increment("memory_graph_operations_total")
                self.metrics.record("memory_graph_operation_duration_seconds", time.time() - start_time)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update node {node_id}: {str(e)}")
            return False

    async def delete_node(self, node_id: str) -> bool:
        """
        Delete a node and all its edges.
        
        Args:
            node_id: Node identifier
            
        Returns:
            True if successful
        """
        start_time = time.time()
        
        try:
            # Check if node exists
            if node_id not in self._nodes:
                return False
            
            node = self._nodes[node_id]
            
            # Delete all connected edges
            outgoing_edges = self._outgoing_edges.get(node_id, []).copy()
            for edge_id in outgoing_edges:
                await self.delete_edge(edge_id)
            
            incoming_edges = self._incoming_edges.get(node_id, []).copy()
            for edge_id in incoming_edges:
                await self.delete_edge(edge_id)
            
            # Remove from indexes
            self._node_type_index[node.node_type].discard(node_id)
            
            for label in node.labels:
                self._node_label_index[label].discard(node_id)
            
            # Remove memory mapping if applicable
            if node.memory_id and node.memory_id in self._memory_to_node:
                del self._memory_to_node[node.memory_id]
            
            # Delete from main storage
            del self._nodes[node_id]
            
            # If node has outgoing or incoming edges, clean up
            if node_id in self._outgoing_edges:
                del self._outgoing_edges[node_id]
            
            if node_id in self._incoming_edges:
                del self._incoming_edges[node_id]
            
            # If using Neo4j, delete from database
            if self.config.storage_type == 'neo4j' and self._neo4j_driver:
                await self._delete_node_from_neo4j(node_id)
            
            # Remove from NetworkX graph
            if NETWORKX_AVAILABLE and self._nx_graph and self._nx_graph.has_node(node_id):
                self._nx_graph.remove_node(node_id)
            
            # If file storage, persist changes
            if self.config.storage_type == 'file':
                await self._persist_to_file()
            
            # Clear path cache
            self._path_cache.clear()
            self._query_cache.clear()
            
            # Emit event
            await self.event_bus.emit(GraphNodeDeleted(
                node_id=node_id,
                node_type=node.node_type.value,
                associated_memory=node.memory_id
            ))
            
            # Update metrics
            if self.metrics:
                self.metrics.increment("memory_graph_operations_total")
                self.metrics.record("memory_graph_operation_duration_seconds", time.time() - start_time)
                self.metrics.gauge("memory_graph_node_count", len(self._nodes))
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete node {node_id}: {str(e)}")
            return False

    async def add_edge(
        self,
        source_id: str,
        target_id: str,
        relationship: RelationshipType,
        properties: Dict[str, Any] = None,
        weight: float = 1.0,
        bidirectional: bool = False
    ) -> str:
        """
        Add an edge between two nodes.
        
        Args:
            source_id: Source node ID
            target_id: Target node ID
            relationship: Relationship type
            properties: Edge properties
            weight: Edge weight
            bidirectional: Whether the relationship is bidirectional
            
        Returns:
            Edge ID
        """
        start_time = time.time()
        
        try:
            # Check if nodes exist
            if source_id not in self._nodes or target_id not in self._nodes:
                raise MemoryError("Source or target node does not exist")
            
            # Create edge ID
            edge_id = str(uuid.uuid4())
            
            # Create edge
            edge = GraphEdge(
                edge_id=edge_id,
                source_id=source_id,
                target_id=target_id,
                relationship=relationship,
                properties=properties or {},
                weight=weight,
                bidirectional=bidirectional
            )
            
            # Store edge
            self._edges[edge_id] = edge
            self._outgoing_edges[source_id].append(edge_id)
            self._incoming_edges[target_id].append(edge_id)
            self._relationship_index[relationship].add(edge_id)
            
            # If using Neo4j, store in database
            if self.config.storage_type == 'neo4j' and self._neo4j_driver:
                await self._store_edge_in_neo4j(edge)
            
            # Add to NetworkX graph
            if NETWORKX_AVAILABLE and self._nx_graph:
                self._nx_graph.add_edge(
                    source_id,
                    target_id,
                    key=edge_id,
                    relationship=relationship.value,
                    weight=weight,
                    bidirectional=bidirectional,
                    **edge.properties
                )
                
                # If bidirectional, add reverse edge in NetworkX
                if bidirectional:
                    self._nx_graph.add_edge(
                        target_id,
                        source_id,
                        key=f"{edge_id}_reverse",
                        relationship=relationship.value,
                        weight=weight,
                        bidirectional=bidirectional,
                        **edge.properties
                    )
            
            # If bidirectional, create reverse relationship for querying
            if bidirectional:
                # Just update indexes, don't create a new edge object
                self._outgoing_edges[target_id].append(edge_id)
                self._incoming_edges[source_id].append(edge_id)
            
            # If file storage, persist changes
            if self.config.storage_type == 'file':
                await self._persist_to_file()
            
            # Clear path cache
            self._path_cache.clear()
            self._query_cache.clear()
            
            # Emit event
            await self.event_bus.emit(GraphEdgeAdded(
                edge_id=edge_id,
                source_id=source_id,
                target_id=target_id,
                relationship=relationship.value
            ))
            
            # Update metrics
            if self.metrics:
                self.metrics.increment("memory_graph_operations_total")
                self.metrics.increment("memory_graph_edges_added")
                self.metrics.record("memory_graph_operation_duration_seconds", time.time() - start_time)
                self.metrics.gauge("memory_graph_edge_count", len(self._edges))
            
            return edge_id
            
        except Exception as e:
            self.logger.error(f"Failed to add edge: {str(e)}")
            traceback.print_exc()
            await self.event_bus.emit(ErrorOccurred(
                component="memory_graph",
                error_type=type(e).__name__,
                error_message=str(e),
                severity="error"
            ))
            raise MemoryError(f"Failed to add edge: {str(e)}")

    async def update_edge(
        self,
        edge_id: str,
        properties: Dict[str, Any],
        weight: Optional[float] = None
    ) -> bool:
        """
        Update an existing edge.
        
        Args:
            edge_id: Edge identifier
            properties: Updated properties
            weight: Updated weight
            
        Returns:
            True if successful
        """
        start_time = time.time()
        
        try:
            # Check if edge exists
            if edge_id not in self._edges:
                return False
            
            edge = self._edges[edge_id]
            
            # Update edge
            edge.update(properties, weight)
            
            # If using Neo4j, update in database
            if self.config.storage_type == 'neo4j' and self._neo4j_driver:
                await self._update_edge_in_neo4j(edge)
            
            # Update in NetworkX graph
            if NETWORKX_AVAILABLE and self._nx_graph:
                for _, _, edge_data in self._nx_graph.edges(data=True, keys=True):
                    if edge_data.get('key') == edge_id:
                        edge_data.update(edge.properties)
                        if weight is not None:
                            edge_data['weight'] = weight
                
                # Update reverse edge if bidirectional
                if edge.bidirectional:
                    for _, _, edge_data in self._nx_graph.edges(data=True, keys=True):
                        if edge_data.get('key') == f"{edge_id}_reverse":
                            edge_data.update(edge.properties)
                            if weight is not None:
                                edge_data['weight'] = weight
            
            # If file storage, persist changes
            if self.config.storage_type == 'file':
                await self._persist_to_file()
            
            # Clear path cache
            self._path_cache.clear()
            self._query_cache.clear()
            
            # Emit event
            await self.event_bus.emit(GraphEdgeUpdated(
                edge_id=edge_id,
                source_id=edge.source_id,
                target_id=edge.target_id,
                relationship=edge.relationship.value
            ))
            
            # Update metrics
            if self.metrics:
                self.metrics.increment("memory_graph_operations_total")
                self.metrics.record("memory_graph_operation_duration_seconds", time.time() - start_time)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update edge {edge_id}: {str(e)}")
            return False

    async def delete_edge(self, edge_id: str) -> bool:
        """
        Delete an edge.
        
        Args:
            edge_id: Edge identifier
            
        Returns:
            True if successful
        """
        start_time = time.time()
        
        try:
            # Check if edge exists
            if edge_id not in self._edges:
                return False
            
            edge = self._edges[edge_id]
            
            # Remove from indexes
            self._outgoing_edges[edge.source_id].remove(edge_id)
            self._incoming_edges[edge.target_id].remove(edge_id)
            self._relationship_index[edge.relationship].discard(edge_id)
            
            # If bidirectional, remove reverse links
            if edge.bidirectional:
                if edge_id in self._outgoing_edges[edge.target_id]:
                    self._outgoing_edges[edge.target_id].remove(edge_id)
                if edge_id in self._incoming_edges[edge.source_id]:
                    self._incoming_edges[edge.source_id].remove(edge_id)
            
            # Delete from main storage
            del self._edges[edge_id]
            
            # If using Neo4j, delete from database
            if self.config.storage_type == 'neo4j' and self._neo4j_driver:
                await self._delete_edge_from_neo4j(edge_id)
            
            # Remove from NetworkX graph
            if NETWORKX_AVAILABLE and self._nx_graph:
                # Find and remove the edge
                for u, v, k in list(self._nx_graph.edges(keys=True)):
                    if k == edge_id:
                        self._nx_graph.remove_edge(u, v, k)
                
                # Remove reverse edge if bidirectional
                if edge.bidirectional:
                    for u, v, k in list(self._nx_graph.edges(keys=True)):
                        if k == f"{edge_id}_reverse":
                            self._nx_graph.remove_edge(u, v, k)
            
            # If file storage, persist changes
            if self.config.storage_type == 'file':
                await self._persist_to_file()
            
            # Clear path cache
            self._path_cache.clear()
            self._query_cache.clear()
            
            # Emit event
            await self.event_bus.emit(GraphEdgeDeleted(
                edge_id=edge_id,
                source_id=edge.source_id,
                target_id=edge.target_id,
                relationship=edge.relationship.value
            ))
            
            # Update metrics
            if self.metrics:
                self.metrics.increment("memory_graph_operations_total")
                self.metrics.record("memory_graph_operation_duration_seconds", time.time() - start_time)
                self.metrics.gauge("memory_graph_edge_count", len(self._edges))
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete edge {edge_id}: {str(e)}")
            return False

    async def get_neighbors(
        self,
        node_id: str,
        direction: str = "both",
        relationships: Optional[List[RelationshipType]] = None,
        node_types: Optional[List[GraphNodeType]] = None
    ) -> List[Tuple[GraphNode, GraphEdge]]:
        """
        Get neighbors of a node.
        
        Args:
            node_id: Node identifier
            direction: Direction of relationships ("outgoing", "incoming", "both")
            relationships: Filter by relationship types
            node_types: Filter by node types
            
        Returns:
            List of (neighbor_node, connecting_edge) tuples
        """
        try:
            # Check if node exists
            if node_id not in self._nodes:
                return []
            
            result = []
            
            # Get outgoing edges
            if direction in ["outgoing", "both"]:
                for edge_id in self._outgoing_edges.get(node_id, []):
                    edge = self._edges[edge_id]
                    
                    # Skip if not matching relationship filter
                    if relationships and edge.relationship not in relationships:
                        continue
                    
                    target_node = self._nodes.get(edge.target_id)
                    if not target_node:
                        continue
                    
                    # Skip if not matching node type filter
                    if node_types and target_node.node_type not in node_types:
                        continue
                    
                    result.append((target_node, edge))
            
            # Get incoming edges
            if direction in ["incoming", "both"]:
                for edge_id in self._incoming_edges.get(node_id, []):
                    edge = self._edges[edge_id]
                    
                    # Skip if not matching relationship filter
                    if relationships and edge.relationship not in relationships:
                        continue
                    
                    source_node = self._nodes.get(edge.source_id)
                    if not source_node:
                        continue
                    
                    # Skip if not matching node type filter
                    if node_types and source_node.node_type not in node_types:
                        continue
                    
                    result.append((source_node, edge))
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to get neighbors for node {node_id}: {str(e)}")
            return []

    async def find_path(
        self,
        start_node_id: str,
        end_node_id: str,
        max_depth: int = 5,
        relationships: Optional[List[RelationshipType]] = None,
        min_weight: float = 0.0
    ) -> Optional[GraphPath]:
        """
        Find a path between two nodes.
        
        Args:
            start_node_id: Start node identifier
            end_node_id: End node identifier
            max_depth: Maximum path depth
            relationships: Filter by relationship types
            min_weight: Minimum edge weight
            
        Returns:
            Path if found, None otherwise
        """
        start_time = time.time()
        
        try:
            # Check if nodes exist
            if start_node_id not in self._nodes or end_node_id not in self._nodes:
                return None
            
            # Check cache
            cache_key = f"{start_node_id}_{end_node_id}_{max_depth}_{min_weight}"
            if relationships:
                cache_key += "_" + "_".join(sorted(r.value for r in relationships))
                
            if cache_key in self._path_cache:
                return self._path_cache[cache_key]
            
            # Emit event
            await self.event_bus.emit(GraphTraversalStarted(
                start_node_id=start_node_id,
                max_depth=max_depth,
                traversal_type="path"
            ))
            
            # Use NetworkX if available
            if NETWORKX_AVAILABLE and self._nx_graph:
                try:
                    # Create a view of the graph with filtered edges
                    view = nx.MultiDiGraph()
                    for u, v, data in self._nx_graph.edges(data=True):
                        if 'weight' not in data or data['weight'] < min_weight:
                            continue
                        if relationships and RelationshipType(data.get('relationship', 'related_to')) not in relationships:
                            continue
                        view.add_edge(u, v, **data)
                    
                    # Find shortest path
                    path = nx.shortest_path(view, start_node_id, end_node_id, weight='weight')
                    
                    # Convert to GraphPath
                    path_nodes = []
                    prev_node_id = None
                    for i, node_id in enumerate(path):
                        node = self._nodes[node_id]
                        incoming_edge = None
                        
                        if prev_node_id:
                            # Find edge between prev_node and current node
                            for edge_id in self._outgoing_edges.get(prev_node_id, []):
                                edge = self._edges[edge_id]
                                if edge.target_id == node_id:
                                    incoming_edge = edge
                                    break
                        
                        path_nodes.append(GraphPathNode(
                            node=node,
                            incoming_edge=incoming_edge,
                            distance=i
                        ))
                        prev_node_id = node_id
                    
                    result = GraphPath(path_nodes)
                    
                    # Update cache
                    self._path_cache[cache_key] = result
                    
                    # Emit event
                    await self.event_bus.emit(GraphTraversalCompleted(
                        start_node_id=start_node_id,
                        node_count=len(path),
                        traversal_time=time.time() - start_time
                    ))
                    
                    # Update metrics
                    if self.metrics:
                        self.metrics.increment("memory_graph_traversals")
                        self.metrics.record("memory_graph_traversal_time_seconds", time.time() - start_time)
                    
                    return result
                
                except nx.NetworkXNoPath:
                    return None
            
            # Fallback to custom BFS implementation
            visited = set()
            queue = deque([(start_node_id, [])])  # (node_id, path_so_far)
            
            while queue:
                current_id, path = queue.popleft()
                
                # Skip if already visited or max depth reached
                if current_id in visited or len(path) > max_depth:
                    continue
                
                visited.add(current_id)
                
                # Check if we reached the target
                if current_id == end_node_id:
                    # Reconstruct path
                    path_nodes = []
                    
                    for i, (node_id, edge) in enumerate([(start_node_id, None)] + path):
                        node = self._nodes[node_id]
                        path_nodes.append(GraphPathNode(
                            node=node,
                            incoming_edge=edge,
                            distance=i
                        ))
                    
                    result = GraphPath(path_nodes)
                    
                    # Update cache
                    self._path_cache[cache_key] = result
                    
                    # Emit event
                    await self.event_bus.emit(GraphTraversalCompleted(
                        start_node_id=start_node_id,
                        node_count=len(path) + 1,
                        traversal_time=time.time() - start_time
                    ))
                    
                    # Update metrics
                    if self.metrics:
                        self.metrics.increment("memory_graph_traversals")
                        self.metrics.record("memory_graph_traversal_time_seconds", time.time() - start_time)
                    
                    return result
                
                # Explore neighbors
                for edge_id in self._outgoing_edges.get(current_id, []):
                    edge = self._edges[edge_id]
                    
                    # Skip if not meeting criteria
                    if edge.weight < min_weight:
                        continue
                    if relationships and edge.relationship not in relationships:
                        continue
                    
                    neighbor_id = edge.target_id
                    if neighbor_id not in visited:
                        new_path = path + [(neighbor_id, edge)]
                        queue.append((neighbor_id, new_path))
            
            # No path found
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to find path: {str(e)}")
            traceback.print_exc()
            return None

    async def find_subgraph(
        self,
        node_ids: List[str],
        include_edges: bool = True,
        max_distance: int = 1
    ) -> Tuple[List[GraphNode], List[GraphEdge]]:
        """
        Extract a subgraph containing the specified nodes and their connections.
        
        Args:
            node_ids: List of node identifiers
            include_edges: Whether to include edges between nodes
            max_distance: Maximum distance from specified nodes to include
            
        Returns:
            Tuple of (nodes, edges) in the subgraph
        """
        try:
            # Validate node IDs
            valid_node_ids = [nid for nid in node_ids if nid in self._nodes]
            if not valid_node_ids:
                return [], []
            
            included_nodes = set(valid_node_ids)
            included_edges = set()
            
            # If include_edges, add direct connections between nodes
            if include_edges:
                # Start with the specified nodes
                current_distance = 0
                frontier = set(valid_node_ids)
                
                # Expand to neighbors up to max_distance
                while current_distance < max_distance:
                    next_frontier = set()
                    
                    for node_id in frontier:
                        # Outgoing connections
                        for edge_id in self._outgoing_edges.get(node_id, []):
                            edge = self._edges[edge_id]
                            included_edges.add(edge_id)
                            
                            if edge.target_id not in included_nodes:
                                included_nodes.add(edge.target_id)
                                next_frontier.add(edge.target_id)
                        
                        # Incoming connections
                        for edge_id in self._incoming_edges.get(node_id, []):
                            edge = self._edges[edge_id]
                            included_edges.add(edge_id)
                            
                            if edge.source_id not in included_nodes:
                                included_nodes.add(edge.source_id)
                                next_frontier.add(edge.source_id)
                    
                    current_distance += 1
                    frontier = next_frontier
            
            # Collect nodes and edges
            nodes = [self._nodes[nid] for nid in included_nodes if nid in self._nodes]
            edges = [self._edges[eid] for eid in included_edges if eid in self._edges]
            
            return nodes, edges
            
        except Exception as e:
            self.logger.error(f"Failed to find subgraph: {str(e)}")
            return [], []

    async def query(self, query: Union[SimpleMemoryQuery, GraphQuery]) -> List[GraphNode]:
        """
        Query the graph for nodes.
        
        Args:
            query: Query parameters
            
        Returns:
            List of matching nodes
        """
        start_time = time.time()
        
        try:
            # Handle SimpleMemoryQuery
            if isinstance(query, SimpleMemoryQuery):
                return await self._simple_memory_query(query)
            
            # Handle GraphQuery
            elif isinstance(query, GraphQuery):
                # Generate cache key
                cache_key = self._generate_query_cache_key(query)
                
                # Check cache
                if cache_key in self._query_cache:
                    return self._query_cache[cache_key]
                
                # Emit event
                await self.event_bus.emit(GraphTraversalStarted(
                    start_node_id=query.start_node_id,
                    max_depth=query.max_depth,
                    traversal_type="query"
                ))
                
                # Execute query
                results = await self._execute_graph_query(query)
                
                # Update cache
                self._query_cache[cache_key] = results
                
                # Emit event
                await self.event_bus.emit(GraphTraversalCompleted(
                    start_node_id=query.start_node_id,
                    node_count=len(results),
                    traversal_time=time.time() - start_time
                ))
                
                # Update metrics
                if self.metrics:
                    self.metrics.increment("memory_graph_traversals")
                    self.metrics.record("memory_graph_traversal_time_seconds", time.time() - start_time)
                
                return results
            
            else:
                raise ValueError(f"Unsupported query type: {type(query)}")
            
        except Exception as e:
            self.logger.error(f"Failed to execute query: {str(e)}")
            traceback.print_exc()
            await self.event_bus.emit(ErrorOccurred(
                component="memory_graph",
                error_type=type(e).__name__,
                error_message=str(e),
                severity="error"
            ))
            return []

    async def find_related_memories(
        self,
        memory_id: str,
        max_distance: int = 2,
        limit: int = 10,
        relationship_types: Optional[List[RelationshipType]] = None
    ) -> List[GraphNode]:
        """
        Find memories related to a given memory.
        
        Args:
            memory_id: Memory identifier
            max_distance: Maximum traversal distance
            limit: Maximum number of results
            relationship_types: Filter by relationship types
            
        Returns:
            List of related memory nodes
        """
        try:
            # Get node ID for memory
            if memory_id not in self._memory_to_node:
                return []
            
            node_id = self._memory_to_node[memory_id]
            
            # Build graph query
            query = GraphQuery(
                start_node_id=node_id,
                node_types=[GraphNodeType.MEMORY],
                relationships=relationship_types,
                max_depth=max_distance,
                limit=limit
            )
            
            # Execute query
            results = await self.query(query)
            
            # Remove the original memory node
            return [node for node in results if node.node_id != node_id]
            
        except Exception as e:
            self.logger.error(f"Failed to find related memories: {str(e)}")
