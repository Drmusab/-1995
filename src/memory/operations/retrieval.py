"""
Memory Retrieval System
Author: Drmusab
Last Modified: 2025-07-05 10:09:35 UTC

This module provides the memory retrieval system for the AI assistant, enabling
efficient and context-aware retrieval of memories across different memory types.
It supports various retrieval strategies, relevance scoring, and query optimization
to ensure the most appropriate memories are retrieved for a given context.
"""

from typing import Optional, Dict, Any, List, Set, Tuple, Union, Callable
import asyncio
import time
from datetime import datetime, timezone, timedelta
import json
import uuid
import logging
from dataclasses import dataclass, field
from enum import Enum
import traceback
import heapq
import math
from collections import defaultdict

# Core imports
from src.core.config.loader import ConfigLoader
from src.core.events.event_bus import EventBus
from src.core.events.event_types import (
    MemoryRetrievalStarted, MemoryRetrievalCompleted, MemoryRetrievalFailed,
    SemanticSearchStarted, SemanticSearchCompleted, SemanticSearchFailed,
    TemporalRetrievalStarted, TemporalRetrievalCompleted,
    GraphRetrievalStarted, GraphRetrievalCompleted
)
from src.core.error_handling import ErrorHandler, handle_exceptions
from src.core.dependency_injection import Container

# Memory system imports
from src.memory.core_memory.base_memory import (
    BaseMemory, MemoryItem, MemoryType, MemoryError, SimpleMemoryQuery
)
from src.memory.core_memory.memory_types import (
    WorkingMemory, EpisodicMemory, SemanticMemory
)
from src.memory.storage.vector_store import VectorMemoryStore
from src.memory.storage.memory_graph import (
    MemoryGraphStore, GraphQuery, GraphNodeType, RelationshipType
)
from src.memory.operations.context_manager import MemoryContextManager

# Integration components
from src.integrations.llm.model_router import ModelRouter

# Observability
from src.observability.monitoring.metrics import MetricsCollector
from src.observability.monitoring.tracing import TraceManager
from src.observability.logging.config import get_logger


class RetrievalStrategy(Enum):
    """Strategies for memory retrieval."""
    SEMANTIC = "semantic"           # Retrieval based on semantic similarity
    TEMPORAL = "temporal"           # Retrieval based on time/sequence
    ASSOCIATIVE = "associative"     # Retrieval based on associations/relationships
    IMPORTANCE = "importance"       # Retrieval based on importance/priority
    RECENCY = "recency"             # Retrieval based on recency
    FREQUENCY = "frequency"         # Retrieval based on access frequency
    HYBRID = "hybrid"               # Combination of multiple strategies


class MemoryRetrievalMode(Enum):
    """Operating modes for memory retrieval."""
    PRECISE = "precise"             # High precision, potentially lower recall
    EXPLORATORY = "exploratory"     # Lower precision, higher recall
    BALANCED = "balanced"           # Balance between precision and recall
    COMPREHENSIVE = "comprehensive" # Retrieve from all memory systems
    CONTEXTUAL = "contextual"       # Prioritize context-relevant memories
    FOCUSED = "focused"             # Focus on specific memory types


@dataclass
class RetrievalRequest:
    """
    A request for memory retrieval.
    
    This class encapsulates the parameters for a memory retrieval operation,
    including query, context, filters, and retrieval preferences.
    """
    
    # Core query parameters
    query: str
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    
    # Memory selection parameters
    memory_types: List[MemoryType] = field(default_factory=lambda: [
        MemoryType.WORKING, MemoryType.EPISODIC, MemoryType.SEMANTIC
    ])
    
    # Retrieval configuration
    strategy: RetrievalStrategy = RetrievalStrategy.HYBRID
    mode: MemoryRetrievalMode = MemoryRetrievalMode.BALANCED
    
    # Filtering and limiting
    time_range: Optional[Tuple[datetime, datetime]] = None
    tags: Optional[Set[str]] = None
    max_results: int = 10
    min_relevance: float = 0.6
    
    # Context information
    context: Dict[str, Any] = field(default_factory=dict)
    context_window: Optional[List[str]] = None  # Recent memory IDs to use as context
    
    # Embedding information
    query_embedding: Optional[List[float]] = None
    use_embeddings: bool = True
    
    # Additional parameters
    params: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize derived fields after initialization."""
        if not self.time_range and 'recency_hours' in self.params:
            # Set time range based on recency parameter
            hours = self.params['recency_hours']
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(hours=hours)
            self.time_range = (start_time, end_time)


@dataclass
class RetrievalResult:
    """
    The result of a memory retrieval operation.
    
    This class encapsulates the retrieved memories, along with metadata about
    the retrieval operation and relevance scores.
    """
    
    # Retrieved memories with relevance scores
    items: List[Tuple[MemoryItem, float]]
    
    # Request that generated this result
    request: RetrievalRequest
    
    # Retrieval statistics
    retrieval_time: float = 0.0
    total_candidates: int = 0
    
    # Result metadata
    source_breakdowns: Dict[MemoryType, int] = field(default_factory=dict)
    strategies_used: List[RetrievalStrategy] = field(default_factory=list)
    
    @property
    def memories(self) -> List[MemoryItem]:
        """Get just the memory items without scores."""
        return [item for item, _ in self.items]
    
    @property
    def top_memory(self) -> Optional[MemoryItem]:
        """Get the highest-relevance memory, if any."""
        if not self.items:
            return None
        return self.items[0][0]
    
    def get_content_list(self) -> List[Any]:
        """Get a list of just the content from each memory."""
        return [memory.content for memory in self.memories]
    
    def filter_by_type(self, memory_type: MemoryType) -> List[MemoryItem]:
        """Filter memories by type."""
        return [item for item, _ in self.items if item.memory_type == memory_type]
    
    def filter_by_min_score(self, min_score: float) -> List[MemoryItem]:
        """Filter memories by minimum relevance score."""
        return [item for item, score in self.items if score >= min_score]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "items": [
                {
                    "memory_id": memory.memory_id,
                    "content": memory.content,
                    "memory_type": memory.memory_type.value,
                    "relevance": score,
                    "created_at": memory.metadata.created_at.isoformat() if memory.metadata else None,
                    "metadata": memory.metadata.to_dict() if memory.metadata else None
                }
                for memory, score in self.items
            ],
            "retrieval_time": self.retrieval_time,
            "total_candidates": self.total_candidates,
            "source_breakdowns": {k.value: v for k, v in self.source_breakdowns.items()},
            "strategies_used": [s.value for s in self.strategies_used],
            "query": self.request.query,
            "mode": self.request.mode.value,
            "memory_types": [mt.value for mt in self.request.memory_types]
        }


class MemoryRetriever:
    """
    Memory retrieval system for the AI assistant.
    
    This class provides methods for retrieving memories from different memory
    systems using various strategies, with relevance scoring and filtering.
    It serves as the main entry point for memory retrieval operations.
    """
    
    def __init__(self, container: Container):
        """
        Initialize the memory retriever.
        
        Args:
            container: Dependency injection container
        """
        self.container = container
        self.logger = get_logger(__name__)
        
        # Load configuration
        self.config_loader = container.get(ConfigLoader)
        self.retrieval_config = self.config_loader.get("memory.retrieval", {})
        
        # Event system
        self.event_bus = container.get(EventBus)
        
        # Get memory systems
        self.working_memory = container.get(WorkingMemory)
        self.episodic_memory = container.get(EpisodicMemory)
        self.semantic_memory = container.get(SemanticMemory)
        
        # Get storage systems
        self.vector_store = container.get(VectorMemoryStore)
        try:
            self.graph_store = container.get(MemoryGraphStore)
        except Exception:
            self.logger.warning("MemoryGraphStore not available, graph-based retrieval will be limited")
            self.graph_store = None
        
        # Get context manager
        try:
            self.context_manager = container.get(MemoryContextManager)
        except Exception:
            self.logger.warning("MemoryContextManager not available, contextual retrieval will be limited")
            self.context_manager = None
        
        # Get model router for embeddings
        try:
            self.model_router = container.get(ModelRouter)
        except Exception:
            self.logger.warning("ModelRouter not available, semantic retrieval will be limited")
            self.model_router = None
        
        # Monitoring components
        try:
            self.metrics = container.get(MetricsCollector)
            self.tracer = container.get(TraceManager)
        except Exception:
            self.logger.warning("Monitoring components not available")
            self.metrics = None
            self.tracer = None
        
        # Initialize caches
        self._query_cache: Dict[str, RetrievalResult] = {}
        self._embedding_cache: Dict[str, List[float]] = {}
        
        # Configure settings
        self.max_cache_size = self.retrieval_config.get("max_cache_size", 100)
        self.cache_ttl = self.retrieval_config.get("cache_ttl", 300)  # seconds
        self.default_max_results = self.retrieval_config.get("default_max_results", 10)
        self.min_embedding_similarity = self.retrieval_config.get("min_embedding_similarity", 0.7)
        
        # Register metrics
        if self.metrics:
            self.metrics.register_counter("memory_retrieval_total")
            self.metrics.register_counter("memory_retrieval_cache_hits")
            self.metrics.register_counter("memory_retrieval_cache_misses")
            self.metrics.register_histogram("memory_retrieval_duration_seconds")
            self.metrics.register_gauge("memory_retrieval_result_count")
        
        self.logger.info("MemoryRetriever initialized")

    async def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        """
        Retrieve memories based on the given request.
        
        This is the main entry point for memory retrieval, orchestrating the
        process across different memory systems and strategies.
        
        Args:
            request: Retrieval request parameters
            
        Returns:
            Retrieval result with memories and metadata
        """
        start_time = time.time()
        
        # Generate cache key for potential caching
        cache_key = self._generate_cache_key(request)
        
        # Check cache for identical recent query
        if cache_key in self._query_cache:
            if self.metrics:
                self.metrics.increment("memory_retrieval_cache_hits")
            return self._query_cache[cache_key]
        
        # Emit retrieval started event
        await self.event_bus.emit(MemoryRetrievalStarted(
            query=request.query,
            user_id=request.user_id,
            session_id=request.session_id,
            strategy=request.strategy.value,
            mode=request.mode.value
        ))
        
        if self.metrics:
            self.metrics.increment("memory_retrieval_total")
            self.metrics.increment("memory_retrieval_cache_misses")
        
        try:
            # Generate embeddings if needed and not provided
            if request.use_embeddings and not request.query_embedding and self.model_router:
                embedding_key = f"query:{request.query}"
                if embedding_key in self._embedding_cache:
                    request.query_embedding = self._embedding_cache[embedding_key]
                else:
                    request.query_embedding = await self.model_router.get_embeddings(request.query)
                    self._embedding_cache[embedding_key] = request.query_embedding
            
            # Choose retrieval strategy based on request
            if request.strategy == RetrievalStrategy.SEMANTIC:
                result = await self._semantic_retrieval(request)
            elif request.strategy == RetrievalStrategy.TEMPORAL:
                result = await self._temporal_retrieval(request)
            elif request.strategy == RetrievalStrategy.ASSOCIATIVE:
                result = await self._associative_retrieval(request)
            elif request.strategy == RetrievalStrategy.IMPORTANCE:
                result = await self._importance_retrieval(request)
            elif request.strategy == RetrievalStrategy.RECENCY:
                result = await self._recency_retrieval(request)
            elif request.strategy == RetrievalStrategy.FREQUENCY:
                result = await self._frequency_retrieval(request)
            else:  # Default to hybrid
                result = await self._hybrid_retrieval(request)
            
            # Update retrieval time
            result.retrieval_time = time.time() - start_time
            
            # Cache result
            self._update_cache(cache_key, result)
            
            # Emit retrieval completed event
            await self.event_bus.emit(MemoryRetrievalCompleted(
                query=request.query,
                user_id=request.user_id,
                session_id=request.session_id,
                item_count=len(result.items),
                retrieval_time=result.retrieval_time
            ))
            
            # Update metrics
            if self.metrics:
                self.metrics.record("memory_retrieval_duration_seconds", result.retrieval_time)
                self.metrics.gauge("memory_retrieval_result_count", len(result.items))
            
            return result
            
        except Exception as e:
            error_time = time.time() - start_time
            self.logger.error(f"Memory retrieval failed: {str(e)}")
            traceback.print_exc()
            
            # Emit retrieval failed event
            await self.event_bus.emit(MemoryRetrievalFailed(
                query=request.query,
                user_id=request.user_id,
                session_id=request.session_id,
                error=str(e),
                retrieval_time=error_time
            ))
            
            # Return empty result
            return RetrievalResult(
                items=[],
                request=request,
                retrieval_time=error_time,
                total_candidates=0,
                strategies_used=[request.strategy]
            )

    async def retrieve_by_id(self, memory_id: str) -> Optional[MemoryItem]:
        """
        Retrieve a specific memory by ID.
        
        Args:
            memory_id: Memory identifier
            
        Returns:
            Memory item or None if not found
        """
        # Try working memory first
        memory = await self.working_memory.retrieve(memory_id)
        if memory:
            return memory
        
        # Try episodic memory
        memory = await self.episodic_memory.retrieve(memory_id)
        if memory:
            return memory
        
        # Try semantic memory
        memory = await self.semantic_memory.retrieve(memory_id)
        if memory:
            return memory
        
        return None

    async def retrieve_recent(
        self, 
        session_id: str, 
        limit: int = 5, 
        memory_types: Optional[List[MemoryType]] = None
    ) -> List[MemoryItem]:
        """
        Retrieve recent memories for a session.
        
        Args:
            session_id: Session identifier
            limit: Maximum number of memories to retrieve
            memory_types: Types of memories to retrieve
            
        Returns:
            List of recent memory items
        """
        memory_types = memory_types or [MemoryType.WORKING, MemoryType.EPISODIC]
        results = []
        
        # Retrieve from working memory
        if MemoryType.WORKING in memory_types:
            working_items = await self.working_memory.get_recent_items(session_id, limit)
            results.extend(working_items)
        
        # Retrieve from episodic memory
        if MemoryType.EPISODIC in memory_types:
            episodic_items = await self.episodic_memory.get_session_memories(session_id, limit)
            results.extend(episodic_items)
        
        # Sort by recency and limit results
        results.sort(
            key=lambda x: x.metadata.last_accessed or x.metadata.created_at, 
            reverse=True
        )
        
        return results[:limit]

    async def retrieve_for_context(
        self, 
        context: Dict[str, Any], 
        session_id: Optional[str] = None,
        limit: int = 5
    ) -> List[MemoryItem]:
        """
        Retrieve memories relevant to a specific context.
        
        Args:
            context: Context information
            session_id: Optional session identifier
            limit: Maximum number of memories to retrieve
            
        Returns:
            List of context-relevant memory items
        """
        if not self.context_manager:
            # Fall back to simple retrieval if context manager not available
            request = RetrievalRequest(
                query="",
                session_id=session_id,
                mode=MemoryRetrievalMode.CONTEXTUAL,
                context=context,
                max_results=limit
            )
            result = await self._hybrid_retrieval(request)
            return result.memories
        
        # Use context manager for more sophisticated retrieval
        context_memories = await self.context_manager.get_memories_for_context(
            context, session_id, limit
        )
        
        return context_memories

    async def retrieve_by_query(
        self, 
        query: str, 
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        strategy: RetrievalStrategy = RetrievalStrategy.SEMANTIC,
        max_results: int = 10
    ) -> List[MemoryItem]:
        """
        Simplified retrieval by query string.
        
        Args:
            query: Query string
            session_id: Optional session identifier
            user_id: Optional user identifier
            strategy: Retrieval strategy to use
            max_results: Maximum number of results
            
        Returns:
            List of relevant memory items
        """
        request = RetrievalRequest(
            query=query,
            session_id=session_id,
            user_id=user_id,
            strategy=strategy,
            max_results=max_results
        )
        
        result = await self.retrieve(request)
        return result.memories

    async def _semantic_retrieval(self, request: RetrievalRequest) -> RetrievalResult:
        """
        Retrieve memories based on semantic similarity.
        
        Args:
            request: Retrieval request
            
        Returns:
            Retrieval result
        """
        start_time = time.time()
        
        # Emit semantic search started event
        await self.event_bus.emit(SemanticSearchStarted(
            query=request.query,
            user_id=request.user_id,
            session_id=request.session_id
        ))
        
        # Initialize aggregated results
        all_items: List[Tuple[MemoryItem, float]] = []
        source_breakdowns: Dict[MemoryType, int] = defaultdict(int)
        total_candidates = 0
        
        try:
            # Check if we have vector embeddings for semantic search
            if not request.query_embedding and not self.model_router:
                self.logger.warning("No query embeddings available for semantic search")
                
                # Fall back to simple text search
                if MemoryType.WORKING in request.memory_types:
                    working_items = await self.working_memory.search(request.query)
                    all_items.extend((item, 0.7) for item in working_items)
                    source_breakdowns[MemoryType.WORKING] = len(working_items)
                    total_candidates += len(working_items)
                
                if MemoryType.EPISODIC in request.memory_types:
                    episodic_items = await self.episodic_memory.search(request.query)
                    all_items.extend((item, 0.7) for item in episodic_items)
                    source_breakdowns[MemoryType.EPISODIC] = len(episodic_items)
                    total_candidates += len(episodic_items)
                
                # Emit semantic search completed event
                await self.event_bus.emit(SemanticSearchCompleted(
                    query=request.query,
                    user_id=request.user_id,
                    session_id=request.session_id,
                    result_count=len(all_items),
                    search_time=time.time() - start_time
                ))
                
                return RetrievalResult(
                    items=all_items[:request.max_results],
                    request=request,
                    retrieval_time=time.time() - start_time,
                    total_candidates=total_candidates,
                    source_breakdowns=source_breakdowns,
                    strategies_used=[RetrievalStrategy.SEMANTIC]
                )
            
            # Use vector store for semantic search if available
            if self.vector_store and request.query_embedding:
                # Prepare filters based on request
                filters = {}
                if request.memory_types:
                    if len(request.memory_types) == 1:
                        filters["memory_type"] = request.memory_types[0]
                
                if request.user_id:
                    filters["owner_id"] = request.user_id
                
                if request.session_id:
                    filters["session_id"] = request.session_id
                
                # Execute vector search
                vector_results = await self.vector_store.similarity_search(
                    query_vector=request.query_embedding,
                    similarity_threshold=request.min_relevance,
                    top_k=request.max_results * 2,  # Get more for filtering
                    filters=filters
                )
                
                # Process vector results
                for memory in vector_results:
                    # Skip if memory type not requested
                    if memory.memory_type not in request.memory_types:
                        continue
                    
                    # Calculate similarity
                    similarity = self._calculate_vector_similarity(
                        request.query_embedding, memory.embeddings
                    )
                    
                    # Only include if above threshold
                    if similarity >= request.min_relevance:
                        all_items.append((memory, similarity))
                        source_breakdowns[memory.memory_type] += 1
                
                total_candidates += len(vector_results)
            
            # Add results from working memory (if needed)
            if MemoryType.WORKING in request.memory_types and request.session_id:
                if request.context:
                    # Get most relevant to context
                    working_items = await self.working_memory.get_most_relevant(
                        request.session_id, request.context, limit=request.max_results
                    )
                else:
                    # Get recent items
                    working_items = await self.working_memory.get_recent_items(
                        request.session_id, limit=request.max_results
                    )
                
                # Score items
                for item in working_items:
                    # Skip if already included
                    if any(existing.memory_id == item.memory_id for existing, _ in all_items):
                        continue
                    
                    # Calculate score (less precise without embeddings)
                    text_similarity = self._text_similarity(request.query, self._get_content_text(item.content))
                    recency_factor = self._calculate_recency_factor(item)
                    
                    # Combine factors
                    score = (text_similarity * 0.7) + (recency_factor * 0.3)
                    
                    if score >= request.min_relevance:
                        all_items.append((item, score))
                        source_breakdowns[MemoryType.WORKING] += 1
                
                total_candidates += len(working_items)
            
            # Add results from semantic memory
            if MemoryType.SEMANTIC in request.memory_types:
                # Use context for better retrieval
                semantic_items = await self.semantic_memory.retrieve_relevant(
                    request.query, 
                    context=request.context, 
                    limit=request.max_results
                )
                
                if semantic_items:
                    for fact_data in semantic_items:
                        # Skip if we don't have a memory_id
                        if "id" not in fact_data:
                            continue
                            
                        # Retrieve full memory item
                        memory = await self.semantic_memory.retrieve(fact_data["id"])
                        if not memory:
                            continue
                        
                        # Skip if already included
                        if any(existing.memory_id == memory.memory_id for existing, _ in all_items):
                            continue
                        
                        # Use confidence as score
                        score = fact_data.get("confidence", 0.7)
                        
                        if score >= request.min_relevance:
                            all_items.append((memory, score))
                            source_breakdowns[MemoryType.SEMANTIC] += 1
                
                total_candidates += len(semantic_items)
            
            # Sort by score (descending) and limit results
            all_items.sort(key=lambda x: x[1], reverse=True)
            results = all_items[:request.max_results]
            
            # Emit semantic search completed event
            await self.event_bus.emit(SemanticSearchCompleted(
                query=request.query,
                user_id=request.user_id,
                session_id=request.session_id,
                result_count=len(results),
                search_time=time.time() - start_time
            ))
            
            return RetrievalResult(
                items=results,
                request=request,
                retrieval_time=time.time() - start_time,
                total_candidates=total_candidates,
                source_breakdowns=source_breakdowns,
                strategies_used=[RetrievalStrategy.SEMANTIC]
            )
            
        except Exception as e:
            self.logger.error(f"Semantic retrieval failed: {str(e)}")
            traceback.print_exc()
            
            # Emit semantic search failed event
            await self.event_bus.emit(SemanticSearchFailed(
                query=request.query,
                user_id=request.user_id,
                session_id=request.session_id,
                error=str(e)
            ))
            
            # Return empty result
            return RetrievalResult(
                items=[],
                request=request,
                retrieval_time=time.time() - start_time,
                total_candidates=0,
                strategies_used=[RetrievalStrategy.SEMANTIC]
            )

    async def _temporal_retrieval(self, request: RetrievalRequest) -> RetrievalResult:
        """
        Retrieve memories based on temporal relationships.
        
        Args:
            request: Retrieval request
            
        Returns:
            Retrieval result
        """
        start_time = time.time()
        
        # Emit temporal retrieval started event
        await self.event_bus.emit(TemporalRetrievalStarted(
            query=request.query,
            user_id=request.user_id,
            session_id=request.session_id
        ))
        
        # Initialize result tracking
        all_items: List[Tuple[MemoryItem, float]] = []
        source_breakdowns: Dict[MemoryType, int] = defaultdict(int)
        total_candidates = 0
        
        try:
            # Process time range from request
            time_range = request.time_range
            if not time_range and 'time_frame' in request.params:
                # Parse time frame from request params
                frame = request.params['time_frame']
                end_time = datetime.now(timezone.utc)
                
                if frame == 'today':
                    start_time_dt = end_time.replace(hour=0, minute=0, second=0, microsecond=0)
                elif frame == 'yesterday':
                    start_time_dt = (end_time - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
                    end_time = end_time.replace(hour=0, minute=0, second=0, microsecond=0)
                elif frame == 'this_week':
                    start_time_dt = (end_time - timedelta(days=end_time.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
                elif frame == 'last_week':
                    start_of_this_week = (end_time - timedelta(days=end_time.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
                    start_time_dt = start_of_this_week - timedelta(days=7)
                    end_time = start_of_this_week
                elif frame == 'this_month':
                    start_time_dt = end_time.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
                else:
                    # Default to last 24 hours
                    start_time_dt = end_time - timedelta(hours=24)
                
                time_range = (start_time_dt, end_time)
            
            # Retrieve from episodic memory based on time
            if MemoryType.EPISODIC in request.memory_types:
                if time_range:
                    # Format date strings for query
                    start_str = time_range[0].strftime('%Y-%m-%d')
                    end_str = time_range[1].strftime('%Y-%m-%d')
                    
                    # Get memories in time range
                    episodic_items = await self.episodic_memory.get_memories_by_timerange(
                        start_str, end_str, limit=request.max_results * 2
                    )
                else:
                    # Get recent memories if no time range specified
                    if request.session_id:
                        episodic_items = await self.episodic_memory.get_session_memories(
                            request.session_id, limit=request.max_results * 2
                        )
                    elif request.user_id:
                        episodic_items = await self.episodic_memory.get_user_memories(
                            request.user_id, limit=request.max_results * 2
                        )
                    else:
                        # Without session or user, just get by date
                        date_str = datetime.now(timezone.utc).strftime('%Y-%m-%d')
                        episodic_items = await self.episodic_memory.get_memories_by_date(
                            date_str, limit=request.max_results * 2
                        )
                
                # Process episodic items
                for item in episodic_items:
                    # Check if matching query text (if provided)
                    relevance = 1.0  # Default high relevance for temporal matches
                    
                    if request.query:
                        # Adjust relevance based on text similarity
                        text_similarity = self._text_similarity(
                            request.query, self._get_content_text(item.content)
                        )
                        relevance = text_similarity
                    
                    # Check if above threshold
                    if relevance >= request.min_relevance:
                        all_items.append((item, relevance))
                        source_breakdowns[MemoryType.EPISODIC] += 1
                
                total_candidates += len(episodic_items)
            
            # Get working memory items if needed
            if MemoryType.WORKING in request.memory_types and request.session_id:
                working_items = await self.working_memory.get_recent_items(
                    request.session_id, limit=request.max_results
                )
                
                # Filter by time range if specified
                if time_range:
                    working_items = [
                        item for item in working_items
                        if time_range[0] <= item.metadata.created_at <= time_range[1]
                    ]
                
                # Process working memory items
                for item in working_items:
                    # Calculate relevance
                    relevance = 0.8  # Default good relevance for recent working memory
                    
                    if request.query:
                        # Adjust based on text similarity
                        text_similarity = self._text_similarity(
                            request.query, self._get_content_text(item.content)
                        )
                        relevance = (relevance + text_similarity) / 2
                    
                    # Check if above threshold
                    if relevance >= request.min_relevance:
                        all_items.append((item, relevance))
                        source_breakdowns[MemoryType.WORKING] += 1
                
                total_candidates += len(working_items)
            
            # Use graph store to find temporal relationships if available
            if self.graph_store and MemoryType.EPISODIC in request.memory_types:
                # Find nodes with temporal relationships
                temporal_relationships = [
                    RelationshipType.OCCURRED_BEFORE,
                    RelationshipType.OCCURRED_AFTER,
                    RelationshipType.OCCURRED_DURING
                ]
                
                # Emit graph retrieval started event
                await self.event_bus.emit(GraphRetrievalStarted(
                    query=request.query,
                    relationship_types=[r.value for r in temporal_relationships]
                ))
                
                # Build graph query
                graph_query = GraphQuery(
                    node_types=[GraphNodeType.MEMORY, GraphNodeType.EVENT, GraphNodeType.TIME],
                    relationships=temporal_relationships,
                    max_depth=2,
                    limit=request.max_results,
                    time_range=time_range
                )
                
                # Execute graph query
                graph_nodes = await self.graph_store.query(graph_query)
                
                # Process graph nodes
                for node in graph_nodes:
                    # Skip nodes without memory_id
                    if not node.memory_id:
                        continue
                    
                    # Get the actual memory
                    memory = await self.retrieve_by_id(node.memory_id)
                    if not memory:
                        continue
                    
                    # Skip if already included
                    if any(existing.memory_id == memory.memory_id for existing, _ in all_items):
                        continue
                    
                    # Calculate relevance
                    relevance = 0.7  # Default relevance for graph matches
                    
                    if request.query:
                        # Adjust based on text similarity
                        text_similarity = self._text_similarity(
                            request.query, self._get_content_text(memory.content)
                        )
                        relevance = (relevance + text_similarity) / 2
                    
                    # Add to results if above threshold
                    if relevance >= request.min_relevance:
                        all_items.append((memory, relevance))
                        source_breakdowns[memory.memory_type] += 1
                
                total_candidates += len(graph_nodes)
                
                # Emit graph retrieval completed event
                await self.event_bus.emit(GraphRetrievalCompleted(
                    query=request.query,
                    node_count=len(graph_nodes),
                    result_count=sum(1 for _, score in all_items if score >= request.min_relevance)
                ))
            
            # Sort by time (most recent first) and then by relevance
            all_items.sort(key=lambda x: (
                -(x[0].metadata.created_at.timestamp() if x[0].metadata else 0),
                -x[1]
            ))
            
            results = all_items[:request.max_results]
            
            # Emit temporal retrieval completed event
            await self.event_bus.emit(TemporalRetrievalCompleted(
                query=request.query,
                user_id=request.user_id,
                session_id=request.session_id,
                result_count=len(results)
            ))
            
            return RetrievalResult(
                items=results,
                request=request,
                retrieval_time=time.time() - start_time,
                total_candidates=total_candidates,
                source_breakdowns=source_breakdowns,
                strategies_used=[RetrievalStrategy.TEMPORAL]
            )
            
        except Exception as e:
            self.logger.error(f"Temporal retrieval failed: {str(e)}")
            traceback.print_exc()
            
            # Return empty result
            return RetrievalResult(
                items=[],
                request=request,
                retrieval_time=time.time() - start_time,
                total_candidates=0,
                strategies_used=[RetrievalStrategy.TEMPORAL]
            )

    async def _associative_retrieval(self, request: RetrievalRequest) -> RetrievalResult:
        """
        Retrieve memories based on associative relationships.
        
        Args:
            request: Retrieval request
            
        Returns:
            Retrieval result
        """
        start_time = time.time()
        
        # Initialize result tracking
        all_items: List[Tuple[MemoryItem, float]] = []
        source_breakdowns: Dict[MemoryType, int] = defaultdict(int)
        total_candidates = 0
        
        try:
            # First, get a seed set of memories from semantic search
            semantic_request = RetrievalRequest(
                query=request.query,
                session_id=request.session_id,
                user_id=request.user_id,
                memory_types=request.memory_types,
                strategy=RetrievalStrategy.SEMANTIC,
                max_results=min(5, request.max_results),
                min_relevance=request.min_relevance
            )
            
            seed_result = await self._semantic_retrieval(semantic_request)
            seed_memories = seed_result.memories
            total_candidates += seed_result.total_candidates
            
            # Use graph store to find associated memories
            if self.graph_store and seed_memories:
                # Emit graph retrieval started event
                await self.event_bus.emit(GraphRetrievalStarted(
                    query=request.query,
                    relationship_types=["related_to", "similar_to", "instance_of"]
                ))
                
                # Get memory IDs from seed memories
                seed_memory_ids = [memory.memory_id for memory in seed_memories]
                
                # Convert to node IDs
                seed_node_ids = []
                for memory_id in seed_memory_ids:
                    node = await self.graph_store.get_node_by_memory_id(memory_id)
                    if node:
                        seed_node_ids.append(node.node_id)
                
                if seed_node_ids:
                    # Find related nodes in the graph
                    relationship_types = [
                        RelationshipType.RELATED_TO,
                        RelationshipType.SIMILAR_TO,
                        RelationshipType.INSTANCE_OF,
                        RelationshipType.IS_A,
                        RelationshipType.HAS_A,
                        RelationshipType.PART_OF
                    ]
                    
                    # Extract subgraph around seed nodes
                    nodes, edges = await self.graph_store.find_subgraph(
                        seed_node_ids, 
                        include_edges=True,
                        max_distance=2
                    )
                    
                    # Convert nodes back to memories
                    for node in nodes:
                        # Skip if no memory ID
                        if not node.memory_id:
                            continue
                            
                        # Skip if already in seed memories
                        if node.memory_id in seed_memory_ids:
                            continue
                        
                        # Get the memory item
                        memory = await self.retrieve_by_id(node.memory_id)
                        if not memory:
                            continue
                            
                        # Skip if memory type not requested
                        if memory.memory_type not in request.memory_types:
                            continue
                        
                        # Calculate relevance based on graph properties
                        # Here we use node properties and edge weights
                        relevance = node.properties.get("importance", 0.5)
                        
                        # Adjust based on connection strength
                        connected_edges = [
                            edge for edge in edges 
                            if edge.source_id in seed_node_ids and edge.target_id == node.node_id
                        ]
                        
                        if connected_edges:
                            # Use maximum edge weight as a factor
                            max_weight = max(edge.weight for edge in connected_edges)
                            relevance = (relevance + max_weight) / 2
                        
                        # Add to results if above threshold
                        if relevance >= request.min_relevance:
                            all_items.append((memory, relevance))
                            source_breakdowns[memory.memory_type] += 1
                    
                    total_candidates += len(nodes)
                    
                    # Emit graph retrieval completed event
                    await self.event_bus.emit(GraphRetrievalCompleted(
                        query=request.query,
                        node_count=len(nodes),
                        result_count=len(all_items)
                    ))
            
            # Add seed memories to results
            for memory in seed_memories:
                # Find existing score or use 1.0 as default
                score = 1.0
                for existing_memory, existing_score in all_items:
                    if existing_memory.memory_id == memory.memory_id:
                        score = existing_score
                        break
                
                # Add to results if not already included
                if not any(existing.memory_id == memory.memory_id for existing, _ in all_items):
                    all_items.append((memory, score))
                    source_breakdowns[memory.memory_type] += 1
            
            # Sort by relevance (descending)
            all_items.sort(key=lambda x: x[1], reverse=True)
            
            # Limit results
            results = all_items[:request.max_results]
            
            return RetrievalResult(
                items=results,
                request=request,
                retrieval_time=time.time() - start_time,
                total_candidates=total_candidates,
                source_breakdowns=source_breakdowns,
                strategies_used=[RetrievalStrategy.ASSOCIATIVE]
            )
            
        except Exception as e:
            self.logger.error(f"Associative retrieval failed: {str(e)}")
            traceback.print_exc()
            
            # Return empty result
            return RetrievalResult(
                items=[],
                request=request,
                retrieval_time=time.time() - start_time,
                total_candidates=0,
                strategies_used=[RetrievalStrategy.ASSOCIATIVE]
            )

    async def _importance_retrieval(self, request: RetrievalRequest) -> RetrievalResult:
        """
        Retrieve memories based on importance.
        
        Args:
            request: Retrieval request
            
        Returns:
            Retrieval result
        """
        start_time = time.time()
        
        # Initialize result tracking
        all_items: List[Tuple[MemoryItem, float]] = []
        source_breakdowns: Dict[MemoryType, int] = defaultdict(int)
        total_candidates = 0
        
        try:
            # Collect items from different memory types
            tasks = []
            
            # Working memory
            if MemoryType.WORKING in request.memory_types and request.session_id:
                tasks.append(self.working_memory.get_recent_items(request.session_id, limit=request.max_results * 2))
            
            # Episodic memory
            if MemoryType.EPISODIC in request.memory_types:
                if request.session_id:
                    tasks.append(self.episodic_memory.get_session_memories(request.session_id, limit=request.max_results * 2))
                elif request.user_id:
                    tasks.append(self.episodic_memory.get_user_memories(request.user_id, limit=request.max_results * 2))
            
            # Semantic memory
            if MemoryType.SEMANTIC in request.memory_types:
                tasks.append(self.semantic_memory.retrieve_relevant(request.query, request.context, limit=request.max_results))
            
            # Execute tasks concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for result in results:
                if isinstance(result, Exception):
                    self.logger.error(f"Error during importance retrieval: {str(result)}")
                    continue
                
                if isinstance(result, list):
                    # Handle list of memory items
                    for memory in result:
                        if isinstance(memory, MemoryItem):
                            # Calculate importance score
                            importance = memory.metadata.importance if memory.metadata else 0.5
                            
                            # If query specified, adjust importance by text similarity
                            if request.query:
                                text_similarity = self._text_similarity(
                                    request.query, self._get_content_text(memory.content)
                                )
                                
                                # Weight importance higher than similarity
                                score = (importance * 0.7) + (text_similarity * 0.3)
                            else:
                                score = importance
                            
                            # Add to results if above threshold
                            if score >= request.min_relevance:
                                all_items.append((memory, score))
                                source_breakdowns[memory.memory_type] += 1
                            
                            total_candidates += 1
                        else:
                            # Handle semantic memory results (dictionaries)
                            if isinstance(memory, dict) and "id" in memory:
                                memory_item = await self.semantic_memory.retrieve(memory["id"])
                                if memory_item:
                                    # Use confidence as score
                                    score = memory.get("confidence", 0.7)
                                    
                                    # Add to results if above threshold
                                    if score >= request.min_relevance:
                                        all_items.append((memory_item, score))
                                        source_breakdowns[MemoryType.SEMANTIC] += 1
                                
                                total_candidates += 1
            
            # Sort by importance score (descending)
            all_items.sort(key=lambda x: x[1], reverse=True)
            
            # Limit results
            results = all_items[:request.max_results]
            
            return RetrievalResult(
                items=results,
                request=request,
                retrieval_time=time.time() - start_time,
                total_candidates=total_candidates,
                source_breakdowns=source_breakdowns,
                strategies_used=[RetrievalStrategy.IMPORTANCE]
            )
            
        except Exception as e:
            self.logger.error(f"Importance retrieval failed: {str(e)}")
            traceback.print_exc()
            
            # Return empty result
            return RetrievalResult(
                items=[],
                request=request,
                retrieval_time=time.time() - start_time,
                total_candidates=0,
                strategies_used=[RetrievalStrategy.IMPORTANCE]
            )

    async def _recency_retrieval(self, request: RetrievalRequest) -> RetrievalResult:
        """
        Retrieve memories based on recency.
        
        Args:
            request: Retrieval request
            
        Returns:
            Retrieval result
        """
        start_time = time.time()
        
        # Initialize result tracking
        all_items: List[Tuple[MemoryItem, float]] = []
        source_breakdowns: Dict[MemoryType, int] = defaultdict(int)
        total_candidates = 0
        
        try:
            # Working memory (most recent items)
            if MemoryType.WORKING in request.memory_types and request.session_id:
                working_items = await self.working_memory.get_recent_items(
                    request.session_id, limit=request.max_results
                )
                
                for item in working_items:
                    # Calculate recency score
                    recency_score = self._calculate_recency_factor(item)
                    
                    # If query specified, adjust by text similarity
                    if request.query:
                        text_similarity = self._text_similarity(
                            request.query, self._get_content_text(item.content)
                        )
                        score = (recency_score * 0.8) + (text_similarity * 0.2)
                    else:
                        score = recency_score
                    
                    # Add to results if above threshold
                    if score >= request.min_relevance:
                        all_items.append((item, score))
                        source_breakdowns[MemoryType.WORKING] += 1
                
                total_candidates += len(working_items)
            
            # Episodic memory (recent episodes)
            if MemoryType.EPISODIC in request.memory_types:
                # Determine scope of retrieval
                if request.session_id:
                    episodic_items = await self.episodic_memory.get_session_memories(
                        request.session_id, limit=request.max_results
                    )
                elif request.user_id:
                    episodic_items = await self.episodic_memory.get_user_memories(
                        request.user_id, limit=request.max_results
                    )
                else:
                    # Get today's memories
                    date_str = datetime.now(timezone.utc).strftime('%Y-%m-%d')
                    episodic_items = await self.episodic_memory.get_memories_by_date(
                        date_str, limit=request.max_results
                    )
                
                for item in episodic_items:
                    # Calculate recency score
                    recency_score = self._calculate_recency_factor(item)
                    
                    # If query specified, adjust by text similarity
                    if request.query:
                        text_similarity = self._text_similarity(
                            request.query, self._get_content_text(item.content)
                        )
                        score = (recency_score * 0.7) + (text_similarity * 0.3)
                    else:
                        score = recency_score
                    
                    # Add to results if above threshold
                    if score >= request.min_relevance:
                        all_items.append((item, score))
                        source_breakdowns[MemoryType.EPISODIC] += 1
                
                total_candidates += len(episodic_items)
            
            # Sort by recency (most recent first)
            all_items.sort(key=lambda x: (
                -(x[0].metadata.last_accessed or x[0].metadata.created_at).timestamp() 
                if x[0].metadata else 0
            ))
            
            # Limit results
            results = all_items[:request.max_results]
            
            return RetrievalResult(
                items=results,
                request=request,
                retrieval_time=time.time() - start_time,
                total_candidates=total_candidates,
                source_breakdowns=source_breakdowns,
                strategies_used=[RetrievalStrategy.RECENCY]
            )
            
        except Exception as e:
            self.logger.error(f"Recency retrieval failed: {str(e)}")
            traceback.print_exc()
            
            # Return empty result
            return RetrievalResult(
                items=[],
                request=request,
                retrieval_time=time.time() - start_time,
                total_candidates=0,
                strategies_used=[RetrievalStrategy.RECENCY]
            )

    async def _frequency_retrieval(self, request: RetrievalRequest) -> RetrievalResult:
        """
        Retrieve memories based on access frequency.
        
        Args:
            request: Retrieval request
            
        Returns:
            Retrieval result
        """
        start_time = time.time()
        
        # Initialize result tracking
        all_items: List[Tuple[MemoryItem, float]] = []
        source_breakdowns: Dict[MemoryType, int] = defaultdict(int)
        total_candidates = 0
        
        try:
            # This retrieval strategy is primarily useful for episodic memory
            if MemoryType.EPISODIC in request.memory_types:
                # Start with a semantic search to get candidate items
                query = request.query or "frequently accessed"
                
                # Get initial candidates
                episodic_items = await self.episodic_memory.search(query)
                total_candidates += len(episodic_items)
                
                # Score based on access frequency
                for item in episodic_items:
                    # Calculate frequency score
                    access_count = item.metadata.access_count if item.metadata else 0
                    
                    # Normalize access count to a 0-1 score
                    max_count = 50  # Arbitrary cap for normalization
                    frequency_score = min(1.0, access_count / max_count)
                    
                    # Adjust score based on semantic relevance if query provided
                    if request.query:
                        text_similarity = self._text_similarity(
                            request.query, self._get_content_text(item.content)
                        )
                        score = (frequency_score * 0.7) + (text_similarity * 0.3)
                    else:
                        score = frequency_score
                    
                    # Add to results if above threshold
                    if score >= request.min_relevance:
                        all_items.append((item, score))
                        source_breakdowns[MemoryType.EPISODIC] += 1
            
            # For working memory, access frequency is less meaningful, but we can add some
            if MemoryType.WORKING in request.memory_types and request.session_id:
                working_items = await self.working_memory.get_recent_items(
                    request.session_id, limit=request.max_results
                )
                total_candidates += len(working_items)
                
                for item in working_items:
                    # Working memory items are frequently accessed by nature
                    base_score = 0.7
                    
                    # Adjust based on query if provided
                    if request.query:
                        text_similarity = self._text_similarity(
                            request.query, self._get_content_text(item.content)
                        )
                        score = (base_score * 0.6) + (text_similarity * 0.4)
                    else:
                        score = base_score
                    
                    # Add to results if above threshold
                    if score >= request.min_relevance:
                        all_items.append((item, score))
                        source_breakdowns[MemoryType.WORKING] += 1
            
            # Sort by frequency score (descending)
            all_items.sort(key=lambda x: x[1], reverse=True)
            
            # Limit results
            results = all_items[:request.max_results]
            
            return RetrievalResult(
                items=results,
                request=request,
                retrieval_time=time.time() - start_time,
                total_candidates=total_candidates,
                source_breakdowns=source_breakdowns,
                strategies_used=[RetrievalStrategy.FREQUENCY]
            )
            
        except Exception as e:
            self.logger.error(f"Frequency retrieval failed: {str(e)}")
            traceback.print_exc()
            
            # Return empty result
            return RetrievalResult(
                items=[],
                request=request,
                retrieval_time=time.time() - start_time,
                total_candidates=0,
                strategies_used=[RetrievalStrategy.FREQUENCY]
            )

    async def _hybrid_retrieval(self, request: RetrievalRequest) -> RetrievalResult:
        """
        Retrieve memories using a hybrid approach combining multiple strategies.
        
        Args:
            request: Retrieval request
            
        Returns:
            Retrieval result
        """
        start_time = time.time()
        
        # Initialize result tracking
        all_results: Dict[str, Tuple[MemoryItem, float, RetrievalStrategy]] = {}
        source_breakdowns: Dict[MemoryType, int] = defaultdict(int)
        total_candidates = 0
        strategies_used = set()
        
        try:
            # Choose strategies based on mode
            if request.mode == MemoryRetrievalMode.PRECISE:
                strategies = [RetrievalStrategy.SEMANTIC]
            elif request.mode == MemoryRetrievalMode.EXPLORATORY:
                strategies = [RetrievalStrategy.SEMANTIC, RetrievalStrategy.ASSOCIATIVE]
            elif request.mode == MemoryRetrievalMode.COMPREHENSIVE:
                strategies = [
                    RetrievalStrategy.SEMANTIC, 
                    RetrievalStrategy.TEMPORAL, 
                    RetrievalStrategy.ASSOCIATIVE
                ]
            elif request.mode == MemoryRetrievalMode.CONTEXTUAL:
                strategies = [RetrievalStrategy.SEMANTIC, RetrievalStrategy.RECENCY]
            elif request.mode == MemoryRetrievalMode.FOCUSED:
                strategies = [RetrievalStrategy.IMPORTANCE, RetrievalStrategy.SEMANTIC]
            else:  # Balanced (default)
                strategies = [
                    RetrievalStrategy.SEMANTIC, 
                    RetrievalStrategy.RECENCY, 
                    RetrievalStrategy.IMPORTANCE
                ]
            
            # Execute strategies in parallel
            tasks = []
            for strategy in strategies:
                # Create a strategy-specific request
                strategy_request = RetrievalRequest(
                    query=request.query,
                    session_id=request.session_id,
                    user_id=request.user_id,
                    memory_types=request.memory_types,
                    strategy=strategy,
                    mode=request.mode,
                    time_range=request.time_range,
                    tags=request.tags,
                    max_results=request.max_results,
                    min_relevance=request.min_relevance,
                    context=request.context,
                    context_window=request.context_window,
                    query_embedding=request.query_embedding,
                    use_embeddings=request.use_embeddings,
                    params=request.params
                )
                
                # Add task based on strategy
                if strategy == RetrievalStrategy.SEMANTIC:
                    tasks.append(self._semantic_retrieval(strategy_request))
                elif strategy == RetrievalStrategy.TEMPORAL:
                    tasks.append(self._temporal_retrieval(strategy_request))
                elif strategy == RetrievalStrategy.ASSOCIATIVE:
                    tasks.append(self._associative_retrieval(strategy_request))
                elif strategy == RetrievalStrategy.IMPORTANCE:
                    tasks.append(self._importance_retrieval(strategy_request))
                elif strategy == RetrievalStrategy.RECENCY:
                    tasks.append(self._recency_retrieval(strategy_request))
                elif strategy == RetrievalStrategy.FREQUENCY:
                    tasks.append(self._frequency_retrieval(strategy_request))
            
            # Execute all retrieval strategies in parallel
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    self.logger.error(f"Error in hybrid retrieval strategy: {str(result)}")
                    continue
                
                if isinstance(result, RetrievalResult):
                    # Track strategy
                    strategy = strategies[i]
                    strategies_used.add(strategy)
                    
                    # Update total candidates
                    total_candidates += result.total_candidates
                    
                    # Merge source breakdowns
                    for memory_type, count in result.source_breakdowns.items():
                        source_breakdowns[memory_type] += count
                    
                    # Add items to combined results
                    for memory, score in result.items:
                        # If memory already exists, take max score
                        if memory.memory_id in all_results:
                            existing_score = all_results[memory.memory_id][1]
                            if score > existing_score:
                                all_results[memory.memory_id] = (memory, score, strategy)
                        else:
                            all_results[memory.memory_id] = (memory, score, strategy)
            
            # Convert to list and sort by score
            combined_items = [
                (memory, score) 
                for memory_id, (memory, score, _) in all_results.items()
                if score >= request.min_relevance
            ]
            
            combined_items.sort(key=lambda x: x[1], reverse=True)
            
            # Limit results
            results = combined_items[:request.max_results]
            
            return RetrievalResult(
                items=results,
                request=request,
                retrieval_time=time.time() - start_time,
                total_candidates=total_candidates,
                source_breakdowns=source_breakdowns,
                strategies_used=list(strategies_used)
            )
            
        except Exception as e:
            self.logger.error(f"Hybrid retrieval failed: {str(e)}")
            traceback.print_exc()
            
            # Return empty result
            return RetrievalResult(
                items=[],
                request=request,
                retrieval_time=time.time() - start_time,
                total_candidates=0,
                strategies_used=[RetrievalStrategy.HYBRID]
            )

    
