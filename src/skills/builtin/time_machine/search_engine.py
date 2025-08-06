"""
Search Engine for Context Time Machine
Author: Drmusab  
Last Modified: 2025-01-08

Provides advanced search capabilities across different memory types:
- Semantic search using embeddings
- Keyword-based search
- Time-based filtering
- Cross-modal search integration
- Bilingual search support
"""

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from src.core.dependency_injection import Container
from src.memory.core_memory.base_memory import (
    MemoryItem, MemoryType, SimpleMemoryQuery, MemorySearchResult
)
from src.memory.core_memory.memory_manager import MemoryManager
from src.observability.logging.config import get_logger
from src.processing.natural_language.bilingual_manager import BilingualManager, Language

from .query_parser import ParsedQuery, QueryType, TimeRange


class SearchMode(Enum):
    """Different search modes available."""
    SEMANTIC = "semantic"  # Vector-based similarity search
    KEYWORD = "keyword"    # Traditional keyword matching
    HYBRID = "hybrid"      # Combination of semantic and keyword
    TEMPORAL = "temporal"  # Time-based search
    CONTEXTUAL = "contextual"  # Context-aware search


@dataclass
class SearchResult:
    """Individual search result item."""
    memory_id: str
    content: Any
    memory_type: MemoryType
    relevance_score: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    snippet: str = ""
    highlights: List[str] = field(default_factory=list)


@dataclass 
class SearchResponse:
    """Complete search response with results and metadata."""
    results: List[SearchResult]
    total_count: int
    search_time: float
    query_info: Dict[str, Any]
    suggestions: List[str] = field(default_factory=list)
    aggregations: Dict[str, Any] = field(default_factory=dict)


class SearchEngine:
    """
    Advanced search engine for the Context Time Machine.
    
    Provides semantic search, keyword search, temporal filtering,
    and cross-modal search capabilities with bilingual support.
    """
    
    def __init__(self, container: Container):
        """Initialize the search engine."""
        self.container = container
        self.logger = get_logger(__name__)
        self.memory_manager = container.get(MemoryManager)
        self.bilingual_manager = container.get(BilingualManager)
        
        # Search configuration
        self.max_results = 50
        self.min_relevance_threshold = 0.3
        self.snippet_length = 200
        
        self.logger.info("SearchEngine initialized")
    
    async def search(
        self,
        parsed_query: ParsedQuery,
        mode: SearchMode = SearchMode.HYBRID,
        limit: int = 20,
        offset: int = 0
    ) -> SearchResponse:
        """
        Execute search based on parsed query.
        
        Args:
            parsed_query: Parsed query object
            mode: Search mode to use
            limit: Maximum results to return
            offset: Pagination offset
            
        Returns:
            SearchResponse with results and metadata
        """
        start_time = datetime.now()
        
        try:
            # Route to appropriate search method based on mode
            if mode == SearchMode.SEMANTIC:
                results = await self._semantic_search(parsed_query, limit, offset)
            elif mode == SearchMode.KEYWORD:
                results = await self._keyword_search(parsed_query, limit, offset)
            elif mode == SearchMode.TEMPORAL:
                results = await self._temporal_search(parsed_query, limit, offset)
            elif mode == SearchMode.CONTEXTUAL:
                results = await self._contextual_search(parsed_query, limit, offset)
            else:  # HYBRID
                results = await self._hybrid_search(parsed_query, limit, offset)
            
            # Calculate search time
            search_time = (datetime.now() - start_time).total_seconds()
            
            # Generate suggestions
            suggestions = await self._generate_suggestions(parsed_query, results)
            
            # Generate aggregations
            aggregations = await self._generate_aggregations(results)
            
            response = SearchResponse(
                results=results,
                total_count=len(results),
                search_time=search_time,
                query_info={
                    "original_text": parsed_query.original_text,
                    "language": parsed_query.language.value,
                    "query_type": parsed_query.query_type.value,
                    "search_mode": mode.value
                },
                suggestions=suggestions,
                aggregations=aggregations
            )
            
            self.logger.debug(
                f"Search completed: {len(results)} results in {search_time:.3f}s"
            )
            
            return response
            
        except Exception as e:
            self.logger.error(f"Search failed: {str(e)}")
            return SearchResponse(
                results=[],
                total_count=0,
                search_time=(datetime.now() - start_time).total_seconds(),
                query_info={
                    "original_text": parsed_query.original_text,
                    "error": str(e)
                }
            )
    
    async def _semantic_search(
        self, 
        parsed_query: ParsedQuery, 
        limit: int, 
        offset: int
    ) -> List[SearchResult]:
        """Perform semantic search using embeddings."""
        results = []
        
        try:
            # Get semantic memory items for the query
            semantic_memories = await self.memory_manager.search_memories(
                query=parsed_query.original_text,
                memory_type=MemoryType.SEMANTIC
            )
            
            # Also search episodic memories for conversation context
            episodic_memories = await self.memory_manager.search_memories(
                query=parsed_query.original_text,
                memory_type=MemoryType.EPISODIC
            )
            
            # Combine and process results
            all_memories = semantic_memories.items + episodic_memories.items
            
            for memory_item in all_memories:
                if self._should_include_memory(memory_item, parsed_query):
                    result = await self._create_search_result(memory_item, parsed_query)
                    if result.relevance_score >= self.min_relevance_threshold:
                        results.append(result)
            
            # Sort by relevance score
            results.sort(key=lambda x: x.relevance_score, reverse=True)
            
        except Exception as e:
            self.logger.error(f"Semantic search failed: {str(e)}")
        
        return results[offset:offset + limit]
    
    async def _keyword_search(
        self, 
        parsed_query: ParsedQuery, 
        limit: int, 
        offset: int
    ) -> List[SearchResult]:
        """Perform keyword-based search."""
        results = []
        
        try:
            # Search across different memory types
            memory_types = [MemoryType.EPISODIC, MemoryType.SEMANTIC, MemoryType.WORKING]
            
            for memory_type in memory_types:
                # Create simple memory query
                query = SimpleMemoryQuery(
                    memory_type=memory_type,
                    limit=self.max_results
                )
                
                # Apply time range filter if present
                if parsed_query.time_range:
                    query.time_range = (
                        parsed_query.time_range.start,
                        parsed_query.time_range.end
                    )
                
                memories = await self.memory_manager.memory_store.query(query)
                
                for memory_item in memories:
                    if self._matches_keywords(memory_item, parsed_query):
                        result = await self._create_search_result(memory_item, parsed_query)
                        if result.relevance_score >= self.min_relevance_threshold:
                            results.append(result)
            
            # Sort by relevance and timestamp
            results.sort(key=lambda x: (x.relevance_score, x.timestamp), reverse=True)
            
        except Exception as e:
            self.logger.error(f"Keyword search failed: {str(e)}")
        
        return results[offset:offset + limit]
    
    async def _temporal_search(
        self, 
        parsed_query: ParsedQuery, 
        limit: int, 
        offset: int
    ) -> List[SearchResult]:
        """Perform time-based search."""
        results = []
        
        if not parsed_query.time_range:
            # If no time range specified, default to recent memories
            self.logger.debug("No time range specified, returning recent memories")
            recent_memories = await self.memory_manager.get_recent_memories(limit=limit)
            
            for memory_item in recent_memories:
                result = await self._create_search_result(memory_item, parsed_query)
                results.append(result)
            
            return results[offset:offset + limit]
        
        try:
            # Search with time range filter
            query = SimpleMemoryQuery(
                time_range=(parsed_query.time_range.start, parsed_query.time_range.end),
                limit=self.max_results
            )
            
            memories = await self.memory_manager.memory_store.query(query)
            
            for memory_item in memories:
                result = await self._create_search_result(memory_item, parsed_query)
                results.append(result)
            
            # Sort by timestamp (most recent first)
            results.sort(key=lambda x: x.timestamp, reverse=True)
            
        except Exception as e:
            self.logger.error(f"Temporal search failed: {str(e)}")
        
        return results[offset:offset + limit]
    
    async def _contextual_search(
        self, 
        parsed_query: ParsedQuery, 
        limit: int, 
        offset: int
    ) -> List[SearchResult]:
        """Perform context-aware search."""
        results = []
        
        try:
            # Use the parsed query context to guide search
            if parsed_query.query_type == QueryType.CONVERSATION_RECALL:
                # Focus on episodic memories (conversations)
                memories = await self.memory_manager.search_memories(
                    query=parsed_query.original_text,
                    memory_type=MemoryType.EPISODIC
                )
                all_memories = memories.items
                
            elif parsed_query.query_type == QueryType.PROJECT_MENTION:
                # Search for project-related content across all memory types
                all_memories = []
                for memory_type in [MemoryType.SEMANTIC, MemoryType.EPISODIC]:
                    memories = await self.memory_manager.search_memories(
                        query=" ".join(parsed_query.topics) if parsed_query.topics else parsed_query.original_text,
                        memory_type=memory_type
                    )
                    all_memories.extend(memories.items)
                    
            elif parsed_query.query_type == QueryType.BEHAVIORAL_ANALYSIS:
                # Focus on memories with behavioral metadata
                all_memories = []
                query = SimpleMemoryQuery(limit=self.max_results)
                memories = await self.memory_manager.memory_store.query(query)
                
                # Filter memories that might contain behavioral data
                for memory in memories:
                    if any(aspect in str(memory.content).lower() 
                          for aspect in parsed_query.behavioral_aspects):
                        all_memories.append(memory)
                        
            else:
                # Default to general search
                memories = await self.memory_manager.search_memories(
                    query=parsed_query.original_text
                )
                all_memories = memories.items
            
            # Process all memories
            for memory_item in all_memories:
                if self._should_include_memory(memory_item, parsed_query):
                    result = await self._create_search_result(memory_item, parsed_query)
                    if result.relevance_score >= self.min_relevance_threshold:
                        results.append(result)
            
            # Sort by relevance and recency
            results.sort(key=lambda x: (x.relevance_score, x.timestamp), reverse=True)
            
        except Exception as e:
            self.logger.error(f"Contextual search failed: {str(e)}")
        
        return results[offset:offset + limit]
    
    async def _hybrid_search(
        self, 
        parsed_query: ParsedQuery, 
        limit: int, 
        offset: int
    ) -> List[SearchResult]:
        """Combine multiple search methods for better results."""
        try:
            # Run semantic and keyword searches in parallel
            semantic_task = asyncio.create_task(
                self._semantic_search(parsed_query, limit * 2, 0)
            )
            keyword_task = asyncio.create_task(
                self._keyword_search(parsed_query, limit * 2, 0)
            )
            
            semantic_results, keyword_results = await asyncio.gather(
                semantic_task, keyword_task
            )
            
            # Merge and deduplicate results
            combined_results = {}
            
            # Add semantic results with higher weight
            for result in semantic_results:
                combined_results[result.memory_id] = result
                result.relevance_score *= 1.2  # Boost semantic results
            
            # Add keyword results, combining scores if duplicate
            for result in keyword_results:
                if result.memory_id in combined_results:
                    # Combine scores
                    existing = combined_results[result.memory_id]
                    existing.relevance_score = min(1.0, 
                        existing.relevance_score + result.relevance_score * 0.5
                    )
                else:
                    combined_results[result.memory_id] = result
            
            # Convert to list and sort
            final_results = list(combined_results.values())
            final_results.sort(key=lambda x: x.relevance_score, reverse=True)
            
            return final_results[offset:offset + limit]
            
        except Exception as e:
            self.logger.error(f"Hybrid search failed: {str(e)}")
            # Fallback to keyword search
            return await self._keyword_search(parsed_query, limit, offset)
    
    def _should_include_memory(self, memory_item: MemoryItem, parsed_query: ParsedQuery) -> bool:
        """Determine if a memory item should be included in results."""
        # Apply time range filter
        if parsed_query.time_range:
            memory_time = memory_item.metadata.created_at
            if memory_time < parsed_query.time_range.start or memory_time > parsed_query.time_range.end:
                return False
        
        # Apply topic filter
        if parsed_query.topics:
            memory_content = str(memory_item.content).lower()
            if not any(topic in memory_content for topic in parsed_query.topics):
                return False
        
        return True
    
    def _matches_keywords(self, memory_item: MemoryItem, parsed_query: ParsedQuery) -> bool:
        """Check if memory item matches query keywords."""
        if not parsed_query.extracted_keywords:
            return True
        
        memory_content = str(memory_item.content).lower()
        
        # Check if any keywords match
        return any(keyword in memory_content for keyword in parsed_query.extracted_keywords)
    
    async def _create_search_result(
        self, 
        memory_item: MemoryItem, 
        parsed_query: ParsedQuery
    ) -> SearchResult:
        """Create a SearchResult from a MemoryItem."""
        # Calculate relevance score
        relevance_score = await self._calculate_relevance_score(memory_item, parsed_query)
        
        # Generate snippet
        snippet = await self._generate_snippet(memory_item, parsed_query)
        
        # Generate highlights
        highlights = await self._generate_highlights(memory_item, parsed_query)
        
        return SearchResult(
            memory_id=memory_item.memory_id,
            content=memory_item.content,
            memory_type=memory_item.memory_type,
            relevance_score=relevance_score,
            timestamp=memory_item.metadata.created_at,
            metadata={
                "importance": memory_item.metadata.importance,
                "tags": list(memory_item.metadata.tags),
                "confidence": memory_item.metadata.custom_metadata.get("confidence", 0.0),
                "access_count": memory_item.metadata.access_count
            },
            snippet=snippet,
            highlights=highlights
        )
    
    async def _calculate_relevance_score(
        self, 
        memory_item: MemoryItem, 
        parsed_query: ParsedQuery
    ) -> float:
        """Calculate relevance score for a memory item."""
        score = 0.0
        
        memory_content = str(memory_item.content).lower()
        
        # Keyword matching score
        if parsed_query.extracted_keywords:
            keyword_matches = sum(
                1 for keyword in parsed_query.extracted_keywords 
                if keyword in memory_content
            )
            keyword_score = keyword_matches / len(parsed_query.extracted_keywords)
            score += keyword_score * 0.4
        
        # Topic matching score
        if parsed_query.topics:
            topic_matches = sum(
                1 for topic in parsed_query.topics
                if topic in memory_content
            )
            topic_score = topic_matches / len(parsed_query.topics)
            score += topic_score * 0.3
        
        # Importance and recency boost
        importance_score = memory_item.metadata.importance
        score += importance_score * 0.2
        
        # Recency boost (memories from last week get higher scores)
        days_old = (datetime.now(timezone.utc) - memory_item.metadata.created_at).days
        recency_score = max(0, 1.0 - (days_old / 30))  # Decay over 30 days
        score += recency_score * 0.1
        
        return min(1.0, score)
    
    async def _generate_snippet(
        self, 
        memory_item: MemoryItem, 
        parsed_query: ParsedQuery
    ) -> str:
        """Generate a snippet of the memory content."""
        content = str(memory_item.content)
        
        # If content is short enough, return as is
        if len(content) <= self.snippet_length:
            return content
        
        # Try to find content around keywords
        if parsed_query.extracted_keywords:
            for keyword in parsed_query.extracted_keywords:
                keyword_pos = content.lower().find(keyword)
                if keyword_pos >= 0:
                    start = max(0, keyword_pos - self.snippet_length // 2)
                    end = start + self.snippet_length
                    snippet = content[start:end]
                    
                    if start > 0:
                        snippet = "..." + snippet
                    if end < len(content):
                        snippet = snippet + "..."
                    
                    return snippet
        
        # Default to beginning of content
        snippet = content[:self.snippet_length]
        if len(content) > self.snippet_length:
            snippet += "..."
        
        return snippet
    
    async def _generate_highlights(
        self, 
        memory_item: MemoryItem, 
        parsed_query: ParsedQuery
    ) -> List[str]:
        """Generate highlighted terms in the content."""
        highlights = []
        content = str(memory_item.content).lower()
        
        # Highlight keywords
        for keyword in parsed_query.extracted_keywords:
            if keyword in content:
                highlights.append(keyword)
        
        # Highlight topics
        for topic in parsed_query.topics:
            if topic in content:
                highlights.append(topic)
        
        return list(set(highlights))  # Remove duplicates
    
    async def _generate_suggestions(
        self, 
        parsed_query: ParsedQuery, 
        results: List[SearchResult]
    ) -> List[str]:
        """Generate search suggestions based on results."""
        suggestions = []
        
        # Extract common topics from results
        topic_counts = {}
        for result in results[:10]:  # Look at top 10 results
            content = str(result.content).lower()
            words = content.split()
            
            for word in words:
                if len(word) > 4:  # Only consider longer words
                    topic_counts[word] = topic_counts.get(word, 0) + 1
        
        # Get most common topics
        common_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Generate suggestions in appropriate language
        if parsed_query.language == Language.ARABIC:
            for topic, count in common_topics:
                suggestions.append(f"ابحث عن {topic}")
        else:
            for topic, count in common_topics:
                suggestions.append(f"search for {topic}")
        
        return suggestions
    
    async def _generate_aggregations(self, results: List[SearchResult]) -> Dict[str, Any]:
        """Generate aggregations from search results."""
        aggregations = {}
        
        # Memory type distribution
        memory_type_counts = {}
        for result in results:
            memory_type = result.memory_type.value
            memory_type_counts[memory_type] = memory_type_counts.get(memory_type, 0) + 1
        
        aggregations["memory_types"] = memory_type_counts
        
        # Time distribution (by day)
        time_counts = {}
        for result in results:
            date_key = result.timestamp.strftime("%Y-%m-%d")
            time_counts[date_key] = time_counts.get(date_key, 0) + 1
        
        aggregations["timeline"] = time_counts
        
        # Tag distribution
        tag_counts = {}
        for result in results:
            tags = result.metadata.get("tags", [])
            for tag in tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
        
        aggregations["tags"] = tag_counts
        
        return aggregations