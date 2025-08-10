"""
Simple Vector Store
Author: Drmusab

Minimal vector store implementation for basic similarity search.
"""

import json
from typing import Any, Dict, List, Optional, Tuple

from .simple_memory import MemoryItem


class VectorStore:
    """Simple vector store for basic semantic search."""

    def __init__(self, dimension: int = 768):
        """Initialize vector store."""
        self.dimension = dimension
        self._vectors: Dict[str, List[float]] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}

    async def add_vector(
        self,
        vector_id: str,
        vector: List[float],
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Add a vector with metadata."""
        try:
            if len(vector) != self.dimension:
                return False
                
            self._vectors[vector_id] = vector
            self._metadata[vector_id] = metadata or {}
            return True
        except Exception:
            return False

    async def get_vector(self, vector_id: str) -> Optional[Tuple[List[float], Dict[str, Any]]]:
        """Get vector and metadata by ID."""
        if vector_id in self._vectors:
            return self._vectors[vector_id], self._metadata.get(vector_id, {})
        return None

    async def similarity_search(
        self,
        query_vector: List[float],
        top_k: int = 10,
        threshold: float = 0.0
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Find similar vectors using cosine similarity."""
        try:
            if len(query_vector) != self.dimension:
                return []

            results = []
            
            for vector_id, vector in self._vectors.items():
                similarity = self._cosine_similarity(query_vector, vector)
                if similarity >= threshold:
                    results.append((
                        vector_id,
                        similarity,
                        self._metadata.get(vector_id, {})
                    ))
            
            # Sort by similarity (descending)
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:top_k]
            
        except Exception:
            return []

    async def delete_vector(self, vector_id: str) -> bool:
        """Delete a vector."""
        try:
            if vector_id in self._vectors:
                del self._vectors[vector_id]
                self._metadata.pop(vector_id, None)
                return True
            return False
        except Exception:
            return False

    async def update_metadata(self, vector_id: str, metadata: Dict[str, Any]) -> bool:
        """Update metadata for a vector."""
        try:
            if vector_id in self._vectors:
                self._metadata[vector_id] = metadata
                return True
            return False
        except Exception:
            return False

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        try:
            # Dot product
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            
            # Magnitudes
            magnitude1 = sum(a * a for a in vec1) ** 0.5
            magnitude2 = sum(b * b for b in vec2) ** 0.5
            
            if magnitude1 == 0 or magnitude2 == 0:
                return 0.0
                
            return dot_product / (magnitude1 * magnitude2)
        except Exception:
            return 0.0

    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        return {
            "total_vectors": len(self._vectors),
            "dimension": self.dimension,
            "memory_usage_mb": self._estimate_memory_usage()
        }

    def _estimate_memory_usage(self) -> float:
        """Estimate memory usage in MB."""
        try:
            # Rough estimation: each float is 8 bytes
            vector_size = len(self._vectors) * self.dimension * 8
            metadata_size = len(json.dumps(self._metadata).encode('utf-8'))
            total_bytes = vector_size + metadata_size
            return round(total_bytes / (1024 * 1024), 2)
        except Exception:
            return 0.0

    async def clear(self) -> bool:
        """Clear all vectors."""
        try:
            self._vectors.clear()
            self._metadata.clear()
            return True
        except Exception:
            return False


# Compatibility aliases
VectorMemoryStore = VectorStore