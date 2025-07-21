"""
Vector Storage System for Memory Embeddings
Author: Drmusab
Last Modified: 2025-07-05 09:50:18 UTC

This module provides a vector-based storage system for the AI assistant's memory,
enabling efficient storage and similarity-based retrieval of embedding vectors.
It supports multiple backend implementations including in-memory, file-based,
and integration with specialized vector databases.
"""

from pathlib import Path
from typing import Optional, Dict, Any, List, Set, Tuple, Union, TypeVar, Generic, Callable
import asyncio
import time
import json
import pickle
import numpy as np
import logging
import os
import shutil
from datetime import datetime, timezone
import heapq
from concurrent.futures import ThreadPoolExecutor
import traceback
import uuid
import math
from enum import Enum

# Core imports
from src.core.config.loader import ConfigLoader
from src.core.events.event_bus import EventBus
from src.core.events.event_types import (
    MemoryItemStored, MemoryItemRetrieved, MemoryItemUpdated, MemoryItemDeleted,
    VectorIndexUpdated, VectorSearchStarted, VectorSearchCompleted,
    VectorIndexRebuildStarted, VectorIndexRebuildCompleted,
    VectorStoreBackupStarted, VectorStoreBackupCompleted,
    VectorStoreRestoreStarted, VectorStoreRestoreCompleted,
    VectorStoreCacheHit, VectorStoreCacheMiss, ErrorOccurred
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
from src.integrations.llm.model_router import ModelRouter

# Observability
from src.observability.monitoring.metrics import MetricsCollector
from src.observability.monitoring.tracing import TraceManager
from src.observability.logging.config import get_logger

# Optional integration - try to import vector database libraries
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    import hnswlib
    HNSW_AVAILABLE = True
except ImportError:
    HNSW_AVAILABLE = False

try:
    import annoy
    ANNOY_AVAILABLE = True
except ImportError:
    ANNOY_AVAILABLE = False


class VectorIndexType(Enum):
    """Types of vector indices supported."""
    FLAT = "flat"                # Simple brute-force, exact search
    IVF_FLAT = "ivf_flat"        # Inverted file index with flat storage
    IVF_PQ = "ivf_pq"            # Inverted file with product quantization
    HNSW = "hnsw"                # Hierarchical Navigable Small World graphs
    ANNOY = "annoy"              # Approximate Nearest Neighbors Oh Yeah
    LSH = "lsh"                  # Locality-Sensitive Hashing
    NONE = "none"                # No index, brute force search


class VectorDistanceMetric(Enum):
    """Distance metrics for vector similarity."""
    COSINE = "cosine"            # Cosine similarity (1 - cosine distance)
    EUCLIDEAN = "euclidean"      # Euclidean distance (L2 norm)
    DOT_PRODUCT = "dot_product"  # Dot product (inner product)
    ANGULAR = "angular"          # Angular distance
    MANHATTAN = "manhattan"      # Manhattan distance (L1 norm)
    HAMMING = "hamming"          # Hamming distance (for binary vectors)


class VectorStoreConfig:
    """Configuration settings for vector store."""
    
    def __init__(self, config_loader: ConfigLoader):
        """
        Initialize vector store configuration.
        
        Args:
            config_loader: Configuration loader
        """
        vector_config = config_loader.get("memory.vector_store", {})
        
        # General settings
        self.dimension = vector_config.get("dimension", 1536)  # Default for many embedding models
        self.default_index_type = VectorIndexType(vector_config.get("index_type", "flat"))
        self.distance_metric = VectorDistanceMetric(vector_config.get("distance_metric", "cosine"))
        
        # Storage settings
        self.storage_type = vector_config.get("storage_type", "memory")  # memory, file, database
        self.file_path = vector_config.get("file_path", "data/cache/vector_cache")
        self.database_table = vector_config.get("database_table", "vector_embeddings")
        
        # Cache settings
        self.cache_enabled = vector_config.get("cache_enabled", True)
        self.cache_size = vector_config.get("cache_size", 1000)
        self.cache_ttl = vector_config.get("cache_ttl", 3600)  # seconds
        
        # Performance settings
        self.batch_size = vector_config.get("batch_size", 100)
        self.num_threads = vector_config.get("num_threads", max(1, os.cpu_count() // 2))
        self.rebuild_threshold = vector_config.get("rebuild_threshold", 0.2)  # Rebuild when 20% of items changed
        
        # Index-specific settings
        self.hnsw_ef_construction = vector_config.get("hnsw_ef_construction", 200)
        self.hnsw_m = vector_config.get("hnsw_m", 16)
        self.hnsw_ef_search = vector_config.get("hnsw_ef_search", 100)
        
        self.ivf_nlist = vector_config.get("ivf_nlist", 100)
        self.pq_m = vector_config.get("pq_m", 8)
        
        self.annoy_n_trees = vector_config.get("annoy_n_trees", 100)
        
        # Backup settings
        self.backup_enabled = vector_config.get("backup_enabled", True)
        self.backup_interval = vector_config.get("backup_interval", 86400)  # seconds (daily)
        self.backup_path = vector_config.get("backup_path", "data/backups/vector_store")
        self.max_backups = vector_config.get("max_backups", 5)


class VectorIndex:
    """
    Abstract vector index implementation.
    
    This class provides a common interface for different vector index implementations,
    allowing the VectorMemoryStore to switch between them as needed.
    """
    
    def __init__(self, dimension: int, metric: VectorDistanceMetric):
        """
        Initialize vector index.
        
        Args:
            dimension: Vector dimension
            metric: Distance metric to use
        """
        self.dimension = dimension
        self.metric = metric
        self.is_initialized = False
        self.logger = get_logger(__name__)
    
    async def initialize(self) -> None:
        """Initialize the index."""
        self.is_initialized = True
    
    async def add(self, id: str, vector: List[float]) -> None:
        """
        Add a vector to the index.
        
        Args:
            id: Vector identifier
            vector: Vector to add
        """
        raise NotImplementedError("Subclasses must implement add()")
    
    async def remove(self, id: str) -> None:
        """
        Remove a vector from the index.
        
        Args:
            id: Vector identifier
        """
        raise NotImplementedError("Subclasses must implement remove()")
    
    async def search(self, query_vector: List[float], k: int) -> List[Tuple[str, float]]:
        """
        Search for similar vectors.
        
        Args:
            query_vector: Query vector
            k: Number of results to return
            
        Returns:
            List of (id, similarity) tuples
        """
        raise NotImplementedError("Subclasses must implement search()")
    
    async def rebuild(self, vectors: Dict[str, List[float]]) -> None:
        """
        Rebuild the index with new vectors.
        
        Args:
            vectors: Dictionary of id -> vector
        """
        raise NotImplementedError("Subclasses must implement rebuild()")
    
    async def save(self, path: Path) -> None:
        """
        Save the index to disk.
        
        Args:
            path: Path to save to
        """
        raise NotImplementedError("Subclasses must implement save()")
    
    async def load(self, path: Path) -> None:
        """
        Load the index from disk.
        
        Args:
            path: Path to load from
        """
        raise NotImplementedError("Subclasses must implement load()")
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get index statistics.
        
        Returns:
            Dictionary of statistics
        """
        return {
            "dimension": self.dimension,
            "metric": self.metric.value,
            "is_initialized": self.is_initialized,
            "type": self.__class__.__name__
        }


class FlatVectorIndex(VectorIndex):
    """
    Simple flat vector index with brute-force search.
    
    This index stores vectors in memory and performs exhaustive search.
    Suitable for small to medium datasets (up to ~100K vectors).
    """
    
    def __init__(self, dimension: int, metric: VectorDistanceMetric):
        """
        Initialize flat vector index.
        
        Args:
            dimension: Vector dimension
            metric: Distance metric to use
        """
        super().__init__(dimension, metric)
        self.vectors: Dict[str, np.ndarray] = {}
        self.vector_count = 0
    
    async def add(self, id: str, vector: List[float]) -> None:
        """
        Add a vector to the index.
        
        Args:
            id: Vector identifier
            vector: Vector to add
        """
        if not self.is_initialized:
            await self.initialize()
        
        # Convert to numpy array if needed
        if not isinstance(vector, np.ndarray):
            vector = np.array(vector, dtype=np.float32)
        
        # Normalize for cosine similarity if needed
        if self.metric == VectorDistanceMetric.COSINE:
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm
        
        self.vectors[id] = vector
        self.vector_count += 1
    
    async def remove(self, id: str) -> None:
        """
        Remove a vector from the index.
        
        Args:
            id: Vector identifier
        """
        if id in self.vectors:
            del self.vectors[id]
            self.vector_count -= 1
    
    async def search(self, query_vector: List[float], k: int) -> List[Tuple[str, float]]:
        """
        Search for similar vectors.
        
        Args:
            query_vector: Query vector
            k: Number of results to return
            
        Returns:
            List of (id, similarity) tuples
        """
        if not self.vectors:
            return []
        
        # Convert to numpy array if needed
        if not isinstance(query_vector, np.ndarray):
            query_vector = np.array(query_vector, dtype=np.float32)
        
        # Normalize for cosine similarity if needed
        if self.metric == VectorDistanceMetric.COSINE:
            norm = np.linalg.norm(query_vector)
            if norm > 0:
                query_vector = query_vector / norm
        
        # Calculate distances based on metric
        similarities = []
        for id, vector in self.vectors.items():
            if self.metric == VectorDistanceMetric.COSINE:
                # Cosine similarity: 1 - cosine distance
                similarity = float(np.dot(query_vector, vector))
            elif self.metric == VectorDistanceMetric.EUCLIDEAN:
                # Convert Euclidean distance to similarity (closer = higher similarity)
                distance = float(np.linalg.norm(query_vector - vector))
                similarity = 1.0 / (1.0 + distance)
            elif self.metric == VectorDistanceMetric.DOT_PRODUCT:
                # Dot product as similarity
                similarity = float(np.dot(query_vector, vector))
            elif self.metric == VectorDistanceMetric.ANGULAR:
                # Angular similarity
                similarity = 1.0 - (np.arccos(
                    min(1.0, max(-1.0, np.dot(query_vector, vector)))
                ) / np.pi)
            elif self.metric == VectorDistanceMetric.MANHATTAN:
                # Convert Manhattan distance to similarity
                distance = float(np.sum(np.abs(query_vector - vector)))
                similarity = 1.0 / (1.0 + distance)
            elif self.metric == VectorDistanceMetric.HAMMING:
                # Convert Hamming distance to similarity
                distance = float(np.sum(query_vector != vector))
                similarity = 1.0 - (distance / len(query_vector))
            else:
                # Default to cosine
                similarity = float(np.dot(query_vector, vector))
            
            similarities.append((id, similarity))
        
        # Sort by similarity (descending) and return top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]
    
    async def rebuild(self, vectors: Dict[str, List[float]]) -> None:
        """
        Rebuild the index with new vectors.
        
        Args:
            vectors: Dictionary of id -> vector
        """
        self.vectors = {}
        self.vector_count = 0
        
        for id, vector in vectors.items():
            await self.add(id, vector)
    
    async def save(self, path: Path) -> None:
        """
        Save the index to disk.
        
        Args:
            path: Path to save to
        """
        # Create directory if it doesn't exist
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save vectors to file
        with open(path, 'wb') as f:
            pickle.dump(self.vectors, f)
    
    async def load(self, path: Path) -> None:
        """
        Load the index from disk.
        
        Args:
            path: Path to load from
        """
        if path.exists():
            with open(path, 'rb') as f:
                self.vectors = pickle.load(f)
                self.vector_count = len(self.vectors)
            self.is_initialized = True
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get index statistics.
        
        Returns:
            Dictionary of statistics
        """
        stats = await super().get_stats()
        stats.update({
            "vector_count": self.vector_count,
            "memory_usage_mb": sum(v.nbytes for v in self.vectors.values()) / (1024 * 1024)
        })
        return stats


class HNSWVectorIndex(VectorIndex):
    """
    HNSW (Hierarchical Navigable Small World) vector index.
    
    This index uses Hierarchical Navigable Small World graphs for approximate
    nearest neighbor search. Suitable for large datasets with high performance
    requirements.
    """
    
    def __init__(self, dimension: int, metric: VectorDistanceMetric, config: VectorStoreConfig):
        """
        Initialize HNSW vector index.
        
        Args:
            dimension: Vector dimension
            metric: Distance metric to use
            config: Vector store configuration
        """
        super().__init__(dimension, metric)
        self.config = config
        self.id_to_index: Dict[str, int] = {}
        self.index_to_id: Dict[int, str] = {}
        self.next_index = 0
        self.index = None
        
        if not HNSW_AVAILABLE:
            self.logger.warning("HNSWlib not available, falling back to flat index")
    
    async def initialize(self) -> None:
        """Initialize the index."""
        if HNSW_AVAILABLE:
            # Convert metric to HNSW metric
            if self.metric == VectorDistanceMetric.COSINE:
                space = 'cosine'
            elif self.metric == VectorDistanceMetric.EUCLIDEAN:
                space = 'l2'
            elif self.metric == VectorDistanceMetric.DOT_PRODUCT:
                space = 'ip'  # inner product
            else:
                space = 'l2'  # default
            
            self.index = hnswlib.Index(space=space, dim=self.dimension)
            self.index.init_index(
                max_elements=1000,  # Will be resized as needed
                ef_construction=self.config.hnsw_ef_construction,
                M=self.config.hnsw_m
            )
            self.index.set_ef(self.config.hnsw_ef_search)
            
            self.is_initialized = True
        else:
            # Fall back to flat index
            self.index = FlatVectorIndex(self.dimension, self.metric)
            await self.index.initialize()
    
    async def add(self, id: str, vector: List[float]) -> None:
        """
        Add a vector to the index.
        
        Args:
            id: Vector identifier
            vector: Vector to add
        """
        if not self.is_initialized:
            await self.initialize()
        
        if isinstance(self.index, FlatVectorIndex):
            await self.index.add(id, vector)
            return
        
        # Convert to numpy array if needed
        if not isinstance(vector, np.ndarray):
            vector = np.array(vector, dtype=np.float32)
        
        # Check if we need to resize the index
        if self.next_index >= self.index.get_max_elements():
            new_size = max(1000, self.index.get_max_elements() * 2)
            self.index.resize_index(new_size)
        
        # Add to index
        self.index.add_items(vector, self.next_index)
        
        # Update id mappings
        self.id_to_index[id] = self.next_index
        self.index_to_id[self.next_index] = id
        
        self.next_index += 1
    
    async def remove(self, id: str) -> None:
        """
        Remove a vector from the index.
        
        Args:
            id: Vector identifier
        """
        if isinstance(self.index, FlatVectorIndex):
            await self.index.remove(id)
            return
        
        if id in self.id_to_index:
            index = self.id_to_index[id]
            
            # HNSW doesn't support true removal, but we can mark it as removed
            # by removing from our id mappings
            del self.id_to_index[id]
            del self.index_to_id[index]
    
    async def search(self, query_vector: List[float], k: int) -> List[Tuple[str, float]]:
        """
        Search for similar vectors.
        
        Args:
            query_vector: Query vector
            k: Number of results to return
            
        Returns:
            List of (id, similarity) tuples
        """
        if isinstance(self.index, FlatVectorIndex):
            return await self.index.search(query_vector, k)
        
        if not self.is_initialized or self.next_index == 0:
            return []
        
        # Adjust k based on how many vectors we actually have
        k = min(k, self.next_index)
        
        # Convert to numpy array if needed
        if not isinstance(query_vector, np.ndarray):
            query_vector = np.array(query_vector, dtype=np.float32)
        
        # Perform search
        labels, distances = self.index.knn_query(query_vector, k=k)
        
        # Convert to results
        results = []
        for i in range(len(labels[0])):
            index = labels[0][i]
            distance = distances[0][i]
            
            # Skip if the index was marked as removed
            if index not in self.index_to_id:
                continue
            
            id = self.index_to_id[index]
            
            # Convert distance to similarity based on metric
            if self.metric == VectorDistanceMetric.COSINE:
                # HNSW cosine distance is already 1 - cosine similarity
                similarity = 1.0 - distance
            elif self.metric == VectorDistanceMetric.DOT_PRODUCT:
                # For inner product, higher is more similar
                similarity = distance
            else:
                # For distance metrics, convert to similarity (closer = higher similarity)
                similarity = 1.0 / (1.0 + distance)
            
            results.append((id, similarity))
        
        return results
    
    async def rebuild(self, vectors: Dict[str, List[float]]) -> None:
        """
        Rebuild the index with new vectors.
        
        Args:
            vectors: Dictionary of id -> vector
        """
        if isinstance(self.index, FlatVectorIndex):
            await self.index.rebuild(vectors)
            return
        
        # Reset index
        self.id_to_index = {}
        self.index_to_id = {}
        self.next_index = 0
        
        # Reinitialize
        await self.initialize()
        
        # Add all vectors
        for id, vector in vectors.items():
            await self.add(id, vector)
    
    async def save(self, path: Path) -> None:
        """
        Save the index to disk.
        
        Args:
            path: Path to save to
        """
        if isinstance(self.index, FlatVectorIndex):
            await self.index.save(path)
            return
        
        # Create directory if it doesn't exist
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save index
        index_path = path.with_suffix('.hnswlib')
        self.index.save_index(str(index_path))
        
        # Save id mappings
        mappings_path = path.with_suffix('.mappings')
        with open(mappings_path, 'wb') as f:
            pickle.dump({
                'id_to_index': self.id_to_index,
                'index_to_id': self.index_to_id,
                'next_index': self.next_index
            }, f)
    
    async def load(self, path: Path) -> None:
        """
        Load the index from disk.
        
        Args:
            path: Path to load from
        """
        if isinstance(self.index, FlatVectorIndex):
            await self.index.load(path)
            return
        
        index_path = path.with_suffix('.hnswlib')
        mappings_path = path.with_suffix('.mappings')
        
        if index_path.exists() and mappings_path.exists():
            # Initialize index
            await self.initialize()
            
            # Load id mappings
            with open(mappings_path, 'rb') as f:
                mappings = pickle.load(f)
                self.id_to_index = mappings['id_to_index']
                self.index_to_id = mappings['index_to_id']
                self.next_index = mappings['next_index']
            
            # Load index
            max_elements = max(1000, self.next_index * 2)
            self.index.resize_index(max_elements)
            self.index.load_index(str(index_path), max_elements)
            
            self.is_initialized = True
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get index statistics.
        
        Returns:
            Dictionary of statistics
        """
        if isinstance(self.index, FlatVectorIndex):
            return await self.index.get_stats()
        
        stats = await super().get_stats()
        stats.update({
            "vector_count": len(self.id_to_index),
            "max_elements": self.index.get_max_elements() if self.is_initialized else 0,
            "ef_construction": self.config.hnsw_ef_construction,
            "ef_search": self.config.hnsw_ef_search,
            "M": self.config.hnsw_m
        })
        return stats


class FAISSVectorIndex(VectorIndex):
    """
    FAISS vector index implementation.
    
    This index uses the FAISS library for efficient similarity search.
    Supports multiple index types and runs on CPU or GPU.
    """
    
    def __init__(self, dimension: int, metric: VectorDistanceMetric, config: VectorStoreConfig):
        """
        Initialize FAISS vector index.
        
        Args:
            dimension: Vector dimension
            metric: Distance metric to use
            config: Vector store configuration
        """
        super().__init__(dimension, metric)
        self.config = config
        self.id_to_index: Dict[str, int] = {}
        self.index_to_id: Dict[int, str] = {}
        self.next_index = 0
        self.index = None
        
        if not FAISS_AVAILABLE:
            self.logger.warning("FAISS not available, falling back to flat index")
    
    async def initialize(self) -> None:
        """Initialize the index."""
        if FAISS_AVAILABLE:
            # Set metric
            if self.metric == VectorDistanceMetric.COSINE:
                faiss_metric = faiss.METRIC_INNER_PRODUCT  # Will normalize vectors before adding
            elif self.metric == VectorDistanceMetric.DOT_PRODUCT:
                faiss_metric = faiss.METRIC_INNER_PRODUCT
            else:
                faiss_metric = faiss.METRIC_L2  # Default to L2 (Euclidean)
            
            # Create index based on configuration
            if self.config.default_index_type == VectorIndexType.FLAT:
                self.index = faiss.IndexFlat(self.dimension, faiss_metric)
            
            elif self.config.default_index_type == VectorIndexType.IVF_FLAT:
                # IVF needs a training set, so we'll initialize but not train yet
                quantizer = faiss.IndexFlat(self.dimension, faiss_metric)
                self.index = faiss.IndexIVFFlat(
                    quantizer, self.dimension, self.config.ivf_nlist, faiss_metric
                )
                self.index.nprobe = min(10, self.config.ivf_nlist)  # Number of clusters to visit during search
            
            elif self.config.default_index_type == VectorIndexType.IVF_PQ:
                # Product Quantization with IVF
                quantizer = faiss.IndexFlat(self.dimension, faiss_metric)
                self.index = faiss.IndexIVFPQ(
                    quantizer, self.dimension, self.config.ivf_nlist, 
                    self.config.pq_m, 8  # 8 bits per subquantizer
                )
                self.index.nprobe = min(10, self.config.ivf_nlist)
            
            else:
                # Default to flat
                self.index = faiss.IndexFlat(self.dimension, faiss_metric)
            
            self.is_initialized = True
        else:
            # Fall back to flat index
            self.index = FlatVectorIndex(self.dimension, self.metric)
            await self.index.initialize()
    
    async def add(self, id: str, vector: List[float]) -> None:
        """
        Add a vector to the index.
        
        Args:
            id: Vector identifier
            vector: Vector to add
        """
        if not self.is_initialized:
            await self.initialize()
        
        if isinstance(self.index, FlatVectorIndex):
            await self.index.add(id, vector)
            return
        
        # Convert to numpy array if needed
        if not isinstance(vector, np.ndarray):
            vector = np.array(vector, dtype=np.float32).reshape(1, -1)
        elif len(vector.shape) == 1:
            vector = vector.reshape(1, -1)
        
        # Normalize for cosine similarity if needed
        if self.metric == VectorDistanceMetric.COSINE:
            faiss.normalize_L2(vector)
        
        # Check if index needs training
        if not self.index.is_trained and self.next_index == 0:
            if isinstance(self.index, (faiss.IndexIVFFlat, faiss.IndexIVFPQ)):
                self.logger.info("Training FAISS IVF index...")
                self.index.train(vector)
        
        # Add to index
        self.index.add(vector)
        
        # Update id mappings
        self.id_to_index[id] = self.next_index
        self.index_to_id[self.next_index] = id
        
        self.next_index += 1
    
    async def remove(self, id: str) -> None:
        """
        Remove a vector from the index.
        
        Args:
            id: Vector identifier
        """
        if isinstance(self.index, FlatVectorIndex):
            await self.index.remove(id)
            return
        
        if id in self.id_to_index:
            # FAISS doesn't support direct removal, so we'll handle it through our id mappings
            del self.id_to_index[id]
            # Note: index_to_id entry is kept for indexing integrity, but marked as not found during search
    
    async def search(self, query_vector: List[float], k: int) -> List[Tuple[str, float]]:
        """
        Search for similar vectors.
        
        Args:
            query_vector: Query vector
            k: Number of results to return
            
        Returns:
            List of (id, similarity) tuples
        """
        if isinstance(self.index, FlatVectorIndex):
            return await self.index.search(query_vector, k)
        
        if not self.is_initialized or self.next_index == 0:
            return []
        
        # Adjust k based on how many vectors we actually have
        k = min(k, self.next_index)
        
        # Convert to numpy array if needed
        if not isinstance(query_vector, np.ndarray):
            query_vector = np.array(query_vector, dtype=np.float32).reshape(1, -1)
        elif len(query_vector.shape) == 1:
            query_vector = query_vector.reshape(1, -1)
        
        # Normalize for cosine similarity if needed
        if self.metric == VectorDistanceMetric.COSINE:
            faiss.normalize_L2(query_vector)
        
        # Perform search
        distances, indices = self.index.search(query_vector, k)
        
        # Convert to results
        results = []
        for i in range(len(indices[0])):
            if indices[0][i] == -1:  # FAISS returns -1 for not found
                continue
                
            index = indices[0][i]
            distance = distances[0][i]
            
            # Skip if the index was marked as removed
            if index not in self.index_to_id:
                continue
            
            id = self.index_to_id[index]
            
            # Skip if id was removed
            if id not in self.id_to_index:
                continue
            
            # Convert distance to similarity based on metric
            if self.metric == VectorDistanceMetric.COSINE:
                # For normalized vectors with inner product, distance is already similarity
                similarity = distance
            elif self.metric == VectorDistanceMetric.DOT_PRODUCT:
                # For inner product, higher is more similar
                similarity = distance
            else:
                # For L2 distance, convert to similarity (closer = higher similarity)
                similarity = 1.0 / (1.0 + distance)
            
            results.append((id, similarity))
        
        return results
    
    async def rebuild(self, vectors: Dict[str, List[float]]) -> None:
        """
        Rebuild the index with new vectors.
        
        Args:
            vectors: Dictionary of id -> vector
        """
        if isinstance(self.index, FlatVectorIndex):
            await self.index.rebuild(vectors)
            return
        
        # Reset index
        self.id_to_index = {}
        self.index_to_id = {}
        self.next_index = 0
        
        # Reinitialize
        await self.initialize()
        
        # Prepare batch of vectors for training
        if vectors and isinstance(self.index, (faiss.IndexIVFFlat, faiss.IndexIVFPQ)):
            # Convert first to numpy arrays
            vector_list = []
            for id, vec in vectors.items():
                if not isinstance(vec, np.ndarray):
                    vec = np.array(vec, dtype=np.float32)
                vector_list.append(vec)
            
            if vector_list:
                # Create training data
                train_vectors = np.vstack(vector_list)
                
                # Normalize for cosine similarity if needed
                if self.metric == VectorDistanceMetric.COSINE:
                    faiss.normalize_L2(train_vectors)
                
                # Train index
                self.logger.info(f"Training FAISS index with {len(train_vectors)} vectors...")
                self.index.train(train_vectors)
        
        # Add all vectors
        for id, vector in vectors.items():
            await self.add(id, vector)
    
    async def save(self, path: Path) -> None:
        """
        Save the index to disk.
        
        Args:
            path: Path to save to
        """
        if isinstance(self.index, FlatVectorIndex):
            await self.index.save(path)
            return
        
        # Create directory if it doesn't exist
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save index
        index_path = path.with_suffix('.faissindex')
        faiss.write_index(self.index, str(index_path))
        
        # Save id mappings
        mappings_path = path.with_suffix('.mappings')
        with open(mappings_path, 'wb') as f:
            pickle.dump({
                'id_to_index': self.id_to_index,
                'index_to_id': self.index_to_id,
                'next_index': self.next_index
            }, f)
    
    async def load(self, path: Path) -> None:
        """
        Load the index from disk.
        
        Args:
            path: Path to load from
        """
        if isinstance(self.index, FlatVectorIndex):
            await self.index.load(path)
            return
        
        index_path = path.with_suffix('.faissindex')
        mappings_path = path.with_suffix('.mappings')
        
        if index_path.exists() and mappings_path.exists():
            # Load index
            self.index = faiss.read_index(str(index_path))
            
            # Load id mappings
            with open(mappings_path, 'rb') as f:
                mappings = pickle.load(f)
                self.id_to_index = mappings['id_to_index']
                self.index_to_id = mappings['index_to_id']
                self.next_index = mappings['next_index']
            
            self.is_initialized = True
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get index statistics.
        
        Returns:
            Dictionary of statistics
        """
        if isinstance(self.index, FlatVectorIndex):
            return await self.index.get_stats()
        
        stats = await super().get_stats()
        stats.update({
            "vector_count": len(self.id_to_index),
            "index_type": self.config.default_index_type.value,
            "is_trained": getattr(self.index, 'is_trained', True)
        })
        
        # Add index-specific stats
        if isinstance(self.index, faiss.IndexIVFFlat):
            stats.update({
                "nlist": self.config.ivf_nlist,
                "nprobe": self.index.nprobe
            })
        elif isinstance(self.index, faiss.IndexIVFPQ):
            stats.update({
                "nlist": self.config.ivf_nlist,
                "nprobe": self.index.nprobe,
                "pq_m": self.config.pq_m
            })
        
        return stats


class VectorMemoryStore(BaseMemoryStore):
    """
    Vector-based memory storage system.
    
    This class provides an implementation of BaseMemoryStore that uses vector
    embeddings for efficient similarity-based retrieval. It supports multiple
    backend implementations and index types for different use cases.
    
    Features:
    - Efficient similarity search for semantic retrieval
    - Support for multiple vector index implementations (FAISS, HNSW, etc.)
    - Automatic index rebuilding and optimization
    - Persistence and serialization
    - Caching for frequent access patterns
    - Monitoring and performance tracking
    """
    
    def __init__(self, container: Container):
        """
        Initialize vector memory store.
        
        Args:
            container: Dependency injection container
        """
        self.container = container
        self.logger = get_logger(__name__)
        
        # Load configuration
        self.config_loader = container.get(ConfigLoader)
        self.config = VectorStoreConfig(self.config_loader)
        
        # Core integrations
        self.event_bus = container.get(EventBus)
        self.error_handler = container.get(ErrorHandler)
        self.health_check = container.get(HealthCheck)
        
        # Try to get optional components
        try:
            self.model_router = container.get(ModelRouter)
        except Exception:
            self.logger.warning("ModelRouter not available, embedding generation will be disabled")
            self.model_router = None
        
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
        self._memory_items: Dict[str, MemoryItem] = {}
        self._vector_index = None
        self._vector_map: Dict[str, List[float]] = {}
        self._memory_types: Dict[MemoryType, Set[str]] = defaultdict(set)
        
        # Cache
        self._item_cache: Dict[str, MemoryItem] = {}
        self._max_cache_size = self.config.cache_size
        
        # Statistics
        self._last_rebuild_time = None
        self._operation_counts = defaultdict(int)
        self._update_count_since_rebuild = 0
        
        # Threading
        self._executor = ThreadPoolExecutor(max_workers=self.config.num_threads)
        
        # Register health check
        self.health_check.register_component("vector_store", self._health_check_callback)
        
        self.logger.info("VectorMemoryStore initialized")

    async def initialize(self) -> None:
        """Initialize the vector store."""
        try:
            # Create index based on configuration
            if FAISS_AVAILABLE and self.config.default_index_type in [
                VectorIndexType.FLAT, VectorIndexType.IVF_FLAT, VectorIndexType.IVF_PQ
            ]:
                self._vector_index = FAISSVectorIndex(self.config.dimension, self.config.distance_metric, self.config)
            elif HNSW_AVAILABLE and self.config.default_index_type == VectorIndexType.HNSW:
                self._vector_index = HNSWVectorIndex(self.config.dimension, self.config.distance_metric, self.config)
            else:
                self._vector_index = FlatVectorIndex(self.config.dimension, self.config.distance_metric)
            
            # Initialize index
            await self._vector_index.initialize()
            
            # Load data from storage
            await self._load_from_storage()
            
            # Register metrics
            if self.metrics:
                self.metrics.register_counter("vector_store_operations_total")
                self.metrics.register_counter("vector_store_items_stored")
                self.metrics.register_counter("vector_store_items_retrieved")
                self.metrics.register_counter("vector_store_search_operations")
                self.metrics.register_histogram("vector_store_operation_duration_seconds")
                self.metrics.register_histogram("vector_store_search_duration_seconds")
                self.metrics.register_gauge("vector_store_items_count")
                self.metrics.register_gauge("vector_store_cache_size")
            
            # Start background tasks
            if self.config.backup_enabled:
                asyncio.create_task(self._backup_loop())
            
            self.logger.info("VectorMemoryStore initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize vector store: {str(e)}")
            traceback.print_exc()
            await self.event_bus.emit(ErrorOccurred(
                component="vector_store",
                error_type=type(e).__name__,
                error_message=str(e),
                severity="critical"
            ))

    async def store_item(self, item: MemoryItem) -> None:
        """
        Store a memory item.
        
        Args:
            item: Memory item to store
        """
        start_time = time.time()
        self._operation_counts['store'] += 1
        
        try:
            # Generate embeddings if needed
            if not item.embeddings and self.model_router:
                try:
                    # Convert to text for embedding
                    if isinstance(item.content, dict):
                        embed_text = json.dumps(item.content)
                    elif not isinstance(item.content, str):
                        embed_text = str(item.content)
                    else:
                        embed_text = item.content
                    
                    # Generate embeddings
                    item.embeddings = await self.model_router.get_embeddings(embed_text)
                except Exception as e:
                    self.logger.warning(f"Failed to generate embeddings: {str(e)}")
            
            # Store item
            self._memory_items[item.memory_id] = item
            
            # Add to memory type index
            self._memory_types[item.memory_type].add(item.memory_id)
            
            # Add to vector index if embeddings available
            if item.embeddings:
                self._vector_map[item.memory_id] = item.embeddings
                await self._vector_index.add(item.memory_id, item.embeddings)
            
            # Add to cache
            if self.config.cache_enabled:
                self._add_to_cache(item)
            
            # Track modifications for index rebuilding
            self._update_count_since_rebuild += 1
            
            # Persist to storage
            if self.config.storage_type == 'file':
                await self._persist_item(item)
            elif self.config.storage_type == 'database' and self.database:
                await self._store_in_database(item)
            
            # Check if index rebuild needed
            if (self._update_count_since_rebuild >= 
                max(100, len(self._memory_items) * self.config.rebuild_threshold)):
                asyncio.create_task(self._rebuild_index())
            
            # Emit event
            await self.event_bus.emit(MemoryItemStored(
                memory_id=item.memory_id,
                memory_type=item.memory_type.value,
                owner_id=item.owner_id,
                context_id=item.context_id
            ))
            
            # Update metrics
            if self.metrics:
                self.metrics.increment("vector_store_operations_total")
                self.metrics.increment("vector_store_items_stored")
                self.metrics.record("vector_store_operation_duration_seconds", time.time() - start_time)
                self.metrics.gauge("vector_store_items_count", len(self._memory_items))
            
        except Exception as e:
            self.logger.error(f"Failed to store item {item.memory_id}: {str(e)}")
            await self.event_bus.emit(ErrorOccurred(
                component="vector_store",
                error_type=type(e).__name__,
                error_message=str(e),
                severity="error"
            ))
            raise MemoryError(f"Failed to store vector memory item: {str(e)}")

    async def get_item(self, memory_id: str) -> Optional[MemoryItem]:
        """
        Get a memory item by ID.
        
        Args:
            memory_id: Memory identifier
            
        Returns:
            Memory item or None if not found
        """
        start_time = time.time()
        self._operation_counts['get'] += 1
        
        try:
            # Check cache first
            if self.config.cache_enabled and memory_id in self._item_cache:
                item = self._item_cache[memory_id]
                
                # Emit cache hit event
                await self.event_bus.emit(VectorStoreCacheHit(memory_id=memory_id))
                
                # Update metrics
                if self.metrics:
                    self.metrics.increment("vector_store_cache_hits")
                
                return item
            
            # Check in-memory storage
            if memory_id in self._memory_items:
                item = self._memory_items[memory_id]
                
                # Add to cache
                if self.config.cache_enabled:
                    self._add_to_cache(item)
                
                # Emit event
                await self.event_bus.emit(MemoryItemRetrieved(
                    memory_id=memory_id,
                    memory_type=item.memory_type.value,
                    owner_id=item.owner_id
                ))
                
                # Update metrics
                if self.metrics:
                    self.metrics.increment("vector_store_operations_total")
                    self.metrics.increment("vector_store_items_retrieved")
                    self.metrics.record("vector_store_operation_duration_seconds", time.time() - start_time)
                
                return item
            
            # If using database and not found in memory, try database
            if self.config.storage_type == 'database' and self.database:
                item = await self._retrieve_from_database(memory_id)
                if item:
                    # Store in memory for future use
                    self._memory_items[memory_id] = item
                    
                    # Add to memory type index
                    self._memory_types[item.memory_type].add(memory_id)
                    
                    # Add to vector index if embeddings available
                    if item.embeddings:
                        self._vector_map[memory_id] = item.embeddings
                        await self._vector_index.add(memory_id, item.embeddings)
                    
                    # Add to cache
                    if self.config.cache_enabled:
                        self._add_to_cache(item)
                    
                    # Emit event
                    await self.event_bus.emit(MemoryItemRetrieved(
                        memory_id=memory_id,
                        memory_type=item.memory_type.value,
                        owner_id=item.owner_id
                    ))
                    
                    return item
            
            # Not found
            if self.config.cache_enabled:
                await self.event_bus.emit(VectorStoreCacheMiss(memory_id=memory_id))
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve item {memory_id}: {str(e)}")
            return None

    async def update_item(self, memory_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update a memory item.
        
        Args:
            memory_id: Memory identifier
            updates: Dictionary of updates to apply
            
        Returns:
            True if successful
        """
        start_time = time.time()
        self._operation_counts['update'] += 1
        
        try:
            # Get existing item
            item = await self.get_item(memory_id)
            if not item:
                return False
            
            # Apply updates
            updated = False
            
            if 'content' in updates:
                item.content = updates['content']
                updated = True
            
            if 'metadata' in updates:
                item.metadata = updates['metadata']
                updated = True
            
            if 'embeddings' in updates:
                # Update embeddings in vector index
                if updates['embeddings']:
                    if memory_id in self._vector_map:
                        # Remove old embedding first
                        await self._vector_index.remove(memory_id)
                    
                    # Add new embedding
                    self._vector_map[memory_id] = updates['embeddings']
                    await self._vector_index.add(memory_id, updates['embeddings'])
                    
                item.embeddings = updates['embeddings']
                updated = True
            
            if 'relationships' in updates:
                item.relationships = updates['relationships']
                updated = True
            
            if not updated:
                return True  # Nothing to update
            
            # Update in-memory storage
            self._memory_items[memory_id] = item
            
            # Update cache
            if self.config.cache_enabled:
                self._add_to_cache(item)
            
            # Track modifications for index rebuilding
            self._update_count_since_rebuild += 1
            
            # Persist to storage
            if self.config.storage_type == 'file':
                await self._persist_item(item)
            elif self.config.storage_type == 'database' and self.database:
                await self._store_in_database(item)
            
            # Emit event
            await self.event_bus.emit(MemoryItemUpdated(
                memory_id=memory_id,
                memory_type=item.memory_type.value,
                owner_id=item.owner_id
            ))
            
            # Update metrics
            if self.metrics:
                self.metrics.increment("vector_store_operations_total")
                self.metrics.record("vector_store_operation_duration_seconds", time.time() - start_time)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update item {memory_id}: {str(e)}")
            return False

    async def delete_item(self, memory_id: str) -> bool:
        """
        Delete a memory item.
        
        Args:
            memory_id: Memory identifier
            
        Returns:
            True if successful
        """
        start_time = time.time()
        self._operation_counts['delete'] += 1
        
        try:
            # Get existing item
            item = await self.get_item(memory_id)
            if not item:
                return False
            
            # Remove from in-memory storage
            if memory_id in self._memory_items:
                del self._memory_items[memory_id]
            
            # Remove from memory type index
            if item.memory_type in self._memory_types:
                self._memory_types[item.memory_type].discard(memory_id)
            
            # Remove from vector index
            if memory_id in self._vector_map:
                await self._vector_index.remove(memory_id)
                del self._vector_map[memory_id]
            
            # Remove from cache
            if self.config.cache_enabled and memory_id in self._item_cache:
                del self._item_cache[memory_id]
            
            # Remove from storage
            if self.config.storage_type == 'file':
                await self._remove_from_file(memory_id)
            elif self.config.storage_type == 'database' and self.database:
                await self._remove_from_database(memory_id)
            
            # Emit event
            await self.event_bus.emit(MemoryItemDeleted(
                memory_id=memory_id,
                memory_type=item.memory_type.value,
                owner_id=item.owner_id
            ))
            
            # Update metrics
            if self.metrics:
                self.metrics.increment("vector_store_operations_total")
                self.metrics.record("vector_store_operation_duration_seconds", time.time() - start_time)
                self.metrics.gauge("vector_store_items_count", len(self._memory_items))
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete item {memory_id}: {str(e)}")
            return False

    async def query(self, query: SimpleMemoryQuery) -> List[MemoryItem]:
        """
        Query memory items.
        
        Args:
            query: Query parameters
            
        Returns:
            List of matching memory items
        """
        start_time = time.time()
        self._operation_counts['query'] += 1
        
        try:
            results = []
            
            # Filter by memory type
            candidate_ids = set()
            if query.memory_type:
                candidate_ids.update(self._memory_types.get(query.memory_type, set()))
            else:
                # All memory IDs
                candidate_ids.update(self._memory_items.keys())
            
            # Apply other filters
            filtered_items = []
            for memory_id in candidate_ids:
                item = self._memory_items.get(memory_id)
                if not item:
                    continue
                
                # Filter by owner
                if query.owner_id and item.owner_id != query.owner_id:
                    continue
                
                # Filter by session
                if query.session_id and item.session_id != query.session_id:
                    continue
                
                # Filter by context
                if query.context_id and item.context_id != query.context_id:
                    continue
                
                # Filter by tags
                if query.tags and not query.tags.issubset(item.metadata.tags):
                    continue
                
                # Filter by time range
                if query.time_range:
                    start_time, end_time = query.time_range
                    if item.metadata.created_at < start_time or item.metadata.created_at > end_time:
                        continue
                
                filtered_items.append(item)
            
            # Sort by recency and paginate
            filtered_items.sort(
                key=lambda x: x.metadata.last_accessed or x.metadata.created_at,
                reverse=True
            )
            
            # Apply offset and limit
            results = filtered_items[query.offset:query.offset + query.limit]
            
            # Update metrics
            if self.metrics:
                self.metrics.increment("vector_store_operations_total")
                self.metrics.increment("vector_store_search_operations")
                self.metrics.record("vector_store_operation_duration_seconds", time.time() - start_time)
                self.metrics.record("vector_store_search_duration_seconds", time.time() - start_time)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to query items: {str(e)}")
            return []

    async def similarity_search(
        self, 
        query_vector: List[float],
        memory_type: Optional[MemoryType] = None,
        similarity_threshold: float = 0.7,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[MemoryItem]:
        """
        Perform similarity search using vector embeddings.
        
        Args:
            query_vector: Query vector
            memory_type: Optional memory type filter
            similarity_threshold: Minimum similarity score
            top_k: Maximum number of results
            filters: Additional filters
            
        Returns:
            List of matching memory items
        """
        start_time = time.time()
        self._operation_counts['similarity_search'] += 1
        
        try:
            # Emit search started event
            await self.event_bus.emit(VectorSearchStarted(
                memory_type=memory_type.value if memory_type else "all",
                top_k=top_k
            ))
            
            # Perform search
            results = await self._vector_index.search(query_vector, top_k * 2)  # Get more for filtering
            
            # Get items and filter
            items = []
            for memory_id, similarity in results:
                # Skip if below threshold
                if similarity < similarity_threshold:
                    continue
                
                # Get item
                item = await self.get_item(memory_id)
                if not item:
                    continue
                
                # Filter by memory type
                if memory_type and item.memory_type != memory_type:
                    continue
                
                # Apply additional filters
                if filters:
                    # Filter by owner
                    if 'owner_id' in filters and item.owner_id != filters['owner_id']:
                        continue
                    
                    # Filter by session
                    if 'session_id' in filters and item.session_id != filters['session_id']:
                        continue
                    
                    # Filter by context
                    if 'context_id' in filters and item.context_id != filters['context_id']:
                        continue
                    
                    # Filter by tags
                    if 'tags' in filters and not set(filters['tags']).issubset(item.metadata.tags):
                        continue
                
                items.append((item, similarity))
            
            # Sort by similarity and limit
            items.sort(key=lambda x: x[1], reverse=True)
            items = items[:top_k]
            
            # Extract items
            result_items = [item for item, _ in items]
            
            # Emit search completed event
            await self.event_bus.emit(VectorSearchCompleted(
                memory_type=memory_type.value if memory_type else "all",
                result_count=len(result_items),
                query_time=time.time() - start_time
            ))
            
            # Update metrics
            if self.metrics:
                self.metrics.increment("vector_store_operations_total")
                self.metrics.increment("vector_store_search_operations")
                self.metrics.record("vector_store_operation_duration_seconds", time.time() - start_time)
                self.metrics.record("vector_store_search_duration_seconds", time.time() - start_time)
            
            return result_items
            
        except Exception as e:
            self.logger.error(f"Failed to perform similarity search: {str(e)}")
            traceback.print_exc()
            return []

    async def count(self, memory_type: Optional[MemoryType] = None) -> int:
        """
        Count memory items.
        
        Args:
            memory_type: Optional memory type filter
            
        Returns:
            Number of items
        """
        if memory_type:
            return len(self._memory_types.get(memory_type, set()))
        else:
            return len(self._memory_items)

    async def clear_all(self) -> None:
        """Clear all memory items."""
        # Clear in-memory storage
        self._memory_items.clear()
        self._memory_types.clear()
        self._vector_map.clear()
        self._item_cache.clear()
        
        # Reset index
        vectors = {}
        await self._vector_index.rebuild(vectors)
        
        # Reset storage
        if self.config.storage_type == 'file':
            storage_path = Path(self.config.file_path)
            if storage_path.exists():
                for file in storage_path.glob('*.vecmem'):
                    file.unlink()
        elif self.config.storage_type == 'database' and self.database:
            await self._clear_database()
        
        # Reset metrics
        if self.metrics:
            self.metrics.gauge("vector_store_items_count", 0)
            self.metrics.gauge("vector_store_cache_size", 0)
        
        self.logger.info("Cleared all vector memory items")

    async def backup(self, backup_path: Path) -> bool:
        """
        Create a backup of the vector store.
        
        Args:
            backup_path: Path to save backup
            
        Returns:
            True if successful
        """
        try:
            # Emit backup started event
            await self.event_bus.emit(VectorStoreBackupStarted(
                backup_path=str(backup_path)
            ))
            
            # Create directory if it doesn't exist
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Serialize memory items
            memory_items_path = backup_path.with_suffix('.memitems')
            with open(memory_items_path, 'wb') as f:
                pickle.dump(self._memory_items, f)
            
            # Serialize vector map
            vector_map_path = backup_path.with_suffix('.vecmap')
            with open(vector_map_path, 'wb') as f:
                pickle.dump(self._vector_map, f)
            
            # Save vector index
            index_path = backup_path.with_suffix('.vecindex')
            await self._vector_index.save(index_path)
            
            # Emit backup complete
            await self.event_bus.emit(VectorStoreBackupCompleted(
                backup_path=str(backup_path),
                vector_count=len(self._vectors),
                backup_size=backup_path.stat().st_size
            ))
            
            self.logger.info(f"Vector store backup completed: {backup_path}")
            
        except Exception as e:
            self.logger.error(f"Vector store backup failed: {str(e)}")
            raise
