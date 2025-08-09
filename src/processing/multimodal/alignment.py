"""
Advanced Multimodal Alignment System
Author: Drmusab
Last Modified: 2025-01-12 12:12:59 UTC

This module provides comprehensive multimodal alignment capabilities for the AI assistant,
handling temporal, semantic, and contextual alignment of different modalities (text, speech,
vision, gestures) to enable effective multimodal fusion and understanding.
"""

import hashlib
import json
import logging
import threading
import time
import uuid
import weakref
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
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
    List,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import asyncio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.assistant.core import ComponentMetadata, EnhancedComponentManager

# Assistant components
from src.assistant.core import CoreAssistantEngine, ProcessingContext, ProcessingResult
from src.assistant.core import EnhancedSessionManager
from src.assistant.core import WorkflowOrchestrator

# Core imports
from src.core.config.loader import ConfigLoader
from src.core.dependency_injection import Container
from src.core.error_handling import ErrorHandler, handle_exceptions
from src.core.events.event_bus import EventBus
from src.core.events.event_types import (
    AlignmentQualityAssessed,
    AlignmentStrategyChanged,
    ComponentHealthChanged,
    ContextualAlignmentPerformed,
    CrossModalAttentionComputed,
    ErrorOccurred,
    ModalityAlignmentCompleted,
    ModalityAlignmentFailed,
    ModalityAlignmentStarted,
    SemanticAlignmentPerformed,
    SystemStateChanged,
    TemporalAlignmentPerformed,
)
from src.core.health_check import HealthCheck

# Learning systems
from src.learning.continual_learning import ContinualLearner
from src.learning.feedback_processor import FeedbackProcessor
from src.learning.preference_learning import PreferenceLearner

# Memory systems
from src.memory.core_memory.memory_manager import MemoryManager
from src.memory.core_memory.memory_types import EpisodicMemory, WorkingMemory
from src.memory.operations.context_manager import ContextManager
from src.observability.logging.config import get_logger

# Observability
from src.observability.monitoring.metrics import MetricsCollector
from src.observability.monitoring.tracing import TraceManager

# Type definitions
T = TypeVar("T")


class AlignmentType(Enum):
    """Types of multimodal alignment."""

    TEMPORAL = "temporal"  # Time-based alignment
    SEMANTIC = "semantic"  # Content-based alignment
    CONTEXTUAL = "contextual"  # Context-aware alignment
    CROSS_MODAL = "cross_modal"  # Cross-modal attention alignment
    HYBRID = "hybrid"  # Combined alignment strategies


class ModalityType(Enum):
    """Types of modalities that can be aligned."""

    TEXT = "text"
    SPEECH = "speech"
    VISION = "vision"
    AUDIO = "audio"
    GESTURE = "gesture"
    TOUCH = "touch"
    PHYSIOLOGICAL = "physiological"


class AlignmentStrategy(Enum):
    """Alignment strategy selection."""

    FIXED = "fixed"  # Fixed alignment strategy
    ADAPTIVE = "adaptive"  # Adaptive strategy selection
    LEARNED = "learned"  # ML-based strategy selection
    CONTEXT_AWARE = "context_aware"  # Context-dependent selection


class AlignmentQuality(Enum):
    """Quality levels for alignment assessment."""

    EXCELLENT = "excellent"  # >0.9 confidence
    GOOD = "good"  # 0.7-0.9 confidence
    FAIR = "fair"  # 0.5-0.7 confidence
    POOR = "poor"  # 0.3-0.5 confidence
    FAILED = "failed"  # <0.3 confidence


@dataclass
class ModalityData:
    """Container for modality-specific data with metadata."""

    modality_type: ModalityType
    data: Any
    timestamp: datetime
    duration: Optional[float] = None
    confidence: float = 1.0

    # Spatial information (for vision/gesture)
    spatial_coords: Optional[Tuple[float, ...]] = None
    spatial_resolution: Optional[Tuple[int, int]] = None

    # Feature representations
    raw_features: Optional[np.ndarray] = None
    encoded_features: Optional[np.ndarray] = None
    embedding: Optional[np.ndarray] = None

    # Quality metrics
    quality_score: float = 1.0
    noise_level: float = 0.0

    # Metadata
    source: Optional[str] = None
    processing_stage: str = "raw"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AlignmentResult:
    """Result of multimodal alignment operation."""

    alignment_id: str
    alignment_type: AlignmentType
    modalities: List[ModalityType]

    # Alignment data
    aligned_timestamps: List[datetime]
    alignment_matrix: Optional[np.ndarray] = None
    attention_weights: Optional[np.ndarray] = None
    correspondence_map: Optional[Dict[str, str]] = None

    # Quality assessment
    alignment_quality: AlignmentQuality
    confidence_score: float
    quality_metrics: Dict[str, float] = field(default_factory=dict)

    # Processing information
    processing_time: float = 0.0
    strategy_used: str = "unknown"
    parameters: Dict[str, Any] = field(default_factory=dict)

    # Error handling
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    session_id: Optional[str] = None
    user_id: Optional[str] = None


@dataclass
class AlignmentConfiguration:
    """Configuration for alignment operations."""

    # Strategy settings
    alignment_strategy: AlignmentStrategy = AlignmentStrategy.ADAPTIVE
    preferred_alignment_types: List[AlignmentType] = field(
        default_factory=lambda: [AlignmentType.TEMPORAL, AlignmentType.SEMANTIC]
    )

    # Temporal alignment
    temporal_window_ms: float = 100.0
    temporal_tolerance_ms: float = 50.0
    max_temporal_drift_ms: float = 500.0

    # Semantic alignment
    semantic_similarity_threshold: float = 0.7
    semantic_weight: float = 0.5
    enable_cross_modal_semantics: bool = True

    # Attention mechanisms
    attention_heads: int = 8
    attention_dimension: int = 512
    attention_dropout: float = 0.1

    # Quality thresholds
    min_confidence_threshold: float = 0.3
    good_quality_threshold: float = 0.7
    excellent_quality_threshold: float = 0.9

    # Performance settings
    batch_size: int = 32
    max_sequence_length: int = 1024
    enable_gpu: bool = True
    memory_limit_mb: float = 1024.0

    # Adaptation settings
    enable_learning: bool = True
    adaptation_rate: float = 0.01
    feedback_window: int = 100


class AlignmentError(Exception):
    """Custom exception for alignment operations."""

    def __init__(
        self,
        message: str,
        alignment_id: Optional[str] = None,
        error_code: Optional[str] = None,
        modalities: Optional[List[ModalityType]] = None,
    ):
        super().__init__(message)
        self.alignment_id = alignment_id
        self.error_code = error_code
        self.modalities = modalities
        self.timestamp = datetime.now(timezone.utc)


class BaseAligner(ABC):
    """Abstract base class for modality aligners."""

    @abstractmethod
    async def align(
        self, modality_data: List[ModalityData], context: Dict[str, Any]
    ) -> AlignmentResult:
        """Perform alignment on modality data."""
        pass

    @abstractmethod
    def can_align(self, modalities: List[ModalityType]) -> bool:
        """Check if this aligner can handle the given modalities."""
        pass

    @abstractmethod
    def get_alignment_type(self) -> AlignmentType:
        """Get the type of alignment this aligner performs."""
        pass

    async def initialize(self) -> None:
        """Initialize the aligner."""
        pass

    async def cleanup(self) -> None:
        """Cleanup aligner resources."""
        pass


class TemporalAligner(BaseAligner):
    """Aligner for temporal synchronization of modalities."""

    def __init__(self, config: AlignmentConfiguration):
        self.config = config
        self.logger = get_logger(__name__)
        self._reference_modality: Optional[ModalityType] = None

    def get_alignment_type(self) -> AlignmentType:
        return AlignmentType.TEMPORAL

    def can_align(self, modalities: List[ModalityType]) -> bool:
        """Can align any modalities with timestamps."""
        return len(modalities) >= 2

    async def align(
        self, modality_data: List[ModalityData], context: Dict[str, Any]
    ) -> AlignmentResult:
        """Perform temporal alignment of modalities."""
        try:
            # Select reference modality (usually the most stable one)
            ref_modality = self._select_reference_modality(modality_data)
            ref_data = next(md for md in modality_data if md.modality_type == ref_modality)

            # Calculate temporal offsets
            aligned_timestamps = []
            alignment_matrix = np.zeros((len(modality_data), len(modality_data)))

            for i, data in enumerate(modality_data):
                # Calculate temporal offset from reference
                offset = (data.timestamp - ref_data.timestamp).total_seconds() * 1000  # ms

                # Apply temporal correction within tolerance
                if abs(offset) <= self.config.temporal_tolerance_ms:
                    aligned_timestamp = ref_data.timestamp
                elif abs(offset) <= self.config.max_temporal_drift_ms:
                    # Interpolate timestamp
                    correction_factor = min(1.0, self.config.temporal_tolerance_ms / abs(offset))
                    corrected_offset_ms = offset * correction_factor
                    aligned_timestamp = ref_data.timestamp + timedelta(
                        milliseconds=corrected_offset_ms
                    )
                else:
                    # Use original timestamp if drift is too large
                    aligned_timestamp = data.timestamp

                aligned_timestamps.append(aligned_timestamp)

                # Build alignment matrix
                for j, other_data in enumerate(modality_data):
                    other_offset = (other_data.timestamp - data.timestamp).total_seconds() * 1000
                    alignment_matrix[i, j] = max(
                        0, 1 - abs(other_offset) / self.config.max_temporal_drift_ms
                    )

            # Calculate quality metrics
            max_drift = max(
                abs((ts - ref_data.timestamp).total_seconds() * 1000) for ts in aligned_timestamps
            )

            confidence = max(0, 1 - max_drift / self.config.max_temporal_drift_ms)
            quality = self._assess_temporal_quality(confidence)

            result = AlignmentResult(
                alignment_id=str(uuid.uuid4()),
                alignment_type=AlignmentType.TEMPORAL,
                modalities=[data.modality_type for data in modality_data],
                aligned_timestamps=aligned_timestamps,
                alignment_matrix=alignment_matrix,
                alignment_quality=quality,
                confidence_score=confidence,
                quality_metrics={
                    "max_drift_ms": max_drift,
                    "avg_drift_ms": np.mean(
                        [
                            abs((ts - ref_data.timestamp).total_seconds() * 1000)
                            for ts in aligned_timestamps
                        ]
                    ),
                    "sync_ratio": sum(
                        1
                        for ts in aligned_timestamps
                        if abs((ts - ref_data.timestamp).total_seconds() * 1000)
                        <= self.config.temporal_tolerance_ms
                    )
                    / len(aligned_timestamps),
                },
                strategy_used=f"temporal_reference_{ref_modality.value}",
            )

            return result

        except Exception as e:
            raise AlignmentError(
                f"Temporal alignment failed: {str(e)}",
                modalities=[data.modality_type for data in modality_data],
            )

    def _select_reference_modality(self, modality_data: List[ModalityData]) -> ModalityType:
        """Select the most stable modality as temporal reference."""
        # Priority order for temporal reference
        priority_order = [
            ModalityType.AUDIO,
            ModalityType.SPEECH,
            ModalityType.TEXT,
            ModalityType.VISION,
            ModalityType.GESTURE,
        ]

        available_modalities = [data.modality_type for data in modality_data]

        for modality in priority_order:
            if modality in available_modalities:
                return modality

        # Fallback to first modality
        return available_modalities[0]

    def _assess_temporal_quality(self, confidence: float) -> AlignmentQuality:
        """Assess temporal alignment quality."""
        if confidence >= self.config.excellent_quality_threshold:
            return AlignmentQuality.EXCELLENT
        elif confidence >= self.config.good_quality_threshold:
            return AlignmentQuality.GOOD
        elif confidence >= 0.5:
            return AlignmentQuality.FAIR
        elif confidence >= self.config.min_confidence_threshold:
            return AlignmentQuality.POOR
        else:
            return AlignmentQuality.FAILED


class SemanticAligner(BaseAligner):
    """Aligner for semantic content alignment across modalities."""

    def __init__(self, config: AlignmentConfiguration):
        self.config = config
        self.logger = get_logger(__name__)
        self.device = torch.device(
            "cuda" if config.enable_gpu and torch.cuda.is_available() else "cpu"
        )

        # Initialize semantic models
        self._init_semantic_models()

    def _init_semantic_models(self):
        """Initialize semantic embedding models."""
        # Text embedding model
        self.text_embedding_dim = 768
        self.text_encoder = nn.Sequential(
            nn.Linear(self.text_embedding_dim, 512), nn.ReLU(), nn.Dropout(0.1), nn.Linear(512, 256)
        ).to(self.device)

        # Audio/speech embedding model
        self.audio_embedding_dim = 512
        self.audio_encoder = nn.Sequential(
            nn.Linear(self.audio_embedding_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
        ).to(self.device)

        # Vision embedding model
        self.vision_embedding_dim = 2048
        self.vision_encoder = nn.Sequential(
            nn.Linear(self.vision_embedding_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
        ).to(self.device)

        # Cross-modal projection layer
        self.common_dim = 256
        self.cross_modal_projector = nn.Linear(self.common_dim, self.common_dim).to(self.device)

    def get_alignment_type(self) -> AlignmentType:
        return AlignmentType.SEMANTIC

    def can_align(self, modalities: List[ModalityType]) -> bool:
        """Can align modalities with semantic content."""
        semantic_modalities = {
            ModalityType.TEXT,
            ModalityType.SPEECH,
            ModalityType.VISION,
            ModalityType.AUDIO,
        }
        return len(set(modalities).intersection(semantic_modalities)) >= 2

    async def align(
        self, modality_data: List[ModalityData], context: Dict[str, Any]
    ) -> AlignmentResult:
        """Perform semantic alignment of modalities."""
        try:
            # Extract semantic embeddings for each modality
            embeddings = []
            valid_data = []

            for data in modality_data:
                embedding = await self._extract_semantic_embedding(data)
                if embedding is not None:
                    embeddings.append(embedding)
                    valid_data.append(data)

            if len(embeddings) < 2:
                raise AlignmentError("Insufficient semantic data for alignment")

            # Convert to tensors
            embeddings_tensor = torch.stack(embeddings).to(self.device)

            # Compute semantic similarity matrix
            similarity_matrix = self._compute_semantic_similarity(embeddings_tensor)

            # Generate correspondence map
            correspondence_map = self._generate_correspondence_map(
                valid_data, similarity_matrix.cpu().numpy()
            )

            # Calculate quality metrics
            avg_similarity = torch.mean(similarity_matrix).item()
            max_similarity = torch.max(similarity_matrix).item()
            confidence = min(avg_similarity, max_similarity)

            quality = self._assess_semantic_quality(confidence)

            # Align timestamps based on semantic correspondences
            aligned_timestamps = self._align_timestamps_semantically(
                valid_data, similarity_matrix.cpu().numpy()
            )

            result = AlignmentResult(
                alignment_id=str(uuid.uuid4()),
                alignment_type=AlignmentType.SEMANTIC,
                modalities=[data.modality_type for data in valid_data],
                aligned_timestamps=aligned_timestamps,
                alignment_matrix=similarity_matrix.cpu().numpy(),
                correspondence_map=correspondence_map,
                alignment_quality=quality,
                confidence_score=confidence,
                quality_metrics={
                    "avg_similarity": avg_similarity,
                    "max_similarity": max_similarity,
                    "min_similarity": torch.min(similarity_matrix).item(),
                    "semantic_coherence": self._calculate_semantic_coherence(similarity_matrix),
                },
                strategy_used="semantic_embedding",
            )

            return result

        except Exception as e:
            raise AlignmentError(
                f"Semantic alignment failed: {str(e)}",
                modalities=[data.modality_type for data in modality_data],
            )

    async def _extract_semantic_embedding(self, data: ModalityData) -> Optional[torch.Tensor]:
        """Extract semantic embedding from modality data."""
        try:
            if data.embedding is not None:
                # Use pre-computed embedding
                embedding = torch.tensor(data.embedding, dtype=torch.float32)
            elif data.encoded_features is not None:
                # Use encoded features
                features = torch.tensor(data.encoded_features, dtype=torch.float32)

                # Apply appropriate encoder based on modality
                if data.modality_type in [ModalityType.TEXT]:
                    if features.shape[-1] != self.text_embedding_dim:
                        # Resize if needed
                        features = F.adaptive_avg_pool1d(
                            features.unsqueeze(0), self.text_embedding_dim
                        ).squeeze(0)
                    embedding = self.text_encoder(features)
                elif data.modality_type in [ModalityType.SPEECH, ModalityType.AUDIO]:
                    if features.shape[-1] != self.audio_embedding_dim:
                        features = F.adaptive_avg_pool1d(
                            features.unsqueeze(0), self.audio_embedding_dim
                        ).squeeze(0)
                    embedding = self.audio_encoder(features)
                elif data.modality_type == ModalityType.VISION:
                    if features.shape[-1] != self.vision_embedding_dim:
                        features = F.adaptive_avg_pool1d(
                            features.unsqueeze(0), self.vision_embedding_dim
                        ).squeeze(0)
                    embedding = self.vision_encoder(features)
                else:
                    return None
            else:
                return None

            # Project to common semantic space
            embedding = self.cross_modal_projector(embedding)
            return F.normalize(embedding, p=2, dim=-1)

        except Exception as e:
            self.logger.warning(
                f"Failed to extract semantic embedding for {data.modality_type}: {str(e)}"
            )
            return None

    def _compute_semantic_similarity(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Compute semantic similarity matrix between embeddings."""
        # Cosine similarity
        similarity_matrix = torch.matmul(embeddings, embeddings.transpose(0, 1))

        # Apply softmax to normalize
        similarity_matrix = F.softmax(similarity_matrix / 0.1, dim=-1)

        return similarity_matrix

    def _generate_correspondence_map(
        self, data: List[ModalityData], similarity_matrix: np.ndarray
    ) -> Dict[str, str]:
        """Generate correspondence map between modalities."""
        correspondence_map = {}

        # Find best matches above threshold
        for i, data_i in enumerate(data):
            best_match_idx = np.argmax(similarity_matrix[i])
            best_similarity = similarity_matrix[i, best_match_idx]

            if best_similarity >= self.config.semantic_similarity_threshold and best_match_idx != i:
                best_match_data = data[best_match_idx]
                correspondence_map[f"{data_i.modality_type.value}_{i}"] = (
                    f"{best_match_data.modality_type.value}_{best_match_idx}"
                )

        return correspondence_map

    def _align_timestamps_semantically(
        self, data: List[ModalityData], similarity_matrix: np.ndarray
    ) -> List[datetime]:
        """Align timestamps based on semantic correspondences."""
        aligned_timestamps = []

        # Find the most semantically central modality as reference
        centrality_scores = np.mean(similarity_matrix, axis=1)
        ref_idx = np.argmax(centrality_scores)
        ref_timestamp = data[ref_idx].timestamp

        for i, modal_data in enumerate(data):
            if i == ref_idx:
                aligned_timestamps.append(ref_timestamp)
            else:
                # Weight timestamp based on semantic similarity to reference
                similarity_weight = similarity_matrix[ref_idx, i]

                if similarity_weight >= self.config.semantic_similarity_threshold:
                    # High semantic similarity - align closely to reference
                    aligned_timestamps.append(ref_timestamp)
                else:
                    # Low semantic similarity - keep original timestamp
                    aligned_timestamps.append(modal_data.timestamp)

        return aligned_timestamps

    def _calculate_semantic_coherence(self, similarity_matrix: torch.Tensor) -> float:
        """Calculate overall semantic coherence."""
        # Remove diagonal (self-similarity)
        mask = ~torch.eye(
            similarity_matrix.shape[0], dtype=torch.bool, device=similarity_matrix.device
        )
        off_diagonal = similarity_matrix[mask]

        return torch.mean(off_diagonal).item()

    def _assess_semantic_quality(self, confidence: float) -> AlignmentQuality:
        """Assess semantic alignment quality."""
        if confidence >= self.config.excellent_quality_threshold:
            return AlignmentQuality.EXCELLENT
        elif confidence >= self.config.good_quality_threshold:
            return AlignmentQuality.GOOD
        elif confidence >= 0.5:
            return AlignmentQuality.FAIR
        elif confidence >= self.config.min_confidence_threshold:
            return AlignmentQuality.POOR
        else:
            return AlignmentQuality.FAILED


class CrossModalAttentionAligner(BaseAligner):
    """Advanced aligner using cross-modal attention mechanisms."""

    def __init__(self, config: AlignmentConfiguration):
        self.config = config
        self.logger = get_logger(__name__)
        self.device = torch.device(
            "cuda" if config.enable_gpu and torch.cuda.is_available() else "cpu"
        )

        # Initialize attention models
        self._init_attention_models()

    def _init_attention_models(self):
        """Initialize cross-modal attention models."""
        self.hidden_dim = self.config.attention_dimension
        self.num_heads = self.config.attention_heads

        # Multi-head cross-modal attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=self.num_heads,
            dropout=self.config.attention_dropout,
            batch_first=True,
        ).to(self.device)

        # Modality-specific query/key/value projections
        self.modality_projections = nn.ModuleDict(
            {
                "text": nn.Linear(768, self.hidden_dim),
                "speech": nn.Linear(512, self.hidden_dim),
                "audio": nn.Linear(512, self.hidden_dim),
                "vision": nn.Linear(2048, self.hidden_dim),
                "gesture": nn.Linear(256, self.hidden_dim),
            }
        ).to(self.device)

        # Position encoding for temporal information
        self.position_encoding = self._create_positional_encoding(
            self.config.max_sequence_length, self.hidden_dim
        )

        # Alignment prediction head
        self.alignment_predictor = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.config.attention_dropout),
            nn.Linear(self.hidden_dim // 2, 1),
            nn.Sigmoid(),
        ).to(self.device)

    def _create_positional_encoding(self, max_length: int, hidden_dim: int) -> torch.Tensor:
        """Create sinusoidal positional encoding."""
        pos_encoding = torch.zeros(max_length, hidden_dim)
        position = torch.arange(0, max_length).unsqueeze(1).float()

        div_term = torch.exp(
            torch.arange(0, hidden_dim, 2).float() * -(np.log(10000.0) / hidden_dim)
        )

        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)

        return pos_encoding.to(self.device)

    def get_alignment_type(self) -> AlignmentType:
        return AlignmentType.CROSS_MODAL

    def can_align(self, modalities: List[ModalityType]) -> bool:
        """Can align modalities with attention mechanisms."""
        supported_modalities = {
            ModalityType.TEXT,
            ModalityType.SPEECH,
            ModalityType.VISION,
            ModalityType.AUDIO,
            ModalityType.GESTURE,
        }
        return len(set(modalities).intersection(supported_modalities)) >= 2

    async def align(
        self, modality_data: List[ModalityData], context: Dict[str, Any]
    ) -> AlignmentResult:
        """Perform cross-modal attention-based alignment."""
        try:
            # Prepare input sequences
            sequences, sequence_lengths, modality_types = self._prepare_sequences(modality_data)

            if len(sequences) < 2:
                raise AlignmentError("Insufficient data for cross-modal attention")

            # Apply cross-modal attention
            attention_outputs, attention_weights = await self._apply_cross_modal_attention(
                sequences, sequence_lengths, modality_types
            )

            # Predict alignment scores
            alignment_scores = self._predict_alignment_scores(attention_outputs)

            # Generate aligned timestamps
            aligned_timestamps = self._generate_aligned_timestamps(
                modality_data, attention_weights, alignment_scores
            )

            # Calculate quality metrics
            confidence = torch.mean(alignment_scores).item()
            quality = self._assess_attention_quality(confidence, attention_weights)

            result = AlignmentResult(
                alignment_id=str(uuid.uuid4()),
                alignment_type=AlignmentType.CROSS_MODAL,
                modalities=[data.modality_type for data in modality_data],
                aligned_timestamps=aligned_timestamps,
                attention_weights=attention_weights.cpu().numpy(),
                alignment_quality=quality,
                confidence_score=confidence,
                quality_metrics={
                    "avg_attention_score": confidence,
                    "attention_entropy": self._calculate_attention_entropy(attention_weights),
                    "cross_modal_coherence": self._calculate_cross_modal_coherence(
                        attention_weights
                    ),
                    "temporal_consistency": self._calculate_temporal_consistency(attention_weights),
                },
                strategy_used="cross_modal_attention",
            )

            return result

        except Exception as e:
            raise AlignmentError(
                f"Cross-modal attention alignment failed: {str(e)}",
                modalities=[data.modality_type for data in modality_data],
            )

    def _prepare_sequences(
        self, modality_data: List[ModalityData]
    ) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        """Prepare input sequences for attention computation."""
        sequences = []
        sequence_lengths = []
        modality_types = []

        for data in modality_data:
            if data.encoded_features is not None:
                # Use encoded features
                features = torch.tensor(data.encoded_features, dtype=torch.float32)

                # Project to common dimension
                modality_key = data.modality_type.value
                if modality_key in self.modality_projections:
                    if features.dim() == 1:
                        features = features.unsqueeze(0)  # Add sequence dimension

                    projected = self.modality_projections[modality_key](features)

                    # Add positional encoding
                    seq_len = projected.shape[0]
                    if seq_len <= self.position_encoding.shape[0]:
                        projected += self.position_encoding[:seq_len]

                    sequences.append(projected)
                    sequence_lengths.append(seq_len)
                    modality_types.append(modality_key)

        if not sequences:
            raise AlignmentError("No valid sequences for attention alignment")

        # Pad sequences to same length
        max_length = max(seq.shape[0] for seq in sequences)
        padded_sequences = []

        for seq in sequences:
            if seq.shape[0] < max_length:
                padding = torch.zeros(
                    max_length - seq.shape[0], seq.shape[1], device=seq.device, dtype=seq.dtype
                )
                padded_seq = torch.cat([seq, padding], dim=0)
            else:
                padded_seq = seq
            padded_sequences.append(padded_seq)

        # Stack into batch
        batch_sequences = torch.stack(padded_sequences).to(self.device)
        sequence_lengths_tensor = torch.tensor(sequence_lengths, device=self.device)

        return batch_sequences, sequence_lengths_tensor, modality_types

    async def _apply_cross_modal_attention(
        self, sequences: torch.Tensor, sequence_lengths: torch.Tensor, modality_types: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply cross-modal attention across sequences."""
        batch_size, seq_len, hidden_dim = sequences.shape

        # Reshape for multi-head attention (batch_size * num_modalities, seq_len, hidden_dim)
        num_modalities = batch_size

        all_attention_outputs = []
        all_attention_weights = []

        # Apply attention between each pair of modalities
        for i in range(num_modalities):
            for j in range(num_modalities):
                if i != j:
                    query = sequences[i].unsqueeze(0)  # (1, seq_len, hidden_dim)
                    key = sequences[j].unsqueeze(0)
                    value = sequences[j].unsqueeze(0)

                    # Apply cross-attention
                    attn_output, attn_weights = self.cross_attention(query, key, value)

                    all_attention_outputs.append(attn_output.squeeze(0))
                    all_attention_weights.append(attn_weights.squeeze(0))

        # Aggregate attention outputs
        if all_attention_outputs:
            aggregated_output = torch.stack(all_attention_outputs).mean(dim=0)
            aggregated_weights = torch.stack(all_attention_weights).mean(dim=0)
        else:
            aggregated_output = sequences.mean(dim=0)
            aggregated_weights = torch.ones(seq_len, seq_len, device=self.device) / seq_len

        return aggregated_output, aggregated_weights

    def _predict_alignment_scores(self, attention_outputs: torch.Tensor) -> torch.Tensor:
        """Predict alignment scores from attention outputs."""
        # Apply alignment prediction head
        alignment_scores = self.alignment_predictor(attention_outputs)
        return alignment_scores.squeeze(-1)  # Remove last dimension

    def _generate_aligned_timestamps(
        self,
        modality_data: List[ModalityData],
        attention_weights: torch.Tensor,
        alignment_scores: torch.Tensor,
    ) -> List[datetime]:
        """Generate aligned timestamps based on attention weights."""
        aligned_timestamps = []

        # Use weighted average of timestamps based on attention
        if len(modality_data) > 0:
            # Calculate attention-weighted timestamp
            timestamps_ms = [
                (data.timestamp - modality_data[0].timestamp).total_seconds() * 1000
                for data in modality_data
            ]

            for i, data in enumerate(modality_data):
                if i < attention_weights.shape[0] and i < len(alignment_scores):
                    # Weight by attention and alignment scores
                    attention_weight = torch.mean(attention_weights[i]).item()
                    alignment_score = (
                        alignment_scores[i].item() if i < len(alignment_scores) else 0.5
                    )

                    combined_weight = attention_weight * alignment_score

                    if combined_weight > 0.5:
                        # High attention - align to most attended timestamp
                        max_attention_idx = torch.argmax(attention_weights[i]).item()
                        if max_attention_idx < len(modality_data):
                            aligned_timestamps.append(modality_data[max_attention_idx].timestamp)
                        else:
                            aligned_timestamps.append(data.timestamp)
                    else:
                        # Low attention - keep original timestamp
                        aligned_timestamps.append(data.timestamp)
                else:
                    aligned_timestamps.append(data.timestamp)

        return aligned_timestamps

    def _calculate_attention_entropy(self, attention_weights: torch.Tensor) -> float:
        """Calculate entropy of attention weights."""
        # Normalize attention weights
        normalized_weights = F.softmax(attention_weights.flatten(), dim=0)

        # Calculate entropy
        entropy = -torch.sum(normalized_weights * torch.log(normalized_weights + 1e-8))
        return entropy.item()

    def _calculate_cross_modal_coherence(self, attention_weights: torch.Tensor) -> float:
        """Calculate cross-modal coherence score."""
        # Calculate consistency of attention patterns
        mean_attention = torch.mean(attention_weights, dim=0)
        variance = torch.mean((attention_weights - mean_attention) ** 2)
        coherence = 1.0 / (1.0 + variance.item())

        return coherence

    def _calculate_temporal_consistency(self, attention_weights: torch.Tensor) -> float:
        """Calculate temporal consistency of attention."""
        # Check if attention weights are temporally smooth
        if attention_weights.shape[-1] < 2:
            return 1.0

        diff = attention_weights[:, 1:] - attention_weights[:, :-1]
        temporal_variance = torch.mean(diff**2)
        consistency = 1.0 / (1.0 + temporal_variance.item())

        return consistency

    def _assess_attention_quality(
        self, confidence: float, attention_weights: torch.Tensor
    ) -> AlignmentQuality:
        """Assess attention-based alignment quality."""
        # Combine confidence with attention pattern quality
        attention_entropy = self._calculate_attention_entropy(attention_weights)
        coherence = self._calculate_cross_modal_coherence(attention_weights)

        # Normalized entropy (lower is better for focused attention)
        max_entropy = np.log(attention_weights.numel())
        normalized_entropy = attention_entropy / max_entropy

        # Combined quality score
        quality_score = (confidence + coherence + (1 - normalized_entropy)) / 3

        if quality_score >= self.config.excellent_quality_threshold:
            return AlignmentQuality.EXCELLENT
        elif quality_score >= self.config.good_quality_threshold:
            return AlignmentQuality.GOOD
        elif quality_score >= 0.5:
            return AlignmentQuality.FAIR
        elif quality_score >= self.config.min_confidence_threshold:
            return AlignmentQuality.POOR
        else:
            return AlignmentQuality.FAILED


class AdaptiveAlignmentStrategy:
    """Adaptive strategy selection for optimal alignment."""

    def __init__(self, config: AlignmentConfiguration):
        self.config = config
        self.logger = get_logger(__name__)
        self.strategy_history: deque = deque(maxlen=config.feedback_window)
        self.strategy_performance: Dict[str, List[float]] = defaultdict(list)

    def select_strategy(
        self, modality_data: List[ModalityData], context: Dict[str, Any]
    ) -> AlignmentType:
        """Select optimal alignment strategy based on context and history."""
        modalities = [data.modality_type for data in modality_data]

        # Check data quality and availability
        has_temporal_info = all(data.timestamp is not None for data in modality_data)
        has_semantic_features = any(
            data.encoded_features is not None or data.embedding is not None
            for data in modality_data
        )

        # Context-based selection
        real_time_requirement = context.get("real_time", False)
        quality_requirement = context.get("quality_preference", "balanced")

        # Performance-based selection
        if self.config.enable_learning and self.strategy_performance:
            best_strategy = max(
                self.strategy_performance.items(), key=lambda x: np.mean(x[1]) if x[1] else 0
            )[0]
            try:
                return AlignmentType(best_strategy)
            except ValueError:
                pass

        # Rule-based selection
        if real_time_requirement:
            # Prefer faster temporal alignment for real-time
            if has_temporal_info:
                return AlignmentType.TEMPORAL

        if quality_requirement == "high":
            # Prefer sophisticated cross-modal attention for high quality
            if has_semantic_features and len(modalities) >= 2:
                return AlignmentType.CROSS_MODAL

        # Default selection based on available data
        if has_semantic_features:
            return AlignmentType.SEMANTIC
        elif has_temporal_info:
            return AlignmentType.TEMPORAL
        else:
            return AlignmentType.CONTEXTUAL

    def update_strategy_performance(
        self, strategy: AlignmentType, performance_score: float
    ) -> None:
        """Update strategy performance tracking."""
        self.strategy_performance[strategy.value].append(performance_score)

        # Limit history size
        if len(self.strategy_performance[strategy.value]) > self.config.feedback_window:
            self.strategy_performance[strategy.value].pop(0)

        # Add to history
        self.strategy_history.append(
            {
                "strategy": strategy.value,
                "performance": performance_score,
                "timestamp": datetime.now(timezone.utc),
            }
        )


class MultimodalAlignmentSystem:
    """
    Advanced Multimodal Alignment System for the AI Assistant.

    This system provides comprehensive alignment capabilities for different modalities:
    - Temporal alignment based on timestamps and synchronization
    - Semantic alignment using content similarity and embeddings
    - Cross-modal attention for sophisticated alignment patterns
    - Adaptive strategy selection based on context and performance
    - Quality assessment and confidence scoring
    - Integration with all core system components
    - Learning-based optimization and adaptation
    """

    def __init__(self, container: Container):
        """
        Initialize the multimodal alignment system.

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

        # Assistant components
        self.core_engine = container.get(CoreAssistantEngine)
        self.component_manager = container.get(EnhancedComponentManager)
        self.session_manager = container.get(EnhancedSessionManager)
        self.workflow_orchestrator = container.get(WorkflowOrchestrator)

        # Memory systems
        self.memory_manager = container.get(MemoryManager)
        self.context_manager = container.get(ContextManager)
        self.working_memory = container.get(WorkingMemory)
        self.episodic_memory = container.get(EpisodicMemory)

        # Learning systems
        self.continual_learner = container.get(ContinualLearner)
        self.preference_learner = container.get(PreferenceLearner)
        self.feedback_processor = container.get(FeedbackProcessor)

        # Observability
        self.metrics = container.get(MetricsCollector)
        self.tracer = container.get(TraceManager)

        # Configuration
        self.config = self._load_alignment_config()

        # Alignment components
        self.aligners: Dict[AlignmentType, BaseAligner] = {}
        self.adaptive_strategy = AdaptiveAlignmentStrategy(self.config)

        # State management
        self.active_alignments: Dict[str, AlignmentResult] = {}
        self.alignment_history: deque = deque(maxlen=1000)
        self.alignment_cache: Dict[str, AlignmentResult] = {}

        # Performance tracking
        self.alignment_stats: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.quality_trends: Dict[AlignmentType, deque] = {
            alignment_type: deque(maxlen=100) for alignment_type in AlignmentType
        }

        # Threading
        self.alignment_semaphore = asyncio.Semaphore(self.config.batch_size)
        self.thread_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="alignment")

        # Initialize components
        self._setup_aligners()
        self._setup_monitoring()

        # Register health check
        self.health_check.register_component("multimodal_alignment", self._health_check_callback)

        self.logger.info("MultimodalAlignmentSystem initialized successfully")

    def _load_alignment_config(self) -> AlignmentConfiguration:
        """Load alignment configuration from config loader."""
        try:
            alignment_config_dict = self.config_loader.get("alignment", {})

            # Create configuration with defaults
            config = AlignmentConfiguration()

            # Override with loaded values
            for key, value in alignment_config_dict.items():
                if hasattr(config, key):
                    setattr(config, key, value)

            return config

        except Exception as e:
            self.logger.warning(f"Failed to load alignment config, using defaults: {str(e)}")
            return AlignmentConfiguration()

    def _setup_aligners(self) -> None:
        """Setup alignment algorithms."""
        try:
            # Initialize aligners
            self.aligners[AlignmentType.TEMPORAL] = TemporalAligner(self.config)
            self.aligners[AlignmentType.SEMANTIC] = SemanticAligner(self.config)
            self.aligners[AlignmentType.CROSS_MODAL] = CrossModalAttentionAligner(self.config)

            self.logger.info(f"Initialized {len(self.aligners)} alignment algorithms")

        except Exception as e:
            self.logger.error(f"Failed to setup aligners: {str(e)}")

    def _setup_monitoring(self) -> None:
        """Setup monitoring and metrics."""
        try:
            # Register alignment metrics
            self.metrics.register_counter("alignment_operations_total")
            self.metrics.register_counter("alignment_operations_successful")
            self.metrics.register_counter("alignment_operations_failed")
            self.metrics.register_histogram("alignment_duration_seconds")
            self.metrics.register_gauge("active_alignments")
            self.metrics.register_histogram("alignment_confidence_score")
            self.metrics.register_counter("alignment_quality_distribution")

        except Exception as e:
            self.logger.warning(f"Failed to setup monitoring: {str(e)}")

    async def initialize(self) -> None:
        """Initialize the alignment system."""
        try:
            # Initialize aligners
            for aligner in self.aligners.values():
                await aligner.initialize()

            # Start background tasks
            asyncio.create_task(self._alignment_quality_monitor_loop())
            asyncio.create_task(self._cache_cleanup_loop())

            if self.config.enable_learning:
                asyncio.create_task(self._learning_adaptation_loop())

            # Register event handlers
            await self._register_event_handlers()

            self.logger.info("MultimodalAlignmentSystem initialization completed")

        except Exception as e:
            self.logger.error(f"Failed to initialize MultimodalAlignmentSystem: {str(e)}")
            raise AlignmentError(f"Initialization failed: {str(e)}")

    async def _register_event_handlers(self) -> None:
        """Register event handlers for system integration."""
        # Processing events
        self.event_bus.subscribe("processing_started", self._handle_processing_started)
        self.event_bus.subscribe(
            "modality_processing_completed", self._handle_modality_processing_completed
        )

        # Component health events
        self.event_bus.subscribe("component_health_changed", self._handle_component_health_change)

        # System events
        self.event_bus.subscribe("system_shutdown_started", self._handle_system_shutdown)

    @handle_exceptions
    async def align_modalities(
        self,
        modality_data: List[ModalityData],
        session_id: str,
        context: Optional[Dict[str, Any]] = None,
        alignment_type: Optional[AlignmentType] = None,
    ) -> AlignmentResult:
        """
        Align multiple modalities using appropriate alignment strategy.

        Args:
            modality_data: List of modality data to align
            session_id: Session ID for context
            context: Additional context information
            alignment_type: Specific alignment type (if not provided, will be selected adaptively)

        Returns:
            Alignment result
        """
        async with self.alignment_semaphore:
            start_time = time.time()
            context = context or {}

            try:
                with self.tracer.trace("multimodal_alignment") as span:
                    span.set_attributes(
                        {
                            "session_id": session_id,
                            "num_modalities": len(modality_data),
                            "modality_types": [data.modality_type.value for data in modality_data],
                            "alignment_type": (
                                alignment_type.value if alignment_type else "adaptive"
                            ),
                        }
                    )

                    # Validate input
                    if len(modality_data) < 2:
                        raise AlignmentError("At least 2 modalities required for alignment")

                    # Add session context
                    session_context = await self._get_session_context(session_id)
                    context.update(session_context)

                    # Select alignment strategy
                    if alignment_type is None:
                        alignment_type = self.adaptive_strategy.select_strategy(
                            modality_data, context
                        )

                    # Check cache
                    cache_key = self._generate_cache_key(modality_data, alignment_type, context)
                    if cache_key in self.alignment_cache:
                        cached_result = self.alignment_cache[cache_key]
                        self.logger.debug(
                            f"Using cached alignment result: {cached_result.alignment_id}"
                        )
                        return cached_result

                    # Emit alignment started event
                    await self.event_bus.emit(
                        ModalityAlignmentStarted(
                            session_id=session_id,
                            modalities=[data.modality_type.value for data in modality_data],
                            alignment_type=alignment_type.value,
                        )
                    )

                    # Perform alignment
                    aligner = self.aligners.get(alignment_type)
                    if not aligner:
                        raise AlignmentError(f"No aligner available for type {alignment_type}")

                    if not aligner.can_align([data.modality_type for data in modality_data]):
                        # Fallback to temporal alignment
                        aligner = self.aligners.get(AlignmentType.TEMPORAL)
                        if not aligner or not aligner.can_align(
                            [data.modality_type for data in modality_data]
                        ):
                            raise AlignmentError(
                                "No suitable aligner found for the given modalities"
                            )
                        alignment_type = AlignmentType.TEMPORAL

                    result = await aligner.align(modality_data, context)

                    # Update result metadata
                    result.session_id = session_id
                    result.processing_time = time.time() - start_time

                    # Store in active alignments
                    self.active_alignments[result.alignment_id] = result

                    # Cache result
                    self.alignment_cache[cache_key] = result

                    # Update strategy performance
                    performance_score = result.confidence_score
                    self.adaptive_strategy.update_strategy_performance(
                        alignment_type, performance_score
                    )

                    # Store in memory for learning
                    await self._store_alignment_for_learning(result, modality_data, context)

                    # Update metrics
                    self.metrics.increment("alignment_operations_total")
                    self.metrics.increment("alignment_operations_successful")
                    self.metrics.record("alignment_duration_seconds", result.processing_time)
                    self.metrics.record("alignment_confidence_score", result.confidence_score)
                    self.metrics.increment(
                        "alignment_quality_distribution",
                        tags={"quality": result.alignment_quality.value},
                    )

                    # Track quality trends
                    self.quality_trends[alignment_type].append(result.confidence_score)

                    # Emit completion event
                    await self.event_bus.emit(
                        ModalityAlignmentCompleted(
                            session_id=session_id,
                            alignment_id=result.alignment_id,
                            alignment_type=alignment_type.value,
                            confidence_score=result.confidence_score,
                            processing_time=result.processing_time,
                        )
                    )

                    self.logger.info(
                        f"Alignment completed: {result.alignment_id} "
                        f"({alignment_type.value}) with confidence {result.confidence_score:.3f} "
                        f"in {result.processing_time:.3f}s"
                    )

                    return result

            except Exception as e:
                # Handle alignment failure
                processing_time = time.time() - start_time

                self.metrics.increment("alignment_operations_failed")

                await self.event_bus.emit(
                    ModalityAlignmentFailed(
                        session_id=session_id,
                        error_message=str(e),
                        error_type=type(e).__name__,
                        processing_time=processing_time,
                    )
                )

                self.logger.error(f"Alignment failed for session {session_id}: {str(e)}")
                raise

    async def _get_session_context(self, session_id: str) -> Dict[str, Any]:
        """Get session context for alignment."""
        try:
            # Get session information
            session_info = await self.session_manager.get_session(session_id)
            if session_info:
                context = {
                    "user_id": session_info.context.user_id,
                    "session_duration": (
                        datetime.now(timezone.utc) - session_info.created_at
                    ).total_seconds(),
                    "interaction_count": session_info.interaction_count,
                    "device_info": session_info.context.device_info,
                    "user_preferences": session_info.context.user_preferences,
                }

                # Add working memory context
                working_memory_data = await self.working_memory.get_session_data(session_id)
                context["working_memory"] = working_memory_data

                return context

            return {}

        except Exception as e:
            self.logger.warning(f"Failed to get session context for {session_id}: {str(e)}")
            return {}

    def _generate_cache_key(
        self,
        modality_data: List[ModalityData],
        alignment_type: AlignmentType,
        context: Dict[str, Any],
    ) -> str:
        """Generate cache key for alignment result."""
        # Create hash from modality data and parameters
        key_components = [
            alignment_type.value,
            len(modality_data),
            tuple(sorted([data.modality_type.value for data in modality_data])),
            context.get("real_time", False),
            context.get("quality_preference", "balanced"),
        ]

        # Add timestamps (quantized to reduce cache misses)
        timestamps = [int(data.timestamp.timestamp() // 10) * 10 for data in modality_data]
        key_components.append(tuple(timestamps))

        key_string = json.dumps(key_components, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()

    async def _store_alignment_for_learning(
        self, result: AlignmentResult, modality_data: List[ModalityData], context: Dict[str, Any]
    ) -> None:
        """Store alignment result for learning and adaptation."""
        if not self.config.enable_learning:
            return

        try:
            learning_data = {
                "alignment_id": result.alignment_id,
                "alignment_type": result.alignment_type.value,
                "modalities": [data.modality_type.value for data in modality_data],
                "confidence_score": result.confidence_score,
                "quality_metrics": result.quality_metrics,
                "processing_time": result.processing_time,
                "context": context,
                "session_id": result.session_id,
                "timestamp": result.created_at,
            }

            # Store in episodic memory
            await self.episodic_memory.store(
                {"event_type": "multimodal_alignment", "data": learning_data}
            )

            # Update continual learning
            await self.continual_learner.learn_from_alignment(learning_data)

        except Exception as e:
            self.logger.warning(f"Failed to store alignment for learning: {str(e)}")

    @handle_exceptions
    async def assess_alignment_quality(self, alignment_id: str) -> Dict[str, Any]:
        """
        Assess the quality of a completed alignment.

        Args:
            alignment_id: Alignment identifier

        Returns:
            Quality assessment results
        """
        if alignment_id not in self.active_alignments:
            # Check alignment history
            for result in self.alignment_history:
                if result.alignment_id == alignment_id:
                    return self._format_quality_assessment(result)

            raise AlignmentError(f"Alignment {alignment_id} not found")

        result = self.active_alignments[alignment_id]

        # Emit quality assessment event
        await self.event_bus.emit(
            AlignmentQualityAssessed(
                alignment_id=alignment_id,
                quality=result.alignment_quality.value,
                confidence_score=result.confidence_score,
                quality_metrics=result.quality_metrics,
            )
        )

        return self._format_quality_assessment(result)

    def _format_quality_assessment(self, result: AlignmentResult) -> Dict[str, Any]:
        """Format quality assessment results."""
        return {
            "alignment_id": result.alignment_id,
            "alignment_type": result.alignment_type.value,
            "quality": result.alignment_quality.value,
            "confidence_score": result.confidence_score,
            "quality_metrics": result.quality_metrics,
            "processing_time": result.processing_time,
            "strategy_used": result.strategy_used,
            "modalities": [modality.value for modality in result.modalities],
            "errors": result.errors,
            "warnings": result.warnings,
            "created_at": result.created_at.isoformat(),
        }

    @handle_exceptions
    def get_alignment_statistics(self) -> Dict[str, Any]:
        """Get comprehensive alignment system statistics."""
        try:
            total_alignments = len(self.alignment_history) + len(self.active_alignments)

            # Quality distribution
            quality_counts = defaultdict(int)
            confidence_scores = []

            all_results = list(self.active_alignments.values()) + list(self.alignment_history)
            for result in all_results:
                quality_counts[result.alignment_quality.value] += 1
                confidence_scores.append(result.confidence_score)

            # Strategy performance
            strategy_performance = {}
            for strategy, scores in self.adaptive_strategy.strategy_performance.items():
                if scores:
                    strategy_performance[strategy] = {
                        "avg_performance": np.mean(scores),
                        "std_performance": np.std(scores),
                        "num_uses": len(scores),
                    }

            # Quality trends
            quality_trends = {}
            for alignment_type, scores in self.quality_trends.items():
                if scores:
                    quality_trends[alignment_type.value] = {
                        "recent_avg": (
                            np.mean(list(scores)[-10:])
                            if len(scores) >= 10
                            else np.mean(list(scores))
                        ),
                        "overall_avg": np.mean(list(scores)),
                        "trend": (
                            "improving" if len(scores) >= 2 and scores[-1] > scores[0] else "stable"
                        ),
                    }

            return {
                "total_alignments": total_alignments,
                "active_alignments": len(self.active_alignments),
                "cache_size": len(self.alignment_cache),
                "quality_distribution": dict(quality_counts),
                "average_confidence": np.mean(confidence_scores) if confidence_scores else 0.0,
                "strategy_performance": strategy_performance,
                "quality_trends": quality_trends,
                "configuration": asdict(self.config),
            }

        except Exception as e:
            self.logger.error(f"Failed to get alignment statistics: {str(e)}")
            return {"error": str(e)}

    async def _alignment_quality_monitor_loop(self) -> None:
        """Background task for monitoring alignment quality."""
        while True:
            try:
                # Monitor quality trends
                for alignment_type, scores in self.quality_trends.items():
                    if len(scores) >= 10:
                        recent_avg = np.mean(list(scores)[-10:])
                        overall_avg = np.mean(list(scores))

                        # Alert if quality is degrading
                        if recent_avg < overall_avg * 0.8:
                            self.logger.warning(
                                f"Quality degradation detected for {alignment_type.value}: "
                                f"recent={recent_avg:.3f}, overall={overall_avg:.3f}"
                            )

                # Update metrics
                active_count = len(self.active_alignments)
                self.metrics.set("active_alignments", active_count)

                await asyncio.sleep(30)  # Monitor every 30 seconds

            except Exception as e:
                self.logger.error(f"Error in quality monitor: {str(e)}")
                await asyncio.sleep(30)

    async def _cache_cleanup_loop(self) -> None:
        """Background task for cache cleanup."""
        while True:
            try:
                # Remove old cache entries
                current_time = datetime.now(timezone.utc)
                cutoff_time = current_time - timedelta(seconds=self.cache_ttl)

                # Clean alignment cache
                expired_keys = []
                for key, entry in self.alignment_cache.items():
                    if entry.get("timestamp", current_time) < cutoff_time:
                        expired_keys.append(key)

                for key in expired_keys:
                    del self.alignment_cache[key]

                if expired_keys:
                    self.logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")

                await asyncio.sleep(300)  # Run every 5 minutes

            except Exception as e:
                self.logger.error(f"Error in cache cleanup: {str(e)}")
                await asyncio.sleep(300)
