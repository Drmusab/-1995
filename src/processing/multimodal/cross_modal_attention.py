"""
Advanced Cross-Modal Attention for Multimodal AI Assistant
Author: Drmusab
Last Modified: 2025-06-12 04:53:13 UTC

This module provides specialized cross-modal attention mechanisms for enhanced
multimodal understanding and alignment across different modalities (text, speech,
vision, emotion, gesture) with deep integration into the AI assistant core system.
"""

from pathlib import Path
from typing import Optional, Dict, Any, List, Union, Tuple, Protocol, TypeVar, Generic
import asyncio
import threading
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime, timezone
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import logging
from collections import defaultdict, deque
import weakref
from contextlib import asynccontextmanager
import json
import pickle
import math
from concurrent.futures import ThreadPoolExecutor

# Core imports
from src.core.config.loader import ConfigLoader
from src.core.events.event_bus import EventBus
from src.core.events.event_types import (
    CrossModalAlignmentStarted, CrossModalAlignmentCompleted, CrossModalAlignmentFailed,
    AttentionWeightsComputed, ModalityCorrelationDetected, CrossModalPatternDiscovered,
    TemporalAlignmentCompleted, SemanticAlignmentCompleted, SpatialAlignmentCompleted,
    AttentionMapGenerated, CrossModalLearningUpdated, QualityThresholdExceeded
)
from src.core.error_handling import ErrorHandler, handle_exceptions
from src.core.dependency_injection import Container
from src.core.health_check import HealthCheck

# Processing components
from src.processing.speech.emotion_detection import EmotionResult
from src.processing.speech.speaker_recognition import SpeakerRecognitionResult
from src.processing.speech.speech_to_text import TranscriptionResult
from src.processing.natural_language.sentiment_analyzer import SentimentAnalyzer
from src.processing.natural_language.entity_extractor import EntityExtractor

# Memory and learning
from src.memory.core_memory.memory_manager import MemoryManager
from src.memory.operations.context_manager import ContextManager
from src.learning.continual_learning import ContinualLearner
from src.learning.feedback_processor import FeedbackProcessor
from src.learning.preference_learning import PreferenceLearner

# Observability
from src.observability.monitoring.metrics import MetricsCollector
from src.observability.monitoring.tracing import TraceManager
from src.observability.logging.config import get_logger

# Assistant components
from src.assistant.session_manager import SessionManager
from src.assistant.component_manager import ComponentManager


# Type definitions
T = TypeVar('T')


class AttentionType(Enum):
    """Types of cross-modal attention mechanisms."""
    SELF_ATTENTION = "self_attention"
    CROSS_ATTENTION = "cross_attention"
    MULTI_HEAD_ATTENTION = "multi_head_attention"
    SCALED_DOT_PRODUCT = "scaled_dot_product"
    ADDITIVE_ATTENTION = "additive_attention"
    TEMPORAL_ATTENTION = "temporal_attention"
    SPATIAL_ATTENTION = "spatial_attention"
    SEMANTIC_ATTENTION = "semantic_attention"
    ADAPTIVE_ATTENTION = "adaptive_attention"


class AlignmentStrategy(Enum):
    """Strategies for cross-modal alignment."""
    TEMPORAL = "temporal"
    SEMANTIC = "semantic"
    SPATIAL = "spatial"
    FEATURE_BASED = "feature_based"
    CANONICAL_CORRELATION = "canonical_correlation"
    ADVERSARIAL = "adversarial"
    CONTRASTIVE = "contrastive"
    TRANSFORMER_BASED = "transformer_based"


class ModalityType(Enum):
    """Types of input modalities."""
    TEXT = "text"
    SPEECH = "speech"
    VISION = "vision"
    AUDIO = "audio"
    GESTURE = "gesture"
    EMOTION = "emotion"
    CONTEXT = "context"
    TEMPORAL = "temporal"
    SPATIAL = "spatial"


@dataclass
class AttentionConfiguration:
    """Configuration for attention mechanisms."""
    attention_type: AttentionType = AttentionType.MULTI_HEAD_ATTENTION
    alignment_strategy: AlignmentStrategy = AlignmentStrategy.TRANSFORMER_BASED
    
    # Attention parameters
    attention_dim: int = 512
    num_heads: int = 8
    dropout_rate: float = 0.1
    temperature: float = 1.0
    
    # Performance settings
    enable_caching: bool = True
    enable_pruning: bool = True
    pruning_threshold: float = 0.01
    
    # Quality settings
    min_attention_score: float = 0.1
    max_sequence_length: int = 1024
    enable_gradient_checkpointing: bool = False
    
    # Real-time settings
    real_time_mode: bool = False
    max_latency_ms: float = 100.0
    batch_processing: bool = True


@dataclass
class ModalityFeatures:
    """Container for modality-specific features and metadata."""
    modality: ModalityType
    features: torch.Tensor
    embeddings: Optional[torch.Tensor] = None
    sequence_length: int = 1
    temporal_info: Optional[Dict[str, Any]] = None
    spatial_info: Optional[Dict[str, Any]] = None
    confidence: float = 1.0
    quality_score: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Attention-specific information
    attention_mask: Optional[torch.Tensor] = None
    position_embeddings: Optional[torch.Tensor] = None
    segment_ids: Optional[torch.Tensor] = None
    
    # Processing timestamps
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    processing_time: float = 0.0


@dataclass
class AttentionOutput:
    """Output from attention mechanisms."""
    success: bool
    attention_type: AttentionType
    processing_time: float
    
    # Attention results
    attended_features: torch.Tensor
    attention_weights: torch.Tensor
    cross_modal_attention: Optional[torch.Tensor] = None
    
    # Alignment results
    alignment_scores: Dict[Tuple[ModalityType, ModalityType], float] = field(default_factory=dict)
    temporal_alignment: Optional[torch.Tensor] = None
    semantic_alignment: Optional[torch.Tensor] = None
    spatial_alignment: Optional[torch.Tensor] = None
    
    # Quality metrics
    attention_entropy: float = 0.0
    alignment_quality: float = 0.0
    consistency_score: float = 0.0
    
    # Performance metrics
    memory_usage: float = 0.0
    computational_cost: float = 0.0
    cache_hit_rate: float = 0.0
    
    # Metadata
    modalities_processed: Set[ModalityType] = field(default_factory=set)
    attention_patterns: Dict[str, Any] = field(default_factory=dict)
    learned_associations: List[Dict[str, Any]] = field(default_factory=list)
    
    # Error handling
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Context
    session_id: str = ""
    request_id: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class CrossModalAttentionError(Exception):
    """Custom exception for cross-modal attention operations."""
    
    def __init__(self, message: str, attention_type: Optional[AttentionType] = None,
                 modalities: Optional[List[ModalityType]] = None):
        super().__init__(message)
        self.attention_type = attention_type
        self.modalities = modalities
        self.timestamp = datetime.now(timezone.utc)


class BaseAttentionMechanism(ABC):
    """Abstract base class for attention mechanisms."""
    
    def __init__(self, config: AttentionConfiguration, device: str = "cpu"):
        self.config = config
        self.device = device
        self.logger = get_logger(__name__)
        self.is_initialized = False
        self._attention_cache = {}
        self._performance_stats = defaultdict(list)
    
    @abstractmethod
    async def compute_attention(
        self,
        modality_features: Dict[ModalityType, ModalityFeatures],
        context: Optional[Dict[str, Any]] = None
    ) -> AttentionOutput:
        """Compute attention across modalities."""
        pass
    
    @abstractmethod
    def get_attention_patterns(self) -> Dict[str, Any]:
        """Get learned attention patterns."""
        pass
    
    async def initialize(self) -> None:
        """Initialize the attention mechanism."""
        self.is_initialized = True
    
    async def cleanup(self) -> None:
        """Cleanup resources."""
        self.is_initialized = False
        self._attention_cache.clear()
    
    def _validate_inputs(self, modality_features: Dict[ModalityType, ModalityFeatures]) -> None:
        """Validate input modality features."""
        if not modality_features:
            raise CrossModalAttentionError("No modality features provided")
        
        if len(modality_features) < 2:
            raise CrossModalAttentionError("Cross-modal attention requires at least 2 modalities")
        
        # Validate feature dimensions
        for modality, features in modality_features.items():
            if features.features.numel() == 0:
                raise CrossModalAttentionError(f"Empty features for modality {modality.value}")


class MultiHeadCrossModalAttention(BaseAttentionMechanism):
    """Multi-head cross-modal attention mechanism."""
    
    def __init__(self, config: AttentionConfiguration, device: str = "cpu"):
        super().__init__(config, device)
        self.attention_layers = None
        self.alignment_network = None
        self.position_encoding = None
    
    async def initialize(self) -> None:
        """Initialize multi-head attention components."""
        await super().initialize()
        
        # Initialize attention layers
        self.attention_layers = nn.ModuleDict()
        
        # Standard modality dimensions (can be configured)
        modality_dims = {
            ModalityType.TEXT: 768,
            ModalityType.SPEECH: 1024,
            ModalityType.VISION: 2048,
            ModalityType.EMOTION: 128,
            ModalityType.AUDIO: 512,
            ModalityType.GESTURE: 256
        }
        
        for modality, dim in modality_dims.items():
            self.attention_layers[modality.value] = MultiHeadAttentionLayer(
                input_dim=dim,
                attention_dim=self.config.attention_dim,
                num_heads=self.config.num_heads,
                dropout_rate=self.config.dropout_rate
            ).to(self.device)
        
        # Cross-modal attention layer
        self.cross_modal_attention = CrossModalAttentionLayer(
            attention_dim=self.config.attention_dim,
            num_heads=self.config.num_heads,
            dropout_rate=self.config.dropout_rate
        ).to(self.device)
        
        # Alignment network
        self.alignment_network = AlignmentNetwork(
            modality_dims=modality_dims,
            hidden_dim=self.config.attention_dim
        ).to(self.device)
        
        # Position encoding
        self.position_encoding = PositionalEncoding(
            d_model=self.config.attention_dim,
            max_length=self.config.max_sequence_length
        ).to(self.device)
        
        self.logger.info("MultiHeadCrossModalAttention initialized")
    
    async def compute_attention(
        self,
        modality_features: Dict[ModalityType, ModalityFeatures],
        context: Optional[Dict[str, Any]] = None
    ) -> AttentionOutput:
        """Compute multi-head cross-modal attention."""
        start_time = time.time()
        
        try:
            self._validate_inputs(modality_features)
            
            # Prepare features for attention
            processed_features = {}
            attention_masks = {}
            
            for modality, features in modality_features.items():
                if modality.value in self.attention_layers:
                    # Project to attention dimension
                    projected = self.attention_layers[modality.value].project_input(features.features)
                    
                    # Add positional encoding
                    if features.sequence_length > 1:
                        projected = self.position_encoding(projected)
                    
                    processed_features[modality] = projected
                    attention_masks[modality] = features.attention_mask
            
            # Compute cross-modal alignments
            alignment_scores = await self._compute_alignments(processed_features)
            
            # Apply cross-modal attention
            attended_features, attention_weights = await self._apply_cross_modal_attention(
                processed_features, attention_masks, alignment_scores
            )
            
            # Compute quality metrics
            attention_entropy = self._compute_attention_entropy(attention_weights)
            alignment_quality = self._compute_alignment_quality(alignment_scores)
            consistency_score = self._compute_consistency_score(attention_weights)
            
            processing_time = time.time() - start_time
            
            return AttentionOutput(
                success=True,
                attention_type=AttentionType.MULTI_HEAD_ATTENTION,
                processing_time=processing_time,
                attended_features=attended_features,
                attention_weights=attention_weights,
                cross_modal_attention=attention_weights,
                alignment_scores=alignment_scores,
                attention_entropy=attention_entropy,
                alignment_quality=alignment_quality,
                consistency_score=consistency_score,
                modalities_processed=set(modality_features.keys()),
                session_id=context.get('session_id', '') if context else '',
                request_id=context.get('request_id', '') if context else ''
            )
            
        except Exception as e:
            self.logger.error(f"Multi-head cross-modal attention failed: {str(e)}")
            return AttentionOutput(
                success=False,
                attention_type=AttentionType.MULTI_HEAD_ATTENTION,
                processing_time=time.time() - start_time,
                attended_features=torch.zeros(1),
                attention_weights=torch.zeros(1),
                errors=[str(e)],
                session_id=context.get('session_id', '') if context else '',
                request_id=context.get('request_id', '') if context else ''
            )
    
    async def _compute_alignments(
        self,
        processed_features: Dict[ModalityType, torch.Tensor]
    ) -> Dict[Tuple[ModalityType, ModalityType], float]:
        """Compute cross-modal alignments."""
        alignment_scores = {}
        
        if self.alignment_network:
            modality_list = list(processed_features.keys())
            
            for i, mod1 in enumerate(modality_list):
                for j, mod2 in enumerate(modality_list):
                    if i != j:
                        score = await self.alignment_network.compute_alignment(
                            processed_features[mod1], processed_features[mod2]
                        )
                        alignment_scores[(mod1, mod2)] = score
        
        return alignment_scores
    
    async def _apply_cross_modal_attention(
        self,
        processed_features: Dict[ModalityType, torch.Tensor],
        attention_masks: Dict[ModalityType, Optional[torch.Tensor]],
        alignment_scores: Dict[Tuple[ModalityType, ModalityType], float]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply cross-modal attention mechanism."""
        if self.cross_modal_attention:
            return await self.cross_modal_attention(
                processed_features, attention_masks, alignment_scores
            )
        else:
            # Fallback to simple cross-modal attention
            return self._simple_cross_modal_attention(processed_features, alignment_scores)
    
    def _simple_cross_modal_attention(
        self,
        processed_features: Dict[ModalityType, torch.Tensor],
        alignment_scores: Dict[Tuple[ModalityType, ModalityType], float]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Simple cross-modal attention fallback."""
        modality_list = list(processed_features.keys())
        num_modalities = len(modality_list)
        
        # Create attention matrix
        attention_matrix = torch.zeros(num_modalities, num_modalities, device=self.device)
        
        for i, mod1 in enumerate(modality_list):
            for j, mod2 in enumerate(modality_list):
                if (mod1, mod2) in alignment_scores:
                    attention_matrix[i, j] = alignment_scores[(mod1, mod2)]
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_matrix, dim=-1)
        
        # Apply attention to features
        feature_tensors = [processed_features[mod] for mod in modality_list]
        feature_stack = torch.stack(feature_tensors, dim=0)
        
        # Weighted combination
        attended_features = torch.einsum('ij,jkl->ikl', attention_weights, feature_stack)
        attended_features = torch.mean(attended_features, dim=0)
        
        return attended_features, attention_weights
    
    def _compute_attention_entropy(self, attention_weights: torch.Tensor) -> float:
        """Compute entropy of attention weights."""
        # Avoid log(0) by adding small epsilon
        epsilon = 1e-8
        weights_with_eps = attention_weights + epsilon
        
        # Compute entropy
        entropy = -torch.sum(attention_weights * torch.log(weights_with_eps))
        return entropy.item()
    
    def _compute_alignment_quality(
        self,
        alignment_scores: Dict[Tuple[ModalityType, ModalityType], float]
    ) -> float:
        """Compute overall alignment quality."""
        if not alignment_scores:
            return 0.0
        
        scores = list(alignment_scores.values())
        return np.mean(scores)
    
    def _compute_consistency_score(self, attention_weights: torch.Tensor) -> float:
        """Compute consistency of attention patterns."""
        # Compute variance in attention weights as a measure of consistency
        variance = torch.var(attention_weights)
        # Convert to consistency score (lower variance = higher consistency)
        consistency = 1.0 / (1.0 + variance.item())
        return consistency
    
    def get_attention_patterns(self) -> Dict[str, Any]:
        """Get learned attention patterns."""
        patterns = {
            'attention_type': self.config.attention_type.value,
            'num_heads': self.config.num_heads,
            'attention_dim': self.config.attention_dim,
            'performance_stats': dict(self._performance_stats)
        }
        
        # Add pattern analysis if available
        if hasattr(self, '_pattern_analyzer'):
            patterns['learned_patterns'] = self._pattern_analyzer.get_patterns()
        
        return patterns


class SemanticCrossModalAttention(BaseAttentionMechanism):
    """Semantic-based cross-modal attention mechanism."""
    
    def __init__(self, config: AttentionConfiguration, device: str = "cpu"):
        super().__init__(config, device)
        self.semantic_encoder = None
        self.concept_mapper = None
        self.knowledge_graph = None
    
    async def initialize(self) -> None:
        """Initialize semantic attention components."""
        await super().initialize()
        
        # Initialize semantic encoder
        self.semantic_encoder = SemanticEncoder(
            input_dim=self.config.attention_dim,
            hidden_dim=self.config.attention_dim * 2,
            output_dim=self.config.attention_dim
        ).to(self.device)
        
        # Initialize concept mapper
        self.concept_mapper = ConceptMapper(
            embedding_dim=self.config.attention_dim,
            num_concepts=1000  # Can be configured
        ).to(self.device)
        
        self.logger.info("SemanticCrossModalAttention initialized")
    
    async def compute_attention(
        self,
        modality_features: Dict[ModalityType, ModalityFeatures],
        context: Optional[Dict[str, Any]] = None
    ) -> AttentionOutput:
        """Compute semantic-based cross-modal attention."""
        start_time = time.time()
        
        try:
            self._validate_inputs(modality_features)
            
            # Extract semantic representations
            semantic_representations = {}
            for modality, features in modality_features.items():
                semantic_rep = await self._extract_semantic_representation(features)
                semantic_representations[modality] = semantic_rep
            
            # Compute semantic similarities
            semantic_attention = await self._compute_semantic_attention(semantic_representations)
            
            # Apply concept mapping
            concept_mappings = await self._apply_concept_mapping(semantic_representations)
            
            # Generate attended features
            attended_features = await self._generate_attended_features(
                semantic_representations, semantic_attention
            )
            
            processing_time = time.time() - start_time
            
            return AttentionOutput(
                success=True,
                attention_type=AttentionType.SEMANTIC_ATTENTION,
                processing_time=processing_time,
                attended_features=attended_features,
                attention_weights=semantic_attention,
                semantic_alignment=semantic_attention,
                modalities_processed=set(modality_features.keys()),
                attention_patterns={
                    'concept_mappings': concept_mappings,
                    'semantic_similarities': semantic_attention.detach().cpu().numpy().tolist()
                },
                session_id=context.get('session_id', '') if context else '',
                request_id=context.get('request_id', '') if context else ''
            )
            
        except Exception as e:
            self.logger.error(f"Semantic cross-modal attention failed: {str(e)}")
            return AttentionOutput(
                success=False,
                attention_type=AttentionType.SEMANTIC_ATTENTION,
                processing_time=time.time() - start_time,
                attended_features=torch.zeros(1),
                attention_weights=torch.zeros(1),
                errors=[str(e)],
                session_id=context.get('session_id', '') if context else '',
                request_id=context.get('request_id', '') if context else ''
            )
    
    async def _extract_semantic_representation(
        self,
        features: ModalityFeatures
    ) -> torch.Tensor:
        """Extract semantic representation from modality features."""
        if self.semantic_encoder:
            return self.semantic_encoder(features.features)
        else:
            # Fallback to feature normalization
            return F.normalize(features.features, dim=-1)
    
    async def _compute_semantic_attention(
        self,
        semantic_representations: Dict[ModalityType, torch.Tensor]
    ) -> torch.Tensor:
        """Compute attention based on semantic similarities."""
        modality_list = list(semantic_representations.keys())
        num_modalities = len(modality_list)
        
        # Compute pairwise semantic similarities
        similarity_matrix = torch.zeros(num_modalities, num_modalities, device=self.device)
        
        for i, mod1 in enumerate(modality_list):
            for j, mod2 in enumerate(modality_list):
                if i != j:
                    sim = F.cosine_similarity(
                        semantic_representations[mod1].flatten(),
                        semantic_representations[mod2].flatten(),
                        dim=0
                    )
                    similarity_matrix[i, j] = sim
        
        # Convert similarities to attention weights
        attention_weights = F.softmax(similarity_matrix / self.config.temperature, dim=-1)
        
        return attention_weights
    
    async def _apply_concept_mapping(
        self,
        semantic_representations: Dict[ModalityType, torch.Tensor]
    ) -> Dict[ModalityType, List[str]]:
        """Apply concept mapping to semantic representations."""
        concept_mappings = {}
        
        for modality, representation in semantic_representations.items():
            if self.concept_mapper:
                concepts = await self.concept_mapper.map_to_concepts(representation)
                concept_mappings[modality] = concepts
            else:
                # Fallback: use modality type as concept
                concept_mappings[modality] = [modality.value]
        
        return concept_mappings
    
    async def _generate_attended_features(
        self,
        semantic_representations: Dict[ModalityType, torch.Tensor],
        attention_weights: torch.Tensor
    ) -> torch.Tensor:
        """Generate attended features using semantic attention."""
        modality_list = list(semantic_representations.keys())
        
        # Stack representations
        representation_stack = torch.stack([
            semantic_representations[mod] for mod in modality_list
        ], dim=0)
        
        # Apply attention weights
        attended = torch.einsum('ij,jkl->ikl', attention_weights, representation_stack)
        
        # Aggregate across modalities
        attended_features = torch.mean(attended, dim=0)
        
        return attended_features
    
    def get_attention_patterns(self) -> Dict[str, Any]:
        """Get semantic attention patterns."""
        return {
            'attention_type': AttentionType.SEMANTIC_ATTENTION.value,
            'semantic_concepts': self.concept_mapper.get_concept_vocabulary() if self.concept_mapper else [],
            'performance_stats': dict(self._performance_stats)
        }


class TemporalCrossModalAttention(BaseAttentionMechanism):
    """Temporal-based cross-modal attention for sequential data."""
    
    def __init__(self, config: AttentionConfiguration, device: str = "cpu"):
        super().__init__(config, device)
        self.temporal_encoder = None
        self.sequence_aligner = None
    
    async def initialize(self) -> None:
        """Initialize temporal attention components."""
        await super().initialize()
        
        # Initialize temporal encoder
        self.temporal_encoder = TemporalEncoder(
            input_dim=self.config.attention_dim,
            hidden_dim=self.config.attention_dim,
            num_layers=2
        ).to(self.device)
        
        # Initialize sequence aligner
        self.sequence_aligner = SequenceAligner(
            feature_dim=self.config.attention_dim
        ).to(self.device)
        
        self.logger.info("TemporalCrossModalAttention initialized")
    
    async def compute_attention(
        self,
        modality_features: Dict[ModalityType, ModalityFeatures],
        context: Optional[Dict[str, Any]] = None
    ) -> AttentionOutput:
        """Compute temporal cross-modal attention."""
        start_time = time.time()
        
        try:
            self._validate_inputs(modality_features)
            
            # Extract temporal information
            temporal_features = {}
            for modality, features in modality_features.items():
                temporal_feat = await self._extract_temporal_features(features)
                temporal_features[modality] = temporal_feat
            
            # Align sequences temporally
            aligned_features = await self._align_temporal_sequences(temporal_features)
            
            # Compute temporal attention
            temporal_attention = await self._compute_temporal_attention(aligned_features)
            
            # Generate temporally-aware attended features
            attended_features = await self._generate_temporal_attended_features(
                aligned_features, temporal_attention
            )
            
            processing_time = time.time() - start_time
            
            return AttentionOutput(
                success=True,
                attention_type=AttentionType.TEMPORAL_ATTENTION,
                processing_time=processing_time,
                attended_features=attended_features,
                attention_weights=temporal_attention,
                temporal_alignment=temporal_attention,
                modalities_processed=set(modality_features.keys()),
                session_id=context.get('session_id', '') if context else '',
                request_id=context.get('request_id', '') if context else ''
            )
            
        except Exception as e:
            self.logger.error(f"Temporal cross-modal attention failed: {str(e)}")
            return AttentionOutput(
                success=False,
                attention_type=AttentionType.TEMPORAL_ATTENTION,
                processing_time=time.time() - start_time,
                attended_features=torch.zeros(1),
                attention_weights=torch.zeros(1),
                errors=[str(e)],
                session_id=context.get('session_id', '') if context else '',
                request_id=context.get('request_id', '') if context else ''
            )
    
    async def _extract_temporal_features(self, features: ModalityFeatures) -> torch.Tensor:
        """Extract temporal features from modality data."""
        if self.temporal_encoder and features.sequence_length > 1:
            return self.temporal_encoder(features.features)
        else:
            # For non-sequential data, add temporal dimension
            return features.features.unsqueeze(0)
    
    async def _align_temporal_sequences(
        self,
        temporal_features: Dict[ModalityType, torch.Tensor]
    ) -> Dict[ModalityType, torch.Tensor]:
        """Align temporal sequences across modalities."""
        if self.sequence_aligner:
            return await self.sequence_aligner.align_sequences(temporal_features)
        else:
            # Simple padding/truncation alignment
            max_length = max(feat.size(0) for feat in temporal_features.values())
            aligned = {}
            
            for modality, features in temporal_features.items():
                if features.size(0) < max_length:
                    # Pad sequence
                    padding = torch.zeros(
                        max_length - features.size(0), *features.shape[1:],
                        device=features.device
                    )
                    aligned[modality] = torch.cat([features, padding], dim=0)
                else:
                    # Truncate sequence
                    aligned[modality] = features[:max_length]
            
            return aligned
    
    async def _compute_temporal_attention(
        self,
        aligned_features: Dict[ModalityType, torch.Tensor]
    ) -> torch.Tensor:
        """Compute attention across temporal dimensions."""
        modality_list = list(aligned_features.keys())
        sequence_length = list(aligned_features.values())[0].size(0)
        num_modalities = len(modality_list)
        
        # Compute temporal attention matrix
        attention_matrix = torch.zeros(
            sequence_length, num_modalities, num_modalities,
            device=self.device
        )
        
        for t in range(sequence_length):
            for i, mod1 in enumerate(modality_list):
                for j, mod2 in enumerate(modality_list):
                    if i != j:
                        feat1 = aligned_features[mod1][t]
                        feat2 = aligned_features[mod2][t]
                        
                        # Compute similarity at this time step
                        sim = F.cosine_similarity(feat1.flatten(), feat2.flatten(), dim=0)
                        attention_matrix[t, i, j] = sim
        
        # Apply softmax across modalities for each time step
        attention_weights = F.softmax(attention_matrix, dim=-1)
        
        return attention_weights
    
    async def _generate_temporal_attended_features(
        self,
        aligned_features: Dict[ModalityType, torch.Tensor],
        temporal_attention: torch.Tensor
    ) -> torch.Tensor:
        """Generate attended features using temporal attention."""
        modality_list = list(aligned_features.keys())
        sequence_length, num_modalities, _ = temporal_attention.shape
        
        # Stack features across modalities
        feature_stack = torch.stack([aligned_features[mod] for mod in modality_list], dim=1)
        
        # Apply temporal attention
        attended_sequence = []
        for t in range(sequence_length):
            attended_t = torch.einsum('ij,jk->ik', temporal_attention[t], feature_stack[t])
            attended_sequence.append(torch.mean(attended_t, dim=0))
        
        # Aggregate temporal sequence
        attended_features = torch.mean(torch.stack(attended_sequence), dim=0)
        
        return attended_features
    
    def get_attention_patterns(self) -> Dict[str, Any]:
        """Get temporal attention patterns."""
        return {
            'attention_type': AttentionType.TEMPORAL_ATTENTION.value,
            'sequence_alignment_strategy': 'padding_truncation',
            'performance_stats': dict(self._performance_stats)
        }


# Neural Network Components

class MultiHeadAttentionLayer(nn.Module):
    """Multi-head attention layer for individual modalities."""
    
    def __init__(self, input_dim: int, attention_dim: int, num_heads: int, dropout_rate: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.attention_dim = attention_dim
        self.num_heads = num_heads
        self.head_dim = attention_dim // num_heads
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, attention_dim)
        
        # Multi-head attention
        self.query_projection = nn.Linear(attention_dim, attention_dim)
        self.key_projection = nn.Linear(attention_dim, attention_dim)
        self.value_projection = nn.Linear(attention_dim, attention_dim)
        
        self.output_projection = nn.Linear(attention_dim, attention_dim)
        self.dropout = nn.Dropout(dropout_rate)
        
        self.layer_norm = nn.LayerNorm(attention_dim)
        
    def project_input(self, x: torch.Tensor) -> torch.Tensor:
        """Project input to attention dimension."""
        return self.input_projection(x)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        queries = self.query_projection(x)
        keys = self.key_projection(x)
        values = self.value_projection(x)
        
        # Reshape for multi-head attention
        queries = queries.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1))
        attention_scores = attention_scores / math.sqrt(self.head_dim)
        
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)
        
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attended = torch.matmul(attention_weights, values)
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.attention_dim
        )
        
        # Output projection and residual connection
        output = self.output_projection(attended)
        output = self.layer_norm(output + x)
        
        return output


class CrossModalAttentionLayer(nn.Module):
    """Cross-modal attention layer."""
    
    def __init__(self, attention_dim: int, num_heads: int, dropout_rate: float = 0.1):
        super().__init__()
        self.attention_dim = attention_dim
        self.num_heads = num_heads
        self.head_dim = attention_dim // num_heads
        
        self.query_projections = nn.ModuleDict()
        self.key_projections = nn.ModuleDict()
        self.value_projections = nn.ModuleDict()
        
        # Will be initialized dynamically based on modalities
        self.output_projection = nn.Linear(attention_dim, attention_dim)
        self.dropout = nn.Dropout(dropout_rate)
        
    def _ensure_projections(self, modalities: List[ModalityType]) -> None:
        """Ensure projections exist for all modalities."""
        for modality in modalities:
            if modality.value not in self.query_projections:
                self.query_projections[modality.value] = nn.Linear(
                    self.attention_dim, self.attention_dim
                )
                self.key_projections[modality.value] = nn.Linear(
                    self.attention_dim, self.attention_dim
                )
                self.value_projections[modality.value] = nn.Linear(
                    self.attention_dim, self.attention_dim
                )
    
    async def __call__(
        self,
        modality_features: Dict[ModalityType, torch.Tensor],
        attention_masks: Dict[ModalityType, Optional[torch.Tensor]],
        alignment_scores: Dict[Tuple[ModalityType, ModalityType], float]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply cross-modal attention."""
        modality_list = list(modality_features.keys())
        self._ensure_projections(modality_list)
        
        # Compute Q, K, V for each modality
        queries = {}
        keys = {}
        values = {}
        
        for modality in modality_list:
            features = modality_features[modality]
            queries[modality] = self.query_projections[modality.value](features)
            keys[modality] = self.key_projections[modality.value](features)
            values[modality] = self.value_projections[modality.value](features)
        
        # Compute cross-modal attention
        num_modalities = len(modality_list)
        attention_matrix = torch.zeros(num_modalities, num_modalities)
        attended_values = []
        
        for i, mod1 in enumerate(modality_list):
            attended_value = torch.zeros_like(values[mod1])
            
            for j, mod2 in enumerate(modality_list):
                # Compute attention score
                score = torch.matmul(queries[mod1], keys[mod2].transpose(-2, -1))
                score = score / math.sqrt(self.head_dim)
                
                # Apply alignment scores if available
                if (mod1, mod2) in alignment_scores:
                    score = score * alignment_scores[(mod1, mod2)]
                
                attention_weight = F.softmax(score, dim=-1)
                attention_matrix[i, j] = attention_weight.mean()
                
                # Apply attention to values
                attended_value += torch.matmul(attention_weight, values[mod2])
            
            attended_values.append(attended_value)
        
        # Combine attended values
        fused_output = torch.mean(torch.stack(attended_values), dim=0)
        fused_output = self.output_projection(fused_output)
        
        return fused_output, attention_matrix


class AlignmentNetwork(nn.Module):
    """Network for computing cross-modal alignments."""
    
    def __init__(self, modality_dims: Dict[ModalityType, int], hidden_dim: int):
        super().__init__()
        self.modality_dims = modality_dims
        self.hidden_dim = hidden_dim
        
        # Alignment projection layers
        self.alignment_projections = nn.ModuleDict()
        for modality, dim in modality_dims.items():
            self.alignment_projections[modality.value] = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, hidden_dim // 4)
            )
    
    async def compute_alignment(
        self,
        features1: torch.Tensor,
        features2: torch.Tensor
    ) -> float:
        """Compute alignment score between two feature sets."""
        # Compute cosine similarity
        similarity = F.cosine_similarity(features1.flatten(), features2.flatten(), dim=0)
        return similarity.item()


class SemanticEncoder(nn.Module):
    """Encoder for semantic representations."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Tanh()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class ConceptMapper(nn.Module):
    """Maps features to semantic concepts."""
    
    def __init__(self, embedding_dim: int, num_concepts: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_concepts = num_concepts
        
        # Concept embeddings
        self.concept_embeddings = nn.Embedding(num_concepts, embedding_dim)
        
        # Concept vocabulary (would be loaded from external source)
        self.concept_vocabulary = [f"concept_{i}" for i in range(num_concepts)]
        
    async def map_to_concepts(self, features: torch.Tensor, top_k: int = 5) -> List[str]:
        """Map features to top-k concepts."""
        # Compute similarities to all concept embeddings
        all_embeddings = self.concept_embeddings.weight
        similarities = F.cosine_similarity(
            features.unsqueeze(0), all_embeddings, dim=1
        )
        
        # Get top-k concepts
        top_k_indices = torch.topk(similarities, k=top_k).indices
        
        return [self.concept_vocabulary[idx.item()] for idx in top_k_indices]
    
    def get_concept_vocabulary(self) -> List[str]:
        """Get the concept vocabulary."""
        return self.concept_vocabulary


class TemporalEncoder(nn.Module):
    """Encoder for temporal features."""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=0.1
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # If input is not sequential, add time dimension
        if x.dim() == 2:
            x = x.unsqueeze(0)  # Add batch dimension
            x = x.unsqueeze(1)  # Add time dimension
        
        output, (hidden, cell) = self.lstm(x)
        output = self.layer_norm(output)
        
        return output


class SequenceAligner(nn.Module):
    """Aligns sequences across different modalities."""
    
    def __init__(self, feature_dim: int):
        super().__init__()
        self.feature_dim = feature_dim
        
        # Alignment networks
        self.alignment_network = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, 1),
            nn.Sigmoid()
        )
    
    async def align_sequences(
        self,
        sequences: Dict[ModalityType, torch.Tensor]
    ) -> Dict[ModalityType, torch.Tensor]:
        """Align sequences using learned alignment."""
        # For now, implement simple padding/truncation
        # More sophisticated alignment could use attention mechanisms
        max_length = max(seq.size(0) for seq in sequences.values())
        
        aligned = {}
        for modality, sequence in sequences.items():
            if sequence.size(0) < max_length:
                # Pad sequence
                padding = torch.zeros(
                    max_length - sequence.size(0), *sequence.shape[1:],
                    device=sequence.device
                )
                aligned[modality] = torch.cat([sequence, padding], dim=0)
            else:
                # Truncate sequence
                aligned[modality] = sequence[:max_length]
        
        return aligned


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer-style attention."""
    
    def __init__(self, d_model: int, max_length: int = 1024):
        super().__init__()
        self.d_model = d_model
        
        # Create positional encoding matrix
        pe = torch.zeros(max_length, d_model)
        position = torch.arange(0, max_length).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Add positional encoding to input
        seq_len = x.size(-2) if x.dim() > 2 else 1
        return x + self.pe[:, :seq_len]


class EnhancedCrossModalAttention:
    """
    Enhanced Cross-Modal Attention System for the AI Assistant.
    
    This system provides comprehensive cross-modal attention mechanisms
    integrated with the core AI assistant infrastructure, enabling sophisticated
    multimodal understanding and processing capabilities.
    """
    
    def __init__(self, container: Container):
        """
        Initialize the enhanced cross-modal attention system.
        
        Args:
            container: Dependency injection container
        """
        self.container = container
        self.logger = get_logger(__name__)
        self.config = container.get(ConfigLoader)
        self.event_bus = container.get(EventBus)
        self.error_handler = container.get(ErrorHandler)
        self.health_check = container.get(HealthCheck)
        
        # Core component integration
        try:
            self.memory_manager = container.get(MemoryManager)
            self.context_manager = container.get(ContextManager)
            self.session_manager = container.get(SessionManager)
            self.component_manager = container.get(ComponentManager)
        except Exception:
            self.logger.warning("Some core components not available")
            self.memory_manager = None
            self.context_manager = None
            self.session_manager = None
            self.component_manager = None
        
        # Learning integration
        try:
            self.continual_learner = container.get(ContinualLearner)
            self.preference_learner = container.get(PreferenceLearner)
            self.feedback_processor = container.get(FeedbackProcessor)
        except Exception:
            self.logger.warning("Learning components not available")
            self.continual_learner = None
            self.preference_learner = None
            self.feedback_processor = None
        
        # Monitoring setup
        self._setup_monitoring()
        
        # Configuration
        self.device = self.config.get("attention.device", "cpu")
        self.default_config = AttentionConfiguration(
            attention_dim=self.config.get("attention.attention_dim", 512),
            num_heads=self.config.get("attention.num_heads", 8),
            dropout_rate=self.config.get("attention.dropout_rate", 0.1),
            enable_caching=self.config.get("attention.enable_caching", True),
            real_time_mode=self.config.get("attention.real_time_mode", False)
        )
        
        # Attention mechanisms
        self.attention_mechanisms: Dict[AttentionType, BaseAttentionMechanism] = {}
        self._mechanism_lock = asyncio.Lock()
        
        # Performance tracking
        self._attention_cache = {}
        self._performance_stats = defaultdict(list)
        self._pattern_history = deque(maxlen=1000)
        
        # Background tasks
        self._background_tasks: List[asyncio.Task] = []
        self._shutdown_event = asyncio.Event()
        
        # Register health check
        self.health_check.register_component("cross_modal_attention", self._health_check_callback)
        
        self.logger.info("EnhancedCrossModalAttention initialized")

    def _setup_monitoring(self) -> None:
        """Setup monitoring and metrics collection."""
        try:
            self.metrics = self.container.get(MetricsCollector)
            self.tracer = self.container.get(TraceManager)
            
            # Register attention metrics
            self.metrics.register_counter("attention_computations_total")
            self.metrics.register_histogram("attention_computation_duration_seconds")
            self.metrics.register_counter("attention_mechanism_usage")
            self.metrics.register_gauge("attention_cache_size")
            self.metrics.register_histogram("attention_quality_score")
            self.metrics.register_counter("attention_errors_total")
            self.metrics.register_histogram("cross_modal_alignment_score")
            
        except Exception as e:
            self.logger.warning(f"Failed to setup monitoring: {str(e)}")
            self.metrics = None
            self.tracer = None

    async def initialize(self) -> None:
        """Initialize cross-modal attention mechanisms."""
        try:
            self.logger.info("Initializing cross-modal attention mechanisms...")
            
            # Initialize attention mechanisms
            await self._initialize_attention_mechanisms()
            
            # Register event handlers
            await self._register_event_handlers()
            
            # Start background tasks
            await self._start_background_tasks()
            
            self.logger.info("Cross-modal attention system initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize cross-modal attention: {str(e)}")
            raise CrossModalAttentionError(f"Initialization failed: {str(e)}")

    async def _initialize_attention_mechanisms(self) -> None:
        """Initialize different attention mechanisms."""
        mechanism_configs = {
            AttentionType.MULTI_HEAD_ATTENTION: MultiHeadCrossModalAttention,
            AttentionType.SEMANTIC_ATTENTION: SemanticCrossModalAttention,
            AttentionType.TEMPORAL_ATTENTION: TemporalCrossModalAttention
        }
        
        for attention_type, mechanism_class in mechanism_configs.items():
            try:
                mechanism = mechanism_class(self.default_config, self.device)
                await mechanism.initialize()
                self.attention_mechanisms[attention_type] = mechanism
                self.logger.debug(f"Initialized attention mechanism: {attention_type.value}")
            except Exception as e:
                self.logger.error(f"Failed to initialize {attention_type.value}: {str(e)}")

    async def _register_event_handlers(self) -> None:
        """Register event handlers for system events."""
        # Session events
        self.event_bus.subscribe("session_started", self._handle_session_started)
        self.event_bus.subscribe("session_ended", self._handle_session_ended)
        
        # Processing events
        self.event_bus.subscribe("modality_processing_completed", self._handle_modality_processed)
        self.event_bus.subscribe("fusion_started", self._handle_fusion_started)
        
        # Learning events
        if self.feedback_processor:
            self.event_bus.subscribe("feedback_received", self._handle_feedback_received)
        
        # System events
        self.event_bus.subscribe("system_shutdown_started", self._handle_system_shutdown)

    async def _start_background_tasks(self) -> None:
        """Start background maintenance tasks."""
        # Cache cleanup
        if self.default_config.enable_caching:
            self._background_tasks.append(
                asyncio.create_task(self._cache_cleanup_loop())
            )
        
        # Performance monitoring
        self._background_tasks.append(
            asyncio.create_task(self._performance_monitoring_loop())
        )
        
        # Pattern analysis
        self._background_tasks.append(
            asyncio.create_task(self._pattern_analysis_loop())
        )
        
        # Learning updates
        if self.continual_learner:
            self._background_tasks.append(
                asyncio.create_task(self._learning_update_loop())
            )

    @handle_exceptions
    async def compute_cross_modal_attention(
        self,
        modality_features: Dict[str, Any],
        attention_type: Optional[AttentionType] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> AttentionOutput:
        """
        Compute cross-modal attention across different modalities.
        
        Args:
            modality_features: Dictionary of modality features
            attention_type: Specific attention mechanism to use
            context: Optional context information
            
        Returns:
            AttentionOutput containing attention results and metadata
        """
        start_time = time.time()
        
        try:
            with self.tracer.trace("cross_modal_attention") if self.tracer else None:
                # Convert to ModalityFeatures format
                processed_features = self._convert_modality_features(modality_features)
                
                if not processed_features:
                    return AttentionOutput(
                        success=False,
                        attention_type=attention_type or AttentionType.MULTI_HEAD_ATTENTION,
                        processing_time=time.time() - start_time,
                        attended_features=torch.zeros(1),
                        attention_weights=torch.zeros(1),
                        errors=["No valid modality features provided"],
                        session_id=context.get('session_id', '') if context else '',
                        request_id=context.get('request_id', '') if context else ''
                    )
                
                # Check cache if enabled
                cache_key = None
                if self.default_config.enable_caching:
                    cache_key = self._compute_cache_key(processed_features, attention_type, context)
                    cached_result = self._attention_cache.get(cache_key)
                    if cached_result:
                        self.logger.debug("Using cached attention result")
                        if self.metrics:
                            self.metrics.increment("attention_cache_hits")
                        return cached_result
                
                # Emit attention started event
                await self.event_bus.emit(CrossModalAlignmentStarted(
                    session_id=context.get('session_id', '') if context else '',
                    modalities=list(processed_features.keys()),
                    request_id=context.get('request_id', '') if context else ''
                ))
                
                # Select attention mechanism
                selected_type = attention_type or self._select_optimal_attention_type(
                    processed_features, context
                )
                
                mechanism = self.attention_mechanisms.get(selected_type)
                if not mechanism:
                    raise CrossModalAttentionError(f"Attention mechanism {selected_type} not available")
                
                # Compute attention
                result = await mechanism.compute_attention(processed_features, context)
                
                # Cache result if successful
                if self.default_config.enable_caching and cache_key and result.success:
                    self._attention_cache[cache_key] = result
                
                # Update performance statistics
                self._performance_stats[selected_type].append({
                    'processing_time': result.processing_time,
                    'quality': result.alignment_quality,
                    'entropy': result.attention_entropy,
                    'timestamp': datetime.now(timezone.utc)
                })
                
                # Store pattern for learning
                self._pattern_history.append({
                    'attention_type': selected_type,
                    'modalities': list(processed_features.keys()),
                    'quality': result.alignment_quality,
                    'context': context
                })
                
                # Emit completion event
                await self.event_bus.emit(CrossModalAlignmentCompleted(
                    session_id=context.get('session_id', '') if context else '',
                    alignment_quality=result.alignment_quality,
                    request_id=context.get('request_id', '') if context else ''
                ))
                
                # Update metrics
                if self.metrics:
                    self.metrics.increment("attention_computations_total")
                    self.metrics.record("attention_computation_duration_seconds", result.processing_time)
                    self.metrics.increment("attention_mechanism_usage", 
                                         tags={'mechanism': selected_type.value})
                    self.metrics.record("attention_quality_score", result.alignment_quality)
                    self.metrics.record("cross_modal_alignment_score", result.alignment_quality)
                
                # Store for learning if available
                if self.memory_manager and result.success:
                    await self._store_attention_for_learning(processed_features, result, context)
                
                self.logger.debug(
                    f"Cross-modal attention completed: type={selected_type.value}, "
                    f"quality={result.alignment_quality:.3f}, time={result.processing_time:.3f}s"
                )
                
                return result
                
        except Exception as e:
            # Handle attention error
            processing_time = time.time() - start_time
            
            error_result = AttentionOutput(
                success=False,
                attention_type=attention_type or AttentionType.MULTI_HEAD_ATTENTION,
                processing_time=processing_time,
                attended_features=torch.zeros(1),
                attention_weights=torch.zeros(1),
                errors=[str(e)],
                session_id=context.get('session_id', '') if context else '',
                request_id=context.get('request_id', '') if context else ''
            )
            
            # Emit error event
            await self.event_bus.emit(CrossModalAlignmentFailed(
                session_id=context.get('session_id', '') if context else '',
                error_message=str(e),
                request_id=context.get('request_id', '') if context else ''
            ))
            
            if self.metrics:
                self.metrics.increment("attention_errors_total")
            
            self.logger.error(f"Cross-modal attention failed: {str(e)}")
            return error_result

    def _convert_modality_features(
        self,
        modality_features: Dict[str, Any]
    ) -> Dict[ModalityType, ModalityFeatures]:
        """Convert raw modality features to ModalityFeatures format."""
        converted = {}
        
        for modality_str, data in modality_features.items():
            try:
                # Determine modality type
                modality_type = self._infer_modality_type(modality_str, data)
                if not modality_type:
                    continue
                
                # Extract tensor features
                features = self._extract_tensor_features(data)
                if features is None:
                    continue
                
                # Extract metadata
                confidence, quality, metadata = self._extract_feature_metadata(data)
                
                converted[modality_type] = ModalityFeatures(
                    modality=modality_type,
                    features=features,
                    confidence=confidence,
                    quality_score=quality,
                    metadata=metadata
                )
                
            except Exception as e:
                self.logger.warning(f"Failed to convert modality {modality_str}: {str(e)}")
        
        return converted

    def _infer_modality_type(self, modality_str: str, data: Any) -> Optional[ModalityType]:
        """Infer modality type from string and data."""
        modality_str = modality_str.lower()
        
        # Direct mapping
        modality_mapping = {
            'text': ModalityType.TEXT,
            'speech': ModalityType.SPEECH,
            'audio': ModalityType.AUDIO,
            'vision': ModalityType.VISION,
            'image': ModalityType.VISION,
            'gesture': ModalityType.GESTURE,
            'emotion': ModalityType.EMOTION,
            'transcription': ModalityType.TEXT,
            'speaker': ModalityType.SPEECH
        }
        
        for key, modality_type in modality_mapping.items():
            if key in modality_str:
                return modality_type
        
        return None

    def _extract_tensor_features(self, data: Any) -> Optional[torch.Tensor]:
        """Extract tensor features from various data formats."""
        if isinstance(data, torch.Tensor):
            return data.to(self.device)
        elif isinstance(data, np.ndarray):
            return torch.tensor(data, dtype=torch.float32, device=self.device)
        elif isinstance(data, dict):
            # Look for features in dictionary
            for key in ['features', 'embeddings', 'data
