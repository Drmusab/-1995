"""
Advanced Multimodal Fusion Strategies
Author: Drmusab
Last Modified: 2025-06-03 19:53:04 UTC

This module provides comprehensive multimodal fusion strategies for combining
information from different modalities (text, speech, vision, gesture) into
unified representations for improved AI assistant performance.
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
from concurrent.futures import ThreadPoolExecutor

# Core imports
from src.core.config.loader import ConfigLoader
from src.core.events.event_bus import EventBus
from src.core.events.event_types import (
    FusionStarted, FusionCompleted, FusionError, ModalityProcessingStarted,
    ModalityProcessingCompleted, AttentionWeightsComputed, FusionStrategyChanged,
    CrossModalAlignmentCompleted, ModalityDropped, QualityThresholdExceeded
)
from src.core.error_handling import ErrorHandler, handle_exceptions
from src.core.dependency_injection import Container
from src.core.health_check import HealthCheck

# Processing components
from src.processing.speech.emotion_detection import EmotionResult
from src.processing.speech.speaker_recognition import SpeakerRecognitionResult
from src.processing.speech.speech_to_text import TranscriptionResult

# Memory and learning
from src.memory.memory_manager import MemoryManager
from src.learning.feedback_processor import FeedbackProcessor

# Observability
from src.observability.monitoring.metrics import MetricsCollector
from src.observability.monitoring.tracing import TraceManager
from src.observability.logging.config import get_logger


# Type definitions
T = TypeVar('T')


class FusionStrategy(Enum):
    """Types of multimodal fusion strategies."""
    EARLY_FUSION = "early_fusion"              # Feature-level fusion
    LATE_FUSION = "late_fusion"                # Decision-level fusion
    HYBRID_FUSION = "hybrid_fusion"            # Combination of early and late
    ATTENTION_FUSION = "attention_fusion"      # Attention-based fusion
    HIERARCHICAL_FUSION = "hierarchical_fusion"  # Multi-level hierarchical
    ADAPTIVE_FUSION = "adaptive_fusion"        # Context-adaptive fusion
    CROSS_MODAL_ATTENTION = "cross_modal_attention"  # Cross-modal attention
    TRANSFORMER_FUSION = "transformer_fusion"  # Transformer-based fusion
    GRAPH_FUSION = "graph_fusion"              # Graph neural network fusion
    PROGRESSIVE_FUSION = "progressive_fusion"   # Progressive refinement


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


class FusionQuality(Enum):
    """Quality levels for fusion processing."""
    FAST = "fast"              # Speed-optimized
    BALANCED = "balanced"      # Balance of speed and quality
    QUALITY = "quality"        # Quality-optimized
    ADAPTIVE = "adaptive"      # Context-adaptive quality


class AlignmentType(Enum):
    """Types of cross-modal alignment."""
    TEMPORAL = "temporal"      # Temporal alignment
    SEMANTIC = "semantic"      # Semantic alignment
    SPATIAL = "spatial"        # Spatial alignment
    ATTENTION = "attention"    # Attention-based alignment
    LEARNED = "learned"        # Learned alignment


@dataclass
class ModalityData:
    """Container for modality-specific data."""
    modality: ModalityType
    data: Any
    features: Optional[np.ndarray] = None
    embeddings: Optional[np.ndarray] = None
    confidence: float = 0.0
    quality_score: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    processing_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    attention_weights: Optional[np.ndarray] = None
    alignment_info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FusionContext:
    """Context for fusion operations."""
    session_id: str
    request_id: str
    user_id: Optional[str] = None
    
    # Fusion configuration
    strategy: FusionStrategy = FusionStrategy.ADAPTIVE_FUSION
    quality: FusionQuality = FusionQuality.BALANCED
    modality_weights: Dict[ModalityType, float] = field(default_factory=dict)
    
    # Processing hints
    real_time_mode: bool = False
    streaming_mode: bool = False
    max_latency_ms: float = 1000.0
    
    # Context information
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    environmental_context: Dict[str, Any] = field(default_factory=dict)
    
    # Technical constraints
    available_modalities: Set[ModalityType] = field(default_factory=set)
    resource_constraints: Dict[str, float] = field(default_factory=dict)
    
    # Metadata
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    tags: Set[str] = field(default_factory=set)


@dataclass
class FusionResult:
    """Result of multimodal fusion operation."""
    success: bool
    strategy_used: FusionStrategy
    processing_time: float
    
    # Fused representations
    fused_features: Optional[np.ndarray] = None
    fused_embeddings: Optional[np.ndarray] = None
    fused_attention: Optional[np.ndarray] = None
    
    # Quality metrics
    overall_confidence: float = 0.0
    fusion_quality: float = 0.0
    modality_contributions: Dict[ModalityType, float] = field(default_factory=dict)
    alignment_scores: Dict[str, float] = field(default_factory=dict)
    
    # Attention and importance
    attention_weights: Dict[ModalityType, np.ndarray] = field(default_factory=dict)
    cross_modal_attention: Optional[np.ndarray] = None
    importance_scores: Dict[ModalityType, float] = field(default_factory=dict)
    
    # Semantic information
    unified_representation: Optional[Dict[str, Any]] = None
    semantic_alignment: Dict[str, Any] = field(default_factory=dict)
    contextual_information: Dict[str, Any] = field(default_factory=dict)
    
    # Performance metrics
    latency_breakdown: Dict[str, float] = field(default_factory=dict)
    memory_usage: Dict[str, float] = field(default_factory=dict)
    computational_cost: float = 0.0
    
    # Error handling
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    fallback_used: bool = False
    
    # Metadata
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    request_id: str = ""
    session_id: str = ""


class FusionError(Exception):
    """Custom exception for fusion operations."""
    
    def __init__(self, message: str, strategy: Optional[FusionStrategy] = None, 
                 modalities: Optional[List[ModalityType]] = None):
        super().__init__(message)
        self.strategy = strategy
        self.modalities = modalities
        self.timestamp = datetime.now(timezone.utc)


class BaseFusionStrategy(ABC):
    """Abstract base class for fusion strategies."""
    
    def __init__(self, logger: logging.Logger, device: str = "cpu"):
        self.logger = logger
        self.device = device
        self.is_initialized = False
        self._model_cache = {}
        self._performance_stats = defaultdict(list)
    
    @abstractmethod
    async def fuse_modalities(
        self, 
        modalities: Dict[ModalityType, ModalityData], 
        context: FusionContext
    ) -> FusionResult:
        """Fuse multiple modalities into unified representation."""
        pass
    
    @abstractmethod
    def supports_real_time(self) -> bool:
        """Check if strategy supports real-time processing."""
        pass
    
    @abstractmethod
    def get_computational_cost(self) -> float:
        """Get relative computational cost of the strategy."""
        pass
    
    async def initialize(self) -> None:
        """Initialize the fusion strategy."""
        self.is_initialized = True
    
    async def cleanup(self) -> None:
        """Cleanup resources."""
        self.is_initialized = False
        self._model_cache.clear()
    
    def _validate_modalities(self, modalities: Dict[ModalityType, ModalityData]) -> None:
        """Validate input modalities."""
        if not modalities:
            raise FusionError("No modalities provided for fusion")
        
        for modality_type, data in modalities.items():
            if data.confidence < 0.1:
                self.logger.warning(f"Low confidence modality: {modality_type.value}")
    
    def _compute_quality_score(self, modalities: Dict[ModalityType, ModalityData]) -> float:
        """Compute overall quality score from modalities."""
        if not modalities:
            return 0.0
        
        quality_scores = [data.quality_score for data in modalities.values()]
        confidence_scores = [data.confidence for data in modalities.values()]
        
        # Weighted average of quality and confidence
        weights = [0.6, 0.4]  # Quality has higher weight
        weighted_scores = [
            np.mean(quality_scores) * weights[0],
            np.mean(confidence_scores) * weights[1]
        ]
        
        return sum(weighted_scores)


class EarlyFusionStrategy(BaseFusionStrategy):
    """Early fusion at feature level."""
    
    def __init__(self, logger: logging.Logger, device: str = "cpu"):
        super().__init__(logger, device)
        self.feature_dim = 512
        self.fusion_network = None
    
    async def initialize(self) -> None:
        """Initialize early fusion components."""
        await super().initialize()
        
        # Initialize feature fusion network
        self.fusion_network = EarlyFusionNetwork(
            input_dims={
                ModalityType.TEXT: 768,
                ModalityType.SPEECH: 1024,
                ModalityType.VISION: 2048,
                ModalityType.EMOTION: 128
            },
            output_dim=self.feature_dim
        ).to(self.device)
        
        self.logger.info("EarlyFusionStrategy initialized")
    
    async def fuse_modalities(
        self, 
        modalities: Dict[ModalityType, ModalityData], 
        context: FusionContext
    ) -> FusionResult:
        """Perform early fusion of modalities."""
        start_time = time.time()
        
        try:
            self._validate_modalities(modalities)
            
            # Extract and concatenate features
            feature_tensors = {}
            attention_weights = {}
            
            for modality_type, data in modalities.items():
                if data.features is not None:
                    features = torch.tensor(data.features, dtype=torch.float32).to(self.device)
                    feature_tensors[modality_type] = features
                    
                    # Compute attention weights based on confidence
                    attention_weights[modality_type] = np.array([data.confidence])
            
            if not feature_tensors:
                raise FusionError("No features available for early fusion")
            
            # Apply fusion network
            fused_features = None
            if self.fusion_network:
                fused_features = self.fusion_network(feature_tensors)
                fused_features = fused_features.detach().cpu().numpy()
            else:
                # Simple concatenation fallback
                concatenated = torch.cat(list(feature_tensors.values()), dim=-1)
                fused_features = concatenated.detach().cpu().numpy()
            
            processing_time = time.time() - start_time
            
            return FusionResult(
                success=True,
                strategy_used=FusionStrategy.EARLY_FUSION,
                processing_time=processing_time,
                fused_features=fused_features,
                overall_confidence=np.mean([data.confidence for data in modalities.values()]),
                fusion_quality=self._compute_quality_score(modalities),
                modality_contributions={
                    mod: data.confidence for mod, data in modalities.items()
                },
                attention_weights=attention_weights,
                request_id=context.request_id,
                session_id=context.session_id
            )
            
        except Exception as e:
            self.logger.error(f"Early fusion failed: {str(e)}")
            return FusionResult(
                success=False,
                strategy_used=FusionStrategy.EARLY_FUSION,
                processing_time=time.time() - start_time,
                errors=[str(e)],
                request_id=context.request_id,
                session_id=context.session_id
            )
    
    def supports_real_time(self) -> bool:
        return True
    
    def get_computational_cost(self) -> float:
        return 0.3  # Low cost


class LateFusionStrategy(BaseFusionStrategy):
    """Late fusion at decision level."""
    
    def __init__(self, logger: logging.Logger, device: str = "cpu"):
        super().__init__(logger, device)
        self.decision_weights = {}
    
    async def fuse_modalities(
        self, 
        modalities: Dict[ModalityType, ModalityData], 
        context: FusionContext
    ) -> FusionResult:
        """Perform late fusion of modalities."""
        start_time = time.time()
        
        try:
            self._validate_modalities(modalities)
            
            # Process each modality independently first
            modality_decisions = {}
            modality_confidences = {}
            
            for modality_type, data in modalities.items():
                # Extract decision-level representations
                if data.embeddings is not None:
                    decision = self._process_modality_decision(modality_type, data)
                    modality_decisions[modality_type] = decision
                    modality_confidences[modality_type] = data.confidence
            
            # Weighted fusion of decisions
            fused_decision = self._weighted_decision_fusion(
                modality_decisions, modality_confidences, context
            )
            
            processing_time = time.time() - start_time
            
            return FusionResult(
                success=True,
                strategy_used=FusionStrategy.LATE_FUSION,
                processing_time=processing_time,
                unified_representation=fused_decision,
                overall_confidence=np.mean(list(modality_confidences.values())),
                fusion_quality=self._compute_quality_score(modalities),
                modality_contributions=modality_confidences,
                request_id=context.request_id,
                session_id=context.session_id
            )
            
        except Exception as e:
            self.logger.error(f"Late fusion failed: {str(e)}")
            return FusionResult(
                success=False,
                strategy_used=FusionStrategy.LATE_FUSION,
                processing_time=time.time() - start_time,
                errors=[str(e)],
                request_id=context.request_id,
                session_id=context.session_id
            )
    
    def _process_modality_decision(
        self, 
        modality_type: ModalityType, 
        data: ModalityData
    ) -> Dict[str, Any]:
        """Process individual modality to decision level."""
        # This would contain modality-specific decision processing
        return {
            "type": modality_type.value,
            "embeddings": data.embeddings,
            "confidence": data.confidence,
            "metadata": data.metadata
        }
    
    def _weighted_decision_fusion(
        self,
        decisions: Dict[ModalityType, Dict[str, Any]],
        confidences: Dict[ModalityType, float],
        context: FusionContext
    ) -> Dict[str, Any]:
        """Fuse decisions with weighted voting."""
        # Normalize confidence weights
        total_confidence = sum(confidences.values())
        normalized_weights = {
            mod: conf / total_confidence 
            for mod, conf in confidences.items()
        } if total_confidence > 0 else {}
        
        # Apply context-based weight adjustments
        if context.modality_weights:
            for mod, weight in context.modality_weights.items():
                if mod in normalized_weights:
                    normalized_weights[mod] *= weight
        
        return {
            "decisions": decisions,
            "weights": normalized_weights,
            "fusion_method": "weighted_voting"
        }
    
    def supports_real_time(self) -> bool:
        return True
    
    def get_computational_cost(self) -> float:
        return 0.4  # Medium cost


class AttentionFusionStrategy(BaseFusionStrategy):
    """Attention-based multimodal fusion."""
    
    def __init__(self, logger: logging.Logger, device: str = "cpu"):
        super().__init__(logger, device)
        self.attention_dim = 256
        self.attention_network = None
    
    async def initialize(self) -> None:
        """Initialize attention fusion components."""
        await super().initialize()
        
        # Initialize attention network
        self.attention_network = MultiModalAttentionNetwork(
            modality_dims={
                ModalityType.TEXT: 768,
                ModalityType.SPEECH: 1024,
                ModalityType.VISION: 2048,
                ModalityType.EMOTION: 128
            },
            attention_dim=self.attention_dim
        ).to(self.device)
        
        self.logger.info("AttentionFusionStrategy initialized")
    
    async def fuse_modalities(
        self, 
        modalities: Dict[ModalityType, ModalityData], 
        context: FusionContext
    ) -> FusionResult:
        """Perform attention-based fusion."""
        start_time = time.time()
        
        try:
            self._validate_modalities(modalities)
            
            # Prepare modality features
            modality_features = {}
            for modality_type, data in modalities.items():
                if data.features is not None:
                    features = torch.tensor(data.features, dtype=torch.float32).to(self.device)
                    modality_features[modality_type] = features
            
            if not modality_features:
                raise FusionError("No features available for attention fusion")
            
            # Apply attention mechanism
            if self.attention_network:
                fused_output, attention_weights = self.attention_network(modality_features)
                fused_features = fused_output.detach().cpu().numpy()
                attention_weights_np = {
                    mod: weights.detach().cpu().numpy() 
                    for mod, weights in attention_weights.items()
                }
            else:
                # Fallback simple attention
                fused_features, attention_weights_np = self._simple_attention_fusion(
                    modality_features, modalities
                )
            
            processing_time = time.time() - start_time
            
            return FusionResult(
                success=True,
                strategy_used=FusionStrategy.ATTENTION_FUSION,
                processing_time=processing_time,
                fused_features=fused_features,
                attention_weights=attention_weights_np,
                overall_confidence=np.mean([data.confidence for data in modalities.values()]),
                fusion_quality=self._compute_quality_score(modalities),
                modality_contributions={
                    mod: np.mean(attention_weights_np.get(mod, [0])) 
                    for mod in modalities.keys()
                },
                request_id=context.request_id,
                session_id=context.session_id
            )
            
        except Exception as e:
            self.logger.error(f"Attention fusion failed: {str(e)}")
            return FusionResult(
                success=False,
                strategy_used=FusionStrategy.ATTENTION_FUSION,
                processing_time=time.time() - start_time,
                errors=[str(e)],
                request_id=context.request_id,
                session_id=context.session_id
            )
    
    def _simple_attention_fusion(
        self,
        modality_features: Dict[ModalityType, torch.Tensor],
        modalities: Dict[ModalityType, ModalityData]
    ) -> Tuple[np.ndarray, Dict[ModalityType, np.ndarray]]:
        """Simple attention-based fusion fallback."""
        # Compute attention weights based on confidence and quality
        attention_weights = {}
        weighted_features = []
        
        total_weight = 0
        for modality_type, features in modality_features.items():
            data = modalities[modality_type]
            weight = data.confidence * data.quality_score
            attention_weights[modality_type] = np.array([weight])
            weighted_features.append(features * weight)
            total_weight += weight
        
        # Normalize weights
        if total_weight > 0:
            for modality_type in attention_weights:
                attention_weights[modality_type] /= total_weight
        
        # Sum weighted features
        fused_tensor = torch.sum(torch.stack(weighted_features), dim=0)
        fused_features = fused_tensor.detach().cpu().numpy()
        
        return fused_features, attention_weights
    
    def supports_real_time(self) -> bool:
        return False  # More computationally intensive
    
    def get_computational_cost(self) -> float:
        return 0.7  # High cost


class AdaptiveFusionStrategy(BaseFusionStrategy):
    """Adaptive fusion that selects optimal strategy based on context."""
    
    def __init__(self, logger: logging.Logger, device: str = "cpu"):
        super().__init__(logger, device)
        self.available_strategies = {}
        self.strategy_selector = None
        self.performance_history = defaultdict(list)
    
    async def initialize(self) -> None:
        """Initialize adaptive fusion components."""
        await super().initialize()
        
        # Initialize available strategies
        self.available_strategies = {
            FusionStrategy.EARLY_FUSION: EarlyFusionStrategy(self.logger, self.device),
            FusionStrategy.LATE_FUSION: LateFusionStrategy(self.logger, self.device),
            FusionStrategy.ATTENTION_FUSION: AttentionFusionStrategy(self.logger, self.device)
        }
        
        # Initialize all strategies
        for strategy in self.available_strategies.values():
            await strategy.initialize()
        
        # Initialize strategy selector
        self.strategy_selector = StrategySelector(self.logger)
        
        self.logger.info("AdaptiveFusionStrategy initialized")
    
    async def fuse_modalities(
        self, 
        modalities: Dict[ModalityType, ModalityData], 
        context: FusionContext
    ) -> FusionResult:
        """Adaptively select and apply fusion strategy."""
        start_time = time.time()
        
        try:
            self._validate_modalities(modalities)
            
            # Select optimal strategy
            selected_strategy_type = self.strategy_selector.select_strategy(
                modalities, context, self.performance_history
            )
            
            selected_strategy = self.available_strategies.get(selected_strategy_type)
            if not selected_strategy:
                # Fallback to early fusion
                selected_strategy = self.available_strategies[FusionStrategy.EARLY_FUSION]
                selected_strategy_type = FusionStrategy.EARLY_FUSION
            
            # Apply selected strategy
            result = await selected_strategy.fuse_modalities(modalities, context)
            
            # Update performance history
            self.performance_history[selected_strategy_type].append({
                'quality': result.fusion_quality,
                'confidence': result.overall_confidence,
                'processing_time': result.processing_time,
                'timestamp': datetime.now(timezone.utc)
            })
            
            # Keep only recent history
            max_history = 100
            if len(self.performance_history[selected_strategy_type]) > max_history:
                self.performance_history[selected_strategy_type] = \
                    self.performance_history[selected_strategy_type][-max_history:]
            
            # Update result to reflect adaptive strategy
            result.strategy_used = FusionStrategy.ADAPTIVE_FUSION
            result.contextual_information = {
                'selected_strategy': selected_strategy_type.value,
                'selection_reason': self.strategy_selector.get_last_selection_reason(),
                'available_strategies': list(self.available_strategies.keys())
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Adaptive fusion failed: {str(e)}")
            return FusionResult(
                success=False,
                strategy_used=FusionStrategy.ADAPTIVE_FUSION,
                processing_time=time.time() - start_time,
                errors=[str(e)],
                request_id=context.request_id,
                session_id=context.session_id
            )
    
    def supports_real_time(self) -> bool:
        return True  # Can select real-time capable strategies
    
    def get_computational_cost(self) -> float:
        return 0.5  # Variable based on selected strategy


class CrossModalAttentionStrategy(BaseFusionStrategy):
    """Cross-modal attention fusion for better alignment."""
    
    def __init__(self, logger: logging.Logger, device: str = "cpu"):
        super().__init__(logger, device)
        self.cross_attention_network = None
        self.alignment_network = None
    
    async def initialize(self) -> None:
        """Initialize cross-modal attention components."""
        await super().initialize()
        
        # Initialize cross-modal attention
        self.cross_attention_network = CrossModalAttentionNetwork(
            modality_dims={
                ModalityType.TEXT: 768,
                ModalityType.SPEECH: 1024,
                ModalityType.VISION: 2048,
                ModalityType.EMOTION: 128
            },
            hidden_dim=512
        ).to(self.device)
        
        # Initialize alignment network
        self.alignment_network = ModalityAlignmentNetwork(
            modality_dims={
                ModalityType.TEXT: 768,
                ModalityType.SPEECH: 1024,
                ModalityType.VISION: 2048
            }
        ).to(self.device)
        
        self.logger.info("CrossModalAttentionStrategy initialized")
    
    async def fuse_modalities(
        self, 
        modalities: Dict[ModalityType, ModalityData], 
        context: FusionContext
    ) -> FusionResult:
        """Perform cross-modal attention fusion."""
        start_time = time.time()
        
        try:
            self._validate_modalities(modalities)
            
            # Prepare modality features
            modality_features = {}
            for modality_type, data in modalities.items():
                if data.features is not None:
                    features = torch.tensor(data.features, dtype=torch.float32).to(self.device)
                    modality_features[modality_type] = features
            
            if len(modality_features) < 2:
                raise FusionError("Cross-modal attention requires at least 2 modalities")
            
            # Compute cross-modal alignments
            alignment_scores = {}
            if self.alignment_network:
                alignment_scores = self.alignment_network(modality_features)
            
            # Apply cross-modal attention
            if self.cross_attention_network:
                fused_output, cross_attention_weights = self.cross_attention_network(
                    modality_features, alignment_scores
                )
                fused_features = fused_output.detach().cpu().numpy()
                cross_attention_np = cross_attention_weights.detach().cpu().numpy()
            else:
                # Fallback implementation
                fused_features, cross_attention_np = self._compute_cross_modal_attention(
                    modality_features, modalities
                )
            
            processing_time = time.time() - start_time
            
            return FusionResult(
                success=True,
                strategy_used=FusionStrategy.CROSS_MODAL_ATTENTION,
                processing_time=processing_time,
                fused_features=fused_features,
                cross_modal_attention=cross_attention_np,
                alignment_scores={
                    f"{mod1.value}_{mod2.value}": score 
                    for (mod1, mod2), score in alignment_scores.items()
                } if alignment_scores else {},
                overall_confidence=np.mean([data.confidence for data in modalities.values()]),
                fusion_quality=self._compute_quality_score(modalities),
                request_id=context.request_id,
                session_id=context.session_id
            )
            
        except Exception as e:
            self.logger.error(f"Cross-modal attention fusion failed: {str(e)}")
            return FusionResult(
                success=False,
                strategy_used=FusionStrategy.CROSS_MODAL_ATTENTION,
                processing_time=time.time() - start_time,
                errors=[str(e)],
                request_id=context.request_id,
                session_id=context.session_id
            )
    
    def _compute_cross_modal_attention(
        self,
        modality_features: Dict[ModalityType, torch.Tensor],
        modalities: Dict[ModalityType, ModalityData]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute cross-modal attention fallback."""
        # Simple cross-modal attention based on feature similarity
        modality_list = list(modality_features.keys())
        attention_matrix = torch.zeros(len(modality_list), len(modality_list))
        
        for i, mod1 in enumerate(modality_list):
            for j, mod2 in enumerate(modality_list):
                if i != j:
                    # Compute cosine similarity
                    feat1 = modality_features[mod1].flatten()
                    feat2 = modality_features[mod2].flatten()
                    similarity = F.cosine_similarity(feat1, feat2, dim=0)
                    attention_matrix[i, j] = similarity
        
        # Apply attention to features
        weighted_features = []
        for i, modality_type in enumerate(modality_list):
            weight = torch.softmax(attention_matrix[i], dim=0)[i]
            weighted_features.append(modality_features[modality_type] * weight)
        
        fused_tensor = torch.mean(torch.stack(weighted_features), dim=0)
        fused_features = fused_tensor.detach().cpu().numpy()
        cross_attention_np = attention_matrix.detach().cpu().numpy()
        
        return fused_features, cross_attention_np
    
    def supports_real_time(self) -> bool:
        return False  # Computationally intensive
    
    def get_computational_cost(self) -> float:
        return 0.9  # Very high cost


# Neural Network Components

class EarlyFusionNetwork(nn.Module):
    """Neural network for early fusion."""
    
    def __init__(self, input_dims: Dict[ModalityType, int], output_dim: int):
        super().__init__()
        self.input_dims = input_dims
        self.output_dim = output_dim
        
        # Projection layers for each modality
        self.projections = nn.ModuleDict()
        for modality, dim in input_dims.items():
            self.projections[modality.value] = nn.Sequential(
                nn.Linear(dim, output_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
        
        # Fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(output_dim * len(input_dims), output_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim, output_dim)
        )
    
    def forward(self, modality_features: Dict[ModalityType, torch.Tensor]) -> torch.Tensor:
        projected_features = []
        
        for modality, features in modality_features.items():
            if modality.value in self.projections:
                projected = self.projections[modality.value](features)
                projected_features.append(projected)
        
        if not projected_features:
            raise ValueError("No valid modality features provided")
        
        # Concatenate and fuse
        concatenated = torch.cat(projected_features, dim=-1)
        fused = self.fusion_layer(concatenated)
        
        return fused


class MultiModalAttentionNetwork(nn.Module):
    """Multi-modal attention network."""
    
    def __init__(self, modality_dims: Dict[ModalityType, int], attention_dim: int):
        super().__init__()
        self.modality_dims = modality_dims
        self.attention_dim = attention_dim
        
        # Attention layers for each modality
        self.attention_layers = nn.ModuleDict()
        for modality, dim in modality_dims.items():
            self.attention_layers[modality.value] = nn.Sequential(
                nn.Linear(dim, attention_dim),
                nn.Tanh(),
                nn.Linear(attention_dim, 1)
            )
        
        # Feature projection layers
        self.feature_projections = nn.ModuleDict()
        for modality, dim in modality_dims.items():
            self.feature_projections[modality.value] = nn.Linear(dim, attention_dim)
    
    def forward(
        self, 
        modality_features: Dict[ModalityType, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[ModalityType, torch.Tensor]]:
        
        projected_features = {}
        attention_scores = {}
        
        # Compute attention scores and project features
        for modality, features in modality_features.items():
            if modality.value in self.attention_layers:
                # Compute attention
                attention = self.attention_layers[modality.value](features)
                attention_scores[modality] = attention
                
                # Project features
                projected = self.feature_projections[modality.value](features)
                projected_features[modality] = projected
        
        # Normalize attention scores
        all_attention = torch.cat(list(attention_scores.values()), dim=-1)
        normalized_attention = torch.softmax(all_attention, dim=-1)
        
        # Apply attention to features
        attention_weights = {}
        weighted_features = []
        
        for i, (modality, features) in enumerate(projected_features.items()):
            weight = normalized_attention[:, i:i+1]
            attention_weights[modality] = weight
            weighted_features.append(features * weight)
        
        # Combine weighted features
        fused_features = torch.sum(torch.stack(weighted_features), dim=0)
        
        return fused_features, attention_weights


class CrossModalAttentionNetwork(nn.Module):
    """Cross-modal attention network."""
    
    def __init__(self, modality_dims: Dict[ModalityType, int], hidden_dim: int):
        super().__init__()
        self.modality_dims = modality_dims
        self.hidden_dim = hidden_dim
        
        # Query, Key, Value projections for each modality
        self.query_projections = nn.ModuleDict()
        self.key_projections = nn.ModuleDict()
        self.value_projections = nn.ModuleDict()
        
        for modality, dim in modality_dims.items():
            self.query_projections[modality.value] = nn.Linear(dim, hidden_dim)
            self.key_projections[modality.value] = nn.Linear(dim, hidden_dim)
            self.value_projections[modality.value] = nn.Linear(dim, hidden_dim)
        
        self.attention_scale = hidden_dim ** -0.5
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(
        self, 
        modality_features: Dict[ModalityType, torch.Tensor],
        alignment_scores: Optional[Dict] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        queries = {}
        keys = {}
        values = {}
        
        # Compute Q, K, V for each modality
        for modality, features in modality_features.items():
            if modality.value in self.query_projections:
                queries[modality] = self.query_projections[modality.value](features)
                keys[modality] = self.key_projections[modality.value](features)
                values[modality] = self.value_projections[modality.value](features)
        
        # Compute cross-modal attention
        modality_list = list(queries.keys())
        attention_matrix = torch.zeros(len(modality_list), len(modality_list))
        attended_values = []
        
        for i, mod1 in enumerate(modality_list):
            attended_value = torch.zeros_like(values[mod1])
            
            for j, mod2 in enumerate(modality_list):
                # Compute attention score
                score = torch.matmul(queries[mod1], keys[mod2].transpose(-2, -1))
                score = score * self.attention_scale
                
                # Apply alignment scores if available
                if alignment_scores and (mod1, mod2) in alignment_scores:
                    score = score * alignment_scores[(mod1, mod2)]
                
                attention_weight = torch.softmax(score, dim=-1)
                attention_matrix[i, j] = attention_weight.mean()
                
                # Apply attention to values
                attended_value += torch.matmul(attention_weight, values[mod2])
            
            attended_values.append(attended_value)
        
        # Combine attended values
        fused_output = torch.mean(torch.stack(attended_values), dim=0)
        fused_output = self.output_projection(fused_output)
        
        return fused_output, attention_matrix


class ModalityAlignmentNetwork(nn.Module):
    """Network for computing modality alignments."""
    
    def __init__(self, modality_dims: Dict[ModalityType, int]):
        super().__init__()
        self.modality_dims = modality_dims
        
        # Alignment projection layers
        self.alignment_projections = nn.ModuleDict()
        for modality, dim in modality_dims.items():
            self.alignment_projections[modality.value] = nn.Sequential(
                nn.Linear(dim, 256),
                nn.ReLU(),
                nn.Linear(256, 128)
            )
    
    def forward(
        self, 
        modality_features: Dict[ModalityType, torch.Tensor]
    ) -> Dict[Tuple[ModalityType, ModalityType], float]:
        
        # Project features to alignment space
        projected_features = {}
        for modality, features in modality_features.items():
            if modality.value in self.alignment_projections:
                projected = self.alignment_projections[modality.value](features)
                projected_features[modality] = projected
        
        # Compute pairwise alignments
        alignment_scores = {}
        modality_list = list(projected_features.keys())
        
        for i, mod1 in enumerate(modality_list):
            for j, mod2 in enumerate(modality_list):
                if i != j:
                    # Compute cosine similarity
                    feat1 = projected_features[mod1]
                    feat2 = projected_features[mod2]
                    
                    similarity = F.cosine_similarity(feat1, feat2, dim=-1)
                    alignment_scores[(mod1, mod2)] = similarity.mean().item()
        
        return alignment_scores


class StrategySelector:
    """Intelligent strategy selection for adaptive fusion."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.last_selection_reason = ""
    
    def select_strategy(
        self,
        modalities: Dict[ModalityType, ModalityData],
        context: FusionContext,
        performance_history: Dict[FusionStrategy, List[Dict[str, Any]]]
    ) -> FusionStrategy:
        """Select optimal fusion strategy based on context and history."""
        
        # Strategy selection criteria
        criteria = self._analyze_context(modalities, context)
        
        # Real-time requirements
        if context.real_time_mode or context.max_latency_ms < 500:
            if criteria['complexity'] == 'low':
                self.last_selection_reason = "Real-time mode with low complexity"
                return FusionStrategy.EARLY_FUSION
            else:
                self.last_selection_reason = "Real-time mode with medium complexity"
                return FusionStrategy.LATE_FUSION
        
        # Quality requirements
        if context.quality == FusionQuality.QUALITY:
            if len(modalities) >= 3:
                self.last_selection_reason = "High quality with multiple modalities"
                return FusionStrategy.CROSS_MODAL_ATTENTION
            else:
                self.last_selection_reason = "High quality with few modalities"
                return FusionStrategy.ATTENTION_FUSION
        
        # Performance-based selection
        if performance_history:
            best_strategy = self._select_best_performing_strategy(performance_history)
            if best_strategy:
                self.last_selection_reason = "Best historical performance"
                return best_strategy
        
        # Default adaptive selection
        if criteria['modality_count'] <= 2:
            self.last_selection_reason = "Few modalities, using early fusion"
            return FusionStrategy.EARLY_FUSION
        elif criteria['confidence_variance'] > 0.3:
            self.last_selection_reason = "High confidence variance, using attention"
            return FusionStrategy.ATTENTION_FUSION
        else:
            self.last_selection_reason = "Balanced scenario, using late fusion"
            return FusionStrategy.LATE_FUSION
    
    def _analyze_context(
        self,
        modalities: Dict[ModalityType, ModalityData],
        context: FusionContext
    ) -> Dict[str, Any]:
        """Analyze context for strategy selection."""
        confidences = [data.confidence for data in modalities.values()]
        quality_scores = [data.quality_score for data in modalities.values()]
        
        return {
            'modality_count': len(modalities),
            'average_confidence': np.mean(confidences),
            'confidence_variance': np.var(confidences),
            'average_quality': np.mean(quality_scores),
            'complexity': self._estimate_complexity(modalities),
            'temporal_alignment': self._check_temporal_alignment(modalities)
        }
    
    def _estimate_complexity(self, modalities: Dict[ModalityType, ModalityData]) -> str:
        """Estimate processing complexity."""
        complexity_scores = {
            ModalityType.TEXT: 1,
            ModalityType.SPEECH: 2,
            ModalityType.VISION: 3,
            ModalityType.EMOTION: 1,
            ModalityType.GESTURE: 2
        }
        
        total_complexity = sum(
            complexity_scores.get(mod, 1) for mod in modalities.keys()
        )
        
        if total_complexity <= 3:
            return 'low'
        elif total_complexity <= 6:
            return 'medium'
        else:
            return 'high'
    
    def _check_temporal_alignment(self, modalities: Dict[ModalityType, ModalityData]) -> bool:
        """Check if modalities are temporally aligned."""
        timestamps = [data.timestamp for data in modalities.values()]
        if len(timestamps) <= 1:
            return True
        
        # Check if all timestamps are within 1 second of each other
        min_time = min(timestamps)
        max_time = max(timestamps)
        time_diff = (max_time - min_time).total_seconds()
        
        return time_diff <= 1.0
    
    def _select_best_performing_strategy(
        self,
        performance_history: Dict[FusionStrategy, List[Dict[str, Any]]]
    ) -> Optional[FusionStrategy]:
        """Select strategy with best historical performance."""
        strategy_scores = {}
        
        for strategy, history in performance_history.items():
            if len(history) >= 5:  # Need sufficient history
                recent_history = history[-10:]  # Consider recent performance
                
                avg_quality = np.mean([h['quality'] for h in recent_history])
                avg_confidence = np.mean([h['confidence'] for h in recent_history])
                avg_time = np.mean([h['processing_time'] for h in recent_history])
                
                # Composite score (higher is better)
                score = (avg_quality * 0.4 + avg_confidence * 0.4) / (avg_time * 0.2)
                strategy_scores[strategy] = score
        
        if strategy_scores:
            return max(strategy_scores, key=strategy_scores.get)
        
        return None
    
    def get_last_selection_reason(self) -> str:
        """Get reason for last strategy selection."""
        return self.last_selection_reason


class MultimodalFusionStrategy:
    """
    Main multimodal fusion strategy coordinator.
    
    This class manages different fusion strategies and provides a unified
    interface for multimodal fusion operations integrated with the core
    AI assistant system.
    """
    
    def __init__(self, container: Container):
        """
        Initialize the multimodal fusion strategy coordinator.
        
        Args:
            container: Dependency injection container
        """
        self.container = container
        self.logger = get_logger(__name__)
        self.config = container.get(ConfigLoader)
        self.event_bus = container.get(EventBus)
        self.error_handler = container.get(ErrorHandler)
        self.health_check = container.get(HealthCheck)
        
        # Memory and learning integration
        try:
            self.memory_manager = container.get(MemoryManager)
            self.feedback_processor = container.get(FeedbackProcessor)
        except Exception:
            self.logger.warning("Memory/learning components not available")
            self.memory_manager = None
            self.feedback_processor = None
        
        # Monitoring
        self._setup_monitoring()
        
        # Configuration
        self.device = self.config.get("fusion.device", "cpu")
        self.default_strategy = FusionStrategy(
            self.config.get("fusion.default_strategy", "adaptive_fusion")
        )
        self.enable_caching = self.config.get("fusion.enable_caching", True)
        self.cache_ttl = self.config.get("fusion.cache_ttl", 3600)
        
        # Strategy management
        self.strategies: Dict[FusionStrategy, BaseFusionStrategy] = {}
        self._strategy_lock = asyncio.Lock()
        self._fusion_cache = {}
        self._performance_stats = defaultdict(list)
        
        # Background tasks
        self._background_tasks: List[asyncio.Task] = []
        self._shutdown_event = asyncio.Event()
        
        # Register health check
        self.health_check.register_component("fusion_strategies", self._health_check_callback)
        
        self.logger.info("MultimodalFusionStrategy initialized")

    def _setup_monitoring(self) -> None:
        """Setup monitoring and metrics collection."""
        try:
            self.metrics = self.container.get(MetricsCollector)
            self.tracer = self.container.get(TraceManager)
            
            # Register fusion metrics
            self.metrics.register_counter("fusion_operations_total")
            self.metrics.register_histogram("fusion_duration_seconds")
            self.metrics.register_counter("fusion_strategy_usage")
            self.metrics.register_gauge("fusion_cache_size")
            self.metrics.register_histogram("fusion_quality_score")
            self.metrics.register_counter("fusion_errors_total")
            
        except Exception as e:
            self.logger.warning(f"Failed to setup monitoring: {str(e)}")
            self.metrics = None
            self.tracer = None

    async def initialize(self) -> None:
        """Initialize fusion strategies and components."""
        try:
            self.logger.info("Initializing multimodal fusion strategies...")
            
            # Initialize strategies
            await self._initialize_strategies()
            
            # Register event handlers
            await self._register_event_handlers()
            
            # Start background tasks
            await self._start_background_tasks()
            
            self.logger.info("Multimodal fusion strategies initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize fusion strategies: {str(e)}")
            raise FusionError(f"Fusion initialization failed: {str(e)}")

    async def _initialize_strategies(self) -> None:
        """Initialize all fusion strategies."""
        strategy_configs = {
            FusionStrategy.EARLY_FUSION: EarlyFusionStrategy,
            FusionStrategy.LATE_FUSION: LateFusionStrategy,
            FusionStrategy.ATTENTION_FUSION: AttentionFusionStrategy,
            FusionStrategy.ADAPTIVE_FUSION: AdaptiveFusionStrategy,
            FusionStrategy.CROSS_MODAL_ATTENTION: CrossModalAttentionStrategy
        }
        
        for strategy_type, strategy_class in strategy_configs.items():
            try:
                strategy = strategy_class(self.logger, self.device)
                await strategy.initialize()
                self.strategies[strategy_type] = strategy
                self.logger.debug(f"Initialized strategy: {strategy_type.value}")
            except Exception as e:
                self.logger.error(f"Failed to initialize {strategy_type.value}: {str(e)}")

    async def _register_event_handlers(self) -> None:
        """Register event handlers for system events."""
        # Session events
        self.event_bus.subscribe("session_started", self._handle_session_started)
        self.event_bus.subscribe("session_ended", self._handle_session_ended)
        
        # Processing events
        self.event_bus.subscribe("processing_started", self._handle_processing_started)
        self.event_bus.subscribe("processing_completed", self._handle_processing_completed)
        
        # Component health events
        self.event_bus.subscribe("component_health_changed", self._handle_component_health_change)
        
        # System shutdown
        self.event_bus.subscribe("system_shutdown_started", self._handle_system_shutdown)

    async def _start_background_tasks(self) -> None:
        """Start background maintenance tasks."""
        # Cache cleanup task
        if self.enable_caching:
            self._background_tasks.append(
                asyncio.create_task(self._cache_cleanup_loop())
            )
        
        # Performance monitoring task
        self._background_tasks.append(
            asyncio.create_task(self._performance_monitoring_loop())
        )
        
        # Strategy optimization task
        self._background_tasks.append(
            asyncio.create_task(self._strategy_optimization_loop())
        )

    @handle_exceptions
    async def fuse_modalities(
        self,
        modalities: Dict[str, Any],
        weights: Optional[Dict[str, float]] = None,
        strategy: Optional[FusionStrategy] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> FusionResult:
        """
        Fuse multiple modalities into a unified representation.
        
        Args:
            modalities: Dictionary of modality data
            weights: Optional weights for each modality
            strategy: Optional specific strategy to use
            context: Optional context information
            
        Returns:
            FusionResult containing fused representation and metadata
        """
        start_time = time.time()
        
        # Create fusion context
        fusion_context = self._create_fusion_context(context, weights)
        
        # Convert modalities to ModalityData format
        modality_data = self._convert_modalities(modalities)
        
        if not modality_data:
            return FusionResult(
                success=False,
                strategy_used=strategy or self.default_strategy,
                processing_time=time.time() - start_time,
                errors=["No valid modalities provided"],
                request_id=fusion_context.request_id,
                session_id=fusion_context.session_id
            )
        
        try:
            with self.tracer.trace("multimodal_fusion") if self.tracer else None:
                # Emit fusion started event
                await self.event_bus.emit(FusionStarted(
                    session_id=fusion_context.session_id,
                    modalities=list(modality_data.keys()),
                    request_id=fusion_context.request_id
                ))
                
                # Check cache if enabled
                cache_key = None
                if self.enable_caching:
                    cache_key = self._compute_cache_key(modality_data, fusion_context)
                    cached_result = self._fusion_cache.get(cache_key)
                    if cached_result:
                        self.logger.debug("Using cached fusion result")
                        if self.metrics:
                            self.metrics.increment("fusion_cache_hits")
                        return cached_result
                
                # Select strategy
                selected_strategy = strategy or fusion_context.strategy
                if selected_strategy not in self.strategies:
                    selected_strategy = self.default_strategy
                
                strategy_instance = self.strategies.get(selected_strategy)
                if not strategy_instance:
                    raise FusionError(f"Strategy {selected_strategy} not available")
                
                # Perform fusion
                result = await strategy_instance.fuse_modalities(modality_data, fusion_context)
                
                # Cache result if enabled
                if self.enable_caching and cache_key and result.success:
                    self._fusion_cache[cache_key] = result
                
                # Update performance statistics
                self._performance_stats[selected_strategy].append({
                    'processing_time': result.processing_time,
                    'quality': result.fusion_quality,
                    'confidence': result.overall_confidence,
                    'timestamp': datetime.now(timezone.utc)
                })
                
                # Emit fusion completed event
                await self.event_bus.emit(FusionCompleted(
                    session_id=fusion_context.session_id,
                    fusion_confidence=result.overall_confidence,
                    request_id=fusion_context.request_id
                ))
                
                # Update metrics
                if self.metrics:
                    self.metrics.increment("fusion_operations_total")
                    self.metrics.record("fusion_duration_seconds", result.processing_time)
                    self.metrics.increment("fusion_strategy_usage", 
                                         tags={'strategy': selected_strategy.value})
                    self.metrics.record("fusion_quality_score", result.fusion_quality)
                
                # Store for learning if available
                if self.memory_manager and result.success:
                    await self._store_fusion_for_learning(modality_data, result, fusion_context)
                
                self.logger.debug(
                    f"Fusion completed: strategy={selected_strategy.value}, "
                    f"quality={result.fusion_quality:.3f}, time={result.processing_time:.3f}s"
                )
                
                return result
                
        except Exception as e:
            # Handle fusion error
            processing_time = time.time() - start_time
            
            error_result = FusionResult(
                success=False,
                strategy_used=strategy or self.default_strategy,
                processing_time=processing_time,
                errors=[str(e)],
                request_id=fusion_context.request_id,
                session_id=fusion_context.session_id
            )
            
            # Emit error event
            await self.event_bus.emit(FusionError(
                session_id=fusion_context.session_id,
                error_message=str(e),
                request_id=fusion_context.request_id
            ))
            
            if self.metrics:
                self.metrics.increment("fusion_errors_total")
            
            self.logger.error(f"Fusion failed: {str(e)}")
            return error_result

    def _create_fusion_context(
        self, 
        context: Optional[Dict[str, Any]], 
        weights: Optional[Dict[str, float]]
    ) -> FusionContext:
        """Create fusion context from input parameters."""
        ctx = context or {}
        
        # Convert string weights to ModalityType weights
        modality_weights = {}
        if weights:
            for mod_str, weight in weights.items():
                try:
                    modality_type = ModalityType(mod_str.lower())
                    modality_weights[modality_type] = weight
                except ValueError:
                    self.logger.warning(f"Unknown modality type: {mod_str}")
        
        return FusionContext(
            session_id=ctx.get('session_id', 'default_session'),
            request_id=ctx.get('request_id', f"fusion_{int(time.time())}"),
            user_id=ctx.get('user_id'),
            strategy=FusionStrategy(ctx.get('strategy', self.default_strategy.value)),
            quality=FusionQuality(ctx.get('quality', 'balanced')),
            modality_weights=modality_weights,
            real_time_mode=ctx.get('real_time_mode', False),
            streaming_mode=ctx.get('streaming_mode', False),
            max_latency_ms=ctx.get('max_latency_ms', 1000.0),
            user_preferences=ctx.get('user_preferences', {}),
            environmental_context=ctx.get('environmental_context', {})
        )

    def _convert_modalities(self, modalities: Dict[str, Any]) -> Dict[ModalityType, ModalityData]:
        """Convert raw modality data to ModalityData format."""
        converted = {}
        
        for modality_str, data in modalities.items():
            try:
                # Determine modality type
                modality_type = self._infer_modality_type(modality_str, data)
                if not modality_type:
                    continue
                
                # Extract features and metadata
                features, embeddings, confidence, quality, metadata = self._extract_modality_info(data)
                
                converted[modality_type] = ModalityData(
                    modality=modality_type,
                    data=data,
                    features=features,
                    embeddings=embeddings,
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
        
        # Infer from data type
        if isinstance(data, str):
            return ModalityType.TEXT
        elif isinstance(data, np.ndarray):
            if data.ndim == 1:  # Audio signal
                return ModalityType.AUDIO
            elif data.ndim >= 2:  # Image/video
                return ModalityType.VISION
        elif isinstance(data, (TranscriptionResult, dict)) and 'text' in str(data):
            return ModalityType.TEXT
        elif isinstance(data, (EmotionResult, dict)) and 'emotion' in str(data):
            return ModalityType.EMOTION
        
        return None

    def _extract_modality_info(
        self, 
        data: Any
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float, float, Dict[str, Any]]:
        """Extract features, embeddings, confidence, quality, and metadata from
