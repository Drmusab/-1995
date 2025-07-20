
"""
Advanced Speech Emotion Detection Module
Author: Drmusab
Last Modified: 2025-05-26 13:43:26 UTC

This module provides comprehensive speech emotion detection capabilities integrated 
with the AI assistant's core architecture, including real-time emotion analysis,
multimodal fusion, memory integration, and adaptive learning systems.
"""

from pathlib import Path
from typing import Optional, Dict, Any, Union, List, Tuple, Callable, AsyncGenerator
import tempfile
from datetime import datetime, timezone
import asyncio
import json
import hashlib
import numpy as np
from dataclasses import dataclass, asdict, field
from enum import Enum
import threading
from contextlib import asynccontextmanager
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import librosa
import soundfile as sf
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib

# Core imports
from src.core.config.loader import ConfigLoader
from src.core.events.event_bus import EventBus
from src.core.events.event_types import (
    EmotionDetectionStarted,
    EmotionDetectionCompleted,
    EmotionDetectionError,
    EmotionalStateChanged,
    BiometricDataProcessed
)
from src.core.error_handling import ErrorHandler, handle_exceptions
from src.core.dependency_injection import Container
from src.core.health_check import HealthCheck

# Processing imports
from src.processing.speech.audio_utils import (
    EnhancedAudioProcessor,
    AudioProcessingError,
    AudioMetadata,
    ProcessingSettings,
    AudioFormat
)
from src.processing.multimodal.fusion_strategies import MultimodalFusionStrategy
from src.processing.natural_language.sentiment_analyzer import SentimentAnalyzer

# Memory and caching
from src.memory.cache_manager import CacheManager
from src.memory.memory_manager import MemoryManager
from src.memory.context_manager import ContextManager
from src.integrations.cache.cache_strategy import CacheStrategy

# Observability
from src.observability.monitoring.metrics import MetricsCollector
from src.observability.monitoring.tracing import TraceManager
from src.observability.logging.config import get_logger

# Learning and adaptation
from src.learning.feedback_processor import FeedbackProcessor
from src.learning.preference_learning import PreferenceLearner
from src.learning.continual_learning import ContinualLearner


class EmotionCategory(Enum):
    """Primary emotion categories based on psychological research."""
    HAPPINESS = "happiness"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    CONTEMPT = "contempt"
    NEUTRAL = "neutral"


class EmotionIntensity(Enum):
    """Emotion intensity levels."""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class AnalysisMode(Enum):
    """Emotion detection analysis modes."""
    REAL_TIME = "real_time"
    BATCH = "batch"
    STREAMING = "streaming"
    CONTEXTUAL = "contextual"


class FeatureSet(Enum):
    """Audio feature sets for emotion detection."""
    BASIC = "basic"          # Fundamental frequency, energy, spectral features
    ADVANCED = "advanced"    # MFCC, chromagram, spectral contrast
    COMPREHENSIVE = "comprehensive"  # All features + prosodic analysis
    DEEP_LEARNING = "deep_learning"  # CNN/RNN extracted features


@dataclass
class EmotionFeatures:
    """Container for extracted emotion-relevant audio features."""
    # Prosodic features
    fundamental_frequency: np.ndarray
    intensity: np.ndarray
    formants: np.ndarray
    jitter: float
    shimmer: float
    
    # Spectral features
    mfcc: np.ndarray
    spectral_centroid: np.ndarray
    spectral_contrast: np.ndarray
    spectral_rolloff: np.ndarray
    zero_crossing_rate: np.ndarray
    
    # Temporal features
    speech_rate: float
    pause_duration: float
    voice_activity_ratio: float
    
    # Deep features (if available)
    deep_embeddings: Optional[np.ndarray] = None
    
    # Metadata
    extraction_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    feature_set: FeatureSet = FeatureSet.COMPREHENSIVE
    audio_duration: float = 0.0
    sample_rate: int = 16000


@dataclass
class EmotionResult:
    """Comprehensive emotion detection result."""
    primary_emotion: EmotionCategory
    emotion_probabilities: Dict[str, float]
    confidence_score: float
    intensity: EmotionIntensity
    valence: float  # Positive/negative emotional value (-1 to 1)
    arousal: float  # Emotional activation level (0 to 1)
    dominance: float  # Control/power dimension (0 to 1)
    
    # Temporal analysis
    emotion_trajectory: Optional[List[Dict[str, Any]]] = None
    temporal_stability: float = 0.0
    
    # Context and metadata
    features: Optional[EmotionFeatures] = None
    processing_time: float = 0.0
    analysis_mode: AnalysisMode = AnalysisMode.REAL_TIME
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    
    # Multimodal integration
    multimodal_confidence: Optional[float] = None
    cross_modal_consistency: Optional[float] = None
    
    # Timestamp and tracking
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    quality_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class EmotionDetectionRequest:
    """Configuration for emotion detection requests."""
    audio_source: str = "buffer"
    analysis_mode: AnalysisMode = AnalysisMode.REAL_TIME
    feature_set: FeatureSet = FeatureSet.COMPREHENSIVE
    enable_temporal_analysis: bool = True
    enable_multimodal_fusion: bool = False
    context_aware: bool = True
    cache_results: bool = True
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class EmotionDetectionError(Exception):
    """Custom exception for emotion detection operations."""
    
    def __init__(self, message: str, error_code: Optional[str] = None):
        super().__init__(message)
        self.error_code = error_code
        self.timestamp = datetime.now(timezone.utc)


class EmotionFeatureExtractor:
    """Advanced feature extraction for emotion detection."""
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.frame_length = 2048
        self.hop_length = 512
        self.n_mels = 128
        self.n_mfcc = 13
        
        # Initialize feature scalers
        self.feature_scaler = StandardScaler()
        self.pca_reducer = PCA(n_components=50)
        
    def extract_prosodic_features(self, audio: np.ndarray) -> Dict[str, Any]:
        """Extract prosodic features related to emotion."""
        features = {}
        
        # Fundamental frequency (F0)
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            frame_length=self.frame_length
        )
        
        # Remove unvoiced frames
        f0_voiced = f0[voiced_flag]
        if len(f0_voiced) > 0:
            features['f0_mean'] = np.nanmean(f0_voiced)
            features['f0_std'] = np.nanstd(f0_voiced)
            features['f0_range'] = np.nanmax(f0_voiced) - np.nanmin(f0_voiced)
            features['f0_slope'] = np.polyfit(range(len(f0_voiced)), f0_voiced, 1)[0]
        else:
            features.update({
                'f0_mean': 0.0, 'f0_std': 0.0, 
                'f0_range': 0.0, 'f0_slope': 0.0
            })
        
        # Intensity/Energy
        rms_energy = librosa.feature.rms(y=audio, frame_length=self.frame_length)[0]
        features['energy_mean'] = np.mean(rms_energy)
        features['energy_std'] = np.std(rms_energy)
        features['energy_range'] = np.max(rms_energy) - np.min(rms_energy)
        
        # Speech rate estimation
        onset_frames = librosa.onset.onset_detect(y=audio, sr=self.sample_rate)
        features['speech_rate'] = len(onset_frames) / (len(audio) / self.sample_rate)
        
        # Jitter and shimmer (voice quality measures)
        features['jitter'] = self._calculate_jitter(f0_voiced)
        features['shimmer'] = self._calculate_shimmer(rms_energy)
        
        return features
    
    def extract_spectral_features(self, audio: np.ndarray) -> Dict[str, Any]:
        """Extract spectral features for emotion detection."""
        features = {}
        
        # MFCC features
        mfcc = librosa.feature.mfcc(
            y=audio, 
            sr=self.sample_rate, 
            n_mfcc=self.n_mfcc,
            n_fft=self.frame_length,
            hop_length=self.hop_length
        )
        
        # Statistical moments of MFCC
        for i in range(self.n_mfcc):
            features[f'mfcc_{i}_mean'] = np.mean(mfcc[i])
            features[f'mfcc_{i}_std'] = np.std(mfcc[i])
            features[f'mfcc_{i}_skew'] = self._calculate_skewness(mfcc[i])
            features[f'mfcc_{i}_kurtosis'] = self._calculate_kurtosis(mfcc[i])
        
        # Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)[0]
        features['spectral_centroid_mean'] = np.mean(spectral_centroids)
        features['spectral_centroid_std'] = np.std(spectral_centroids)
        
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate)[0]
        features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
        features['spectral_rolloff_std'] = np.std(spectral_rolloff)
        
        # Spectral contrast
        spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=self.sample_rate)
        features['spectral_contrast_mean'] = np.mean(spectral_contrast)
        features['spectral_contrast_std'] = np.std(spectral_contrast)
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        features['zcr_mean'] = np.mean(zcr)
        features['zcr_std'] = np.std(zcr)
        
        return features
    
    def extract_temporal_features(self, audio: np.ndarray) -> Dict[str, Any]:
        """Extract temporal features related to emotion."""
        features = {}
        
        # Voice activity detection
        rms_energy = librosa.feature.rms(y=audio)[0]
        energy_threshold = np.mean(rms_energy) * 0.1
        voice_frames = rms_energy > energy_threshold
        
        features['voice_activity_ratio'] = np.sum(voice_frames) / len(voice_frames)
        
        # Pause detection and analysis
        silence_frames = ~voice_frames
        silence_regions = self._find_silence_regions(silence_frames)
        
        if len(silence_regions) > 0:
            pause_durations = [(end - start) * self.hop_length / self.sample_rate 
                             for start, end in silence_regions]
            features['avg_pause_duration'] = np.mean(pause_durations)
            features['max_pause_duration'] = np.max(pause_durations)
            features['pause_frequency'] = len(pause_durations) / (len(audio) / self.sample_rate)
        else:
            features.update({
                'avg_pause_duration': 0.0,
                'max_pause_duration': 0.0,
                'pause_frequency': 0.0
            })
        
        return features
    
    def _calculate_jitter(self, f0: np.ndarray) -> float:
        """Calculate jitter (F0 variability)."""
        if len(f0) < 2:
            return 0.0
        
        period_diffs = np.abs(np.diff(1.0 / f0))
        avg_period = np.mean(1.0 / f0)
        return np.mean(period_diffs) / avg_period if avg_period > 0 else 0.0
    
    def _calculate_shimmer(self, rms_energy: np.ndarray) -> float:
        """Calculate shimmer (amplitude variability)."""
        if len(rms_energy) < 2:
            return 0.0
        
        amp_diffs = np.abs(np.diff(rms_energy))
        avg_amp = np.mean(rms_energy)
        return np.mean(amp_diffs) / avg_amp if avg_amp > 0 else 0.0
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data."""
        if len(data) == 0:
            return 0.0
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 3)
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of data."""
        if len(data) == 0:
            return 0.0
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 4) - 3.0
    
    def _find_silence_regions(self, silence_frames: np.ndarray) -> List[Tuple[int, int]]:
        """Find continuous silence regions in frame indices."""
        regions = []
        in_silence = False
        start_idx = 0
        
        for i, is_silence in enumerate(silence_frames):
            if is_silence and not in_silence:
                start_idx = i
                in_silence = True
            elif not is_silence and in_silence:
                regions.append((start_idx, i))
                in_silence = False
        
        # Handle case where audio ends in silence
        if in_silence:
            regions.append((start_idx, len(silence_frames)))
        
        return regions


class EmotionClassificationModel(nn.Module):
    """Deep learning model for emotion classification."""
    
    def __init__(self, input_dim: int = 256, num_emotions: int = 8, dropout: float = 0.3):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_emotions = num_emotions
        
        # Feature processing layers
        self.feature_layers = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Emotion classification head
        self.emotion_classifier = nn.Linear(128, num_emotions)
        
        # Dimensional emotion prediction (valence, arousal, dominance)
        self.valence_predictor = nn.Linear(128, 1)
        self.arousal_predictor = nn.Linear(128, 1)
        self.dominance_predictor = nn.Linear(128, 1)
        
        # Confidence estimation
        self.confidence_estimator = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Extract deep features
        features = self.feature_layers(x)
        
        # Emotion classification
        emotion_logits = self.emotion_classifier(features)
        emotion_probs = F.softmax(emotion_logits, dim=-1)
        
        # Dimensional predictions
        valence = torch.tanh(self.valence_predictor(features))
        arousal = torch.sigmoid(self.arousal_predictor(features))
        dominance = torch.sigmoid(self.dominance_predictor(features))
        
        # Confidence estimation
        confidence = self.confidence_estimator(features)
        
        return {
            'emotion_logits': emotion_logits,
            'emotion_probs': emotion_probs,
            'valence': valence,
            'arousal': arousal,
            'dominance': dominance,
            'confidence': confidence,
            'features': features
        }


class EnhancedEmotionDetector:
    """
    Advanced emotion detection system with comprehensive AI assistant integration.
    
    Features:
    - Multi-modal emotion detection from speech audio
    - Real-time and batch processing capabilities
    - Advanced feature extraction with prosodic, spectral, and temporal analysis
    - Deep learning models for emotion classification
    - Dimensional emotion analysis (valence, arousal, dominance)
    - Context-aware emotion tracking with memory integration
    - Adaptive learning and personalization
    - Event-driven architecture with comprehensive monitoring
    - Caching and performance optimization
    - Integration with multimodal fusion strategies
    """
    
    EMOTION_LABELS = [e.value for e in EmotionCategory]
    
    def __init__(self, container: Container):
        """
        Initialize the enhanced emotion detection system.
        
        Args:
            container: Dependency injection container
        """
        self.container = container
        self.logger = get_logger(__name__)
        self.config = container.get(ConfigLoader)
        self.event_bus = container.get(EventBus)
        self.error_handler = container.get(ErrorHandler)
        self.health_check = container.get(HealthCheck)
        
        # Initialize components
        self._setup_device()
        self._setup_audio_config()
        self._setup_models()
        self._setup_feature_extraction()
        self._setup_integrations()
        self._setup_monitoring()
        self._setup_caching()
        self._setup_learning()
        
        # Register health check
        self.health_check.register_component(
            "emotion_detection",
            self._health_check_callback
        )
        
        self.logger.info(
            f"EnhancedEmotionDetector initialized "
            f"(Device: {self.device}, Models: {len(self.models)})"
        )
    
    def _setup_device(self) -> None:
        """Setup compute device and memory management."""
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and 
            self.config.get("emotion_detection.device.use_gpu", True) else "cpu"
        )
        
        if self.device.type == "cuda":
            torch.cuda.set_per_process_memory_fraction(
                self.config.get("emotion_detection.device.gpu_memory_fraction", 0.5)
            )
    
    def _setup_audio_config(self) -> None:
        """Configure audio processing settings."""
        self.sample_rate = self.config.get("emotion_detection.audio.sample_rate", 16000)
        self.frame_duration = self.config.get("emotion_detection.audio.frame_duration", 0.025)
        self.hop_duration = self.config.get("emotion_detection.audio.hop_duration", 0.010)
        
        # Initialize audio processor
        self.audio_processor = EnhancedAudioProcessor(
            sample_rate=self.sample_rate,
            container=self.container
        )
        
        # Temporal analysis settings
        self.temporal_window_size = self.config.get("emotion_detection.temporal.window_size", 3.0)
        self.temporal_overlap = self.config.get("emotion_detection.temporal.overlap", 0.5)
    
    def _setup_models(self) -> None:
        """Setup emotion detection models."""
        self.models = {}
        self.model_loading_lock = threading.Lock()
        
        # Model paths
        self.model_dir = Path(self.config.get(
            "emotion_detection.models.model_dir", 
            "data/models/emotion_detection"
        ))
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Load emotion classification model
        self._load_classification_model()
        
        # Model performance tracking
        self.model_performance = {}
    
    def _load_classification_model(self) -> None:
        """Load the emotion classification model."""
        try:
            # Initialize model
            input_dim = self.config.get("emotion_detection.models.input_dim", 256)
            num_emotions = len(self.EMOTION_LABELS)
            
            self.classification_model = EmotionClassificationModel(
                input_dim=input_dim,
                num_emotions=num_emotions,
                dropout=self.config.get("emotion_detection.models.dropout", 0.3)
            ).to(self.device)
            
            # Load pretrained weights if available
            model_path = self.model_dir / "emotion_classifier.pth"
            if model_path.exists():
                checkpoint = torch.load(model_path, map_location=self.device)
                self.classification_model.load_state_dict(checkpoint['model_state_dict'])
                self.logger.info("Loaded pretrained emotion classification model")
                
                # Load associated metadata
                if 'performance_metrics' in checkpoint:
                    self.model_performance = checkpoint['performance_metrics']
            
            self.models['classification'] = self.classification_model
            
        except Exception as e:
            self.logger.error(f"Failed to load classification model: {str(e)}")
            raise EmotionDetectionError(f"Model loading failed: {str(e)}")
    
    def _setup_feature_extraction(self) -> None:
        """Setup feature extraction components."""
        self.feature_extractor = EmotionFeatureExtractor(sample_rate=self.sample_rate)
        
        # Feature processing settings
        self.feature_cache = {}
        self.feature_cache_lock = threading.Lock()
        
        # Feature normalization
        self.feature_normalizer = StandardScaler()
        self.feature_selector = PCA(n_components=0.95)  # Keep 95% of variance
        
        # Load pretrained feature processors if available
        self._load_feature_processors()
    
    def _load_feature_processors(self) -> None:
        """Load pretrained feature processing components."""
        try:
            scaler_path = self.model_dir / "feature_scaler.joblib"
            if scaler_path.exists():
                self.feature_normalizer = joblib.load(scaler_path)
                self.logger.info("Loaded pretrained feature scaler")
            
            pca_path = self.model_dir / "feature_selector.joblib"
            if pca_path.exists():
                self.feature_selector = joblib.load(pca_path)
                self.logger.info("Loaded pretrained feature selector")
                
        except Exception as e:
            self.logger.warning(f"Failed to load feature processors: {str(e)}")
    
    def _setup_integrations(self) -> None:
        """Setup integrations with other AI assistant components."""
        # Memory integration
        self.memory_manager = self.container.get(MemoryManager)
        self.context_manager = self.container.get(ContextManager)
        
        # Natural language processing
        self.sentiment_analyzer = self.container.get_optional(SentimentAnalyzer)
        
        # Multimodal fusion
        self.fusion_strategy = self.container.get_optional(MultimodalFusionStrategy)
        
        # Learning components
        self.feedback_processor = self.container.get_optional(FeedbackProcessor)
        self.preference_learner = self.container.get_optional(PreferenceLearner)
        self.continual_learner = self.container.get_optional(ContinualLearner)
    
    def _setup_monitoring(self) -> None:
        """Setup monitoring and metrics collection."""
        self.metrics = self.container.get(MetricsCollector)
        self.tracer = self.container.get(TraceManager)
        
        # Register metrics
        self.metrics.register_counter("emotion_detection_requests_total")
        self.metrics.register_histogram("emotion_detection_processing_duration_seconds")
        self.metrics.register_gauge("emotion_detection_confidence_score")
        self.metrics.register_counter("emotion_detection_errors_total")
        self.metrics.register_histogram("emotion_detection_feature_extraction_duration")
    
    def _setup_caching(self) -> None:
        """Setup caching for emotion detection results."""
        self.cache_manager = self.container.get(CacheManager)
        self.cache_strategy = self.container.get(CacheStrategy)
        
        self.cache_enabled = self.config.get("emotion_detection.caching.enabled", True)
        self.cache_ttl = self.config.get("emotion_detection.caching.ttl_seconds", 1800)
    
    def _setup_learning(self) -> None:
        """Setup learning and adaptation capabilities."""
        self.learning_enabled = self.config.get("emotion_detection.learning.enabled", True)
        self.adaptation_enabled = self.config.get("emotion_detection.learning.adaptation", True)
        
        # User-specific emotion profiles
        self.user_emotion_profiles = {}
        self.emotion_history = {}
    
    @handle_exceptions
    async def detect_emotion(
        self,
        audio: Union[np.ndarray, str, Path],
        request: Optional[EmotionDetectionRequest] = None
    ) -> EmotionResult:
        """
        Comprehensive emotion detection from speech audio.
        
        Args:
            audio: Audio data or path to audio file
            request: Emotion detection request configuration
            
        Returns:
            Comprehensive emotion detection result
        """
        start_time = datetime.now(timezone.utc)
        
        # Default request if not provided
        if request is None:
            request = EmotionDetectionRequest()
        
        # Generate session ID if not provided
        if not request.session_id:
            request.session_id = self._generate_session_id()
        
        # Emit detection started event
        await self.event_bus.emit(EmotionDetectionStarted(
            session_id=request.session_id,
            audio_source=request.audio_source,
            analysis_mode=request.analysis_mode.value,
            feature_set=request.feature_set.value
        ))
        
        try:
            with self.tracer.trace("emotion_detection") as span:
                span.set_attributes({
                    "session_id": request.session_id,
                    "analysis_mode": request.analysis_mode.value,
                    "feature_set": request.feature_set.value
                })
                
                # Check cache first
                cache_key = None
                if request.cache_results and self.cache_enabled:
                    cache_key = self._generate_cache_key(audio, request)
                    cached_result = await self._get_cached_result(cache_key)
                    if cached_result:
                        self.logger.info(f"Returning cached emotion result for session {request.session_id}")
                        return cached_result
                
                # Process audio input
                processed_audio = await self._process_audio_input(audio, request)
                
                # Extract features
                features = await self._extract_emotion_features(processed_audio, request)
                
                # Perform emotion detection
                emotion_result = await self._classify_emotion(features, request)
                
                # Enhance with temporal analysis if enabled
                if request.enable_temporal_analysis:
                    emotion_result = await self._enhance_with_temporal_analysis(
                        emotion_result, processed_audio, request
                    )
                
                # Enhance with multimodal fusion if enabled
                if request.enable_multimodal_fusion and self.fusion_strategy:
                    emotion_result = await self._enhance_with_multimodal_fusion(
                        emotion_result, request
                    )
                
                # Context-aware enhancement
                if request.context_aware:
                    emotion_result = await self._enhance_with_context(emotion_result, request)
                
                # Calculate processing time and quality metrics
                processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
                emotion_result.processing_time = processing_time
                emotion_result.quality_metrics = self._calculate_quality_metrics(
                    emotion_result, features, processing_time
                )
                
                # Cache result if enabled
                if cache_key:
                    await self._cache_result(cache_key, emotion_result)
                
                # Store in memory and context
                if request.context_aware:
                    await self._store_emotion_context(emotion_result, request)
                
                # Update metrics
                self._update_metrics(emotion_result, processing_time)
                
                # Learn from result if enabled
                if self.learning_enabled:
                    await self._learn_from_detection(emotion_result, request)
                
                # Emit completion event
                await self.event_bus.emit(EmotionDetectionCompleted(
                    session_id=request.session_id,
                    primary_emotion=emotion_result.primary_emotion.value,
                    confidence=emotion_result.confidence_score,
                    processing_time=processing_time
                ))
                
                # Emit emotional state change if significant
                await self._check_and_emit_state_change(emotion_result, request)
                
                self.logger.info(
                    f"Emotion detection completed for session {request.session_id} "
                    f"(Emotion: {emotion_result.primary_emotion.value}, "
                    f"Confidence: {emotion_result.confidence_score:.2f}, "
                    f"Time: {processing_time:.2f}s)"
                )
                
                return emotion_result
                
        except Exception as e:
            # Emit error event
            await self.event_bus.emit(EmotionDetectionError(
                session_id=request.session_id,
                error_type=type(e).__name__,
                error_message=str(e)
            ))
            
            self.metrics.increment("emotion_detection_errors_total")
            self.logger.error(f"Emotion detection failed for session {request.session_id}: {str(e)}")
            raise EmotionDetectionError(f"Emotion detection failed: {str(e)}") from e
    
    async def _process_audio_input(
        self, 
        audio: Union[np.ndarray, str, Path], 
        request: EmotionDetectionRequest
    ) -> np.ndarray:
        """Process and prepare audio input for emotion detection."""
        if isinstance(audio, (str, Path)):
            audio_data, sr = await self.audio_processor.load_audio(
                audio, target_sr=self.sample_rate, normalize=True
            )
        else:
            audio_data = audio.copy()
        
        # Apply preprocessing optimized for emotion detection
        audio_data = await self._preprocess_for_emotion_detection(audio_data)
        
        return audio_data
    
    async def _preprocess_for_emotion_detection(self, audio: np.ndarray) -> np.ndarray:
        """Apply preprocessing specifically optimized for emotion detection."""
        # Normalize audio
        audio = self.audio_processor.normalize_audio(audio, target_db=-20.0)
        
        # Apply gentle noise reduction to preserve emotional cues
        audio = self.audio_processor.advanced_noise_reduction(
            audio, self.sample_rate, method="spectral_gating"
        )
        
        # Trim silence but preserve emotional pauses
        audio = self.audio_processor.trim_silence(audio, threshold_db=-40.0)
        
        return audio
    
    async def _extract_emotion_features(
        self, 
        audio: np.ndarray, 
        request: EmotionDetectionRequest
    ) -> EmotionFeatures:
        """Extract comprehensive emotion-relevant features from audio."""
        start_time = datetime.now(timezone.utc)
        
        try:
            # Initialize feature containers
            prosodic_features = {}
            spectral_features = {}
            temporal_features = {}
            
            # Extract different feature sets based on request
            if request.feature_set in [FeatureSet.BASIC, FeatureSet.ADVANCED, FeatureSet.COMPREHENSIVE]:
                prosodic_features = self.feature_extractor.extract_prosodic_features(audio)
                
            if request.feature_set in [FeatureSet.ADVANCED, FeatureSet.COMPREHENSIVE]:
                spectral_features = self.feature_extractor.extract_spectral_features(audio)
                
            if request.feature_set == FeatureSet.COMPREHENSIVE:
                temporal_features = self.feature_extractor.extract_temporal_features(audio)
            
            # Extract fundamental frequency and formants
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio, fmin=50, fmax=500, frame_length=2048
            )
            f0_voiced = f0[voiced_flag] if np.any(voiced_flag) else np.array([0])
            
            # Extract formants (simplified - would use more sophisticated method in practice)
            formants = self._extract_formants(audio)
            
            # Calculate jitter and shimmer
            jitter = prosodic_features.get('jitter', 0.0)
            shimmer = prosodic_features.get('shimmer', 0.0)
            
            # Extract spectral features
            mfcc = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=13)
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)[0]
            spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=self.sample_rate)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate)[0]
            zcr = librosa.feature.zero_crossing_rate(audio)[0]
            
            # Extract intensity
            rms_energy = librosa.feature.rms(y=audio)[0]
            
            # Calculate temporal features
            speech_rate = temporal_features.get('speech_rate', 0.0)
            pause_duration = temporal_features.get('avg_pause_duration', 0.0)
            voice_activity_ratio = temporal_features.get('voice_activity_ratio', 1.0)
            
            # Deep learning features (if model available)
            deep_embeddings = None
            if request.feature_set == FeatureSet.DEEP_LEARNING:
                deep_embeddings = await self._extract_deep_features(audio)
            
            # Create features object
            features = EmotionFeatures(
                fundamental_frequency=f0_voiced,
                intensity=rms_energy,
                formants=formants,
                jitter=jitter,
                shimmer=shimmer,
                mfcc=mfcc,
                spectral_centroid=spectral_centroid,
                spectral_contrast=spectral_contrast,
                spectral_rolloff=spectral_rolloff,
                zero_crossing_rate=zcr,
                speech_rate=speech_rate,
                pause_duration=pause_duration,
                voice_activity_ratio=voice_activity_ratio,
                deep_embeddings=deep_embeddings,
                feature_set=request.feature_set,
                audio_duration=len(audio) / self.sample_rate,
                sample_rate=self.sample_rate
            )
            
            # Record feature extraction metrics
            extraction_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            self.metrics.record("emotion_detection_feature_extraction_duration", extraction_time)
            
            return features
            
        except Exception as e:
            raise EmotionDetectionError(f"Feature extraction failed: {str(e)}") from e
    
    def _extract_formants(self, audio: np.ndarray) -> np.ndarray:
        """Extract formant frequencies (simplified implementation)."""
        try:
            # Compute LPC coefficients
            from scipy.signal import lfilter
            
            # Pre-emphasis
            emphasized = lfilter([1, -0.95], [1], audio)
            
            # Windowing and autocorrelation-based formant estimation
            # This is a simplified approach; production systems would use more sophisticated methods
            frame_length = int(0.025 * self.sample_rate)
            hop_length = int(0.010 * self.sample_rate)
            
            formants = []
            for i in range(0, len(emphasized) - frame_length, hop_length):
                frame = emphasized[i:i + frame_length]
                if len(frame) == frame_length:
                    # Basic formant estimation
                    fft = np.fft.fft(frame)
                    magnitude = np.abs(fft[:len(fft)//2])
                    freqs = np.fft.fftfreq(len(fft), 1/self.sample_rate)[:len(fft)//2]
                    
                    # Find peaks (simplified formant detection)
                    peaks = self._find_spectral_peaks(magnitude, freqs)
                    formants.append(peaks[:3] if len(peaks) >= 3 else [0, 0, 0])
            
            return np.array(formants) if formants else np.array([[0, 0, 0]])
            
        except Exception as e:
            self.logger.warning(f"Formant extraction failed: {str(e)}")
            return np.array([[0, 0, 0]])
    
    def _find_spectral_peaks(self, magnitude: np.ndarray, freqs: np.ndarray) -> List[float]:
        """Find spectral peaks for formant estimation."""
        from scipy.signal import find_peaks
        
        # Find peaks in the magnitude spectrum
        peaks, _ = find_peaks(magnitude, height=np.max(magnitude) * 0.1, distance=20)
        
        # Return frequencies of peaks
        return freqs[peaks].tolist()
    
    async def _extract_deep_features(self, audio: np.ndarray) -> Optional[np.ndarray]:
        """Extract deep learning features (placeholder for future implementation)."""
        # This would integrate with pre-trained deep learning models for feature extraction
        # For now, return None
        return None
    
    async def _classify_emotion(
        self, 
        features: EmotionFeatures, 
        request: EmotionDetectionRequest
    ) -> EmotionResult:
        """Classify emotion from extracted features."""
        try:
            # Prepare feature vector
            feature_vector = self._prepare_feature_vector(features)
            
            # Convert to tensor
            feature_tensor = torch.tensor(feature_vector, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            # Run inference
            with torch.no_grad():
                self.classification_model.eval()
                outputs = self.classification_model(feature_tensor)
            
            # Extract predictions
            emotion_probs = outputs['emotion_probs'].cpu().numpy()[0]
            valence = float(outputs['valence'].cpu().numpy()[0, 0])
            arousal = float(outputs['arousal'].cpu().numpy()[0, 0])
            dominance = float(outputs['dominance'].cpu().numpy()[0, 0])
            confidence = float(outputs['confidence'].cpu().numpy()[0, 0])
            
            # Determine primary emotion
            primary_emotion_idx = np.argmax(emotion_probs)
            primary_emotion = EmotionCategory(self.EMOTION_LABELS[primary_emotion_idx])
            
            # Create emotion probabilities dictionary
            emotion_probabilities = {
                label: float(prob) for label, prob in zip(self.EMOTION_LABELS, emotion_probs)
            }
            
            # Determine intensity based on confidence and arousal
            intensity = self._determine_intensity(confidence, arousal)
            
            # Create result
            result = EmotionResult(
                primary_emotion=primary_emotion,
                emotion_probabilities=emotion_probabilities,
                confidence_score=confidence,
                intensity=intensity,
                valence=valence,
                arousal=arousal,
                dominance=dominance,
                features=features,
                analysis_mode=request.analysis_mode,
                session_id=request.session_id,
                user_id=request.user_id
            )
            
            return result
            
        except Exception as e:
            raise EmotionDetectionError(f"Emotion classification failed: {str(e)}") from e
    
    def _prepare_feature_vector(self, features: EmotionFeatures) -> np.ndarray:
        """Prepare feature vector for model input."""
        # Combine all features into a single vector
        feature_list = []
        
        # Prosodic features
        if len(features.fundamental_frequency) > 0:
            feature_list.extend([
                np.mean(features.fundamental_frequency),
                np.std(features.fundamental_frequency),
                np.min(features.fundamental_frequency),
                np.max(features.fundamental_frequency)
            ])
        else:
            feature_list.extend([0, 0, 0, 0])
        
        feature_list.extend([
            np.mean(features.intensity),
            np.std(features.intensity),
            features.jitter,
            features.shimmer
        ])
        
        # Spectral features
        feature_list.extend([
            np.mean(features.mfcc, axis=1).tolist(),
            np.mean(features.spectral_centroid),
            np.std(features.spectral_centroid),
            np.mean(features.spectral_contrast),
            np.mean(features.spectral_rolloff),
            np.mean(features.zero_crossing_rate)
        ])
        
        # Flatten nested lists
        flattened_features = []
        for item in feature_list:
            if isinstance(item, list):
                flattened_features.extend(item)
            else:
                flattened_features.append(item)
        
        # Temporal features
        flattened_features.extend([
            features.speech_rate,
            features.pause_duration,
            features.voice_activity_ratio
        ])
        
        # Convert to numpy array and handle NaN values
        feature_vector = np.array(flattened_features, dtype=np.float32)
        feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Normalize features if normalizer is trained
        if hasattr(self.feature_normalizer, 'mean_'):
            # Ensure feature vector has correct dimensions
            if len(feature_vector) == len(self.feature_normalizer.mean_):
                feature_vector = self.feature_normalizer.transform(feature_vector.reshape(1, -1))[0]
        
        # Pad or truncate to expected model input size
        expected_size = self.config.get("emotion_detection.models.input_dim", 256)
        if len(feature_vector) < expected_size:
            feature_vector = np.pad(feature_vector, (0, expected_size - len(feature_vector)))
        elif len(feature_vector) > expected_size:
            feature_vector = feature_vector[:expected_size]
        
        return feature_vector
    
    def _determine_intensity(self, confidence: float, arousal: float) -> EmotionIntensity:
        """Determine emotion intensity from confidence and arousal levels."""
        # Combine confidence and arousal to determine intensity
        intensity_score = (confidence + arousal) / 2.0
        
        if intensity_score < 0.2:
            return EmotionIntensity.VERY_LOW
        elif intensity_score < 0.4:
            return EmotionIntensity.LOW
        elif intensity_score < 0.6:
            return EmotionIntensity.MEDIUM
        elif intensity_score < 0.8:
            return EmotionIntensity.HIGH
        else:
            return EmotionIntensity.VERY_HIGH
    
    async def _enhance_with_temporal_analysis(
        self,
        emotion_result: EmotionResult,
        audio: np.ndarray,
        request: EmotionDetectionRequest
    ) -> EmotionResult:
        """Enhance emotion result with temporal analysis."""
        try:
            # Analyze emotion trajectory over time
            window_size = int(self.temporal_window_size * self.sample_rate)
            hop_size = int(window_size * (1 - self.temporal_overlap))
            
            emotion_trajectory = []
            
            for i in range(0, len(audio) - window_size, hop_size):
                window_audio = audio[i:i + window_size]
                
                # Extract features for this window
                window_features = await self._extract_emotion_features(
                    window_audio, 
                    EmotionDetectionRequest(feature_set=FeatureSet.BASIC)
                )
                
                # Classify emotion for this window
                window_result = await self._classify_emotion(window_features, request)
                
                emotion_trajectory.append({
                    'timestamp': i / self.sample_rate,
                    'emotion': window_result.primary_emotion.value,
                    'confidence': window_result.confidence_score,
                    'valence': window_result.valence,
                    'arousal': window_result.arousal
                })
            
            # Calculate temporal stability
            if len(emotion_trajectory) > 1:
                emotions = [frame['emotion'] for frame in emotion_trajectory]
                temporal_stability = self._calculate_temporal_stability(emotions)
            else:
                temporal_stability = 1.0
            
            emotion_result.emotion_trajectory = emotion_trajectory
            emotion_result.temporal_stability = temporal_stability
            
            return emotion_result
            
        except Exception as e:
            self.logger.warning(f"Temporal analysis failed: {str(e)}")
            return emotion_result
    
    def _calculate_temporal_stability(self, emotions: List[str]) -> float:
        """Calculate temporal stability of emotions."""
        if len(emotions) <= 1:
            return 1.0
        
        # Calculate the proportion of consecutive frames with the same emotion
        stable_transitions = sum(1 for i in range(1, len(emotions)) 
                               if emotions[i] == emotions[i-1])
        
        return stable_transitions / (len(emotions) - 1)
    
    async def _enhance_with_multimodal_fusion(
        self,
        emotion_result: EmotionResult,
        request: EmotionDetectionRequest
    ) -> EmotionResult:
        """Enhance emotion result with multimodal fusion."""
        try:
            if not self.fusion_strategy:
                return emotion_result
            
            # This would integrate with text-based sentiment analysis if available
            if self.sentiment_analyzer:
                # For now, just set multimodal confidence
                emotion_result.multimodal_confidence = emotion_result.confidence_score
                emotion_result.cross_modal_consistency = 1.0
            
            return emotion_result
            
        except Exception as e:
            self.logger.warning(f"Multimodal fusion failed: {str(e)}")
            return emotion_result
    
    async def _enhance_with_context(
        self,
        emotion_result: EmotionResult,
        request: EmotionDetectionRequest
    ) -> EmotionResult:
        """Enhance emotion result with contextual information."""
        try:
            if not request.session_id:
                return emotion_result
            
            # Retrieve emotion history for context
            emotion_history = await self._get_emotion_history(request.session_id, request.user_id)
            
            if emotion_history:
                # Adjust confidence based on emotional consistency
                recent_emotions = [entry['emotion'] for entry in emotion_history[-5:]]
                if recent_emotions:
                    consistency_bonus = self._calculate_consistency_bonus(
                        emotion_result.primary_emotion.value, 
                        recent_emotions
                    )
                    emotion_result.confidence_score = min(1.0, 
                        emotion_result.confidence_score + consistency_bonus
                    )
            
            return emotion_result
            
        except Exception as e:
            self.logger.warning(f"Context enhancement failed: {str(e)}")
            return emotion_result
    
    def _calculate_consistency_bonus(self, current_emotion: str, recent_emotions: List[str]) -> float:
        """Calculate confidence bonus based on emotional consistency."""
        if not recent_emotions:
            return 0.0
        
        # Count how many recent emotions match current emotion
        matches = sum(1 for emotion in recent_emotions if emotion == current_emotion)
        consistency_ratio = matches / len(recent_emotions)
        
        # Provide small bonus for consistency (max 0.1)
        return consistency_ratio * 0.1
    
    def _calculate_quality_metrics(
        self,
        emotion_result: EmotionResult,
        features: EmotionFeatures,
        processing_time: float
    ) -> Dict[str, float]:
        """Calculate quality metrics for emotion detection."""
        metrics = {
            'processing_time': processing_time,
            'confidence_score': emotion_result.confidence_score,
            'feature_quality': self._assess_feature_quality(features),
            'audio_duration': features.audio_duration,
            'voice_activity_ratio': features.voice_activity_ratio
        }
        
        # Temporal stability if available
        if emotion_result.temporal_stability is not None:
            metrics['temporal_stability'] = emotion_result.temporal_stability
        
        return metrics
    
    def _assess_feature_quality(self, features: EmotionFeatures) -> float:
        """Assess the quality of extracted features."""
        quality_score = 1.0
        
        # Penalize for very short audio
        if features.audio_duration < 1.0:
            quality_score *= 0.5
        
        # Penalize for low voice activity
        if features.voice_activity_ratio < 0.3:
            quality_score *= 0.7
        
        # Penalize for excessive jitter/shimmer (poor audio quality)
        if features.jitter > 0.05 or features.shimmer > 0.1:
            quality_score *= 0.8
        
        return max(0.0, min(1.0, quality_score))
    
    async def _store_emotion_context(
        self,
        emotion_result: EmotionResult,
        request: EmotionDetectionRequest
    ) -> None:
        """Store emotion result in context memory."""
        try:
            context_data = {
                'emotion': emotion_result.primary_emotion.value,
                'confidence': emotion_result.confidence_score,
                'valence': emotion_result.valence,
                'arousal': emotion_result.arousal,
                'dominance': emotion_result.dominance,
                'intensity': emotion_result.intensity.value,
                'timestamp': emotion_result.timestamp.isoformat()
            }
            
            # Store in context manager
            if request.session_id:
                await self.context_manager.update_context(
                    request.session_id,
                    "emotion_detection",
                    context_data
                )
            
            # Store in user emotion history
            if request.user_id:
                await self._update_user_emotion_history(request.user_id, context_data)
            
        except Exception as e:
            self.logger.warning(f"Context storage failed: {str(e)}")
    
    async def _update_user_emotion_history(self, user_id: str, emotion_data: Dict[str, Any]) -> None:
        """Update user-specific emotion history."""
        try:
            if user_id not in self.emotion_history:
                self.emotion_history[user_id] = []
            
            self.emotion_history[user_id].append(emotion_data)
            
            # Keep only recent history (last 100 entries)
            if len(self.emotion_history[user_id]) > 100:
                self.emotion_history[user_id] = self.emotion_history[user_id][-100:]
            
            # Store in persistent memory
            await self.memory_manager.store(
                f"emotion_history_{user_id}",
                self.emotion_history[user_id],
                memory_type="episodic"
            )
            
        except Exception as e:
            self.logger.warning(f"Emotion history update failed: {str(e)}")
    
    async def _get_emotion_history(
        self, 
        session_id: str, 
        user_id: Optional[str]
    ) -> Optional[List[Dict[str, Any]]]:
        """Retrieve emotion history for context."""
        try:
            if user_id and user_id in self.emotion_history:
                return self.emotion_history[user_id]
            
            # Try to retrieve from persistent memory
            if user_id:
                history = await self.memory_manager.retrieve(
                    f"emotion_history_{user_id}",
                    memory_type="episodic"
                )
                if history:
                    self.emotion_history[user_id] = history
                    return history
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Emotion history retrieval failed: {str(e)}")
            return None
    
    def _update_metrics(self, emotion_result: EmotionResult, processing_time: float) -> None:
        """Update monitoring metrics."""
        self.metrics.increment("emotion_detection_requests_total")
        self.metrics.record("emotion_detection_processing_duration_seconds", processing_time)
        self.metrics.set("emotion_detection_confidence_score", emotion_result.confidence_score)
    
    async def _learn_from_detection(
        self,
        emotion_result: EmotionResult,
        request: EmotionDetectionRequest
    ) -> None:
        """Learn from emotion detection results."""
        if not self.learning_enabled or not self.feedback_processor:
            return
        
        try:
            learning_data = {
                'emotion': emotion_result.primary_emotion.value,
                'confidence': emotion_result.confidence_score,
                'valence': emotion_result.valence,
                'arousal': emotion_result.arousal,
                'processing_time': emotion_result.processing_time,
                'feature_quality': emotion_result.quality_metrics.get('feature_quality', 0.0),
                'temporal_stability': emotion_result.temporal_stability or 0.0
            }
            
            await self.feedback_processor.process_feedback(
                "emotion_detection",
                learning_data,
                request.user_id
            )
            
        except Exception as e:
            self.logger.warning(f"Learning from detection failed: {str(e)}")
    
    async def _check_and_emit_state_change(
        self,
        emotion_result: EmotionResult,
        request: EmotionDetectionRequest
    ) -> None:
        """Check for significant emotional state changes and emit events."""
        try:
            if not request.user_id:
                return
            
            # Get previous emotion state
            emotion_history = await self._get_emotion_history(request.session_id, request.user_id)
            
            if not emotion_history or len(emotion_history) == 0:
                return
            
            previous_emotion = emotion_history[-1]['emotion']
            previous_valence = emotion_history[-1]['valence']
            
            # Check for significant changes
            emotion_changed = emotion_result.primary_emotion.value != previous_emotion
            valence_change = abs(emotion_result.valence - previous_valence) > 0.3
            
            if emotion_changed or valence_change:
                await self.event_bus.emit(EmotionalStateChanged(
                    user_id=request.user_id,
                    session_id=request.session_id,
                    previous_emotion=previous_emotion,
                    current_emotion=emotion_result.primary_emotion.value,
                    valence_change=emotion_result.valence - previous_valence,
                    confidence=emotion_result.confidence_score
                ))
            
        except Exception as e:
            self.logger.warning(f"State change detection failed: {str(e)}")
    
    def _generate_session_id(self) -> str:
        """Generate unique session identifier."""
        timestamp = datetime.now(timezone.utc).isoformat()
        return hashlib.md5(f"{timestamp}_{id(self)}".encode()).hexdigest()[:16]
    
    def _generate_cache_key(
        self,
        audio: Union[np.ndarray, str, Path],
        request: EmotionDetectionRequest
    ) -> str:
        """Generate cache key for emotion detection request."""
        # Create hash of audio content
        if isinstance(audio, (str, Path)):
            audio_hash = hashlib.md5(Path(audio).read_bytes()).hexdigest()
        else:
            audio_hash = hashlib.md5(audio.tobytes()).hexdigest()
        
        # Create hash of request parameters
        request_dict = asdict(request)
        request_dict.pop("session_id", None)
        request_dict.pop("metadata", None)
        request_str = json.dumps(request_dict, sort_keys=True, default=str)
        request_hash = hashlib.md5(request_str.encode()).hexdigest()
        
        return f"emotion:{audio_hash[:16]}:{request_hash[:16]}"
    
    async def _get_cached_result(self, cache_key: str) -> Optional[EmotionResult]:
        """Retrieve cached emotion detection result."""
        try:
            cached_data = await self.cache_manager.get(cache_key)
            if cached_data:
                # Reconstruct EmotionResult
                cached_data['primary_emotion'] = EmotionCategory(cached_data['primary_emotion'])
                cached_data['intensity'] = EmotionIntensity(cached_data['intensity'])
                cached_data['analysis_mode'] = AnalysisMode(cached_data['analysis_mode'])
                cached_data['timestamp'] = datetime.fromisoformat(cached_data['timestamp'])
                
                return EmotionResult(**cached_data)
        except Exception as e:
            self.logger.warning(f"Cache retrieval failed: {str(e)}")
        return None
    
    async def _cache_result(self, cache_key: str, result: EmotionResult) -> None:
        """Cache emotion detection result."""
        try:
            # Convert for serialization
            cache_data = asdict(result)
            cache_data['primary_emotion'] = result.primary_emotion.value
            cache_data['intensity'] = result.intensity.value
            cache_data['analysis_mode'] = result.analysis_mode.value
            cache_data['timestamp'] = result.timestamp.isoformat()
            cache_data.pop('features', None)  # Don't cache large feature objects
            
            await self.cache_manager.set(cache_key, cache_data, ttl=self.cache_ttl)
            
        except Exception as e:
            self.logger.warning(f"Cache storage failed: {str(e)}")
    
    async def _health_check_callback(self) -> Dict[str, Any]:
        """Health check callback for emotion detection system."""
        try:
            # Test basic functionality
            test_audio = np.random.randn(self.sample_rate).astype(np.float32) * 0.1
            start_time = datetime.now(timezone.utc)
            
            test_request = EmotionDetectionRequest(
                analysis_mode=AnalysisMode.REAL_TIME,
                feature_set=FeatureSet.BASIC,
                cache_results=False,
                context_aware=False
            )
            
            result = await self.detect_emotion(test_audio, test_request)
            response_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            return {
                "status": "healthy",
                "models_loaded": len(self.models),
                "device": str(self.device),
                "response_time_ms": response_time * 1000,
                "cache_enabled": self.cache_enabled,
                "learning_enabled": self.learning_enabled,
                "feature_extractor_ready": self.feature_extractor is not None
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "models_loaded": len(self.models) if hasattr(self, 'models') else 0
            }
