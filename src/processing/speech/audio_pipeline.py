"""
Comprehensive Audio Processing Pipeline
Author: Drmusab
Last Modified: 2025-05-26 13:51:29 UTC

This module provides a unified audio processing pipeline that orchestrates all
audio-related tasks including preprocessing, speech-to-text, emotion detection,
speaker recognition, and text-to-speech, with full integration into the AI
assistant's core architecture.
"""

from pathlib import Path
from typing import Optional, Dict, Any, Union, List, Tuple, Callable, AsyncGenerator, Set
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
from concurrent.futures import ThreadPoolExecutor
import queue

import torch
import sounddevice as sd
import soundfile as sf
import librosa

# Core imports
from src.core.config.loader import ConfigLoader
from src.core.events.event_bus import EventBus
from src.core.events.event_types import (
    AudioPipelineStarted,
    AudioPipelineCompleted,
    AudioPipelineError,
    AudioProcessingStarted,
    AudioProcessingCompleted,
    SpeechProcessingStarted,
    SpeechProcessingCompleted,
    EmotionDetectionStarted,
    EmotionDetectionCompleted,
    AudioStreamStarted,
    AudioStreamStopped,
    AudioQualityAssessed,
    WorkflowStateChanged
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
from src.processing.speech.speech_to_text import (
    EnhancedWhisperTranscriber,
    TranscriptionRequest,
    TranscriptionResult,
    TranscriptionQuality,
    AudioSource
)
from src.processing.speech.emotion_detection import (
    EnhancedEmotionDetector,
    EmotionDetectionRequest,
    EmotionResult,
    AnalysisMode,
    FeatureSet
)

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


class PipelineMode(Enum):
    """Audio pipeline processing modes."""
    REAL_TIME = "real_time"
    BATCH = "batch"
    STREAMING = "streaming"
    INTERACTIVE = "interactive"
    BACKGROUND = "background"


class WorkflowType(Enum):
    """Types of audio processing workflows."""
    TRANSCRIPTION_ONLY = "transcription_only"
    EMOTION_ONLY = "emotion_only"
    FULL_ANALYSIS = "full_analysis"
    PREPROCESSING_ONLY = "preprocessing_only"
    CUSTOM = "custom"


class QualityLevel(Enum):
    """Audio processing quality levels."""
    FAST = "fast"
    BALANCED = "balanced"
    HIGH_QUALITY = "high_quality"
    ULTRA_HIGH = "ultra_high"


class StreamingState(Enum):
    """Audio streaming states."""
    IDLE = "idle"
    RECORDING = "recording"
    PROCESSING = "processing"
    PAUSED = "paused"
    ERROR = "error"


@dataclass
class AudioPipelineRequest:
    """Comprehensive audio pipeline request configuration."""
    # Basic configuration
    workflow_type: WorkflowType = WorkflowType.FULL_ANALYSIS
    mode: PipelineMode = PipelineMode.REAL_TIME
    quality_level: QualityLevel = QualityLevel.BALANCED
    
    # Audio source configuration
    audio_source: Optional[Union[str, Path, np.ndarray]] = None
    audio_format: Optional[AudioFormat] = None
    sample_rate: Optional[int] = None
    
    # Processing settings
    enable_preprocessing: bool = True
    enable_noise_reduction: bool = True
    enable_normalization: bool = True
    enable_voice_activity_detection: bool = True
    
    # Component-specific settings
    transcription_config: Optional[TranscriptionRequest] = None
    emotion_config: Optional[EmotionDetectionRequest] = None
    
    # Pipeline behavior
    parallel_processing: bool = True
    cache_results: bool = True
    store_intermediate_results: bool = False
    
    # Session management
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    conversation_id: Optional[str] = None
    
    # Advanced options
    custom_processors: List[str] = field(default_factory=list)
    quality_thresholds: Dict[str, float] = field(default_factory=dict)
    timeout_seconds: float = 30.0
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: Set[str] = field(default_factory=set)


@dataclass
class AudioPipelineResult:
    """Comprehensive audio pipeline processing result."""
    # Core results
    audio_metadata: Optional[AudioMetadata] = None
    transcription_result: Optional[TranscriptionResult] = None
    emotion_result: Optional[EmotionResult] = None
    
    # Processing information
    workflow_type: WorkflowType = WorkflowType.FULL_ANALYSIS
    processing_time: float = 0.0
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Pipeline state
    success: bool = True
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    
    # Intermediate results
    preprocessed_audio: Optional[np.ndarray] = None
    audio_features: Optional[Dict[str, Any]] = None
    
    # Session information
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    conversation_id: Optional[str] = None
    
    # Performance metrics
    component_timings: Dict[str, float] = field(default_factory=dict)
    memory_usage: Dict[str, float] = field(default_factory=dict)
    
    # Quality assessment
    overall_confidence: float = 0.0
    quality_score: float = 0.0
    
    # Timestamp and tracking
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    pipeline_version: str = "1.0.0"


@dataclass
class StreamingConfig:
    """Configuration for audio streaming operations."""
    chunk_size: int = 1024
    buffer_duration: float = 5.0
    overlap_duration: float = 0.5
    vad_threshold: float = 0.5
    silence_timeout: float = 2.0
    max_duration: float = 300.0
    auto_save: bool = True
    callback_interval: float = 0.1


class AudioPipelineError(Exception):
    """Custom exception for audio pipeline operations."""
    
    def __init__(self, message: str, error_code: Optional[str] = None, component: Optional[str] = None):
        super().__init__(message)
        self.error_code = error_code
        self.component = component
        self.timestamp = datetime.now(timezone.utc)


class AudioQualityAssessor:
    """Component for assessing audio quality and providing recommendations."""
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.logger = get_logger(__name__)
    
    def assess_quality(self, audio: np.ndarray, metadata: AudioMetadata) -> Dict[str, float]:
        """Assess comprehensive audio quality metrics."""
        metrics = {}
        
        try:
            # Signal-to-noise ratio estimation
            metrics['snr_estimate'] = self._estimate_snr(audio)
            
            # Dynamic range assessment
            metrics['dynamic_range'] = self._calculate_dynamic_range(audio)
            
            # Clipping detection
            metrics['clipping_ratio'] = self._detect_clipping(audio)
            
            # Frequency content analysis
            metrics['frequency_balance'] = self._analyze_frequency_balance(audio)
            
            # Voice activity ratio
            metrics['voice_activity_ratio'] = self._calculate_voice_activity_ratio(audio)
            
            # Overall quality score
            metrics['overall_quality'] = self._calculate_overall_quality(metrics)
            
            return metrics
            
        except Exception as e:
            self.logger.warning(f"Quality assessment failed: {str(e)}")
            return {'overall_quality': 0.5}  # Default moderate quality
    
    def _estimate_snr(self, audio: np.ndarray) -> float:
        """Estimate signal-to-noise ratio."""
        # Simple SNR estimation using energy-based voice activity detection
        frame_length = int(0.025 * self.sample_rate)
        hop_length = int(0.010 * self.sample_rate)
        
        frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=hop_length)
        frame_energies = np.sum(frames ** 2, axis=0)
        
        # Assume top 60% energy frames contain speech
        speech_threshold = np.percentile(frame_energies, 40)
        speech_energy = np.mean(frame_energies[frame_energies > speech_threshold])
        noise_energy = np.mean(frame_energies[frame_energies <= speech_threshold])
        
        if noise_energy > 0:
            snr = 10 * np.log10(speech_energy / noise_energy)
            return max(0.0, min(40.0, snr)) / 40.0  # Normalize to 0-1
        return 1.0
    
    def _calculate_dynamic_range(self, audio: np.ndarray) -> float:
        """Calculate dynamic range of audio signal."""
        if len(audio) == 0:
            return 0.0
        
        # Calculate RMS in dB
        rms = np.sqrt(np.mean(audio ** 2))
        if rms > 0:
            rms_db = 20 * np.log10(rms)
            peak_db = 20 * np.log10(np.max(np.abs(audio)))
            dynamic_range = peak_db - rms_db
            return max(0.0, min(60.0, dynamic_range)) / 60.0  # Normalize to 0-1
        return 0.0
    
    def _detect_clipping(self, audio: np.ndarray) -> float:
        """Detect audio clipping ratio."""
        if len(audio) == 0:
            return 0.0
        
        # Count samples near maximum amplitude
        threshold = 0.95
        clipped_samples = np.sum(np.abs(audio) > threshold)
        return clipped_samples / len(audio)
    
    def _analyze_frequency_balance(self, audio: np.ndarray) -> float:
        """Analyze frequency content balance."""
        # Compute spectral centroid and spread
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate)[0]
        
        # Good balance if centroid is in speech range (roughly 1-4 kHz)
        mean_centroid = np.mean(spectral_centroids)
        optimal_range = (1000, 4000)
        
        if optimal_range[0] <= mean_centroid <= optimal_range[1]:
            return 1.0
        else:
            # Penalize deviation from optimal range
            deviation = min(abs(mean_centroid - optimal_range[0]), 
                          abs(mean_centroid - optimal_range[1]))
            return max(0.0, 1.0 - deviation / 2000.0)
    
    def _calculate_voice_activity_ratio(self, audio: np.ndarray) -> float:
        """Calculate ratio of voice activity in audio."""
        # Simple energy-based VAD
        frame_length = int(0.025 * self.sample_rate)
        hop_length = int(0.010 * self.sample_rate)
        
        rms_energy = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
        energy_threshold = np.mean(rms_energy) * 0.1
        
        voice_frames = np.sum(rms_energy > energy_threshold)
        return voice_frames / len(rms_energy) if len(rms_energy) > 0 else 0.0
    
    def _calculate_overall_quality(self, metrics: Dict[str, float]) -> float:
        """Calculate overall quality score from individual metrics."""
        # Weighted combination of quality metrics
        weights = {
            'snr_estimate': 0.3,
            'dynamic_range': 0.2,
            'frequency_balance': 0.2,
            'voice_activity_ratio': 0.2,
            'clipping_ratio': -0.1  # Negative weight for clipping
        }
        
        score = 0.0
        for metric, weight in weights.items():
            if metric in metrics:
                if metric == 'clipping_ratio':
                    score += weight * (1.0 - metrics[metric])  # Invert clipping ratio
                else:
                    score += weight * metrics[metric]
        
        return max(0.0, min(1.0, score))


class EnhancedAudioPipeline:
    """
    Comprehensive audio processing pipeline with full AI assistant integration.
    
    Features:
    - Unified orchestration of all audio processing workflows
    - Real-time and batch processing capabilities
    - Advanced audio preprocessing and quality assessment
    - Parallel processing of multiple audio analysis tasks
    - Comprehensive caching and memory integration
    - Event-driven architecture with detailed monitoring
    - Adaptive learning and personalization
    - Streaming audio processing with voice activity detection
    - Error handling and recovery mechanisms
    - Performance optimization and resource management
    """
    
    def __init__(self, container: Container):
        """
        Initialize the enhanced audio processing pipeline.
        
        Args:
            container: Dependency injection container
        """
        self.container = container
        self.logger = get_logger(__name__)
        self.config = container.get(ConfigLoader)
        self.event_bus = container.get(EventBus)
        self.error_handler = container.get(ErrorHandler)
        self.health_check = container.get(HealthCheck)
        
        # Initialize core components
        self._setup_audio_config()
        self._setup_processors()
        self._setup_monitoring()
        self._setup_caching()
        self._setup_streaming()
        self._setup_learning()
        
        # Pipeline state management
        self._active_sessions = {}
        self._session_lock = threading.Lock()
        self._processing_queue = queue.Queue()
        self._thread_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="audio_pipeline")
        
        # Register health check
        self.health_check.register_component(
            "audio_pipeline",
            self._health_check_callback
        )
        
        self.logger.info("EnhancedAudioPipeline initialized successfully")
    
    def _setup_audio_config(self) -> None:
        """Setup audio processing configuration."""
        self.sample_rate = self.config.get("audio_pipeline.sample_rate", 16000)
        self.default_chunk_size = self.config.get("audio_pipeline.chunk_size", 1024)
        self.max_audio_duration = self.config.get("audio_pipeline.max_duration", 600)
        
        # Quality settings
        self.quality_thresholds = {
            'min_snr': self.config.get("audio_pipeline.quality.min_snr", 0.3),
            'min_voice_activity': self.config.get("audio_pipeline.quality.min_voice_activity", 0.1),
            'max_clipping': self.config.get("audio_pipeline.quality.max_clipping", 0.05)
        }
        
        # Initialize quality assessor
        self.quality_assessor = AudioQualityAssessor(sample_rate=self.sample_rate)
    
    def _setup_processors(self) -> None:
        """Setup audio processing components."""
        # Core audio processor
        self.audio_processor = self.container.get(EnhancedAudioProcessor)
        
        # Speech-to-text processor
        self.speech_transcriber = self.container.get(EnhancedWhisperTranscriber)
        
        # Emotion detection processor
        self.emotion_detector = self.container.get(EnhancedEmotionDetector)
        
        # Component availability flags
        self.components_available = {
            'audio_processor': self.audio_processor is not None,
            'speech_transcriber': self.speech_transcriber is not None,
            'emotion_detector': self.emotion_detector is not None
        }
    
    def _setup_monitoring(self) -> None:
        """Setup monitoring and metrics collection."""
        self.metrics = self.container.get(MetricsCollector)
        self.tracer = self.container.get(TraceManager)
        
        # Register pipeline metrics
        self.metrics.register_counter("audio_pipeline_requests_total")
        self.metrics.register_histogram("audio_pipeline_processing_duration_seconds")
        self.metrics.register_gauge("audio_pipeline_quality_score")
        self.metrics.register_counter("audio_pipeline_errors_total")
        self.metrics.register_gauge("audio_pipeline_active_sessions")
        self.metrics.register_histogram("audio_pipeline_component_duration_seconds")
    
    def _setup_caching(self) -> None:
        """Setup caching for pipeline results."""
        self.cache_manager = self.container.get(CacheManager)
        self.cache_strategy = self.container.get(CacheStrategy)
        
        self.cache_enabled = self.config.get("audio_pipeline.caching.enabled", True)
        self.cache_ttl = self.config.get("audio_pipeline.caching.ttl_seconds", 3600)
    
    def _setup_streaming(self) -> None:
        """Setup audio streaming capabilities."""
        self.streaming_enabled = self.config.get("audio_pipeline.streaming.enabled", True)
        self.default_streaming_config = StreamingConfig(
            chunk_size=self.config.get("audio_pipeline.streaming.chunk_size", 1024),
            buffer_duration=self.config.get("audio_pipeline.streaming.buffer_duration", 5.0),
            overlap_duration=self.config.get("audio_pipeline.streaming.overlap_duration", 0.5),
            vad_threshold=self.config.get("audio_pipeline.streaming.vad_threshold", 0.5),
            silence_timeout=self.config.get("audio_pipeline.streaming.silence_timeout", 2.0),
            max_duration=self.config.get("audio_pipeline.streaming.max_duration", 300.0)
        )
        
        # Streaming state tracking
        self.streaming_sessions = {}
        self.streaming_lock = threading.Lock()
    
    def _setup_learning(self) -> None:
        """Setup learning and adaptation capabilities."""
        self.memory_manager = self.container.get(MemoryManager)
        self.context_manager = self.container.get(ContextManager)
        self.feedback_processor = self.container.get_optional(FeedbackProcessor)
        self.preference_learner = self.container.get_optional(PreferenceLearner)
        self.continual_learner = self.container.get_optional(ContinualLearner)
        
        self.learning_enabled = self.config.get("audio_pipeline.learning.enabled", True)
    
    @handle_exceptions
    async def process_audio(
        self,
        request: AudioPipelineRequest
    ) -> AudioPipelineResult:
        """
        Main audio processing pipeline with comprehensive workflow orchestration.
        
        Args:
            request: Audio pipeline processing request
            
        Returns:
            Comprehensive audio processing result
        """
        start_time = datetime.now(timezone.utc)
        
        # Generate session ID if not provided
        if not request.session_id:
            request.session_id = self._generate_session_id()
        
        # Track active session
        with self._session_lock:
            self._active_sessions[request.session_id] = {
                'start_time': start_time,
                'status': 'processing',
                'request': request
            }
            self.metrics.set("audio_pipeline_active_sessions", len(self._active_sessions))
        
        # Emit pipeline started event
        await self.event_bus.emit(AudioPipelineStarted(
            session_id=request.session_id,
            workflow_type=request.workflow_type.value,
            mode=request.mode.value,
            quality_level=request.quality_level.value
        ))
        
        try:
            with self.tracer.trace("audio_pipeline_processing") as span:
                span.set_attributes({
                    "session_id": request.session_id,
                    "workflow_type": request.workflow_type.value,
                    "mode": request.mode.value,
                    "quality_level": request.quality_level.value
                })
                
                # Check cache first
                cache_key = None
                if request.cache_results and self.cache_enabled:
                    cache_key = self._generate_cache_key(request)
                    cached_result = await self._get_cached_result(cache_key)
                    if cached_result:
                        self.logger.info(f"Returning cached result for session {request.session_id}")
                        return cached_result
                
                # Load and preprocess audio
                processed_audio, audio_metadata = await self._load_and_preprocess_audio(request)
                
                # Assess audio quality
                quality_metrics = await self._assess_audio_quality(processed_audio, audio_metadata)
                
                # Check quality thresholds
                quality_warnings = self._check_quality_thresholds(quality_metrics)
                
                # Execute processing workflow
                result = await self._execute_workflow(
                    request, processed_audio, audio_metadata, quality_metrics
                )
                
                # Add quality warnings
                result.warnings.extend(quality_warnings)
                
                # Calculate overall metrics
                processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
                result.processing_time = processing_time
                result.overall_confidence = self._calculate_overall_confidence(result)
                result.quality_score = quality_metrics.get('overall_quality', 0.5)
                
                # Cache result if enabled
                if cache_key:
                    await self._cache_result(cache_key, result)
                
                # Store in memory and context
                await self._store_pipeline_context(result, request)
                
                # Update metrics
                self._update_metrics(result, processing_time)
                
                # Learn from result
                if self.learning_enabled:
                    await self._learn_from_processing(result, request)
                
                # Emit completion event
                await self.event_bus.emit(AudioPipelineCompleted(
                    session_id=request.session_id,
                    processing_time=processing_time,
                    quality_score=result.quality_score,
                    overall_confidence=result.overall_confidence,
                    success=result.success
                ))
                
                self.logger.info(
                    f"Audio pipeline completed for session {request.session_id} "
                    f"(Time: {processing_time:.2f}s, Quality: {result.quality_score:.2f})"
                )
                
                return result
                
        except Exception as e:
            # Create error result
            error_result = AudioPipelineResult(
                workflow_type=request.workflow_type,
                success=False,
                error_message=str(e),
                session_id=request.session_id,
                user_id=request.user_id,
                conversation_id=request.conversation_id
            )
            
            # Emit error event
            await self.event_bus.emit(AudioPipelineError(
                session_id=request.session_id,
                error_type=type(e).__name__,
                error_message=str(e),
                component="audio_pipeline"
            ))
            
            self.metrics.increment("audio_pipeline_errors_total")
            self.logger.error(f"Audio pipeline failed for session {request.session_id}: {str(e)}")
            
            return error_result
            
        finally:
            # Clean up session tracking
            with self._session_lock:
                self._active_sessions.pop(request.session_id, None)
                self.metrics.set("audio_pipeline_active_sessions", len(self._active_sessions))
    
    async def _load_and_preprocess_audio(
        self, 
        request: AudioPipelineRequest
    ) -> Tuple[np.ndarray, AudioMetadata]:
        """Load and preprocess audio data."""
        start_time = datetime.now(timezone.utc)
        
        try:
            # Load audio data
            if isinstance(request.audio_source, (str, Path)):
                audio_data, sr = await self.audio_processor.load_audio(
                    request.audio_source,
                    target_sr=request.sample_rate or self.sample_rate,
                    normalize=request.enable_normalization
                )
                file_path = Path(request.audio_source)
                audio_metadata = AudioMetadata(
                    sample_rate=sr,
                    channels=1,  # Converted to mono
                    duration=len(audio_data) / sr,
                    format=file_path.suffix.lstrip('.'),
                    file_size=file_path.stat().st_size if file_path.exists() else None,
                    created_at=datetime.now(timezone.utc)
                )
            elif isinstance(request.audio_source, np.ndarray):
                audio_data = request.audio_source.copy()
                sr = request.sample_rate or self.sample_rate
                audio_metadata = AudioMetadata(
                    sample_rate=sr,
                    channels=1,
                    duration=len(audio_data) / sr,
                    format="array",
                    created_at=datetime.now(timezone.utc)
                )
            else:
                raise AudioPipelineError("Invalid audio source provided")
            
            # Apply preprocessing if enabled
            if request.enable_preprocessing:
                processed_audio = await self._apply_preprocessing(audio_data, request)
            else:
                processed_audio = audio_data
            
            # Record preprocessing time
            preprocessing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            self.metrics.record("audio_pipeline_component_duration_seconds", 
                              preprocessing_time, {"component": "preprocessing"})
            
            return processed_audio, audio_metadata
            
        except Exception as e:
            raise AudioPipelineError(f"Audio loading/preprocessing failed: {str(e)}", component="preprocessing")
    
    async def _apply_preprocessing(self, audio: np.ndarray, request: AudioPipelineRequest) -> np.ndarray:
        """Apply comprehensive audio preprocessing."""
        processed_audio = audio.copy()
        
        # Set quality preset based on request
        quality_preset = request.quality_level.value
        self.audio_processor.set_quality_preset(quality_preset)
        
        # Apply normalization
        if request.enable_normalization:
            processed_audio = self.audio_processor.normalize_audio(processed_audio)
        
        # Apply noise reduction
        if request.enable_noise_reduction:
            processed_audio = self.audio_processor.advanced_noise_reduction(
                processed_audio, 
                self.sample_rate,
                method="spectral_gating"
            )
        
        # Trim silence while preserving emotional cues
        processed_audio = self.audio_processor.trim_silence(
            processed_audio, 
            threshold_db=-40.0
        )
        
        return processed_audio
    
    async def _assess_audio_quality(
        self, 
        audio: np.ndarray, 
        metadata: AudioMetadata
    ) -> Dict[str, float]:
        """Assess comprehensive audio quality."""
        start_time = datetime.now(timezone.utc)
        
        try:
            quality_metrics = self.quality_assessor.assess_quality(audio, metadata)
            
            # Emit quality assessment event
            await self.event_bus.emit(AudioQualityAssessed(
                session_id=getattr(self, '_current_session_id', 'unknown'),
                quality_score=quality_metrics.get('overall_quality', 0.5),
                snr_estimate=quality_metrics.get('snr_estimate', 0.0),
                voice_activity_ratio=quality_metrics.get('voice_activity_ratio', 0.0),
                clipping_ratio=quality_metrics.get('clipping_ratio', 0.0)
            ))
            
            # Record assessment time
            assessment_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            self.metrics.record("audio_pipeline_component_duration_seconds", 
                              assessment_time, {"component": "quality_assessment"})
            
            return quality_metrics
            
        except Exception as e:
            self.logger.warning(f"Quality assessment failed: {str(e)}")
            return {'overall_quality': 0.5}
    
    def _check_quality_thresholds(self, quality_metrics: Dict[str, float]) -> List[str]:
        """Check quality metrics against thresholds and return warnings."""
        warnings = []
        
        # Check SNR
        snr = quality_metrics.get('snr_estimate', 1.0)
        if snr < self.quality_thresholds['min_snr']:
            warnings.append(f"Low signal-to-noise ratio: {snr:.2f}")
        
        # Check voice activity
        voice_activity = quality_metrics.get('voice_activity_ratio', 1.0)
        if voice_activity < self.quality_thresholds['min_voice_activity']:
            warnings.append(f"Low voice activity: {voice_activity:.2f}")
        
        # Check clipping
        clipping = quality_metrics.get('clipping_ratio', 0.0)
        if clipping > self.quality_thresholds['max_clipping']:
            warnings.append(f"Audio clipping detected: {clipping:.2f}")
        
        return warnings
    
    async def _execute_workflow(
        self,
        request: AudioPipelineRequest,
        audio: np.ndarray,
        metadata: AudioMetadata,
        quality_metrics: Dict[str, float]
    ) -> AudioPipelineResult:
        """Execute the specified audio processing workflow."""
        result = AudioPipelineResult(
            workflow_type=request.workflow_type,
            audio_metadata=metadata,
            quality_metrics=quality_metrics,
            session_id=request.session_id,
            user_id=request.user_id,
            conversation_id=request.conversation_id
        )
        
        # Store preprocessed audio if requested
        if request.store_intermediate_results:
            result.preprocessed_audio = audio.copy()
        
        try:
            if request.workflow_type == WorkflowType.TRANSCRIPTION_ONLY:
                await self._execute_transcription_workflow(request, audio, result)
                
            elif request.workflow_type == WorkflowType.EMOTION_ONLY:
                await self._execute_emotion_workflow(request, audio, result)
                
            elif request.workflow_type == WorkflowType.FULL_ANALYSIS:
                if request.parallel_processing:
                    await self._execute_parallel_workflow(request, audio, result)
                else:
                    await self._execute_sequential_workflow(request, audio, result)
                    
            elif request.workflow_type == WorkflowType.PREPROCESSING_ONLY:
                # Already done in preprocessing step
                pass
                
            elif request.workflow_type == WorkflowType.CUSTOM:
                await self._execute_custom_workflow(request, audio, result)
            
            result.success = True
            
        except Exception as e:
            result.success = False
            result.error_message = str(e)
            self.logger.error(f"Workflow execution failed: {str(e)}")
        
        return result
    
    async def _execute_transcription_workflow(
        self,
        request: AudioPipelineRequest,
        audio: np.ndarray,
        result: AudioPipelineResult
    ) -> None:
        """Execute transcription-only workflow."""
        start_time = datetime.now(timezone.utc)
        
        # Configure transcription request
        transcription_request = request.transcription_config or TranscriptionRequest(
            audio_source=AudioSource.BUFFER,
            quality=self._map_quality_level_to_transcription(request.quality_level),
            session_id=request.session_id,
            user_id=request.user_id
        )
        
        # Perform transcription
        transcription_result = await self.speech_transcriber.transcribe(
            audio, transcription_request
        )
        
        result.transcription_result = transcription_result
        
        # Record timing
        transcription_time = (datetime.now(timezone.utc) - start_time).total_seconds()
        result.component_timings['transcription'] = transcription_time
        self.metrics.record("audio_pipeline_component_duration_seconds", 
                          transcription_time, {"component": "transcription"})
    
    async def _execute_emotion_workflow(
        self,
        request: AudioPipelineRequest,
        audio: np.ndarray,
        result: AudioPipelineResult
    ) -> None:
        """Execute emotion detection-only workflow."""
        start_time = datetime.now(timezone.utc)
        
        # Configure emotion detection request
        emotion_request = request.emotion_config or EmotionDetectionRequest(
            analysis_mode=self._map_mode_to_analysis_mode(request.mode),
            feature_set=self._map_quality_level_to_feature_set(request.quality_level),
            session_id=request.session_id,
            user_id=request.user_id
        )
        
        # Perform emotion detection
        emotion_result = await self.emotion_detector.detect_emotion(
            audio, emotion_request
        )
        
        result.emotion_result = emotion_result
        
        # Record timing
        emotion_time = (datetime.now(timezone.utc) - start_time).total_seconds()
        result.component_timings['emotion_detection'] = emotion_time
        self.metrics.record("audio_pipeline_component_duration_seconds", 
                          emotion_time, {"component": "emotion_detection"})
    
    async def _execute_parallel_workflow(
        self,
        request: AudioPipelineRequest,
        audio: np.ndarray,
        result: AudioPipelineResult
    ) -> None:
        """Execute full analysis workflow with parallel processing."""
        start_time = datetime.now(timezone.utc)
        
        # Create tasks for parallel execution
        tasks = []
        
        if self.components_available['speech_transcriber']:
            transcription_request = request.transcription_config or TranscriptionRequest(
                audio_source=AudioSource.BUFFER,
                quality=self._map_quality_level_to_transcription(request.quality_level),
                session_id=request.session_id,
                user_id=request.user_id
            )
            tasks.append(self.speech_transcriber.transcribe(audio, transcription_request))
        
        if self.components_available['emotion_detector']:
            emotion_request = request.emotion_config or EmotionDetectionRequest(
                analysis_mode=self._map_mode_to_analysis_mode(request.mode),
                feature_set=self._map_quality_level_to_feature_set(request.quality_level),
                session_id=request.session_id,
                user_id=request.user_id
            )
            tasks.append(self.emotion_detector.detect_emotion(audio, emotion_request))
        
        # Execute tasks in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for i, task_result in enumerate(results):
            if isinstance(task_result, Exception):
                self.logger.error(f"Parallel task {i} failed: {str(task_result)}")
                result.warnings.append(f"Component {i} failed: {str(task_result)}")
            else:
                if isinstance(task_result, TranscriptionResult):
                    result.transcription_result = task_result
                elif isinstance(task_result, EmotionResult):
                    result.emotion_result = task_result
        
        # Record timing
        parallel_time = (datetime.now(timezone.utc) - start_time).total_seconds()
        result.component_timings['parallel_processing'] = parallel_time
    
    async def _execute_sequential_workflow(
        self,
        request: AudioPipelineRequest,
        audio: np.ndarray,
        result: AudioPipelineResult
    ) -> None:
        """Execute full analysis workflow with sequential processing."""
        # Execute transcription first
        if self.components_available['speech_transcriber']:
            await self._execute_transcription_workflow(request, audio, result)
        
        # Execute emotion detection
        if self.components_available['emotion_detector']:
            await self._execute_emotion_workflow(request, audio, result)
    
    async def _execute_custom_workflow(
        self,
        request: AudioPipelineRequest,
        audio: np.ndarray,
        result: AudioPipelineResult
    ) -> None:
        """Execute custom processing workflow."""
        # Custom workflow implementation would be based on request.custom_processors
        # For now, default to full analysis
        await self._execute_parallel_workflow(request, audio, result)
    
    def _calculate_overall_confidence(self, result: AudioPipelineResult) -> float:
        """Calculate overall confidence score from component results."""
        confidences = []
        
        if result.transcription_result:
            confidences.append(result.transcription_result.confidence)
        
        if result.emotion_result:
            confidences.append(result.emotion_result.confidence_score)
        
        if result.quality_metrics:
            confidences.append(result.quality_metrics.get('overall_quality', 0.5))
        
        return np.mean(confidences) if confidences else 0.0
    
    async def start_streaming(
        self,
        request: AudioPipelineRequest,
        callback: Callable[[AudioPipelineResult], None],
        streaming_config: Optional[StreamingConfig] = None
    ) -> str:
        """Start streaming audio processing session."""
        if not self.streaming_enabled:
            raise AudioPipelineError("Streaming is not enabled")
        
        session_id = request.session_id or self._generate_session_id()
        config = streaming_config or self.default_streaming_config
        
        with self.streaming_lock:
            if session_id in self.streaming_sessions:
                raise AudioPipelineError(f"Streaming session {session_id} already exists")
            
            self.streaming_sessions[session_id] = {
                'state': StreamingState.IDLE,
                'request': request,
                'config': config,
                'callback': callback,
                'start_time': datetime.now(timezone.utc),
                'audio_buffer': [],
                'processing_task': None
            }
        
        # Emit streaming started event
        await self.event_bus.emit(AudioStreamStarted(
            session_id=session_id,
            sample_rate=self.sample_rate,
            chunk_size=config.chunk_size
        ))
        
        # Start streaming processing task
        task = asyncio.create_task(self._streaming_processing_loop(session_id))
        self.streaming_sessions[session_id]['processing_task'] = task
        
        self.logger.info(f"Started streaming session: {session_id}")
        return session_id
    
    async def stop_streaming(self, session_id: str) -> None:
        """Stop streaming audio processing session."""
        with self.streaming_lock:
            if session_id not in self.streaming_sessions:
                raise AudioPipelineError(f"Streaming session {session_id} not found")
            
            session = self.streaming_sessions[session_id]
            session['state'] = StreamingState.IDLE
            
            # Cancel processing task
            if session['processing_task']:
                session['processing_task'].cancel()
            
            # Clean up session
            del self.streaming_sessions[session_id]
        
        # Emit streaming stopped event
        await self.event_bus.emit(AudioStreamStopped(
            session_id=session_id,
            duration=(datetime.now(timezone.utc) - session['start_time']).total_seconds()
        ))
        
        self.logger.info(f"Stopped streaming session: {session_id}")
    
    async def _streaming_processing_loop(self, session_id: str) -> None:
        """Main processing loop for streaming audio."""
        try:
            with self.streaming_lock:
                session = self.streaming_sessions.get(session_id)
                if not session:
                    return
                
                config = session['config']
                request = session['request']
                callback = session['callback']
            
            # Initialize audio input stream
            audio_queue = asyncio.Queue()
            
            def audio_callback(indata: np.ndarray, frames: int, time, status):
                if status:
                    self.logger.warning(f"Audio input status: {status}")
                try:
                    asyncio.get_event_loop().call_soon_threadsafe(
                        audio_queue.put_nowait, indata.copy()
                    )
                except:
                    pass  # Queue might be full or loop might be closed
            
            # Start audio stream
            stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype=np.float32,
                blocksize=config.chunk_size,
                callback=audio_callback
            )
            
            with stream:
                session['state'] = StreamingState.RECORDING
                buffer = []
                silence_start = None
                
                while session['state'] in [StreamingState.RECORDING, StreamingState.PROCESSING]:
                    try:
                        # Get audio chunk with timeout
                        chunk = await asyncio.wait_for(audio_queue.get(), timeout=1.0)
                        buffer.append(chunk.flatten())
                        
                        # Check for voice activity
                        chunk_energy = np.mean(chunk ** 2)
                        is_speech = chunk_energy > config.vad_threshold
                        
                        if is_speech:
                            silence_start = None
                        elif silence_start is None:
                            silence_start = datetime.now(timezone.utc)
                        
                        # Check if we have enough audio to process
                        buffer_duration = len(buffer) * config.chunk_size / self.sample_rate
                        
                        if buffer_duration >= config.buffer_duration:
                            # Process accumulated audio
                            audio_segment = np.concatenate(buffer)
                            
                            # Create processing request
                            segment_request = AudioPipelineRequest(
                                workflow_type=request.workflow_type,
                                mode=PipelineMode.STREAMING,
                                quality_level=request.quality_level,
                                audio_source=audio_segment,
                                session_id=session_id,
                                user_id=request.user_id,
                                parallel_processing=True,
                                cache_results=False
                            )
                            
                            # Process segment
                            session['state'] = StreamingState.PROCESSING
                            result = await self.process_audio(segment_request)
                            session['state'] = StreamingState.RECORDING
                            
                            # Call callback with result
                            if callback:
                                try:
                                    callback(result)
                                except Exception as e:
                                    self.logger.error(f"Streaming callback failed: {str(e)}")
                            
                            # Overlap handling
                            overlap_samples = int(config.overlap_duration * self.sample_rate)
                            if overlap_samples > 0 and len(audio_segment) > overlap_samples:
                                buffer = [audio_segment[-overlap_samples:]]
                            else:
                                buffer = []
                        
                        # Check for silence timeout
                        if (silence_start and 
                            (datetime.now(timezone.utc) - silence_start).total_seconds() > config.silence_timeout):
                            break
                        
                        # Check for maximum duration
                        total_duration = (datetime.now(timezone.utc) - session['start_time']).total_seconds()
                        if total_duration > config.max_duration:
                            break
                            
                    except asyncio.TimeoutError:
                        continue  # Continue if no audio received
                    except Exception as e:
                        self.logger.error(f"Streaming processing error: {str(e)}")
                        session['state'] = StreamingState.ERROR
                        break
                
        except Exception as e:
            self.logger.error(f"Streaming session {session_id} failed: {str(e)}")
            with self.streaming_lock:
                if session_id in self.streaming_sessions:
                    self.streaming_sessions[session_id]['state'] = StreamingState.ERROR
    
    def _map_quality_level_to_transcription(self, quality: QualityLevel) -> TranscriptionQuality:
        """Map pipeline quality level to transcription quality."""
        mapping = {
            QualityLevel.FAST: TranscriptionQuality.FAST,
            QualityLevel.BALANCED: TranscriptionQuality.BALANCED,
            QualityLevel.HIGH_QUALITY: TranscriptionQuality.HIGH_QUALITY,
            QualityLevel.ULTRA_HIGH: TranscriptionQuality.ULTRA_HIGH
        }
        return mapping.get(quality, TranscriptionQuality.BALANCED)
    
    def _map_mode_to_analysis_mode(self, mode: PipelineMode) -> AnalysisMode:
        """Map pipeline mode to emotion analysis mode."""
        mapping = {
            PipelineMode.REAL_TIME: AnalysisMode.REAL_TIME,
            PipelineMode.BATCH: AnalysisMode.BATCH,
            PipelineMode.STREAMING: AnalysisMode.STREAMING,
            PipelineMode.INTERACTIVE: AnalysisMode.CONTEXTUAL,
            PipelineMode.BACKGROUND: AnalysisMode.BATCH
        }
        return mapping.get(mode, AnalysisMode.REAL_TIME)
    
    def _map_quality_level_to_feature_set(self, quality: QualityLevel) -> FeatureSet:
        """Map pipeline quality level to emotion feature set."""
        mapping = {
            QualityLevel.FAST: FeatureSet.BASIC,
            QualityLevel.BALANCED: FeatureSet.ADVANCED,
            QualityLevel.HIGH_QUALITY: FeatureSet.COMPREHENSIVE,
            QualityLevel.ULTRA_HIGH: FeatureSet.COMPREHENSIVE
        }
        return mapping.get(quality, FeatureSet.ADVANCED)
    
    def _generate_session_id(self) -> str:
        """Generate unique session identifier."""
        timestamp = datetime.now(timezone.utc).isoformat()
        return hashlib.md5(f"{timestamp}_{id(self)}".encode()).hexdigest()[:16]
    
    def _generate_cache_key(self, request: AudioPipelineRequest) -> str:
        """Generate cache key for pipeline request."""
        # Create hash of audio content
        if isinstance(request.audio_source, (str, Path)):
            audio_hash = hashlib.md5(Path(request.audio_source).read_bytes()).hexdigest()
        elif isinstance(request.audio_source, np.ndarray):
            audio_hash = hashlib.md5(request.audio_source.tobytes()).hexdigest()
        else:
            audio_hash = "no_audio"
        
        # Create hash of request parameters
        request_dict = asdict(request)
        request_dict.pop("session_id", None)
        request_dict.pop("metadata", None)
        request_str = json.dumps(request_dict, sort_keys=True, default=str)
        request_hash = hashlib.md5(request_str.encode()).hexdigest()
        
        return f"pipeline:{audio_hash[:16]}:{request_hash[:16]}"
    
    async def _get_cached_result(self, cache_key: str) -> Optional[AudioPipelineResult]:
        """Retrieve cached pipeline result."""
        try:
            cached_data = await self.cache_manager.get(cache_key)
            if cached_data:
                # Reconstruct result object
                result = AudioPipelineResult(**cached_data)
                return result
        except Exception as e:
            self.logger.warning(f"Cache retrieval failed: {str(e)}")
        return None
    
    async def _cache_result(self, cache_key: str, result: AudioPipelineResult) -> None:
        """Cache pipeline result."""
        try:
            # Convert for serialization
            cache_data = asdict(result)
            cache_data['timestamp'] = result.timestamp.isoformat()
            # Remove large objects that shouldn't be cached
            cache_data.pop('preprocessed_audio', None)
            
            await self.cache_manager.set(cache_key, cache_data, ttl=self.cache_ttl)
            
        except Exception as e:
            self.logger.warning(f"Cache storage failed: {str(e)}")
    
    async def _store_pipeline_context(
        self,
        result: AudioPipelineResult,
        request: AudioPipelineRequest
    ) -> None:
        """Store pipeline result in context memory."""
        try:
            context_data = {
                'workflow_type': result.workflow_type.value,
                'quality_score': result.quality_score,
                'overall_confidence': result.overall_confidence,
                'processing_time': result.processing_time,
                'success': result.success,
                'timestamp': result.timestamp.isoformat()
            }
            
            # Add component-specific data
            if result.transcription_result:
                context_data['transcription'] = {
                    'text': result.transcription_result.text,
                    'confidence': result.transcription_result.confidence,
                    'language': result.transcription_result.language
                }
            
            if result.emotion_result:
                context_data['emotion'] = {
                    'primary_emotion': result.emotion_result.primary_emotion.value,
                    'confidence': result.emotion_result.confidence_score,
                    'valence': result.emotion_result.valence,
                    'arousal': result.emotion_result.arousal
                }
            
            # Store in context manager
            if request.session_id:
                await self.context_manager.update_context(
                    request.session_id,
                    "audio_pipeline",
                    context_data
                )
            
        except Exception as e:
            self.logger.warning(f"Context storage failed: {str(e)}")
    
    def _update_metrics(self, result: AudioPipelineResult, processing_time: float) -> None:
        """Update monitoring metrics."""
        self.metrics.increment("audio_pipeline_requests_total")
        self.metrics.record("audio_pipeline_processing_duration_seconds", processing_time)
        self.metrics.set("audio_pipeline_quality_score", result.quality_score)
        
        if not result.success:
            self.metrics.increment("audio_pipeline_errors_total")
    
    async def _learn_from_processing(
        self,
        result: AudioPipelineResult,
        request: AudioPipelineRequest
    ) -> None:
        """Learn from pipeline processing results."""
        if not self.learning_enabled or not self.feedback_processor:
            return
        
        try:
            learning_data = {
                'workflow_type': result.workflow_type.value,
                'processing_time': result.processing_time,
                'quality_score': result.quality_score,
                'overall_confidence': result.overall_confidence,
                'success': result.success,
                'component_timings': result.component_timings,
                'quality_metrics': result.quality_metrics
            }
            
            await self.feedback_processor.process_feedback(
                "audio_pipeline",
                learning_data,
                request.user_id
            )
            
        except Exception as e:
            self.logger.warning(f"Learning from processing failed: {str(e)}")
    
    async def _health_check_callback(self) -> Dict[str, Any]:
        """Health check callback for audio pipeline."""
        try:
            # Test basic functionality
            test_audio = np.random.randn(self.sample_rate).astype(np.float32) * 0.01
            start_time = datetime.now(timezone.utc)
            
            test_request = AudioPipelineRequest(
                workflow_type=WorkflowType.PREPROCESSING_ONLY,
                mode=PipelineMode.REAL_TIME,
                quality_level=QualityLevel.FAST,
                audio_source=test_audio,
                cache_results=False
            )
            
            result = await self.process_audio(test_request)
            response_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            return {
                "status": "healthy",
                "components_available": self.components_available,
                "active_sessions": len(self._active_sessions),
                "streaming_sessions": len(self.streaming_sessions),
                "response_time_ms": response_time * 1000,
                "cache_enabled": self.cache_enabled,
                "learning_enabled": self.learning_enabled,
                "streaming_enabled": self.streaming_enabled
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "components_available": getattr(self, 'components_available', {}),
                "active_sessions": len(getattr(self, '_active_sessions', {}))
            }
    
    async def get_session_status(self, session_id: str) -> Dict[str, Any]:
        """Get status of a processing session."""
        with self._session_lock:
            if session_id in self._active_sessions:
                session = self._active_sessions[session_id]
                return {
                    'session_id': session_id,
                    'status': session['status'],
                    'start_time': session['start_time'].isoformat(),
                    'duration': (datetime.now(timezone.utc) - session['start_time']).total_seconds(),
                    'workflow_type': session['request'].workflow_type.value
                }
        
        with self.streaming_lock:
            if session_id in self.streaming_sessions:
                session = self.streaming_sessions[session_id]
                return {
                    'session_id': session_id,
                    'status': 'streaming',
                    'state': session['state'].value,
                    'start_time': session['start_time'].isoformat(),
                    'duration': (datetime.now(timezone.utc) - session['start_time']).total_seconds()
                }
        
        return {'session_id': session_id, 'status': 'not_found'}
    
    def get_pipeline_statistics(self) -> Dict[str, Any]:
        """Get comprehensive pipeline statistics."""
        return {
            'active_sessions': len(self._active_sessions),
            'streaming_sessions': len(self.streaming_sessions),
            'components_available': self.components_available,
            'cache_enabled': self.cache_enabled,
            'learning_enabled': self.learning_enabled,
            'streaming_enabled': self.streaming_enabled,
            'sample_rate': self.sample_rate,
            'quality_thresholds': self.quality_thresholds
        }
    
    async def cleanup(self) -> None:
        """Comprehensive cleanup of pipeline resources."""
        self.logger.info("Starting AudioPipeline cleanup...")
        
        # Stop all streaming sessions
        streaming_sessions = list(self.streaming_sessions.keys())
        for session_id in streaming_sessions:
            try:
                await self.stop_streaming(session_id)
            except Exception as e:
                self.logger.warning(f"Failed to stop streaming session {session_id}: {str(e)}")
        
        # Clean up active sessions
        with self._session_lock:
            self._active_sessions.clear()
        
        # Shutdown thread pool
        self._thread_pool.shutdown(wait=True)
        
        # Cleanup component processors
        if hasattr(self.audio_processor, 'cleanup'):
            await self.audio_processor.cleanup()
        
        if hasattr(self.speech_transcriber, 'cleanup'):
            await self.speech_transcriber.cleanup()
        
        # Unregister health check
        if hasattr(self, 'health_check'):
            self.health_check.unregister_component("audio_pipeline")
        
        self.logger.info("AudioPipeline cleanup completed")
    
    def __del__(self):
        """Destructor to ensure cleanup on object deletion."""
        try:
            # Basic cleanup for synchronous destruction
            if hasattr(self, '_thread_pool'):
                self._thread_pool.shutdown(wait=False)
        except Exception:
            pass  # Ignore cleanup errors in destructor
