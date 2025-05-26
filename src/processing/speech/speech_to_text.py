"""
Advanced Speech to Text Transcription Module
Author: Drmusab
Last Modified: 2025-05-26 13:26:17 UTC

This module provides comprehensive speech-to-text capabilities integrated with the
AI assistant's core architecture, including memory systems, event handling,
caching, and multimodal processing.
"""

from pathlib import Path
from typing import Optional, Dict, Any, Union, List, Callable, AsyncGenerator
import tempfile
from datetime import datetime, timezone
import asyncio
import json
import hashlib
from dataclasses import dataclass, asdict
from enum import Enum
import threading
from contextlib import asynccontextmanager

import numpy as np
import torch
import whisper
import sounddevice as sd
import soundfile as sf

# Core imports
from src.core.config.loader import ConfigLoader
from src.core.events.event_bus import EventBus
from src.core.events.event_types import (
    SpeechProcessingStarted,
    SpeechProcessingCompleted,
    SpeechProcessingError,
    AudioRecordingStarted,
    AudioRecordingCompleted
)
from src.core.error_handling import ErrorHandler, handle_exceptions
from src.core.dependency_injection import Container
from src.core.health_check import HealthCheck

# Processing imports
from src.processing.speech.audio_utils import (
    AudioProcessor,
    AudioProcessingError,
    AudioProcessorProtocol,
    AudioData
)
from src.processing.multimodal.fusion_strategies import MultimodalFusionStrategy
from src.processing.natural_language.entity_extractor import EntityExtractor
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


class TranscriptionQuality(Enum):
    """Enumeration for transcription quality levels."""
    FAST = "fast"
    BALANCED = "balanced"
    HIGH_QUALITY = "high_quality"
    ULTRA_HIGH = "ultra_high"


class AudioSource(Enum):
    """Enumeration for audio input sources."""
    MICROPHONE = "microphone"
    FILE = "file"
    STREAM = "stream"
    BUFFER = "buffer"


@dataclass
class TranscriptionRequest:
    """Data class for transcription requests."""
    audio_source: AudioSource
    language: Optional[str] = None
    task: str = "transcribe"
    quality: TranscriptionQuality = TranscriptionQuality.BALANCED
    enable_preprocessing: bool = True
    enable_emotion_detection: bool = False
    enable_speaker_identification: bool = False
    context_aware: bool = True
    cache_result: bool = True
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class TranscriptionResult:
    """Data class for transcription results."""
    text: str
    confidence: float
    language: str
    segments: List[Dict[str, Any]]
    processing_time: float
    quality_metrics: Dict[str, float]
    emotions: Optional[Dict[str, float]] = None
    speaker_info: Optional[Dict[str, Any]] = None
    entities: Optional[List[Dict[str, Any]]] = None
    sentiment: Optional[Dict[str, Any]] = None
    audio_features: Optional[Dict[str, Any]] = None
    session_id: Optional[str] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)


class EnhancedWhisperTranscriber:
    """
    Advanced Speech recognition system with comprehensive AI assistant integration.
    
    Features:
    - Multi-model Whisper support with dynamic model switching
    - Advanced audio preprocessing and quality enhancement
    - Context-aware transcription with memory integration
    - Real-time emotion and sentiment analysis
    - Speaker identification and voice profiling
    - Intelligent caching and result optimization
    - Event-driven architecture with comprehensive monitoring
    - Learning and adaptation capabilities
    - Multimodal fusion support
    """

    AVAILABLE_MODELS = {
        "tiny": {"size": 39, "multilingual": False, "speed": "fastest", "quality": "low"},
        "tiny.en": {"size": 39, "multilingual": False, "speed": "fastest", "quality": "low"},
        "base": {"size": 74, "multilingual": False, "speed": "fast", "quality": "medium"},
        "base.en": {"size": 74, "multilingual": False, "speed": "fast", "quality": "medium"},
        "small": {"size": 244, "multilingual": True, "speed": "medium", "quality": "good"},
        "small.en": {"size": 244, "multilingual": False, "speed": "medium", "quality": "good"},
        "medium": {"size": 769, "multilingual": True, "speed": "slow", "quality": "high"},
        "medium.en": {"size": 769, "multilingual": False, "speed": "slow", "quality": "high"},
        "large-v1": {"size": 1550, "multilingual": True, "speed": "slowest", "quality": "highest"},
        "large-v2": {"size": 1550, "multilingual": True, "speed": "slowest", "quality": "highest"},
        "large-v3": {"size": 1550, "multilingual": True, "speed": "slowest", "quality": "highest"},
    }

    QUALITY_MODEL_MAPPING = {
        TranscriptionQuality.FAST: ["tiny", "tiny.en"],
        TranscriptionQuality.BALANCED: ["base", "base.en"],
        TranscriptionQuality.HIGH_QUALITY: ["small", "medium"],
        TranscriptionQuality.ULTRA_HIGH: ["large-v3", "large-v2"]
    }

    def __init__(self, container: Container):
        """
        Initialize the enhanced speech recognition system.
        
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
        self._setup_preprocessing()
        self._setup_integrations()
        self._setup_monitoring()
        self._setup_caching()
        self._setup_learning()
        
        # Register health check
        self.health_check.register_component(
            "speech_to_text", 
            self._health_check_callback
        )
        
        self.logger.info(
            f"EnhancedWhisperTranscriber initialized "
            f"(Primary Model: {self.primary_model_name}, Device: {self.device})"
        )

    def _setup_device(self) -> None:
        """Setup compute device and memory management."""
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and 
            self.config.get("speech.device.use_gpu", True) else "cpu"
        )
        
        if self.device.type == "cuda":
            # Setup GPU memory management
            torch.cuda.set_per_process_memory_fraction(
                self.config.get("speech.device.gpu_memory_fraction", 0.7)
            )
            self.gpu_memory_reserved = True
        else:
            self.gpu_memory_reserved = False

    def _setup_audio_config(self) -> None:
        """Configure comprehensive audio settings."""
        self.sample_rate = self.config.get("speech.input.sample_rate", 16000)
        self.channels = self.config.get("speech.input.channels", 1)
        self.chunk_size = int(self.config.get("speech.input.chunk_size", 1024))
        self.audio_format = np.float32
        
        # Advanced audio settings
        self.max_recording_duration = self.config.get("speech.input.max_duration", 300)
        self.auto_gain_control = self.config.get("speech.input.auto_gain_control", True)
        self.noise_suppression = self.config.get("speech.input.noise_suppression", True)
        
        # Initialize audio processor
        self.audio_processor = AudioProcessor(sample_rate=self.sample_rate)
        
        # Setup temporary directory with proper cleanup
        self.temp_dir = Path(tempfile.gettempdir()) / "ai_assistant_audio" 
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def _setup_models(self) -> None:
        """Setup and load Whisper models with intelligent switching."""
        self.models = {}
        self.model_loading_lock = threading.Lock()
        
        # Primary model configuration
        self.primary_model_name = self.config.get("speech.models.whisper.primary", "base")
        self.fallback_models = self.config.get(
            "speech.models.whisper.fallbacks", 
            ["tiny", "base"]
        )
        
        # Load primary model
        self._load_model(self.primary_model_name, primary=True)
        
        # Model switching configuration
        self.auto_model_switching = self.config.get(
            "speech.models.auto_switching.enabled", 
            True
        )
        self.model_switch_threshold = self.config.get(
            "speech.models.auto_switching.confidence_threshold", 
            0.8
        )

    def _load_model(self, model_name: str, primary: bool = False) -> None:
        """
        Load a specific Whisper model with error handling.
        
        Args:
            model_name: Name of the model to load
            primary: Whether this is the primary model
        """
        try:
            if model_name not in self.AVAILABLE_MODELS:
                valid_models = ", ".join(self.AVAILABLE_MODELS.keys())
                raise ValueError(f"Invalid model name. Available models: {valid_models}")

            with self.model_loading_lock:
                if model_name not in self.models:
                    self.logger.info(f"Loading Whisper model: {model_name}")
                    
                    model = whisper.load_model(model_name).to(self.device)
                    self.models[model_name] = {
                        "model": model,
                        "info": self.AVAILABLE_MODELS[model_name],
                        "load_time": datetime.now(timezone.utc),
                        "usage_count": 0
                    }
                    
                    if primary:
                        self.primary_model = model
                        self.multilingual = self.AVAILABLE_MODELS[model_name]["multilingual"]
                        
                    self.logger.info(f"Successfully loaded model: {model_name}")
                    
        except Exception as e:
            error_msg = f"Failed to load Whisper model {model_name}: {str(e)}"
            self.logger.error(error_msg)
            
            if primary:
                # Try fallback models
                for fallback in self.fallback_models:
                    if fallback != model_name:
                        try:
                            self._load_model(fallback, primary=True)
                            self.primary_model_name = fallback
                            self.logger.warning(f"Using fallback model: {fallback}")
                            return
                        except Exception:
                            continue
                            
                raise RuntimeError(f"Failed to load any Whisper model: {error_msg}")

    def _setup_preprocessing(self) -> None:
        """Configure advanced audio preprocessing."""
        preprocessing_config = self.config.get("speech.preprocessing", {})
        
        self.trim_enabled = preprocessing_config.get("trim_silence", True)
        self.normalize_enabled = preprocessing_config.get("normalize", True)
        self.noise_reduction_enabled = preprocessing_config.get("noise_reduction", True)
        self.trim_threshold_db = preprocessing_config.get("trim_threshold_db", -50.0)
        self.noise_reduction_strength = preprocessing_config.get("noise_reduction_strength", 0.5)
        
        # Advanced preprocessing
        self.dynamic_range_compression = preprocessing_config.get("dynamic_range_compression", False)
        self.spectral_gating = preprocessing_config.get("spectral_gating", True)
        self.adaptive_filtering = preprocessing_config.get("adaptive_filtering", True)

    def _setup_integrations(self) -> None:
        """Setup integrations with other AI assistant components."""
        # Memory integration
        self.memory_manager = self.container.get(MemoryManager)
        self.context_manager = self.container.get(ContextManager)
        
        # Natural language processing
        self.entity_extractor = self.container.get_optional(EntityExtractor)
        self.sentiment_analyzer = self.container.get_optional(SentimentAnalyzer)
        
        # Multimodal fusion
        self.fusion_strategy = self.container.get_optional(MultimodalFusionStrategy)
        
        # Learning components
        self.feedback_processor = self.container.get_optional(FeedbackProcessor)
        self.preference_learner = self.container.get_optional(PreferenceLearner)

    def _setup_monitoring(self) -> None:
        """Setup comprehensive monitoring and metrics collection."""
        self.metrics = self.container.get(MetricsCollector)
        self.tracer = self.container.get(TraceManager)
        
        # Initialize metrics
        self.metrics.register_counter("speech_transcriptions_total")
        self.metrics.register_histogram("speech_transcription_duration_seconds")
        self.metrics.register_gauge("speech_transcription_confidence")
        self.metrics.register_counter("speech_transcription_errors_total")

    def _setup_caching(self) -> None:
        """Setup intelligent caching for transcription results."""
        self.cache_manager = self.container.get(CacheManager)
        self.cache_strategy = self.container.get(CacheStrategy)
        
        # Cache configuration
        self.cache_enabled = self.config.get("speech.caching.enabled", True)
        self.cache_ttl = self.config.get("speech.caching.ttl_seconds", 3600)
        self.cache_max_size = self.config.get("speech.caching.max_size_mb", 100)

    def _setup_learning(self) -> None:
        """Setup learning and adaptation capabilities."""
        self.learning_enabled = self.config.get("speech.learning.enabled", True)
        self.adaptation_threshold = self.config.get("speech.learning.adaptation_threshold", 0.1)
        
        # User preference tracking
        self.user_preferences = {}
        self.quality_feedback_history = []

    @handle_exceptions
    async def transcribe(
        self,
        audio: Union[np.ndarray, str, Path],
        request: Optional[TranscriptionRequest] = None
    ) -> TranscriptionResult:
        """
        Advanced transcription with comprehensive AI integration.
        
        Args:
            audio: Audio data or path to audio file
            request: Detailed transcription request configuration
            
        Returns:
            Comprehensive transcription result
        """
        start_time = datetime.now(timezone.utc)
        
        # Default request if not provided
        if request is None:
            request = TranscriptionRequest(
                audio_source=AudioSource.FILE if isinstance(audio, (str, Path)) else AudioSource.BUFFER
            )
        
        # Generate session ID if not provided
        if not request.session_id:
            request.session_id = self._generate_session_id()
        
        # Emit processing started event
        await self.event_bus.emit(SpeechProcessingStarted(
            session_id=request.session_id,
            audio_source=request.audio_source.value,
            language=request.language,
            quality=request.quality.value
        ))
        
        try:
            with self.tracer.trace("speech_transcription") as span:
                span.set_attributes({
                    "audio_source": request.audio_source.value,
                    "quality": request.quality.value,
                    "language": request.language or "auto"
                })
                
                # Check cache first
                cache_key = None
                if request.cache_result and self.cache_enabled:
                    cache_key = self._generate_cache_key(audio, request)
                    cached_result = await self._get_cached_result(cache_key)
                    if cached_result:
                        self.logger.info(f"Returning cached transcription for session {request.session_id}")
                        return cached_result
                
                # Process audio
                processed_audio = await self._process_audio_input(audio, request)
                
                # Select optimal model based on request quality
                model_name = self._select_optimal_model(request.quality, request.language)
                
                # Perform transcription
                raw_result = await self._perform_transcription(
                    processed_audio, 
                    model_name, 
                    request
                )
                
                # Enhance result with additional processing
                enhanced_result = await self._enhance_transcription_result(
                    raw_result, 
                    processed_audio, 
                    request
                )
                
                # Calculate processing time and quality metrics
                processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
                quality_metrics = self._calculate_quality_metrics(enhanced_result, processing_time)
                
                # Create comprehensive result
                result = TranscriptionResult(
                    text=enhanced_result["text"],
                    confidence=enhanced_result.get("confidence", 0.0),
                    language=enhanced_result.get("language", "unknown"),
                    segments=enhanced_result.get("segments", []),
                    processing_time=processing_time,
                    quality_metrics=quality_metrics,
                    emotions=enhanced_result.get("emotions"),
                    speaker_info=enhanced_result.get("speaker_info"),
                    entities=enhanced_result.get("entities"),
                    sentiment=enhanced_result.get("sentiment"),
                    audio_features=enhanced_result.get("audio_features"),
                    session_id=request.session_id
                )
                
                # Cache result if enabled
                if cache_key and request.cache_result:
                    await self._cache_result(cache_key, result)
                
                # Store in memory if context-aware
                if request.context_aware:
                    await self._store_transcription_context(result, request)
                
                # Update metrics
                self.metrics.increment("speech_transcriptions_total")
                self.metrics.record("speech_transcription_duration_seconds", processing_time)
                self.metrics.set("speech_transcription_confidence", result.confidence)
                
                # Learn from result if enabled
                if self.learning_enabled:
                    await self._learn_from_transcription(result, request)
                
                # Emit completion event
                await self.event_bus.emit(SpeechProcessingCompleted(
                    session_id=request.session_id,
                    processing_time=processing_time,
                    confidence=result.confidence,
                    word_count=len(result.text.split())
                ))
                
                self.logger.info(
                    f"Transcription completed for session {request.session_id} "
                    f"in {processing_time:.2f}s with confidence {result.confidence:.2f}"
                )
                
                return result
                
        except Exception as e:
            # Emit error event
            await self.event_bus.emit(SpeechProcessingError(
                session_id=request.session_id,
                error_type=type(e).__name__,
                error_message=str(e)
            ))
            
            self.metrics.increment("speech_transcription_errors_total")
            self.logger.error(f"Transcription failed for session {request.session_id}: {str(e)}")
            raise

    async def _process_audio_input(
        self, 
        audio: Union[np.ndarray, str, Path], 
        request: TranscriptionRequest
    ) -> np.ndarray:
        """Process and prepare audio input for transcription."""
        if isinstance(audio, (str, Path)):
            audio_data, file_sr = self.audio_processor.load_audio(
                audio, target_sr=self.sample_rate
            )
        else:
            audio_data = audio
            
        if request.enable_preprocessing:
            audio_data = await self._advanced_preprocessing(audio_data, request)
            
        return audio_data

    async def _advanced_preprocessing(
        self, 
        audio: np.ndarray, 
        request: TranscriptionRequest
    ) -> np.ndarray:
        """Apply advanced audio preprocessing based on configuration and request."""
        processed_audio = audio.copy()
        
        # Standard preprocessing
        if self.trim_enabled:
            processed_audio = self.audio_processor.trim_silence(
                processed_audio, threshold_db=self.trim_threshold_db
            )
            
        if self.normalize_enabled:
            processed_audio = self.audio_processor.normalize_audio(processed_audio)
            
        if self.noise_reduction_enabled:
            processed_audio = self.audio_processor.apply_noise_reduction(
                processed_audio, self.sample_rate
            )
        
        # Advanced preprocessing based on request
        if hasattr(request, 'enable_enhancement') and request.enable_enhancement:
            processed_audio = await self._apply_audio_enhancement(processed_audio)
            
        return processed_audio

    def _select_optimal_model(
        self, 
        quality: TranscriptionQuality, 
        language: Optional[str]
    ) -> str:
        """Select the optimal model based on quality requirements and language."""
        candidate_models = self.QUALITY_MODEL_MAPPING.get(quality, ["base"])
        
        # Filter by language requirements
        if language and language != "en":
            # Prefer multilingual models for non-English
            candidate_models = [
                model for model in candidate_models 
                if self.AVAILABLE_MODELS.get(model, {}).get("multilingual", False)
            ]
        
        # Select best available model
        for model_name in candidate_models:
            if model_name in self.models:
                return model_name
                
        # Load and return first candidate if not already loaded
        if candidate_models:
            model_name = candidate_models[0]
            self._load_model(model_name)
            return model_name
            
        return self.primary_model_name

    async def _perform_transcription(
        self, 
        audio: np.ndarray, 
        model_name: str, 
        request: TranscriptionRequest
    ) -> Dict[str, Any]:
        """Perform the actual transcription using the selected model."""
        temp_file_created = False
        audio_path = None
        
        try:
            # Write to temporary file
            audio_path = self._write_temp_audio(audio)
            temp_file_created = True
            
            # Get model
            model_info = self.models[model_name]
            model = model_info["model"]
            
            # Update usage count
            model_info["usage_count"] += 1
            
            # Configure transcription options
            options = {
                "language": request.language if model_info["info"]["multilingual"] else "en",
                "task": request.task,
                "fp16": torch.cuda.is_available(),
                "temperature": self.config.get("speech.models.whisper.temperature", 0.0),
                "best_of": self.config.get("speech.models.whisper.best_of", 1),
                "beam_size": self.config.get("speech.models.whisper.beam_size", 1),
                "patience": self.config.get("speech.models.whisper.patience", 1.0),
                "length_penalty": self.config.get("speech.models.whisper.length_penalty", 1.0),
                "suppress_tokens": self.config.get("speech.models.whisper.suppress_tokens", "-1"),
                "initial_prompt": await self._get_context_prompt(request),
                "condition_on_previous_text": self.config.get(
                    "speech.models.whisper.condition_on_previous", True
                ),
                "compression_ratio_threshold": self.config.get(
                    "speech.models.whisper.compression_ratio_threshold", 2.4
                ),
                "logprob_threshold": self.config.get(
                    "speech.models.whisper.logprob_threshold", -1.0
                ),
                "no_speech_threshold": self.config.get(
                    "speech.models.whisper.no_speech_threshold", 0.6
                )
            }
            
            # Perform transcription
            result = model.transcribe(str(audio_path), **options)
            
            # Add confidence scoring
            result["confidence"] = self._calculate_confidence_score(result)
            
            return result
            
        finally:
            if temp_file_created and audio_path and audio_path.exists():
                audio_path.unlink(missing_ok=True)
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

    async def _enhance_transcription_result(
        self, 
        raw_result: Dict[str, Any], 
        audio: np.ndarray, 
        request: TranscriptionRequest
    ) -> Dict[str, Any]:
        """Enhance transcription result with additional AI processing."""
        enhanced_result = raw_result.copy()
        
        # Entity extraction
        if self.entity_extractor and enhanced_result.get("text"):
            try:
                entities = await self.entity_extractor.extract(enhanced_result["text"])
                enhanced_result["entities"] = entities
            except Exception as e:
                self.logger.warning(f"Entity extraction failed: {str(e)}")
        
        # Sentiment analysis
        if self.sentiment_analyzer and enhanced_result.get("text"):
            try:
                sentiment = await self.sentiment_analyzer.analyze(enhanced_result["text"])
                enhanced_result["sentiment"] = sentiment
            except Exception as e:
                self.logger.warning(f"Sentiment analysis failed: {str(e)}")
        
        # Audio feature extraction
        enhanced_result["audio_features"] = self._extract_audio_features(audio)
        
        # Emotion detection (if enabled and available)
        if request.enable_emotion_detection:
            try:
                emotions = await self._detect_emotions(audio)
                enhanced_result["emotions"] = emotions
            except Exception as e:
                self.logger.warning(f"Emotion detection failed: {str(e)}")
        
        # Speaker identification (if enabled and available)
        if request.enable_speaker_identification:
            try:
                speaker_info = await self._identify_speaker(audio)
                enhanced_result["speaker_info"] = speaker_info
            except Exception as e:
                self.logger.warning(f"Speaker identification failed: {str(e)}")
        
        return enhanced_result

    async def listen_continuous(
        self,
        callback: Callable[[TranscriptionResult], None],
        request: Optional[TranscriptionRequest] = None,
        stop_event: Optional[asyncio.Event] = None
    ) -> AsyncGenerator[TranscriptionResult, None]:
        """
        Continuous speech recognition with real-time transcription.
        
        Args:
            callback: Function to call with each transcription result
            request: Transcription request configuration
            stop_event: Event to signal stopping
            
        Yields:
            Transcription results as they become available
        """
        if request is None:
            request = TranscriptionRequest(audio_source=AudioSource.MICROPHONE)
        
        stop_event = stop_event or asyncio.Event()
        audio_buffer = []
        
        def audio_callback(indata: np.ndarray, frames: int, time, status):
            if status:
                self.logger.warning(f"Audio input status: {status}")
            audio_buffer.append(indata.copy())
        
        try:
            # Emit recording started event
            await self.event_bus.emit(AudioRecordingStarted(
                session_id=request.session_id or self._generate_session_id(),
                sample_rate=self.sample_rate,
                channels=self.channels
            ))
            
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=self.audio_format,
                blocksize=self.chunk_size,
                callback=audio_callback
            ):
                while not stop_event.is_set():
                    if len(audio_buffer) >= self.sample_rate // self.chunk_size:  # 1 second of audio
                        # Process accumulated audio
                        audio_segment = np.concatenate(audio_buffer)
                        audio_buffer.clear()
                        
                        # Transcribe segment
                        try:
                            result = await self.transcribe(audio_segment, request)
                            if result.text.strip():  # Only yield non-empty results
                                if callback:
                                    callback(result)
                                yield result
                        except Exception as e:
                            self.logger.error(f"Continuous transcription error: {str(e)}")
                    
                    await asyncio.sleep(0.1)
                    
        finally:
            # Emit recording completed event
            await self.event_bus.emit(AudioRecordingCompleted(
                session_id=request.session_id or "unknown",
                duration=len(audio_buffer) * self.chunk_size / self.sample_rate
            ))

    async def batch_transcribe(
        self,
        audio_files: List[Union[str, Path]],
        request: Optional[TranscriptionRequest] = None,
        max_concurrent: int = 3
    ) -> List[TranscriptionResult]:
        """
        Batch transcribe multiple audio files with concurrency control.
        
        Args:
            audio_files: List of paths to audio files
            request: Base transcription request configuration
            max_concurrent: Maximum number of concurrent transcriptions
            
        Returns:
            List of transcription results
        """
        if request is None:
            request = TranscriptionRequest(audio_source=AudioSource.FILE)
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def transcribe_single(file_path: Union[str, Path]) -> TranscriptionResult:
            async with semaphore:
                file_request = TranscriptionRequest(
                    audio_source=AudioSource.FILE,
                    language=request.language,
                    task=request.task,
                    quality=request.quality,
                    enable_preprocessing=request.enable_preprocessing,
                    enable_emotion_detection=request.enable_emotion_detection,
                    enable_speaker_identification=request.enable_speaker_identification,
                    context_aware=request.context_aware,
                    cache_result=request.cache_result,
                    session_id=f"{request.session_id}_{hash(str(file_path))}",
                    user_id=request.user_id,
                    metadata={**(request.metadata or {}), "file_path": str(file_path)}
                )
                return await self.transcribe(file_path, file_request)
        
        # Process all files concurrently
        tasks = [transcribe_single(file_path) for file_path in audio_files]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions and return valid results
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"Failed to transcribe {audio_files[i]}: {str(result)}")
            else:
                valid_results.append(result)
        
        return valid_results

    def _generate_session_id(self) -> str:
        """Generate a unique session identifier."""
        timestamp = datetime.now(timezone.utc).isoformat()
        return hashlib.md5(f"{timestamp}_{id(self)}".encode()).hexdigest()[:16]

    def _generate_cache_key(
        self, 
        audio: Union[np.ndarray, str, Path], 
        request: TranscriptionRequest
    ) -> str:
        """Generate a cache key for the transcription request."""
        # Create hash of audio content
        if isinstance(audio, (str, Path)):
            audio_hash = hashlib.md5(Path(audio).read_bytes()).hexdigest()
        else:
            audio_hash = hashlib.md5(audio.tobytes()).hexdigest()
        
        # Create hash of request parameters
        request_dict = asdict(request)
        request_dict.pop("session_id", None)  # Exclude session-specific data
        request_dict.pop("metadata", None)
        request_str = json.dumps(request_dict, sort_keys=True)
        request_hash = hashlib.md5(request_str.encode()).hexdigest()
        
        return f"stt:{audio_hash[:16]}:{request_hash[:16]}"

    async def _get_cached_result(self, cache_key: str) -> Optional[TranscriptionResult]:
        """Retrieve cached transcription result."""
        try:
            cached_data = await self.cache_manager.get(cache_key)
            if cached_data:
                return TranscriptionResult(**cached_data)
        except Exception as e:
            self.logger.warning(f"Cache retrieval failed: {str(e)}")
        return None

    async def _cache_result(self, cache_key: str, result: TranscriptionResult) -> None:
        """Cache transcription result."""
        try:
            await self.cache_manager.set(
                cache_key, 
                asdict(result), 
                ttl=self.cache_ttl
            )
        except Exception as e:
            self.logger.warning(f"Cache storage failed: {str(e)}")

    async def _get_context_prompt(self, request: TranscriptionRequest) -> Optional[str]:
        """Get context-aware prompt for transcription."""
        if not request.context_aware or not request.session_id:
            return None
        
        try:
            # Retrieve recent transcription context
            context = await self.context_manager.get_context(
                request.session_id, 
                context_type="speech_transcription"
            )
            
            if context and context.get("recent_text"):
                return context["recent_text"][-200:]  # Last 200 characters
                
        except Exception as e:
            self.logger.warning(f"Context retrieval failed: {str(e)}")
        
        return None

    async def _store_transcription_context(
        self, 
        result: TranscriptionResult, 
        request: TranscriptionRequest
    ) -> None:
        """Store transcription result in context memory."""
        try:
            context_data = {
                "text": result.text,
                "timestamp": result.timestamp.isoformat(),
                "confidence": result.confidence,
                "language": result.language
            }
            
            await self.context_manager.update_context(
                request.session_id,
                "speech_transcription",
                context_data
            )
            
        except Exception as e:
            self.logger.warning(f"Context storage failed: {str(e)}")

    def _write_temp_audio(self, audio: np.ndarray) -> Path:
        """Write audio data to a temporary file."""
        timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
        temp_path = self.temp_dir / f"audio_{timestamp}.wav"
        sf.write(temp_path, audio, self.sample_rate)
        return temp_path

    def _calculate_confidence_score(self, result: Dict[str, Any]) -> float:
        """Calculate confidence score from transcription result."""
        if "segments" in result and result["segments"]:
            # Average confidence from segments
            total_conf = sum(seg.get("avg_logprob", 0) for seg in result["segments"])
            avg_conf = total_conf / len(result["segments"])
            # Convert log probability to confidence score (0-1)
            return max(0.0, min(1.0, np.exp(avg_conf)))
        return 0.0

    def _calculate_quality_metrics(
        self, 
        result: Dict[str, Any], 
        processing_time: float
    ) -> Dict[str, float]:
        """Calculate comprehensive quality metrics."""
        text = result.get("text", "")
        segments = result.get("segments", [])
        
        metrics = {
            "processing_time": processing_time,
            "words_per_second": len(text.split()) / processing_time if processing_time > 0 else 0,
            "segment_count": len(segments),
            "average_segment_confidence": np.mean([
                seg.get("avg_logprob", 0) for seg in segments
            ]) if segments else 0,
            "text_length": len(text),
            "word_count": len(text.split())
        }
        
        return metrics

    def _extract_audio_features(self, audio: np.ndarray) -> Dict[str, Any]:
        """Extract basic audio features for analysis."""
        return {
            "duration": len(audio) / self.sample_rate,
            "rms_energy": float(np.sqrt(np.mean(audio**2))),
            "max_amplitude": float(np.max(np.abs(audio))),
            "zero_crossing_rate": float(np.mean(np.abs(np.diff(np.sign(audio))))),
            "sample_rate": self.sample_rate,
            "sample_count": len(audio)
        }

    async def _detect_emotions(self, audio: np.ndarray) -> Dict[str, float]:
        """Placeholder for emotion detection (to be implemented)."""
        # This would integrate with src/processing/speech/emotion_detection.py
        return {"neutral": 1.0}

    async def _identify_speaker(self, audio: np.ndarray) -> Dict[str, Any]:
        """Placeholder for speaker identification (to be implemented)."""
        # This would integrate with src/processing/speech/speaker_recognition.py
        return {"speaker_id": "unknown", "confidence": 0.0}

    async def _learn_from_transcription(
        self, 
        result: TranscriptionResult, 
        request: TranscriptionRequest
    ) -> None:
        """Learn from transcription results to improve future performance."""
        if not self.learning_enabled or not self.feedback_processor:
            return
        
        try:
            # Create learning data
            learning_data = {
                "transcription_quality": result.confidence,
                "processing_time": result.processing_time,
                "audio_features": result.audio_features,
                "model_used": self.primary_model_name,
                "preprocessing_enabled": request.enable_preprocessing,
                "language": result.language
            }
            
            # Send to feedback processor
            await self.feedback_processor.process_feedback(
                "speech_transcription",
                learning_data,
                request.user_id
            )
            
        except Exception as e:
            self.logger.warning(f"Learning from transcription failed: {str(e)}")

    async def _health_check_callback(self) -> Dict[str, Any]:
        """Health check callback for the speech-to-text system."""
        try:
            # Test basic functionality
            test_audio = np.random.randn(self.sample_rate).astype(np.float32) * 0.01
            start_time = datetime.now(timezone.utc)
            
            # Perform a quick test transcription
            test_request = TranscriptionRequest(
                audio_source=AudioSource.BUFFER,
                quality=TranscriptionQuality.FAST,
                cache_result=False,
                context_aware=False
            )
            
            result = await self.transcribe(test_audio, test_request)
            response_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            return {
                "status": "healthy",
                "models_loaded": len(self.models),
                "primary_model": self.primary_model_name,
                "device": str(self.device),
                "response_time_ms": response_time * 1000,
                "cache_enabled": self.cache_enabled,
                "learning_enabled": self.learning_enabled
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "models_loaded": len(self.models) if hasattr(self, 'models') else 0
            }

    def get_available_languages(self) -> List[str]:
        """Get supported languages for the current model setup."""
        if self.multilingual:
            return list(whisper.tokenizer.LANGUAGES.keys())
        return ["en"]

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models."""
        return {
            "primary_model": self.primary_model_name,
            "loaded_models": {
                name: {
                    "info": model_data["info"],
                    "load_time": model_data["load_time"].isoformat(),
                    "usage_count": model_data["usage_count"]
                }
                for name, model_data in self.models.items()
            },
            "device": str(self.device),
            "multilingual_support": self.multilingual
        }

    @asynccontextmanager
    async def recording_session(
        self, 
        request: Optional[TranscriptionRequest] = None
    ):
        """Context manager for recording sessions with proper cleanup."""
        session_id = self._generate_session_id()
        if request:
            request.session_id = session_id
        
        try:
            await self.event_bus.emit(AudioRecordingStarted(
                session_id=session_id,
                sample_rate=self.sample_rate,
                channels=self.channels
            ))
            yield session_id
        finally:
            await self.event_bus.emit(AudioRecordingCompleted(
                session_id=session_id,
                duration=0.0  # Would be calculated in actual implementation
            ))

    async def cleanup(self) -> None:
        """Comprehensive cleanup of resources."""
        self.logger.info("Starting EnhancedWhisperTranscriber cleanup...")
        
        # Clean up temporary files
        if hasattr(self, 'temp_dir') and self.temp_dir.exists():
            for file in self.temp_dir.glob('*.wav'):
                try:
                    file.unlink()
                except Exception as e:
                    self.logger.warning(f"Failed to delete temporary file {file}: {str(e)}")
            
            try:
                self.temp_dir.rmdir()
            except Exception as e:
                self.logger.warning(f"Failed to remove temporary directory: {str(e)}")
        
        # Clean up audio processor
        if hasattr(self, 'audio_processor'):
            self.audio_processor.cleanup()
        
        # Clean up models and GPU memory
        if hasattr(self, 'models'):
            for model_name, model_data in self.models.items():
                try:
                    del model_data["model"]
                    self.logger.debug(f"Cleaned up model: {model_name}")
                except Exception as e:
                    self.logger.warning(f"Failed to cleanup model {model_name}: {str(e)}")
        
        # Clear GPU memory
        if self.device.type == "cuda" and self.gpu_memory_reserved:
            torch.cuda.empty_cache()
            self.logger.debug("Cleared CUDA cache")
        
        # Unregister health check
        if hasattr(self, 'health_check'):
            self.health_check.unregister_component("speech_to_text")
        
        self.logger.info("EnhancedWhisperTranscriber cleanup completed")

    def __del__(self):
        """Destructor to ensure cleanup on object deletion."""
        try:
            if hasattr(self, 'temp_dir') and self.temp_dir.exists():
                for file in self.temp_dir.glob('*.wav'):
                    file.unlink(missing_ok=True)
        except Exception:
            pass  # Ignore cleanup errors in destructor
