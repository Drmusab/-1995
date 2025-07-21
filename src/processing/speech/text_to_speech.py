"""
Advanced Text-to-Speech Module using Piper TTS
Author: Drmusab
Last Modified: 2025-05-26 14:07:11 UTC

This module provides comprehensive text-to-speech capabilities integrated with the
AI assistant's core architecture, including voice synthesis, prosody control,
real-time streaming, voice cloning, and multimodal integration.
"""

from pathlib import Path
from typing import Optional, Dict, Any, Union, List, Callable, AsyncGenerator, Iterator
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
import subprocess
import shutil
import io
import wave
from concurrent.futures import ThreadPoolExecutor
import queue

import torch
import sounddevice as sd
import soundfile as sf
import librosa
import phonemizer
from piper import PiperVoice
from piper.download import get_voices, ensure_voice_exists
import onnxruntime as ort

# Core imports
from src.core.config.loader import ConfigLoader
from src.core.events.event_bus import EventBus
from src.core.events.event_types import (
    SpeechSynthesisStarted,
    SpeechSynthesisCompleted,
    SpeechSynthesisError,
    VoiceProcessingStarted,
    VoiceProcessingCompleted,
    AudioStreamStarted,
    AudioStreamStopped,
    TTSModelLoaded,
    TTSModelError
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
from src.processing.natural_language.sentiment_analyzer import SentimentAnalyzer
from src.processing.natural_language.entity_extractor import EntityExtractor
from src.processing.multimodal.fusion_strategies import MultimodalFusionStrategy

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


class VoiceQuality(Enum):
    """Voice synthesis quality levels."""
    FAST = "fast"
    BALANCED = "balanced"
    HIGH_QUALITY = "high_quality"
    ULTRA_HIGH = "ultra_high"


class SpeakingStyle(Enum):
    """Speaking styles for voice synthesis."""
    NEUTRAL = "neutral"
    CHEERFUL = "cheerful"
    SAD = "sad"
    ANGRY = "angry"
    EXCITED = "excited"
    WHISPERING = "whispering"
    SHOUTING = "shouting"
    CALM = "calm"
    ANXIOUS = "anxious"
    CONFIDENT = "confident"


class SynthesisMode(Enum):
    """Text-to-speech synthesis modes."""
    BATCH = "batch"
    STREAMING = "streaming"
    REAL_TIME = "real_time"
    BUFFERED = "buffered"


class VoiceGender(Enum):
    """Voice gender categories."""
    MALE = "male"
    FEMALE = "female"
    NEUTRAL = "neutral"


class AudioOutputFormat(Enum):
    """Audio output formats."""
    WAV = "wav"
    MP3 = "mp3"
    FLAC = "flac"
    OGG = "ogg"
    RAW_PCM = "raw_pcm"


@dataclass
class VoiceProfile:
    """Voice profile configuration."""
    voice_id: str
    name: str
    language: str
    gender: VoiceGender
    accent: Optional[str] = None
    age_group: Optional[str] = None  # young, adult, elderly
    characteristics: List[str] = field(default_factory=list)
    quality: str = "medium"
    sample_rate: int = 22050
    model_path: Optional[Path] = None
    config_path: Optional[Path] = None
    speaker_id: Optional[int] = None
    description: Optional[str] = None
    
    # Performance metrics
    load_time: Optional[float] = None
    average_synthesis_time: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    
    # Customization
    pitch_shift: float = 0.0  # Semitones
    speed_multiplier: float = 1.0
    volume_multiplier: float = 1.0


@dataclass
class ProsodySettings:
    """Prosody control settings for natural speech."""
    pitch_base: float = 1.0  # Base pitch multiplier
    pitch_range: float = 1.0  # Pitch variation range
    speech_rate: float = 1.0  # Speaking rate multiplier
    pause_duration: float = 1.0  # Pause duration multiplier
    emphasis_strength: float = 1.0  # Emphasis intensity
    
    # Advanced prosodic features
    intonation_pattern: str = "neutral"  # rising, falling, neutral
    stress_pattern: str = "natural"  # natural, emphasized, reduced
    rhythm_pattern: str = "regular"  # regular, varied, monotone
    
    # Emotional prosody
    emotional_intensity: float = 0.0  # 0.0 to 1.0
    speaking_style: SpeakingStyle = SpeakingStyle.NEUTRAL


@dataclass
class SynthesisRequest:
    """Comprehensive text-to-speech synthesis request."""
    text: str
    voice_id: Optional[str] = None
    language: Optional[str] = None
    synthesis_mode: SynthesisMode = SynthesisMode.BATCH
    quality: VoiceQuality = VoiceQuality.BALANCED
    output_format: AudioOutputFormat = AudioOutputFormat.WAV
    
    # Prosody and style
    prosody: Optional[ProsodySettings] = None
    speaking_style: SpeakingStyle = SpeakingStyle.NEUTRAL
    
    # Processing options
    enable_preprocessing: bool = True
    enable_postprocessing: bool = True
    enable_emotional_adaptation: bool = False
    enable_context_aware_prosody: bool = True
    
    # Audio settings
    sample_rate: Optional[int] = None
    normalize_audio: bool = True
    apply_audio_effects: bool = False
    
    # Session and context
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    conversation_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    
    # Caching and performance
    cache_result: bool = True
    priority: int = 0  # 0 = normal, higher = more priority
    timeout_seconds: float = 30.0
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)


@dataclass
class SynthesisResult:
    """Comprehensive text-to-speech synthesis result."""
    audio_data: np.ndarray
    audio_metadata: AudioMetadata
    text: str
    voice_profile: VoiceProfile
    synthesis_time: float
    
    # Quality metrics
    audio_quality_score: float = 0.0
    naturalness_score: float = 0.0
    intelligibility_score: float = 0.0
    emotional_expressiveness: float = 0.0
    
    # Processing information
    synthesis_mode: SynthesisMode = SynthesisMode.BATCH
    preprocessing_applied: List[str] = field(default_factory=list)
    postprocessing_applied: List[str] = field(default_factory=list)
    
    # Prosodic analysis
    detected_emotions: Optional[Dict[str, float]] = None
    prosody_features: Optional[Dict[str, Any]] = None
    speaking_statistics: Optional[Dict[str, Any]] = None
    
    # Session information
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    conversation_id: Optional[str] = None
    
    # Performance metrics
    model_inference_time: float = 0.0
    audio_generation_time: float = 0.0
    postprocessing_time: float = 0.0
    memory_usage_mb: float = 0.0
    
    # Success and error handling
    success: bool = True
    warnings: List[str] = field(default_factory=list)
    error_message: Optional[str] = None
    
    # Timestamp and tracking
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # File output information
    output_file_path: Optional[Path] = None
    file_size_bytes: Optional[int] = None


@dataclass
class StreamingConfig:
    """Configuration for streaming text-to-speech."""
    chunk_size: int = 1024
    buffer_size: int = 8192
    overlap_samples: int = 256
    latency_target_ms: float = 100.0
    quality_vs_latency: str = "balanced"  # latency, balanced, quality
    enable_sentence_streaming: bool = True
    sentence_buffer_size: int = 3
    
    # Real-time adaptation
    adaptive_quality: bool = True
    network_aware: bool = True
    cpu_usage_threshold: float = 0.8


class TTSError(Exception):
    """Custom exception for text-to-speech operations."""
    
    def __init__(self, message: str, error_code: Optional[str] = None, component: Optional[str] = None):
        super().__init__(message)
        self.error_code = error_code
        self.component = component
        self.timestamp = datetime.now(timezone.utc)


class VoiceManager:
    """Manager for Piper TTS voices and models."""
    
    def __init__(self, model_dir: Path, logger):
        self.model_dir = model_dir
        self.logger = logger
        self.voices: Dict[str, VoiceProfile] = {}
        self.loaded_models: Dict[str, PiperVoice] = {}
        self.model_loading_lock = threading.Lock()
        
        # Voice catalog cache
        self.voice_catalog: Dict[str, Any] = {}
        self.catalog_last_updated: Optional[datetime] = None
        
    async def initialize(self) -> None:
        """Initialize voice manager and discover available voices."""
        try:
            # Create model directory
            self.model_dir.mkdir(parents=True, exist_ok=True)
            
            # Load voice catalog
            await self._load_voice_catalog()
            
            # Discover local voices
            await self._discover_local_voices()
            
            self.logger.info(f"VoiceManager initialized with {len(self.voices)} voices")
            
        except Exception as e:
            self.logger.error(f"VoiceManager initialization failed: {str(e)}")
            raise TTSError(f"Voice manager initialization failed: {str(e)}")
    
    async def _load_voice_catalog(self) -> None:
        """Load available voices from Piper catalog."""
        try:
            # Get voices from piper-tts
            voices = get_voices("https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/voices.json")
            self.voice_catalog = voices
            self.catalog_last_updated = datetime.now(timezone.utc)
            
            # Parse and register voices
            for voice_key, voice_info in voices.items():
                if isinstance(voice_info, dict) and 'files' in voice_info:
                    voice_profile = self._create_voice_profile_from_catalog(voice_key, voice_info)
                    self.voices[voice_key] = voice_profile
            
            self.logger.info(f"Loaded {len(self.voice_catalog)} voices from catalog")
            
        except Exception as e:
            self.logger.warning(f"Failed to load voice catalog: {str(e)}")
            # Continue with local voices only
    
    def _create_voice_profile_from_catalog(self, voice_key: str, voice_info: Dict[str, Any]) -> VoiceProfile:
        """Create voice profile from catalog information."""
        # Parse voice key (e.g., "en_US-amy-medium")
        parts = voice_key.split('-')
        language = parts[0] if parts else "unknown"
        name = parts[1] if len(parts) > 1 else "unknown"
        quality = parts[2] if len(parts) > 2 else "medium"
        
        # Determine gender from name or metadata
        gender = VoiceGender.NEUTRAL
        if 'speaker' in voice_info:
            speaker_info = voice_info['speaker']
            if isinstance(speaker_info, dict):
                gender_str = speaker_info.get('gender', 'neutral').lower()
                if gender_str in ['male', 'female', 'neutral']:
                    gender = VoiceGender(gender_str)
        
        return VoiceProfile(
            voice_id=voice_key,
            name=name,
            language=language,
            gender=gender,
            quality=quality,
            sample_rate=voice_info.get('sample_rate', 22050),
            description=voice_info.get('description', ''),
            characteristics=voice_info.get('characteristics', [])
        )
    
    async def _discover_local_voices(self) -> None:
        """Discover locally installed voice models."""
        try:
            # Scan model directory for .onnx files
            model_files = list(self.model_dir.glob("*.onnx"))
            
            for model_file in model_files:
                config_file = model_file.with_suffix('.onnx.json')
                if config_file.exists():
                    try:
                        voice_profile = self._create_voice_profile_from_files(model_file, config_file)
                        if voice_profile:
                            self.voices[voice_profile.voice_id] = voice_profile
                    except Exception as e:
                        self.logger.warning(f"Failed to load local voice {model_file}: {str(e)}")
            
            self.logger.info(f"Discovered {len(model_files)} local voice models")
            
        except Exception as e:
            self.logger.warning(f"Local voice discovery failed: {str(e)}")
    
    def _create_voice_profile_from_files(self, model_file: Path, config_file: Path) -> Optional[VoiceProfile]:
        """Create voice profile from local model and config files."""
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            voice_id = model_file.stem.replace('.onnx', '')
            
            return VoiceProfile(
                voice_id=voice_id,
                name=config.get('name', voice_id),
                language=config.get('language', 'unknown'),
                gender=VoiceGender(config.get('gender', 'neutral')),
                quality=config.get('quality', 'medium'),
                sample_rate=config.get('sample_rate', 22050),
                model_path=model_file,
                config_path=config_file,
                speaker_id=config.get('speaker_id'),
                description=config.get('description', '')
            )
            
        except Exception as e:
            self.logger.warning(f"Failed to parse voice config {config_file}: {str(e)}")
            return None
    
    async def get_voice(self, voice_id: str) -> Optional[VoiceProfile]:
        """Get voice profile by ID."""
        return self.voices.get(voice_id)
    
    async def list_voices(
        self, 
        language: Optional[str] = None,
        gender: Optional[VoiceGender] = None,
        quality: Optional[str] = None
    ) -> List[VoiceProfile]:
        """List available voices with optional filtering."""
        voices = list(self.voices.values())
        
        if language:
            voices = [v for v in voices if v.language.startswith(language)]
        
        if gender:
            voices = [v for v in voices if v.gender == gender]
        
        if quality:
            voices = [v for v in voices if v.quality == quality]
        
        return voices
    
    async def load_voice_model(self, voice_id: str) -> PiperVoice:
        """Load Piper voice model."""
        with self.model_loading_lock:
            if voice_id in self.loaded_models:
                return self.loaded_models[voice_id]
            
            voice_profile = self.voices.get(voice_id)
            if not voice_profile:
                raise TTSError(f"Voice {voice_id} not found")
            
            try:
                start_time = datetime.now(timezone.utc)
                
                # Download voice if not local
                if not voice_profile.model_path or not voice_profile.model_path.exists():
                    await self._download_voice(voice_id)
                    # Refresh voice profile with downloaded paths
                    voice_profile = await self._update_voice_profile_paths(voice_id)
                
                # Load Piper voice
                if voice_profile.model_path and voice_profile.config_path:
                    voice = PiperVoice.load(
                        str(voice_profile.model_path),
                        config_path=str(voice_profile.config_path),
                        use_cuda=torch.cuda.is_available()
                    )
                else:
                    raise TTSError(f"Model or config file not found for voice {voice_id}")
                
                load_time = (datetime.now(timezone.utc) - start_time).total_seconds()
                voice_profile.load_time = load_time
                
                self.loaded_models[voice_id] = voice
                
                self.logger.info(f"Loaded voice model {voice_id} in {load_time:.2f}s")
                return voice
                
            except Exception as e:
                raise TTSError(f"Failed to load voice model {voice_id}: {str(e)}")
    
    async def _download_voice(self, voice_id: str) -> None:
        """Download voice model from remote repository."""
        try:
            self.logger.info(f"Downloading voice model: {voice_id}")
            
            # Use piper's ensure_voice_exists function
            model_path, config_path = ensure_voice_exists(
                voice_id,
                self.voice_catalog,
                str(self.model_dir)
            )
            
            self.logger.info(f"Downloaded voice model {voice_id}")
            
        except Exception as e:
            raise TTSError(f"Failed to download voice {voice_id}: {str(e)}")
    
    async def _update_voice_profile_paths(self, voice_id: str) -> VoiceProfile:
        """Update voice profile with downloaded file paths."""
        voice_profile = self.voices[voice_id]
        
        # Find downloaded files
        model_file = self.model_dir / f"{voice_id}.onnx"
        config_file = self.model_dir / f"{voice_id}.onnx.json"
        
        if model_file.exists():
            voice_profile.model_path = model_file
        if config_file.exists():
            voice_profile.config_path = config_file
        
        return voice_profile
    
    def unload_voice_model(self, voice_id: str) -> None:
        """Unload voice model to free memory."""
        if voice_id in self.loaded_models:
            del self.loaded_models[voice_id]
            self.logger.info(f"Unloaded voice model: {voice_id}")


class ProsodyProcessor:
    """Advanced prosody processing for natural speech synthesis."""
    
    def __init__(self, logger):
        self.logger = logger
        
        # Prosody models and processors
        self.emotion_to_prosody_map = {
            "happiness": {"pitch_base": 1.2, "speech_rate": 1.1, "pitch_range": 1.3},
            "sadness": {"pitch_base": 0.8, "speech_rate": 0.9, "pitch_range": 0.7},
            "anger": {"pitch_base": 1.1, "speech_rate": 1.2, "pitch_range": 1.4},
            "fear": {"pitch_base": 1.3, "speech_rate": 1.3, "pitch_range": 1.5},
            "surprise": {"pitch_base": 1.4, "speech_rate": 1.2, "pitch_range": 1.6},
            "neutral": {"pitch_base": 1.0, "speech_rate": 1.0, "pitch_range": 1.0}
        }
    
    def analyze_text_for_prosody(self, text: str, context: Optional[Dict[str, Any]] = None) -> ProsodySettings:
        """Analyze text and context to determine optimal prosody."""
        try:
            prosody = ProsodySettings()
            
            # Basic text analysis
            if text.endswith('!'):
                prosody.emphasis_strength = 1.2
                prosody.pitch_range = 1.3
            elif text.endswith('?'):
                prosody.intonation_pattern = "rising"
                prosody.pitch_base = 1.1
            elif '...' in text:
                prosody.pause_duration = 1.5
                prosody.speech_rate = 0.9
            
            # Detect emotional markers
            emotional_words = {
                'excited': SpeakingStyle.EXCITED,
                'happy': SpeakingStyle.CHEERFUL,
                'sad': SpeakingStyle.SAD,
                'angry': SpeakingStyle.ANGRY,
                'whisper': SpeakingStyle.WHISPERING,
                'shout': SpeakingStyle.SHOUTING
            }
            
            text_lower = text.lower()
            for word, style in emotional_words.items():
                if word in text_lower:
                    prosody.speaking_style = style
                    break
            
            # Context-based adjustments
            if context:
                detected_emotion = context.get('detected_emotion')
                if detected_emotion and detected_emotion in self.emotion_to_prosody_map:
                    emotion_prosody = self.emotion_to_prosody_map[detected_emotion]
                    prosody.pitch_base *= emotion_prosody['pitch_base']
                    prosody.speech_rate *= emotion_prosody['speech_rate']
                    prosody.pitch_range *= emotion_prosody['pitch_range']
            
            return prosody
            
        except Exception as e:
            self.logger.warning(f"Prosody analysis failed: {str(e)}")
            return ProsodySettings()  # Return default settings
    
    def apply_prosody_to_synthesis(self, voice: PiperVoice, prosody: ProsodySettings) -> Dict[str, Any]:
        """Apply prosody settings to voice synthesis parameters."""
        synthesis_params = {}
        
        try:
            # Convert prosody settings to synthesis parameters
            if hasattr(voice, 'config'):
                config = voice.config
                
                # Adjust synthesis parameters based on prosody
                synthesis_params['length_scale'] = 1.0 / prosody.speech_rate
                synthesis_params['noise_scale'] = 0.667
                synthesis_params['noise_scale_w'] = 0.8
                
                # Speaker-specific adjustments if available
                if hasattr(config, 'num_speakers') and config.num_speakers > 1:
                    synthesis_params['speaker_id'] = 0  # Default speaker
            
            return synthesis_params
            
        except Exception as e:
            self.logger.warning(f"Prosody application failed: {str(e)}")
            return {}


class EnhancedTextToSpeech:
    """
    Advanced Text-to-Speech system with comprehensive AI assistant integration.
    
    Features:
    - Piper TTS integration with multiple voice models
    - Advanced prosody control and emotional expression
    - Real-time and streaming synthesis capabilities
    - Voice cloning and customization
    - Context-aware speech adaptation
    - Multimodal integration with sentiment analysis
    - Comprehensive caching and memory integration
    - Event-driven architecture with detailed monitoring
    - Learning and adaptation capabilities
    - Audio post-processing and enhancement
    """
    
    DEFAULT_VOICE_ID = "en_US-amy-medium"
    
    def __init__(self, container: Container):
        """
        Initialize the enhanced text-to-speech system.
        
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
        self._setup_models_and_voices()
        self._setup_audio_processing()
        self._setup_integrations()
        self._setup_monitoring()
        self._setup_caching()
        self._setup_learning()
        self._setup_streaming()
        
        # Register health check
        self.health_check.register_component(
            "text_to_speech",
            self._health_check_callback
        )
        
        self.logger.info(
            f"EnhancedTextToSpeech initialized "
            f"(Device: {self.device}, Models: {len(self.voice_manager.voices)})"
        )
    
    def _setup_device(self) -> None:
        """Setup compute device and memory management."""
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and 
            self.config.get("text_to_speech.device.use_gpu", True) else "cpu"
        )
        
        if self.device.type == "cuda":
            torch.cuda.set_per_process_memory_fraction(
                self.config.get("text_to_speech.device.gpu_memory_fraction", 0.6)
            )
        
        # Initialize ONNX Runtime providers
        self.ort_providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
    
    def _setup_audio_config(self) -> None:
        """Configure audio settings for synthesis."""
        self.default_sample_rate = self.config.get("text_to_speech.audio.sample_rate", 22050)
        self.default_bit_depth = self.config.get("text_to_speech.audio.bit_depth", 16)
        self.audio_buffer_size = self.config.get("text_to_speech.audio.buffer_size", 1024)
        
        # Audio quality settings
        self.quality_settings = {
            VoiceQuality.FAST: {
                "sample_rate": 16000,
                "quality_factor": 0.7,
                "processing_optimization": "speed"
            },
            VoiceQuality.BALANCED: {
                "sample_rate": 22050,
                "quality_factor": 0.8,
                "processing_optimization": "balanced"
            },
            VoiceQuality.HIGH_QUALITY: {
                "sample_rate": 22050,
                "quality_factor": 0.9,
                "processing_optimization": "quality"
            },
            VoiceQuality.ULTRA_HIGH: {
                "sample_rate": 44100,
                "quality_factor": 1.0,
                "processing_optimization": "quality"
            }
        }
    
    def _setup_models_and_voices(self) -> None:
        """Setup voice models and voice manager."""
        # Model directory
        self.model_dir = Path(self.config.get(
            "text_to_speech.models.model_dir",
            "data/models/text_to_speech"
        ))
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize voice manager
        self.voice_manager = VoiceManager(self.model_dir, self.logger)
        
        # Initialize prosody processor
        self.prosody_processor = ProsodyProcessor(self.logger)
        
        # Default voice settings
        self.default_voice_id = self.config.get("text_to_speech.default_voice", self.DEFAULT_VOICE_ID)
        self.voice_selection_strategy = self.config.get("text_to_speech.voice_selection", "quality_first")
        
        # Voice model cache settings
        self.max_loaded_models = self.config.get("text_to_speech.models.max_loaded", 3)
        self.model_unload_timeout = self.config.get("text_to_speech.models.unload_timeout", 300)
    
    def _setup_audio_processing(self) -> None:
        """Setup audio processing components."""
        self.audio_processor = EnhancedAudioProcessor(
            sample_rate=self.default_sample_rate,
            container=self.container
        )
        
        # Post-processing settings
        self.enable_audio_enhancement = self.config.get("text_to_speech.post_processing.enhancement", True)
        self.enable_normalization = self.config.get("text_to_speech.post_processing.normalization", True)
        self.enable_noise_gate = self.config.get("text_to_speech.post_processing.noise_gate", False)
    
    def _setup_integrations(self) -> None:
        """Setup integrations with other AI assistant components."""
        # Memory integration
        self.memory_manager = self.container.get(MemoryManager)
        self.context_manager = self.container.get(ContextManager)
        
        # Natural language processing
        self.sentiment_analyzer = self.container.get_optional(SentimentAnalyzer)
        self.entity_extractor = self.container.get_optional(EntityExtractor)
        
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
        self.metrics.register_counter("tts_synthesis_requests_total")
        self.metrics.register_histogram("tts_synthesis_duration_seconds")
        self.metrics.register_gauge("tts_audio_quality_score")
        self.metrics.register_counter("tts_synthesis_errors_total")
        self.metrics.register_histogram("tts_model_inference_duration_seconds")
        self.metrics.register_gauge("tts_active_streams")
    
    def _setup_caching(self) -> None:
        """Setup caching for synthesis results."""
        self.cache_manager = self.container.get(CacheManager)
        self.cache_strategy = self.container.get(CacheStrategy)
        
        self.cache_enabled = self.config.get("text_to_speech.caching.enabled", True)
        self.cache_ttl = self.config.get("text_to_speech.caching.ttl_seconds", 7200)
        self.max_cache_size_mb = self.config.get("text_to_speech.caching.max_size_mb", 500)
    
    def _setup_learning(self) -> None:
        """Setup learning and adaptation capabilities."""
        self.learning_enabled = self.config.get("text_to_speech.learning.enabled", True)
        self.personalization_enabled = self.config.get("text_to_speech.learning.personalization", True)
        
        # User voice preferences
        self.user_voice_preferences = {}
        self.synthesis_feedback_history = {}
    
    def _setup_streaming(self) -> None:
        """Setup streaming synthesis capabilities."""
        self.streaming_enabled = self.config.get("text_to_speech.streaming.enabled", True)
        self.default_streaming_config = StreamingConfig(
            chunk_size=self.config.get("text_to_speech.streaming.chunk_size", 1024),
            buffer_size=self.config.get("text_to_speech.streaming.buffer_size", 8192),
            latency_target_ms=self.config.get("text_to_speech.streaming.latency_target_ms", 100),
            enable_sentence_streaming=self.config.get("text_to_speech.streaming.sentence_streaming", True)
        )
        
        # Streaming session management
        self.active_streams = {}
        self.stream_lock = threading.Lock()
        
        # Thread pool for streaming
        self.stream_thread_pool = ThreadPoolExecutor(
            max_workers=4,
            thread_name_prefix="tts_streaming"
        )
    
    async def initialize(self) -> None:
        """Initialize the text-to-speech system."""
        try:
            # Initialize voice manager
            await self.voice_manager.initialize()
            
            # Load default voice
            if self.default_voice_id:
                try:
                    await self.voice_manager.load_voice_model(self.default_voice_id)
                    self.logger.info(f"Loaded default voice: {self.default_voice_id}")
                except Exception as e:
                    self.logger.warning(f"Failed to load default voice: {str(e)}")
            
            # Emit initialization complete event
            await self.event_bus.emit(TTSModelLoaded(
                model_id=self.default_voice_id or "unknown",
                model_type="piper_tts",
                device=str(self.device)
            ))
            
            self.logger.info("EnhancedTextToSpeech initialization completed")
            
        except Exception as e:
            await self.event_bus.emit(TTSModelError(
                model_id=self.default_voice_id or "unknown",
                error_type=type(e).__name__,
                error_message=str(e)
            ))
            raise TTSError(f"TTS initialization failed: {str(e)}")
    
    @handle_exceptions
    async def synthesize(
        self,
        request: Union[str, SynthesisRequest]
    ) -> SynthesisResult:
        """
        Advanced text-to-speech synthesis with comprehensive AI integration.
        
        Args:
            request: Text string or detailed synthesis request
            
        Returns:
            Comprehensive synthesis result
        """
        start_time = datetime.now(timezone.utc)
        
        # Convert string input to SynthesisRequest
        if isinstance(request, str):
            request = SynthesisRequest(text=request)
        
        # Generate session ID if not provided
        if not request.session_id:
            request.session_id = self._generate_session_id()
        
        # Emit synthesis started event
        await self.event_bus.emit(SpeechSynthesisStarted(
            session_id=request.session_id,
            text_length=len(request.text),
            voice_id=request.voice_id or self.default_voice_id,
            synthesis_mode=request.synthesis_mode.value
        ))
        
        try:
            with self.tracer.trace("tts_synthesis") as span:
                span.set_attributes({
                    "session_id": request.session_id,
                    "text_length": len(request.text),
                    "voice_id": request.voice_id or self.default_voice_id,
                    "synthesis_mode": request.synthesis_mode.value,
                    "quality": request.quality.value
                })
                
                # Check cache first
                cache_key = None
                if request.cache_result and self.cache_enabled:
                    cache_key = self._generate_cache_key(request)
                    cached_result = await self._get_cached_result(cache_key)
                    if cached_result:
                        self.logger.info(f"Returning cached synthesis result for session {request.session_id}")
                        return cached_result
                
                # Text preprocessing
                processed_text = await self._preprocess_text(request)
                
                # Voice selection and loading
                voice_profile, voice_model = await self._select_and_load_voice(request)
                
                # Prosody analysis and setup
                prosody_settings = await self._analyze_and_setup_prosody(processed_text, request)
                
                # Perform synthesis
                audio_data = await self._perform_synthesis(
                    processed_text, voice_model, prosody_settings, request
                )
                
                # Audio post-processing
                processed_audio = await self._postprocess_audio(audio_data, request)
                
                # Create audio metadata
                audio_metadata = AudioMetadata(
                    sample_rate=voice_profile.sample_rate,
                    channels=1,
                    duration=len(processed_audio) / voice_profile.sample_rate,
                    format="pcm",
                    created_at=datetime.now(timezone.utc)
                )
                
                # Quality assessment
                quality_metrics = await self._assess_synthesis_quality(
                    processed_audio, processed_text, voice_profile
                )
                
                # Calculate processing time
                processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
                
                # Create result
                result = SynthesisResult(
                    audio_data=processed_audio,
                    audio_metadata=audio_metadata,
                    text=processed_text,
                    voice_profile=voice_profile,
                    synthesis_time=processing_time,
                    audio_quality_score=quality_metrics.get('audio_quality', 0.0),
                    naturalness_score=quality_metrics.get('naturalness', 0.0),
                    intelligibility_score=quality_metrics.get('intelligibility', 0.0),
                    synthesis_mode=request.synthesis_mode,
                    session_id=request.session_id,
                    user_id=request.user_id,
                    conversation_id=request.conversation_id
                )
                
                # Save to file if requested
                if request.output_format != AudioOutputFormat.RAW_PCM:
                    output_path = await self._save_audio_to_file(
                        processed_audio, audio_metadata, request
                    )
                    result.output_file_path = output_path
                    if output_path:
                        result.file_size_bytes = output_path.stat().st_size
                
                # Cache result if enabled
                if cache_key:
                    await self._cache_result(cache_key, result)
                
                # Store in memory and context
                await self._store_synthesis_context(result, request)
                
                # Update metrics
                self._update_metrics(result, processing_time)
                
                # Learn from synthesis if enabled
                if self.learning_enabled:
                    await self._learn_from_synthesis(result, request)
                
                # Emit completion event
                await self.event_bus.emit(SpeechSynthesisCompleted(
                    session_id=request.session_id,
                    processing_time=processing_time,
                    audio_duration=result.audio_metadata.duration,
                    quality_score=result.audio_quality_score
                ))
                
                self.logger.info(
                    f"TTS synthesis completed for session {request.session_id} "
                    f"(Duration: {result.audio_metadata.duration:.2f}s, "
                    f"Quality: {result.audio_quality_score:.2f}, "
                    f"Time: {processing_time:.2f}s)"
                )
                
                return result
                
        except Exception as e:
            # Emit error event
            await self.event_bus.emit(SpeechSynthesisError(
                session_id=request.session_id,
                error_type=type(e).__name__,
                error_message=str(e)
            ))
            
            self.metrics.increment("tts_synthesis_errors_total")
            self.logger.error(f"TTS synthesis failed for session {request.session_id}: {str(e)}")
            raise TTSError(f"Speech synthesis failed: {str(e)}") from e
    
    async def _preprocess_text(self, request: SynthesisRequest) -> str:
        """Preprocess text for optimal synthesis."""
        text = request.text
        
        if not request.enable_preprocessing:
            return text
        
        try:
            # Text normalization
            text = self._normalize_text(text)
            
            # Handle special characters and abbreviations
            text = self._expand_abbreviations(text)
            
            # Phoneme preprocessing if needed
            if hasattr(self, 'phonemizer'):
                text = self._preprocess_phonemes(text)
            
            # Remove or convert unsupported characters
            text = self._clean_unsupported_characters(text)
            
            return text
            
        except Exception as e:
            self.logger.warning(f"Text preprocessing failed: {str(e)}")
            return request.text  # Return original text
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for synthesis."""
        # Basic normalization
        text = text.strip()
        
        # Handle multiple spaces
        text = ' '.join(text.split())
        
        # Handle common replacements
        replacements = {
            ' & ': ' and ',
            ' @ ': ' at ',
            ' # ': ' number ',
            ' $ ': ' dollars ',
            ' % ': ' percent ',
            ' + ': ' plus ',
            ' = ': ' equals ',
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return text
    
    def _expand_abbreviations(self, text: str) -> str:
        """Expand common abbreviations."""
        abbreviations = {
            'Dr.': 'Doctor',
            'Mr.': 'Mister',
            'Mrs.': 'Missus',
            'Ms.': 'Miss',
            'Prof.': 'Professor',
            'Inc.': 'Incorporated',
            'Corp.': 'Corporation',
            'LLC': 'Limited Liability Company',
            'etc.': 'etcetera',
            'vs.': 'versus',
            'e.g.': 'for example',
            'i.e.': 'that is',
        }
        
        for abbrev, expansion in abbreviations.items():
            text = text.replace(abbrev, expansion)
        
        return text
    
    def _clean_unsupported_characters(self, text: str) -> str:
        """Remove or convert characters that might cause synthesis issues."""
        # Remove or replace problematic characters
        text = text.replace('\n', ' ')
        text = text.replace('\t', ' ')
        text = text.replace('\r', ' ')
        
        # Remove excessive punctuation
        text = text.replace('...', '.')
        text = text.replace('!!!', '!')
        text = text.replace('???', '?')
        
        return text
    
    async def _select_and_load_voice(self, request: SynthesisRequest) -> tuple[VoiceProfile, PiperVoice]:
        """Select optimal voice and load model."""
        # Determine voice ID
        voice_id = request.voice_id or await self._select_optimal_voice(request)
        
        # Get voice profile
        voice_profile = await self.voice_manager.get_voice(voice_id)
        if not voice_profile:
            # Fallback to default voice
            voice_id = self.default_voice_id
            voice_profile = await self.voice_manager.get_voice(voice_id)
            if not voice_profile:
                raise TTSError(f"No suitable voice found")
        
        # Load voice model
        voice_model = await self.voice_manager.load_voice_model(voice_id)
        
        return voice_profile, voice_model
    
    async def _select_optimal_voice(self, request: SynthesisRequest) -> str:
        """Select optimal voice based on request parameters and user preferences."""
        # Check user preferences
        if request.user_id and request.user_id in self.user_voice_preferences:
            preferred_voice = self.user_voice_preferences[request.user_id]
            voice_profile = await self.voice_manager.get_voice(preferred_voice)
            if voice_profile:
                return preferred_voice
        
        # Language-based selection
        if request.language:
            voices = await self.voice_manager.list_voices(language=request.language)
            if voices:
                # Sort by quality
                voices.sort(key=lambda v: self._voice_quality_score(v), reverse=True)
                return voices[0].voice_id
        
        # Default voice selection strategy
        if self.voice_selection_strategy == "quality_first":
            all_voices = await self.voice_manager.list_voices()
            if all_voices:
                all_voices.sort(key=lambda v: self._voice_quality_score(v), reverse=True)
                return all_voices[0].voice_id
        
        return self.default_voice_id
    
    def _voice_quality_score(self, voice_profile: VoiceProfile) -> float:
        """Calculate quality score for voice selection."""
        quality_scores = {
            "low": 0.3,
            "medium": 0.7,
            "high": 1.0,
            "x_low": 0.1,
            "x_high": 1.2
        }
        
        base_score = quality_scores.get(voice_profile.quality, 0.5)
        
        # Bonus for higher sample rate
        if voice_profile.sample_rate >= 22050:
            base_score += 0.1
        
        # Bonus for recent load times (performance indicator)
        if voice_profile.load_time and voice_profile.load_time < 2.0:
            base_score += 0.1
        
        return base_score
    
    async def _analyze_and_setup_prosody(
        self, 
        text: str, 
        request: SynthesisRequest
    ) -> ProsodySettings:
        """Analyze text and setup prosody for natural speech."""
        # Use provided prosody or analyze text
        if request.prosody:
            prosody_settings = request.prosody
        else:
            # Prepare context for prosody analysis
            context = {}
            
            # Add sentiment analysis if available
            if self.sentiment_analyzer and request.enable_emotional_adaptation:
                try:
                    sentiment = await self.sentiment_analyzer.analyze(text)
                    context['sentiment'] = sentiment
                    if hasattr(sentiment, 'emotion'):
                        context['detected_emotion'] = sentiment.emotion
                except Exception as e:
                    self.logger.warning(f"Sentiment analysis failed: {str(e)}")
            
            # Add conversational context if available
            if request.context:
                context.update(request.context)
            
            prosody_settings = self.prosody_processor.analyze_text_for_prosody(text, context)
        
        # Apply speaking style
        if request.speaking_style != SpeakingStyle.NEUTRAL:
            prosody_settings = self._apply_speaking_style(prosody_settings, request.speaking_style)
        
        return prosody_settings
    
    def _apply_speaking_style(self, prosody: ProsodySettings, style: SpeakingStyle) -> ProsodySettings:
        """Apply speaking style to prosody settings."""
        style_adjustments = {
            SpeakingStyle.CHEERFUL: {
                "pitch_base": 1.2, "speech_rate": 1.1, "emphasis_strength": 1.3
            },
            SpeakingStyle.SAD: {
                "pitch_base": 0.8, "speech_rate": 0.9, "emphasis_strength": 0.7
            },
            SpeakingStyle.ANGRY: {
                "pitch_base": 1.1, "speech_rate": 1.2, "emphasis_strength": 1.4
            },
            SpeakingStyle.EXCITED: {
                "pitch_base": 1.3, "speech_rate": 1.3, "emphasis_strength": 1.5
            },
            SpeakingStyle.WHISPERING: {
                "pitch_base": 0.9, "speech_rate": 0.8, "emphasis_strength": 0.5
            },
            SpeakingStyle.SHOUTING: {
                "pitch_base": 1.4, "speech_rate": 1.1, "emphasis_strength": 1.8
            },
            SpeakingStyle.CALM: {
                "pitch_base": 1.0, "speech_rate": 0.95, "emphasis_strength": 0.8
            },
            SpeakingStyle.ANXIOUS: {
                "pitch_base": 1.15, "speech_rate": 1.2, "emphasis_strength": 1.1
            },
            SpeakingStyle.CONFIDENT: {
                "pitch_base": 1.05, "speech_rate": 1.0, "emphasis_strength": 1.2
            }
        }
        
        if style in style_adjustments:
            adjustments = style_adjustments[style]
            prosody.pitch_base *= adjustments.get("pitch_base", 1.0)
            prosody.speech_rate *= adjustments.get("speech_rate", 1.0)
            prosody.emphasis_strength *= adjustments.get("emphasis_strength", 1.0)
            prosody.speaking_style = style
        
        return prosody
    
    async def _perform_synthesis(
        self,
        text: str,
        voice_model: PiperVoice,
        prosody_settings: ProsodySettings,
        request: SynthesisRequest
    ) -> np.ndarray:
        """Perform the actual text-to-speech synthesis."""
        start_time = datetime.now(timezone.utc)
        
        try:
            # Apply prosody to synthesis parameters
            synthesis_params = self.prosody_processor.apply_prosody_to_synthesis(
                voice_model, prosody_settings
            )
            
            # Perform synthesis based on mode
            if request.synthesis_mode == SynthesisMode.STREAMING:
                audio_data = await self._synthesize_streaming(text, voice_model, synthesis_params)
            else:
                audio_data = await self._synthesize_batch(text, voice_model, synthesis_params)
            
            # Record inference time
            inference_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            self.metrics.record("tts_model_inference_duration_seconds", inference_time)
            
            return audio_data
            
        except Exception as e:
            raise TTSError(f"Synthesis failed: {str(e)}") from e
    
    async def _synthesize_batch(
        self,
        text: str,
        voice_model: PiperVoice,
        synthesis_params: Dict[str, Any]
    ) -> np.ndarray:
        """Perform batch synthesis."""
        try:
            # Synthesize using Piper
            audio_stream = voice_model.synthesize_stream(text, **synthesis_params)
            
            # Collect audio chunks
            audio_chunks = []
            for audio_chunk in audio_stream:
                audio_chunks.append(audio_chunk)
            
            # Concatenate audio
            if audio_chunks:
                audio_data = np.concatenate(audio_chunks)
            else:
                audio_data = np.array([], dtype=np.float32)
            
            return audio_data
            
        except Exception as e:
            raise TTSError(f"Batch synthesis failed: {str(e)}") from e
    
    async def _synthesize_streaming(
        self,
        text: str,
        voice_model: PiperVoice,
        synthesis_params: Dict[str, Any]
    ) -> np.ndarray:
        """Perform streaming synthesis (collect all chunks for now)."""
        # For now, streaming mode still returns complete audio
        # Real streaming would yield chunks as they're generated
        return await self._synthesize_batch(text, voice_model, synthesis_params)
    
    async def _postprocess_audio(
        self,
        audio_data: np.ndarray,
        request: SynthesisRequest
    ) -> np.ndarray:
        """Apply post-processing to synthesized audio."""
        if not request.enable_postprocessing:
            return audio_data
        
        processed_audio = audio_data.copy()
        
        try:
            # Normalization
            if self.enable_normalization and request.normalize_audio:
                processed_audio = self.audio_processor.normalize_audio(
                    processed_audio, target_db=-20.0
                )
            
            # Audio enhancement
            if self.enable_audio_enhancement:
                processed_audio = await self._enhance_audio_quality(processed_audio)
            
            # Apply audio effects if requested
            if request.apply_audio_effects:
                processed_audio = await self._apply_audio_effects(processed_audio, request)
            
            # Noise gate
            if self.enable_noise_gate:
                processed_audio = self._apply_noise_gate(processed_audio)
            
            return processed_audio
            
        except Exception as e:
            self.logger.warning(f"Audio post-processing failed: {str(e)}")
            return audio_data  # Return original audio
    
    async def _enhance_audio_quality(self, audio: np.ndarray) -> np.ndarray:
        """Enhance audio quality through processing."""
        try:
            # Apply subtle noise reduction
            enhanced_audio = self.audio_processor.advanced_noise_reduction(
                audio, self.default_sample_rate, method="spectral_gating"
            )
            
            # Apply dynamic range optimization
            enhanced_audio = self._optimize_dynamic_range(enhanced_audio)
            
            return enhanced_audio
            
        except Exception as e:
            self.logger.warning(f"Audio enhancement failed: {str(e)}")
            return audio
    
    def _optimize_dynamic_range(self, audio: np.ndarray) -> np.ndarray:
        """Optimize dynamic range for better listening experience."""
        # Simple dynamic range compression
        threshold = 0.8
        ratio = 4.0
        
        # Find peaks above threshold
        peaks = np.abs(audio) > threshold
        
        if np.any(peaks):
            # Apply compression to peaks
            compressed_peaks = threshold + (np.abs(audio[peaks]) - threshold) / ratio
            audio[peaks] = np.sign(audio[peaks]) * compressed_peaks
        
        return audio
    
    def _apply_noise_gate(self, audio: np.ndarray) -> np.ndarray:
        """Apply noise gate to reduce background noise."""
        # Simple noise gate implementation
        gate_threshold = 0.01
        below_threshold = np.abs(audio) < gate_threshold
        audio[below_threshold] *= 0.1  # Reduce rather than completely silence
        
        return audio
    
    async def _apply_audio_effects(
        self,
        audio: np.ndarray,
        request: SynthesisRequest
    ) -> np.ndarray:
        """Apply requested audio effects."""
        # Placeholder for audio effects
        # Could include reverb, echo, EQ, etc.
        return audio
    
    async def _assess_synthesis_quality(
        self,
        audio: np.ndarray,
        text: str,
        voice_profile: VoiceProfile
    ) -> Dict[str, float]:
        """Assess quality of synthesized audio."""
        metrics = {}
        
        try:
            # Audio quality metrics
            rms_energy = np.sqrt(np.mean(audio ** 2))
            dynamic_range = np.max(np.abs(audio)) - np.mean(np.abs(audio))
            
            metrics['audio_quality'] = min(1.0, rms_energy * 10)  # Simple quality score
            metrics['dynamic_range'] = min(1.0, dynamic_range * 5)
            
            # Naturalness estimation (based on audio characteristics)
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=voice_profile.sample_rate))
            metrics['naturalness'] = min(1.0, spectral_centroid / 2000.0)  # Normalize
            
            # Intelligibility estimation (based on clarity metrics)
            zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(audio))
            metrics['intelligibility'] = min(1.0, 1.0 - zero_crossing_rate)
            
            # Overall quality score
            metrics['overall_quality'] = np.mean([
                metrics['audio_quality'],
                metrics['naturalness'],
                metrics['intelligibility']
            ])
            
        except Exception as e:
            self.logger.warning(f"Quality assessment failed: {str(e)}")
            # Default moderate quality scores
            metrics = {
                'audio_quality': 0.7,
                'naturalness': 0.6,
                'intelligibility': 0.7,
                'overall_quality': 0.67
            }
        
        return metrics
    
    async def _save_audio_to_file(
        self,
        audio: np.ndarray,
        metadata: AudioMetadata,
        request: SynthesisRequest
    ) -> Optional[Path]:
        """Save synthesized audio to file."""
        try:
            # Create output directory
            output_dir = Path("data/tts_output")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate filename
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            session_id = request.session_id[:8] if request.session_id else "unknown"
            filename = f"tts_{session_id}_{timestamp}.{request.output_format.value}"
            output_path = output_dir / filename
            
            # Save based on format
            if request.output_format == AudioOutputFormat.WAV:
                sf.write(output_path, audio, metadata.sample_rate, format='WAV', subtype='PCM_16')
            elif request.output_format == AudioOutputFormat.FLAC:
                sf.write(output_path, audio, metadata.sample_rate, format='FLAC')
            elif request.output_format == AudioOutputFormat.OGG:
                sf.write(output_path, audio, metadata.sample_rate, format='OGG', subtype='VORBIS')
            else:
                # Default to WAV
                sf.write(output_path, audio, metadata.sample_rate)
            
            return output_path
            
        except Exception as e:
            self.logger.warning(f"Failed to save audio file: {str(e)}")
            return None
    
    async def synthesize_streaming(
        self,
        request: Union[str, SynthesisRequest],
        callback: Callable[[np.ndarray], None],
        streaming_config: Optional[StreamingConfig] = None
    ) -> str:
        """
        Start streaming text-to-speech synthesis.
        
        Args:
            request: Text or synthesis request
            callback: Function called with each audio chunk
            streaming_config: Streaming configuration
            
        Returns:
            Stream session ID
        """
        if not self.streaming_enabled:
            raise TTSError("Streaming synthesis is not enabled")
        
        # Convert string to request
        if isinstance(request, str):
            request = SynthesisRequest(
                text=request,
                synthesis_mode=SynthesisMode.STREAMING
            )
        else:
            request.synthesis_mode = SynthesisMode.STREAMING
        
        # Generate session ID
        stream_id = request.session_id or self._generate_session_id()
        config = streaming_config or self.default_streaming_config
        
        with self.stream_lock:
            if stream_id in self.active_streams:
                raise TTSError(f"Stream {stream_id} already exists")
            
            self.active_streams[stream_id] = {
                'request': request,
                'config': config,
                'callback': callback,
                'start_time': datetime.now(timezone.utc),
                'status': 'active',
                'task': None
            }
        
        # Start streaming task
        task = asyncio.create_task(self._streaming_synthesis_loop(stream_id))
        self.active_streams[stream_id]['task'] = task
        
        # Emit stream started event
        await self.event_bus.emit(AudioStreamStarted(
            session_id=stream_id,
            sample_rate=self.default_sample_rate,
            chunk_size=config.chunk_size
        ))
        
        self.logger.info(f"Started TTS streaming session: {stream_id}")
        return stream_id
    
    async def _streaming_synthesis_loop(self, stream_id: str) -> None:
        """Main loop for streaming synthesis."""
        try:
            with self.stream_lock:
                stream_info = self.active_streams.get(stream_id)
                if not stream_info:
                    return
                
                request = stream_info['request']
                config = stream_info['config']
                callback = stream_info['callback']
                
                # Process streaming synthesis
                await self._process_streaming_synthesis(stream_id, request, config, callback)
                
        except Exception as e:
            self.logger.error(f"Error in streaming loop: {str(e)}")
            await asyncio.sleep(1)
