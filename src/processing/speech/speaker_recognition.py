"""
Advanced Speaker Recognition Module
Author: Drmusab
Last Modified: 2025-05-26 14:19:02 UTC

This module provides comprehensive speaker recognition capabilities integrated
with the AI assistant's core architecture, including speaker identification,
verification, voice profiling, anti-spoofing, and adaptive learning systems.
"""

import hashlib
import json
import pickle
import queue
import tempfile
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Set, Tuple, Union

import asyncio
import joblib
import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from scipy import signal
from scipy.spatial.distance import euclidean
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

# Core imports
from src.core.config.loader import ConfigLoader
from src.core.dependency_injection import Container
from src.core.error_handling import ErrorHandler, handle_exceptions
from src.core.events.event_bus import EventBus
from src.core.events.event_types import (
    BiometricDataProcessed,
    SecurityAlert,
    SpeakerIdentified,
    SpeakerProcessingCompleted,
    SpeakerProcessingError,
    SpeakerProcessingStarted,
    SpeakerRegistered,
    SpeakerVerified,
    VoiceProfileUpdated,
)
from src.core.health_check import HealthCheck
from src.core.security.authentication import SecurityValidator
from src.integrations.cache.cache_strategy import CacheStrategy
from src.learning.continual_learning import ContinualLearner

# Learning and adaptation
from src.learning.feedback_processor import FeedbackProcessor
from src.learning.preference_learning import PreferenceLearner

# Memory and caching
from src.memory.cache_manager import CacheManager
from src.memory.context_manager import ContextManager
from src.memory.memory_manager import MemoryManager
from src.observability.logging.config import get_logger

# Observability
from src.observability.monitoring.metrics import MetricsCollector
from src.observability.monitoring.tracing import TraceManager
from src.processing.multimodal.fusion_strategies import MultimodalFusionStrategy

# Processing imports
from src.processing.speech.audio_utils import (
    AudioFormat,
    AudioMetadata,
    AudioProcessingError,
    EnhancedAudioProcessor,
    ProcessingSettings,
)
from src.processing.speech.emotion_detection import EmotionDetector


class VerificationMode(Enum):
    """Speaker verification modes."""

    IDENTIFICATION = "identification"  # Who is this speaker?
    VERIFICATION = "verification"  # Is this speaker X?
    DETECTION = "detection"  # Is there a speaker present?


class ProcessingMode(Enum):
    """Speaker recognition processing modes."""

    REAL_TIME = "real_time"
    BATCH = "batch"
    STREAMING = "streaming"
    ENROLLMENT = "enrollment"
    VERIFICATION = "verification"


class SecurityLevel(Enum):
    """Security levels for speaker verification."""

    LOW = "low"  # Basic verification
    MEDIUM = "medium"  # Standard verification
    HIGH = "high"  # Enhanced verification with anti-spoofing
    CRITICAL = "critical"  # Maximum security with multi-factor


class SpoofingType(Enum):
    """Types of spoofing attacks."""

    REPLAY = "replay"  # Recorded audio playback
    TTS = "text_to_speech"  # Synthetic speech
    VOICE_CONVERSION = "voice_conversion"  # Voice morphing
    DEEPFAKE = "deepfake"  # AI-generated voice
    NONE = "none"  # Legitimate speech


@dataclass
class VoiceProfile:
    """Comprehensive voice profile for a speaker."""

    speaker_id: str
    name: Optional[str] = None

    # Voice embeddings and features
    primary_embedding: Optional[np.ndarray] = None
    embeddings_history: List[np.ndarray] = field(default_factory=list)
    statistical_features: Optional[Dict[str, float]] = None

    # Biometric characteristics
    fundamental_frequency_range: Tuple[float, float] = (0.0, 0.0)
    formant_characteristics: Optional[Dict[str, Any]] = None
    vocal_tract_length: Optional[float] = None
    speaking_rate_range: Tuple[float, float] = (0.0, 0.0)

    # Quality and confidence metrics
    enrollment_quality: float = 0.0
    confidence_threshold: float = 0.8
    verification_count: int = 0
    last_verification: Optional[datetime] = None

    # Adaptive learning data
    adaptation_data: Dict[str, Any] = field(default_factory=dict)
    embedding_drift_history: List[float] = field(default_factory=list)

    # Security and anti-spoofing
    security_features: Optional[Dict[str, Any]] = None
    spoofing_resistance_score: float = 0.0

    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    version: str = "1.0"

    # Demographics and context (optional)
    age_group: Optional[str] = None
    gender: Optional[str] = None
    language: Optional[str] = None
    accent: Optional[str] = None

    # Usage statistics
    total_sessions: int = 0
    successful_verifications: int = 0
    failed_verifications: int = 0

    def update_verification_stats(self, success: bool):
        """Update verification statistics."""
        self.verification_count += 1
        self.last_verification = datetime.now(timezone.utc)
        self.updated_at = datetime.now(timezone.utc)

        if success:
            self.successful_verifications += 1
        else:
            self.failed_verifications += 1

    @property
    def verification_success_rate(self) -> float:
        """Calculate verification success rate."""
        if self.verification_count == 0:
            return 0.0
        return self.successful_verifications / self.verification_count


@dataclass
class SpeakerRecognitionRequest:
    """Configuration for speaker recognition requests."""

    audio_source: str = "buffer"
    mode: ProcessingMode = ProcessingMode.REAL_TIME
    verification_mode: VerificationMode = VerificationMode.IDENTIFICATION
    security_level: SecurityLevel = SecurityLevel.MEDIUM

    # Target speaker for verification
    target_speaker_id: Optional[str] = None

    # Processing options
    enable_anti_spoofing: bool = True
    enable_multi_speaker_detection: bool = False
    enable_voice_activity_detection: bool = True
    enable_quality_assessment: bool = True

    # Adaptation and learning
    enable_adaptation: bool = True
    update_profile: bool = False

    # Session and context
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    conversation_id: Optional[str] = None

    # Caching and performance
    cache_results: bool = True
    timeout_seconds: float = 30.0

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: Set[str] = field(default_factory=set)


@dataclass
class SpeakerRecognitionResult:
    """Comprehensive speaker recognition result."""

    # Primary results
    identified_speaker_id: Optional[str] = None
    verification_score: float = 0.0
    confidence_score: float = 0.0
    verification_decision: bool = False

    # Speaker candidates (for identification)
    speaker_candidates: List[Dict[str, Any]] = field(default_factory=list)

    # Multi-speaker results
    detected_speakers: List[Dict[str, Any]] = field(default_factory=list)
    speaker_segments: List[Dict[str, Any]] = field(default_factory=list)

    # Voice characteristics
    extracted_embedding: Optional[np.ndarray] = None
    voice_characteristics: Dict[str, Any] = field(default_factory=dict)
    quality_metrics: Dict[str, float] = field(default_factory=dict)

    # Security and anti-spoofing
    spoofing_detection: Dict[str, Any] = field(default_factory=dict)
    security_assessment: Dict[str, Any] = field(default_factory=dict)

    # Processing information
    mode: ProcessingMode = ProcessingMode.REAL_TIME
    processing_time: float = 0.0
    model_version: str = "1.0"

    # Session information
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    conversation_id: Optional[str] = None

    # Success and warnings
    success: bool = True
    warnings: List[str] = field(default_factory=list)
    error_message: Optional[str] = None

    # Timestamp
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class EnrollmentRequest:
    """Request for speaker enrollment."""

    speaker_id: str
    speaker_name: Optional[str] = None
    audio_samples: List[Union[np.ndarray, str, Path]] = field(default_factory=list)

    # Quality requirements
    min_duration: float = 10.0  # Minimum total audio duration
    max_duration: float = 300.0  # Maximum total audio duration
    quality_threshold: float = 0.7

    # Metadata
    demographics: Dict[str, Any] = field(default_factory=dict)
    security_level: SecurityLevel = SecurityLevel.MEDIUM

    # Processing options
    enable_quality_assessment: bool = True
    enable_augmentation: bool = True

    def validate(self) -> bool:
        """Validate enrollment request."""
        if not self.speaker_id or not self.audio_samples:
            return False

        total_duration = 0
        for sample in self.audio_samples:
            if isinstance(sample, np.ndarray):
                total_duration += len(sample) / 16000  # Assume 16kHz
            # For file paths, would need to check file duration

        return self.min_duration <= total_duration <= self.max_duration


class SpeakerRecognitionError(Exception):
    """Custom exception for speaker recognition operations."""

    def __init__(
        self, message: str, error_code: Optional[str] = None, component: Optional[str] = None
    ):
        super().__init__(message)
        self.error_code = error_code
        self.component = component
        self.timestamp = datetime.now(timezone.utc)


class EcapaTdnnModel(nn.Module):
    """ECAPA-TDNN model for speaker embeddings."""

    def __init__(self, input_dim: int = 80, embedding_dim: int = 512):
        super().__init__()

        self.input_dim = input_dim
        self.embedding_dim = embedding_dim

        # Initial convolution
        self.conv1 = nn.Conv1d(input_dim, 512, kernel_size=5, padding=2)

        # ECAPA-TDNN blocks
        self.tdnn_blocks = nn.ModuleList(
            [
                self._make_tdnn_block(512, 512, kernel_size=3, dilation=1),
                self._make_tdnn_block(512, 512, kernel_size=3, dilation=2),
                self._make_tdnn_block(512, 512, kernel_size=3, dilation=3),
                self._make_tdnn_block(512, 512, kernel_size=1, dilation=1),
            ]
        )

        # Attention and pooling
        self.attention = nn.MultiheadAttention(512, num_heads=8, batch_first=True)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Embedding layer
        self.embedding_layer = nn.Sequential(
            nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, embedding_dim)
        )

        # Classification head (for training)
        self.classifier = nn.Linear(embedding_dim, 1000)  # Adjust based on number of speakers

    def _make_tdnn_block(
        self, in_channels: int, out_channels: int, kernel_size: int, dilation: int
    ):
        """Create a TDNN block with residual connections."""
        return nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                dilation=dilation,
                padding=dilation * (kernel_size - 1) // 2,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

    def forward(self, x: torch.Tensor, return_embedding: bool = True) -> Dict[str, torch.Tensor]:
        """
        Forward pass through ECAPA-TDNN.

        Args:
            x: Input tensor of shape (batch, features, time)
            return_embedding: Whether to return embeddings

        Returns:
            Dictionary containing embeddings and/or logits
        """
        # Initial convolution
        x = F.relu(self.conv1(x))

        # TDNN blocks with residual connections
        for block in self.tdnn_blocks:
            residual = x
            x = block(x)
            if x.shape == residual.shape:
                x = x + residual

        # Attention mechanism
        x_transposed = x.transpose(1, 2)  # (batch, time, features)
        attended, _ = self.attention(x_transposed, x_transposed, x_transposed)
        x = attended.transpose(1, 2)  # Back to (batch, features, time)

        # Global pooling
        x = self.global_pool(x).squeeze(-1)  # (batch, features)

        # Generate embeddings
        embeddings = self.embedding_layer(x)

        result = {"embeddings": embeddings}

        if not return_embedding:
            # Classification for training
            logits = self.classifier(embeddings)
            result["logits"] = logits

        return result


class AntiSpoofingModel(nn.Module):
    """Anti-spoofing model for detecting synthetic speech."""

    def __init__(self, input_dim: int = 80):
        super().__init__()

        # Feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(input_dim, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, len(SpoofingType) - 1),  # Exclude NONE
        )

        # Confidence estimation
        self.confidence_head = nn.Sequential(
            nn.Linear(512, 128), nn.ReLU(), nn.Linear(128, 1), nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass for anti-spoofing detection."""
        # Extract features
        features = self.feature_extractor(x).squeeze(-1)

        # Classification
        spoofing_logits = self.classifier(features)
        spoofing_probs = F.softmax(spoofing_logits, dim=-1)

        # Confidence
        confidence = self.confidence_head(features)

        return {
            "spoofing_logits": spoofing_logits,
            "spoofing_probs": spoofing_probs,
            "confidence": confidence,
            "features": features,
        }


class SpeakerEmbeddingExtractor:
    """Advanced speaker embedding extraction with multiple models."""

    def __init__(self, model_dir: Path, device: torch.device, logger):
        self.model_dir = model_dir
        self.device = device
        self.logger = logger

        # Models
        self.ecapa_model: Optional[EcapaTdnnModel] = None
        self.anti_spoofing_model: Optional[AntiSpoofingModel] = None

        # Feature extraction settings
        self.n_mels = 80
        self.n_fft = 512
        self.hop_length = 160
        self.win_length = 400
        self.sample_rate = 16000

        # Load models
        self._load_models()

    def _load_models(self):
        """Load speaker recognition models."""
        try:
            # Load ECAPA-TDNN model
            ecapa_path = self.model_dir / "ecapa_tdnn.pth"
            if ecapa_path.exists():
                self.ecapa_model = EcapaTdnnModel().to(self.device)
                checkpoint = torch.load(ecapa_path, map_location=self.device)
                self.ecapa_model.load_state_dict(checkpoint["model_state_dict"])
                self.ecapa_model.eval()
                self.logger.info("Loaded ECAPA-TDNN model")
            else:
                self.logger.warning("ECAPA-TDNN model not found, initializing new model")
                self.ecapa_model = EcapaTdnnModel().to(self.device)
                self.ecapa_model.eval()

            # Load anti-spoofing model
            antispoofing_path = self.model_dir / "antispoofing.pth"
            if antispoofing_path.exists():
                self.anti_spoofing_model = AntiSpoofingModel().to(self.device)
                checkpoint = torch.load(antispoofing_path, map_location=self.device)
                self.anti_spoofing_model.load_state_dict(checkpoint["model_state_dict"])
                self.anti_spoofing_model.eval()
                self.logger.info("Loaded anti-spoofing model")
            else:
                self.logger.warning("Anti-spoofing model not found, initializing new model")
                self.anti_spoofing_model = AntiSpoofingModel().to(self.device)
                self.anti_spoofing_model.eval()

        except Exception as e:
            self.logger.error(f"Failed to load models: {str(e)}")
            raise SpeakerRecognitionError(f"Model loading failed: {str(e)}")

    def extract_features(self, audio: np.ndarray) -> torch.Tensor:
        """Extract mel-spectrogram features from audio."""
        # Convert to torch tensor
        audio_tensor = torch.from_numpy(audio).float()

        # Extract mel-spectrogram
        mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_mels=self.n_mels,
            power=2.0,
        )(audio_tensor)

        # Convert to log scale
        log_mel = torch.log(mel_spec + 1e-8)

        return log_mel.unsqueeze(0).to(self.device)

    def extract_embedding(self, audio: np.ndarray) -> np.ndarray:
        """Extract speaker embedding from audio."""
        try:
            # Extract features
            features = self.extract_features(audio)

            # Get embedding
            with torch.no_grad():
                outputs = self.ecapa_model(features, return_embedding=True)
                embedding = outputs["embeddings"].cpu().numpy()[0]

            # Normalize embedding
            embedding = embedding / np.linalg.norm(embedding)

            return embedding

        except Exception as e:
            raise SpeakerRecognitionError(f"Embedding extraction failed: {str(e)}")

    def detect_spoofing(self, audio: np.ndarray) -> Dict[str, Any]:
        """Detect spoofing attacks in audio."""
        try:
            # Extract features
            features = self.extract_features(audio)

            # Run anti-spoofing detection
            with torch.no_grad():
                outputs = self.anti_spoofing_model(features)

                spoofing_probs = outputs["spoofing_probs"].cpu().numpy()[0]
                confidence = float(outputs["confidence"].cpu().numpy()[0, 0])

            # Interpret results
            spoofing_types = [t for t in SpoofingType if t != SpoofingType.NONE]
            spoofing_scores = {
                spoofing_types[i].value: float(prob) for i, prob in enumerate(spoofing_probs)
            }

            # Determine if spoofed
            max_spoofing_score = max(spoofing_scores.values())
            is_spoofed = max_spoofing_score > 0.5

            if is_spoofed:
                detected_type = max(spoofing_scores, key=spoofing_scores.get)
            else:
                detected_type = SpoofingType.NONE.value

            return {
                "is_spoofed": is_spoofed,
                "spoofing_type": detected_type,
                "spoofing_scores": spoofing_scores,
                "confidence": confidence,
                "overall_score": max_spoofing_score,
            }

        except Exception as e:
            self.logger.warning(f"Spoofing detection failed: {str(e)}")
            return {
                "is_spoofed": False,
                "spoofing_type": SpoofingType.NONE.value,
                "spoofing_scores": {},
                "confidence": 0.0,
                "overall_score": 0.0,
            }


class VoiceProfileManager:
    """Manager for speaker voice profiles with persistent storage."""

    def __init__(self, storage_dir: Path, logger):
        self.storage_dir = storage_dir
        self.logger = logger
        self.profiles: Dict[str, VoiceProfile] = {}
        self.profile_lock = threading.Lock()

        # Ensure storage directory exists
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # Load existing profiles
        self._load_profiles()

    def _load_profiles(self):
        """Load existing voice profiles from storage."""
        try:
            profiles_file = self.storage_dir / "voice_profiles.pkl"
            if profiles_file.exists():
                with open(profiles_file, "rb") as f:
                    self.profiles = pickle.load(f)
                self.logger.info(f"Loaded {len(self.profiles)} voice profiles")

        except Exception as e:
            self.logger.warning(f"Failed to load voice profiles: {str(e)}")
            self.profiles = {}

    def _save_profiles(self):
        """Save voice profiles to persistent storage."""
        try:
            profiles_file = self.storage_dir / "voice_profiles.pkl"
            with open(profiles_file, "wb") as f:
                pickle.dump(self.profiles, f)

        except Exception as e:
            self.logger.error(f"Failed to save voice profiles: {str(e)}")

    def create_profile(self, speaker_id: str, name: Optional[str] = None) -> VoiceProfile:
        """Create a new voice profile."""
        with self.profile_lock:
            if speaker_id in self.profiles:
                raise SpeakerRecognitionError(f"Profile for speaker {speaker_id} already exists")

            profile = VoiceProfile(speaker_id=speaker_id, name=name)
            self.profiles[speaker_id] = profile
            self._save_profiles()

            return profile

    def get_profile(self, speaker_id: str) -> Optional[VoiceProfile]:
        """Get voice profile by speaker ID."""
        return self.profiles.get(speaker_id)

    def update_profile(self, speaker_id: str, **updates) -> bool:
        """Update voice profile with new data."""
        with self.profile_lock:
            if speaker_id not in self.profiles:
                return False

            profile = self.profiles[speaker_id]

            # Update fields
            for key, value in updates.items():
                if hasattr(profile, key):
                    setattr(profile, key, value)

            profile.updated_at = datetime.now(timezone.utc)
            self._save_profiles()

            return True

    def add_embedding(self, speaker_id: str, embedding: np.ndarray) -> bool:
        """Add new embedding to speaker profile."""
        with self.profile_lock:
            profile = self.profiles.get(speaker_id)
            if not profile:
                return False

            # Update primary embedding (moving average)
            if profile.primary_embedding is None:
                profile.primary_embedding = embedding
            else:
                # Exponential moving average
                alpha = 0.1
                profile.primary_embedding = (
                    alpha * embedding + (1 - alpha) * profile.primary_embedding
                )

            # Add to history
            profile.embeddings_history.append(embedding)

            # Keep only recent embeddings
            if len(profile.embeddings_history) > 50:
                profile.embeddings_history = profile.embeddings_history[-50:]

            profile.updated_at = datetime.now(timezone.utc)
            self._save_profiles()

            return True

    def list_profiles(self) -> List[VoiceProfile]:
        """Get list of all voice profiles."""
        return list(self.profiles.values())

    def delete_profile(self, speaker_id: str) -> bool:
        """Delete a voice profile."""
        with self.profile_lock:
            if speaker_id in self.profiles:
                del self.profiles[speaker_id]
                self._save_profiles()
                return True
            return False

    def get_similar_profiles(
        self, embedding: np.ndarray, threshold: float = 0.8
    ) -> List[Tuple[str, float]]:
        """Find profiles similar to given embedding."""
        similarities = []

        for speaker_id, profile in self.profiles.items():
            if profile.primary_embedding is not None:
                similarity = cosine_similarity(
                    embedding.reshape(1, -1), profile.primary_embedding.reshape(1, -1)
                )[0, 0]

                if similarity >= threshold:
                    similarities.append((speaker_id, similarity))

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities


class EnhancedSpeakerRecognition:
    """
    Advanced Speaker Recognition system with comprehensive AI assistant integration.

    Features:
    - State-of-the-art speaker identification and verification
    - Multi-modal voice profiling and biometric analysis
    - Real-time anti-spoofing and security assessment
    - Adaptive learning and personalization
    - Multi-speaker detection and tracking
    - Comprehensive integration with AI assistant core systems
    - Event-driven architecture with detailed monitoring
    - Caching and performance optimization
    - Voice activity detection integration
    - Quality assessment and enhancement
    """

    def __init__(self, container: Container):
        """
        Initialize the enhanced speaker recognition system.

        Args:
            container: Dependency injection container
        """
        self.container = container
        self.logger = get_logger(__name__)
        self.config = container.get(ConfigLoader)
        self.event_bus = container.get(EventBus)
        self.error_handler = container.get(ErrorHandler)
        self.health_check = container.get(HealthCheck)
        self.security_validator = container.get_optional(SecurityValidator)

        # Initialize components
        self._setup_device()
        self._setup_audio_config()
        self._setup_models_and_storage()
        self._setup_processing()
        self._setup_integrations()
        self._setup_monitoring()
        self._setup_caching()
        self._setup_learning()
        self._setup_security()

        # Register health check
        self.health_check.register_component("speaker_recognition", self._health_check_callback)

        self.logger.info(
            f"EnhancedSpeakerRecognition initialized "
            f"(Device: {self.device}, Profiles: {len(self.profile_manager.profiles)})"
        )

    def _setup_device(self) -> None:
        """Setup compute device and memory management."""
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            and self.config.get("speaker_recognition.device.use_gpu", True)
            else "cpu"
        )

        if self.device.type == "cuda":
            torch.cuda.set_per_process_memory_fraction(
                self.config.get("speaker_recognition.device.gpu_memory_fraction", 0.4)
            )

    def _setup_audio_config(self) -> None:
        """Configure audio processing settings."""
        self.sample_rate = self.config.get("speaker_recognition.audio.sample_rate", 16000)
        self.min_audio_duration = self.config.get("speaker_recognition.audio.min_duration", 1.0)
        self.max_audio_duration = self.config.get("speaker_recognition.audio.max_duration", 60.0)

        # Initialize audio processor
        self.audio_processor = EnhancedAudioProcessor(
            sample_rate=self.sample_rate, container=self.container
        )

    def _setup_models_and_storage(self) -> None:
        """Setup models and storage components."""
        # Model directory
        self.model_dir = Path(
            self.config.get(
                "speaker_recognition.models.model_dir", "data/models/speaker_recognition"
            )
        )
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Profile storage
        self.profile_storage_dir = Path(
            self.config.get("speaker_recognition.storage.profile_dir", "data/speaker_profiles")
        )

        # Initialize components
        self.embedding_extractor = SpeakerEmbeddingExtractor(
            self.model_dir, self.device, self.logger
        )

        self.profile_manager = VoiceProfileManager(self.profile_storage_dir, self.logger)

        # Recognition thresholds
        self.verification_threshold = self.config.get(
            "speaker_recognition.thresholds.verification", 0.8
        )
        self.identification_threshold = self.config.get(
            "speaker_recognition.thresholds.identification", 0.7
        )

    def _setup_processing(self) -> None:
        """Setup processing components."""
        # Voice activity detection
        self.vad_enabled = self.config.get("speaker_recognition.vad.enabled", True)
        self.vad_threshold = self.config.get("speaker_recognition.vad.threshold", 0.5)

        # Quality assessment
        self.quality_enabled = self.config.get("speaker_recognition.quality.enabled", True)
        self.min_quality_score = self.config.get("speaker_recognition.quality.min_score", 0.5)

        # Multi-speaker detection
        self.multi_speaker_enabled = self.config.get(
            "speaker_recognition.multi_speaker.enabled", True
        )
        self.speaker_clustering = DBSCAN(eps=0.3, min_samples=2)

    def _setup_integrations(self) -> None:
        """Setup integrations with other AI assistant components."""
        # Memory integration
        self.memory_manager = self.container.get(MemoryManager)
        self.context_manager = self.container.get(ContextManager)

        # Other processing components
        self.emotion_detector = self.container.get_optional(EmotionDetector)

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
        self.metrics.register_counter("speaker_recognition_requests_total")
        self.metrics.register_histogram("speaker_recognition_processing_duration_seconds")
        self.metrics.register_gauge("speaker_verification_score")
        self.metrics.register_counter("speaker_recognition_errors_total")
        self.metrics.register_counter("spoofing_attempts_detected_total")
        self.metrics.register_gauge("speaker_profile_count")

    def _setup_caching(self) -> None:
        """Setup caching for speaker recognition results."""
        self.cache_manager = self.container.get(CacheManager)
        self.cache_strategy = self.container.get(CacheStrategy)

        self.cache_enabled = self.config.get("speaker_recognition.caching.enabled", True)
        self.cache_ttl = self.config.get("speaker_recognition.caching.ttl_seconds", 3600)

    def _setup_learning(self) -> None:
        """Setup learning and adaptation capabilities."""
        self.learning_enabled = self.config.get("speaker_recognition.learning.enabled", True)
        self.adaptation_enabled = self.config.get("speaker_recognition.learning.adaptation", True)

        # Adaptation settings
        self.adaptation_threshold = self.config.get(
            "speaker_recognition.learning.adaptation_threshold", 0.1
        )
        self.max_adaptation_rate = self.config.get(
            "speaker_recognition.learning.max_adaptation_rate", 0.05
        )

    def _setup_security(self) -> None:
        """Setup security and anti-spoofing components."""
        self.security_enabled = self.config.get("speaker_recognition.security.enabled", True)
        self.anti_spoofing_enabled = self.config.get(
            "speaker_recognition.security.anti_spoofing", True
        )

        # Security thresholds
        self.security_thresholds = {
            SecurityLevel.LOW: 0.5,
            SecurityLevel.MEDIUM: 0.7,
            SecurityLevel.HIGH: 0.85,
            SecurityLevel.CRITICAL: 0.95,
        }

    @handle_exceptions
    async def recognize_speaker(
        self,
        audio: Union[np.ndarray, str, Path],
        request: Optional[SpeakerRecognitionRequest] = None,
    ) -> SpeakerRecognitionResult:
        """
        Comprehensive speaker recognition with identification/verification.

        Args:
            audio: Audio data or path to audio file
            request: Speaker recognition request configuration

        Returns:
            Comprehensive speaker recognition result
        """
        start_time = datetime.now(timezone.utc)

        # Default request if not provided
        if request is None:
            request = SpeakerRecognitionRequest()

        # Generate session ID if not provided
        if not request.session_id:
            request.session_id = self._generate_session_id()

        # Emit processing started event
        await self.event_bus.emit(
            SpeakerProcessingStarted(
                session_id=request.session_id,
                mode=request.mode.value,
                verification_mode=request.verification_mode.value,
                security_level=request.security_level.value,
            )
        )

        try:
            with self.tracer.trace("speaker_recognition") as span:
                span.set_attributes(
                    {
                        "session_id": request.session_id,
                        "mode": request.mode.value,
                        "verification_mode": request.verification_mode.value,
                        "security_level": request.security_level.value,
                    }
                )

                # Check cache first
                cache_key = None
                if request.cache_results and self.cache_enabled:
                    cache_key = self._generate_cache_key(audio, request)
                    cached_result = await self._get_cached_result(cache_key)
                    if cached_result:
                        self.logger.info(
                            f"Returning cached recognition result for session {request.session_id}"
                        )
                        return cached_result

                # Process audio input
                processed_audio = await self._process_audio_input(audio, request)

                # Quality assessment
                quality_metrics = await self._assess_audio_quality(processed_audio)

                # Voice activity detection
                if request.enable_voice_activity_detection:
                    vad_result = await self._detect_voice_activity(processed_audio)
                    if not vad_result["has_speech"]:
                        return self._create_no_speech_result(request)

                # Anti-spoofing detection
                spoofing_result = {}
                if request.enable_anti_spoofing and self.anti_spoofing_enabled:
                    spoofing_result = await self._detect_spoofing(processed_audio, request)
                    if spoofing_result["is_spoofed"]:
                        return self._create_spoofing_detected_result(spoofing_result, request)

                # Extract speaker embedding
                embedding = await self._extract_speaker_embedding(processed_audio)

                # Perform recognition based on mode
                if request.verification_mode == VerificationMode.IDENTIFICATION:
                    recognition_result = await self._identify_speaker(embedding, request)
                elif request.verification_mode == VerificationMode.VERIFICATION:
                    recognition_result = await self._verify_speaker(embedding, request)
                else:  # DETECTION
                    recognition_result = await self._detect_speaker_presence(embedding, request)

                # Multi-speaker detection if enabled
                if request.enable_multi_speaker_detection:
                    multi_speaker_result = await self._detect_multiple_speakers(
                        processed_audio, request
                    )
                    recognition_result.detected_speakers = multi_speaker_result.get("speakers", [])
                    recognition_result.speaker_segments = multi_speaker_result.get("segments", [])

                # Extract voice characteristics
                voice_characteristics = await self._extract_voice_characteristics(processed_audio)

                # Update result with additional information
                recognition_result.extracted_embedding = embedding
                recognition_result.voice_characteristics = voice_characteristics
                recognition_result.quality_metrics = quality_metrics
                recognition_result.spoofing_detection = spoofing_result
                recognition_result.security_assessment = await self._assess_security(
                    recognition_result, request
                )

                # Calculate processing time
                processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
                recognition_result.processing_time = processing_time
                recognition_result.mode = request.mode
                recognition_result.session_id = request.session_id
                recognition_result.user_id = request.user_id
                recognition_result.conversation_id = request.conversation_id

                # Update speaker profile if enabled
                if request.update_profile and recognition_result.identified_speaker_id:
                    await self._update_speaker_profile(
                        recognition_result.identified_speaker_id,
                        embedding,
                        voice_characteristics,
                        recognition_result.verification_score > self.verification_threshold,
                    )

                # Cache result if enabled
                if cache_key:
                    await self._cache_result(cache_key, recognition_result)

                # Store in memory and context
                await self._store_recognition_context(recognition_result, request)

                # Update metrics
                self._update_metrics(recognition_result, processing_time)

                # Learn from recognition if enabled
                if self.learning_enabled:
                    await self._learn_from_recognition(recognition_result, request)

                # Emit completion events
                await self._emit_completion_events(recognition_result, request)

                self.logger.info(
                    f"Speaker recognition completed for session {request.session_id} "
                    f"(Speaker: {recognition_result.identified_speaker_id or 'Unknown'}, "
                    f"Score: {recognition_result.verification_score:.3f}, "
                    f"Time: {processing_time:.2f}s)"
                )

                return recognition_result

        except Exception as e:
            # Emit error event
            await self.event_bus.emit(
                SpeakerProcessingError(
                    session_id=request.session_id, error_type=type(e).__name__, error_message=str(e)
                )
            )

            self.metrics.increment("speaker_recognition_errors_total")
            self.logger.error(
                f"Speaker recognition failed for session {request.session_id}: {str(e)}"
            )
            raise SpeakerRecognitionError(f"Speaker recognition failed: {str(e)}") from e

    async def enroll_speaker(self, enrollment_request: EnrollmentRequest) -> VoiceProfile:
        """
        Enroll a new speaker with comprehensive voice profiling.

        Args:
            enrollment_request: Speaker enrollment request

        Returns:
            Created voice profile
        """
        start_time = datetime.now(timezone.utc)

        # Validate enrollment request
        if not enrollment_request.validate():
            raise SpeakerRecognitionError("Invalid enrollment request")

        session_id = self._generate_session_id()

        try:
            with self.tracer.trace("speaker_enrollment") as span:
                span.set_attributes(
                    {
                        "speaker_id": enrollment_request.speaker_id,
                        "num_samples": len(enrollment_request.audio_samples),
                        "session_id": session_id,
                    }
                )

                # Create new voice profile
                profile = self.profile_manager.create_profile(
                    enrollment_request.speaker_id, enrollment_request.speaker_name
                )

                embeddings = []
                quality_scores = []
                voice_characteristics_list = []

                # Process each audio sample
                for i, audio_sample in enumerate(enrollment_request.audio_samples):
                    try:
                        # Process audio
                        if isinstance(audio_sample, (str, Path)):
                            audio_data, _ = await self.audio_processor.load_audio(
                                audio_sample, target_sr=self.sample_rate
                            )
                        else:
                            audio_data = audio_sample

                        # Preprocess audio
                        processed_audio = await self._preprocess_for_enrollment(audio_data)

                        # Quality assessment
                        quality = await self._assess_audio_quality(processed_audio)
                        quality_score = quality.get("overall_quality", 0.0)

                        if quality_score < enrollment_request.quality_threshold:
                            self.logger.warning(f"Sample {i} quality too low: {quality_score:.3f}")
                            continue

                        # Extract embedding
                        embedding = await self._extract_speaker_embedding(processed_audio)
                        embeddings.append(embedding)
                        quality_scores.append(quality_score)

                        # Extract voice characteristics
                        characteristics = await self._extract_voice_characteristics(processed_audio)
                        voice_characteristics_list.append(characteristics)

                    except Exception as e:
                        self.logger.warning(f"Failed to process sample {i}: {str(e)}")
                        continue

                if len(embeddings) == 0:
                    raise SpeakerRecognitionError("No valid audio samples for enrollment")

                # Compute average embedding
                embeddings_array = np.array(embeddings)
                primary_embedding = np.mean(embeddings_array, axis=0)
                primary_embedding = primary_embedding / np.linalg.norm(primary_embedding)

                # Compute voice characteristics statistics
                aggregated_characteristics = self._aggregate_voice_characteristics(
                    voice_characteristics_list
                )

                # Update profile
                profile.primary_embedding = primary_embedding
                profile.embeddings_history = embeddings
                profile.statistical_features = aggregated_characteristics
                profile.enrollment_quality = np.mean(quality_scores)

                # Extract biometric characteristics
                profile.fundamental_frequency_range = self._extract_f0_range(
                    voice_characteristics_list
                )
                profile.formant_characteristics = self._extract_formant_characteristics(
                    voice_characteristics_list
                )
                profile.speaking_rate_range = self._extract_speaking_rate_range(
                    voice_characteristics_list
                )

                # Set security features
                if self.security_enabled:
                    profile.security_features = await self._extract_security_features(
                        embeddings_array
                    )
                    profile.spoofing_resistance_score = await self._assess_spoofing_resistance(
                        embeddings_array
                    )

                # Set demographics if provided
                if enrollment_request.demographics:
                    profile.age_group = enrollment_request.demographics.get("age_group")
                    profile.gender = enrollment_request.demographics.get("gender")
                    profile.language = enrollment_request.demographics.get("language")
                    profile.accent = enrollment_request.demographics.get("accent")

                # Save updated profile
                self.profile_manager.update_profile(profile.speaker_id, **asdict(profile))

                # Emit enrollment event
                await self.event_bus.emit(
                    SpeakerRegistered(
                        speaker_id=profile.speaker_id,
                        speaker_name=profile.name,
                        enrollment_quality=profile.enrollment_quality,
                        session_id=session_id,
                    )
                )

                # Update metrics
                self.metrics.set("speaker_profile_count", len(self.profile_manager.profiles))

                enrollment_time = (datetime.now(timezone.utc) - start_time).total_seconds()
                self.logger.info(
                    f"Speaker {profile.speaker_id} enrolled successfully "
                    f"(Quality: {profile.enrollment_quality:.3f}, Time: {enrollment_time:.2f}s)"
                )

                return profile

        except Exception as e:
            self.logger.error(f"Speaker enrollment failed: {str(e)}")

            # Clean up partial profile if created
            if hasattr(profile, "speaker_id"):
                self.profile_manager.delete_profile(profile.speaker_id)

            raise SpeakerRecognitionError(f"Speaker enrollment failed: {str(e)}") from e

    async def _process_audio_input(
        self, audio: Union[np.ndarray, str, Path], request: SpeakerRecognitionRequest
    ) -> np.ndarray:
        """Process and prepare audio input for speaker recognition."""
        if isinstance(audio, (str, Path)):
            audio_data, sr = await self.audio_processor.load_audio(
                audio, target_sr=self.sample_rate, normalize=True
            )
        else:
            audio_data = audio.copy()

        # Basic preprocessing
        audio_data = await self._preprocess_for_recognition(audio_data)

        # Validate audio duration
        duration = len(audio_data) / self.sample_rate
        if duration < self.min_audio_duration:
            raise SpeakerRecognitionError(
                f"Audio too short: {duration:.2f}s (minimum: {self.min_audio_duration}s)"
            )

        if duration > self.max_audio_duration:
            # Truncate to maximum duration
            max_samples = int(self.max_audio_duration * self.sample_rate)
            audio_data = audio_data[:max_samples]

        return audio_data

    async def _preprocess_for_recognition(self, audio: np.ndarray) -> np.ndarray:
        """Apply preprocessing optimized for speaker recognition."""
        # Normalize audio
        audio = self.audio_processor.normalize_audio(audio, target_db=-23.0)

        # Apply gentle noise reduction
        audio = self.audio_processor.advanced_noise_reduction(
            audio, self.sample_rate, method="spectral_gating"
        )

        # Trim silence but preserve speaker characteristics
        audio = self.audio_processor.trim_silence(audio, threshold_db=-35.0)

        return audio

    async def _preprocess_for_enrollment(self, audio: np.ndarray) -> np.ndarray:
        """Apply preprocessing specifically for enrollment."""
        # More aggressive preprocessing for enrollment
        audio = self.audio_processor.normalize_audio(audio, target_db=-20.0)

        # Noise reduction
        audio = self.audio_processor.advanced_noise_reduction(
            audio, self.sample_rate, method="spectral_gating"
        )

        # Remove silence
        audio = self.audio_processor.trim_silence(audio, threshold_db=-40.0)

        return audio

    async def _assess_audio_quality(self, audio: np.ndarray) -> Dict[str, float]:
        """Assess audio quality for speaker recognition."""
        quality_metrics = {}

        # Signal-to-noise ratio estimation
        rms_energy = np.sqrt(np.mean(audio**2))
        quality_metrics["rms_energy"] = float(rms_energy)

        # Dynamic range
        dynamic_range = np.max(np.abs(audio)) - np.mean(np.abs(audio))
        quality_metrics["dynamic_range"] = float(dynamic_range)

        # Spectral quality
        stft = librosa.stft(audio)
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate))
        quality_metrics["spectral_centroid"] = float(spectral_centroid)

        # Voice activity ratio
        energy_threshold = np.mean(rms_energy) * 0.1
        voice_frames = np.sum(np.abs(audio) > energy_threshold)
        voice_ratio = voice_frames / len(audio)
        quality_metrics["voice_activity_ratio"] = float(voice_ratio)

        # Overall quality score
        quality_score = (
            min(rms_energy * 10, 1.0) * 0.3
            + min(dynamic_range * 2, 1.0) * 0.2
            + min(spectral_centroid / 2000, 1.0) * 0.2
            + voice_ratio * 0.3
        )
        quality_metrics["overall_quality"] = quality_score

        return quality_metrics

    async def _detect_voice_activity(self, audio: np.ndarray) -> Dict[str, Any]:
        """Detect voice activity in audio."""
        # Energy-based VAD
        frame_length = int(0.025 * self.sample_rate)
        hop_length = int(0.010 * self.sample_rate)

        frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=hop_length)
        frame_energies = np.sum(frames**2, axis=0)

        # Threshold-based detection
        energy_threshold = np.mean(frame_energies) * self.vad_threshold
        voice_frames = frame_energies > energy_threshold

        voice_ratio = np.sum(voice_frames) / len(voice_frames)
        has_speech = voice_ratio > 0.3  # At least 30% voice activity

        return {
            "has_speech": has_speech,
            "voice_activity_ratio": float(voice_ratio),
            "voice_frames": voice_frames.tolist(),
        }

    async def _detect_spoofing(
        self, audio: np.ndarray, request: SpeakerRecognitionRequest
    ) -> Dict[str, Any]:
        """Detect spoofing attacks in audio."""
        try:
            spoofing_result = self.embedding_extractor.detect_spoofing(audio)

            # Emit security alert if spoofing detected
            if spoofing_result["is_spoofed"]:
                await self.event_bus.emit(
                    SecurityAlert(
                        alert_type="spoofing_detected",
                        severity="high",
                        details=spoofing_result,
                        session_id=request.session_id,
                        user_id=request.user_id,
                    )
                )

                self.metrics.increment("spoofing_attempts_detected_total")

            return spoofing_result

        except Exception as e:
            self.logger.warning(f"Spoofing detection failed: {str(e)}")
            return {
                "is_spoofed": False,
                "spoofing_type": SpoofingType.NONE.value,
                "confidence": 0.0,
            }

    async def _extract_speaker_embedding(self, audio: np.ndarray) -> np.ndarray:
        """Extract speaker embedding from audio."""
        try:
            embedding = self.embedding_extractor.extract_embedding(audio)
            return embedding

        except Exception as e:
            raise SpeakerRecognitionError(f"Embedding extraction failed: {str(e)}") from e

    async def _identify_speaker(
        self, embedding: np.ndarray, request: SpeakerRecognitionRequest
    ) -> SpeakerRecognitionResult:
        """Identify speaker from embedding."""
        result = SpeakerRecognitionResult()

        # Find similar profiles
        similar_profiles = self.profile_manager.get_similar_profiles(
            embedding, threshold=self.identification_threshold
        )

        if similar_profiles:
            # Best match
            best_speaker_id, best_score = similar_profiles[0]
            result.identified_speaker_id = best_speaker_id
            result.verification_score = best_score
            result.confidence_score = best_score
            result.verification_decision = best_score >= self.verification_threshold

            # All candidates
            result.speaker_candidates = [
                {"speaker_id": speaker_id, "score": score, "confidence": score}
                for speaker_id, score in similar_profiles[:5]  # Top 5 candidates
            ]
        else:
            result.verification_decision = False
            result.verification_score = 0.0
            result.confidence_score = 0.0

        return result

    async def _verify_speaker(
        self, embedding: np.ndarray, request: SpeakerRecognitionRequest
    ) -> SpeakerRecognitionResult:
        """Verify speaker against target profile."""
        result = SpeakerRecognitionResult()

        if not request.target_speaker_id:
            raise SpeakerRecognitionError("Target speaker ID required for verification")

        target_profile = self.profile_manager.get_profile(request.target_speaker_id)
        if not target_profile or target_profile.primary_embedding is None:
            raise SpeakerRecognitionError(
                f"Target speaker profile not found: {request.target_speaker_id}"
            )

        # Calculate similarity
        similarity = cosine_similarity(
            embedding.reshape(1, -1), target_profile.primary_embedding.reshape(1, -1)
        )[0, 0]

        # Apply security level threshold
        threshold = self.security_thresholds[request.security_level]

        result.identified_speaker_id = request.target_speaker_id
        result.verification_score = float(similarity)
        result.confidence_score = float(similarity)
        result.verification_decision = similarity >= threshold

        # Update profile statistics
        target_profile.update_verification_stats(result.verification_decision)

        return result

    async def _detect_speaker_presence(
        self, embedding: np.ndarray, request: SpeakerRecognitionRequest
    ) -> SpeakerRecognitionResult:
        """Detect if any known speaker is present."""
        result = SpeakerRecognitionResult()

        # Check against all profiles
        max_similarity = 0.0
        best_speaker = None

        for speaker_id, profile in self.profile_manager.profiles.items():
            if profile.primary_embedding is not None:
                similarity = cosine_similarity(
                    embedding.reshape(1, -1), profile.primary_embedding.reshape(1, -1)
                )[0, 0]

                if similarity > max_similarity:
                    max_similarity = similarity
                    best_speaker = speaker_id

        if max_similarity >= self.identification_threshold:
            result.identified_speaker_id = best_speaker
            result.verification_score = max_similarity
            result.confidence_score = max_similarity
            result.verification_decision = True
        else:
            result.verification_decision = False
            result.verification_score = max_similarity
            result.confidence_score = max_similarity

        return result

    async def _detect_multiple_speakers(
        self, audio: np.ndarray, request: SpeakerRecognitionRequest
    ) -> Dict[str, Any]:
        """Detect multiple speakers in audio."""
        try:
            # Segment audio into overlapping windows
            window_size = int(2.0 * self.sample_rate)  # 2-second windows
            hop_size = int(1.0 * self.sample_rate)  # 1-second hop

            embeddings = []
            timestamps = []

            for i in range(0, len(audio) - window_size, hop_size):
                window = audio[i : i + window_size]

                # Extract embedding for this window
                try:
                    embedding = await self._extract_speaker_embedding(window)
                    embeddings.append(embedding)
                    timestamps.append(i / self.sample_rate)
                except Exception:
                    continue

            if len(embeddings) < 2:
                return {"speakers": [], "segments": []}

            # Cluster embeddings to identify speakers
            embeddings_array = np.array(embeddings)
            clusters = self.speaker_clustering.fit_predict(embeddings_array)

            # Identify unique speakers
            unique_clusters = set(clusters)
            unique_clusters.discard(-1)  # Remove noise cluster

            speakers = []
            segments = []

            for cluster_id in unique_clusters:
                cluster_indices = np.where(clusters == cluster_id)[0]
                cluster_embeddings = embeddings_array[cluster_indices]

                # Average embedding for this speaker
                avg_embedding = np.mean(cluster_embeddings, axis=0)

                # Try to identify this speaker
                similar_profiles = self.profile_manager.get_similar_profiles(
                    avg_embedding, threshold=self.identification_threshold
                )

                speaker_id = similar_profiles[0][0] if similar_profiles else f"unknown_{cluster_id}"
                confidence = similar_profiles[0][1] if similar_profiles else 0.0

                speakers.append(
                    {
                        "speaker_id": speaker_id,
                        "confidence": confidence,
                        "segment_count": len(cluster_indices),
                    }
                )

                # Create segments for this speaker
                for idx in cluster_indices:
                    segments.append(
                        {
                            "speaker_id": speaker_id,
                            "start_time": timestamps[idx],
                            "end_time": timestamps[idx] + 2.0,  # Window size
                            "confidence": confidence,
                        }
                    )

            return {"speakers": speakers, "segments": segments}

        except Exception as e:
            self.logger.warning(f"Multi-speaker detection failed: {str(e)}")
            return {"speakers": [], "segments": []}

    async def _extract_voice_characteristics(self, audio: np.ndarray) -> Dict[str, Any]:
        """Extract comprehensive voice characteristics."""
        characteristics = {}

        try:
            # Fundamental frequency
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio, fmin=50, fmax=500, frame_length=2048
            )
            f0_voiced = f0[voiced_flag] if np.any(voiced_flag) else np.array([])

            if len(f0_voiced) > 0:
                characteristics["f0_mean"] = float(np.mean(f0_voiced))
                characteristics["f0_std"] = float(np.std(f0_voiced))
                characteristics["f0_min"] = float(np.min(f0_voiced))
                characteristics["f0_max"] = float(np.max(f0_voiced))
            else:
                characteristics["f0_mean"] = 0.0
                characteristics["f0_std"] = 0.0
                characteristics["f0_min"] = 0.0
                characteristics["f0_max"] = 0.0

        except Exception as e:
            self.logger.error(f"Error extracting voice characteristics: {str(e)}")
            return {"error": str(e)}
