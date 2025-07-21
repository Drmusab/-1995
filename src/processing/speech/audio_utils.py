"""
Enhanced Audio Processing Utilities
Author: Drmusab
Last Modified: 2025-05-26 13:26:17 UTC

This module provides comprehensive audio processing capabilities integrated with the
AI assistant's architecture, including advanced signal processing, real-time
audio handling, and integration with caching and monitoring systems.
"""

from pathlib import Path
from typing import Optional, Tuple, Union, Dict, Protocol, TypeVar, List, Callable, Any
import asyncio
from datetime import datetime, timezone
from dataclasses import dataclass
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import sounddevice as sd
import soundfile as sf
import librosa
import noisereduce as nr
from scipy import signal, ndimage
from scipy.fft import fft, fftfreq
import webrtcvad

# Core imports
from src.core.error_handling import ErrorHandler, handle_exceptions
from src.core.dependency_injection import Container
from src.observability.monitoring.metrics import MetricsCollector
from src.observability.logging.config import get_logger
from src.integrations.cache.cache_strategy import CacheStrategy


# Type definitions
AudioData = TypeVar('AudioData', bound=np.ndarray)


class AudioFormat(Enum):
    """Supported audio formats."""
    WAV = "wav"
    MP3 = "mp3"
    FLAC = "flac"
    M4A = "m4a"
    OGG = "ogg"
    WEBM = "webm"


class ProcessingMode(Enum):
    """Audio processing modes."""
    REAL_TIME = "real_time"
    BATCH = "batch"
    STREAMING = "streaming"
    HIGH_QUALITY = "high_quality"


@dataclass
class AudioMetadata:
    """Audio metadata container."""
    sample_rate: int
    channels: int
    duration: float
    format: str
    bit_depth: Optional[int] = None
    codec: Optional[str] = None
    file_size: Optional[int] = None
    created_at: Optional[datetime] = None


@dataclass
class ProcessingSettings:
    """Audio processing settings."""
    normalize: bool = True
    trim_silence: bool = True
    noise_reduction: bool = True
    dynamic_range_compression: bool = False
    spectral_gating: bool = True
    adaptive_filtering: bool = True
    real_time_optimization: bool = False
    quality_level: str = "balanced"  # "fast", "balanced", "high_quality"


class AudioProcessingError(Exception):
    """Custom exception for audio processing errors."""
    
    def __init__(self, message: str, error_code: Optional[str] = None):
        super().__init__(message)
        self.error_code = error_code
        self.timestamp = datetime.now(timezone.utc)


class AudioProcessorProtocol(Protocol):
    """Protocol defining the interface for audio processing operations."""
    
    def normalize_audio(self, audio: AudioData, target_db: float = -20.0) -> AudioData: ...
    def trim_silence(self, audio: AudioData, threshold_db: float = -50.0) -> AudioData: ...
    def apply_noise_reduction(self, audio: AudioData, sample_rate: int) -> AudioData: ...


class EnhancedAudioProcessor:
    """
    Advanced audio processing utilities with comprehensive AI assistant integration.
    
    Features:
    - Advanced signal processing with multiple algorithms
    - Real-time audio processing capabilities
    - Intelligent noise reduction and enhancement
    - Voice activity detection (VAD)
    - Audio quality assessment and optimization
    - Caching and performance monitoring
    - Asynchronous processing support
    - Memory-efficient streaming processing
    """

    # Audio processing constants
    DEFAULT_SAMPLE_RATE = 16000
    DEFAULT_CHANNELS = 1
    DEFAULT_DTYPE = np.float32
    
    # Quality presets
    QUALITY_PRESETS = {
        "fast": {
            "frame_length": 1024,
            "hop_length": 256,
            "n_fft": 1024,
            "noise_reduction_strength": 0.3,
            "spectral_gating_strength": 0.5
        },
        "balanced": {
            "frame_length": 2048,
            "hop_length": 512,
            "n_fft": 2048,
            "noise_reduction_strength": 0.5,
            "spectral_gating_strength": 0.7
        },
        "high_quality": {
            "frame_length": 4096,
            "hop_length": 1024,
            "n_fft": 4096,
            "noise_reduction_strength": 0.8,
            "spectral_gating_strength": 0.9
        }
    }

    def __init__(
        self, 
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        container: Optional[Container] = None
    ):
        """
        Initialize the enhanced audio processor.
        
        Args:
            sample_rate: Default sample rate for processing
            container: Dependency injection container
        """
        self.logger = get_logger(__name__)
        self.default_sample_rate = sample_rate
        self.container = container
        
        # Initialize components
        self._setup_processing_parameters()
        self._setup_monitoring()
        self._setup_caching()
        self._setup_threading()
        self._setup_vad()
        
        self.logger.info(f"EnhancedAudioProcessor initialized (SR: {sample_rate})")

    def _setup_processing_parameters(self) -> None:
        """Setup audio processing parameters."""
        self.default_channels: int = self.DEFAULT_CHANNELS
        self.default_dtype = self.DEFAULT_DTYPE
        self.min_silence_duration: float = 0.1
        
        # Use balanced quality as default
        self.current_preset = self.QUALITY_PRESETS["balanced"]
        self.frame_length: int = self.current_preset["frame_length"]
        self.hop_length: int = self.current_preset["hop_length"]
        self.n_fft: int = self.current_preset["n_fft"]
        
        # Advanced processing parameters
        self.noise_reduction_strength = self.current_preset["noise_reduction_strength"]
        self.spectral_gating_strength = self.current_preset["spectral_gating_strength"]
        self.adaptive_filter_length = 256
        self.dynamic_range_target = 12.0  # dB

    def _setup_monitoring(self) -> None:
        """Setup monitoring and metrics collection."""
        if self.container:
            try:
                self.metrics = self.container.get(MetricsCollector)
                # Register audio processing metrics
                self.metrics.register_counter("audio_processing_operations_total")
                self.metrics.register_histogram("audio_processing_duration_seconds")
                self.metrics.register_gauge("audio_processing_quality_score")
                self.metrics.register_counter("audio_processing_errors_total")
            except Exception as e:
                self.logger.warning(f"Failed to setup metrics: {str(e)}")
                self.metrics = None
        else:
            self.metrics = None

    def _setup_caching(self) -> None:
        """Setup caching for processed audio segments."""
        if self.container:
            try:
                self.cache_strategy = self.container.get(CacheStrategy)
                self.cache_enabled = True
            except Exception as e:
                self.logger.warning(f"Failed to setup caching: {str(e)}")
                self.cache_strategy = None
                self.cache_enabled = False
        else:
            self.cache_strategy = None
            self.cache_enabled = False

    def _setup_threading(self) -> None:
        """Setup thread pool for concurrent processing."""
        self.thread_pool = ThreadPoolExecutor(
            max_workers=4, 
            thread_name_prefix="audio_processor"
        )
        self.processing_lock = threading.Lock()

    def _setup_vad(self) -> None:
        """Setup Voice Activity Detection."""
        try:
            # Initialize WebRTC VAD with different aggressiveness levels
            self.vad_modes = {
                0: webrtcvad.Vad(0),  # Least aggressive
                1: webrtcvad.Vad(1),  # Less aggressive
                2: webrtcvad.Vad(2),  # More aggressive
                3: webrtcvad.Vad(3)   # Most aggressive
            }
            self.default_vad_mode = 2
            self.vad_available = True
        except Exception as e:
            self.logger.warning(f"Failed to initialize VAD: {str(e)}")
            self.vad_available = False

    @handle_exceptions
    def set_quality_preset(self, preset: str) -> None:
        """
        Set audio processing quality preset.
        
        Args:
            preset: Quality preset name ("fast", "balanced", "high_quality")
        """
        if preset not in self.QUALITY_PRESETS:
            raise AudioProcessingError(
                f"Invalid quality preset. Available: {list(self.QUALITY_PRESETS.keys())}"
            )
        
        self.current_preset = self.QUALITY_PRESETS[preset]
        self.frame_length = self.current_preset["frame_length"]
        self.hop_length = self.current_preset["hop_length"]
        self.n_fft = self.current_preset["n_fft"]
        self.noise_reduction_strength = self.current_preset["noise_reduction_strength"]
        self.spectral_gating_strength = self.current_preset["spectral_gating_strength"]
        
        self.logger.info(f"Audio processing quality set to: {preset}")

    @handle_exceptions
    async def load_audio(
        self,
        file_path: Union[str, Path],
        target_sr: Optional[int] = None,
        normalize: bool = True
    ) -> Tuple[np.ndarray, int]:
        """
        Enhanced audio file loading with format detection and optimization.
        
        Args:
            file_path: Path to the audio file
            target_sr: Target sampling rate (None to keep original)
            normalize: Whether to normalize the audio
        
        Returns:
            Tuple of (audio_data, sample_rate)
            
        Raises:
            AudioProcessingError: If loading fails
        """
        start_time = datetime.now(timezone.utc)
        
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                raise AudioProcessingError(f"Audio file not found: {file_path}")
            
            # Detect and handle different audio formats
            file_format = file_path.suffix.lower().lstrip('.')
            
            if file_format in ['wav', 'flac', 'aiff']:
                # Use soundfile for lossless formats
                audio, sr = sf.read(str(file_path))
                if len(audio.shape) > 1:
                    audio = audio.mean(axis=1)  # Convert to mono
            else:
                # Use librosa for other formats (MP3, M4A, etc.)
                audio, sr = librosa.load(str(file_path), sr=target_sr, mono=True)
            
            # Resample if needed
            if target_sr and sr != target_sr:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
                sr = target_sr
            
            # Normalize if requested
            if normalize:
                audio = self.normalize_audio(audio)
            
            # Record metrics
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            if self.metrics:
                self.metrics.increment("audio_processing_operations_total")
                self.metrics.record("audio_processing_duration_seconds", processing_time)
            
            self.logger.debug(
                f"Loaded audio file: {file_path.name} "
                f"(Duration: {len(audio)/sr:.2f}s, SR: {sr})"
            )
            
            return audio, sr
            
        except Exception as e:
            if self.metrics:
                self.metrics.increment("audio_processing_errors_total")
            raise AudioProcessingError(f"Failed to load audio file: {str(e)}") from e

    @handle_exceptions
    async def save_audio(
        self,
        audio: np.ndarray,
        file_path: Union[str, Path],
        sample_rate: Optional[int] = None,
        format: AudioFormat = AudioFormat.WAV,
        quality: Optional[str] = None
    ) -> Path:
        """
        Enhanced audio file saving with format optimization.
        
        Args:
            audio: Audio data as numpy array
            file_path: Output file path
            sample_rate: Sampling rate (defaults to default_sample_rate)
            format: Audio format to save
            quality: Quality setting for lossy formats
        
        Returns:
            Path to the saved file
            
        Raises:
            AudioProcessingError: If saving fails
        """
        try:
            file_path = Path(file_path)
            sample_rate = sample_rate or self.default_sample_rate
            
            # Ensure audio is in correct format
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            
            # Handle different output formats
            if format == AudioFormat.WAV:
                sf.write(file_path, audio, sample_rate, format='WAV', subtype='PCM_16')
            elif format == AudioFormat.FLAC:
                sf.write(file_path, audio, sample_rate, format='FLAC')
            elif format == AudioFormat.OGG:
                sf.write(file_path, audio, sample_rate, format='OGG', subtype='VORBIS')
            else:
                # Default to WAV for unsupported formats
                sf.write(file_path, audio, sample_rate)
            
            self.logger.debug(f"Saved audio to: {file_path}")
            return file_path
            
        except Exception as e:
            raise AudioProcessingError(f"Failed to save audio file: {str(e)}") from e

    @handle_exceptions
    def normalize_audio(
        self,
        audio: np.ndarray,
        target_db: float = -20.0,
        method: str = "peak"
    ) -> np.ndarray:
        """
        Enhanced audio normalization with multiple methods.
        
        Args:
            audio: Input audio data
            target_db: Target dB level
            method: Normalization method ("peak", "rms", "lufs")
        
        Returns:
            Normalized audio data
            
        Raises:
            AudioProcessingError: If normalization fails
        """
        try:
            if len(audio) == 0:
                return audio
            
            if method == "peak":
                # Peak normalization
                peak = np.max(np.abs(audio))
                if peak > 0:
                    target_amplitude = 10 ** (target_db / 20.0)
                    audio = audio * (target_amplitude / peak)
            
            elif method == "rms":
                # RMS normalization
                rms = np.sqrt(np.mean(audio ** 2))
                if rms > 0:
                    target_rms = 10 ** (target_db / 20.0)
                    audio = audio * (target_rms / rms)
            
            elif method == "lufs":
                # LUFS-based normalization (simplified)
                # This is a basic implementation; for true LUFS, use specialized libraries
                rms = np.sqrt(np.mean(audio ** 2))
                if rms > 0:
                    # Approximate LUFS calculation
                    lufs = 20 * np.log10(rms) - 0.691
                    target_lufs = target_db
                    gain_db = target_lufs - lufs
                    gain_linear = 10 ** (gain_db / 20.0)
                    audio = audio * gain_linear
            
            else:
                raise AudioProcessingError(f"Unknown normalization method: {method}")
            
            # Prevent clipping
            audio = np.clip(audio, -1.0, 1.0)
            
            return audio
            
        except Exception as e:
            raise AudioProcessingError(f"Normalization failed: {str(e)}") from e

    @handle_exceptions
    def advanced_noise_reduction(
        self,
        audio: np.ndarray,
        sample_rate: int,
        method: str = "spectral_gating",
        noise_profile: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Advanced noise reduction with multiple algorithms.
        
        Args:
            audio: Input audio data
            sample_rate: Audio sampling rate
            method: Noise reduction method
            noise_profile: Optional noise profile for reduction
        
        Returns:
            Noise-reduced audio data
            
        Raises:
            AudioProcessingError: If noise reduction fails
        """
        try:
            if method == "spectral_gating":
                return self._spectral_gating_nr(audio, sample_rate)
            elif method == "spectral_subtraction":
                return self._spectral_subtraction_nr(audio, sample_rate, noise_profile)
            elif method == "wiener_filter":
                return self._wiener_filter_nr(audio, sample_rate)
            else:
                self.logger.warning(f"Unknown noise reduction method: {method}")
                return audio
                
        except Exception as e:
            self.logger.error(f"Noise reduction failed: {str(e)}")
            return audio
