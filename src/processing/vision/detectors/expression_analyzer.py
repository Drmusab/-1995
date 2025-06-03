"""
Advanced Facial Expression Analysis System
Author: Drmusab
Last Modified: 2025-06-03 19:34:15 UTC

This module provides comprehensive facial expression analysis capabilities for the AI assistant,
including real-time emotion detection, micro-expression analysis, attention assessment, and
seamless integration with the core system architecture.
"""

from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Union, AsyncGenerator, Set, Callable
import asyncio
import threading
import time
import numpy as np
import cv2
from datetime import datetime, timezone
from dataclasses import dataclass, field
from enum import Enum
from contextlib import asynccontextmanager
import logging
import json
import hashlib
from collections import deque, defaultdict
import weakref
from abc import ABC, abstractmethod
import uuid
import warnings
from concurrent.futures import ThreadPoolExecutor
import math

# Core imports
from src.core.config.loader import ConfigLoader
from src.core.events.event_bus import EventBus
from src.core.events.event_types import (
    ComponentRegistered, ComponentInitialized, ComponentStarted,
    ComponentStopped, ComponentFailed, ComponentHealthChanged,
    ProcessingStarted, ProcessingCompleted, ProcessingError,
    VisionProcessingStarted, VisionProcessingCompleted,
    EmotionDetected, ExpressionAnalyzed, AttentionAssessed
)
from src.core.error_handling import ErrorHandler, handle_exceptions
from src.core.dependency_injection import Container
from src.core.health_check import HealthCheck

# Assistant components
from src.assistant.component_manager import ComponentInterface, ComponentMetadata, ComponentPriority
from src.assistant.session_manager import SessionContext

# Processing components
from src.processing.vision.vision_processor import VisionProcessor
from src.processing.vision.image_analyzer import ImageAnalyzer

# Observability
from src.observability.monitoring.metrics import MetricsCollector
from src.observability.monitoring.tracing import TraceManager
from src.observability.logging.config import get_logger

# Memory and caching
from src.integrations.cache.cache_strategy import CacheStrategy
from src.memory.cache_manager import CacheManager

# Learning and adaptation
from src.learning.feedback_processor import FeedbackProcessor

# Optional dependencies (with graceful fallbacks)
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    warnings.warn("MediaPipe not available. Some expression analysis features will be disabled.")

try:
    import dlib
    DLIB_AVAILABLE = True
except ImportError:
    DLIB_AVAILABLE = False
    warnings.warn("dlib not available. Some facial landmark features will be disabled.")

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision.transforms as transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. Some expression analysis features will be disabled.")

try:
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("scikit-learn not available. Some analysis features will be disabled.")


class ExpressionModel(Enum):
    """Available expression analysis models."""
    MEDIAPIPE = "mediapipe"
    DLIB = "dlib"
    PYTORCH_CNN = "pytorch_cnn"
    OPENCV_DNN = "opencv_dnn"
    ENSEMBLE = "ensemble"
    AUTO = "auto"


class EmotionType(Enum):
    """Basic emotion types (Ekman's 6 + neutral)."""
    NEUTRAL = "neutral"
    HAPPINESS = "happiness"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    
    # Extended emotions
    CONTEMPT = "contempt"
    EXCITEMENT = "excitement"
    CONFUSION = "confusion"
    CONCENTRATION = "concentration"
    BOREDOM = "boredom"
    ANXIETY = "anxiety"
    RELIEF = "relief"
    PRIDE = "pride"
    SHAME = "shame"
    GUILT = "guilt"
    LOVE = "love"
    JEALOUSY = "jealousy"
    UNKNOWN = "unknown"


class ExpressionIntensity(Enum):
    """Intensity levels of expressions."""
    SUBTLE = "subtle"        # Barely noticeable
    MILD = "mild"           # Clearly visible
    MODERATE = "moderate"    # Prominent
    STRONG = "strong"       # Very prominent
    EXTREME = "extreme"     # Overwhelming


class FacialRegion(Enum):
    """Facial regions for detailed analysis."""
    FOREHEAD = "forehead"
    EYEBROWS = "eyebrows"
    EYES = "eyes"
    NOSE = "nose"
    CHEEKS = "cheeks"
    MOUTH = "mouth"
    CHIN = "chin"
    JAW = "jaw"
    OVERALL = "overall"


class AnalysisMode(Enum):
    """Expression analysis modes."""
    REAL_TIME = "real_time"
    DETAILED = "detailed"
    MICRO_EXPRESSION = "micro_expression"
    CONTINUOUS = "continuous"
    COMPARATIVE = "comparative"


class AttentionState(Enum):
    """Attention and engagement states."""
    FOCUSED = "focused"
    DISTRACTED = "distracted"
    ENGAGED = "engaged"
    DISENGAGED = "disengaged"
    CONFUSED = "confused"
    INTERESTED = "interested"
    BORED = "bored"
    OVERWHELMED = "overwhelmed"
    UNKNOWN = "unknown"


class QualityLevel(Enum):
    """Quality levels for expression analysis."""
    FAST = "fast"           # Speed optimized
    BALANCED = "balanced"   # Speed/accuracy balance
    ACCURATE = "accurate"   # Accuracy optimized
    RESEARCH = "research"   # Maximum accuracy for research


@dataclass
class FacialLandmark:
    """Individual facial landmark point."""
    x: float
    y: float
    z: float = 0.0
    confidence: float = 0.0
    landmark_id: Optional[int] = None
    region: Optional[FacialRegion] = None


@dataclass
class FaceGeometry:
    """Face geometric properties."""
    bbox: Tuple[float, float, float, float]  # x, y, width, height
    landmarks: List[FacialLandmark]
    pose_angles: Tuple[float, float, float]  # pitch, yaw, roll
    face_area: float
    landmark_quality: float
    symmetry_score: float
    
    # Face measurements
    eye_distance: float = 0.0
    face_width: float = 0.0
    face_height: float = 0.0
    mouth_width: float = 0.0
    nose_width: float = 0.0


@dataclass
class EmotionScore:
    """Individual emotion with confidence score."""
    emotion: EmotionType
    confidence: float
    intensity: ExpressionIntensity
    
    # Regional scores
    regional_scores: Dict[FacialRegion, float] = field(default_factory=dict)
    
    # Temporal information
    onset_time: Optional[datetime] = None
    peak_time: Optional[datetime] = None
    offset_time: Optional[datetime] = None
    duration_ms: float = 0.0
    
    # Additional metrics
    authenticity: float = 0.0  # How genuine the emotion appears
    stability: float = 0.0     # How stable over time


@dataclass
class MicroExpression:
    """Detected micro-expression."""
    expression_id: str
    emotion: EmotionType
    confidence: float
    duration_ms: float
    
    # Timing
    start_frame: int
    end_frame: int
    peak_frame: int
    
    # Characteristics
    intensity: ExpressionIntensity
    suppression_detected: bool = False
    leaked_emotion: Optional[EmotionType] = None
    
    # Facial regions involved
    primary_regions: List[FacialRegion] = field(default_factory=list)
    movement_vectors: Dict[FacialRegion, Tuple[float, float]] = field(default_factory=dict)


@dataclass
class ExpressionAnalysisRequest:
    """Request for expression analysis."""
    image: np.ndarray
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    
    # Analysis parameters
    model: ExpressionModel = ExpressionModel.AUTO
    mode: AnalysisMode = AnalysisMode.REAL_TIME
    quality: QualityLevel = QualityLevel.BALANCED
    
    # Options
    detect_micro_expressions: bool = False
    analyze_attention: bool = True
    track_changes: bool = True
    cultural_context: Optional[str] = None
    age_range: Optional[Tuple[int, int]] = None
    gender_hint: Optional[str] = None
    
    # Technical parameters
    min_face_size: int = 50
    max_faces: int = 10
    confidence_threshold: float = 0.5
    enable_face_tracking: bool = True
    
    # Context
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    frame_number: int = 0
    previous_analysis: Optional['ExpressionAnalysisResult'] = None
    
    # Processing hints
    roi: Optional[Tuple[int, int, int, int]] = None  # Region of interest
    expected_emotion: Optional[EmotionType] = None
    processing_priority: int = 1
    timeout_seconds: float = 5.0


@dataclass
class ExpressionAnalysisResult:
    """Complete expression analysis result."""
    success: bool
    request_id: str
    processing_time: float
    
    # Face detection results
    faces_detected: int
    face_geometries: List[FaceGeometry]
    
    # Expression analysis results
    primary_emotions: List[EmotionScore]
    secondary_emotions: List[EmotionScore]
    emotion_blend: Dict[EmotionType, float]
    
    # Micro-expression analysis
    micro_expressions: List[MicroExpression]
    suppressed_emotions: List[EmotionType]
    
    # Attention and engagement
    attention_state: AttentionState
    engagement_level: float  # 0.0 to 1.0
    focus_direction: Optional[Tuple[float, float]] = None
    eye_contact_probability: float = 0.0
    
    # Temporal analysis
    emotion_stability: float = 0.0
    transition_detected: bool = False
    emotion_changes: List[Dict[str, Any]] = field(default_factory=list)
    
    # Quality metrics
    analysis_confidence: float = 0.0
    landmark_quality: float = 0.0
    lighting_quality: float = 0.0
    pose_quality: float = 0.0
    
    # Metadata
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    model_used: ExpressionModel = ExpressionModel.AUTO
    frame_number: int = 0
    
    # Cultural and demographic context
    cultural_adjustments: Dict[str, float] = field(default_factory=dict)
    demographic_confidence: Dict[str, float] = field(default_factory=dict)
    
    # Additional insights
    personality_indicators: Dict[str, float] = field(default_factory=dict)
    social_signals: Dict[str, float] = field(default_factory=dict)
    interaction_cues: List[str] = field(default_factory=list)
    
    # Error information
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Debug information
    debug_info: Dict[str, Any] = field(default_factory=dict)
    processing_steps: List[str] = field(default_factory=list)


@dataclass
class ExpressionConfiguration:
    """Configuration for expression analysis."""
    # Model settings
    primary_model: ExpressionModel = ExpressionModel.AUTO
    fallback_models: List[ExpressionModel] = field(default_factory=list)
    ensemble_models: List[ExpressionModel] = field(default_factory=list)
    
    # Analysis settings
    default_quality: QualityLevel = QualityLevel.BALANCED
    enable_micro_expressions: bool = False
    enable_attention_analysis: bool = True
    enable_cultural_adaptation: bool = True
    
    # Detection thresholds
    emotion_confidence_threshold: float = 0.5
    micro_expression_threshold: float = 0.7
    attention_threshold: float = 0.6
    face_detection_threshold: float = 0.8
    
    # Temporal settings
    emotion_smoothing_frames: int = 5
    micro_expression_window_ms: float = 500.0
    attention_analysis_window_ms: float = 2000.0
    emotion_transition_threshold: float = 0.3
    
    # Performance settings
    max_concurrent_analyses: int = 4
    enable_gpu_acceleration: bool = True
    batch_processing_size: int = 8
    cache_face_detections: bool = True
    
    # Face processing
    face_tracking_enabled: bool = True
    max_face_tracking_distance: float = 50.0
    landmark_detection_model: str = "mediapipe"
    face_alignment_enabled: bool = True
    
    # Quality control
    min_face_resolution: int = 64
    max_pose_angle: float = 45.0
    min_lighting_quality: float = 0.3
    blur_detection_threshold: float = 100.0
    
    # Cultural adaptation
    default_cultural_context: str = "western"
    cultural_emotion_mappings: Dict[str, Dict[str, float]] = field(default_factory=dict)
    age_adaptation_enabled: bool = True
    gender_adaptation_enabled: bool = True
    
    # Privacy and ethics
    anonymize_results: bool = False
    emotion_data_retention_hours: int = 24
    consent_required: bool = True
    ethical_guidelines_enabled: bool = True
    
    # Caching
    enable_result_caching: bool = True
    cache_ttl_seconds: int = 300
    cache_size_limit: int = 1000
    
    # Learning and adaptation
    enable_online_learning: bool = False
    feedback_learning_rate: float = 0.01
    model_update_threshold: int = 100
    
    # Debug and monitoring
    enable_debug_output: bool = False
    save_debug_images: bool = False
    performance_monitoring: bool = True
    detailed_logging: bool = False


class ExpressionAnalysisError(Exception):
    """Custom exception for expression analysis operations."""
    
    def __init__(self, message: str, error_code: Optional[str] = None, 
                 request_id: Optional[str] = None, component: Optional[str] = None):
        super().__init__(message)
        self.error_code = error_code
        self.request_id = request_id
        self.component = component
        self.timestamp = datetime.now(timezone.utc)


class ExpressionAnalyzer(ABC):
    """Abstract base class for expression analyzers."""
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the expression analyzer."""
        pass
    
    @abstractmethod
    async def analyze_expression(
        self, 
        request: ExpressionAnalysisRequest
    ) -> ExpressionAnalysisResult:
        """Analyze facial expressions in an image."""
        pass
    
    @abstractmethod
    def supports_model(self, model: ExpressionModel) -> bool:
        """Check if the analyzer supports a specific model."""
        pass
    
    def cleanup(self) -> None:
        """Cleanup resources."""
        pass


class MediaPipeExpressionAnalyzer(ExpressionAnalyzer):
    """MediaPipe-based expression analyzer."""
    
    def __init__(self, logger, config: ExpressionConfiguration):
        self.logger = logger
        self.config = config
        self.face_mesh = None
        self.face_detection = None
        self.initialized = False
        
        # Emotion classification model (simplified)
        self.emotion_classifier = None
        
    async def initialize(self) -> None:
        """Initialize MediaPipe components."""
        if not MEDIAPIPE_AVAILABLE:
            raise ExpressionAnalysisError("MediaPipe not available")
        
        try:
            # Initialize MediaPipe solutions
            self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                max_num_faces=self.config.max_concurrent_analyses,
                refine_landmarks=True,
                min_detection_confidence=self.config.face_detection_threshold,
                min_tracking_confidence=0.5
            )
            
            self.face_detection = mp.solutions.face_detection.FaceDetection(
                model_selection=0,
                min_detection_confidence=self.config.face_detection_threshold
            )
            
            # Initialize emotion classifier
            await self._initialize_emotion_classifier()
            
            self.initialized = True
            self.logger.info("MediaPipe expression analyzer initialized")
            
        except Exception as e:
            raise ExpressionAnalysisError(f"Failed to initialize MediaPipe: {str(e)}")
    
    async def _initialize_emotion_classifier(self) -> None:
        """Initialize the emotion classification model."""
        # This would load a pre-trained emotion classification model
        # For now, we'll use a simplified rule-based approach
        self.emotion_classifier = SimplifiedEmotionClassifier()
        await self.emotion_classifier.initialize()
    
    def supports_model(self, model: ExpressionModel) -> bool:
        """Check if MediaPipe model is supported."""
        return model in [ExpressionModel.MEDIAPIPE, ExpressionModel.AUTO]
    
    async def analyze_expression(
        self, 
        request: ExpressionAnalysisRequest
    ) -> ExpressionAnalysisResult:
        """Analyze expressions using MediaPipe."""
        if not self.initialized:
            await self.initialize()
        
        start_time = time.time()
        
        try:
            # Convert image format
            rgb_image = cv2.cvtColor(request.image, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            face_results = self.face_detection.process(rgb_image)
            faces_detected = len(face_results.detections) if face_results.detections else 0
            
            if faces_detected == 0:
                return self._create_empty_result(request, time.time() - start_time)
            
            # Analyze each face
            face_geometries = []
            all_emotions = []
            
            for detection in face_results.detections:
                # Extract face region
                bbox = self._get_face_bbox(detection, request.image.shape[:2])
                face_roi = self._extract_face_roi(request.image, bbox)
                
                # Get face landmarks
                mesh_results = self.face_mesh.process(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB))
                
                if mesh_results.multi_face_landmarks:
                    landmarks = mesh_results.multi_face_landmarks[0]
                    
                    # Create face geometry
                    face_geometry = self._create_face_geometry(landmarks, bbox)
                    face_geometries.append(face_geometry)
                    
                    # Analyze emotions
                    emotions = await self._analyze_face_emotions(face_roi, face_geometry)
                    all_emotions.extend(emotions)
            
            # Create result
            result = self._create_analysis_result(
                request, face_geometries, all_emotions, time.time() - start_time
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"MediaPipe expression analysis failed: {str(e)}")
            return self._create_error_result(request, str(e), time.time() - start_time)
    
    def _get_face_bbox(self, detection, image_shape: Tuple[int, int]) -> Tuple[float, float, float, float]:
        """Extract bounding box from face detection."""
        bbox = detection.location_data.relative_bounding_box
        h, w = image_shape
        
        x = bbox.xmin * w
        y = bbox.ymin * h
        width = bbox.width * w
        height = bbox.height * h
        
        return (x, y, width, height)
    
    def _extract_face_roi(self, image: np.ndarray, bbox: Tuple[float, float, float, float]) -> np.ndarray:
        """Extract face region of interest."""
        x, y, width, height = bbox
        x, y, width, height = int(x), int(y), int(width), int(height)
        
        # Add padding
        padding = int(min(width, height) * 0.1)
        x = max(0, x - padding)
        y = max(0, y - padding)
        width = min(image.shape[1] - x, width + 2 * padding)
        height = min(image.shape[0] - y, height + 2 * padding)
        
        return image[y:y+height, x:x+width]
    
    def _create_face_geometry(self, landmarks, bbox: Tuple[float, float, float, float]) -> FaceGeometry:
        """Create face geometry from landmarks."""
        # Convert landmarks to our format
        facial_landmarks = []
        for i, landmark in enumerate(landmarks.landmark):
            facial_landmarks.append(FacialLandmark(
                x=landmark.x,
                y=landmark.y,
                z=landmark.z,
                confidence=1.0,  # MediaPipe doesn't provide per-landmark confidence
                landmark_id=i
            ))
        
        # Calculate pose angles (simplified)
        pose_angles = self._calculate_pose_angles(facial_landmarks)
        
        # Calculate face measurements
        face_area = bbox[2] * bbox[3]
        
        return FaceGeometry(
            bbox=bbox,
            landmarks=facial_landmarks,
            pose_angles=pose_angles,
            face_area=face_area,
            landmark_quality=0.8,  # Simplified
            symmetry_score=0.8,    # Simplified
            eye_distance=self._calculate_eye_distance(facial_landmarks),
            face_width=bbox[2],
            face_height=bbox[3]
        )
    
    def _calculate_pose_angles(self, landmarks: List[FacialLandmark]) -> Tuple[float, float, float]:
        """Calculate face pose angles from landmarks."""
        # Simplified pose estimation
        # In a real implementation, this would use proper 3D pose estimation
        return (0.0, 0.0, 0.0)
    
    def _calculate_eye_distance(self, landmarks: List[FacialLandmark]) -> float:
        """Calculate distance between eyes."""
        # Simplified calculation using landmark indices for eyes
        # MediaPipe face mesh has specific indices for eye corners
        if len(landmarks) > 468:  # MediaPipe has 468 landmarks
            left_eye = landmarks[33]   # Left eye corner
            right_eye = landmarks[263] # Right eye corner
            
            distance = math.sqrt(
                (left_eye.x - right_eye.x) ** 2 + 
                (left_eye.y - right_eye.y) ** 2
            )
            return distance
        
        return 0.0
    
    async def _analyze_face_emotions(
        self, 
        face_roi: np.ndarray, 
        face_geometry: FaceGeometry
    ) -> List[EmotionScore]:
        """Analyze emotions in a face region."""
        if self.emotion_classifier:
            return await self.emotion_classifier.classify_emotions(face_roi, face_geometry)
        
        # Fallback: return neutral emotion
        return [EmotionScore(
            emotion=EmotionType.NEUTRAL,
            confidence=0.5,
            intensity=ExpressionIntensity.MILD
        )]
    
    def _create_empty_result(self, request: ExpressionAnalysisRequest, processing_time: float) -> ExpressionAnalysisResult:
        """Create empty result when no faces detected."""
        return ExpressionAnalysisResult(
            success=True,
            request_id=request.request_id,
            processing_time=processing_time,
            faces_detected=0,
            face_geometries=[],
            primary_emotions=[],
            secondary_emotions=[],
            emotion_blend={},
            micro_expressions=[],
            suppressed_emotions=[],
            attention_state=AttentionState.UNKNOWN,
            engagement_level=0.0,
            model_used=ExpressionModel.MEDIAPIPE,
            frame_number=request.frame_number
        )
    
    def _create_analysis_result(
        self,
        request: ExpressionAnalysisRequest,
        face_geometries: List[FaceGeometry],
        emotions: List[EmotionScore],
        processing_time: float
    ) -> ExpressionAnalysisResult:
        """Create comprehensive analysis result."""
        # Sort emotions by confidence
        primary_emotions = sorted(emotions, key=lambda x: x.confidence, reverse=True)[:3]
        secondary_emotions = sorted(emotions, key=lambda x: x.confidence, reverse=True)[3:6]
        
        # Create emotion blend
        emotion_blend = {}
        for emotion in emotions:
            emotion_blend[emotion.emotion] = emotion.confidence
        
        # Analyze attention (simplified)
        attention_state = self._analyze_attention(face_geometries)
        engagement_level = self._calculate_engagement(primary_emotions)
        
        return ExpressionAnalysisResult(
            success=True,
            request_id=request.request_id,
            processing_time=processing_time,
            faces_detected=len(face_geometries),
            face_geometries=face_geometries,
            primary_emotions=primary_emotions,
            secondary_emotions=secondary_emotions,
            emotion_blend=emotion_blend,
            micro_expressions=[],  # Not implemented in this version
            suppressed_emotions=[],
            attention_state=attention_state,
            engagement_level=engagement_level,
            analysis_confidence=self._calculate_overall_confidence(emotions),
            model_used=ExpressionModel.MEDIAPIPE,
            frame_number=request.frame_number
        )
    
    def _analyze_attention(self, face_geometries: List[FaceGeometry]) -> AttentionState:
        """Analyze attention state from face geometries."""
        if not face_geometries:
            return AttentionState.UNKNOWN
        
        # Simplified attention analysis based on pose
        face = face_geometries[0]
        pitch, yaw, roll = face.pose_angles
        
        if abs(yaw) < 15 and abs(pitch) < 10:
            return AttentionState.FOCUSED
        elif abs(yaw) > 30 or abs(pitch) > 20:
            return AttentionState.DISTRACTED
        else:
            return AttentionState.ENGAGED
    
    def _calculate_engagement(self, emotions: List[EmotionScore]) -> float:
        """Calculate engagement level from emotions."""
        if not emotions:
            return 0.0
        
        # Higher engagement for positive emotions and attention-related states
        engagement_emotions = {
            EmotionType.HAPPINESS: 0.8,
            EmotionType.SURPRISE: 0.7,
            EmotionType.EXCITEMENT: 0.9,
            EmotionType.CONCENTRATION: 0.8,
            EmotionType.NEUTRAL: 0.5
        }
        
        total_engagement = 0.0
        total_confidence = 0.0
        
        for emotion in emotions:
            weight = engagement_emotions.get(emotion.emotion, 0.3)
            total_engagement += weight * emotion.confidence
            total_confidence += emotion.confidence
        
        return total_engagement / max(total_confidence, 1.0)
    
    def _calculate_overall_confidence(self, emotions: List[EmotionScore]) -> float:
        """Calculate overall analysis confidence."""
        if not emotions:
            return 0.0
        
        return sum(emotion.confidence for emotion in emotions) / len(emotions)
    
    def _create_error_result(
        self, 
        request: ExpressionAnalysisRequest, 
        error_message: str, 
        processing_time: float
    ) -> ExpressionAnalysisResult:
        """Create error result."""
        return ExpressionAnalysisResult(
            success=False,
            request_id=request.request_id,
            processing_time=processing_time,
            faces_detected=0,
            face_geometries=[],
            primary_emotions=[],
            secondary_emotions=[],
            emotion_blend={},
            micro_expressions=[],
            suppressed_emotions=[],
            attention_state=AttentionState.UNKNOWN,
            engagement_level=0.0,
            errors=[error_message],
            model_used=ExpressionModel.MEDIAPIPE,
            frame_number=request.frame_number
        )
    
    def cleanup(self) -> None:
        """Cleanup MediaPipe resources."""
        if self.face_mesh:
            self.face_mesh.close()
        if self.face_detection:
            self.face_detection.close()
        self.initialized = False


class SimplifiedEmotionClassifier:
    """Simplified emotion classifier using facial geometry analysis."""
    
    def __init__(self):
        self.emotion_rules = {}
        
    async def initialize(self) -> None:
        """Initialize the emotion classifier."""
        self.emotion_rules = self._define_emotion_rules()
    
    def _define_emotion_rules(self) -> Dict[EmotionType, Callable]:
        """Define rules for emotion classification based on facial features."""
        return {
            EmotionType.HAPPINESS: self._detect_happiness,
            EmotionType.SADNESS: self._detect_sadness,
            EmotionType.ANGER: self._detect_anger,
            EmotionType.SURPRISE: self._detect_surprise,
            EmotionType.FEAR: self._detect_fear,
            EmotionType.DISGUST: self._detect_disgust,
            EmotionType.NEUTRAL: self._detect_neutral
        }
    
    async def classify_emotions(
        self, 
        face_roi: np.ndarray, 
        face_geometry: FaceGeometry
    ) -> List[EmotionScore]:
        """Classify emotions using simplified rules."""
        emotions = []
        
        for emotion_type, detector in self.emotion_rules.items():
            confidence = detector(face_geometry)
            if confidence > 0.3:  # Threshold for inclusion
                intensity = self._determine_intensity(confidence)
                emotions.append(EmotionScore(
                    emotion=emotion_type,
                    confidence=confidence,
                    intensity=intensity
                ))
        
        return emotions
    
    def _detect_happiness(self, face_geometry: FaceGeometry) -> float:
        """Detect happiness based on facial geometry."""
        # Simplified: check for upward mouth curvature
        # In a real implementation, this would analyze mouth landmarks
        return 0.7  # Placeholder
    
    def _detect_sadness(self, face_geometry: FaceGeometry) -> float:
        """Detect sadness based on facial geometry."""
        return 0.2  # Placeholder
    
    def _detect_anger(self, face_geometry: FaceGeometry) -> float:
        """Detect anger based on facial geometry."""
        return 0.1  # Placeholder
    
    def _detect_surprise(self, face_geometry: FaceGeometry) -> float:
        """Detect surprise based on facial geometry."""
        return 0.3  # Placeholder
    
    def _detect_fear(self, face_geometry: FaceGeometry) -> float:
        """Detect fear based on facial geometry."""
        return 0.1  # Placeholder
    
    def _detect_disgust(self, face_geometry: FaceGeometry) -> float:
        """Detect disgust based on facial geometry."""
        return 0.1  # Placeholder
    
    def _detect_neutral(self, face_geometry: FaceGeometry) -> float:
        """Detect neutral expression."""
        return 0.5  # Placeholder
    
    def _determine_intensity(self, confidence: float) -> ExpressionIntensity:
        """Determine expression intensity from confidence."""
        if confidence >= 0.8:
            return ExpressionIntensity.STRONG
        elif confidence >= 0.6:
            return ExpressionIntensity.MODERATE
        elif confidence >= 0.4:
            return ExpressionIntensity.MILD
        else:
            return ExpressionIntensity.SUBTLE


class ExpressionTracker:
    """Tracks expressions across frames for temporal analysis."""
    
    def __init__(self, max_history: int = 30):
        self.max_history = max_history
        self.expression_history: deque = deque(maxlen=max_history)
        self.face_tracks: Dict[int, Dict[str, Any]] = {}
        self.next_track_id = 0
    
    def update_tracking(
        self, 
        result: ExpressionAnalysisResult
    ) -> ExpressionAnalysisResult:
        """Update tracking information and temporal analysis."""
        self.expression_history.append(result)
        
        # Perform temporal analysis
        result.emotion_stability = self._calculate_emotion_stability()
        result.transition_detected = self._detect_emotion_transitions()
        result.emotion_changes = self._analyze_emotion_changes()
        
        return result
    
    def _calculate_emotion_stability(self) -> float:
        """Calculate emotion stability over time."""
        if len(self.expression_history) < 3:
            return 1.0
        
        recent_emotions = [
            result.primary_emotions[0].emotion if result.primary_emotions else EmotionType.NEUTRAL
            for result in list(self.expression_history)[-5:]
        ]
        
        # Calculate consistency
        if len(set(recent_emotions)) == 1:
            return 1.0
        else:
            return 1.0 / len(set(recent_emotions))
    
    def _detect_emotion_transitions(self) -> bool:
        """Detect if an emotion transition occurred."""
        if len(self.expression_history) < 2:
            return False
        
        current = self.expression_history[-1]
        previous = self.expression_history[-2]
        
        if (current.primary_emotions and previous.primary_emotions and
            current.primary_emotions[0].emotion != previous.primary_emotions[0].emotion):
            return True
        
        return False
    
    def _analyze_emotion_changes(self) -> List[Dict[str, Any]]:
        """Analyze emotion changes over time."""
        changes = []
        
        if len(self.expression_history) >= 2:
            current = self.expression_history[-1]
            previous = self.expression_history[-2]
            
            if (current.primary_emotions and previous.primary_emotions and
                current.primary_emotions[0].emotion != previous.primary_emotions[0].emotion):
                
                changes.append({
                    "from_emotion": previous.primary_emotions[0].emotion.value,
                    "to_emotion": current.primary_emotions[0].emotion.value,
                    "confidence_change": (current.primary_emotions[0].confidence - 
                                        previous.primary_emotions[0].confidence),
                    "timestamp": current.timestamp
                })
        
        return changes


class EnhancedExpressionAnalyzer(ComponentInterface):
    """
    Enhanced facial expression analyzer with comprehensive capabilities.
    
    Features:
    - Multi-model expression analysis
    - Real-time emotion detection
    - Micro-expression analysis
    - Attention and engagement assessment
    - Temporal expression tracking
    - Cultural adaptation
    - Performance optimization
    - Seamless system integration
    """
    
    def __init__(self, container: Container):
        """Initialize the enhanced expression analyzer."""
        self.container = container
        self.logger = get_logger(__name__)
        self.config_loader = container.get(ConfigLoader)
        self.event_bus = container.get(EventBus)
        self.error_handler = container.get(ErrorHandler)
        self.health_check = container.get(HealthCheck)
        
        # Load configuration
        self._load_configuration()
        
        # Core components
        self._setup_core_components()
        
        # Expression analyzers
        self._setup_analyzers()
        
        # Monitoring and caching
        self._setup_monitoring()
        self._setup_caching()
        
        # Tracking and learning
        self.expression_tracker = ExpressionTracker()
        self._processing_queue = asyncio.Queue()
        self._analysis_cache: Dict[str, ExpressionAnalysisResult] = {}
        
        # Performance metrics
        self._total_analyses = 0
        self._total_processing_time = 0.0
        self._error_count = 0
        
        # Background tasks
        self._background_tasks: Set[asyncio.Task] = set()
        
        # State management
        self.initialized = False
        self._shutdown_event = asyncio.Event()
        
        self.logger.info("EnhancedExpressionAnalyzer initialized")
    
    def _load_configuration(self) -> None:
        """Load expression analysis configuration."""
        config_dict = self.config_loader.get("expression_analysis", {})
        
        # Create configuration with defaults
        self.config = ExpressionConfiguration(
            primary_model=ExpressionModel(config_dict.get("primary_model", "auto")),
            default_quality=QualityLevel(config_dict.get("default_quality", "balanced")),
            enable_micro_expressions=config_dict.get("enable_micro_expressions", False),
            enable_attention_analysis=config_dict.get("enable_attention_analysis", True),
            emotion_confidence_threshold=config_dict.get("emotion_confidence_threshold", 0.5),
            enable_gpu_acceleration=config_dict.get("enable_gpu_acceleration", True),
            max_concurrent_analyses=config_dict.get("max_concurrent_analyses", 4),
            enable_result_caching=config_dict.get("enable_result_caching", True),
            cache_ttl_seconds=config_dict.get("cache_ttl_seconds", 300)
        )
    
    def _setup_core_components(self) -> None:
        """Setup core system components."""
        try:
            self.vision_processor = self.container.get(VisionProcessor)
            self.image_analyzer = self.container.get(ImageAnalyzer)
            self.feedback_processor = self.container.get(FeedbackProcessor)
        except Exception as e:
            self.logger.warning(f"Some core components not available: {str(e)}")
    
    def _setup_analyzers(self) -> None:
        """Setup expression analyzers."""
        self.analyzers: Dict[ExpressionModel, ExpressionAnalyzer] = {}
        
        # Setup MediaPipe analyzer
        if MEDIAPIPE_AVAILABLE and self.config.primary_model in [ExpressionModel.MEDIAPIPE, ExpressionModel.AUTO]:
            self.analyzers[ExpressionModel.MEDIAPIPE] = MediaPipeExpressionAnalyzer(
                self.logger, self.config
            )
        
        # Setup fallback analyzers
        if not self.analyzers:
            self.logger.warning("No expression analyzers available")
    
    def _setup_monitoring(self) -> None:
        """Setup monitoring and metrics collection."""
        try:
            self.metrics = self.container.get(MetricsCollector)
            self.tracer = self.container.get(TraceManager)
            
            # Register metrics
            self.metrics.register_counter("expression_analyses_total")
            self.metrics.register_counter("expression_errors_total")
            self.metrics.register_histogram("expression_analysis_duration_seconds")
            self.metrics.register_histogram("expression_confidence_score")
            self.metrics.register_gauge("active_expression_analyses")
            
        except Exception as e:
            self.logger.warning(f"Monitoring setup failed: {str(e)}")
            self.metrics = None
            self.tracer = None
    
    def _setup_caching(self) -> None:
        """Setup caching for expression analysis results."""
        try:
            self.cache_manager = self.container.get(CacheManager)
            self.cache_strategy = CacheStrategy(
                ttl=self.config.cache_ttl_seconds,
                max_size=self.config.cache_size_limit
            )
        except Exception as e:
            self.logger.warning(f"Caching setup failed: {str(e)}")
            self.cache_manager = None
            self.cache_strategy = None
    
    async def initialize(self) -> None:
        """Initialize the expression analyzer."""
        if self.initialized:
            return
        
        self.logger.info("Initializing enhanced expression analyzer...")
        
        try:
            # Initialize analyzers
            for model, analyzer in self.analyzers.items():
                await analyzer.initialize()
                self.logger.info(f"Initialized {model.value} analyzer")
            
            # Register health check
            self.health_check.register_component(
                "expression_analyzer", 
                self._health_check_callback
            )
            
            # Start background tasks
            await self._start_background_tasks()
            
            # Register event handlers
            await self._register_event_handlers()
            
            self.initialized = True
            
            # Emit initialization event
            await self.event_bus.emit(ComponentInitialized(
                component_id="expression_analyzer",
                initialization_time=time.time()
            ))
            
            self.logger.info("Enhanced expression analyzer initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize expression analyzer: {str(e)}")
            raise ExpressionAnalysisError(f"Initialization failed: {str(e)}")
    
    async def _register_event_handlers(self) -> None:
        """Register event handlers."""
        try:
            # Register for vision processing events
            await self.event_bus.subscribe("vision_processing_completed", self._handle_vision_completed)
            await self.event_bus.subscribe("feedback_received", self._handle_feedback)
            await self.event_bus.subscribe("system_shutdown", self._handle_shutdown)
            
        except Exception as e:
            self.logger.warning(f"Event handler registration failed: {str(e)}")
    
    async def _start_background_tasks(self) -> None:
        """Start background monitoring and processing tasks."""
        tasks = [
            self._processing_queue_handler(),
            self._cache_cleanup_loop(),
            self._performance_monitoring_loop()
        ]
        
        for task_coro in tasks:
            task = asyncio.create_task(task_coro)
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)
    
    @handle_exceptions
    async def analyze_expression(
        self, 
        request: ExpressionAnalysisRequest
    ) -> ExpressionAnalysisResult:
        """
        Analyze facial expressions in an image.
        
        Args:
            request: Expression analysis request
            
        Returns:
            Expression analysis result
        """
        if not self.initialized:
            await self.initialize()
        
        # Check cache first
        if self.config.enable_result_caching:
            cached_result = await self._get_cached_result(request)
            if cached_result:
                return cached_result
        
        # Start tracing
        with self.tracer.trace("expression_analysis") if self.tracer else nullcontext():
            start_time = time.time()
            
            try:
                # Emit processing started event
                await self.event_bus.emit(VisionProcessingStarted(
                    processing_id=request.request_id,
                    processing_type="expression_analysis",
                    session_id=request.session_id
                ))
                
                # Select analyzer
                analyzer = self._select_analyzer(request.model)
                if not analyzer:
                    raise ExpressionAnalysisError("No suitable analyzer available")
                
                # Validate request
                self._validate_request(request)
                
                # Perform analysis
                result = await analyzer.analyze_expression(request)
                
                # Post-process result
                result = await self._post_process_result(result, request)
                
                # Update tracking
                result = self.expression_tracker.update_tracking(result)
                
                # Cache result
                if self.config.enable_result_caching and result.success:
                    await self._cache_result(request, result)
                
                # Update metrics
                self._update_metrics(result, time.time() - start_time)
                
                # Store for learning
                await self._store_analysis_for_learning(request, result)
                
                # Emit completion event
                await self.event_bus.emit(VisionProcessingCompleted(
                    processing_id=request.request_id,
                    processing_type="expression_analysis",
                    session_id=request.session_id,
                    success=result.success,
                    processing_time=result.processing_time
                ))
                
                # Emit specific expression events
                if result.success and result.primary_emotions:
                    await self.event_bus.emit(EmotionDetected(
                        emotion=result.primary_emotions[0].emotion.value,
                        confidence=result.primary_emotions[0].confidence,
                        session_id=request.session_id,
                        user_id=request.user_id
                    ))
                
                return result
                
            except Exception as e:
                processing_time = time.time() - start_time
                self.logger.error(f"Expression analysis failed: {str(e)}")
                
                # Create error result
                error_result = self._create_error_result(request, str(e), processing_time)
                
                # Update error metrics
                self._error_count += 1
                if self.metrics:
                    self.metrics.increment("expression_errors_total")
                
                # Emit error event
                await self.event_bus.emit(ProcessingError(
                    processing_id=request.request_id,
                    error_message=str(e),
                    component="expression_analyzer"
                ))
                
                return error_result
    
    def _select_analyzer(self, requested_model: ExpressionModel) -> Optional[ExpressionAnalyzer]:
        """Select appropriate analyzer based on request."""
        if requested_model == ExpressionModel.AUTO:
            # Select best available analyzer
            if ExpressionModel.MEDIAPIPE in self.analyzers:
                return self.analyzers[ExpressionModel.MEDIAPIPE]
        
        return self.analyzers.get(requested_model)
    
    def _validate_request(self, request: ExpressionAnalysisRequest) -> None:
        """Validate expression analysis request."""
        if request.image is None or request.image.size == 0:
            raise ExpressionAnalysisError("Invalid image data")
        
        if len(request.image.shape) != 3:
            raise ExpressionAnalysisError("Image must be 3-channel")
        
        if request.image.shape[0] < self.config.min_face_resolution or request.image.shape[1] < self.config.min_face_resolution:
            raise ExpressionAnalysisError("Image resolution too low")
    
    async def _post_process_result(
        self, 
        result: ExpressionAnalysisResult, 
        request: ExpressionAnalysisRequest
    ) -> ExpressionAnalysisResult:
        """Post-process analysis result."""
        # Apply cultural adaptations
        if self.config.enable_cultural_adaptation and request.cultural_context:
            result = await self._apply_cultural_adaptation(result, request.cultural_context)
        
        # Filter low-confidence results
        result.primary_emotions = [
            emotion for emotion in result.primary_emotions
            if emotion.confidence >= self.config.emotion_confidence_threshold
        ]
        
        # Calculate additional metrics
        result.analysis_confidence = self._calculate_analysis_confidence(result)
        
        return result
    
    async def _apply_cultural_adaptation(
        self, 
        result: ExpressionAnalysisResult, 
        cultural_context: str
    ) -> ExpressionAnalysisResult:
        """Apply cultural adaptations to emotion interpretation."""
        # This would implement cultural emotion mapping
        # For now, it's a placeholder
        cultural_adjustments = self.config.cultural_emotion_mappings.get(cultural_context, {})
        
        for emotion in result.primary_emotions:
            adjustment = cultural_adjustments.get(emotion.emotion.value, 1.0)
            emotion.confidence *= adjustment
        
        result.cultural_adjustments = cultural_adjustments
        return result
    
    def _calculate_analysis_confidence(self, result: ExpressionAnalysisResult) -> float:
        """Calculate overall analysis confidence."""
        if not result.primary_emotions:
            return 0.0
        
        # Weight by emotion confidence and face quality
        emotion_confidence = sum(e.confidence for e in result.primary_emotions) / len(result.primary_emotions)
        
        quality_factors = [
            result.landmark_quality,
            result.lighting_quality,
            result.pose_quality
        ]
        
        quality_confidence = sum(q for q in quality_factors if q > 0) / max(1, sum(1 for q in quality_factors if q > 0))
        
        return (emotion_confidence + quality_confidence) / 2.0
    
    async def _get_cached_result(self, request: ExpressionAnalysisRequest) -> Optional[ExpressionAnalysisResult]:
        """Get cached analysis result."""
        if not self.cache_manager:
            return None
        
        cache_key = self._generate_cache_key(request)
        try:
            cached_data = await self.cache_manager.get(cache_key)
            if cached_data:
                return ExpressionAnalysisResult(**cached_data)
        except Exception as e:
            self.logger.warning(f"Cache retrieval failed: {str(e)}")
        
        return None
    
    async def _cache_result(self, request: ExpressionAnalysisRequest, result: ExpressionAnalysisResult) -> None:
        """Cache analysis result."""
        if not self.cache_manager:
            return
        
        cache_key = self._generate_cache_key(request)
        try:
            # Convert result to dictionary for caching
            result_dict = {
                "success": result.success,
                "request_id": result.request_id,
                "processing_time": result.processing_time,
                "faces_detected": result.faces_detected,
                "primary_emotions": [
                    {
                        "emotion": emotion.emotion.value,
                        "confidence": emotion.confidence,
                        "intensity": emotion.intensity.value
                    }
                    for emotion in result.primary_emotions
                ],
                "attention_state": result.attention_state.value,
                "engagement_level": result.engagement_level,
                "timestamp": result.timestamp.isoformat()
            }
            
            await self.cache_manager.set(cache_key, result_dict, ttl=self.config.cache_ttl_seconds)
            
        except Exception as e:
            self.logger.warning(f"Cache storage failed: {str(e)}")
    
    def _generate_cache_key(self, request: ExpressionAnalysisRequest) -> str:
        """Generate cache key for request."""
        # Create hash from image and key parameters
        image_hash = hashlib.md5(request.image.tobytes()).hexdigest()
        
        key_components = [
            image_hash,
            request.model.value,
            request.mode.value,
            request.quality.value,
            str(request.detect_micro_expressions),
            str(request.analyze_attention),
            str(request.confidence_threshold)
        ]
        
        return f"expression_analysis:{':'.join(key_components)}"
    
    def _update_metrics(self, result: ExpressionAnalysisResult, processing_time: float) -> None:
        """Update performance metrics."""
        self._total_analyses += 1
        self._total_processing_time += processing_time
        
        if self.metrics:
            self.metrics.increment("expression_analyses_total")
            self.metrics.record("expression_analysis_duration_seconds", processing_time)
            
            if result.success and result.primary_emotions:
                self.metrics.record("expression_confidence_score", result.primary_emotions[0].confidence)
    
    async def _store_analysis_for_learning(
        self, 
        request: ExpressionAnalysisRequest, 
        result: ExpressionAnalysisResult
    ) -> None:
        """Store analysis data for learning and improvement."""
        if not self.config.enable_online_learning:
            return
        
        try:
            learning_data = {
                "timestamp": datetime.now(timezone.utc),
                "session_id": request.session_id,
                "user_id": request.user_id,
                "model_used": result.model_used.value,
                "processing_time": result.processing_time,
                "emotions_detected": len(result.primary_emotions),
                "analysis_confidence": result.analysis_confidence,
                "success": result.success
            }
            
            if self.feedback_processor:
                await self.feedback_processor.store_interaction_data(
                    "expression_analysis", learning_data
                )
                
        except Exception as e:
            self.logger.warning(f"Learning data storage failed: {str(e)}")
    
    def _create_error_result(
        self, 
        request: ExpressionAnalysisRequest, 
        error_message: str, 
        processing_time: float
    ) -> ExpressionAnalysisResult:
        """Create error result."""
        return ExpressionAnalysisResult(
            success=False,
            request_id=request.request_id,
            processing_time=processing_time,
            faces_detected=0,
            face_geometries=[],
            primary_emotions=[],
            secondary_emotions=[],
            emotion_blend={},
            micro_expressions=[],
            suppressed_emotions=[],
            attention_state=AttentionState.UNKNOWN,
            engagement_level=0.0,
            errors=[error_message],
            frame_number=request.frame_number
        )
    
    async def get_analysis_statistics(self) -> Dict[str, Any]:
        """Get analysis statistics and performance metrics."""
        avg_processing_time = (
            self._total_processing_time / max(self._total_analyses, 1)
        )
        
        return {
            "total_analyses": self._total_analyses,
            "total_errors": self._error_count,
            "average_processing_time": avg_processing_time,
            "success_rate": (self._total_analyses - self._error_count) / max(self._total_analyses, 1),
            "available_models": [model.value for model in self.analyzers.keys()],
            "config": {
                "primary_model": self.config.primary_model.value,
                "quality_level": self.config.default_quality.value,
                "micro_expressions_enabled": self.config.enable_micro_expressions,
                "attention_analysis_enabled": self.config.enable_attention_analysis
            }
        }
    
    async def switch_model(self, model: ExpressionModel) -> bool:
        """Switch to a different expression analysis model."""
        if model not in self.analyzers:
            self.logger.warning(f"Model {model.value} not available")
            return False
        
        self.config.primary_model = model
        self.logger.info(f"Switched to expression model: {model.value}")
        return True
    
    async def _processing_queue_handler(self) -> None:
        """Handle background processing queue."""
        while not self._shutdown_event.is_set():
            try:
                # Process queued requests
                await asyncio.sleep(1.0)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Processing queue error: {str(e)}")
    
    async def _cache_cleanup_loop(self) -> None:
        """Periodic cache cleanup."""
        while not self._shutdown_event.is_set():
            try:
                if self.cache_manager:
                    await self.cache_manager.cleanup_expired()
                
                await asyncio.sleep(300)  # Every 5 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Cache cleanup error: {str(e)}")
    
    async def _performance_monitoring_loop(self) -> None:
        """Monitor performance and optimize."""
        while not self._shutdown_event.is_set():
            try:
                # Monitor performance metrics
                if self._total_analyses > 0:
                    avg_time = self._total_processing_time / self._total_analyses
                    error_rate = self._error_count / self._total_analyses
                    
                    if avg_time > 5.0:  # Too slow
                        self.logger.warning(f"Expression analysis is slow: {avg_time:.2f}s average")
                    
                    if error_rate > 0.1:  # Too many errors
                        self.logger.warning(f"High error rate in expression analysis: {error_rate:.2%}")
                
                await asyncio.sleep(60)  # Every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Performance monitoring error: {str(e)}")
    
    async def _handle_vision_completed(self, event) -> None:
        """Handle vision processing completion events."""
        try:
            # Process vision results if relevant to expression analysis
            pass
        except Exception as e:
            self.logger.error(f"Vision event handling error: {str(e)}")
    
    async def _handle_feedback(self, event) -> None:
        """Handle user feedback events."""
        try:
            # Use feedback to improve expression analysis
            if hasattr(event, 'component') and event.component == 'expression_analyzer':
                await self._process_feedback(event)
        except Exception as e:
            self.logger.error(f"Feedback handling error: {str(e)}")
    
    async def _process_feedback(self, event) -> None:
        """Process feedback for model improvement."""
        # This would implement feedback processing for model improvement
        pass
    
    async def _handle_shutdown(self, event) -> None:
        """Handle system shutdown."""
        self.logger.info("Expression analyzer shutting down...")
        self._shutdown_event.set()
    
    async def _health_check_callback(self) -> Dict[str, Any]:
        """Health check callback."""
        try:
            health_status = "healthy"
            
            # Check analyzer availability
            available_analyzers = len(self.analyzers)
            if available_analyzers == 0:
                health_status = "unhealthy"
            
            # Check error rate
            if self._total_analyses > 10:
                error_rate = self._error_count / self._total_analyses
                if error_rate > 0.2:
                    health_status = "degraded"
            
            # Check performance
            if self._total_analyses > 0:
                avg_time = self._total_processing_time / self._total_analyses
                if avg_time > 10.0:
                    health_status = "degraded"
            
            return {
                "status": health_status,
                "available_analyzers": available_analyzers,
                "total_analyses": self._total_analyses,
                "error_count": self._error_count,
                "average_processing_time": (
                    self._total_processing_time / max(self._total_analyses, 1)
                ),
                "initialized": self.initialized,
                "config_loaded": bool(self.config)
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    async def cleanup(self) -> None:
        """Cleanup resources and shutdown."""
        self.logger.info("Cleaning up expression analyzer...")
        
        try:
            # Set shutdown flag
            self._shutdown_event.set()
            
            # Cancel background tasks
            for task in self._background_tasks:
                task.cancel()
            
            if self._background_tasks:
                await asyncio.gather(*self._background_tasks, return_exceptions=True)
            
            # Cleanup analyzers
            for analyzer in self.analyzers.values():
                analyzer.cleanup()
            
            # Clear caches
            self._analysis_cache.clear()
            
            self.initialized = False
            self.logger.info("Expression analyzer cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")
    
    def __del__(self):
        """Destructor."""
        try:
            if hasattr(self, 'initialized') and self.initialized:
                self.logger.warning("ExpressionAnalyzer was not properly cleaned up")
        except Exception:
            pass  # Ignore cleanup errors in destructor


# Null context manager for when tracer is not available
class nullcontext:
    """Null context manager for when tracer is not available."""
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
