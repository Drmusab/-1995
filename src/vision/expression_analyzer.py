"""
Advanced Facial Expression Analysis System
Author: Drmusab
Last Modified: 2025-05-28 17:35:00 UTC

This module provides comprehensive facial expression recognition, emotion detection,
and micro-expression analysis for the AI assistant system.
"""

from pathlib import Path
from typing import Optional, Dict, Any, List, Set, Tuple, Union, AsyncGenerator
import asyncio
import time
from datetime import datetime, timezone
from dataclasses import dataclass, field
from enum import Enum
import logging
import numpy as np
import cv2
from abc import ABC, abstractmethod
from collections import deque, defaultdict
import json
import threading
from concurrent.futures import ThreadPoolExecutor

# Core imports
from src.core.config.loader import ConfigLoader
from src.core.events.event_bus import EventBus
from src.core.events.event_types import (
    ComponentInitialized,
    ComponentFailed,
    ProcessingCompleted,
    ProcessingFailed
)
from src.core.error_handling import ErrorHandler, handle_exceptions
from src.core.dependency_injection import Container
from src.core.health_check import HealthCheck

# Observability
from src.observability.monitoring.metrics import MetricsCollector
from src.observability.monitoring.tracing import TraceManager
from src.observability.logging.config import get_logger

# Memory and caching
from src.memory.cache_manager import CacheManager
from src.integrations.cache.redis_cache import RedisCache

# Learning and feedback
from src.learning.feedback_processor import FeedbackProcessor

# Face processing components
from src.vision.face_recognition import FaceDetector, FaceDetectionMethod

try:
    import tensorflow as tf
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False

try:
    import torch
    import torch.nn as nn
    import torchvision.transforms as transforms
    HAS_PYTORCH = True
except ImportError:
    HAS_PYTORCH = False

try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False

try:
    import dlib
    HAS_DLIB = True
except ImportError:
    HAS_DLIB = False


class ExpressionType(Enum):
    """Types of facial expressions."""
    NEUTRAL = "neutral"
    HAPPINESS = "happiness"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    CONTEMPT = "contempt"
    
    # Complex expressions
    CONFUSION = "confusion"
    EXCITEMENT = "excitement"
    EMBARRASSMENT = "embarrassment"
    RELIEF = "relief"
    SKEPTICISM = "skepticism"
    CONCENTRATION = "concentration"
    BOREDOM = "boredom"
    FRUSTRATION = "frustration"


class EmotionCategory(Enum):
    """Emotion categories based on psychological models."""
    BASIC = "basic"           # Ekman's basic emotions
    COMPLEX = "complex"       # Complex emotional states
    SOCIAL = "social"         # Social emotions
    COGNITIVE = "cognitive"   # Cognitive emotions
    PHYSIOLOGICAL = "physiological"  # Stress, fatigue, etc.


class ExpressionDetectionMethod(Enum):
    """Methods for expression detection."""
    LANDMARKS = "landmarks"       # Facial landmark analysis
    CNN = "cnn"                  # Convolutional neural networks
    TRANSFORMER = "transformer"  # Vision transformer models
    ENSEMBLE = "ensemble"        # Multiple methods combined
    MEDIAPIPE = "mediapipe"      # MediaPipe face mesh
    OPENFACE = "openface"        # OpenFace toolkit
    AFFECTNET = "affectnet"      # AffectNet models


class MicroExpressionType(Enum):
    """Types of micro-expressions."""
    GENUINE_SMILE = "genuine_smile"
    FAKE_SMILE = "fake_smile"
    SUPPRESSED_ANGER = "suppressed_anger"
    CONCEALED_FEAR = "concealed_fear"
    HIDDEN_DISGUST = "hidden_disgust"
    LEAKED_CONTEMPT = "leaked_contempt"
    DECEPTION_MARKERS = "deception_markers"


class AnalysisQuality(Enum):
    """Quality levels for expression analysis."""
    FAST = "fast"           # Speed optimized
    BALANCED = "balanced"   # Speed/accuracy balance
    ACCURATE = "accurate"   # Accuracy optimized
    RESEARCH = "research"   # Maximum accuracy for research


@dataclass
class FacialLandmarks:
    """Facial landmarks data structure."""
    landmarks: np.ndarray  # Shape: (N, 2) or (N, 3)
    confidence: float
    landmark_type: str  # '68_point', '478_point', etc.
    visibility: Optional[np.ndarray] = None  # Visibility scores
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExpressionVector:
    """Expression analysis result vector."""
    expression_scores: Dict[str, float]
    primary_expression: str
    secondary_expression: Optional[str] = None
    confidence: float = 0.0
    intensity: float = 0.0
    authenticity: float = 1.0  # How genuine the expression appears
    arousal: float = 0.0      # Emotional arousal level
    valence: float = 0.0      # Emotional valence (positive/negative)
    dominance: float = 0.0    # Emotional dominance
    
    # Temporal information
    duration: float = 0.0
    onset_time: float = 0.0
    peak_time: float = 0.0
    offset_time: float = 0.0
    
    # Spatial information
    facial_regions: Dict[str, float] = field(default_factory=dict)
    asymmetry_score: float = 0.0
    
    # Metadata
    analysis_method: str = "unknown"
    quality_score: float = 0.0


@dataclass
class MicroExpression:
    """Micro-expression detection result."""
    micro_expression_type: MicroExpressionType
    confidence: float
    start_frame: int
    end_frame: int
    peak_frame: int
    duration_ms: float
    intensity: float
    facial_regions: List[str]
    landmarks_evidence: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EmotionalState:
    """Complete emotional state analysis."""
    primary_emotion: str
    secondary_emotions: List[str] = field(default_factory=list)
    emotion_category: EmotionCategory = EmotionCategory.BASIC
    confidence: float = 0.0
    intensity: float = 0.0
    
    # Dimensional emotion model (Russell's circumplex)
    arousal: float = 0.0      # High arousal (excited) vs Low arousal (calm)
    valence: float = 0.0      # Positive (pleasant) vs Negative (unpleasant)
    dominance: float = 0.0    # High control vs Low control
    
    # Temporal dynamics
    stability: float = 0.0    # How stable the emotion is
    change_rate: float = 0.0  # Rate of emotional change
    
    # Contextual factors
    social_context: Optional[str] = None
    cultural_context: Optional[str] = None
    
    # Physiological indicators
    stress_level: float = 0.0
    fatigue_level: float = 0.0
    engagement_level: float = 0.0


@dataclass
class ExpressionAnalysisResult:
    """Complete expression analysis result."""
    success: bool
    request_id: str
    processing_time: float
    
    # Core results
    detected_faces: List[Dict[str, Any]]
    expression_vectors: List[ExpressionVector]
    emotional_states: List[EmotionalState]
    micro_expressions: List[MicroExpression]
    
    # Analysis metadata
    analysis_method: ExpressionDetectionMethod
    quality_level: AnalysisQuality
    frame_count: int = 1
    fps: Optional[float] = None
    
    # Temporal analysis (for video)
    temporal_expressions: Dict[str, List[float]] = field(default_factory=dict)
    expression_transitions: List[Dict[str, Any]] = field(default_factory=list)
    emotional_trajectory: List[Dict[str, Any]] = field(default_factory=list)
    
    # Aggregate metrics
    overall_mood: str = "neutral"
    mood_stability: float = 0.0
    dominant_expressions: List[str] = field(default_factory=list)
    expression_diversity: float = 0.0
    
    # Quality metrics
    detection_quality: float = 0.0
    analysis_confidence: float = 0.0
    temporal_consistency: float = 0.0
    
    # Error information
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Metadata
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExpressionConfiguration:
    """Configuration for expression analysis."""
    # Analysis settings
    detection_method: ExpressionDetectionMethod = ExpressionDetectionMethod.CNN
    quality_level: AnalysisQuality = AnalysisQuality.BALANCED
    enable_micro_expressions: bool = True
    enable_temporal_analysis: bool = True
    enable_emotion_classification: bool = True
    
    # Model settings
    model_confidence_threshold: float = 0.5
    expression_confidence_threshold: float = 0.3
    micro_expression_threshold: float = 0.7
    temporal_smoothing: bool = True
    
    # Face detection settings
    min_face_size: int = 64
    max_faces: int = 10
    face_detection_confidence: float = 0.8
    
    # Temporal analysis settings
    temporal_window_size: int = 30  # frames
    expression_stability_threshold: float = 0.8
    transition_detection_threshold: float = 0.3
    
    # Performance settings
    enable_gpu: bool = True
    batch_size: int = 1
    num_threads: int = 4
    max_memory_mb: int = 1024
    
    # Output settings
    include_landmarks: bool = True
    include_heatmaps: bool = False
    include_feature_vectors: bool = False
    normalize_scores: bool = True
    
    # Cultural adaptation
    cultural_model: Optional[str] = None
    demographic_adaptation: bool = False
    
    # Caching
    enable_caching: bool = True
    cache_ttl: int = 300
    
    # Privacy
    anonymize_results: bool = False
    store_facial_data: bool = False


class ExpressionAnalysisError(Exception):
    """Custom exception for expression analysis operations."""
    
    def __init__(self, message: str, error_code: Optional[str] = None,
                 analysis_stage: Optional[str] = None):
        super().__init__(message)
        self.error_code = error_code
        self.analysis_stage = analysis_stage
        self.timestamp = datetime.now(timezone.utc)


class ExpressionDetector(ABC):
    """Base class for expression detection methods."""
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the detector."""
        pass
    
    @abstractmethod
    async def detect_expressions(
        self,
        image: np.ndarray,
        face_locations: List[Tuple[int, int, int, int]],
        config: ExpressionConfiguration
    ) -> List[ExpressionVector]:
        """Detect expressions in faces."""
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Cleanup resources."""
        pass


class CNNExpressionDetector(ExpressionDetector):
    """CNN-based expression detector."""
    
    def __init__(self, logger, model_path: Optional[str] = None):
        self.logger = logger
        self.model_path = model_path
        self.model = None
        self.preprocessing = None
        self.expression_classes = [
            "neutral", "happiness", "sadness", "anger", 
            "fear", "surprise", "disgust"
        ]
        
    async def initialize(self) -> None:
        """Initialize the CNN model."""
        try:
            if HAS_TENSORFLOW:
                await self._initialize_tensorflow_model()
            elif HAS_PYTORCH:
                await self._initialize_pytorch_model()
            else:
                raise ExpressionAnalysisError(
                    "No deep learning framework available",
                    error_code="NO_FRAMEWORK"
                )
                
            self.logger.info("CNN expression detector initialized")
            
        except Exception as e:
            raise ExpressionAnalysisError(
                f"Failed to initialize CNN detector: {str(e)}",
                error_code="INIT_FAILED"
            )
    
    async def _initialize_tensorflow_model(self) -> None:
        """Initialize TensorFlow model."""
        if self.model_path and Path(self.model_path).exists():
            self.model = tf.keras.models.load_model(self.model_path)
        else:
            # Create a simple CNN model
            self.model = self._create_default_model()
            
        self.preprocessing = tf.keras.utils.normalize
    
    async def _initialize_pytorch_model(self) -> None:
        """Initialize PyTorch model."""
        if self.model_path and Path(self.model_path).exists():
            self.model = torch.load(self.model_path)
        else:
            self.model = self._create_pytorch_model()
            
        self.preprocessing = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def _create_default_model(self):
        """Create a default CNN model."""
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(224, 224, 3)),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(len(self.expression_classes), activation='softmax')
        ])
        return model
    
    def _create_pytorch_model(self):
        """Create a default PyTorch model."""
        class ExpressionCNN(nn.Module):
            def __init__(self, num_classes):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, 64, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(64, 128, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(128, 256, 3, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1, 1))
                )
                self.classifier = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(256, 512),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(512, num_classes)
                )
            
            def forward(self, x):
                x = self.features(x)
                x = self.classifier(x)
                return x
        
        return ExpressionCNN(len(self.expression_classes))
    
    async def detect_expressions(
        self,
        image: np.ndarray,
        face_locations: List[Tuple[int, int, int, int]],
        config: ExpressionConfiguration
    ) -> List[ExpressionVector]:
        """Detect expressions using CNN."""
        try:
            expression_vectors = []
            
            for face_location in face_locations:
                top, right, bottom, left = face_location
                face_image = image[top:bottom, left:right]
                
                # Preprocess face image
                processed_face = self._preprocess_face(face_image)
                
                # Get predictions
                if HAS_TENSORFLOW and hasattr(self.model, 'predict'):
                    predictions = self.model.predict(processed_face[np.newaxis, ...])[0]
                elif HAS_PYTORCH:
                    with torch.no_grad():
                        predictions = torch.softmax(
                            self.model(processed_face.unsqueeze(0)), dim=1
                        )[0].numpy()
                else:
                    continue
                
                # Create expression vector
                expression_scores = {
                    expr: float(score) for expr, score in 
                    zip(self.expression_classes, predictions)
                }
                
                primary_expression = max(expression_scores, key=expression_scores.get)
                confidence = expression_scores[primary_expression]
                
                # Calculate additional metrics
                arousal, valence = self._calculate_arousal_valence(expression_scores)
                
                expression_vector = ExpressionVector(
                    expression_scores=expression_scores,
                    primary_expression=primary_expression,
                    confidence=confidence,
                    arousal=arousal,
                    valence=valence,
                    analysis_method="cnn",
                    quality_score=confidence
                )
                
                expression_vectors.append(expression_vector)
            
            return expression_vectors
            
        except Exception as e:
            raise ExpressionAnalysisError(
                f"CNN expression detection failed: {str(e)}",
                error_code="DETECTION_FAILED",
                analysis_stage="cnn_detection"
            )
    
    def _preprocess_face(self, face_image: np.ndarray) -> np.ndarray:
        """Preprocess face image for model input."""
        # Resize to model input size
        face_resized = cv2.resize(face_image, (224, 224))
        
        if HAS_TENSORFLOW:
            # Normalize for TensorFlow
            face_normalized = face_resized.astype(np.float32) / 255.0
            return face_normalized
        elif HAS_PYTORCH:
            # Apply PyTorch preprocessing
            return self.preprocessing(face_resized)
        
        return face_resized
    
    def _calculate_arousal_valence(self, expression_scores: Dict[str, float]) -> Tuple[float, float]:
        """Calculate arousal and valence from expression scores."""
        # Simplified arousal/valence mapping
        arousal_map = {
            "neutral": 0.0, "happiness": 0.7, "sadness": -0.3,
            "anger": 0.8, "fear": 0.9, "surprise": 0.8, "disgust": 0.4
        }
        
        valence_map = {
            "neutral": 0.0, "happiness": 0.9, "sadness": -0.8,
            "anger": -0.6, "fear": -0.7, "surprise": 0.2, "disgust": -0.8
        }
        
        arousal = sum(expression_scores[expr] * arousal_map.get(expr, 0.0) 
                     for expr in expression_scores)
        valence = sum(expression_scores[expr] * valence_map.get(expr, 0.0) 
                     for expr in expression_scores)
        
        return arousal, valence
    
    def cleanup(self) -> None:
        """Cleanup model resources."""
        self.model = None
        self.preprocessing = None


class MediaPipeExpressionDetector(ExpressionDetector):
    """MediaPipe-based expression detector using face landmarks."""
    
    def __init__(self, logger):
        self.logger = logger
        self.face_mesh = None
        self.landmark_indices = self._get_expression_landmarks()
        
    async def initialize(self) -> None:
        """Initialize MediaPipe components."""
        try:
            if not HAS_MEDIAPIPE:
                raise ExpressionAnalysisError(
                    "MediaPipe not available",
                    error_code="MEDIAPIPE_NOT_AVAILABLE"
                )
            
            mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=10,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            
            self.logger.info("MediaPipe expression detector initialized")
            
        except Exception as e:
            raise ExpressionAnalysisError(
                f"Failed to initialize MediaPipe detector: {str(e)}",
                error_code="INIT_FAILED"
            )
    
    def _get_expression_landmarks(self) -> Dict[str, List[int]]:
        """Get landmark indices for different facial regions."""
        return {
            'left_eye': [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
            'right_eye': [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],
            'eyebrows': [70, 63, 105, 66, 107, 55, 65, 52, 53, 46, 276, 283, 282, 295, 285, 336, 296, 334],
            'nose': [1, 2, 5, 4, 6, 168, 8, 9, 10, 151, 195, 197, 196, 3, 51, 48, 115, 131, 134, 102, 49, 220, 305, 281, 360, 279],
            'mouth': [61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318, 13, 82, 81, 80, 78, 95, 88, 178, 87, 14, 317, 402, 318, 324],
            'lips': [0, 11, 12, 13, 14, 15, 16, 17, 18, 200, 199, 175, 0],
            'chin': [172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323]
        }
    
    async def detect_expressions(
        self,
        image: np.ndarray,
        face_locations: List[Tuple[int, int, int, int]],
        config: ExpressionConfiguration
    ) -> List[ExpressionVector]:
        """Detect expressions using facial landmarks."""
        try:
            expression_vectors = []
            
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_image)
            
            if results.multi_face_landmarks:
                for landmarks in results.multi_face_landmarks:
                    # Convert landmarks to numpy array
                    landmark_points = np.array([
                        [lm.x * image.shape[1], lm.y * image.shape[0]] 
                        for lm in landmarks.landmark
                    ])
                    
                    # Analyze expression based on landmarks
                    expression_vector = self._analyze_landmarks(landmark_points)
                    expression_vectors.append(expression_vector)
            
            return expression_vectors
            
        except Exception as e:
            raise ExpressionAnalysisError(
                f"MediaPipe expression detection failed: {str(e)}",
                error_code="DETECTION_FAILED",
                analysis_stage="mediapipe_detection"
            )
    
    def _analyze_landmarks(self, landmarks: np.ndarray) -> ExpressionVector:
        """Analyze expression from facial landmarks."""
        expression_scores = {}
        
        # Eye analysis
        eye_openness = self._calculate_eye_openness(landmarks)
        
        # Mouth analysis
        mouth_curvature = self._calculate_mouth_curvature(landmarks)
        mouth_openness = self._calculate_mouth_openness(landmarks)
        
        # Eyebrow analysis
        eyebrow_height = self._calculate_eyebrow_height(landmarks)
        
        # Classify expressions based on features
        expression_scores['happiness'] = max(0, mouth_curvature * 0.8 + (1 - eye_openness) * 0.2)
        expression_scores['sadness'] = max(0, -mouth_curvature * 0.7 - eyebrow_height * 0.3)
        expression_scores['surprise'] = max(0, eye_openness * 0.6 + mouth_openness * 0.4)
        expression_scores['anger'] = max(0, -eyebrow_height * 0.8 + (1 - eye_openness) * 0.2)
        expression_scores['fear'] = max(0, eye_openness * 0.5 + eyebrow_height * 0.5)
        expression_scores['disgust'] = max(0, -mouth_curvature * 0.5 + mouth_openness * 0.3)
        expression_scores['neutral'] = max(0, 1 - sum(expression_scores.values()))
        
        # Normalize scores
        total_score = sum(expression_scores.values())
        if total_score > 0:
            expression_scores = {k: v / total_score for k, v in expression_scores.items()}
        
        primary_expression = max(expression_scores, key=expression_scores.get)
        confidence = expression_scores[primary_expression]
        
        return ExpressionVector(
            expression_scores=expression_scores,
            primary_expression=primary_expression,
            confidence=confidence,
            analysis_method="mediapipe_landmarks",
            quality_score=confidence
        )
    
    def _calculate_eye_openness(self, landmarks: np.ndarray) -> float:
        """Calculate average eye openness."""
        left_eye_points = landmarks[self.landmark_indices['left_eye']]
        right_eye_points = landmarks[self.landmark_indices['right_eye']]
        
        # Calculate eye aspect ratio
        def eye_aspect_ratio(eye_points):
            # Vertical distances
            A = np.linalg.norm(eye_points[1] - eye_points[5])
            B = np.linalg.norm(eye_points[2] - eye_points[4])
            # Horizontal distance
            C = np.linalg.norm(eye_points[0] - eye_points[3])
            return (A + B) / (2.0 * C)
        
        left_ear = eye_aspect_ratio(left_eye_points[:6])
        right_ear = eye_aspect_ratio(right_eye_points[:6])
        
        return (left_ear + right_ear) / 2.0
    
    def _calculate_mouth_curvature(self, landmarks: np.ndarray) -> float:
        """Calculate mouth curvature (smile/frown)."""
        mouth_points = landmarks[self.landmark_indices['mouth']]
        
        # Get mouth corners and center
        left_corner = mouth_points[0]
        right_corner = mouth_points[6]
        mouth_center = mouth_points[3]
        
        # Calculate vertical positions relative to center
        corner_height = (left_corner[1] + right_corner[1]) / 2
        center_height = mouth_center[1]
        
        # Positive for smile, negative for frown
        return (center_height - corner_height) / 100.0
    
    def _calculate_mouth_openness(self, landmarks: np.ndarray) -> float:
        """Calculate mouth openness."""
        mouth_points = landmarks[self.landmark_indices['mouth']]
        
        # Vertical distance between top and bottom of mouth
        top_lip = mouth_points[2]
        bottom_lip = mouth_points[4]
        
        openness = np.linalg.norm(top_lip - bottom_lip)
        return openness / 50.0  # Normalize
    
    def _calculate_eyebrow_height(self, landmarks: np.ndarray) -> float:
        """Calculate eyebrow height relative to eyes."""
        eyebrow_points = landmarks[self.landmark_indices['eyebrows']]
        eye_points = np.concatenate([
            landmarks[self.landmark_indices['left_eye']],
            landmarks[self.landmark_indices['right_eye']]
        ])
        
        avg_eyebrow_y = np.mean(eyebrow_points[:, 1])
        avg_eye_y = np.mean(eye_points[:, 1])
        
        return (avg_eye_y - avg_eyebrow_y) / 100.0
    
    def cleanup(self) -> None:
        """Cleanup MediaPipe resources."""
        if self.face_mesh:
            self.face_mesh.close()


class MicroExpressionDetector:
    """Detector for micro-expressions in video sequences."""
    
    def __init__(self, logger):
        self.logger = logger
        self.frame_buffer = deque(maxlen=30)  # Buffer for temporal analysis
        self.baseline_expression = None
        
    def detect_micro_expressions(
        self,
        expression_sequence: List[ExpressionVector],
        timestamps: List[float]
    ) -> List[MicroExpression]:
        """Detect micro-expressions in a sequence of expressions."""
        micro_expressions = []
        
        if len(expression_sequence) < 3:
            return micro_expressions
        
        # Analyze temporal patterns
        for i in range(2, len(expression_sequence)):
            # Check for rapid expression changes
            current = expression_sequence[i]
            previous = expression_sequence[i-1]
            baseline = expression_sequence[i-2]
            
            # Detect flash expressions
            flash_expr = self._detect_flash_expression(baseline, previous, current)
            if flash_expr:
                micro_expressions.append(flash_expr)
            
            # Detect suppressed expressions
            suppressed_expr = self._detect_suppressed_expression(baseline, previous, current)
            if suppressed_expr:
                micro_expressions.append(suppressed_expr)
        
        return micro_expressions
    
    def _detect_flash_expression(
        self,
        baseline: ExpressionVector,
        flash: ExpressionVector,
        return_expr: ExpressionVector
    ) -> Optional[MicroExpression]:
        """Detect flash micro-expressions."""
        # Look for brief expression that returns to baseline
        baseline_expr = baseline.primary_expression
        flash_expr = flash.primary_expression
        return_expr_type = return_expr.primary_expression
        
        if (baseline_expr == return_expr_type and 
            flash_expr != baseline_expr and 
            flash.confidence > 0.7):
            
            return MicroExpression(
                micro_expression_type=self._classify_micro_expression(flash_expr),
                confidence=flash.confidence,
                start_frame=0,  # Would need frame numbers
                end_frame=2,
                peak_frame=1,
                duration_ms=100,  # Typical micro-expression duration
                intensity=flash.intensity,
                facial_regions=["mouth", "eyes"]
            )
        
        return None
    
    def _detect_suppressed_expression(
        self,
        baseline: ExpressionVector,
        current: ExpressionVector,
        next_expr: ExpressionVector
    ) -> Optional[MicroExpression]:
        """Detect suppressed expressions."""
        # Look for partial expressions that are quickly controlled
        suppression_indicators = [
            'anger', 'disgust', 'fear', 'contempt'
        ]
        
        for expr in suppression_indicators:
            if (current.expression_scores.get(expr, 0) > 0.3 and
                next_expr.expression_scores.get(expr, 0) < 0.1 and
                current.authenticity < 0.8):
                
                return MicroExpression(
                    micro_expression_type=MicroExpressionType.SUPPRESSED_ANGER,
                    confidence=current.expression_scores[expr],
                    start_frame=0,
                    end_frame=1,
                    peak_frame=0,
                    duration_ms=200,
                    intensity=current.intensity,
                    facial_regions=["eyebrows", "mouth"]
                )
        
        return None
    
    def _classify_micro_expression(self, expression: str) -> MicroExpressionType:
        """Classify micro-expression type."""
        mapping = {
            'happiness': MicroExpressionType.GENUINE_SMILE,
            'anger': MicroExpressionType.SUPPRESSED_ANGER,
            'fear': MicroExpressionType.CONCEALED_FEAR,
            'disgust': MicroExpressionType.HIDDEN_DISGUST,
            'contempt': MicroExpressionType.LEAKED_CONTEMPT
        }
        return mapping.get(expression, MicroExpressionType.DECEPTION_MARKERS)


class EmotionalStateAnalyzer:
    """Analyzes emotional states from expression data."""
    
    def __init__(self, logger):
        self.logger = logger
        self.emotion_history = deque(maxlen=100)
        
    def analyze_emotional_state(
        self,
        expression_vectors: List[ExpressionVector],
        context: Optional[Dict[str, Any]] = None
    ) -> List[EmotionalState]:
        """Analyze emotional states from expression vectors."""
        emotional_states = []
        
        for expression_vector in expression_vectors:
            # Map expressions to emotions
            emotional_state = self._map_expression_to_emotion(expression_vector)
            
            # Apply contextual adjustments
            if context:
                emotional_state = self._apply_context(emotional_state, context)
            
            # Calculate stability
            emotional_state.stability = self._calculate_stability(emotional_state)
            
            emotional_states.append(emotional_state)
        
        return emotional_states
    
    def _map_expression_to_emotion(self, expression_vector: ExpressionVector) -> EmotionalState:
        """Map expression vector to emotional state."""
        # Basic emotion mapping
        emotion_map = {
            'happiness': 'joy',
            'sadness': 'sadness',
            'anger': 'anger',
            'fear': 'fear',
            'surprise': 'surprise',
            'disgust': 'disgust',
            'neutral': 'neutral'
        }
        
        primary_emotion = emotion_map.get(
            expression_vector.primary_expression, 
            'neutral'
        )
        
        # Determine emotion category
        category = EmotionCategory.BASIC
        if primary_emotion in ['confusion', 'excitement', 'relief']:
            category = EmotionCategory.COMPLEX
        
        return EmotionalState(
            primary_emotion=primary_emotion,
            emotion_category=category,
            confidence=expression_vector.confidence,
            intensity=expression_vector.intensity,
            arousal=expression_vector.arousal,
            valence=expression_vector.valence,
            dominance=expression_vector.dominance
        )
    
    def _apply_context(
        self,
        emotional_state: EmotionalState,
        context: Dict[str, Any]
    ) -> EmotionalState:
        """Apply contextual information to emotional state."""
        # Adjust based on social context
        if context.get('social_context') == 'professional':
            # People may suppress certain emotions in professional settings
            if emotional_state.primary_emotion in ['anger', 'frustration']:
                emotional_state.intensity *= 0.7
                emotional_state.confidence *= 0.8
        
        # Adjust based on cultural context
        if context.get('cultural_context') == 'eastern':
            # Some cultures suppress emotional expression
            emotional_state.intensity *= 0.9
        
        return emotional_state
    
    def _calculate_stability(self, emotional_state: EmotionalState) -> float:
        """Calculate emotional stability."""
        if len(self.emotion_history) < 3:
            return 1.0
        
        # Compare with recent history
        recent_emotions = list(self.emotion_history)[-3:]
        same_emotion_count = sum(
            1 for es in recent_emotions 
            if es.primary_emotion == emotional_state.primary_emotion
        )
        
        return same_emotion_count / len(recent_emotions)


class EnhancedExpressionAnalyzer:
    """
    Advanced facial expression analysis system for the AI assistant.
    
    Features:
    - Multi-method expression detection (CNN, landmarks, ensemble)
    - Micro-expression detection for deception analysis
    - Temporal expression analysis for video
    - Emotional state classification and tracking
    - Cultural and demographic adaptation
    - Real-time processing capabilities
    """
    
    def __init__(self, container: Container):
        """
        Initialize the enhanced expression analyzer.
        
        Args:
            container: Dependency injection container
        """
        self.container = container
        self.logger = get_logger(__name__)
        self.config = container.get(ConfigLoader)
        self.event_bus = container.get(EventBus)
        self.error_handler = container.get(ErrorHandler)
        self.health_check = container.get(HealthCheck)
        
        # Load configuration
        self._load_configuration()
        
        # Setup components
        self._setup_core_components()
        self._setup_detectors()
        self._setup_analyzers()
        self._setup_monitoring()
        self._setup_caching()
        
        # State management
        self._processing_stats = {
            'total_analyses': 0,
            'successful_analyses': 0,
            'failed_analyses': 0,
            'average_processing_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        # Thread pool for CPU-intensive operations
        self._executor = ThreadPoolExecutor(max_workers=self.analysis_config.num_threads)
        
        # Register health check
        self.health_check.register_component("expression_analyzer", self._health_check_callback)
        
        self.logger.info("EnhancedExpressionAnalyzer initialized")
    
    def _load_configuration(self) -> None:
        """Load analyzer configuration."""
        self.analysis_config = ExpressionConfiguration(
            detection_method=ExpressionDetectionMethod(
                self.config.get("expression_analysis.detection_method", "cnn")
            ),
            quality_level=AnalysisQuality(
                self.config.get("expression_analysis.quality_level", "balanced")
            ),
            enable_micro_expressions=self.config.get("expression_analysis.enable_micro_expressions", True),
            enable_temporal_analysis=self.config.get("expression_analysis.enable_temporal_analysis", True),
            enable_emotion_classification=self.config.get("expression_analysis.enable_emotion_classification", True),
            model_confidence_threshold=self.config.get("expression_analysis.model_confidence_threshold", 0.5),
            enable_gpu=self.config.get("expression_analysis.enable_gpu", True),
            enable_caching=self.config.get("expression_analysis.enable_caching", True),
            cache_ttl=self.config.get("expression_analysis.cache_ttl", 300)
        )
    
    def _setup_core_components(self) -> None:
        """Setup core processing components."""
        # Face detector for finding faces
        self.face_detector = FaceDetector(
            method=FaceDetectionMethod.CNN
        )
        
        # Feedback processor for learning
        try:
            self.feedback_processor = self.container.get(FeedbackProcessor)
        except Exception:
            self.feedback_processor = None
            self.logger.warning("FeedbackProcessor not available")
    
    def _setup_detectors(self) -> None:
        """Setup expression detection methods."""
        self.detectors = {}
        
        # CNN detector
        try:
            self.detectors['cnn'] = CNNExpressionDetector(
                self.logger,
                self.config.get("expression_analysis.cnn_model_path")
            )
        except Exception as e:
            self.logger.warning(f"Failed to setup CNN detector: {e}")
        
        # MediaPipe detector
        try:
            self.detectors['mediapipe'] = MediaPipeExpressionDetector(self.logger)
        except Exception as e:
            self.logger.warning(f"Failed to setup MediaPipe detector: {e}")
        
        # Micro-expression detector
        self.micro_expression_detector = MicroExpressionDetector(self.logger)
    
    def _setup_analyzers(self) -> None:
        """Setup analysis components."""
        self.emotional_state_analyzer = EmotionalStateAnalyzer(self.logger)
        
        # Temporal analysis buffer
        self.temporal_buffer = defaultdict(lambda: deque(maxlen=100))
    
    def _setup_monitoring(self) -> None:
        """Setup monitoring and metrics."""
        try:
            self.metrics = self.container.get(MetricsCollector)
            self.tracer = self.container.get(TraceManager)
            
            # Register metrics
            self.metrics.register_counter("expression_analyses_total")
            self.metrics.register_counter("expression_analyses_successful")
            self.metrics.register_counter("expression_analyses_failed")
            self.metrics.register_histogram("expression_analysis_duration_seconds")
            self.metrics.register_gauge("expression_detections_per_image")
            self.metrics.register_counter("micro_expressions_detected")
            
        except Exception as e:
            self.logger.warning(f"Failed to setup monitoring: {e}")
            self.metrics = None
            self.tracer = None
    
    def _setup_caching(self) -> None:
        """Setup caching for expression analysis."""
        try:
            self.cache_manager = self.container.get(CacheManager)
            self.redis_cache = self.container.get(RedisCache)
        except Exception:
            self.cache_manager = None
            self.redis_cache = None
            self.logger.warning("Caching not available")
    
    async def initialize(self) -> None:
        """Initialize all expression analysis components."""
        try:
            # Initialize face detector
            await self.face_detector.initialize()
            
            # Initialize expression detectors
            for name, detector in self.detectors.items():
                try:
                    await detector.initialize()
                    self.logger.info(f"Initialized {name} expression detector")
                except Exception as e:
                    self.logger.error(f"Failed to initialize {name} detector: {e}")
                    del self.detectors[name]
            
            if not self.detectors:
                raise ExpressionAnalysisError(
                    "No expression detectors available",
                    error_code="NO_DETECTORS"
                )
            
            # Emit initialization event
            await self.event_bus.emit(ComponentInitialized(
                component_id="expression_analyzer",
                initialization_time=0.0
            ))
            
            self.logger.info("ExpressionAnalyzer initialized successfully")
            
        except Exception as e:
            await self.event_bus.emit(ComponentFailed(
                component_id="expression_analyzer",
                error_message=str(e),
                error_type=type(e).__name__
            ))
            raise
    
    @handle_exceptions
    async def analyze_expressions(
        self,
        image: np.ndarray,
        session_id: Optional[str] = None,
        config: Optional[ExpressionConfiguration] = None
    ) -> ExpressionAnalysisResult:
        """
        Analyze facial expressions in an image.
        
        Args:
            image: Input image as numpy array
            session_id: Session identifier for temporal analysis
            config: Optional configuration override
            
        Returns:
            ExpressionAnalysisResult containing all analysis results
        """
        start_time = time.time()
        request_id = f"expr_{int(time.time()*1000)}"
        config = config or self.analysis_config
        
        try:
            # Check cache first
            if config.enable_caching and self.redis_cache:
                cache_key = self._generate_cache_key(image, config)
                cached_result = await self._get_cached_result(cache_key)
                if cached_result:
                    self._processing_stats['cache_hits'] += 1
                    return cached_result
                self._processing_stats['cache_misses'] += 1
            
            with self.tracer.trace("expression_analysis") if self.tracer else None:
                # Detect faces
                face_detection_result = await self.face_detector.detect_faces(image)
                
                if not face_detection_result.face_locations:
                    # No faces detected
                    return ExpressionAnalysisResult(
                        success=True,
                        request_id=request_id,
                        processing_time=time.time() - start_time,
                        detected_faces=[],
                        expression_vectors=[],
                        emotional_states=[],
                        micro_expressions=[],
                        analysis_method=config.detection_method,
                        quality_level=config.quality_level
                    )
                
                # Analyze expressions for each face
                detected_faces = []
                expression_vectors = []
                
                for i, face_location in enumerate(face_detection_result.face_locations):
                    face_info = {
                        'face_id': i,
                        'location': face_location,
                        'confidence': face_detection_result.confidence_scores[i]
                    }
                    detected_faces.append(face_info)
                    
                    # Get expression analysis
                    face_expressions = await self._analyze_single_face(
                        image, [face_location], config
                    )
                    expression_vectors.extend(face_expressions)
                
                # Analyze emotional states
                emotional_states = []
                if config.enable_emotion_classification:
                    emotional_states = self.emotional_state_analyzer.analyze_emotional_state(
                        expression_vectors
                    )
                
                # Detect micro-expressions (if temporal data available)
                micro_expressions = []
                if config.enable_micro_expressions and session_id:
                    micro_expressions = await self._analyze_micro_expressions(
                        session_id, expression_vectors
                    )
                
                # Calculate aggregate metrics
                overall_mood = self._calculate_overall_mood(expression_vectors)
                mood_stability = self._calculate_mood_stability(session_id, expression_vectors)
                
                # Create result
                result = ExpressionAnalysisResult(
                    success=True,
                    request_id=request_id,
                    processing_time=time.time() - start_time,
                    detected_faces=detected_faces,
                    expression_vectors=expression_vectors,
                    emotional_states=emotional_states,
                    micro_expressions=micro_expressions,
                    analysis_method=config.detection_method,
                    quality_level=config.quality_level,
                    overall_mood=overall_mood,
                    mood_stability=mood_stability,
                    detection_quality=np.mean([ev.quality_score for ev in expression_vectors]) if expression_vectors else 0.0,
                    analysis_confidence=np.mean([ev.confidence for ev in expression_vectors]) if expression_vectors else 0.0
                )
                
                # Cache result
                if config.enable_caching and self.redis_cache:
                    await self._cache_result(cache_key, result, config.cache_ttl)
                
                # Update metrics
                self._update_metrics(result.processing_time, len(expression_vectors))
                
                # Store for learning
                if self.feedback_processor:
                    await self._store_analysis_for_learning(result, image, config)
                
                # Update temporal buffer
                if session_id:
                    self.temporal_buffer[session_id].append({
                        'timestamp': time.time(),
                        'expressions': expression_vectors,
                        'emotions': emotional_states
                    })
                
                self._processing_stats['successful_analyses'] += 1
                
                return result
                
        except Exception as e:
            self._processing_stats['failed_analyses'] += 1
            
            error_result = ExpressionAnalysisResult(
                success=False,
                request_id=request_id,
                processing_time=time.time() - start_time,
                detected_faces=[],
                expression_vectors=[],
                emotional_states=[],
                micro_expressions=[],
                analysis_method=config.detection_method,
                quality_level=config.quality_level,
                errors=[str(e)]
            )
            
            if self.metrics:
                self.metrics.increment("expression_analyses_failed")
            
            self.logger.error(f"Expression analysis failed: {str(e)}")
            return error_result
        
        finally:
            self._processing_stats['total_analyses'] += 1
    
    async def _analyze_single_face(
        self,
        image: np.ndarray,
        face_locations: List[Tuple[int, int, int, int]],
        config: ExpressionConfiguration
    ) -> List[ExpressionVector]:
        """Analyze expressions for a single face."""
        method = config.detection_method.value
        
        if method in self.detectors:
            return await self.detectors[method].detect_expressions(
                image, face_locations, config
            )
        elif method == 'ensemble':
            return await self._ensemble_detection(image, face_locations, config)
        else:
            # Fallback to first available detector
            first_detector = next(iter(self.detectors.values()))
            return await first_detector.detect_expressions(
                image, face_locations, config
            )
    
    async def _ensemble_detection(
        self,
        image: np.ndarray,
        face_locations: List[Tuple[int, int, int, int]],
        config: ExpressionConfiguration
    ) -> List[ExpressionVector]:
        """Combine results from multiple detection methods."""
        all_results = []
        
        # Run multiple detectors
        for detector in self.detectors.values():
            try:
                results = await detector.detect_expressions(image, face_locations, config)
                all_results.append(results)
            except Exception as e:
                self.logger.warning(f"Detector failed in ensemble: {e}")
        
        if not all_results:
            return []
        
        # Ensemble results (simple averaging)
        ensemble_vectors = []
        for face_idx in range(len(face_locations)):
            if face_idx < len(all_results[0]):
                # Combine expression scores
                combined_scores = defaultdict(list)
                
                for result_set in all_results:
                    if face_idx < len(result_set):
                        for expr, score in result_set[face_idx].expression_scores.items():
                            combined_scores[expr].append(score)
                
                # Average scores
                final_scores = {
                    expr: np.mean(scores) for expr, scores in combined_scores.items()
                }
                
                primary_expression = max(final_scores, key=final_scores.get)
                confidence = final_scores[primary_expression]
                
                ensemble_vector = ExpressionVector(
                    expression_scores=final_scores,
                    primary_expression=primary_expression,
                    confidence=confidence,
                    analysis_method="ensemble",
                    quality_score=confidence
                )
                
                ensemble_vectors.append(ensemble_vector)
        
        return ensemble_vectors
    
    async def _analyze_micro_expressions(
        self,
        session_id: str,
        current_expressions: List[ExpressionVector]
    ) -> List[MicroExpression]:
        """Analyze micro-expressions using temporal data."""
        if session_id not in self.temporal_buffer:
            return []
        
        # Get recent expression history
        history = list(self.temporal_buffer[session_id])
        if len(history) < 3:
            return []
        
        # Extract expression sequences
        expression_sequences = []
        timestamps = []
        
        for entry in history[-10:]:  # Last 10 frames
            expression_sequences.append(entry['expressions'])
            timestamps.append(entry['timestamp'])
        
        # Detect micro-expressions
        micro_expressions = []
        if expression_sequences:
            # Flatten for analysis
            flat_sequences = [seq[0] for seq in expression_sequences if seq]
            micro_expressions = self.micro_expression_detector.detect_micro_expressions(
                flat_sequences, timestamps
            )
        
        return micro_expressions
    
    def _calculate_overall_mood(self, expression_vectors: List[ExpressionVector]) -> str:
        """Calculate overall mood from expression vectors."""
        if not expression_vectors:
            return "neutral"
        
        # Aggregate expression scores
        aggregated_scores = defaultdict(float)
        for vector in expression_vectors:
            for expr, score in vector.expression_scores.items():
                aggregated_scores[expr] += score
        
        # Average scores
        for expr in aggregated_scores:
            aggregated_scores[expr] /= len(expression_vectors)
        
        return max(aggregated_scores, key=aggregated_scores.get)
    
    def _calculate_mood_stability(
        self,
        session_id: Optional[str],
        current_expressions: List[ExpressionVector]
    ) -> float:
        """Calculate mood stability over time."""
        if not session_id or session_id not in self.temporal_buffer:
            return 1.0
        
        history = list(self.temporal_buffer[session_id])
        if len(history) < 3:
            return 1.0
        
        # Calculate consistency of primary expressions
        recent_expressions = [
            entry['expressions'][0].primary_expression 
            for entry in history[-5:] 
            if entry['expressions']
        ]
        
        if not recent_expressions:
            return 1.0
        
        # Calculate stability as consistency
        most_common = max(set(recent_expressions), key=recent_expressions.count)
        stability = recent_expressions.count(most_common) / len(recent_expressions)
        
        return stability
    
    def _generate_cache_key(self, image: np.ndarray, config: ExpressionConfiguration) -> str:
        """Generate cache key for expression analysis."""
        import hashlib
        
        # Create hash from image and config
        image_hash = hashlib.md5(image.tobytes()).hexdigest()[:16]
        config_hash = hashlib.md5(
            f"{config.detection_method.value}_{config.quality_level.value}".encode()
        ).hexdigest()[:8]
        
        return f"expr_analysis:{image_hash}:{config_hash}"
    
    async def _get_cached_result(self, cache_key: str) -> Optional[ExpressionAnalysisResult]:
        """Get cached analysis result."""
        try:
            if self.redis_cache:
                cached_data = await self.redis_cache.get(cache_key)
                if cached_data:
                    # Deserialize result (simplified)
                    return ExpressionAnalysisResult(**json.loads(cached_data))
        except Exception as e:
            self.logger.warning(f"Cache retrieval failed: {e}")
        
        return None
    
    async def _cache_result(
        self,
        cache_key: str,
        result: ExpressionAnalysisResult,
        ttl: int
    ) -> None:
        """Cache analysis result."""
        try:
            if self.redis_cache:
                # Serialize result (simplified)
                cached_data = json.dumps({
                    'success': result.success,
                    'processing_time': result.processing_time,
                    'detected_faces': result.detected_faces,
                    'overall_mood': result.overall_mood,
                    # Add other serializable fields
                })
                await self.redis_cache.set(cache_key, cached_data, ttl)
        except Exception as e:
            self.logger.warning(f"Cache storage failed: {e}")
    
    def _update_metrics(self, processing_time: float, expressions_count: int) -> None:
        """Update performance metrics."""
        if self.metrics:
            self.metrics.increment("expression_analyses_total")
            self.metrics.increment("expression_analyses_successful")
            self.metrics.record("expression_analysis_duration_seconds", processing_time)
            self.metrics.set("expression_detections_per_image", expressions_count)
        
        # Update internal stats
        total = self._processing_stats['total_analyses']
        if total > 0:
            current_avg = self._processing_stats['average_processing_time']
            self._processing_stats['average_processing_time'] = (
                (current_avg * (total - 1) + processing_time) / total
            )
    
    async def _store_analysis_for_learning(
        self,
        result: ExpressionAnalysisResult,
        image: np.ndarray,
        config: ExpressionConfiguration
    ) -> None:
        """Store analysis data for learning and improvement."""
        if not self.feedback_processor:
            return
        
        try:
            learning_data = {
                'analysis_type': 'expression_analysis',
                'result_quality': result.analysis_confidence,
                'processing_time': result.processing_time,
                'expressions_detected': len(result.expression_vectors),
                'method_used': config.detection_method.value,
                'timestamp': result.timestamp.isoformat()
            }
            
            await self.feedback_processor.store_interaction_data(learning_data)
            
        except Exception as e:
            self.logger.warning(f"Failed to store learning data: {e}")
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            'component': 'expression_analyzer',
            'status': 'healthy' if self.detectors else 'degraded',
            'detectors_available': list(self.detectors.keys()),
            'processing_stats': self._processing_stats.copy(),
            'temporal_sessions': len(self.temporal_buffer),
            'cache_enabled': self.redis_cache is not None,
            'gpu_enabled': self.analysis_config.enable_gpu,
            'configuration': {
                'detection_method': self.analysis_config.detection_method.value,
                'quality_level': self.analysis_config.quality_level.value,
                'micro_expressions_enabled': self.analysis_config.enable_micro_expressions,
                'temporal_analysis_enabled': self.analysis_config.enable_temporal_analysis
            }
        }
    
    async def _health_check_callback(self) -> Dict[str, Any]:
        """Health check callback for the component manager."""
        try:
            detectors_healthy = len(self.detectors) > 0
            success_rate = 0.0
            
            if self._processing_stats['total_analyses'] > 0:
                success_rate = (
                    self._processing_stats['successful_analyses'] / 
                    self._processing_stats['total_analyses']
                )
            
            status = "healthy" if detectors_healthy and success_rate >= 0.8 else "degraded"
            
            return {
                'status': status,
                'detectors_available': len(self.detectors),
                'success_rate': success_rate,
                'average_processing_time': self._processing_stats['average_processing_time'],
                'total_analyses': self._processing_stats['total_analyses']
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e)
            }
    
    async def cleanup(self) -> None:
        """Cleanup resources and shutdown gracefully."""
        try:
            # Cleanup detectors
            for detector in self.detectors.values():
                detector.cleanup()
            
            # Shutdown thread pool
            self._executor.shutdown(wait=True)
            
            # Clear temporal buffer
            self.temporal_buffer.clear()
            
            self.logger.info("ExpressionAnalyzer cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            if hasattr(self, '_executor'):
                self._executor.shutdown(wait=False)
        except Exception:
            pass  # Ignore cleanup errors in destructor
