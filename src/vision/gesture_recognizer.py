"""
Advanced Gesture Recognition System
Author: Drmusab
Last Modified: 2025-05-28 17:45:37 UTC

This module provides comprehensive gesture recognition capabilities for the AI assistant,
including hand gestures, body gestures, and dynamic gesture sequences with real-time tracking.
"""

from pathlib import Path
from typing import Optional, Dict, Any, List, Set, Tuple, Union, Callable, Type
import asyncio
import threading
import time
from datetime import datetime, timezone
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import logging
import numpy as np
import cv2
from collections import deque, defaultdict
import hashlib
import json

# Core imports
from src.core.config.loader import ConfigLoader
from src.core.events.event_bus import EventBus
from src.core.events.event_types import (
    ComponentInitialized,
    ComponentFailed,
    ProcessingCompleted,
    HealthCheckPerformed
)
from src.core.error_handling import ErrorHandler, handle_exceptions
from src.core.dependency_injection import Container
from src.core.health_check import HealthCheck

# Observability
from src.observability.monitoring.metrics import MetricsCollector
from src.observability.monitoring.tracing import TraceManager
from src.observability.logging.config import get_logger

# Learning and adaptation
from src.learning.feedback_processor import FeedbackProcessor

# Vision processing imports
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False


class GestureType(Enum):
    """Types of gestures that can be recognized."""
    # Hand gestures
    HAND_WAVE = "hand_wave"
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    POINTING = "pointing"
    PEACE_SIGN = "peace_sign"
    OK_SIGN = "ok_sign"
    FIST = "fist"
    OPEN_PALM = "open_palm"
    STOP_GESTURE = "stop_gesture"
    COME_HERE = "come_here"
    
    # Body gestures
    NOD_YES = "nod_yes"
    SHAKE_NO = "shake_no"
    SHRUG = "shrug"
    ARMS_CROSSED = "arms_crossed"
    HANDS_ON_HIPS = "hands_on_hips"
    CLAPPING = "clapping"
    
    # Dynamic gestures
    SWIPE_LEFT = "swipe_left"
    SWIPE_RIGHT = "swipe_right"
    SWIPE_UP = "swipe_up"
    SWIPE_DOWN = "swipe_down"
    CIRCULAR_MOTION = "circular_motion"
    ZOOM_IN = "zoom_in"
    ZOOM_OUT = "zoom_out"
    TAP = "tap"
    DOUBLE_TAP = "double_tap"
    
    # Complex gestures
    BECKONING = "beckoning"
    DISMISSAL = "dismissal"
    VICTORY = "victory"
    APPLAUSE = "applause"
    SILENCE = "silence"
    HEART_SHAPE = "heart_shape"
    
    # Custom/Unknown
    CUSTOM = "custom"
    UNKNOWN = "unknown"


class GestureCategory(Enum):
    """Categories of gestures based on their nature."""
    STATIC = "static"           # Single frame gestures
    DYNAMIC = "dynamic"         # Motion-based gestures
    SEQUENTIAL = "sequential"   # Multi-step gesture sequences
    CONTINUOUS = "continuous"   # Ongoing gestures


class RecognitionMethod(Enum):
    """Gesture recognition methods."""
    MEDIAPIPE = "mediapipe"
    CNN_CLASSIFIER = "cnn_classifier"
    LSTM_SEQUENCE = "lstm_sequence"
    TRANSFORMER = "transformer"
    HAND_CRAFTED = "hand_crafted"
    ENSEMBLE = "ensemble"
    RULE_BASED = "rule_based"


class GestureQuality(Enum):
    """Quality levels for gesture recognition."""
    FAST = "fast"           # Speed optimized
    BALANCED = "balanced"   # Speed/accuracy balance
    ACCURATE = "accurate"   # Accuracy optimized
    RESEARCH = "research"   # Maximum accuracy for research


class TrackingMode(Enum):
    """Gesture tracking modes."""
    NONE = "none"
    SIMPLE = "simple"
    KALMAN = "kalman"
    OPTICAL_FLOW = "optical_flow"
    DEEP_TRACKING = "deep_tracking"


class HandSide(Enum):
    """Hand side identification."""
    LEFT = "left"
    RIGHT = "right"
    BOTH = "both"
    UNKNOWN = "unknown"


@dataclass
class GestureKeypoint:
    """Individual gesture keypoint."""
    x: float
    y: float
    z: Optional[float] = None
    confidence: float = 0.0
    visibility: float = 1.0
    name: Optional[str] = None
    landmark_id: Optional[int] = None


@dataclass
class HandLandmarks:
    """Hand landmark data structure."""
    landmarks: List[GestureKeypoint]
    hand_side: HandSide
    confidence: float
    bbox: Tuple[float, float, float, float]  # x, y, width, height
    wrist_position: Tuple[float, float]
    palm_center: Tuple[float, float]
    fingers_extended: Dict[str, bool] = field(default_factory=dict)
    hand_orientation: float = 0.0


@dataclass
class GestureSequence:
    """Sequence of gestures for dynamic recognition."""
    sequence_id: str
    gestures: List[Dict[str, Any]]
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: float = 0.0
    confidence: float = 0.0
    is_complete: bool = False


@dataclass
class RecognizedGesture:
    """Individual recognized gesture."""
    gesture_id: str
    gesture_type: GestureType
    category: GestureCategory
    confidence: float
    
    # Spatial information
    bbox: Tuple[float, float, float, float]
    center_point: Tuple[float, float]
    hand_side: HandSide = HandSide.UNKNOWN
    
    # Temporal information
    start_frame: int = 0
    end_frame: int = 0
    duration_ms: float = 0.0
    
    # Gesture characteristics
    velocity: Optional[Tuple[float, float]] = None
    direction: Optional[float] = None
    magnitude: Optional[float] = None
    
    # Context information
    hand_landmarks: Optional[HandLandmarks] = None
    pose_context: Optional[Dict[str, Any]] = None
    
    # Quality metrics
    stability: float = 0.0
    clarity: float = 0.0
    completeness: float = 0.0
    
    # Metadata
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GestureRecognitionResult:
    """Complete gesture recognition result."""
    success: bool
    request_id: str
    processing_time: float
    
    # Recognition results
    recognized_gestures: List[RecognizedGesture]
    gesture_sequences: List[GestureSequence]
    total_gestures: int
    
    # Hand detection results
    detected_hands: List[HandLandmarks]
    hand_count: int
    
    # Image metadata
    image_dimensions: Tuple[int, int]  # height, width
    frame_number: int = 0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Processing metadata
    method_used: RecognitionMethod
    quality_level: GestureQuality
    processing_fps: Optional[float] = None
    
    # Performance metrics
    memory_usage_mb: float = 0.0
    gpu_usage_percent: float = 0.0
    
    # Quality metrics
    overall_confidence: float = 0.0
    detection_quality: float = 0.0
    tracking_quality: float = 0.0
    temporal_consistency: float = 0.0
    
    # Error information
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Additional data
    debug_info: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GestureConfiguration:
    """Configuration for gesture recognition."""
    # Recognition settings
    method: RecognitionMethod = RecognitionMethod.MEDIAPIPE
    quality_level: GestureQuality = GestureQuality.BALANCED
    tracking_mode: TrackingMode = TrackingMode.SIMPLE
    
    # Detection settings
    confidence_threshold: float = 0.7
    min_gesture_confidence: float = 0.5
    max_hands: int = 2
    enable_dynamic_gestures: bool = True
    enable_gesture_sequences: bool = True
    
    # Temporal settings
    sequence_timeout_ms: float = 2000.0
    gesture_smoothing_frames: int = 5
    min_gesture_duration_ms: float = 100.0
    max_gesture_duration_ms: float = 5000.0
    
    # Performance settings
    enable_gpu: bool = True
    batch_size: int = 1
    num_threads: int = 4
    max_memory_mb: int = 512
    
    # Input processing
    input_resolution: Tuple[int, int] = (640, 480)
    normalize_coordinates: bool = True
    flip_horizontally: bool = False
    roi_padding: float = 0.1
    
    # Feature extraction
    enable_hand_landmarks: bool = True
    enable_pose_context: bool = True
    enable_velocity_analysis: bool = True
    enable_directional_analysis: bool = True
    
    # Quality control
    enable_quality_filter: bool = True
    min_hand_area: int = 1000
    max_hand_area: int = 50000
    stability_threshold: float = 0.8
    
    # Caching
    enable_caching: bool = True
    cache_ttl: int = 300
    cache_gesture_templates: bool = True
    
    # Custom gestures
    enable_custom_gestures: bool = True
    custom_gesture_models: List[str] = field(default_factory=list)
    
    # Debugging
    enable_visualization: bool = False
    save_debug_frames: bool = False
    debug_output_dir: str = "/tmp/gesture_debug"


class GestureRecognitionError(Exception):
    """Custom exception for gesture recognition operations."""
    
    def __init__(self, message: str, error_code: Optional[str] = None, 
                 gesture_type: Optional[str] = None):
        super().__init__(message)
        self.error_code = error_code
        self.gesture_type = gesture_type
        self.timestamp = datetime.now(timezone.utc)


class GestureRecognizer(ABC):
    """Abstract base class for gesture recognizers."""
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the gesture recognizer."""
        pass
    
    @abstractmethod
    async def recognize_gestures(
        self, 
        image: np.ndarray, 
        config: Optional[GestureConfiguration] = None
    ) -> GestureRecognitionResult:
        """Recognize gestures in an image."""
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Cleanup resources."""
        pass


class MediaPipeGestureRecognizer(GestureRecognizer):
    """MediaPipe-based gesture recognizer."""
    
    def __init__(self, logger):
        self.logger = logger
        self.mp_hands = None
        self.mp_pose = None
        self.hands_detector = None
        self.pose_detector = None
        self.gesture_classifier = None
        self._initialized = False
        
        # Gesture templates
        self.gesture_templates = self._load_gesture_templates()
        
        # Tracking state
        self.tracking_history = defaultdict(lambda: deque(maxlen=30))
    
    async def initialize(self) -> None:
        """Initialize MediaPipe components."""
        if not MEDIAPIPE_AVAILABLE:
            raise GestureRecognitionError("MediaPipe not available")
        
        try:
            self.mp_hands = mp.solutions.hands
            self.mp_pose = mp.solutions.pose
            
            self.hands_detector = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.5
            )
            
            self.pose_detector = self.mp_pose.Pose(
                static_image_mode=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            
            self.gesture_classifier = GestureClassifier()
            await self.gesture_classifier.initialize()
            
            self._initialized = True
            self.logger.info("MediaPipe gesture recognizer initialized")
            
        except Exception as e:
            raise GestureRecognitionError(f"Failed to initialize MediaPipe: {str(e)}")
    
    async def recognize_gestures(
        self, 
        image: np.ndarray, 
        config: Optional[GestureConfiguration] = None
    ) -> GestureRecognitionResult:
        """Recognize gestures using MediaPipe."""
        if not self._initialized:
            await self.initialize()
        
        config = config or GestureConfiguration()
        start_time = time.time()
        
        try:
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width = image.shape[:2]
            
            # Detect hands
            hands_results = self.hands_detector.process(rgb_image)
            detected_hands = []
            
            if hands_results.multi_hand_landmarks:
                for i, (hand_landmarks, handedness) in enumerate(
                    zip(hands_results.multi_hand_landmarks, hands_results.multi_handedness)
                ):
                    hand_data = self._extract_hand_landmarks(
                        hand_landmarks, handedness, width, height
                    )
                    detected_hands.append(hand_data)
            
            # Detect pose for context
            pose_context = None
            if config.enable_pose_context:
                pose_results = self.pose_detector.process(rgb_image)
                if pose_results.pose_landmarks:
                    pose_context = self._extract_pose_context(pose_results.pose_landmarks)
            
            # Recognize gestures
            recognized_gestures = []
            for hand in detected_hands:
                gesture = await self._classify_hand_gesture(hand, pose_context)
                if gesture and gesture.confidence >= config.min_gesture_confidence:
                    recognized_gestures.append(gesture)
            
            # Recognize dynamic gestures
            gesture_sequences = []
            if config.enable_dynamic_gestures:
                sequences = await self._recognize_dynamic_gestures(
                    detected_hands, config
                )
                gesture_sequences.extend(sequences)
            
            processing_time = time.time() - start_time
            
            return GestureRecognitionResult(
                success=True,
                request_id=f"gesture_{int(time.time() * 1000)}",
                processing_time=processing_time,
                recognized_gestures=recognized_gestures,
                gesture_sequences=gesture_sequences,
                total_gestures=len(recognized_gestures),
                detected_hands=detected_hands,
                hand_count=len(detected_hands),
                image_dimensions=(height, width),
                method_used=RecognitionMethod.MEDIAPIPE,
                quality_level=config.quality_level,
                overall_confidence=self._calculate_overall_confidence(recognized_gestures),
                detection_quality=self._calculate_detection_quality(detected_hands),
                timestamp=datetime.now(timezone.utc)
            )
            
        except Exception as e:
            self.logger.error(f"Gesture recognition failed: {str(e)}")
            return GestureRecognitionResult(
                success=False,
                request_id=f"gesture_{int(time.time() * 1000)}",
                processing_time=time.time() - start_time,
                recognized_gestures=[],
                gesture_sequences=[],
                total_gestures=0,
                detected_hands=[],
                hand_count=0,
                image_dimensions=image.shape[:2],
                method_used=RecognitionMethod.MEDIAPIPE,
                quality_level=config.quality_level,
                errors=[str(e)]
            )
    
    def _extract_hand_landmarks(
        self, 
        hand_landmarks, 
        handedness, 
        width: int, 
        height: int
    ) -> HandLandmarks:
        """Extract hand landmarks from MediaPipe results."""
        landmarks = []
        for i, landmark in enumerate(hand_landmarks.landmark):
            landmarks.append(GestureKeypoint(
                x=landmark.x * width,
                y=landmark.y * height,
                z=landmark.z,
                confidence=1.0,  # MediaPipe doesn't provide per-landmark confidence
                visibility=landmark.visibility if hasattr(landmark, 'visibility') else 1.0,
                landmark_id=i
            ))
        
        # Determine hand side
        hand_side = HandSide.LEFT if handedness.classification[0].label == "Left" else HandSide.RIGHT
        
        # Calculate bounding box
        x_coords = [lm.x for lm in landmarks]
        y_coords = [lm.y for lm in landmarks]
        bbox = (
            min(x_coords),
            min(y_coords),
            max(x_coords) - min(x_coords),
            max(y_coords) - min(y_coords)
        )
        
        # Calculate key points
        wrist_position = (landmarks[0].x, landmarks[0].y)
        palm_center = self._calculate_palm_center(landmarks)
        
        # Analyze finger extensions
        fingers_extended = self._analyze_finger_extensions(landmarks)
        
        # Calculate hand orientation
        hand_orientation = self._calculate_hand_orientation(landmarks)
        
        return HandLandmarks(
            landmarks=landmarks,
            hand_side=hand_side,
            confidence=handedness.classification[0].score,
            bbox=bbox,
            wrist_position=wrist_position,
            palm_center=palm_center,
            fingers_extended=fingers_extended,
            hand_orientation=hand_orientation
        )
    
    def _calculate_palm_center(self, landmarks: List[GestureKeypoint]) -> Tuple[float, float]:
        """Calculate the center of the palm."""
        # Use landmarks 0, 5, 9, 13, 17 (base of each finger)
        palm_points = [landmarks[i] for i in [0, 5, 9, 13, 17]]
        center_x = sum(p.x for p in palm_points) / len(palm_points)
        center_y = sum(p.y for p in palm_points) / len(palm_points)
        return (center_x, center_y)
    
    def _analyze_finger_extensions(self, landmarks: List[GestureKeypoint]) -> Dict[str, bool]:
        """Analyze which fingers are extended."""
        finger_tips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
        finger_pips = [3, 6, 10, 14, 18]  # PIP joints
        
        fingers = {}
        finger_names = ["thumb", "index", "middle", "ring", "pinky"]
        
        for i, (tip, pip, name) in enumerate(zip(finger_tips, finger_pips, finger_names)):
            if i == 0:  # Thumb special case
                fingers[name] = landmarks[tip].x > landmarks[pip].x
            else:
                fingers[name] = landmarks[tip].y < landmarks[pip].y
        
        return fingers
    
    def _calculate_hand_orientation(self, landmarks: List[GestureKeypoint]) -> float:
        """Calculate hand orientation angle."""
        wrist = landmarks[0]
        middle_mcp = landmarks[9]
        
        dx = middle_mcp.x - wrist.x
        dy = middle_mcp.y - wrist.y
        
        return np.arctan2(dy, dx)
    
    def _extract_pose_context(self, pose_landmarks) -> Dict[str, Any]:
        """Extract pose context for gesture recognition."""
        # Extract key pose landmarks for context
        context = {
            "left_shoulder": (pose_landmarks.landmark[11].x, pose_landmarks.landmark[11].y),
            "right_shoulder": (pose_landmarks.landmark[12].x, pose_landmarks.landmark[12].y),
            "left_elbow": (pose_landmarks.landmark[13].x, pose_landmarks.landmark[13].y),
            "right_elbow": (pose_landmarks.landmark[14].x, pose_landmarks.landmark[14].y),
            "left_wrist": (pose_landmarks.landmark[15].x, pose_landmarks.landmark[15].y),
            "right_wrist": (pose_landmarks.landmark[16].x, pose_landmarks.landmark[16].y)
        }
        return context
    
    async def _classify_hand_gesture(
        self, 
        hand: HandLandmarks, 
        pose_context: Optional[Dict[str, Any]]
    ) -> Optional[RecognizedGesture]:
        """Classify a hand gesture."""
        # Use gesture classifier
        gesture_type, confidence = await self.gesture_classifier.classify(hand, pose_context)
        
        if gesture_type == GestureType.UNKNOWN:
            return None
        
        return RecognizedGesture(
            gesture_id=f"gesture_{int(time.time() * 1000)}_{hash(str(hand.landmarks))}",
            gesture_type=gesture_type,
            category=GestureCategory.STATIC,
            confidence=confidence,
            bbox=hand.bbox,
            center_point=hand.palm_center,
            hand_side=hand.hand_side,
            hand_landmarks=hand,
            pose_context=pose_context,
            stability=self._calculate_gesture_stability(hand),
            clarity=confidence,
            completeness=1.0
        )
    
    async def _recognize_dynamic_gestures(
        self, 
        hands: List[HandLandmarks], 
        config: GestureConfiguration
    ) -> List[GestureSequence]:
        """Recognize dynamic gestures from hand movement."""
        sequences = []
        
        for hand in hands:
            hand_id = f"{hand.hand_side.value}_{hash(str(hand.wrist_position))}"
            self.tracking_history[hand_id].append({
                "timestamp": time.time(),
                "position": hand.palm_center,
                "landmarks": hand.landmarks
            })
            
            # Analyze movement patterns
            if len(self.tracking_history[hand_id]) >= 10:
                sequence = self._analyze_movement_sequence(
                    self.tracking_history[hand_id], hand_id
                )
                if sequence:
                    sequences.append(sequence)
        
        return sequences
    
    def _analyze_movement_sequence(
        self, 
        history: deque, 
        hand_id: str
    ) -> Optional[GestureSequence]:
        """Analyze movement sequence for dynamic gestures."""
        if len(history) < 10:
            return None
        
        # Calculate velocity and direction
        positions = [h["position"] for h in history]
        velocities = []
        
        for i in range(1, len(positions)):
            dx = positions[i][0] - positions[i-1][0]
            dy = positions[i][1] - positions[i-1][1]
            velocities.append((dx, dy))
        
        # Detect gesture patterns
        gesture_type = self._detect_movement_pattern(velocities)
        
        if gesture_type != GestureType.UNKNOWN:
            return GestureSequence(
                sequence_id=f"seq_{hand_id}_{int(time.time())}",
                gestures=[{
                    "type": gesture_type.value,
                    "confidence": 0.8,
                    "velocities": velocities[-5:]  # Last 5 velocities
                }],
                start_time=datetime.fromtimestamp(history[0]["timestamp"], timezone.utc),
                end_time=datetime.fromtimestamp(history[-1]["timestamp"], timezone.utc),
                duration_ms=(history[-1]["timestamp"] - history[0]["timestamp"]) * 1000,
                confidence=0.8,
                is_complete=True
            )
        
        return None
    
    def _detect_movement_pattern(self, velocities: List[Tuple[float, float]]) -> GestureType:
        """Detect gesture type from movement velocities."""
        if len(velocities) < 5:
            return GestureType.UNKNOWN
        
        # Calculate dominant direction
        total_dx = sum(v[0] for v in velocities)
        total_dy = sum(v[1] for v in velocities)
        
        # Swipe detection
        if abs(total_dx) > abs(total_dy) * 2:  # Horizontal swipe
            if total_dx > 50:
                return GestureType.SWIPE_RIGHT
            elif total_dx < -50:
                return GestureType.SWIPE_LEFT
        elif abs(total_dy) > abs(total_dx) * 2:  # Vertical swipe
            if total_dy > 50:
                return GestureType.SWIPE_DOWN
            elif total_dy < -50:
                return GestureType.SWIPE_UP
        
        # Circular motion detection
        if self._is_circular_motion(velocities):
            return GestureType.CIRCULAR_MOTION
        
        return GestureType.UNKNOWN
    
    def _is_circular_motion(self, velocities: List[Tuple[float, float]]) -> bool:
        """Detect circular motion in velocities."""
        if len(velocities) < 8:
            return False
        
        # Calculate angle changes
        angles = []
        for dx, dy in velocities:
            angle = np.arctan2(dy, dx)
            angles.append(angle)
        
        # Check for consistent angular change
        angle_changes = []
        for i in range(1, len(angles)):
            change = angles[i] - angles[i-1]
            # Normalize angle change to [-π, π]
            while change > np.pi:
                change -= 2 * np.pi
            while change < -np.pi:
                change += 2 * np.pi
            angle_changes.append(change)
        
        # Circular motion should have consistent direction
        positive_changes = sum(1 for c in angle_changes if c > 0)
        negative_changes = sum(1 for c in angle_changes if c < 0)
        
        return abs(positive_changes - negative_changes) > len(angle_changes) * 0.6
    
    def _calculate_gesture_stability(self, hand: HandLandmarks) -> float:
        """Calculate gesture stability score."""
        # For now, return a default value
        # In a real implementation, this would analyze temporal consistency
        return 0.8
    
    def _calculate_overall_confidence(self, gestures: List[RecognizedGesture]) -> float:
        """Calculate overall confidence score."""
        if not gestures:
            return 0.0
        return sum(g.confidence for g in gestures) / len(gestures)
    
    def _calculate_detection_quality(self, hands: List[HandLandmarks]) -> float:
        """Calculate detection quality score."""
        if not hands:
            return 0.0
        return sum(h.confidence for h in hands) / len(hands)
    
    def _load_gesture_templates(self) -> Dict[str, Any]:
        """Load gesture templates for classification."""
        # This would load pre-trained gesture templates
        return {}
    
    def cleanup(self) -> None:
        """Cleanup MediaPipe resources."""
        if self.hands_detector:
            self.hands_detector.close()
        if self.pose_detector:
            self.pose_detector.close()
        self._initialized = False


class GestureClassifier:
    """Classifies hand gestures based on landmarks."""
    
    def __init__(self):
        self.gesture_rules = self._define_gesture_rules()
    
    async def initialize(self) -> None:
        """Initialize the gesture classifier."""
        pass
    
    async def classify(
        self, 
        hand: HandLandmarks, 
        pose_context: Optional[Dict[str, Any]]
    ) -> Tuple[GestureType, float]:
        """Classify a hand gesture."""
        for gesture_type, rule_func in self.gesture_rules.items():
            confidence = rule_func(hand, pose_context)
            if confidence > 0.5:
                return gesture_type, confidence
        
        return GestureType.UNKNOWN, 0.0
    
    def _define_gesture_rules(self) -> Dict[GestureType, Callable]:
        """Define rule-based gesture classification."""
        return {
            GestureType.THUMBS_UP: self._is_thumbs_up,
            GestureType.THUMBS_DOWN: self._is_thumbs_down,
            GestureType.PEACE_SIGN: self._is_peace_sign,
            GestureType.OK_SIGN: self._is_ok_sign,
            GestureType.FIST: self._is_fist,
            GestureType.OPEN_PALM: self._is_open_palm,
            GestureType.POINTING: self._is_pointing,
            GestureType.STOP_GESTURE: self._is_stop_gesture
        }
    
    def _is_thumbs_up(self, hand: HandLandmarks, pose_context: Optional[Dict[str, Any]]) -> float:
        """Check if gesture is thumbs up."""
        fingers = hand.fingers_extended
        if (fingers.get("thumb", False) and 
            not fingers.get("index", True) and 
            not fingers.get("middle", True) and 
            not fingers.get("ring", True) and 
            not fingers.get("pinky", True)):
            return 0.9
        return 0.0
    
    def _is_thumbs_down(self, hand: HandLandmarks, pose_context: Optional[Dict[str, Any]]) -> float:
        """Check if gesture is thumbs down."""
        # Implementation would check for inverted thumb position
        return 0.0
    
    def _is_peace_sign(self, hand: HandLandmarks, pose_context: Optional[Dict[str, Any]]) -> float:
        """Check if gesture is peace sign."""
        fingers = hand.fingers_extended
        if (fingers.get("index", False) and 
            fingers.get("middle", False) and 
            not fingers.get("ring", True) and 
            not fingers.get("pinky", True)):
            return 0.9
        return 0.0
    
    def _is_ok_sign(self, hand: HandLandmarks, pose_context: Optional[Dict[str, Any]]) -> float:
        """Check if gesture is OK sign."""
        # Implementation would check for thumb-index circle
        return 0.0
    
    def _is_fist(self, hand: HandLandmarks, pose_context: Optional[Dict[str, Any]]) -> float:
        """Check if gesture is a fist."""
        fingers = hand.fingers_extended
        if not any(fingers.values()):
            return 0.9
        return 0.0
    
    def _is_open_palm(self, hand: HandLandmarks, pose_context: Optional[Dict[str, Any]]) -> float:
        """Check if gesture is open palm."""
        fingers = hand.fingers_extended
        if all(fingers.values()):
            return 0.9
        return 0.0
    
    def _is_pointing(self, hand: HandLandmarks, pose_context: Optional[Dict[str, Any]]) -> float:
        """Check if gesture is pointing."""
        fingers = hand.fingers_extended
        if (fingers.get("index", False) and 
            not fingers.get("middle", True) and 
            not fingers.get("ring", True) and 
            not fingers.get("pinky", True)):
            return 0.9
        return 0.0
    
    def _is_stop_gesture(self, hand: HandLandmarks, pose_context: Optional[Dict[str, Any]]) -> float:
        """Check if gesture is stop gesture."""
        return self._is_open_palm(hand, pose_context)


class GestureTracker:
    """Tracks gestures across frames for temporal consistency."""
    
    def __init__(self, max_tracks: int = 10):
        self.active_tracks = {}
        self.max_tracks = max_tracks
        self.next_track_id = 0
    
    def update_tracks(
        self, 
        gestures: List[RecognizedGesture]
    ) -> List[RecognizedGesture]:
        """Update gesture tracks with new detections."""
        # Simple tracking based on spatial proximity
        updated_gestures = []
        
        for gesture in gestures:
            track_id = self._find_matching_track(gesture)
            if track_id is None:
                track_id = self._create_new_track(gesture)
            
            self._update_track(track_id, gesture)
            gesture.metadata["track_id"] = track_id
            updated_gestures.append(gesture)
        
        # Remove old tracks
        self._cleanup_old_tracks()
        
        return updated_gestures
    
    def _find_matching_track(self, gesture: RecognizedGesture) -> Optional[int]:
        """Find matching track for a gesture."""
        min_distance = float('inf')
        best_track = None
        
        for track_id, track_data in self.active_tracks.items():
            distance = self._calculate_distance(
                gesture.center_point, 
                track_data["last_position"]
            )
            if distance < min_distance and distance < 100:  # 100 pixel threshold
                min_distance = distance
                best_track = track_id
        
        return best_track
    
    def _create_new_track(self, gesture: RecognizedGesture) -> int:
        """Create a new track for a gesture."""
        track_id = self.next_track_id
        self.next_track_id += 1
        
        self.active_tracks[track_id] = {
            "last_position": gesture.center_point,
            "last_update": time.time(),
            "gesture_history": [gesture.gesture_type]
        }
        
        return track_id
    
    def _update_track(self, track_id: int, gesture: RecognizedGesture) -> None:
        """Update an existing track."""
        track_data = self.active_tracks[track_id]
        track_data["last_position"] = gesture.center_point
        track_data["last_update"] = time.time()
        track_data["gesture_history"].append(gesture.gesture_type)
        
        # Keep only recent history
        if len(track_data["gesture_history"]) > 10:
            track_data["gesture_history"] = track_data["gesture_history"][-10:]
    
    def _calculate_distance(
        self, 
        pos1: Tuple[float, float], 
        pos2: Tuple[float, float]
    ) -> float:
        """Calculate Euclidean distance between two positions."""
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def _cleanup_old_tracks(self) -> None:
        """Remove tracks that haven't been updated recently."""
        current_time = time.time()
        tracks_to_remove = []
        
        for track_id, track_data in self.active_tracks.items():
            if current_time - track_data["last_update"] > 2.0:  # 2 second timeout
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.active_tracks[track_id]


class EnhancedGestureRecognizer:
    """
    Enhanced gesture recognition system for the AI assistant.
    
    Features:
    - Multi-method gesture recognition (MediaPipe, CNN, LSTM)
    - Real-time hand tracking and gesture classification
    - Dynamic gesture sequence recognition
    - Temporal consistency and tracking
    - Integration with pose estimation for context
    - Performance monitoring and caching
    - Learning and adaptation capabilities
    """
    
    def __init__(self, container: Container):
        """
        Initialize the enhanced gesture recognizer.
        
        Args:
            container: Dependency injection container
        """
        self.container = container
        self.logger = get_logger(__name__)
        
        # Load configuration
        self._load_configuration()
        
        # Setup core components
        self._setup_core_components()
        
        # Setup recognizers
        self._setup_recognizers()
        
        # Setup monitoring and caching
        self._setup_monitoring()
        self._setup_caching()
        
        # State management
        self._initialized = False
        self.gesture_tracker = GestureTracker()
        
        # Performance tracking
        self._gesture_count = 0
        self._total_processing_time = 0.0
        
        self.logger.info("EnhancedGestureRecognizer initialized")
    
    def _load_configuration(self) -> None:
        """Load gesture recognition configuration."""
        try:
            config_loader = self.container.get(ConfigLoader)
            gesture_config = config_loader.get("gesture_recognition", {})
            
            self.config = GestureConfiguration(
                method=RecognitionMethod(gesture_config.get("method", "mediapipe")),
                quality_level=GestureQuality(gesture_config.get("quality_level", "balanced")),
                confidence_threshold=gesture_config.get("confidence_threshold", 0.7),
                enable_dynamic_gestures=gesture_config.get("enable_dynamic_gestures", True),
                enable_caching=gesture_config.get("enable_caching", True),
                cache_ttl=gesture_config.get("cache_ttl", 300)
            )
            
        except Exception as e:
            self.logger.warning(f"Failed to load configuration: {str(e)}")
            self.config = GestureConfiguration()
    
    def _setup_core_components(self) -> None:
        """Setup core system components."""
        try:
            self.event_bus = self.container.get(EventBus)
            self.error_handler = self.container.get(ErrorHandler)
            self.health_check = self.container.get(HealthCheck)
        except Exception as e:
            self.logger.warning(f"Some core components not available: {str(e)}")
            self.event_bus = None
            self.error_handler = None
            self.health_check = None
    
    def _setup_recognizers(self) -> None:
        """Setup gesture recognizers."""
        self.recognizers = {}
        
        # MediaPipe recognizer
        if MEDIAPIPE_AVAILABLE:
            self.recognizers[RecognitionMethod.MEDIAPIPE] = MediaPipeGestureRecognizer(
                self.logger
            )
        
        # Default to MediaPipe if available
        if RecognitionMethod.MEDIAPIPE in self.recognizers:
            self.primary_recognizer = self.recognizers[RecognitionMethod.MEDIAPIPE]
        else:
            self.logger.warning("No gesture recognizers available")
            self.primary_recognizer = None
    
    def _setup_monitoring(self) -> None:
        """Setup monitoring and metrics collection."""
        try:
            self.metrics = self.container.get(MetricsCollector)
            self.tracer = self.container.get(TraceManager)
            
            # Register gesture recognition metrics
            if self.metrics:
                self.metrics.register_counter("gesture_recognitions_total")
                self.metrics.register_counter("gesture_recognition_failures_total")
                self.metrics.register_histogram("gesture_recognition_duration_seconds")
                self.metrics.register_gauge("active_gesture_tracks")
                self.metrics.register_histogram("gesture_confidence_score")
                
        except Exception as e:
            self.logger.warning(f"Failed to setup monitoring: {str(e)}")
            self.metrics = None
            self.tracer = None
    
    def _setup_caching(self) -> None:
        """Setup caching for gesture recognition."""
        try:
            from src.integrations.cache.redis_cache import RedisCache
            self.cache = self.container.get(RedisCache)
        except Exception as e:
            self.logger.warning(f"Cache not available: {str(e)}")
            self.cache = None
    
    async def initialize(self) -> None:
        """Initialize the gesture recognition system."""
        if self._initialized:
            return
        
        try:
            # Initialize recognizers
            if self.primary_recognizer:
                await self.primary_recognizer.initialize()
            
            # Register health check
            if self.health_check:
                self.health_check.register_component(
                    "gesture_recognizer", 
                    self._health_check_callback
                )
            
            # Emit initialization event
            if self.event_bus:
                await self.event_bus.emit(ComponentInitialized(
                    component_id="gesture_recognizer",
                    initialization_time=0.0
                ))
            
            self._initialized = True
            self.logger.info("Gesture recognition system initialized")
            
        except Exception as e:
            error_msg = f"Failed to initialize gesture recognizer: {str(e)}"
            self.logger.error(error_msg)
            
            if self.event_bus:
                await self.event_bus.emit(ComponentFailed(
                    component_id="gesture_recognizer",
                    error_message=error_msg,
                    error_type=type(e).__name__
                ))
            
            raise GestureRecognitionError(error_msg)
    
    @handle_exceptions
    async def recognize_gestures(
        self, 
        image: np.ndarray, 
        config: Optional[GestureConfiguration] = None
    ) -> GestureRecognitionResult:
        """
        Recognize gestures in an image.
        
        Args:
            image: Input image as numpy array
            config: Optional configuration override
            
        Returns:
            Gesture recognition result
        """
        if not self._initialized:
            await self.initialize()
        
        if self.primary_recognizer is None:
            raise GestureRecognitionError("No gesture recognizer available")
        
        config = config or self.config
        request_id = f"gesture_{int(time.time() * 1000)}"
        
        # Check cache
        if config.enable_caching and self.cache:
            cache_key = self._generate_cache_key(image, config)
            cached_result = await self._get_cached_result(cache_key)
            if cached_result:
                return cached_result
        
        start_time = time.time()
        
        try:
            with self.tracer.trace("gesture_recognition") if self.tracer else nullcontext():
                # Perform gesture recognition
                result = await self.primary_recognizer.recognize_gestures(image, config)
                
                # Apply tracking
                if result.success and result.recognized_gestures:
                    result.recognized_gestures = self.gesture_tracker.update_tracks(
                        result.recognized_gestures
                    )
                
                # Update metrics
                processing_time = time.time() - start_time
                self._update_metrics(processing_time, len(result.recognized_gestures))
                
                # Cache result
                if config.enable_caching and self.cache and result.success:
                    await self._cache_result(cache_key, result, config.cache_ttl)
                
                # Store for learning
                if result.success:
                    await self._store_recognition_for_learning(result, image, config)
                
                # Emit event
                if self.event_bus:
                    await self.event_bus.emit(ProcessingCompleted(
                        component_id="gesture_recognizer",
                        processing_time=processing_time,
                        success=result.success
                    ))
                
                return result
                
        except Exception as e:
            error_msg = f"Gesture recognition failed: {str(e)}"
            self.logger.error(error_msg)
            
            if self.metrics:
                self.metrics.increment("gesture_recognition_failures_total")
            
            # Return error result
            return GestureRecognitionResult(
                success=False,
                request_id=request_id,
                processing_time=time.time() - start_time,
                recognized_gestures=[],
                gesture_sequences=[],
                total_gestures=0,
                detected_hands=[],
                hand_count=0,
                image_dimensions=image.shape[:2],
                method_used=config.method,
                quality_level=config.quality_level,
                errors=[error_msg]
            )
    
    def _generate_cache_key(self, image: np.ndarray, config: GestureConfiguration) -> str:
        """Generate cache key for gesture recognition."""
        image_hash = hashlib.md5(image.tobytes()).hexdigest()[:16]
        config_hash = hashlib.md5(
            json.dumps(config.__dict__, default=str, sort_keys=True).encode()
        ).hexdigest()[:8]
        return f"gesture:{image_hash}:{config_hash}"
    
    async def _get_cached_result(self, cache_key: str) -> Optional[GestureRecognitionResult]:
        """Get cached gesture recognition result."""
        try:
            cached_data = await self.cache.get(cache_key)
            if cached_data:
                # Deserialize result (simplified)
                return GestureRecognitionResult(**cached_data)
        except Exception as e:
            self.logger.warning(f"Cache retrieval failed: {str(e)}")
        return None
    
    async def _cache_result(
        self, 
        cache_key: str, 
        result: GestureRecognitionResult, 
        ttl: int
    ) -> None:
        """Cache gesture recognition result."""
        try:
            # Serialize result (simplified)
            cache_data = result.__dict__.copy()
            # Remove non-serializable items
            cache_data.pop('recognized_gestures', None)
            cache_data.pop('gesture_sequences', None)
            cache_data.pop('detected_hands', None)
            
            await self.cache.set(cache_key, cache_data, ttl)
        except Exception as e:
            self.logger.warning(f"Cache storage failed: {str(e)}")
    
    def _update_metrics(self, processing_time: float, gesture_count: int) -> None:
        """Update performance metrics."""
        if self.metrics:
            self.metrics.increment("gesture_recognitions_total")
            self.metrics.record("gesture_recognition_duration_seconds", processing_time)
            self.metrics.set("active_gesture_tracks", len(self.gesture_tracker.active_tracks))
        
        self._gesture_count += gesture_count
        self._total_processing_time += processing_time
    
    async def _store_recognition_for_learning(
        self, 
        result: GestureRecognitionResult, 
        image: np.ndarray, 
        config: GestureConfiguration
    ) -> None:
        """Store gesture recognition result for learning."""
        try:
            feedback_processor = self.container.get(FeedbackProcessor)
            
            learning_data = {
                "component": "gesture_recognizer",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "method": config.method.value,
                "gesture_count": result.total_gestures,
                "confidence": result.overall_confidence,
                "processing_time": result.processing_time,
                "image_dimensions": result.image_dimensions,
                "gestures": [
                    {
                        "type": g.gesture_type.value,
                        "confidence": g.confidence,
                        "hand_side": g.hand_side.value
                    }
                    for g in result.recognized_gestures
                ]
            }
            
            await feedback_processor.process_component_feedback(
                "gesture_recognizer", 
                learning_data
            )
            
        except Exception as e:
            self.logger.warning(f"Failed to store learning data: {str(e)}")
    
    async def switch_method(self, method: RecognitionMethod) -> None:
        """Switch gesture recognition method."""
        if method in self.recognizers:
            self.primary_recognizer = self.recognizers[method]
            self.config.method = method
            self.logger.info(f"Switched to gesture recognition method: {method.value}")
        else:
            raise GestureRecognitionError(f"Method {method.value} not available")
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get gesture recognition system status."""
        return {
            "initialized": self._initialized,
            "primary_method": self.config.method.value if self.config else "unknown",
            "available_methods": [m.value for m in self.recognizers.keys()],
            "active_tracks": len(self.gesture_tracker.active_tracks),
            "total_gestures_recognized": self._gesture_count,
            "average_processing_time": (
                self._total_processing_time / max(1, self._gesture_count)
            ),
            "configuration": self.config.__dict__ if self.config else {}
        }
    
    async def _health_check_callback(self) -> Dict[str, Any]:
        """Health check callback for the gesture recognizer."""
        try:
            # Basic health metrics
            status = "healthy"
            
            if not self._initialized:
                status = "initializing"
            elif self.primary_recognizer is None:
                status = "unhealthy"
            
            return {
                "status": status,
                "initialized": self._initialized,
                "primary_recognizer_available": self.primary_recognizer is not None,
                "active_tracks": len(self.gesture_tracker.active_tracks),
                "total_recognitions": self._gesture_count
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    async def cleanup(self) -> None:
        """Cleanup gesture recognition resources."""
        try:
            # Cleanup recognizers
            for recognizer in self.recognizers.values():
                recognizer.cleanup()
            
            # Clear tracking state
            self.gesture_tracker.active_tracks.clear()
            
            self._initialized = False
            self.logger.info("Gesture recognition system cleaned up")
            
        except Exception as e:
            self.logger.error(f"Cleanup failed: {str(e)}")
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            if hasattr(self, '_initialized') and self._initialized:
                # Run cleanup in a new event loop if needed
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # Schedule cleanup for later
                        loop.create_task(self.cleanup())
                    else:
                        loop.run_until_complete(self.cleanup())
                except RuntimeError:
                    # Create new event loop for cleanup
                    loop = asyncio.new_event_loop()
                    loop.run_until_complete(self.cleanup())
                    loop.close()
        except Exception:
            pass  # Ignore cleanup errors in destructor


# Utility context manager for cases where tracer might be None
class nullcontext:
    """Null context manager for when tracer is not available."""
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
