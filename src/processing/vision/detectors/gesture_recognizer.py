"""
Advanced Gesture Recognition System
Author: Drmusab
Last Modified: 2025-06-03 19:40:19 UTC

This module provides comprehensive gesture recognition capabilities for the AI assistant,
including hand gesture detection, body language interpretation, and dynamic gesture
analysis with real-time processing and integration with the core system components.
"""

from pathlib import Path
from typing import Optional, Dict, Any, List, Set, Tuple, Union, AsyncGenerator, Callable
import asyncio
import threading
import time
from datetime import datetime, timezone
from dataclasses import dataclass, field
from enum import Enum
from contextlib import asynccontextmanager
import logging
import uuid
import json
import pickle
from concurrent.futures import ThreadPoolExecutor
import traceback

import numpy as np
import cv2
import mediapipe as mp
from scipy.spatial.distance import euclidean
from scipy.signal import savgol_filter
import tensorflow as tf
import torch
import torch.nn as nn

# Core imports
from src.core.config.loader import ConfigLoader
from src.core.events.event_bus import EventBus
from src.core.events.event_types import (
    ProcessingStarted, ProcessingCompleted, ProcessingError,
    ComponentHealthChanged, ErrorOccurred, SystemStateChanged,
    UserInteractionStarted, UserInteractionCompleted
)
from src.core.error_handling import ErrorHandler, handle_exceptions
from src.core.dependency_injection import Container
from src.core.health_check import HealthCheck
from src.core.security.sanitization import SecuritySanitizer

# Assistant components
from src.assistant.component_manager import EnhancedComponentManager, ComponentMetadata, ComponentPriority
from src.assistant.workflow_orchestrator import WorkflowOrchestrator, WorkflowDefinition
from src.assistant.session_manager import EnhancedSessionManager
from src.assistant.interaction_handler import InteractionHandler

# Memory and learning
from src.memory.memory_manager import MemoryManager
from src.learning.feedback_processor import FeedbackProcessor
from src.learning.continual_learning import ContinualLearner

# Observability
from src.observability.monitoring.metrics import MetricsCollector
from src.observability.monitoring.tracing import TraceManager
from src.observability.logging.config import get_logger

# Vision processing
from src.processing.vision.vision_processor import VisionProcessor
from src.processing.vision.image_analyzer import ImageAnalyzer


class GestureType(Enum):
    """Types of gestures that can be recognized."""
    STATIC_HAND = "static_hand"           # Static hand poses
    DYNAMIC_HAND = "dynamic_hand"         # Hand movements/trajectories
    BODY_POSE = "body_pose"               # Full body gestures
    FACIAL_EXPRESSION = "facial_expression"  # Face gestures
    POINTING = "pointing"                 # Pointing gestures
    SWIPE = "swipe"                      # Swipe gestures
    PINCH = "pinch"                      # Pinch/zoom gestures
    WAVE = "wave"                        # Waving gestures
    THUMBS_UP = "thumbs_up"              # Approval gestures
    THUMBS_DOWN = "thumbs_down"          # Disapproval gestures
    PEACE_SIGN = "peace_sign"            # Peace sign
    OK_SIGN = "ok_sign"                  # OK gesture
    STOP_SIGN = "stop_sign"              # Stop gesture
    CUSTOM = "custom"                    # Custom defined gestures


class RecognitionMode(Enum):
    """Gesture recognition modes."""
    REAL_TIME = "real_time"              # Real-time processing
    BATCH = "batch"                      # Batch processing
    STREAMING = "streaming"              # Continuous streaming
    TRIGGERED = "triggered"              # Event-triggered recognition
    ADAPTIVE = "adaptive"                # Adaptive recognition


class GestureState(Enum):
    """States of gesture recognition."""
    IDLE = "idle"
    DETECTING = "detecting"
    TRACKING = "tracking"
    RECOGNIZING = "recognizing"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


class ConfidenceLevel(Enum):
    """Confidence levels for gesture recognition."""
    VERY_LOW = 0.2
    LOW = 0.4
    MEDIUM = 0.6
    HIGH = 0.8
    VERY_HIGH = 0.95


@dataclass
class GestureConfiguration:
    """Configuration for gesture recognition."""
    # Recognition settings
    recognition_mode: RecognitionMode = RecognitionMode.REAL_TIME
    min_confidence_threshold: float = 0.7
    max_hands: int = 2
    enable_body_pose: bool = True
    enable_face_detection: bool = True
    
    # Performance settings
    target_fps: int = 30
    max_processing_time: float = 0.1  # 100ms max processing time
    buffer_size: int = 10
    smoothing_window: int = 5
    
    # Quality settings
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    model_complexity: int = 1  # 0=lite, 1=full, 2=heavy
    
    # Gesture specific settings
    gesture_timeout: float = 3.0  # Maximum time for gesture completion
    gesture_hold_time: float = 0.5  # Time to hold gesture for recognition
    trajectory_smoothing: bool = True
    temporal_consistency: bool = True
    
    # Integration settings
    enable_memory_storage: bool = True
    enable_learning: bool = True
    enable_personalization: bool = True
    enable_context_awareness: bool = True


@dataclass
class HandLandmarks:
    """Hand landmark data structure."""
    landmarks: List[Tuple[float, float, float]]  # x, y, z coordinates
    world_landmarks: List[Tuple[float, float, float]]  # world coordinates
    handedness: str  # "Left" or "Right"
    confidence: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class BodyLandmarks:
    """Body pose landmark data structure."""
    landmarks: List[Tuple[float, float, float]]
    visibility: List[float]
    confidence: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class FaceLandmarks:
    """Face landmark data structure."""
    landmarks: List[Tuple[float, float, float]]
    confidence: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class GestureResult:
    """Result of gesture recognition."""
    gesture_id: str
    gesture_type: GestureType
    gesture_name: str
    confidence: float
    
    # Temporal information
    start_time: datetime
    end_time: datetime
    duration: float
    
    # Spatial information
    bounding_box: Optional[Tuple[int, int, int, int]] = None
    center_point: Optional[Tuple[float, float]] = None
    hand_landmarks: List[HandLandmarks] = field(default_factory=list)
    body_landmarks: Optional[BodyLandmarks] = None
    face_landmarks: Optional[FaceLandmarks] = None
    
    # Movement information
    trajectory: List[Tuple[float, float]] = field(default_factory=list)
    velocity: Optional[float] = None
    acceleration: Optional[float] = None
    direction: Optional[float] = None  # angle in radians
    
    # Context information
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    interaction_id: Optional[str] = None
    
    # Quality metrics
    quality_score: float = 0.0
    stability_score: float = 0.0
    clarity_score: float = 0.0
    
    # Metadata
    processing_time: float = 0.0
    frame_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GestureRequest:
    """Request for gesture recognition."""
    request_id: str
    session_id: str
    user_id: Optional[str] = None
    
    # Image/video data
    image: Optional[np.ndarray] = None
    video_frames: Optional[List[np.ndarray]] = None
    video_path: Optional[str] = None
    
    # Configuration
    config: GestureConfiguration = field(default_factory=GestureConfiguration)
    target_gestures: Optional[Set[GestureType]] = None
    context: Dict[str, Any] = field(default_factory=dict)
    
    # Timing
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    timeout: float = 30.0
    priority: int = 1  # 0=low, 1=normal, 2=high
    
    # Callbacks
    progress_callback: Optional[Callable] = None
    completion_callback: Optional[Callable] = None


class GestureRecognitionError(Exception):
    """Custom exception for gesture recognition operations."""
    
    def __init__(self, message: str, gesture_id: Optional[str] = None, 
                 error_code: Optional[str] = None):
        super().__init__(message)
        self.gesture_id = gesture_id
        self.error_code = error_code
        self.timestamp = datetime.now(timezone.utc)


class MediaPipeProcessor:
    """MediaPipe-based gesture processing."""
    
    def __init__(self, config: GestureConfiguration):
        self.config = config
        self.logger = get_logger(__name__)
        
        # Initialize MediaPipe components
        self.mp_hands = mp.solutions.hands
        self.mp_pose = mp.solutions.pose
        self.mp_face = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Setup processors
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=config.max_hands,
            min_detection_confidence=config.min_detection_confidence,
            min_tracking_confidence=config.min_tracking_confidence,
            model_complexity=config.model_complexity
        )
        
        if config.enable_body_pose:
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=config.model_complexity,
                enable_segmentation=False,
                min_detection_confidence=config.min_detection_confidence,
                min_tracking_confidence=config.min_tracking_confidence
            )
        
        if config.enable_face_detection:
            self.face_mesh = self.mp_face.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=config.min_detection_confidence,
                min_tracking_confidence=config.min_tracking_confidence
            )
    
    def process_frame(self, image: np.ndarray) -> Dict[str, Any]:
        """Process a single frame to extract landmarks."""
        try:
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = {}
            
            # Process hands
            hand_results = self.hands.process(rgb_image)
            if hand_results.multi_hand_landmarks:
                hand_landmarks = []
                for idx, landmarks in enumerate(hand_results.multi_hand_landmarks):
                    handedness = hand_results.multi_handedness[idx].classification[0].label
                    confidence = hand_results.multi_handedness[idx].classification[0].score
                    
                    landmark_points = [(lm.x, lm.y, lm.z) for lm in landmarks.landmark]
                    world_landmarks = []
                    if hand_results.multi_hand_world_landmarks:
                        world_landmarks = [(lm.x, lm.y, lm.z) 
                                         for lm in hand_results.multi_hand_world_landmarks[idx].landmark]
                    
                    hand_landmarks.append(HandLandmarks(
                        landmarks=landmark_points,
                        world_landmarks=world_landmarks,
                        handedness=handedness,
                        confidence=confidence
                    ))
                
                results['hands'] = hand_landmarks
            
            # Process body pose
            if self.config.enable_body_pose and hasattr(self, 'pose'):
                pose_results = self.pose.process(rgb_image)
                if pose_results.pose_landmarks:
                    landmark_points = [(lm.x, lm.y, lm.z) for lm in pose_results.pose_landmarks.landmark]
                    visibility = [lm.visibility for lm in pose_results.pose_landmarks.landmark]
                    
                    results['body'] = BodyLandmarks(
                        landmarks=landmark_points,
                        visibility=visibility,
                        confidence=np.mean(visibility)
                    )
            
            # Process face
            if self.config.enable_face_detection and hasattr(self, 'face_mesh'):
                face_results = self.face_mesh.process(rgb_image)
                if face_results.multi_face_landmarks:
                    for face_landmarks in face_results.multi_face_landmarks:
                        landmark_points = [(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark]
                        
                        results['face'] = FaceLandmarks(
                            landmarks=landmark_points,
                            confidence=0.8  # MediaPipe doesn't provide face confidence
                        )
                        break  # Only process first face
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error processing frame: {str(e)}")
            return {}
    
    def cleanup(self):
        """Cleanup MediaPipe resources."""
        if hasattr(self, 'hands'):
            self.hands.close()
        if hasattr(self, 'pose'):
            self.pose.close()
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()


class GestureClassifier:
    """Neural network-based gesture classifier."""
    
    def __init__(self, config: GestureConfiguration):
        self.config = config
        self.logger = get_logger(__name__)
        self.model = None
        self.gesture_classes = []
        self._setup_model()
    
    def _setup_model(self):
        """Setup the gesture classification model."""
        try:
            # Define a simple neural network for gesture classification
            self.model = nn.Sequential(
                nn.Linear(63, 128),  # 21 hand landmarks * 3 coordinates
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, len(GestureType))
            )
            
            # Initialize gesture classes
            self.gesture_classes = list(GestureType)
            
        except Exception as e:
            self.logger.error(f"Failed to setup gesture classifier: {str(e)}")
    
    def classify_gesture(self, landmarks: List[HandLandmarks]) -> Tuple[GestureType, float]:
        """Classify gesture from hand landmarks."""
        try:
            if not landmarks or not self.model:
                return GestureType.CUSTOM, 0.0
            
            # Use first hand for classification
            hand = landmarks[0]
            
            # Flatten landmarks
            features = np.array([coord for point in hand.landmarks for coord in point])
            
            # Normalize features (center around wrist)
            if len(features) >= 3:
                wrist = features[:3]
                features = features.reshape(-1, 3) - wrist
                features = features.flatten()
            
            # Pad or truncate to expected size
            if len(features) < 63:
                features = np.pad(features, (0, 63 - len(features)))
            elif len(features) > 63:
                features = features[:63]
            
            # Convert to tensor and classify
            with torch.no_grad():
                input_tensor = torch.FloatTensor(features).unsqueeze(0)
                outputs = self.model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                
                confidence, predicted = torch.max(probabilities, 1)
                gesture_type = self.gesture_classes[predicted.item()]
                
                return gesture_type, confidence.item()
        
        except Exception as e:
            self.logger.error(f"Error classifying gesture: {str(e)}")
            return GestureType.CUSTOM, 0.0
    
    def classify_static_gestures(self, landmarks: List[HandLandmarks]) -> List[Tuple[GestureType, float]]:
        """Classify static hand gestures using rule-based approach."""
        gestures = []
        
        for hand in landmarks:
            gesture_type, confidence = self._classify_hand_shape(hand)
            if confidence >= self.config.min_confidence_threshold:
                gestures.append((gesture_type, confidence))
        
        return gestures
    
    def _classify_hand_shape(self, hand: HandLandmarks) -> Tuple[GestureType, float]:
        """Classify hand shape using geometric rules."""
        try:
            landmarks = np.array(hand.landmarks)
            
            # Extract key points
            thumb_tip = landmarks[4]
            thumb_ip = landmarks[3]
            index_tip = landmarks[8]
            index_pip = landmarks[6]
            middle_tip = landmarks[12]
            middle_pip = landmarks[10]
            ring_tip = landmarks[16]
            ring_pip = landmarks[14]
            pinky_tip = landmarks[20]
            pinky_pip = landmarks[18]
            
            # Check for thumbs up
            if (thumb_tip[1] < thumb_ip[1] and  # thumb pointing up
                index_tip[1] > index_pip[1] and  # index folded
                middle_tip[1] > middle_pip[1] and  # middle folded
                ring_tip[1] > ring_pip[1] and  # ring folded
                pinky_tip[1] > pinky_pip[1]):  # pinky folded
                return GestureType.THUMBS_UP, 0.9
            
            # Check for thumbs down
            if (thumb_tip[1] > thumb_ip[1] and  # thumb pointing down
                index_tip[1] > index_pip[1] and  # index folded
                middle_tip[1] > middle_pip[1] and  # middle folded
                ring_tip[1] > ring_pip[1] and  # ring folded
                pinky_tip[1] > pinky_pip[1]):  # pinky folded
                return GestureType.THUMBS_DOWN, 0.9
            
            # Check for peace sign
            if (index_tip[1] < index_pip[1] and  # index extended
                middle_tip[1] < middle_pip[1] and  # middle extended
                ring_tip[1] > ring_pip[1] and  # ring folded
                pinky_tip[1] > pinky_pip[1]):  # pinky folded
                return GestureType.PEACE_SIGN, 0.85
            
            # Check for OK sign
            thumb_index_distance = euclidean(thumb_tip[:2], index_tip[:2])
            if (thumb_index_distance < 0.05 and  # thumb and index close
                middle_tip[1] < middle_pip[1] and  # middle extended
                ring_tip[1] < ring_pip[1] and  # ring extended
                pinky_tip[1] < pinky_pip[1]):  # pinky extended
                return GestureType.OK_SIGN, 0.8
            
            # Check for stop sign (open palm)
            fingers_extended = sum([
                index_tip[1] < index_pip[1],
                middle_tip[1] < middle_pip[1],
                ring_tip[1] < ring_pip[1],
                pinky_tip[1] < pinky_pip[1]
            ])
            
            if fingers_extended >= 4:
                return GestureType.STOP_SIGN, 0.7
            
            return GestureType.CUSTOM, 0.1
            
        except Exception as e:
            self.logger.error(f"Error in hand shape classification: {str(e)}")
            return GestureType.CUSTOM, 0.0


class TemporalGestureAnalyzer:
    """Analyzes temporal patterns in gestures."""
    
    def __init__(self, config: GestureConfiguration):
        self.config = config
        self.logger = get_logger(__name__)
        self.gesture_buffer = deque(maxlen=config.buffer_size)
        self.trajectory_buffer = deque(maxlen=config.buffer_size)
    
    def add_frame_data(self, landmarks: Dict[str, Any], timestamp: datetime):
        """Add frame data for temporal analysis."""
        self.gesture_buffer.append({
            'landmarks': landmarks,
            'timestamp': timestamp
        })
        
        # Extract hand center for trajectory
        if 'hands' in landmarks and landmarks['hands']:
            hand = landmarks['hands'][0]  # Use first hand
            center = self._calculate_hand_center(hand)
            self.trajectory_buffer.append({
                'center': center,
                'timestamp': timestamp
            })
    
    def _calculate_hand_center(self, hand: HandLandmarks) -> Tuple[float, float]:
        """Calculate the center point of a hand."""
        landmarks = np.array(hand.landmarks)
        center_x = np.mean(landmarks[:, 0])
        center_y = np.mean(landmarks[:, 1])
        return (center_x, center_y)
    
    def detect_wave_gesture(self) -> Tuple[bool, float]:
        """Detect waving gesture from trajectory."""
        if len(self.trajectory_buffer) < 5:
            return False, 0.0
        
        try:
            # Extract x-coordinates and timestamps
            points = list(self.trajectory_buffer)
            x_coords = [p['center'][0] for p in points]
            timestamps = [p['timestamp'] for p in points]
            
            # Calculate time differences
            time_diffs = [(timestamps[i] - timestamps[i-1]).total_seconds() 
                         for i in range(1, len(timestamps))]
            
            if max(time_diffs) > self.config.gesture_timeout:
                return False, 0.0
            
            # Smooth the trajectory
            if self.config.trajectory_smoothing and len(x_coords) >= 5:
                x_coords = savgol_filter(x_coords, 5, 3)
            
            # Detect oscillations (wave pattern)
            peaks = []
            valleys = []
            
            for i in range(1, len(x_coords) - 1):
                if x_coords[i] > x_coords[i-1] and x_coords[i] > x_coords[i+1]:
                    peaks.append(i)
                elif x_coords[i] < x_coords[i-1] and x_coords[i] < x_coords[i+1]:
                    valleys.append(i)
            
            # Check for alternating peaks and valleys (wave pattern)
            oscillations = len(peaks) + len(valleys)
            if oscillations >= 4:  # At least 2 complete waves
                return True, min(0.9, oscillations * 0.15)
            
            return False, 0.0
            
        except Exception as e:
            self.logger.error(f"Error detecting wave gesture: {str(e)}")
            return False, 0.0
    
    def detect_swipe_gesture(self) -> Tuple[Optional[str], float]:
        """Detect swipe gestures."""
        if len(self.trajectory_buffer) < 3:
            return None, 0.0
        
        try:
            points = list(self.trajectory_buffer)
            start_point = points[0]['center']
            end_point = points[-1]['center']
            
            # Calculate displacement
            dx = end_point[0] - start_point[0]
            dy = end_point[1] - start_point[1]
            distance = np.sqrt(dx**2 + dy**2)
            
            # Check if displacement is significant
            if distance < 0.1:  # Minimum swipe distance
                return None, 0.0
            
            # Calculate direction
            angle = np.arctan2(dy, dx)
            
            # Classify swipe direction
            if -np.pi/4 <= angle <= np.pi/4:
                direction = "right"
            elif np.pi/4 < angle <= 3*np.pi/4:
                direction = "down"
            elif -3*np.pi/4 <= angle < -np.pi/4:
                direction = "up"
            else:
                direction = "left"
            
            confidence = min(0.9, distance * 2)  # Scale confidence with distance
            return direction, confidence
            
        except Exception as e:
            self.logger.error(f"Error detecting swipe gesture: {str(e)}")
            return None, 0.0
    
    def clear_buffers(self):
        """Clear temporal buffers."""
        self.gesture_buffer.clear()
        self.trajectory_buffer.clear()


class EnhancedGestureRecognizer:
    """
    Advanced Gesture Recognition System for the AI Assistant.
    
    Features:
    - Multi-modal gesture detection (hand, body, face)
    - Real-time and batch processing modes
    - Temporal gesture analysis and tracking
    - Integration with core system components
    - Adaptive learning and personalization
    - Context-aware gesture interpretation
    - Performance monitoring and optimization
    - Memory integration for gesture history
    - Event-driven architecture integration
    """
    
    def __init__(self, container: Container):
        """
        Initialize the enhanced gesture recognizer.
        
        Args:
            container: Dependency injection container
        """
        self.container = container
        self.logger = get_logger(__name__)
        self.config = container.get(ConfigLoader)
        self.event_bus = container.get(EventBus)
        self.error_handler = container.get(ErrorHandler)
        self.health_check = container.get(HealthCheck)
        
        # Core component integration
        self.component_manager = container.get(EnhancedComponentManager)
        self.workflow_orchestrator = container.get(WorkflowOrchestrator)
        self.session_manager = container.get(EnhancedSessionManager)
        self.interaction_handler = container.get(InteractionHandler)
        
        # Vision and processing
        try:
            self.vision_processor = container.get(VisionProcessor)
            self.image_analyzer = container.get(ImageAnalyzer)
        except Exception:
            self.logger.warning("Vision components not available")
            self.vision_processor = None
            self.image_analyzer = None
        
        # Memory and learning
        try:
            self.memory_manager = container.get(MemoryManager)
            self.feedback_processor = container.get(FeedbackProcessor)
            self.continual_learner = container.get(ContinualLearner)
        except Exception:
            self.logger.warning("Memory/learning components not available")
            self.memory_manager = None
            self.feedback_processor = None
            self.continual_learner = None
        
        # Security
        try:
            self.security_sanitizer = container.get(SecuritySanitizer)
        except Exception:
            self.logger.warning("Security sanitizer not available")
            self.security_sanitizer = None
        
        # Configuration
        self._load_gesture_config()
        
        # Processing components
        self.mediapipe_processor = MediaPipeProcessor(self.gesture_config)
        self.gesture_classifier = GestureClassifier(self.gesture_config)
        self.temporal_analyzer = TemporalGestureAnalyzer(self.gesture_config)
        
        # State management
        self._active_recognitions: Dict[str, Dict[str, Any]] = {}
        self._gesture_cache: Dict[str, GestureResult] = {}
        self._processing_lock = asyncio.Lock()
        self._shutdown_event = asyncio.Event()
        
        # Performance monitoring
        self._setup_monitoring()
        
        # Threading
        self.thread_pool = ThreadPoolExecutor(
            max_workers=4,
            thread_name_prefix="gesture_recognizer"
        )
        
        # Background tasks
        self._background_tasks: List[asyncio.Task] = []
        
        self.logger.info("EnhancedGestureRecognizer initialized")

    def _load_gesture_config(self) -> None:
        """Load gesture recognition configuration."""
        gesture_config = self.config.get("gesture_recognition", {})
        
        self.gesture_config = GestureConfiguration(
            recognition_mode=RecognitionMode(gesture_config.get("mode", "real_time")),
            min_confidence_threshold=gesture_config.get("min_confidence", 0.7),
            max_hands=gesture_config.get("max_hands", 2),
            enable_body_pose=gesture_config.get("enable_body_pose", True),
            enable_face_detection=gesture_config.get("enable_face_detection", True),
            target_fps=gesture_config.get("target_fps", 30),
            max_processing_time=gesture_config.get("max_processing_time", 0.1),
            buffer_size=gesture_config.get("buffer_size", 10),
            smoothing_window=gesture_config.get("smoothing_window", 5),
            min_detection_confidence=gesture_config.get("min_detection_confidence", 0.5),
            min_tracking_confidence=gesture_config.get("min_tracking_confidence", 0.5),
            model_complexity=gesture_config.get("model_complexity", 1),
            gesture_timeout=gesture_config.get("gesture_timeout", 3.0),
            gesture_hold_time=gesture_config.get("gesture_hold_time", 0.5),
            trajectory_smoothing=gesture_config.get("trajectory_smoothing", True),
            temporal_consistency=gesture_config.get("temporal_consistency", True),
            enable_memory_storage=gesture_config.get("enable_memory_storage", True),
            enable_learning=gesture_config.get("enable_learning", True),
            enable_personalization=gesture_config.get("enable_personalization", True),
            enable_context_awareness=gesture_config.get("enable_context_awareness", True)
        )

    def _setup_monitoring(self) -> None:
        """Setup monitoring and metrics collection."""
        try:
            self.metrics = self.container.get(MetricsCollector)
            self.tracer = self.container.get(TraceManager)
            
            # Register gesture recognition metrics
            self.metrics.register_counter("gesture_recognitions_total")
            self.metrics.register_counter("gesture_recognition_failures_total")
            self.metrics.register_histogram("gesture_recognition_duration_seconds")
            self.metrics.register_histogram("gesture_confidence_score")
            self.metrics.register_gauge("active_gesture_recognitions")
            
        except Exception as e:
            self.logger.warning(f"Failed to setup monitoring: {str(e)}")
            self.metrics = None
            self.tracer = None

    async def initialize(self) -> None:
        """Initialize the gesture recognizer."""
        try:
            self.logger.info("Initializing gesture recognizer...")
            
            # Register event handlers
            await self._register_event_handlers()
            
            # Register health check
            self.health_check.register_component("gesture_recognizer", self._health_check_callback)
            
            # Load gesture models
            await self._load_gesture_models()
            
            # Start background tasks
            await self._start_background_tasks()
            
            self.logger.info("Gesture recognizer initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize gesture recognizer: {str(e)}")
            raise GestureRecognitionError(f"Initialization failed: {str(e)}")

    async def _register_event_handlers(self) -> None:
        """Register event handlers for system events."""
        # User interaction events
        self.event_bus.subscribe("user_interaction_started", self._handle_interaction_started)
        self.event_bus.subscribe("user_interaction_completed", self._handle_interaction_completed)
        
        # Session events
        self.event_bus.subscribe("session_started", self._handle_session_started)
        self.event_bus.subscribe("session_ended", self._handle_session_ended)
        
        # System events
        self.event_bus.subscribe("system_shutdown_started", self._handle_system_shutdown)
        self.event_bus.subscribe("component_health_changed", self._handle_component_health_change)

    async def _load_gesture_models(self) -> None:
        """Load pre-trained gesture recognition models."""
        try:
            # Load custom gesture models if available
            model_path = Path(self.config.get("gesture_recognition.model_path", "data/models/gestures/"))
            if model_path.exists():
                self.logger.info(f"Loading gesture models from {model_path}")
                # Implementation would load actual model files
        
        except Exception as e:
            self.logger.warning(f"Failed to load gesture models: {str(e)}")

    async def _start_background_tasks(self) -> None:
        """Start background maintenance tasks."""
        # Cache cleanup task
        self._background_tasks.append(
            asyncio.create_task(self._cache_cleanup_loop())
        )
        
        # Performance monitoring task
        self._background_tasks.append(
            asyncio.create_task(self._performance_monitoring_loop())
        )
        
        # Model adaptation task
        if self.gesture_config.enable_learning:
            self._background_tasks.append(
                asyncio.create_task(self._model_adaptation_loop())
            )

    @handle_exceptions
    async def recognize_gesture(self, request: GestureRequest) -> GestureResult:
        """
        Recognize gestures from image or video data.
        
        Args:
            request: Gesture recognition request
            
        Returns:
            Gesture recognition result
        """
        start_time = time.time()
        
        async with self._processing_lock:
            try:
                with self.tracer.trace("gesture_recognition") if self.tracer else None:
                    self.logger.info(f"Starting gesture recognition for request {request.request_id}")
                    
                    # Emit processing started event
                    await self.event_bus.emit(ProcessingStarted(
                        session_id=request.session_id,
                        request_id=request.request_id,
                        input_modalities=["vision"]
                    ))
                    
                    # Track active recognition
                    self._active_recognitions[request.request_id] = {
                        'start_time': start_time,
                        'session_id': request.session_id,
                        'user_id': request.user_id
                    }
                    
                    # Process based on input type
                    if request.image is not None:
                        result = await self._recognize_from_image(request)
                    elif request.video_frames:
                        result = await self._recognize_from_video_frames(request)
                    elif request.video_path:
                        result = await self._recognize_from_video_file(request)
                    else:
                        raise GestureRecognitionError("No valid input data provided")
                    
                    # Store in memory if enabled
                    if self.gesture_config.enable_memory_storage and self.memory_manager:
                        await self._store_gesture_memory(result, request)
                    
                    # Learning updates
                    if self.gesture_config.enable_learning and self.continual_learner:
                        await self._update_learning_models(result, request)
                    
                    processing_time = time.time() - start_time
                    result.processing_time = processing_time
                    
                    # Update metrics
                    if self.metrics:
                        self.metrics.increment("gesture_recognitions_total")
                        self.metrics.record("gesture_recognition_duration_seconds", processing_time)
                        self.metrics.record("gesture_confidence_score", result.confidence)
                    
                    # Emit completion event
                    await self.event_bus.emit(ProcessingCompleted(
                        session_id=request.session_id,
                        request_id=request.request_id,
                        processing_time=processing_time,
                        success=True,
                        confidence=result.confidence
                    ))
                    
                    self.logger.info(
                        f"Gesture recognition completed for {request.request_id} "
                        f"in {processing_time:.3f}s with confidence {result.confidence:.3f}"
                    )
                    
                    return result
                    
            except Exception as e:
                processing_time = time.time() - start_time
                
                # Update metrics
                if self.metrics:
                    self.metrics.increment("gesture_recognition_failures_total")
                
                # Emit error event
                await self.event_bus.emit(ProcessingError(
                    session_id=request.session_id,
                    request_id=request.request_id,
                    error_type=type(e).__name__,
                    error_message=str(e)
                ))
                
                self.logger.error(f"Gesture recognition failed for {request.request_id}: {str(e)}")
                raise GestureRecognitionError(f"Recognition failed: {str(e)}")
            
            finally:
                # Clean up active recognition
                self._active_recognitions.pop(request.request_id, None)
                
                # Update active recognition count
                if self.metrics:
                    self.metrics.set("active_gesture_recognitions", len(self._active_recognitions))

    async def _recognize_from_image(self, request: GestureRequest) -> GestureResult:
        """Recognize gestures from a single image."""
        try:
            # Process image with MediaPipe
            landmarks_data = await asyncio.get_event_loop().run_in_executor(
                self.thread_pool,
                self.mediapipe_processor.process_frame,
                request.image
            )
            
            # Create gesture result
            result = GestureResult(
                gesture_id=str(uuid.uuid4()),
                gesture_type=GestureType.CUSTOM,
                gesture_name="unknown",
                confidence=0.0,
                start_time=datetime.now(timezone.utc),
                end_time=datetime.now(timezone.utc),
                duration=0.0,
                session_id=request.session_id,
                user_id=request.user_id,
                frame_count=1
            )
            
            # Analyze gestures if hands detected
            if 'hands' in landmarks_data and landmarks_data['hands']:
                result.hand_landmarks = landmarks_data['hands']
                
                # Classify static gestures
                static_gestures = self.gesture_classifier.classify_static_gestures(
                    landmarks_data['hands']
                )
                
                if static_gestures:
                    best_gesture, confidence = max(static_gestures, key=lambda x: x[1])
                    result.gesture_type = best_gesture
                    result.gesture_name = best_gesture.value
                    result.confidence = confidence
                
                # Calculate bounding box and center
                result.bounding_box = self._calculate_bounding_box(landmarks_data['hands'])
                result.center_point = self._calculate_center_point(landmarks_data['hands'])
            
            # Add body landmarks if available
            if 'body' in landmarks_data:
                result.body_landmarks = landmarks_data['body']
            
            # Add face landmarks if available
            if 'face' in landmarks_data:
                result.face_landmarks = landmarks_data['face']
            
            # Calculate quality scores
            result.quality_score = self._calculate_quality_score(landmarks_data)
            result.stability_score = 1.0  # Single frame is perfectly stable
            result.clarity_score = self._calculate_clarity_score(request.image)
            
            return result
            
        except Exception as e:
            raise GestureRecognitionError(f"Image recognition failed: {str(e)}")

    async def _recognize_from_video_frames(self, request: GestureRequest) -> GestureResult:
        """Recognize gestures from video frames."""
        try:
            # Clear temporal buffers
            self.temporal_analyzer.clear_buffers()
            
            all_landmarks = []
            timestamps = []
            
            # Process each frame
            for i, frame in enumerate(request.video_frames):
                frame_time = datetime.now(timezone.utc)
                
                # Process frame
                landmarks_data = await asyncio.get_event_loop().run_in_executor(
                    self.thread_pool,
                    self.mediapipe_processor.process_frame,
                    frame
                )
                
                all_landmarks.append(landmarks_data)
                timestamps.append(frame_time)
                
                # Add to temporal analyzer
                self.temporal_analyzer.add_frame_data(landmarks_data, frame_time)
                
                # Progress callback
                if request.progress_callback:
                    progress = (i + 1) / len(request.video_frames)
                    await request.progress_callback(progress)
            
            # Create gesture result
            result = GestureResult(
                gesture_id=str(uuid.uuid4()),
                gesture_type=GestureType.CUSTOM,
                gesture_name="unknown",
                confidence=0.0,
                start_time=timestamps[0] if timestamps else datetime.now(timezone.utc),
                end_time=timestamps[-1] if timestamps else datetime.now(timezone.utc),
                duration=(timestamps[-1] - timestamps[0]).total_seconds() if len(timestamps) > 1 else 0.0,
                session_id=request.session_id,
                user_id=request.user_id,
                frame_count=len(request.video_frames)
            )
            
            # Analyze temporal gestures
            if all_landmarks:
                # Check for wave gesture
                is_wave, wave_confidence = self.temporal_analyzer.detect_wave_gesture()
                if is_wave and wave_confidence >= self.gesture_config.min_confidence_threshold:
                    result.gesture_type = GestureType.WAVE
                    result.gesture_name = "wave"
                    result.confidence = wave_confidence
                
                # Check for swipe gesture
                swipe_direction, swipe_confidence = self.temporal_analyzer.detect_swipe_gesture()
                if (swipe_direction and 
                    swipe_confidence >= self.gesture_config.min_confidence_threshold and
                    swipe_confidence > result.confidence):
                    result.gesture_type = GestureType.SWIPE
                    result.gesture_name = f"swipe_{swipe_direction}"
                    result.confidence = swipe_confidence
                
                # Extract trajectory
                result.trajectory = self._extract_trajectory(all_landmarks)
                
                # Calculate movement metrics
                if result.trajectory:
                    result.velocity = self._calculate_velocity(result.trajectory, timestamps)
                    result.acceleration = self._calculate_acceleration(result.trajectory, timestamps)
                    result.direction = self._calculate_direction(result.trajectory)
                
                # Use best frame for landmarks
                best_frame_landmarks = max(all_landmarks, 
                                         key=lambda x: len(x.get('hands', [])))
                
                if 'hands' in best_frame_landmarks:
                    result.hand_landmarks = best_frame_landmarks['hands']
                    result.bounding_box = self._calculate_bounding_box(best_frame_landmarks['hands'])
                    result.center_point = self._calculate_center_point(best_frame_landmarks['hands'])
                
                if 'body' in best_frame_landmarks:
                    result.body_landmarks = best_frame_landmarks['body']
                
                if 'face' in best_frame_landmarks:
                    result.face_landmarks = best_frame_landmarks['face']
            
            # Calculate quality scores
            result.quality_score = self._calculate_temporal_quality(all_landmarks)
            result.stability_score = self._calculate_stability_score(all_landmarks)
            result.clarity_score = np.mean([
                self._calculate_clarity_score(frame) for frame in request.video_frames
            ])
            
            return result
            
        except Exception as e:
            raise GestureRecognitionError(f"Video frame recognition failed: {str(e)}")

    async def _recognize_from_video_file(self, request: GestureRequest) -> GestureResult:
        """Recognize gestures from video file."""
        try:
            # Load video frames
            cap = cv2.VideoCapture(request.video_path)
            frames = []
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            
            cap.release()
            
            if not frames:
                raise GestureRecognitionError("No frames found in video file")
            
            # Create modified request with frames
            frame_request = GestureRequest(
                request_id=request.request_id,
                session_id=request.session_id,
                user_id=request.user_id,
                video_frames=frames,
                config=request.config,
                target_gestures=request.target_gestures,
                context=request.context,
                timeout=request.timeout,
                priority=request.priority,
                progress_callback=request.progress_callback,
                completion_callback=request.completion_callback
            )
            
            return await self._recognize_from_video_frames(frame_request)
            
        except Exception as e:
            raise GestureRecognitionError(f"Video file recognition failed: {str(e)}")

    def _calculate_bounding_box(self, hands: List[HandLandmarks]) -> Tuple[int, int, int, int]:
        """Calculate bounding box for detected hands."""
        if not hands:
            return (0, 0, 0, 0)
        
        all_points = []
        for hand in hands:
            all_points.extend([(lm[0], lm[1]) for lm in hand.landmarks])
        
        if not all_points:
            return (0, 0, 0, 0)
        
        x_coords = [p[0] for p in all_points]
        y_coords = [p[1] for p in all_points]
        
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        
        # Convert to pixel coordinates (assuming normalized coordinates)
        return (int(min_x * 640), int(min_y * 480), 
                int((max_x - min_x) * 640), int((max_y - min_y) * 480))

    def _calculate_center_point(self, hands: List[HandLandmarks]) -> Tuple[float, float]:
        """Calculate center point of detected hands."""
        if not hands:
            return (0.0, 0.0)
        
        all_points = []
        for hand in hands:
            all_points.extend([(lm[0], lm[1]) for lm in hand.landmarks])
        
        if not all_points:
            return (0.0, 0.0)
        
        center_x = np.mean([p[0] for p in all_points])
        center_y = np.mean([p[1] for p in all_points])
        
        return (float(center_x), float(center_y))

    def _calculate_quality_score(self, landmarks_data: Dict[str, Any]) -> float:
        """Calculate quality score based on landmark detection."""
        score = 0.0
        components = 0
        
        if 'hands' in landmarks_data and landmarks_data['hands']:
            hand_score = np.mean([hand.confidence for hand in landmarks_data['hands']])
            score += hand_score
            components += 1
        
        if 'body' in landmarks_data and landmarks_data['body']:
            score += landmarks_data['body'].confidence
            components += 1
        
        if 'face' in landmarks_data and landmarks_data['face']:
            score += landmarks_data['face'].confidence
            components += 1
        
        return score / components if components > 0 else 0.0

    def _calculate_clarity_score(self, image: np.ndarray) -> float:
        """Calculate image clarity score using Laplacian variance."""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Normalize to 0-1 range
            return min(1.0, laplacian_var / 1000.0)
        except Exception:
            return 0.5

    def _calculate_temporal_quality(self, all_landmarks: List[Dict[str, Any]]) -> float:
        """Calculate quality score across temporal sequence."""
        if not all_landmarks:
            return 0.0
        
        quality_scores = [self._calculate_quality_score(landmarks) for landmarks in all_landmarks]
        return np.mean(quality_scores)

    def _calculate_stability_score(self, all_landmarks: List[Dict[str, Any]]) -> float:
        """Calculate stability score based on landmark consistency."""
        if len(all_landmarks) < 2:
            return 1.0
        
        try:
            # Track hand center stability
            centers = []
            for landmarks in all_landmarks:
                if 'hands' in landmarks and landmarks['hands']:
                    center = self.temporal_analyzer._calculate_hand_center(landmarks['hands'][0])
                    centers.append(center)
            
            if len(centers) < 2:
                return 0.5
            
            # Calculate center point variance
            x_coords = [c[0] for c in centers]
            y_coords = [c[1] for c in centers]
            
            x_variance = np.var(x_coords)
            y_variance = np.var(y_coords)
            
            # Convert variance to stability score (lower variance = higher stability)
            stability = 1.0 / (1.0 + (x_variance + y_variance) * 10)
            return float(stability)
        
        except Exception:
            return 0.5

    def _extract_trajectory(self, all_landmarks: List[Dict[str, Any]]) -> List[Tuple[float, float]]:
        """Extract hand trajectory from landmarks."""
        trajectory = []
        
        for landmarks in all_landmarks:
            if 'hands' in landmarks and landmarks['hands']:
                center = self.temporal_analyzer._calculate_hand_center(landmarks['hands'][0])
                trajectory.append(center)
        
        return trajectory

    def _calculate_velocity(self, trajectory: List[Tuple[float, float]], 
                          timestamps: List[datetime]) -> Optional[float]:
        """Calculate average velocity from trajectory."""
        if len(trajectory) < 2 or len(timestamps) < 2:
            return None
        
        try:
            distances = []
            times = []
            
            for i in range(1, len(trajectory)):
                # Calculate distance
                dist = euclidean(trajectory[i], trajectory[i-1])
                distances.append(dist)
                
                # Calculate time difference
                time_diff = (timestamps[i] - timestamps[i-1]).total_seconds()
                times.append(time_diff)
            
            # Calculate velocities
            velocities = [d/t if t > 0 else 0 for d, t in zip(distances, times)]
            return float(np.mean(velocities))
        
        except Exception:
            return None

    def _calculate_acceleration(self, trajectory: List[Tuple[float, float]], 
                              timestamps: List[datetime]) -> Optional[float]:
        """Calculate average acceleration from trajectory."""
        if len(trajectory) < 3 or len(timestamps) < 3:
            return None
        
        try:
            velocities = []
            times = []
            
            for i in range(1, len(trajectory)):
                # Calculate velocity
                dist = euclidean(trajectory[i], trajectory[i-1])
                time_diff = (timestamps[i] - timestamps[i-1]).total_seconds()
                velocity = dist / time_diff if time_diff > 0 else 0
                velocities.append(velocity)
                
                if i > 1:
                    times.append(time_diff)
            
            # Calculate accelerations
            accelerations = []
            for i in range(1, len(velocities)):
                acc = (velocities[i] - velocities[i-1]) / times[i-1] if times[i-1] > 0 else 0
                accelerations.append(acc)
            
            return float(np.mean(accelerations))
        
        except Exception:
            return None

    def _calculate_direction(self, trajectory: List[Tuple[float, float]]) -> Optional[float]:
        """Calculate overall direction from trajectory."""
        if len(trajectory) < 2:
            return None
        
        try:
            start_point = trajectory[0]
            end_point = trajectory[-1]
            
            dx = end_point[0] - start_point[0]
            dy = end_point[1] - start_point[1]
            
            return float(np.arctan2(dy, dx))
        
        except Exception:
            return None

    async def _store_gesture_memory(self, result: GestureResult, request: GestureRequest) -> None:
        """Store gesture result in memory system."""
        try:
            if not self.memory_manager:
                return
            
            memory_data = {
                'gesture_id': result.gesture_id,
                'gesture_type': result.gesture_type.value,
                'gesture_name': result.gesture_name,
                'confidence': result.confidence,
                'session_id': request.session_id,
                'user_id': request.user_id,
                'timestamp': result.start_time,
                'duration': result.duration,
                'context': request.context,
                'quality_metrics': {
                    'quality_score': result.quality_score,
                    'stability_score': result.stability_score,
                    'clarity_score': result.clarity_score
                }
            }
            
            await self.memory_manager.store_interaction_memory(
                interaction_type="gesture_recognition",
                data=memory_data,
                session_id=request.session_id
            )
        
        except Exception as e:
            self.logger.warning(f"Failed to store gesture memory: {str(e)}")

    async def _update_learning_models(self, result: GestureResult, request: GestureRequest) -> None:
        """Update learning models with gesture recognition result."""
        try:
            if not self.continual_learner:
                return
            
            learning_data = {
                'input_type': 'gesture',
                'result': result,
                'request': request,
                'feedback_signal': result.confidence
            }
            
            await self.continual_learner.learn_from_interaction(
                input_data=learning_data,
                result=result,
                context={'session_id': request.session_id, 'user_id': request.user_id}
            )
        
        except Exception as e:
            self.logger.warning(f"Failed to update learning models: {str(e)}")

    async def get_recognition_status(self, request_id: str) -> Dict[str, Any]:
        """Get status of ongoing gesture recognition."""
        if request_id in self._active_recognitions:
            recognition = self._active_recognitions[request_id]
            return {
                'request_id': request_id,
                'status': 'processing',
                'start_time': recognition['start_time'],
                'elapsed_time': time.time() - recognition['start_time'],
                'session_id': recognition['session_id'],
                'user_id': recognition['user_id']
            }
        
        return {
            'request_id': request_id,
            'status': 'not_found'
        }

    async def cancel_recognition(self, request_id: str) -> bool:
        """Cancel ongoing gesture recognition."""
        if request_id in self._active_recognitions:
            # In a more complete implementation, this would signal cancellation
            # to the processing thread
            del self._active_recognitions[request_id]
            return True
        
        return False

    def get_supported_gestures(self) -> List[Dict[str, Any]]:
        """Get list of supported gesture types."""
        return [
            {
                'type': gesture_type.value,
                'name': gesture_type.name,
                'description': f"{gesture_type.name.replace('_', ' ').title()} gesture"
            }
            for gesture_type in GestureType
        ]

    async def _cache_cleanup_loop(self) -> None:
        """Background task for cache cleanup."""
        while not self._shutdown_event.is_set():
            try:
                # Clean up old cache entries
                current_time = time.time()
                cache_ttl = 3600  # 1 hour
                
                expired_keys = [
                    key for key, value in self._gesture_cache.items()
                    if (current_time - value.start_time.timestamp()) > cache_ttl
                ]
                
                for key in expired_keys:
                    del self._gesture_cache[key]
                
                self.logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
                
                await asyncio.sleep(300)  # Clean every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Cache cleanup error: {str(e)}")
                await asyncio.sleep(300)

    async def _performance_monitoring_loop(self) -> None:
        """Background task for performance monitoring."""
        while not self._shutdown_event.is_set():
            try:
                # Update active recognition count
                if self.metrics:
                    self.metrics.set("active_gesture_recognitions", len(self._active_recognitions))
                
                # Monitor long-running recognitions
                current_time = time.time()
                timeout_threshold = 30.0  # 30 seconds
                
                for request_id, recognition in list(self._active_recognitions.items()):
                    if (current_time - recognition['start_time']) > timeout_threshold:
                        self.logger.warning(f"Long-running gesture recognition: {request_id}")
                
                await asyncio.sleep(30)  # Monitor every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Performance monitoring error: {str(e)}")
                await asyncio.sleep(30)

    async def _model_adaptation_loop(self) -> None:
        """Background task for model adaptation and learning."""
        while not self._shutdown_event.is_set():
            try:
                # Implement model adaptation logic
                # This could include retraining based on user feedback,
                # updating gesture thresholds, etc.
                
                await asyncio.sleep(3600)  # Adapt every hour
                
            except Exception as e:
                self.logger.error(f"Model adaptation error: {str(e)}")
                await asyncio.sleep(3600)

    async def _handle_interaction_started(self, event) -> None:
        """Handle user interaction started events."""
        # Prepare for potential gesture input
        self.logger.debug(f"Interaction started: {event.interaction_id}")

    async def _handle_interaction_completed(self, event) -> None:
        """Handle user interaction completed events."""
        # Clean up any interaction-specific data
        self.logger.debug(f"Interaction completed: {event.interaction_id}")

    async def _handle_session_started(self, event) -> None:
        """Handle session start events."""
        # Initialize session-specific gesture settings
        self.logger.debug(f"Session started: {event.session_id}")

    async def _handle_session_ended(self, event) -> None:
        """Handle session end events."""
        # Clean up session-specific data
        session_recognitions = [
            req_id for req_id, recognition in self._active_recognitions.items()
            if recognition.get('session_id') == event.session_id
        ]
        
        for req_id in session_recognitions:
            await self.cancel_recognition(req_id)

    async def _handle_system_shutdown(self, event) -> None:
        """Handle system shutdown events."""
        self._shutdown_event.set()
        await self.cleanup()

    async def _handle_component_health_change(self, event) -> None:
        """Handle component health change events."""
        if event.component in ['vision_processor', 'image_analyzer'] and not event.healthy:
            self.logger.warning(f"Vision component {event.component} unhealthy")

    async def _health_check_callback(self) -> Dict[str, Any]:
        """Health check callback for the gesture recognizer."""
        try:
            return {
                "status": "healthy",
                "active_recognitions": len(self._active_recognitions),
                "cache_size": len(self._gesture_cache),
                "mediapipe_initialized": self.mediapipe_processor is not None,
                "classifier_initialized": self.gesture_classifier is not None,
                "temporal_analyzer_initialized": self.temporal_analyzer is not None,
                "background_tasks_running": len(self._background_tasks)
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }

    async def cleanup(self) -> None:
        """Cleanup gesture recognizer resources."""
        self.logger.info("Starting gesture recognizer cleanup...")
        
        try:
            # Cancel all active recognitions
            for recognition_id in list(self.active_recognitions.keys()):
                await self.stop_recognition(recognition_id)
            
            # Clear gesture history
            self.gesture_history.clear()
            
            # Release mediapipe resources
            if hasattr(self, 'hands'):
                self.hands.close()
            if hasattr(self, 'pose'):
                self.pose.close()
            if hasattr(self, 'face_mesh'):
                self.face_mesh.close()
                
            self.logger.info("Gesture recognizer cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")
