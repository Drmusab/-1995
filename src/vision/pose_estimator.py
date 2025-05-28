"""
Advanced Human Pose Estimation Module
Author: Drmusab
Last Modified: 2025-05-28 17:22:55 UTC

This module provides comprehensive human pose estimation capabilities for the AI assistant,
including 2D/3D pose detection, body landmark analysis, gesture recognition, and real-time
pose tracking with integration into the core system architecture.
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
import numpy as np
import cv2
from abc import ABC, abstractmethod
import json
import hashlib
from collections import defaultdict, deque
import weakref

# Computer vision libraries
try:
    import mediapipe as mp
    import torch
    import torchvision.transforms as transforms
    from ultralytics import YOLO
    import openpose
except ImportError as e:
    logging.warning(f"Optional pose estimation dependency not available: {e}")

# Core imports
from src.core.config.loader import ConfigLoader
from src.core.events.event_bus import EventBus
from src.core.events.event_types import (
    ComponentRegistered,
    ComponentInitialized,
    ComponentStarted,
    ComponentFailed,
    ProcessingCompleted,
    ProcessingFailed,
    QualityThresholdChanged
)
from src.core.error_handling import ErrorHandler, handle_exceptions
from src.core.dependency_injection import Container
from src.core.health_check import HealthCheck

# Observability
from src.observability.monitoring.metrics import MetricsCollector
from src.observability.monitoring.tracing import TraceManager
from src.observability.logging.config import get_logger

# Integration components
from src.integrations.cache.redis_cache import RedisCache
from src.integrations.storage.database import DatabaseManager
from src.core.security.encryption import EncryptionManager

# Learning and adaptation
from src.learning.feedback_processor import FeedbackProcessor
from src.learning.continual_learning import ContinualLearning


class PoseEstimationMethod(Enum):
    """Pose estimation methods."""
    MEDIAPIPE = "mediapipe"
    OPENPOSE = "openpose"
    YOLO_POSE = "yolo_pose"
    ALPHAPOSE = "alphapose"
    DETECTRON2 = "detectron2"
    MOVENET = "movenet"
    POSENET = "posenet"
    HRNET = "hrnet"


class PoseFormat(Enum):
    """Pose data formats."""
    COCO = "coco"           # 17 keypoints
    MPII = "mpii"           # 16 keypoints  
    BODY_25 = "body_25"     # OpenPose 25 keypoints
    FACE_468 = "face_468"   # MediaPipe face landmarks
    HAND_21 = "hand_21"     # MediaPipe hand landmarks
    WHOLEBODY = "wholebody" # Full body + face + hands


class PoseType(Enum):
    """Types of pose estimation."""
    SINGLE_PERSON = "single_person"
    MULTI_PERSON = "multi_person"
    REAL_TIME = "real_time"
    BATCH = "batch"
    TRACKING = "tracking"


class QualityLevel(Enum):
    """Quality levels for pose estimation."""
    FAST = "fast"           # Speed optimized
    BALANCED = "balanced"   # Speed/accuracy balance
    ACCURATE = "accurate"   # Accuracy optimized
    ULTRA = "ultra"         # Maximum accuracy


class TrackingMode(Enum):
    """Pose tracking modes."""
    NONE = "none"
    SIMPLE = "simple"
    KALMAN = "kalman"
    DEEP_SORT = "deep_sort"
    OPTICAL_FLOW = "optical_flow"


@dataclass
class PoseKeypoint:
    """Individual pose keypoint."""
    x: float
    y: float
    z: Optional[float] = None
    confidence: float = 0.0
    visibility: float = 1.0
    name: Optional[str] = None
    id: Optional[int] = None


@dataclass
class PosePerson:
    """Pose data for a single person."""
    person_id: str
    keypoints: List[PoseKeypoint]
    bbox: Tuple[float, float, float, float]  # x, y, width, height
    confidence: float
    pose_format: PoseFormat
    
    # Optional detailed data
    face_landmarks: List[PoseKeypoint] = field(default_factory=list)
    left_hand_landmarks: List[PoseKeypoint] = field(default_factory=list)
    right_hand_landmarks: List[PoseKeypoint] = field(default_factory=list)
    
    # Tracking information
    track_id: Optional[int] = None
    tracking_confidence: float = 0.0
    
    # Pose analysis
    pose_class: Optional[str] = None
    action_label: Optional[str] = None
    gesture_label: Optional[str] = None
    
    # Quality metrics
    pose_quality_score: float = 0.0
    occlusion_score: float = 0.0
    stability_score: float = 0.0


@dataclass
class PoseEstimationResult:
    """Complete pose estimation result."""
    success: bool
    request_id: str
    processing_time: float
    
    # Pose data
    detected_persons: List[PosePerson]
    total_persons: int
    
    # Image metadata
    image_dimensions: Tuple[int, int]  # height, width
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Processing metadata
    method_used: PoseEstimationMethod
    quality_level: QualityLevel
    pose_format: PoseFormat
    
    # Performance metrics
    fps: Optional[float] = None
    memory_usage_mb: float = 0.0
    gpu_usage_percent: float = 0.0
    
    # Quality metrics
    overall_confidence: float = 0.0
    detection_quality: float = 0.0
    tracking_quality: float = 0.0
    
    # Error information
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Additional data
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PoseConfiguration:
    """Configuration for pose estimation."""
    # Model settings
    method: PoseEstimationMethod = PoseEstimationMethod.MEDIAPIPE
    quality_level: QualityLevel = QualityLevel.BALANCED
    pose_format: PoseFormat = PoseFormat.COCO
    pose_type: PoseType = PoseType.MULTI_PERSON
    
    # Detection settings
    confidence_threshold: float = 0.5
    nms_threshold: float = 0.4
    max_persons: int = 10
    min_keypoints: int = 5
    
    # Tracking settings
    tracking_mode: TrackingMode = TrackingMode.SIMPLE
    tracking_threshold: float = 0.7
    max_track_age: int = 30
    min_track_hits: int = 3
    
    # Performance settings
    enable_gpu: bool = True
    batch_size: int = 1
    num_threads: int = 4
    
    # Input processing
    input_resolution: Tuple[int, int] = (640, 480)
    normalize_coordinates: bool = True
    flip_horizontally: bool = False
    
    # Output settings
    include_face: bool = False
    include_hands: bool = False
    include_3d: bool = False
    smooth_landmarks: bool = True
    
    # Quality control
    enable_quality_filter: bool = True
    min_pose_quality: float = 0.3
    enable_occlusion_detection: bool = True
    
    # Caching
    enable_caching: bool = True
    cache_ttl: int = 300
    
    # Debugging
    enable_visualization: bool = False
    save_debug_images: bool = False


class PoseEstimationError(Exception):
    """Custom exception for pose estimation operations."""
    
    def __init__(self, message: str, error_code: Optional[str] = None, 
                 method: Optional[str] = None):
        super().__init__(message)
        self.error_code = error_code
        self.method = method
        self.timestamp = datetime.now(timezone.utc)


class PoseEstimator(ABC):
    """Abstract base class for pose estimators."""
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the pose estimator."""
        pass
    
    @abstractmethod
    async def estimate_pose(self, image: np.ndarray, 
                          config: Optional[PoseConfiguration] = None) -> PoseEstimationResult:
        """Estimate poses in an image."""
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Cleanup resources."""
        pass


class MediaPipePoseEstimator(PoseEstimator):
    """MediaPipe-based pose estimator."""
    
    def __init__(self, logger):
        self.logger = logger
        self.mp_pose = None
        self.mp_holistic = None
        self.pose_detector = None
        self.holistic_detector = None
        self.initialized = False
    
    async def initialize(self) -> None:
        """Initialize MediaPipe pose estimation."""
        try:
            self.mp_pose = mp.solutions.pose
            self.mp_holistic = mp.solutions.holistic
            
            self.pose_detector = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                enable_segmentation=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            
            self.holistic_detector = self.mp_holistic.Holistic(
                static_image_mode=False,
                model_complexity=1,
                enable_segmentation=False,
                refine_face_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            
            self.initialized = True
            self.logger.info("MediaPipe pose estimator initialized")
            
        except Exception as e:
            raise PoseEstimationError(f"Failed to initialize MediaPipe: {str(e)}", 
                                    method="mediapipe")
    
    async def estimate_pose(self, image: np.ndarray, 
                          config: Optional[PoseConfiguration] = None) -> PoseEstimationResult:
        """Estimate poses using MediaPipe."""
        if not self.initialized:
            await self.initialize()
        
        start_time = time.time()
        request_id = hashlib.md5(f"{time.time()}".encode()).hexdigest()[:8]
        
        try:
            # Convert image format
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process with appropriate detector
            if config and (config.include_face or config.include_hands):
                results = self.holistic_detector.process(rgb_image)
                pose_landmarks = results.pose_landmarks
                face_landmarks = results.face_landmarks
                left_hand_landmarks = results.left_hand_landmarks
                right_hand_landmarks = results.right_hand_landmarks
            else:
                results = self.pose_detector.process(rgb_image)
                pose_landmarks = results.pose_landmarks
                face_landmarks = None
                left_hand_landmarks = None
                right_hand_landmarks = None
            
            # Extract pose data
            detected_persons = []
            
            if pose_landmarks:
                person = self._extract_pose_person(
                    pose_landmarks, face_landmarks, 
                    left_hand_landmarks, right_hand_landmarks,
                    image.shape, config
                )
                detected_persons.append(person)
            
            processing_time = time.time() - start_time
            
            return PoseEstimationResult(
                success=True,
                request_id=request_id,
                processing_time=processing_time,
                detected_persons=detected_persons,
                total_persons=len(detected_persons),
                image_dimensions=(image.shape[0], image.shape[1]),
                method_used=PoseEstimationMethod.MEDIAPIPE,
                quality_level=config.quality_level if config else QualityLevel.BALANCED,
                pose_format=PoseFormat.COCO,
                overall_confidence=detected_persons[0].confidence if detected_persons else 0.0
            )
            
        except Exception as e:
            self.logger.error(f"MediaPipe pose estimation failed: {str(e)}")
            return PoseEstimationResult(
                success=False,
                request_id=request_id,
                processing_time=time.time() - start_time,
                detected_persons=[],
                total_persons=0,
                image_dimensions=(image.shape[0], image.shape[1]),
                method_used=PoseEstimationMethod.MEDIAPIPE,
                quality_level=config.quality_level if config else QualityLevel.BALANCED,
                pose_format=PoseFormat.COCO,
                errors=[str(e)]
            )
    
    def _extract_pose_person(self, pose_landmarks, face_landmarks, 
                           left_hand_landmarks, right_hand_landmarks,
                           image_shape: Tuple[int, int, int], 
                           config: Optional[PoseConfiguration]) -> PosePerson:
        """Extract pose person data from MediaPipe results."""
        height, width = image_shape[:2]
        
        # Extract body keypoints
        keypoints = []
        confidence_sum = 0.0
        
        for i, landmark in enumerate(pose_landmarks.landmark):
            x = landmark.x * width if config and config.normalize_coordinates else landmark.x
            y = landmark.y * height if config and config.normalize_coordinates else landmark.y
            z = landmark.z if hasattr(landmark, 'z') else None
            confidence = landmark.visibility if hasattr(landmark, 'visibility') else 1.0
            
            keypoint = PoseKeypoint(
                x=x, y=y, z=z, confidence=confidence,
                visibility=landmark.visibility if hasattr(landmark, 'visibility') else 1.0,
                id=i
            )
            keypoints.append(keypoint)
            confidence_sum += confidence
        
        # Calculate bounding box
        valid_points = [(kp.x, kp.y) for kp in keypoints if kp.confidence > 0.5]
        if valid_points:
            xs, ys = zip(*valid_points)
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            bbox = (x_min, y_min, x_max - x_min, y_max - y_min)
        else:
            bbox = (0, 0, width, height)
        
        # Extract additional landmarks if available
        face_kps = []
        if face_landmarks:
            face_kps = self._extract_face_landmarks(face_landmarks, width, height, config)
        
        left_hand_kps = []
        if left_hand_landmarks:
            left_hand_kps = self._extract_hand_landmarks(left_hand_landmarks, width, height, config)
        
        right_hand_kps = []
        if right_hand_landmarks:
            right_hand_kps = self._extract_hand_landmarks(right_hand_landmarks, width, height, config)
        
        return PosePerson(
            person_id=f"mp_{int(time.time() * 1000)}",
            keypoints=keypoints,
            bbox=bbox,
            confidence=confidence_sum / len(keypoints) if keypoints else 0.0,
            pose_format=PoseFormat.COCO,
            face_landmarks=face_kps,
            left_hand_landmarks=left_hand_kps,
            right_hand_landmarks=right_hand_kps,
            pose_quality_score=confidence_sum / len(keypoints) if keypoints else 0.0
        )
    
    def _extract_face_landmarks(self, face_landmarks, width: int, height: int,
                              config: Optional[PoseConfiguration]) -> List[PoseKeypoint]:
        """Extract face landmarks."""
        landmarks = []
        for i, landmark in enumerate(face_landmarks.landmark):
            x = landmark.x * width if config and config.normalize_coordinates else landmark.x
            y = landmark.y * height if config and config.normalize_coordinates else landmark.y
            z = landmark.z if hasattr(landmark, 'z') else None
            
            landmarks.append(PoseKeypoint(
                x=x, y=y, z=z, confidence=1.0, id=i
            ))
        return landmarks
    
    def _extract_hand_landmarks(self, hand_landmarks, width: int, height: int,
                              config: Optional[PoseConfiguration]) -> List[PoseKeypoint]:
        """Extract hand landmarks."""
        landmarks = []
        for i, landmark in enumerate(hand_landmarks.landmark):
            x = landmark.x * width if config and config.normalize_coordinates else landmark.x
            y = landmark.y * height if config and config.normalize_coordinates else landmark.y
            z = landmark.z if hasattr(landmark, 'z') else None
            
            landmarks.append(PoseKeypoint(
                x=x, y=y, z=z, confidence=1.0, id=i
            ))
        return landmarks
    
    def cleanup(self) -> None:
        """Cleanup MediaPipe resources."""
        if self.pose_detector:
            self.pose_detector.close()
        if self.holistic_detector:
            self.holistic_detector.close()
        self.initialized = False


class YOLOPoseEstimator(PoseEstimator):
    """YOLO-based pose estimator."""
    
    def __init__(self, logger, model_path: Optional[str] = None):
        self.logger = logger
        self.model_path = model_path or "yolov8n-pose.pt"
        self.model = None
        self.initialized = False
    
    async def initialize(self) -> None:
        """Initialize YOLO pose estimation."""
        try:
            self.model = YOLO(self.model_path)
            self.initialized = True
            self.logger.info(f"YOLO pose estimator initialized with model: {self.model_path}")
            
        except Exception as e:
            raise PoseEstimationError(f"Failed to initialize YOLO: {str(e)}", 
                                    method="yolo_pose")
    
    async def estimate_pose(self, image: np.ndarray, 
                          config: Optional[PoseConfiguration] = None) -> PoseEstimationResult:
        """Estimate poses using YOLO."""
        if not self.initialized:
            await self.initialize()
        
        start_time = time.time()
        request_id = hashlib.md5(f"{time.time()}".encode()).hexdigest()[:8]
        
        try:
            # Run inference
            results = self.model(image, verbose=False)
            
            detected_persons = []
            for result in results:
                if hasattr(result, 'keypoints') and result.keypoints is not None:
                    keypoints_data = result.keypoints.data
                    boxes = result.boxes.xyxy if result.boxes is not None else None
                    
                    for i, kps in enumerate(keypoints_data):
                        person = self._extract_yolo_person(
                            kps, boxes[i] if boxes is not None and i < len(boxes) else None,
                            image.shape, config, i
                        )
                        detected_persons.append(person)
            
            processing_time = time.time() - start_time
            
            return PoseEstimationResult(
                success=True,
                request_id=request_id,
                processing_time=processing_time,
                detected_persons=detected_persons,
                total_persons=len(detected_persons),
                image_dimensions=(image.shape[0], image.shape[1]),
                method_used=PoseEstimationMethod.YOLO_POSE,
                quality_level=config.quality_level if config else QualityLevel.BALANCED,
                pose_format=PoseFormat.COCO,
                overall_confidence=np.mean([p.confidence for p in detected_persons]) if detected_persons else 0.0
            )
            
        except Exception as e:
            self.logger.error(f"YOLO pose estimation failed: {str(e)}")
            return PoseEstimationResult(
                success=False,
                request_id=request_id,
                processing_time=time.time() - start_time,
                detected_persons=[],
                total_persons=0,
                image_dimensions=(image.shape[0], image.shape[1]),
                method_used=PoseEstimationMethod.YOLO_POSE,
                quality_level=config.quality_level if config else QualityLevel.BALANCED,
                pose_format=PoseFormat.COCO,
                errors=[str(e)]
            )
    
    def _extract_yolo_person(self, keypoints_tensor, bbox_tensor, 
                           image_shape: Tuple[int, int, int], 
                           config: Optional[PoseConfiguration], person_idx: int) -> PosePerson:
        """Extract pose person data from YOLO results."""
        height, width = image_shape[:2]
        
        # Extract keypoints
        keypoints = []
        confidence_sum = 0.0
        
        for i, (x, y, conf) in enumerate(keypoints_tensor):
            if config and config.normalize_coordinates:
                x = float(x)
                y = float(y)
            else:
                x = float(x) / width
                y = float(y) / height
            
            confidence = float(conf)
            keypoint = PoseKeypoint(x=x, y=y, confidence=confidence, id=i)
            keypoints.append(keypoint)
            confidence_sum += confidence
        
        # Extract bounding box
        if bbox_tensor is not None:
            x1, y1, x2, y2 = map(float, bbox_tensor)
            bbox = (x1, y1, x2 - x1, y2 - y1)
        else:
            # Calculate from keypoints
            valid_points = [(kp.x, kp.y) for kp in keypoints if kp.confidence > 0.5]
            if valid_points:
                xs, ys = zip(*valid_points)
                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)
                bbox = (x_min, y_min, x_max - x_min, y_max - y_min)
            else:
                bbox = (0, 0, width, height)
        
        return PosePerson(
            person_id=f"yolo_{person_idx}_{int(time.time() * 1000)}",
            keypoints=keypoints,
            bbox=bbox,
            confidence=confidence_sum / len(keypoints) if keypoints else 0.0,
            pose_format=PoseFormat.COCO,
            pose_quality_score=confidence_sum / len(keypoints) if keypoints else 0.0
        )
    
    def cleanup(self) -> None:
        """Cleanup YOLO resources."""
        self.model = None
        self.initialized = False


class PoseTracker:
    """Advanced pose tracking across frames."""
    
    def __init__(self, mode: TrackingMode = TrackingMode.SIMPLE):
        self.mode = mode
        self.active_tracks: Dict[int, Dict[str, Any]] = {}
        self.next_track_id = 1
        self.max_track_age = 30
        self.tracking_threshold = 0.7
        
    def update_tracks(self, detected_persons: List[PosePerson]) -> List[PosePerson]:
        """Update tracking information for detected persons."""
        if self.mode == TrackingMode.NONE:
            return detected_persons
        
        # Simple tracking based on spatial proximity
        if self.mode == TrackingMode.SIMPLE:
            return self._simple_tracking(detected_persons)
        
        # TODO: Implement more sophisticated tracking methods
        return detected_persons
    
    def _simple_tracking(self, detected_persons: List[PosePerson]) -> List[PosePerson]:
        """Simple spatial proximity-based tracking."""
        tracked_persons = []
        
        for person in detected_persons:
            best_track_id = None
            best_distance = float('inf')
            
            # Find best matching track
            for track_id, track_data in self.active_tracks.items():
                if track_data['age'] > self.max_track_age:
                    continue
                
                # Calculate distance between current and previous position
                prev_bbox = track_data['bbox']
                curr_bbox = person.bbox
                
                distance = self._calculate_bbox_distance(prev_bbox, curr_bbox)
                
                if distance < best_distance and distance < self.tracking_threshold:
                    best_distance = distance
                    best_track_id = track_id
            
            # Assign track ID
            if best_track_id is not None:
                person.track_id = best_track_id
                person.tracking_confidence = 1.0 - (best_distance / self.tracking_threshold)
                self.active_tracks[best_track_id]['bbox'] = person.bbox
                self.active_tracks[best_track_id]['age'] = 0
            else:
                person.track_id = self.next_track_id
                person.tracking_confidence = 1.0
                self.active_tracks[self.next_track_id] = {
                    'bbox': person.bbox,
                    'age': 0
                }
                self.next_track_id += 1
            
            tracked_persons.append(person)
        
        # Age existing tracks
        for track_id in list(self.active_tracks.keys()):
            self.active_tracks[track_id]['age'] += 1
            if self.active_tracks[track_id]['age'] > self.max_track_age:
                del self.active_tracks[track_id]
        
        return tracked_persons
    
    def _calculate_bbox_distance(self, bbox1: Tuple[float, float, float, float],
                               bbox2: Tuple[float, float, float, float]) -> float:
        """Calculate distance between two bounding boxes."""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Calculate center points
        cx1, cy1 = x1 + w1/2, y1 + h1/2
        cx2, cy2 = x2 + w2/2, y2 + h2/2
        
        # Euclidean distance between centers
        return np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)


class PoseAnalyzer:
    """Analyzes poses for classification and action recognition."""
    
    def __init__(self, logger):
        self.logger = logger
        self.pose_classes = {
            "standing": self._is_standing,
            "sitting": self._is_sitting,
            "lying": self._is_lying,
            "walking": self._is_walking,
            "running": self._is_running,
            "jumping": self._is_jumping,
            "waving": self._is_waving,
            "pointing": self._is_pointing
        }
    
    def analyze_pose(self, person: PosePerson) -> PosePerson:
        """Analyze pose and add classification results."""
        # Pose classification
        for pose_class, classifier in self.pose_classes.items():
            if classifier(person.keypoints):
                person.pose_class = pose_class
                break
        
        # Gesture recognition
        person.gesture_label = self._recognize_gesture(person.keypoints)
        
        # Quality assessment
        person.pose_quality_score = self._assess_pose_quality(person.keypoints)
        person.occlusion_score = self._assess_occlusion(person.keypoints)
        
        return person
    
    def _is_standing(self, keypoints: List[PoseKeypoint]) -> bool:
        """Check if person is standing."""
        # Simple heuristic: legs are extended and person is upright
        try:
            left_hip = next(kp for kp in keypoints if kp.id == 11)
            left_knee = next(kp for kp in keypoints if kp.id == 13)
            left_ankle = next(kp for kp in keypoints if kp.id == 15)
            
            # Check if leg is relatively straight
            hip_knee_dist = abs(left_hip.y - left_knee.y)
            knee_ankle_dist = abs(left_knee.y - left_ankle.y)
            
            return hip_knee_dist > 0.1 and knee_ankle_dist > 0.1
        except (StopIteration, AttributeError):
            return False
    
    def _is_sitting(self, keypoints: List[PoseKeypoint]) -> bool:
        """Check if person is sitting."""
        try:
            left_hip = next(kp for kp in keypoints if kp.id == 11)
            left_knee = next(kp for kp in keypoints if kp.id == 13)
            
            # In sitting position, knee is typically higher than hip
            return left_knee.y < left_hip.y
        except (StopIteration, AttributeError):
            return False
    
    def _is_lying(self, keypoints: List[PoseKeypoint]) -> bool:
        """Check if person is lying down."""
        try:
            nose = next(kp for kp in keypoints if kp.id == 0)
            left_hip = next(kp for kp in keypoints if kp.id == 11)
            
            # If head and hip are at similar y-level, person might be lying
            return abs(nose.y - left_hip.y) < 0.2
        except (StopIteration, AttributeError):
            return False
    
    def _is_walking(self, keypoints: List[PoseKeypoint]) -> bool:
        """Check if person is walking (requires temporal analysis)."""
        # This would require frame-to-frame analysis
        return False
    
    def _is_running(self, keypoints: List[PoseKeypoint]) -> bool:
        """Check if person is running (requires temporal analysis)."""
        # This would require frame-to-frame analysis
        return False
    
    def _is_jumping(self, keypoints: List[PoseKeypoint]) -> bool:
        """Check if person is jumping."""
        try:
            left_ankle = next(kp for kp in keypoints if kp.id == 15)
            right_ankle = next(kp for kp in keypoints if kp.id == 16)
            
            # Simple heuristic: both feet are off the ground (high y values)
            return left_ankle.y < 0.8 and right_ankle.y < 0.8
        except (StopIteration, AttributeError):
            return False
    
    def _is_waving(self, keypoints: List[PoseKeypoint]) -> bool:
        """Check if person is waving."""
        try:
            left_wrist = next(kp for kp in keypoints if kp.id == 9)
            left_shoulder = next(kp for kp in keypoints if kp.id == 5)
            
            # Wrist is above shoulder level
            return left_wrist.y < left_shoulder.y
        except (StopIteration, AttributeError):
            return False
    
    def _is_pointing(self, keypoints: List[PoseKeypoint]) -> bool:
        """Check if person is pointing."""
        try:
            right_wrist = next(kp for kp in keypoints if kp.id == 10)
            right_elbow = next(kp for kp in keypoints if kp.id == 8)
            right_shoulder = next(kp for kp in keypoints if kp.id == 6)
            
            # Arm is extended
            shoulder_elbow_dist = np.sqrt((right_shoulder.x - right_elbow.x)**2 + 
                                        (right_shoulder.y - right_elbow.y)**2)
            elbow_wrist_dist = np.sqrt((right_elbow.x - right_wrist.x)**2 + 
                                     (right_elbow.y - right_wrist.y)**2)
            
            return shoulder_elbow_dist > 0.15 and elbow_wrist_dist > 0.15
        except (StopIteration, AttributeError):
            return False
    
    def _recognize_gesture(self, keypoints: List[PoseKeypoint]) -> Optional[str]:
        """Recognize hand gestures."""
        # This would require more sophisticated gesture recognition
        # For now, return None
        return None
    
    def _assess_pose_quality(self, keypoints: List[PoseKeypoint]) -> float:
        """Assess overall pose quality."""
        if not keypoints:
            return 0.0
        
        # Calculate based on average confidence and visibility
        confidence_sum = sum(kp.confidence for kp in keypoints)
        visibility_sum = sum(kp.visibility for kp in keypoints)
        
        avg_confidence = confidence_sum / len(keypoints)
        avg_visibility = visibility_sum / len(keypoints)
        
        return (avg_confidence + avg_visibility) / 2
    
    def _assess_occlusion(self, keypoints: List[PoseKeypoint]) -> float:
        """Assess level of occlusion."""
        if not keypoints:
            return 1.0
        
        # Count low-confidence/low-visibility keypoints
        occluded_count = sum(1 for kp in keypoints 
                           if kp.confidence < 0.5 or kp.visibility < 0.5)
        
        return occluded_count / len(keypoints)


class EnhancedPoseEstimator:
    """
    Enhanced Pose Estimation System integrated with the AI Assistant architecture.
    
    Features:
    - Multiple pose estimation methods (MediaPipe, YOLO, OpenPose)
    - Real-time and batch processing
    - Pose tracking and analysis
    - Quality assessment and filtering
    - Caching and performance optimization
    - Integration with core system components
    - Event-driven architecture
    - Health monitoring and error handling
    """
    
    def __init__(self, container: Container):
        """
        Initialize the enhanced pose estimator.
        
        Args:
            container: Dependency injection container
        """
        self.container = container
        self.logger = get_logger(__name__)
        self.config = container.get(ConfigLoader)
        self.event_bus = container.get(EventBus)
        self.error_handler = container.get(ErrorHandler)
        self.health_check = container.get(HealthCheck)
        
        # Core configuration
        self._load_configuration()
        
        # Component setup
        self._setup_core_components()
        self._setup_monitoring()
        self._setup_caching()
        
        # Pose estimators
        self.estimators: Dict[PoseEstimationMethod, PoseEstimator] = {}
        self.current_estimator: Optional[PoseEstimator] = None
        
        # Analysis components
        self.pose_tracker = PoseTracker(self.pose_config.tracking_mode)
        self.pose_analyzer = PoseAnalyzer(self.logger)
        
        # State management
        self.initialized = False
        self.processing_count = 0
        self.total_processed = 0
        self.error_count = 0
        
        # Performance tracking
        self.performance_metrics = {
            'total_processing_time': 0.0,
            'average_processing_time': 0.0,
            'frames_per_second': 0.0,
            'memory_usage_mb': 0.0,
            'gpu_usage_percent': 0.0
        }
        
        # Register health check
        self.health_check.register_component("pose_estimator", self._health_check_callback)
        
        self.logger.info("EnhancedPoseEstimator initialized")
    
    def _load_configuration(self) -> None:
        """Load pose estimation configuration."""
        pose_config = self.config.get("pose_estimation", {})
        
        self.pose_config = PoseConfiguration(
            method=PoseEstimationMethod(pose_config.get("method", "mediapipe")),
            quality_level=QualityLevel(pose_config.get("quality_level", "balanced")),
            pose_format=PoseFormat(pose_config.get("pose_format", "coco")),
            pose_type=PoseType(pose_config.get("pose_type", "multi_person")),
            confidence_threshold=pose_config.get("confidence_threshold", 0.5),
            nms_threshold=pose_config.get("nms_threshold", 0.4),
            max_persons=pose_config.get("max_persons", 10),
            tracking_mode=TrackingMode(pose_config.get("tracking_mode", "simple")),
            enable_gpu=pose_config.get("enable_gpu", True),
            include_face=pose_config.get("include_face", False),
            include_hands=pose_config.get("include_hands", False),
            enable_caching=pose_config.get("enable_caching", True),
            cache_ttl=pose_config.get("cache_ttl", 300)
        )
        
        self.logger.info(f"Loaded pose estimation configuration: {self.pose_config.method.value}")
    
    def _setup_core_components(self) -> None:
        """Setup core components."""
        try:
            self.database = self.container.get(DatabaseManager)
            self.encryption = self.container.get(EncryptionManager)
            self.feedback_processor = self.container.get(FeedbackProcessor)
        except Exception as e:
            self.logger.warning(f"Optional components not available: {e}")
            self.database = None
            self.encryption = None
            self.feedback_processor = None
    
    def _setup_monitoring(self) -> None:
        """Setup monitoring and metrics collection."""
        try:
            self.metrics = self.container.get(MetricsCollector)
            self.tracer = self.container.get(TraceManager)
            
            # Register pose estimation metrics
            self.metrics.register_counter("pose_estimations_total")
            self.metrics.register_counter("pose_estimation_errors_total")
            self.metrics.register_histogram("pose_estimation_duration_seconds")
            self.metrics.register_gauge("pose_estimator_active_requests")
            self.metrics.register_gauge("pose_estimator_memory_usage_mb")
            self.metrics.register_counter("poses_detected_total")
            
        except Exception as e:
            self.logger.warning(f"Failed to setup monitoring: {str(e)}")
            self.metrics = None
            self.tracer = None
    
    def _setup_caching(self) -> None:
        """Setup caching for pose estimation results."""
        try:
            if self.pose_config.enable_caching:
                self.cache = self.container.get(RedisCache)
                self.cache_enabled = True
            else:
                self.cache = None
                self.cache_enabled = False
        except Exception as e:
            self.logger.warning(f"Cache not available: {e}")
            self.cache = None
            self.cache_enabled = False
    
    @handle_exceptions
    async def initialize(self) -> None:
        """Initialize pose estimation components."""
        if self.initialized:
            self.logger.warning("Pose estimator already initialized")
            return
        
        try:
            # Initialize estimators based on configuration
            await self._initialize_estimators()
            
            # Set current estimator
            self.current_estimator = self.estimators.get(self.pose_config.method)
            if not self.current_estimator:
                raise PoseEstimationError(f"Failed to initialize {self.pose_config.method.value} estimator")
            
            # Emit initialization event
            await self.event_bus.emit(ComponentInitialized(
                component_id="pose_estimator",
                initialization_time=0.0  # Would track actual time
            ))
            
            self.initialized = True
            self.logger.info("Pose estimator initialization completed")
            
        except Exception as e:
            self.logger.error(f"Pose estimator initialization failed: {str(e)}")
            raise PoseEstimationError(f"Initialization failed: {str(e)}")
    
    async def _initialize_estimators(self) -> None:
        """Initialize available pose estimators."""
        estimator_configs = {
            PoseEstimationMethod.MEDIAPIPE: MediaPipePoseEstimator,
            PoseEstimationMethod.YOLO_POSE: YOLOPoseEstimator,
            # Add more estimators as needed
        }
        
        for method, estimator_class in estimator_configs.items():
            try:
                if method == PoseEstimationMethod.MEDIAPIPE:
                    estimator = estimator_class(self.logger)
                elif method == PoseEstimationMethod.YOLO_POSE:
                    model_path = self.config.get("pose_estimation.yolo_model_path", "yolov8n-pose.pt")
                    estimator = estimator_class(self.logger, model_path)
                else:
                    continue
                
                await estimator.initialize()
                self.estimators[method] = estimator
                self.logger.info(f"Initialized {method.value} estimator")
                
            except Exception as e:
                self.logger.warning(f"Failed to initialize {method.value} estimator: {str(e)}")
    
    @handle_exceptions
    async def estimate_pose(self, image: np.ndarray, 
                          config: Optional[PoseConfiguration] = None,
                          session_id: Optional[str] = None,
                          user_id: Optional[str] = None) -> PoseEstimationResult:
        """
        Estimate poses in an image.
        
        Args:
            image: Input image as numpy array
            config: Optional pose estimation configuration
            session_id: Optional session ID for tracking
            user_id: Optional user ID for personalization
            
        Returns:
            PoseEstimationResult containing detected poses and metadata
        """
        if not self.initialized:
            await self.initialize()
        
        start_time = time.time()
        request_id = hashlib.md5(f"{time.time()}_{session_id}".encode()).hexdigest()[:8]
        
        # Use provided config or default
        estimation_config = config or self.pose_config
        
        # Update metrics
        self.processing_count += 1
        if self.metrics:
            self.metrics.increment("pose_estimations_total")
            self.metrics.set("pose_estimator_active_requests", self.processing_count)
        
        try:
            with self.tracer.trace("pose_estimation") if self.tracer else None:
                # Check cache first
                cache_key = None
                if self.cache_enabled and self.cache:
                    cache_key = self._generate_cache_key(image, estimation_config)
                    cached_result = await self._get_cached_result(cache_key)
                    if cached_result:
                        self.logger.debug(f"Returning cached pose estimation result")
                        return cached_result
                
                # Perform pose estimation
                result = await self.current_estimator.estimate_pose(image, estimation_config)
                
                # Post-process results
                if result.success and result.detected_persons:
                    # Apply tracking
                    if estimation_config.tracking_mode != TrackingMode.NONE:
                        result.detected_persons = self.pose_tracker.update_tracks(result.detected_persons)
                    
                    # Apply pose analysis
                    analyzed_persons = []
                    for person in result.detected_persons:
                        analyzed_person = self.pose_analyzer.analyze_pose(person)
                        analyzed_persons.append(analyzed_person)
                    result.detected_persons = analyzed_persons
                    
                    # Quality filtering
                    if estimation_config.enable_quality_filter:
                        result.detected_persons = [
                            person for person in result.detected_persons
                            if person.pose_quality_score >= estimation_config.min_pose_quality
                        ]
                        result.total_persons = len(result.detected_persons)
                
                # Cache result
                if self.cache_enabled and self.cache and cache_key and result.success:
                    await self._cache_result(cache_key, result, estimation_config.cache_ttl)
                
                # Update performance metrics
                processing_time = time.time() - start_time
                self._update_metrics(processing_time, len(result.detected_persons))
                
                # Store for learning if feedback processor available
                if self.feedback_processor and session_id:
                    await self._store_estimation_for_learning(result, session_id, user_id)
                
                # Emit processing event
                await self.event_bus.emit(ProcessingCompleted(
                    component_id="pose_estimator",
                    processing_time=processing_time,
                    request_id=request_id
                ))
                
                self.total_processed += 1
                return result
                
        except Exception as e:
            self.error_count += 1
            processing_time = time.time() - start_time
            
            error_msg = f"Pose estimation failed: {str(e)}"
            self.logger.error(error_msg)
            
            if self.metrics:
                self.metrics.increment("pose_estimation_errors_total")
            
            # Emit error event
            await self.event_bus.emit(ProcessingFailed(
                component_id="pose_estimator",
                error_message=error_msg,
                request_id=request_id
            ))
            
            return PoseEstimationResult(
                success=False,
                request_id=request_id,
                processing_time=processing_time,
                detected_persons=[],
                total_persons=0,
                image_dimensions=(image.shape[0], image.shape[1]),
                method_used=estimation_config.method,
                quality_level=estimation_config.quality_level,
                pose_format=estimation_config.pose_format,
                errors=[str(e)]
            )
        
        finally:
            self.processing_count -= 1
            if self.metrics:
                self.metrics.set("pose_estimator_active_requests", self.processing_count)
    
    def _generate_cache_key(self, image: np.ndarray, config: PoseConfiguration) -> str:
        """Generate cache key for image and configuration."""
        # Create hash from image and config
        image_hash = hashlib.md5(image.tobytes()).hexdigest()[:16]
        config_hash = hashlib.md5(str(config.__dict__).encode()).hexdigest()[:16]
        return f"pose_est_{image_hash}_{config_hash}"
    
    async def _get_cached_result(self, cache_key: str) -> Optional[PoseEstimationResult]:
        """Get cached pose estimation result."""
        try:
            cached_data = await self.cache.get(cache_key)
            if cached_data:
                # Deserialize cached result
                result_dict = json.loads(cached_data)
                # Convert back to PoseEstimationResult
                # This would require proper serialization/deserialization
                return None  # Simplified for now
        except Exception as e:
            self.logger.warning(f"Failed to get cached result: {e}")
        return None
    
    async def _cache_result(self, cache_key: str, result: PoseEstimationResult, ttl: int) -> None:
        """Cache pose estimation result."""
        try:
            # Serialize result to JSON
            # This would require proper serialization
            result_dict = {
                "success": result.success,
                "processing_time": result.processing_time,
                "total_persons": result.total_persons,
                # Add other serializable fields
            }
            await self.cache.set(cache_key, json.dumps(result_dict), expire=ttl)
        except Exception as e:
            self.logger.warning(f"Failed to cache result: {e}")
    
    def _update_metrics(self, processing_time: float, poses_count: int) -> None:
        """Update performance metrics."""
        self.performance_metrics['total_processing_time'] += processing_time
        if self.total_processed > 0:
            self.performance_metrics['average_processing_time'] = (
                self.performance_metrics['total_processing_time'] / self.total_processed
            )
        
        if processing_time > 0:
            self.performance_metrics['frames_per_second'] = 1.0 / processing_time
        
        if self.metrics:
            self.metrics.record("pose_estimation_duration_seconds", processing_time)
            self.metrics.add("poses_detected_total", poses_count)
    
    async def _store_estimation_for_learning(self, result: PoseEstimationResult, 
                                           session_id: str, user_id: Optional[str]) -> None:
        """Store pose estimation result for learning and improvement."""
        try:
            learning_data = {
                "request_id": result.request_id,
                "session_id": session_id,
                "user_id": user_id,
                "method_used": result.method_used.value,
                "processing_time": result.processing_time,
                "poses_detected": result.total_persons,
                "confidence": result.overall_confidence,
                "quality_metrics": {
                    "detection_quality": result.detection_quality,
                    "tracking_quality": result.tracking_quality
                },
                "timestamp": result.timestamp.isoformat()
            }
            
            await self.feedback_processor.store_interaction_data(
                "pose_estimation", learning_data
            )
            
        except Exception as e:
            self.logger.warning(f"Failed to store estimation for learning: {e}")
    
    async def switch_method(self, method: PoseEstimationMethod) -> None:
        """Switch to a different pose estimation method."""
        if method not in self.estimators:
            raise PoseEstimationError(f"Method {method.value} not available")
        
        self.current_estimator = self.estimators[method]
        self.pose_config.method = method
        
        self.logger.info(f"Switched to {method.value} pose estimation method")
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            "initialized": self.initialized,
            "current_method": self.pose_config.method.value if self.current_estimator else None,
            "available_methods": [method.value for method in self.estimators.keys()],
            "processing_count": self.processing_count,
            "total_processed": self.total_processed,
            "error_count": self.error_count,
            "error_rate": self.error_count / max(self.total_processed, 1),
            "performance_metrics": self.performance_metrics,
            "configuration": {
                "quality_level": self.pose_config.quality_level.value,
                "pose_format": self.pose_config.pose_format.value,
                "tracking_enabled": self.pose_config.tracking_mode != TrackingMode.NONE,
                "caching_enabled": self.cache_enabled
            }
        }
    
    async def _health_check_callback(self) -> Dict[str, Any]:
        """Health check callback for monitoring."""
        try:
            healthy = self.initialized and self.current_estimator is not None
            error_rate = self.error_count / max(self.total_processed, 1)
            
            status = "healthy" if healthy and error_rate < 0.1 else "unhealthy"
            
            return {
                "status": status,
                "initialized": self.initialized,
                "current_method": self.pose_config.method.value if self.current_estimator else None,
                "error_rate": error_rate,
                "processing_count": self.processing_count,
                "total_processed": self.total_processed
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    async def cleanup(self) -> None:
        """Cleanup pose estimation resources."""
        try:
            # Cleanup all estimators
            for estimator in self.estimators.values():
                estimator.cleanup()
            
            self.estimators.clear()
            self.current_estimator = None
            self.initialized = False
            
            self.logger.info("Pose estimator cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            if hasattr(self, 'estimators') and self.estimators:
                for estimator in self.estimators.values():
                    estimator.cleanup()
        except Exception:
            pass  # Ignore cleanup errors in destructor
