"""
Advanced Face Recognition System
Author: Drmusab
Last Modified: 2025-05-28 17:04:12 UTC

This module provides comprehensive face recognition capabilities including detection,
encoding, matching, verification, and identity management with full integration
into the AI assistant's core architecture.
"""

import asyncio
import numpy as np
import cv2
import time
import hashlib
import json
import base64
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Set, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timezone
from contextlib import asynccontextmanager
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
import sqlite3
from collections import defaultdict, deque
import weakref

# Core imports
from src.core.config.loader import ConfigLoader
from src.core.events.event_bus import EventBus
from src.core.events.event_types import (
    ComponentInitialized,
    ComponentFailed,
    ComponentHealthChanged
)
from src.core.error_handling import ErrorHandler, handle_exceptions
from src.core.dependency_injection import Container
from src.core.health_check import HealthCheck
from src.core.security.encryption import EncryptionManager

# Observability
from src.observability.monitoring.metrics import MetricsCollector
from src.observability.monitoring.tracing import TraceManager
from src.observability.logging.config import get_logger

# Processing components
from src.processing.vision.vision_processor import VisionProcessor
from src.integrations.cache.redis_cache import RedisCache
from src.integrations.storage.database import DatabaseManager

# External dependencies
try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False

try:
    import dlib
    DLIB_AVAILABLE = True
except ImportError:
    DLIB_AVAILABLE = False

try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False


class FaceDetectionMethod(Enum):
    """Face detection methods."""
    HOG = "hog"
    CNN = "cnn"
    MTCNN = "mtcnn"
    SSD = "ssd"
    OPENCV_DNN = "opencv_dnn"
    YOLO = "yolo"


class FaceEncodingMethod(Enum):
    """Face encoding methods."""
    FACENET = "facenet"
    ARCFACE = "arcface"
    OPENFACE = "openface"
    DLIB = "dlib"
    VGG_FACE = "vgg_face"
    SFACE = "sface"


class AntiSpoofingMethod(Enum):
    """Anti-spoofing detection methods."""
    LIVENESS_DETECTION = "liveness_detection"
    DEPTH_ANALYSIS = "depth_analysis"
    TEXTURE_ANALYSIS = "texture_analysis"
    MOTION_ANALYSIS = "motion_analysis"
    CHALLENGE_RESPONSE = "challenge_response"


class MatchingStrategy(Enum):
    """Face matching strategies."""
    EUCLIDEAN = "euclidean"
    COSINE = "cosine"
    MANHATTAN = "manhattan"
    THRESHOLD_BASED = "threshold_based"
    ENSEMBLE = "ensemble"


class VerificationMode(Enum):
    """Face verification modes."""
    ONE_TO_ONE = "one_to_one"
    ONE_TO_MANY = "one_to_many"
    MANY_TO_MANY = "many_to_many"


@dataclass
class FaceDetectionResult:
    """Result of face detection operation."""
    face_locations: List[Tuple[int, int, int, int]]  # (top, right, bottom, left)
    confidence_scores: List[float]
    detection_method: FaceDetectionMethod
    processing_time: float
    image_dimensions: Tuple[int, int]  # (height, width)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FaceEncoding:
    """Face encoding data structure."""
    encoding_id: str
    encoding_vector: np.ndarray
    encoding_method: FaceEncodingMethod
    quality_score: float
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "encoding_id": self.encoding_id,
            "encoding_vector": self.encoding_vector.tolist(),
            "encoding_method": self.encoding_method.value,
            "quality_score": self.quality_score,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FaceEncoding':
        """Create from dictionary."""
        return cls(
            encoding_id=data["encoding_id"],
            encoding_vector=np.array(data["encoding_vector"]),
            encoding_method=FaceEncodingMethod(data["encoding_method"]),
            quality_score=data["quality_score"],
            created_at=datetime.fromisoformat(data["created_at"]),
            metadata=data.get("metadata", {})
        )


@dataclass
class FaceIdentity:
    """Identity information for a person."""
    identity_id: str
    name: Optional[str] = None
    encodings: List[FaceEncoding] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_seen: Optional[datetime] = None
    confidence_threshold: float = 0.6
    is_active: bool = True
    
    # Metadata
    tags: Set[str] = field(default_factory=set)
    groups: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Statistics
    recognition_count: int = 0
    false_positive_count: int = 0
    accuracy_score: float = 1.0


@dataclass
class FaceRecognitionResult:
    """Result of face recognition operation."""
    recognized_faces: List[Dict[str, Any]]
    unknown_faces: List[Dict[str, Any]]
    total_faces_detected: int
    processing_time: float
    confidence_scores: Dict[str, float]
    method_used: FaceEncodingMethod
    anti_spoofing_results: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VerificationResult:
    """Result of face verification operation."""
    is_match: bool
    confidence: float
    distance: float
    threshold_used: float
    verification_mode: VerificationMode
    processing_time: float
    anti_spoofing_passed: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


class FaceRecognitionError(Exception):
    """Custom exception for face recognition operations."""
    
    def __init__(self, message: str, error_code: Optional[str] = None, 
                 component: Optional[str] = None):
        super().__init__(message)
        self.error_code = error_code
        self.component = component
        self.timestamp = datetime.now(timezone.utc)


class FaceDetector:
    """Advanced face detection with multiple methods."""
    
    def __init__(self, method: FaceDetectionMethod = FaceDetectionMethod.HOG):
        self.method = method
        self.logger = get_logger(__name__)
        self._models = {}
        self._setup_detector()
    
    def _setup_detector(self) -> None:
        """Setup the face detector based on method."""
        try:
            if self.method == FaceDetectionMethod.OPENCV_DNN:
                # Load OpenCV DNN face detector
                net_path = Path("data/models/opencv_face_detector_uint8.pb")
                config_path = Path("data/models/opencv_face_detector.pbtxt")
                if net_path.exists() and config_path.exists():
                    self._models['opencv_net'] = cv2.dnn.readNetFromTensorflow(
                        str(net_path), str(config_path)
                    )
            
            elif self.method == FaceDetectionMethod.SSD:
                # Setup SSD MobileNet detector
                self._models['ssd_detector'] = cv2.CascadeClassifier(
                    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                )
            
        except Exception as e:
            self.logger.warning(f"Failed to setup detector {self.method}: {e}")
    
    async def detect_faces(self, image: np.ndarray) -> FaceDetectionResult:
        """Detect faces in image."""
        start_time = time.time()
        
        try:
            if self.method == FaceDetectionMethod.HOG and FACE_RECOGNITION_AVAILABLE:
                locations = face_recognition.face_locations(image, model="hog")
                confidences = [1.0] * len(locations)  # HOG doesn't provide confidence
            
            elif self.method == FaceDetectionMethod.CNN and FACE_RECOGNITION_AVAILABLE:
                locations = face_recognition.face_locations(image, model="cnn")
                confidences = [1.0] * len(locations)  # CNN doesn't provide confidence
            
            elif self.method == FaceDetectionMethod.OPENCV_DNN:
                locations, confidences = self._detect_with_opencv_dnn(image)
            
            elif self.method == FaceDetectionMethod.SSD:
                locations, confidences = self._detect_with_ssd(image)
            
            else:
                raise FaceRecognitionError(
                    f"Detection method {self.method} not available or not implemented"
                )
            
            processing_time = time.time() - start_time
            
            return FaceDetectionResult(
                face_locations=locations,
                confidence_scores=confidences,
                detection_method=self.method,
                processing_time=processing_time,
                image_dimensions=(image.shape[0], image.shape[1])
            )
            
        except Exception as e:
            raise FaceRecognitionError(f"Face detection failed: {str(e)}", "DETECTION_ERROR")
    
    def _detect_with_opencv_dnn(self, image: np.ndarray) -> Tuple[List, List]:
        """Detect faces using OpenCV DNN."""
        if 'opencv_net' not in self._models:
            raise FaceRecognitionError("OpenCV DNN model not loaded")
        
        net = self._models['opencv_net']
        h, w = image.shape[:2]
        
        # Create blob from image
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                                   (300, 300), [104, 117, 123])
        net.setInput(blob)
        detections = net.forward()
        
        locations = []
        confidences = []
        
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence > 0.5:  # Confidence threshold
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x, y, x1, y1) = box.astype("int")
                
                # Convert to face_recognition format (top, right, bottom, left)
                locations.append((y, x1, y1, x))
                confidences.append(float(confidence))
        
        return locations, confidences
    
    def _detect_with_ssd(self, image: np.ndarray) -> Tuple[List, List]:
        """Detect faces using SSD/Haar cascades."""
        if 'ssd_detector' not in self._models:
            raise FaceRecognitionError("SSD detector not loaded")
        
        detector = self._models['ssd_detector']
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        faces = detector.detectMultiScale(gray, 1.1, 4)
        
        locations = []
        confidences = []
        
        for (x, y, w, h) in faces:
            # Convert to face_recognition format (top, right, bottom, left)
            locations.append((y, x + w, y + h, x))
            confidences.append(1.0)  # Haar cascades don't provide confidence
        
        return locations, confidences


class FaceEncoder:
    """Advanced face encoding with multiple methods."""
    
    def __init__(self, method: FaceEncodingMethod = FaceEncodingMethod.DLIB):
        self.method = method
        self.logger = get_logger(__name__)
        self._models = {}
        self._setup_encoder()
    
    def _setup_encoder(self) -> None:
        """Setup the face encoder based on method."""
        try:
            if self.method == FaceEncodingMethod.DLIB and FACE_RECOGNITION_AVAILABLE:
                # face_recognition library uses dlib internally
                pass
            
            elif self.method == FaceEncodingMethod.FACENET and DEEPFACE_AVAILABLE:
                # DeepFace will handle FaceNet model loading
                pass
            
            elif self.method == FaceEncodingMethod.VGG_FACE and DEEPFACE_AVAILABLE:
                # DeepFace will handle VGG-Face model loading
                pass
            
        except Exception as e:
            self.logger.warning(f"Failed to setup encoder {self.method}: {e}")
    
    async def encode_faces(self, image: np.ndarray, 
                          face_locations: List[Tuple[int, int, int, int]]) -> List[FaceEncoding]:
        """Encode faces from image."""
        try:
            encodings = []
            
            for i, location in enumerate(face_locations):
                if self.method == FaceEncodingMethod.DLIB and FACE_RECOGNITION_AVAILABLE:
                    encoding_vector = face_recognition.face_encodings(image, [location])
                    if encoding_vector:
                        quality_score = self._calculate_quality_score(image, location)
                        
                        encoding = FaceEncoding(
                            encoding_id=self._generate_encoding_id(encoding_vector[0]),
                            encoding_vector=encoding_vector[0],
                            encoding_method=self.method,
                            quality_score=quality_score
                        )
                        encodings.append(encoding)
                
                elif self.method in [FaceEncodingMethod.FACENET, FaceEncodingMethod.VGG_FACE] and DEEPFACE_AVAILABLE:
                    # Extract face region
                    top, right, bottom, left = location
                    face_img = image[top:bottom, left:right]
                    
                    if face_img.size > 0:
                        # Use DeepFace for encoding
                        model_name = "Facenet" if self.method == FaceEncodingMethod.FACENET else "VGG-Face"
                        embedding = DeepFace.represent(face_img, model_name=model_name, enforce_detection=False)
                        
                        if embedding:
                            encoding_vector = np.array(embedding[0]["embedding"])
                            quality_score = self._calculate_quality_score(image, location)
                            
                            encoding = FaceEncoding(
                                encoding_id=self._generate_encoding_id(encoding_vector),
                                encoding_vector=encoding_vector,
                                encoding_method=self.method,
                                quality_score=quality_score
                            )
                            encodings.append(encoding)
                
                else:
                    raise FaceRecognitionError(
                        f"Encoding method {self.method} not available or not implemented"
                    )
            
            return encodings
            
        except Exception as e:
            raise FaceRecognitionError(f"Face encoding failed: {str(e)}", "ENCODING_ERROR")
    
    def _calculate_quality_score(self, image: np.ndarray, 
                               location: Tuple[int, int, int, int]) -> float:
        """Calculate quality score for face region."""
        try:
            top, right, bottom, left = location
            face_img = image[top:bottom, left:right]
            
            if face_img.size == 0:
                return 0.0
            
            # Convert to grayscale for analysis
            gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            
            # Calculate sharpness using Laplacian variance
            laplacian_var = cv2.Laplacian(gray_face, cv2.CV_64F).var()
            sharpness_score = min(laplacian_var / 1000.0, 1.0)  # Normalize
            
            # Calculate contrast
            contrast_score = gray_face.std() / 255.0
            
            # Calculate size score (larger faces are generally better)
            face_area = (bottom - top) * (right - left)
            size_score = min(face_area / (100 * 100), 1.0)  # Normalize to 100x100
            
            # Combine scores
            quality_score = (sharpness_score + contrast_score + size_score) / 3.0
            
            return min(max(quality_score, 0.0), 1.0)
            
        except Exception:
            return 0.5  # Default medium quality
    
    def _generate_encoding_id(self, encoding_vector: np.ndarray) -> str:
        """Generate unique ID for encoding."""
        # Create hash from encoding vector
        vector_bytes = encoding_vector.tobytes()
        hash_object = hashlib.sha256(vector_bytes)
        return hash_object.hexdigest()[:16]


class AntiSpoofingDetector:
    """Anti-spoofing detection to prevent face spoofing attacks."""
    
    def __init__(self, methods: List[AntiSpoofingMethod] = None):
        self.methods = methods or [AntiSpoofingMethod.LIVENESS_DETECTION]
        self.logger = get_logger(__name__)
        self._models = {}
        self._setup_detectors()
    
    def _setup_detectors(self) -> None:
        """Setup anti-spoofing detectors."""
        try:
            for method in self.methods:
                if method == AntiSpoofingMethod.LIVENESS_DETECTION:
                    # Setup basic liveness detection
                    pass
                elif method == AntiSpoofingMethod.TEXTURE_ANALYSIS:
                    # Setup texture analysis models
                    pass
        except Exception as e:
            self.logger.warning(f"Failed to setup anti-spoofing detectors: {e}")
    
    async def detect_spoofing(self, image: np.ndarray, 
                            face_locations: List[Tuple[int, int, int, int]]) -> Dict[str, Any]:
        """Detect spoofing attempts."""
        results = {
            "is_live": True,
            "confidence": 1.0,
            "methods_used": [],
            "details": {}
        }
        
        try:
            for method in self.methods:
                if method == AntiSpoofingMethod.LIVENESS_DETECTION:
                    liveness_result = await self._check_liveness(image, face_locations)
                    results["details"]["liveness"] = liveness_result
                    results["methods_used"].append(method.value)
                    
                    if not liveness_result["is_live"]:
                        results["is_live"] = False
                        results["confidence"] = min(results["confidence"], 
                                                  liveness_result["confidence"])
                
                elif method == AntiSpoofingMethod.TEXTURE_ANALYSIS:
                    texture_result = await self._analyze_texture(image, face_locations)
                    results["details"]["texture"] = texture_result
                    results["methods_used"].append(method.value)
                    
                    if not texture_result["is_real"]:
                        results["is_live"] = False
                        results["confidence"] = min(results["confidence"], 
                                                  texture_result["confidence"])
            
            return results
            
        except Exception as e:
            self.logger.error(f"Anti-spoofing detection failed: {e}")
            return {
                "is_live": False,
                "confidence": 0.0,
                "methods_used": [],
                "details": {"error": str(e)}
            }
    
    async def _check_liveness(self, image: np.ndarray, 
                            face_locations: List[Tuple[int, int, int, int]]) -> Dict[str, Any]:
        """Basic liveness detection."""
        # Simple heuristic-based liveness detection
        # In production, this would use more sophisticated methods
        
        if not face_locations:
            return {"is_live": False, "confidence": 0.0, "reason": "no_faces"}
        
        # Check for basic image properties that indicate liveness
        # This is a simplified implementation
        
        # Check image quality
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Basic thresholds (would be learned from data in production)
        if sharpness < 50:  # Too blurry, might be a photo of photo
            return {"is_live": False, "confidence": 0.3, "reason": "low_quality"}
        
        # Check for color diversity (real faces have more color variation)
        color_std = np.std(image)
        if color_std < 20:  # Too uniform, might be printed photo
            return {"is_live": False, "confidence": 0.4, "reason": "low_color_diversity"}
        
        return {"is_live": True, "confidence": 0.8, "reason": "basic_checks_passed"}
    
    async def _analyze_texture(self, image: np.ndarray, 
                             face_locations: List[Tuple[int, int, int, int]]) -> Dict[str, Any]:
        """Texture analysis for spoofing detection."""
        # Simplified texture analysis
        # In production, this would use trained models
        
        if not face_locations:
            return {"is_real": False, "confidence": 0.0, "reason": "no_faces"}
        
        # Analyze texture in face regions
        for location in face_locations:
            top, right, bottom, left = location
            face_region = image[top:bottom, left:right]
            
            if face_region.size > 0:
                # Calculate local binary patterns or other texture features
                gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
                
                # Simple texture measure using gradient magnitude
                grad_x = cv2.Sobel(gray_face, cv2.CV_64F, 1, 0, ksize=3)
                grad_y = cv2.Sobel(gray_face, cv2.CV_64F, 0, 1, ksize=3)
                grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
                
                texture_score = np.mean(grad_magnitude)
                
                # Simple threshold (would be learned from data)
                if texture_score < 10:  # Too smooth, might be fake
                    return {"is_real": False, "confidence": 0.4, "reason": "low_texture"}
        
        return {"is_real": True, "confidence": 0.7, "reason": "texture_analysis_passed"}


class FaceMatcher:
    """Advanced face matching with multiple strategies."""
    
    def __init__(self, strategy: MatchingStrategy = MatchingStrategy.EUCLIDEAN):
        self.strategy = strategy
        self.logger = get_logger(__name__)
    
    def calculate_distance(self, encoding1: np.ndarray, encoding2: np.ndarray) -> float:
        """Calculate distance between two face encodings."""
        try:
            if self.strategy == MatchingStrategy.EUCLIDEAN:
                return np.linalg.norm(encoding1 - encoding2)
            
            elif self.strategy == MatchingStrategy.COSINE:
                # Cosine similarity converted to distance
                similarity = np.dot(encoding1, encoding2) / (
                    np.linalg.norm(encoding1) * np.linalg.norm(encoding2)
                )
                return 1.0 - similarity
            
            elif self.strategy == MatchingStrategy.MANHATTAN:
                return np.sum(np.abs(encoding1 - encoding2))
            
            else:
                # Default to Euclidean
                return np.linalg.norm(encoding1 - encoding2)
                
        except Exception as e:
            self.logger.error(f"Distance calculation failed: {e}")
            return float('inf')
    
    def is_match(self, encoding1: np.ndarray, encoding2: np.ndarray, 
                threshold: float = 0.6) -> Tuple[bool, float]:
        """Check if two encodings match."""
        distance = self.calculate_distance(encoding1, encoding2)
        
        if self.strategy == MatchingStrategy.COSINE:
            # For cosine distance, smaller is better, but threshold logic is different
            is_match = distance <= threshold
        else:
            # For Euclidean and Manhattan, smaller distance means better match
            is_match = distance <= threshold
        
        # Convert distance to confidence (0-1 scale)
        if self.strategy == MatchingStrategy.COSINE:
            confidence = 1.0 - distance
        else:
            # Normalize distance to confidence
            confidence = max(0.0, 1.0 - (distance / (threshold * 2)))
        
        return is_match, confidence
    
    async def find_best_match(self, query_encoding: np.ndarray, 
                            candidate_encodings: List[Tuple[str, np.ndarray]], 
                            threshold: float = 0.6) -> Optional[Tuple[str, float, float]]:
        """Find best matching encoding from candidates."""
        best_match = None
        best_distance = float('inf')
        best_confidence = 0.0
        
        for identity_id, encoding in candidate_encodings:
            distance = self.calculate_distance(query_encoding, encoding)
            is_match, confidence = self.is_match(query_encoding, encoding, threshold)
            
            if is_match and distance < best_distance:
                best_match = identity_id
                best_distance = distance
                best_confidence = confidence
        
        if best_match:
            return best_match, best_distance, best_confidence
        
        return None


class IdentityManager:
    """Manages face identities and their encodings."""
    
    def __init__(self, database: DatabaseManager, encryption: Optional[EncryptionManager] = None):
        self.database = database
        self.encryption = encryption
        self.logger = get_logger(__name__)
        self._identities: Dict[str, FaceIdentity] = {}
        self._identity_lock = asyncio.Lock()
    
    async def initialize(self) -> None:
        """Initialize identity manager."""
        try:
            # Create tables if they don't exist
            await self._create_tables()
            
            # Load existing identities
            await self._load_identities()
            
            self.logger.info("Identity manager initialized")
            
        except Exception as e:
            raise FaceRecognitionError(f"Failed to initialize identity manager: {e}")
    
    async def _create_tables(self) -> None:
        """Create database tables for identities."""
        create_identities_sql = """
        CREATE TABLE IF NOT EXISTS face_identities (
            identity_id TEXT PRIMARY KEY,
            name TEXT,
            created_at TEXT,
            last_seen TEXT,
            confidence_threshold REAL,
            is_active BOOLEAN,
            recognition_count INTEGER,
            false_positive_count INTEGER,
            accuracy_score REAL,
            tags TEXT,
            groups_data TEXT,
            metadata TEXT
        )
        """
        
        create_encodings_sql = """
        CREATE TABLE IF NOT EXISTS face_encodings (
            encoding_id TEXT PRIMARY KEY,
            identity_id TEXT,
            encoding_vector BLOB,
            encoding_method TEXT,
            quality_score REAL,
            created_at TEXT,
            metadata TEXT,
            FOREIGN KEY (identity_id) REFERENCES face_identities (identity_id)
        )
        """
        
        await self.database.execute(create_identities_sql)
        await self.database.execute(create_encodings_sql)
    
    async def _load_identities(self) -> None:
        """Load identities from database."""
        try:
            rows = await self.database.fetch_all(
                "SELECT * FROM face_identities WHERE is_active = 1"
            )
            
            for row in rows:
                identity = await self._row_to_identity(row)
                self._identities[identity.identity_id] = identity
            
            self.logger.info(f"Loaded {len(self._identities)} identities")
            
        except Exception as e:
            self.logger.error(f"Failed to load identities: {e}")
    
    async def _row_to_identity(self, row: Dict[str, Any]) -> FaceIdentity:
        """Convert database row to FaceIdentity."""
        # Load encodings for this identity
        encoding_rows = await self.database.fetch_all(
            "SELECT * FROM face_encodings WHERE identity_id = ?",
            (row['identity_id'],)
        )
        
        encodings = []
        for enc_row in encoding_rows:
            # Decrypt encoding vector if encryption is enabled
            encoding_data = enc_row['encoding_vector']
            if self.encryption and isinstance(encoding_data, (str, bytes)):
                encoding_data = self.encryption.decrypt(encoding_data)
            
            encoding_vector = np.frombuffer(encoding_data, dtype=np.float64)
            
            encoding = FaceEncoding(
                encoding_id=enc_row['encoding_id'],
                encoding_vector=encoding_vector,
                encoding_method=FaceEncodingMethod(enc_row['encoding_method']),
                quality_score=enc_row['quality_score'],
                created_at=datetime.fromisoformat(enc_row['created_at']),
                metadata=json.loads(enc_row['metadata']) if enc_row['metadata'] else {}
            )
            encodings.append(encoding)
        
        # Parse JSON fields
        tags = set(json.loads(row['tags'])) if row['tags'] else set()
        groups = set(json.loads(row['groups_data'])) if row['groups_data'] else set()
        metadata = json.loads(row['metadata']) if row['metadata'] else {}
        
        return FaceIdentity(
            identity_id=row['identity_id'],
            name=row['name'],
            encodings=encodings,
            created_at=datetime.fromisoformat(row['created_at']),
            last_seen=datetime.fromisoformat(row['last_seen']) if row['last_seen'] else None,
            confidence_threshold=row['confidence_threshold'],
            is_active=bool(row['is_active']),
            tags=tags,
            groups=groups,
            metadata=metadata,
            recognition_count=row['recognition_count'],
            false_positive_count=row['false_positive_count'],
            accuracy_score=row['accuracy_score']
        )
    
    async def register_identity(self, name: str, encodings: List[FaceEncoding], 
                              identity_id: Optional[str] = None, 
                              **kwargs) -> str:
        """Register a new identity."""
        async with self._identity_lock:
            try:
                if identity_id is None:
                    identity_id = f"identity_{int(time.time())}_{len(self._identities)}"
                
                if identity_id in self._identities:
                    raise FaceRecognitionError(f"Identity {identity_id} already exists")
                
                # Create identity
                identity = FaceIdentity(
                    identity_id=identity_id,
                    name=name,
                    encodings=encodings,
                    **kwargs
                )
                
                # Store in database
                await self._store_identity(identity)
                
                # Cache in memory
                self._identities[identity_id] = identity
                
                self.logger.info(f"Registered identity: {identity_id} ({name})")
                return identity_id
                
            except Exception as e:
                raise FaceRecognitionError(f"Failed to register identity: {e}")
    
    async def _store_identity(self, identity: FaceIdentity) -> None:
        """Store identity in database."""
        # Store identity record
        identity_sql = """
        INSERT OR REPLACE INTO face_identities 
        (identity_id, name, created_at, last_seen, confidence_threshold, 
         is_active, recognition_count, false_positive_count, accuracy_score,
         tags, groups_data, metadata)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        await self.database.execute(
            identity_sql,
            (
                identity.identity_id,
                identity.name,
                identity.created_at.isoformat(),
                identity.last_seen.isoformat() if identity.last_seen else None,
                identity.confidence_threshold,
                identity.is_active,
                identity.recognition_count,
                identity.false_positive_count,
                identity.accuracy_score,
                json.dumps(list(identity.tags)),
                json.dumps(list(identity.groups)),
                json.dumps(identity.metadata)
            )
        )
        
        # Store encodings
        for encoding in identity.encodings:
            await self._store_encoding(encoding, identity.identity_id)
    
    async def _store_encoding(self, encoding: FaceEncoding, identity_id: str) -> None:
        """Store encoding in database."""
        # Encrypt encoding vector if encryption is enabled
        encoding_data = encoding.encoding_vector.tobytes()
        if self.encryption:
            encoding_data = self.encryption.encrypt(encoding_data)
        
        encoding_sql = """
        INSERT OR REPLACE INTO face_encodings
        (encoding_id, identity_id, encoding_vector, encoding_method, 
         quality_score, created_at, metadata)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        
        await self.database.execute(
            encoding_sql,
            (
                encoding.encoding_id,
                identity_id,
                encoding_data,
                encoding.encoding_method.value,
                encoding.quality_score,
                encoding.created_at.isoformat(),
                json.dumps(encoding.metadata)
            )
        )
    
    async def get_identity(self, identity_id: str) -> Optional[FaceIdentity]:
        """Get identity by ID."""
        return self._identities.get(identity_id)
    
    async def update_identity(self, identity_id: str, **updates) -> None:
        """Update identity information."""
        async with self._identity_lock:
            if identity_id not in self._identities:
                raise FaceRecognitionError(f"Identity {identity_id} not found")
            
            identity = self._identities[identity_id]
            
            # Update fields
            for field, value in updates.items():
                if hasattr(identity, field):
                    setattr(identity, field, value)
            
            # Update in database
            await self._store_identity(identity)
    
    async def delete_identity(self, identity_id: str) -> None:
        """Delete identity."""
        async with self._identity_lock:
            try:
                # Delete from database
                await self.database.execute(
                    "DELETE FROM face_encodings WHERE identity_id = ?",
                    (identity_id,)
                )
                await self.database.execute(
                    "DELETE FROM face_identities WHERE identity_id = ?",
                    (identity_id,)
                )
                
                # Remove from cache
                self._identities.pop(identity_id, None)
                
                self.logger.info(f"Deleted identity: {identity_id}")
                
            except Exception as e:
                raise FaceRecognitionError(f"Failed to delete identity: {e}")
    
    def list_identities(self, active_only: bool = True) -> List[str]:
        """List all identity IDs."""
        if active_only:
            return [
                identity_id for identity_id, identity in self._identities.items()
                if identity.is_active
            ]
        return list(self._identities.keys())
    
    def get_all_encodings(self) -> List[Tuple[str, np.ndarray]]:
        """Get all encodings for matching."""
        encodings = []
        for identity_id, identity in self._identities.items():
            if identity.is_active:
                for encoding in identity.encodings:
                    encodings.append((identity_id, encoding.encoding_vector))
        return encodings


class EnhancedFaceRecognition:
    """
    Advanced Face Recognition System with comprehensive functionality.
    
    Features:
    - Multiple detection and encoding methods
    - Anti-spoofing detection
    - Identity management with database persistence
    - Real-time processing capabilities
    - Performance optimization and caching
    - Integration with core AI assistant architecture
    """
    
    def __init__(self, container: Container):
        """Initialize the face recognition system."""
        self.container = container
        self.logger = get_logger(__name__)
        self.config = container.get(ConfigLoader)
        self.event_bus = container.get(EventBus)
        self.error_handler = container.get(ErrorHandler)
        self.health_check = container.get(HealthCheck)
        
        # Core components
        self._setup_core_components()
        
        # Processing components
        self._setup_processing_components()
        
        # Monitoring and caching
        self._setup_monitoring()
        self._setup_caching()
        
        # State management
        self._processing_lock = asyncio.Lock()
        self._is_initialized = False
        
        # Performance tracking
        self._processing_times = deque(maxlen=1000)
        self._error_count = 0
        self._recognition_count = 0
        
        # Configuration
        self._load_configuration()
        
        self.logger.info("Enhanced Face Recognition system initialized")
    
    def _setup_core_components(self) -> None:
        """Setup core components."""
        try:
            self.database = self.container.get(DatabaseManager)
            self.encryption = self.container.get(EncryptionManager, optional=True)
            
        except Exception as e:
            self.logger.error(f"Failed to setup core components: {e}")
            raise FaceRecognitionError("Core components setup failed")
    
    def _setup_processing_components(self) -> None:
        """Setup processing components."""
        try:
            # Get configuration
            face_config = self.config.get("face_recognition", {})
            
            # Initialize detector
            detection_method = FaceDetectionMethod(
                face_config.get("detection_method", "hog")
            )
            self.detector = FaceDetector(detection_method)
            
            # Initialize encoder
            encoding_method = FaceEncodingMethod(
                face_config.get("encoding_method", "dlib")
            )
            self.encoder = FaceEncoder(encoding_method)
            
            # Initialize anti-spoofing
            anti_spoofing_methods = [
                AntiSpoofingMethod(method) 
                for method in face_config.get("anti_spoofing_methods", ["liveness_detection"])
            ]
            self.anti_spoofing = AntiSpoofingDetector(anti_spoofing_methods)
            
            # Initialize matcher
            matching_strategy = MatchingStrategy(
                face_config.get("matching_strategy", "euclidean")
            )
            self.matcher = FaceMatcher(matching_strategy)
            
            # Initialize identity manager
            self.identity_manager = IdentityManager(self.database, self.encryption)
            
            # Thread pool for CPU-intensive operations
            max_workers = face_config.get("max_workers", 4)
            self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)
            
        except Exception as e:
            self.logger.error(f"Failed to setup processing components: {e}")
            raise FaceRecognitionError("Processing components setup failed")
    
    def _setup_monitoring(self) -> None:
        """Setup monitoring and metrics."""
        try:
            self.metrics = self.container.get(MetricsCollector, optional=True)
            self.tracer = self.container.get(TraceManager, optional=True)
            
            if self.metrics:
                # Register metrics
                self.metrics.register_counter("face_recognition_requests_total")
                self.metrics.register_counter("face_recognition_errors_total")
                self.metrics.register_histogram("face_recognition_processing_time_seconds")
                self.metrics.register_gauge("face_recognition_identities_count")
                self.metrics.register_gauge("face_recognition_encodings_count")
                
        except Exception as e:
            self.logger.warning(f"Failed to setup monitoring: {e}")
            self.metrics = None
            self.tracer = None
    
    def _setup_caching(self) -> None:
        """Setup caching layer."""
        try:
            self.cache = self.container.get(RedisCache, optional=True)
            self._cache_ttl = self.config.get("face_recognition.cache_ttl", 3600)
            
        except Exception as e:
            self.logger.warning(f"Failed to setup caching: {e}")
            self.cache = None
    
    def _load_configuration(self) -> None:
        """Load face recognition configuration."""
        face_config = self.config.get("face_recognition", {})
        
        # Detection settings
        self.confidence_threshold = face_config.get("confidence_threshold", 0.6)
        self.max_faces_per_image = face_config.get("max_faces_per_image", 10)
        self.min_face_size = face_config.get("min_face_size", 50)
        
        # Anti-spoofing settings
        self.enable_anti_spoofing = face_config.get("enable_anti_spoofing", True)
        self.anti_spoofing_threshold = face_config.get("anti_spoofing_threshold", 0.5)
        
        # Performance settings
        self.enable_gpu = face_config.get("enable_gpu", False)
        self.batch_size = face_config.get("batch_size", 1)
        self.processing_timeout = face_config.get("processing_timeout", 30.0)
    
    @handle_exceptions
    async def initialize(self) -> None:
        """Initialize the face recognition system."""
        try:
            if self._is_initialized:
                self.logger.warning("Face recognition already initialized")
                return
            
            # Initialize identity manager
            await self.identity_manager.initialize()
            
            # Register health check
            self.health_check.register_component(
                "face_recognition", 
                self._health_check_callback
            )
            
            # Emit initialization event
            await self.event_bus.emit(ComponentInitialized(
                component_id="face_recognition",
                initialization_time=0.0  # Would track actual time
            ))
            
            self._is_initialized = True
            self.logger.info("Face recognition system initialized successfully")
            
        except Exception as e:
            await self.event_bus.emit(ComponentFailed(
                component_id="face_recognition",
                error_message=str(e),
                error_type=type(e).__name__
            ))
            raise FaceRecognitionError(f"Face recognition initialization failed: {e}")
    
    @handle_exceptions
    async def detect_and_recognize_faces(self, image: np.ndarray, 
                                       user_id: Optional[str] = None,
                                       enable_anti_spoofing: Optional[bool] = None) -> FaceRecognitionResult:
        """
        Detect and recognize faces in an image.
        
        Args:
            image: Input image as numpy array
            user_id: Optional user ID for personalization
            enable_anti_spoofing: Override anti-spoofing setting
            
        Returns:
            FaceRecognitionResult with recognition results
        """
        start_time = time.time()
        
        if not self._is_initialized:
            raise FaceRecognitionError("Face recognition not initialized")
        
        async with self._processing_lock:
            try:
                with self.tracer.trace("face_recognition_processing") if self.tracer else None:
                    # Step 1: Detect faces
                    detection_result = await self.detector.detect_faces(image)
                    
                    if not detection_result.face_locations:
                        return FaceRecognitionResult(
                            recognized_faces=[],
                            unknown_faces=[],
                            total_faces_detected=0,
                            processing_time=time.time() - start_time,
                            confidence_scores={},
                            method_used=self.encoder.method
                        )
                    
                    # Limit number of faces processed
                    face_locations = detection_result.face_locations[:self.max_faces_per_image]
                    
                    # Step 2: Anti-spoofing detection
                    anti_spoofing_results = {}
                    if enable_anti_spoofing or (enable_anti_spoofing is None and self.enable_anti_spoofing):
                        anti_spoofing_results = await self.anti_spoofing.detect_spoofing(
                            image, face_locations
                        )
                        
                        if not anti_spoofing_results.get("is_live", True):
                            self.logger.warning("Spoofing attempt detected")
                            # Still continue with recognition but flag results
                    
                    # Step 3: Encode faces
                    encodings = await self.encoder.encode_faces(image, face_locations)
                    
                    # Step 4: Match against known identities
                    recognized_faces = []
                    unknown_faces = []
                    confidence_scores = {}
                    
                    known_encodings = self.identity_manager.get_all_encodings()
                    
                    for i, encoding in enumerate(encodings):
                        # Find best match
                        match_result = await self.matcher.find_best_match(
                            encoding.encoding_vector,
                            known_encodings,
                            self.confidence_threshold
                        )
                        
                        face_info = {
                            "location": face_locations[i],
                            "encoding_id": encoding.encoding_id,
                            "quality_score": encoding.quality_score
                        }
                        
                        if match_result:
                            identity_id, distance, confidence = match_result
                            identity = await self.identity_manager.get_identity(identity_id)
                            
                            face_info.update({
                                "identity_id": identity_id,
                                "name": identity.name if identity else "Unknown",
                                "confidence": confidence,
                                "distance": distance
                            })
                            
                            recognized_faces.append(face_info)
                            confidence_scores[identity_id] = confidence
                            
                            # Update identity statistics
                            if identity:
                                await self.identity_manager.update_identity(
                                    identity_id,
                                    last_seen=datetime.now(timezone.utc),
                                    recognition_count=identity.recognition_count + 1
                                )
                        else:
                            unknown_faces.append(face_info)
                    
                    processing_time = time.time() - start_time
                    
                    # Update metrics
                    self._update_metrics(processing_time, len(face_locations))
                    
                    # Create result
                    result = FaceRecognitionResult(
                        recognized_faces=recognized_faces,
                        unknown_faces=unknown_faces,
                        total_faces_detected=len(face_locations),
                        processing_time=processing_time,
                        confidence_scores=confidence_scores,
                        method_used=self.encoder.method,
                        anti_spoofing_results=anti_spoofing_results
                    )
                    
                    self._recognition_count += 1
                    self.logger.info(
                        f"Processed {len(face_locations)} faces, "
                        f"recognized {len(recognized_faces)}, "
                        f"unknown {len(unknown_faces)} in {processing_time:.2f}s"
                    )
                    
                    return result
                    
            except Exception as e:
                self._error_count += 1
                if self.metrics:
                    self.metrics.increment("face_recognition_errors_total")
                
                self.logger.error(f"Face recognition failed: {e}")
                raise FaceRecognitionError(f"Face recognition processing failed: {e}")
    
    @handle_exceptions
    async def verify_identity(self, image: np.ndarray, identity_id: str,
                            mode: VerificationMode = VerificationMode.ONE_TO_ONE) -> VerificationResult:
        """
        Verify if a face in the image matches a specific identity.
        
        Args:
            image: Input image
            identity_id: Identity to verify against
            mode: Verification mode
            
        Returns:
            VerificationResult with verification outcome
        """
        start_time = time.time()
        
        try:
            # Get identity
            identity = await self.identity_manager.get_identity(identity_id)
            if not identity:
                raise FaceRecognitionError(f"Identity {identity_id} not found")
            
            # Detect and encode faces
            detection_result = await self.detector.detect_faces(image)
            if not detection_result.face_locations:
                return VerificationResult(
                    is_match=False,
                    confidence=0.0,
                    distance=float('inf'),
                    threshold_used=self.confidence_threshold,
                    verification_mode=mode,
                    processing_time=time.time() - start_time,
                    anti_spoofing_passed=True
                )
            
            encodings = await self.encoder.encode_faces(image, detection_result.face_locations)
            
            # Anti-spoofing check
            anti_spoofing_passed = True
            if self.enable_anti_spoofing:
                anti_spoofing_results = await self.anti_spoofing.detect_spoofing(
                    image, detection_result.face_locations
                )
                anti_spoofing_passed = anti_spoofing_results.get("is_live", True)
            
            # Find best match against identity encodings
            best_match = None
            best_confidence = 0.0
            best_distance = float('inf')
            
            for face_encoding in encodings:
                for identity_encoding in identity.encodings:
                    is_match, confidence = self.matcher.is_match(
                        face_encoding.encoding_vector,
                        identity_encoding.encoding_vector,
                        identity.confidence_threshold
                    )
                    
                    distance = self.matcher.calculate_distance(
                        face_encoding.encoding_vector,
                        identity_encoding.encoding_vector
                    )
                    
                    if confidence > best_confidence:
                        best_match = is_match
                        best_confidence = confidence
                        best_distance = distance
            
            processing_time = time.time() - start_time
            
            return VerificationResult(
                is_match=best_match or False,
                confidence=best_confidence,
                distance=best_distance,
                threshold_used=identity.confidence_threshold,
                verification_mode=mode,
                processing_time=processing_time,
                anti_spoofing_passed=anti_spoofing_passed
            )
            
        except Exception as e:
            self.logger.error(f"Identity verification failed: {e}")
            raise FaceRecognitionError(f"Identity verification failed: {e}")
    
    @handle_exceptions
    async def enroll_identity(self, images: List[np.ndarray], name: str,
                            identity_id: Optional[str] = None,
                            **kwargs) -> str:
        """
        Enroll a new identity with multiple face images.
        
        Args:
            images: List of face images for enrollment
            name: Name for the identity
            identity_id: Optional specific identity ID
            **kwargs: Additional identity metadata
            
        Returns:
            Created identity ID
        """
        try:
            all_encodings = []
            
            for i, image in enumerate(images):
                # Detect faces
                detection_result = await self.detector.detect_faces(image)
                
                if not detection_result.face_locations:
                    self.logger.warning(f"No faces detected in enrollment image {i}")
                    continue
                
                # Take the largest face (assuming it's the primary subject)
                largest_face_idx = 0
                largest_area = 0
                
                for j, location in enumerate(detection_result.face_locations):
                    top, right, bottom, left = location
                    area = (bottom - top) * (right - left)
                    if area > largest_area:
                        largest_area = area
                        largest_face_idx = j
                
                largest_face = [detection_result.face_locations[largest_face_idx]]
                
                # Encode the largest face
                encodings = await self.encoder.encode_faces(image, largest_face)
                if encodings:
                    all_encodings.extend(encodings)
            
            if not all_encodings:
                raise FaceRecognitionError("No valid face encodings generated for enrollment")
            
            # Filter by quality score
            quality_threshold = 0.3
            good_encodings = [
                enc for enc in all_encodings 
                if enc.quality_score >= quality_threshold
            ]
            
            if not good_encodings:
                self.logger.warning("No high-quality encodings, using all available")
                good_encodings = all_encodings
            
            # Register identity
            identity_id = await self.identity_manager.register_identity(
                name=name,
                encodings=good_encodings,
                identity_id=identity_id,
                **kwargs
            )
            
            # Update metrics
            if self.metrics:
                self.metrics.set("face_recognition_identities_count", 
                               len(self.identity_manager.list_identities()))
            
            self.logger.info(f"Enrolled identity {identity_id} with {len(good_encodings)} encodings")
            return identity_id
            
        except Exception as e:
            self.logger.error(f"Identity enrollment failed: {e}")
            raise FaceRecognitionError(f"Identity enrollment failed: {e}")
    
    def _update_metrics(self, processing_time: float, faces_count: int) -> None:
        """Update performance metrics."""
        self._processing_times.append(processing_time)
        
        if self.metrics:
            self.metrics.increment("face_recognition_requests_total")
            self.metrics.record("face_recognition_processing_time_seconds", processing_time)
            self.metrics.set("face_recognition_identities_count", 
                           len(self.identity_manager.list_identities()))
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get face recognition system status."""
        try:
            identities_count = len(self.identity_manager.list_identities())
            total_encodings = sum(
                len(identity.encodings) 
                for identity in self.identity_manager._identities.values()
            )
            
            avg_processing_time = (
                sum(self._processing_times) / len(self._processing_times)
                if self._processing_times else 0.0
            )
            
            return {
                "is_initialized": self._is_initialized,
                "identities_count": identities_count,
                "total_encodings": total_encodings,
                "recognition_count": self._recognition_count,
                "error_count": self._error_count,
                "average_processing_time": avg_processing_time,
                "detection_method": self.detector.method.value,
                "encoding_method": self.encoder.method.value,
                "anti_spoofing_enabled": self.enable_anti_spoofing,
                "configuration": {
                    "confidence_threshold": self.confidence_threshold,
                    "max_faces_per_image": self.max_faces_per_image,
                    "min_face_size": self.min_face_size,
                    "processing_timeout": self.processing_timeout
                }
            }
            
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    async def _health_check_callback(self) -> Dict[str, Any]:
        """Health check callback for monitoring."""
        try:
            # Test basic functionality
            test_image = np.zeros((100, 100, 3), dtype=np.uint8)
            
            start_time = time.time()
            await self.detector.detect_faces(test_image)
            detection_time = time.time() - start_time
            
            # Calculate health score
            error_rate = self._error_count / max(self._recognition_count, 1)
            health_score = max(0.0, 1.0 - error_rate)
            
            if detection_time > 5.0:  # Detection taking too long
                health_score *= 0.5
            
            status = "healthy" if health_score >= 0.8 else "degraded" if health_score >= 0.5 else "unhealthy"
            
            return {
                "status": status,
                "health_score": health_score,
                "detection_time": detection_time,
                "error_rate": error_rate,
                "recognition_count": self._recognition_count,
                "identities_registered": len(self.identity_manager.list_identities()),
                "is_initialized": self._is_initialized
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "health_score": 0.0
            }
    
    async def cleanup(self) -> None:
        """Cleanup face recognition system."""
        try:
            # Shutdown thread pool
            if hasattr(self, 'thread_pool'):
                self.thread_pool.shutdown(wait=True)
            
            self.logger.info("Face recognition system cleaned up")
            
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            if hasattr(self, 'thread_pool'):
                self.thread_pool.shutdown(wait=False)
        except Exception:
            pass
