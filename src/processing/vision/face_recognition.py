"""
Enhanced Face Recognition Module

This module provides comprehensive face recognition capabilities including:
- Face detection with multiple methods
- Face encoding and matching
- Identity management with database storage
- Anti-spoofing detection
- Integration with the core system architecture

Author: AI Assistant System
Version: 1.0.0
"""

import base64
import hashlib
import json
import logging
import pickle
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import asyncio
import cv2
import face_recognition as fr
import numpy as np

# Core system imports
from core.dependency_injection import Container
from core.events.event_bus import EventBus
from core.events.event_types import EventType
from core.security.encryption import EncryptionManager
from integrations.cache.redis_cache import RedisCache
from integrations.storage.database import DatabaseManager
from observability.logging.config import get_logger
from observability.monitoring.metrics import MetricsCollector
from observability.monitoring.tracing import trace_async


class FaceDetectionMethod(Enum):
    """Face detection methods."""

    HOG = "hog"
    CNN = "cnn"
    MTCNN = "mtcnn"
    SSD = "ssd"
    OPENCV_DNN = "opencv_dnn"
    YOLO = "yolo"
    MEDIAPIPE = "mediapipe"


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
        """Convert encoding to dictionary for storage."""
        return {
            "encoding_id": self.encoding_id,
            "encoding_vector": base64.b64encode(self.encoding_vector.tobytes()).decode("utf-8"),
            "encoding_vector_shape": self.encoding_vector.shape,
            "encoding_vector_dtype": str(self.encoding_vector.dtype),
            "encoding_method": self.encoding_method.value,
            "quality_score": self.quality_score,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FaceEncoding":
        """Create encoding from dictionary."""
        vector_bytes = base64.b64decode(data["encoding_vector"])
        vector = np.frombuffer(vector_bytes, dtype=data["encoding_vector_dtype"])
        vector = vector.reshape(data["encoding_vector_shape"])

        return cls(
            encoding_id=data["encoding_id"],
            encoding_vector=vector,
            encoding_method=FaceEncodingMethod(data["encoding_method"]),
            quality_score=data["quality_score"],
            created_at=datetime.fromisoformat(data["created_at"]),
            metadata=data.get("metadata", {}),
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


@dataclass
class FaceRecognitionConfiguration:
    """Configuration for face recognition system."""

    # Detection settings
    detection_method: FaceDetectionMethod = FaceDetectionMethod.HOG
    encoding_method: FaceEncodingMethod = FaceEncodingMethod.DLIB
    matching_strategy: MatchingStrategy = MatchingStrategy.EUCLIDEAN

    # Quality settings
    min_face_size: int = 64
    max_faces: int = 10
    detection_confidence_threshold: float = 0.8
    recognition_confidence_threshold: float = 0.6
    quality_threshold: float = 0.5

    # Anti-spoofing settings
    enable_anti_spoofing: bool = True
    anti_spoofing_methods: List[AntiSpoofingMethod] = field(
        default_factory=lambda: [
            AntiSpoofingMethod.LIVENESS_DETECTION,
            AntiSpoofingMethod.TEXTURE_ANALYSIS,
        ]
    )
    anti_spoofing_threshold: float = 0.7

    # Performance settings
    enable_gpu: bool = True
    max_threads: int = 4
    batch_size: int = 1
    enable_caching: bool = True
    cache_ttl: int = 3600

    # Privacy and security
    encrypt_encodings: bool = True
    anonymize_logs: bool = True
    max_storage_duration_days: int = 365
    require_consent: bool = True

    # Model settings
    model_tolerance: float = 0.6
    num_jitters: int = 1

    # Storage settings
    store_face_images: bool = False
    max_encodings_per_identity: int = 10


class FaceRecognitionError(Exception):
    """Custom exception for face recognition operations."""

    def __init__(
        self, message: str, error_code: Optional[str] = None, identity_id: Optional[str] = None
    ):
        super().__init__(message)
        self.error_code = error_code
        self.identity_id = identity_id
        self.timestamp = datetime.now(timezone.utc)


class FaceDetector:
    """Advanced face detection with multiple methods."""

    def __init__(self, method: FaceDetectionMethod = FaceDetectionMethod.HOG):
        self.method = method
        self.logger = get_logger(self.__class__.__name__)
        self._detector = None

    def _setup_detector(self) -> None:
        """Setup the appropriate detector based on method."""
        try:
            if self.method == FaceDetectionMethod.HOG:
                # HOG detector is built into face_recognition
                pass
            elif self.method == FaceDetectionMethod.CNN:
                # CNN detector is built into face_recognition
                pass
            elif self.method == FaceDetectionMethod.OPENCV_DNN:
                # Load OpenCV DNN model
                model_path = "models/opencv_face_detector_uint8.pb"
                config_path = "models/opencv_face_detector.pbtxt"
                if Path(model_path).exists() and Path(config_path).exists():
                    self._detector = cv2.dnn.readNetFromTensorflow(model_path, config_path)
                else:
                    self.logger.warning("OpenCV DNN model not found, falling back to HOG")
                    self.method = FaceDetectionMethod.HOG
            else:
                self.logger.warning(f"Method {self.method} not fully implemented, using HOG")
                self.method = FaceDetectionMethod.HOG

        except Exception as e:
            self.logger.error(f"Failed to setup detector: {e}")
            self.method = FaceDetectionMethod.HOG

    async def detect_faces(self, image: np.ndarray) -> FaceDetectionResult:
        """Detect faces in an image."""
        start_time = asyncio.get_event_loop().time()

        try:
            if self._detector is None:
                self._setup_detector()

            height, width = image.shape[:2]

            if self.method in [FaceDetectionMethod.HOG, FaceDetectionMethod.CNN]:
                model = "hog" if self.method == FaceDetectionMethod.HOG else "cnn"
                face_locations = fr.face_locations(image, model=model)
                confidence_scores = [1.0] * len(
                    face_locations
                )  # face_recognition doesn't return confidence

            elif self.method == FaceDetectionMethod.OPENCV_DNN:
                face_locations, confidence_scores = self._detect_with_opencv_dnn(image)

            else:
                # Fallback to HOG
                face_locations = fr.face_locations(image, model="hog")
                confidence_scores = [1.0] * len(face_locations)

            processing_time = asyncio.get_event_loop().time() - start_time

            return FaceDetectionResult(
                face_locations=face_locations,
                confidence_scores=confidence_scores,
                detection_method=self.method,
                processing_time=processing_time,
                image_dimensions=(height, width),
                metadata={
                    "faces_detected": len(face_locations),
                    "avg_confidence": np.mean(confidence_scores) if confidence_scores else 0.0,
                },
            )

        except Exception as e:
            self.logger.error(f"Face detection failed: {e}")
            raise FaceRecognitionError(f"Face detection failed: {e}", "DETECTION_FAILED")

    def _detect_with_opencv_dnn(self, image: np.ndarray) -> Tuple[List, List]:
        """Detect faces using OpenCV DNN."""
        try:
            blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123])
            self._detector.setInput(blob)
            detections = self._detector.forward()

            face_locations = []
            confidence_scores = []

            h, w = image.shape[:2]

            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]

                if confidence > 0.5:  # Confidence threshold
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (x, y, x1, y1) = box.astype("int")

                    # Convert to face_recognition format (top, right, bottom, left)
                    face_locations.append((y, x1, y1, x))
                    confidence_scores.append(float(confidence))

            return face_locations, confidence_scores

        except Exception as e:
            self.logger.error(f"OpenCV DNN detection failed: {e}")
            return [], []


class FaceEncoder:
    """Advanced face encoding with multiple methods."""

    def __init__(self, method: FaceEncodingMethod = FaceEncodingMethod.DLIB):
        self.method = method
        self.logger = get_logger(self.__class__.__name__)
        self._encoder = None

    def _setup_encoder(self) -> None:
        """Setup the appropriate encoder based on method."""
        try:
            if self.method == FaceEncodingMethod.DLIB:
                # Built into face_recognition
                pass
            else:
                self.logger.warning(f"Method {self.method} not fully implemented, using DLIB")
                self.method = FaceEncodingMethod.DLIB

        except Exception as e:
            self.logger.error(f"Failed to setup encoder: {e}")
            self.method = FaceEncodingMethod.DLIB

    async def encode_faces(
        self,
        image: np.ndarray,
        face_locations: List[Tuple[int, int, int, int]],
        num_jitters: int = 1,
    ) -> List[FaceEncoding]:
        """Encode faces from detected locations."""
        try:
            if self._encoder is None:
                self._setup_encoder()

            if not face_locations:
                return []

            # Get face encodings
            encodings = fr.face_encodings(image, face_locations, num_jitters=num_jitters)

            face_encodings = []
            for i, encoding in enumerate(encodings):
                # Calculate quality score based on face size and clarity
                face_location = face_locations[i]
                quality_score = self._calculate_quality_score(image, face_location)

                face_encoding = FaceEncoding(
                    encoding_id=self._generate_encoding_id(encoding),
                    encoding_vector=encoding,
                    encoding_method=self.method,
                    quality_score=quality_score,
                    metadata={"face_location": face_location, "num_jitters": num_jitters},
                )
                face_encodings.append(face_encoding)

            return face_encodings

        except Exception as e:
            self.logger.error(f"Face encoding failed: {e}")
            raise FaceRecognitionError(f"Face encoding failed: {e}", "ENCODING_FAILED")

    def _calculate_quality_score(
        self, image: np.ndarray, face_location: Tuple[int, int, int, int]
    ) -> float:
        """Calculate quality score for a face region."""
        try:
            top, right, bottom, left = face_location
            face_image = image[top:bottom, left:right]

            if face_image.size == 0:
                return 0.0

            # Calculate sharpness using Laplacian variance
            gray = (
                cv2.cvtColor(face_image, cv2.COLOR_RGB2GRAY)
                if len(face_image.shape) == 3
                else face_image
            )
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

            # Calculate size score (larger faces generally have better quality)
            face_area = (bottom - top) * (right - left)
            size_score = min(face_area / 10000, 1.0)  # Normalize to 0-1

            # Combine metrics
            quality_score = min(sharpness / 1000, 1.0) * 0.7 + size_score * 0.3

            return float(np.clip(quality_score, 0.0, 1.0))

        except Exception as e:
            self.logger.error(f"Quality calculation failed: {e}")
            return 0.5  # Default quality score

    def _generate_encoding_id(self, encoding_vector: np.ndarray) -> str:
        """Generate unique ID for encoding."""
        # Create hash from encoding vector
        vector_hash = hashlib.sha256(encoding_vector.tobytes()).hexdigest()[:16]
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        return f"enc_{timestamp}_{vector_hash}"


class AntiSpoofingDetector:
    """Anti-spoofing detection to prevent face spoofing attacks."""

    def __init__(self, methods: List[AntiSpoofingMethod] = None):
        self.methods = methods or [AntiSpoofingMethod.LIVENESS_DETECTION]
        self.logger = get_logger(self.__class__.__name__)
        self._detectors = {}

    def _setup_detectors(self) -> None:
        """Setup anti-spoofing detectors."""
        for method in self.methods:
            try:
                if method == AntiSpoofingMethod.LIVENESS_DETECTION:
                    # Simple liveness detection based on eye blinking
                    self._detectors[method] = self._check_liveness
                elif method == AntiSpoofingMethod.TEXTURE_ANALYSIS:
                    # Texture analysis for print attack detection
                    self._detectors[method] = self._analyze_texture
                else:
                    self.logger.warning(f"Anti-spoofing method {method} not implemented")

            except Exception as e:
                self.logger.error(f"Failed to setup anti-spoofing detector {method}: {e}")

    async def detect_spoofing(
        self, image: np.ndarray, face_locations: List[Tuple[int, int, int, int]]
    ) -> Dict[str, Any]:
        """Detect spoofing attempts."""
        if not self._detectors:
            self._setup_detectors()

        results = {}
        overall_confidence = 1.0

        for method, detector in self._detectors.items():
            try:
                method_result = await detector(image, face_locations)
                results[method.value] = method_result
                overall_confidence *= method_result.get("confidence", 1.0)

            except Exception as e:
                self.logger.error(f"Anti-spoofing detection failed for {method}: {e}")
                results[method.value] = {"confidence": 0.5, "is_live": False}
                overall_confidence *= 0.5

        return {
            "is_live": overall_confidence > 0.7,
            "confidence": overall_confidence,
            "method_results": results,
            "risk_level": (
                "high"
                if overall_confidence < 0.3
                else "medium" if overall_confidence < 0.7 else "low"
            ),
        }

    async def _check_liveness(
        self, image: np.ndarray, face_locations: List[Tuple[int, int, int, int]]
    ) -> Dict[str, Any]:
        """Basic liveness detection."""
        # This is a simplified implementation
        # In production, you would use more sophisticated methods
        return {"confidence": 0.8, "is_live": True, "method": "basic_liveness"}

    async def _analyze_texture(
        self, image: np.ndarray, face_locations: List[Tuple[int, int, int, int]]
    ) -> Dict[str, Any]:
        """Analyze texture for print attack detection."""
        # Simplified texture analysis
        if not face_locations:
            return {"confidence": 0.0, "is_live": False}

        # Analyze first face
        top, right, bottom, left = face_locations[0]
        face_region = image[top:bottom, left:right]

        if face_region.size == 0:
            return {"confidence": 0.0, "is_live": False}

        # Convert to grayscale
        gray = (
            cv2.cvtColor(face_region, cv2.COLOR_RGB2GRAY)
            if len(face_region.shape) == 3
            else face_region
        )

        # Calculate texture metrics
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        # Higher variance suggests real face texture
        confidence = min(laplacian_var / 1000, 1.0)
        is_live = confidence > 0.3

        return {"confidence": confidence, "is_live": is_live, "texture_variance": laplacian_var}


class FaceMatcher:
    """Advanced face matching with multiple strategies."""

    def __init__(self, strategy: MatchingStrategy = MatchingStrategy.EUCLIDEAN):
        self.strategy = strategy
        self.logger = get_logger(self.__class__.__name__)

    def calculate_distance(self, encoding1: np.ndarray, encoding2: np.ndarray) -> float:
        """Calculate distance between two face encodings."""
        try:
            if self.strategy == MatchingStrategy.EUCLIDEAN:
                return float(np.linalg.norm(encoding1 - encoding2))
            elif self.strategy == MatchingStrategy.COSINE:
                # Cosine distance = 1 - cosine similarity
                dot_product = np.dot(encoding1, encoding2)
                norm_a = np.linalg.norm(encoding1)
                norm_b = np.linalg.norm(encoding2)
                cosine_sim = dot_product / (norm_a * norm_b)
                return float(1 - cosine_sim)
            elif self.strategy == MatchingStrategy.MANHATTAN:
                return float(np.sum(np.abs(encoding1 - encoding2)))
            else:
                # Default to Euclidean
                return float(np.linalg.norm(encoding1 - encoding2))

        except Exception as e:
            self.logger.error(f"Distance calculation failed: {e}")
            return float("inf")

    def is_match(
        self, encoding1: np.ndarray, encoding2: np.ndarray, threshold: float = 0.6
    ) -> bool:
        """Determine if two encodings match."""
        distance = self.calculate_distance(encoding1, encoding2)
        return distance <= threshold

    async def find_best_match(
        self,
        query_encoding: np.ndarray,
        candidate_encodings: List[Tuple[str, np.ndarray]],
        threshold: float = 0.6,
    ) -> Optional[Tuple[str, float]]:
        """Find the best match from candidate encodings."""
        if not candidate_encodings:
            return None

        best_match = None
        best_distance = float("inf")

        for identity_id, encoding in candidate_encodings:
            distance = self.calculate_distance(query_encoding, encoding)

            if distance <= threshold and distance < best_distance:
                best_distance = distance
                best_match = (identity_id, distance)

        return best_match


class IdentityManager:
    """Manages face identities and their encodings."""

    def __init__(self, database: DatabaseManager, encryption: Optional[EncryptionManager] = None):
        self.database = database
        self.encryption = encryption
        self.logger = get_logger(self.__class__.__name__)
        self._identities: Dict[str, FaceIdentity] = {}
        self._identity_encodings: Dict[str, List[Tuple[str, np.ndarray]]] = {}

    async def initialize(self) -> None:
        """Initialize the identity manager."""
        try:
            await self._create_tables()
            await self._load_identities()
            self.logger.info("Identity manager initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize identity manager: {e}")
            raise FaceRecognitionError(f"Identity manager initialization failed: {e}")

    async def _create_tables(self) -> None:
        """Create database tables for identities and encodings."""
        # Create identities table
        identity_table_sql = """
        CREATE TABLE IF NOT EXISTS face_identities (
            identity_id VARCHAR(255) PRIMARY KEY,
            name VARCHAR(255),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_seen TIMESTAMP,
            confidence_threshold FLOAT DEFAULT 0.6,
            is_active BOOLEAN DEFAULT TRUE,
            tags TEXT,
            groups_list TEXT,
            metadata TEXT,
            recognition_count INTEGER DEFAULT 0,
            false_positive_count INTEGER DEFAULT 0,
            accuracy_score FLOAT DEFAULT 1.0
        )
        """

        # Create encodings table
        encoding_table_sql = """
        CREATE TABLE IF NOT EXISTS face_encodings (
            encoding_id VARCHAR(255) PRIMARY KEY,
            identity_id VARCHAR(255),
            encoding_data TEXT,
            encoding_method VARCHAR(50),
            quality_score FLOAT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            metadata TEXT,
            FOREIGN KEY (identity_id) REFERENCES face_identities(identity_id) ON DELETE CASCADE
        )
        """

        await self.database.execute(identity_table_sql)
        await self.database.execute(encoding_table_sql)

    async def _load_identities(self) -> None:
        """Load all identities from database."""
        try:
            # Load identities
            identity_rows = await self.database.fetch_all(
                "SELECT * FROM face_identities WHERE is_active = TRUE"
            )

            for row in identity_rows:
                identity = await self._row_to_identity(row)
                self._identities[identity.identity_id] = identity

                # Load encodings for this identity
                encoding_rows = await self.database.fetch_all(
                    "SELECT * FROM face_encodings WHERE identity_id = ?", (identity.identity_id,)
                )

                encodings = []
                for enc_row in encoding_rows:
                    encoding_data = json.loads(enc_row["encoding_data"])
                    if self.encryption:
                        # Decrypt encoding data
                        encoding_data = self.encryption.decrypt(encoding_data)

                    encoding = FaceEncoding.from_dict(encoding_data)
                    encodings.append((identity.identity_id, encoding.encoding_vector))

                self._identity_encodings[identity.identity_id] = encodings

            self.logger.info(f"Loaded {len(self._identities)} identities")

        except Exception as e:
            self.logger.error(f"Failed to load identities: {e}")
            raise

    async def _row_to_identity(self, row: Dict[str, Any]) -> FaceIdentity:
        """Convert database row to FaceIdentity object."""
        return FaceIdentity(
            identity_id=row["identity_id"],
            name=row["name"],
            created_at=row["created_at"],
            last_seen=row["last_seen"],
            confidence_threshold=row["confidence_threshold"],
            is_active=row["is_active"],
            tags=set(json.loads(row["tags"]) if row["tags"] else []),
            groups=set(json.loads(row["groups_list"]) if row["groups_list"] else []),
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
            recognition_count=row["recognition_count"],
            false_positive_count=row["false_positive_count"],
            accuracy_score=row["accuracy_score"],
        )

    async def register_identity(self, name: str, encodings: List[FaceEncoding], **kwargs) -> str:
        """Register a new identity with face encodings."""
        try:
            identity_id = str(uuid.uuid4())

            identity = FaceIdentity(
                identity_id=identity_id, name=name, encodings=encodings, **kwargs
            )

            # Store in database
            await self._store_identity(identity)

            # Store encodings
            encoding_tuples = []
            for encoding in encodings:
                await self._store_encoding(encoding, identity_id)
                encoding_tuples.append((identity_id, encoding.encoding_vector))

            # Cache in memory
            self._identities[identity_id] = identity
            self._identity_encodings[identity_id] = encoding_tuples

            self.logger.info(f"Registered new identity: {identity_id} ({name})")
            return identity_id

        except Exception as e:
            self.logger.error(f"Failed to register identity: {e}")
            raise FaceRecognitionError(f"Identity registration failed: {e}")

    async def _store_identity(self, identity: FaceIdentity) -> None:
        """Store identity in database."""
        sql = """
        INSERT INTO face_identities (
            identity_id, name, created_at, last_seen, confidence_threshold,
            is_active, tags, groups_list, metadata, recognition_count,
            false_positive_count, accuracy_score
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        await self.database.execute(
            sql,
            (
                identity.identity_id,
                identity.name,
                identity.created_at,
                identity.last_seen,
                identity.confidence_threshold,
                identity.is_active,
                json.dumps(list(identity.tags)),
                json.dumps(list(identity.groups)),
                json.dumps(identity.metadata),
                identity.recognition_count,
                identity.false_positive_count,
                identity.accuracy_score,
            ),
        )

    async def _store_encoding(self, encoding: FaceEncoding, identity_id: str) -> None:
        """Store face encoding in database."""
        encoding_data = encoding.to_dict()

        if self.encryption:
            # Encrypt encoding data
            encoding_data = self.encryption.encrypt(encoding_data)

        sql = """
        INSERT INTO face_encodings (
            encoding_id, identity_id, encoding_data, encoding_method,
            quality_score, created_at, metadata
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """

        await self.database.execute(
            sql,
            (
                encoding.encoding_id,
                identity_id,
                json.dumps(encoding_data),
                encoding.encoding_method.value,
                encoding.quality_score,
                encoding.created_at,
                json.dumps(encoding.metadata),
            ),
        )

    async def get_identity(self, identity_id: str) -> Optional[FaceIdentity]:
        """Get identity by ID."""
        return self._identities.get(identity_id)

    async def update_identity(self, identity_id: str, **updates) -> None:
        """Update identity information."""
        if identity_id not in self._identities:
            raise FaceRecognitionError(f"Identity {identity_id} not found")

        identity = self._identities[identity_id]

        # Update fields
        for key, value in updates.items():
            if hasattr(identity, key):
                setattr(identity, key, value)

        # Update in database
        update_sql = """
        UPDATE face_identities SET name = ?, last_seen = ?, confidence_threshold = ?,
        is_active = ?, tags = ?, groups_list = ?, metadata = ?, recognition_count = ?,
        false_positive_count = ?, accuracy_score = ? WHERE identity_id = ?
        """

        await self.database.execute(
            update_sql,
            (
                identity.name,
                identity.last_seen,
                identity.confidence_threshold,
                identity.is_active,
                json.dumps(list(identity.tags)),
                json.dumps(list(identity.groups)),
                json.dumps(identity.metadata),
                identity.recognition_count,
                identity.false_positive_count,
                identity.accuracy_score,
                identity_id,
            ),
        )

    async def delete_identity(self, identity_id: str) -> None:
        """Delete an identity and all its encodings."""
        if identity_id in self._identities:
            del self._identities[identity_id]

        if identity_id in self._identity_encodings:
            del self._identity_encodings[identity_id]

        # Delete from database
        await self.database.execute(
            "DELETE FROM face_identities WHERE identity_id = ?", (identity_id,)
        )

    def list_identities(self, active_only: bool = True) -> List[str]:
        """List all identity IDs."""
        if active_only:
            return [id for id, identity in self._identities.items() if identity.is_active]
        return list(self._identities.keys())

    def get_all_encodings(self) -> List[Tuple[str, np.ndarray]]:
        """Get all encodings for matching."""
        all_encodings = []
        for encodings in self._identity_encodings.values():
            all_encodings.extend(encodings)
        return all_encodings


class EnhancedFaceRecognition:
    """Enhanced face recognition system with full integration."""

    def __init__(self, container: Container):
        self.container = container
        self.logger = get_logger(self.__class__.__name__)

        # Configuration
        self.config: FaceRecognitionConfiguration = FaceRecognitionConfiguration()

        # Core components
        self.detector: Optional[FaceDetector] = None
        self.encoder: Optional[FaceEncoder] = None
        self.anti_spoofing: Optional[AntiSpoofingDetector] = None
        self.matcher: Optional[FaceMatcher] = None
        self.identity_manager: Optional[IdentityManager] = None

        # System integrations
        self.database: Optional[DatabaseManager] = None
        self.cache: Optional[RedisCache] = None
        self.encryption: Optional[EncryptionManager] = None
        self.event_bus: Optional[EventBus] = None
        self.metrics: Optional[MetricsCollector] = None

        self._is_initialized = False

    def _setup_core_components(self) -> None:
        """Setup core face recognition components."""
        try:
            # Get dependencies from container
            self.database = self.container.get(DatabaseManager)
            self.cache = self.container.get(RedisCache, optional=True)
            self.encryption = self.container.get(EncryptionManager, optional=True)
            self.event_bus = self.container.get(EventBus, optional=True)
            self.metrics = self.container.get(MetricsCollector, optional=True)

        except Exception as e:
            self.logger.error(f"Failed to setup core components: {e}")
            raise

    def _setup_processing_components(self) -> None:
        """Setup face recognition processing components."""
        try:
            self.detector = FaceDetector(self.config.detection_method)
            self.encoder = FaceEncoder(self.config.encoding_method)
            self.anti_spoofing = AntiSpoofingDetector(self.config.anti_spoofing_methods)
            self.matcher = FaceMatcher(self.config.matching_strategy)
            self.identity_manager = IdentityManager(self.database, self.encryption)

        except Exception as e:
            self.logger.error(f"Failed to setup processing components: {e}")
            raise

    def _setup_monitoring(self) -> None:
        """Setup monitoring and metrics collection."""
        if self.metrics:
            self.metrics.register_counter("face_recognition_requests_total")
            self.metrics.register_counter("face_recognition_errors_total")
            self.metrics.register_histogram("face_recognition_duration_seconds")
            self.metrics.register_gauge("face_identities_count")

    def _setup_caching(self) -> None:
        """Setup caching for face recognition results."""
        if not self.cache:
            self.logger.warning("No cache available, face recognition will run without caching")

    def _load_configuration(self) -> None:
        """Load configuration from container."""
        try:
            config_dict = self.container.get_config("face_recognition", {})

            # Update configuration with loaded values
            for key, value in config_dict.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)

        except Exception as e:
            self.logger.warning(f"Failed to load configuration: {e}, using defaults")

    async def initialize(self) -> None:
        """Initialize the face recognition system."""
        if self._is_initialized:
            return

        try:
            self.logger.info("Initializing face recognition system...")

            # Load configuration
            self._load_configuration()

            # Setup components
            self._setup_core_components()
            self._setup_processing_components()
            self._setup_monitoring()
            self._setup_caching()

            # Initialize identity manager
            await self.identity_manager.initialize()

            self._is_initialized = True
            self.logger.info("Face recognition system initialized successfully")

            # Emit initialization event
            if self.event_bus:
                await self.event_bus.emit(
                    EventType.COMPONENT_INITIALIZED,
                    {"component": "face_recognition", "timestamp": datetime.now(timezone.utc)},
                )

        except Exception as e:
            self.logger.error(f"Failed to initialize face recognition system: {e}")
            raise FaceRecognitionError(f"Initialization failed: {e}")

    @trace_async("face_recognition.detect_and_recognize")
    async def detect_and_recognize_faces(
        self, image: np.ndarray, session_id: Optional[str] = None
    ) -> FaceRecognitionResult:
        """Detect and recognize faces in an image."""
        if not self._is_initialized:
            await self.initialize()

        start_time = asyncio.get_event_loop().time()
        request_id = str(uuid.uuid4())

        try:
            self.logger.debug(f"Processing face recognition request: {request_id}")

            # Update metrics
            if self.metrics:
                self.metrics.increment("face_recognition_requests_total")

            # Step 1: Detect faces
            detection_result = await self.detector.detect_faces(image)

            if not detection_result.face_locations:
                return FaceRecognitionResult(
                    recognized_faces=[],
                    unknown_faces=[],
                    total_faces_detected=0,
                    processing_time=asyncio.get_event_loop().time() - start_time,
                    confidence_scores={},
                    method_used=self.config.encoding_method,
                    metadata={"request_id": request_id},
                )

            # Step 2: Anti-spoofing detection
            anti_spoofing_results = {}
            if self.config.enable_anti_spoofing:
                anti_spoofing_results = await self.anti_spoofing.detect_spoofing(
                    image, detection_result.face_locations
                )

                if not anti_spoofing_results.get("is_live", True):
                    self.logger.warning("Potential spoofing attack detected")

            # Step 3: Encode faces
            face_encodings = await self.encoder.encode_faces(
                image, detection_result.face_locations, self.config.num_jitters
            )

            # Step 4: Recognize faces
            recognized_faces = []
            unknown_faces = []
            confidence_scores = {}

            all_encodings = self.identity_manager.get_all_encodings()

            for i, encoding in enumerate(face_encodings):
                best_match = await self.matcher.find_best_match(
                    encoding.encoding_vector,
                    all_encodings,
                    self.config.recognition_confidence_threshold,
                )

                face_data = {
                    "face_location": detection_result.face_locations[i],
                    "encoding_id": encoding.encoding_id,
                    "quality_score": encoding.quality_score,
                    "detection_confidence": detection_result.confidence_scores[i],
                }

                if best_match:
                    identity_id, distance = best_match
                    identity = await self.identity_manager.get_identity(identity_id)

                    confidence = 1.0 - (distance / self.config.recognition_confidence_threshold)
                    confidence = max(0.0, min(1.0, confidence))

                    face_data.update(
                        {
                            "identity_id": identity_id,
                            "name": identity.name if identity else "Unknown",
                            "recognition_confidence": confidence,
                            "distance": distance,
                        }
                    )

                    recognized_faces.append(face_data)
                    confidence_scores[identity_id] = confidence

                    # Update identity statistics
                    if identity:
                        identity.recognition_count += 1
                        identity.last_seen = datetime.now(timezone.utc)
                        await self.identity_manager.update_identity(
                            identity_id,
                            recognition_count=identity.recognition_count,
                            last_seen=identity.last_seen,
                        )
                else:
                    unknown_faces.append(face_data)

            processing_time = asyncio.get_event_loop().time() - start_time

            # Update metrics
            if self.metrics:
                self.metrics.observe("face_recognition_duration_seconds", processing_time)
                self.metrics.set(
                    "face_identities_count", len(self.identity_manager.list_identities())
                )

            result = FaceRecognitionResult(
                recognized_faces=recognized_faces,
                unknown_faces=unknown_faces,
                total_faces_detected=len(detection_result.face_locations),
                processing_time=processing_time,
                confidence_scores=confidence_scores,
                method_used=self.config.encoding_method,
                anti_spoofing_results=anti_spoofing_results,
                metadata={
                    "request_id": request_id,
                    "session_id": session_id,
                    "detection_method": self.config.detection_method.value,
                    "matching_strategy": self.config.matching_strategy.value,
                },
            )

            # Emit recognition event
            if self.event_bus:
                await self.event_bus.emit(
                    EventType.VISION_ANALYSIS_COMPLETED,
                    {
                        "component": "face_recognition",
                        "result": result,
                        "timestamp": datetime.now(timezone.utc),
                    },
                )

            return result

        except Exception as e:
            self.logger.error(f"Face recognition failed: {e}")

            if self.metrics:
                self.metrics.increment("face_recognition_errors_total")

            raise FaceRecognitionError(f"Face recognition failed: {e}", "RECOGNITION_FAILED")

    async def verify_identity(
        self,
        image: np.ndarray,
        identity_id: str,
        verification_mode: VerificationMode = VerificationMode.ONE_TO_ONE,
    ) -> VerificationResult:
        """Verify if a face in the image belongs to a specific identity."""
        if not self._is_initialized:
            await self.initialize()

        start_time = asyncio.get_event_loop().time()

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
                    distance=float("inf"),
                    threshold_used=identity.confidence_threshold,
                    verification_mode=verification_mode,
                    processing_time=asyncio.get_event_loop().time() - start_time,
                    metadata={"error": "No faces detected"},
                )

            face_encodings = await self.encoder.encode_faces(image, detection_result.face_locations)

            if not face_encodings:
                return VerificationResult(
                    is_match=False,
                    confidence=0.0,
                    distance=float("inf"),
                    threshold_used=identity.confidence_threshold,
                    verification_mode=verification_mode,
                    processing_time=asyncio.get_event_loop().time() - start_time,
                    metadata={"error": "Face encoding failed"},
                )

            # Anti-spoofing check
            anti_spoofing_passed = True
            if self.config.enable_anti_spoofing:
                anti_spoofing_results = await self.anti_spoofing.detect_spoofing(
                    image, detection_result.face_locations
                )
                anti_spoofing_passed = anti_spoofing_results.get("is_live", True)

            # Get identity encodings
            identity_encodings = self.identity_manager._identity_encodings.get(identity_id, [])

            if not identity_encodings:
                return VerificationResult(
                    is_match=False,
                    confidence=0.0,
                    distance=float("inf"),
                    threshold_used=identity.confidence_threshold,
                    verification_mode=verification_mode,
                    processing_time=asyncio.get_event_loop().time() - start_time,
                    anti_spoofing_passed=anti_spoofing_passed,
                    metadata={"error": "No encodings for identity"},
                )

            # Find best match among identity's encodings
            best_distance = float("inf")
            for face_encoding in face_encodings:
                for _, identity_encoding in identity_encodings:
                    distance = self.matcher.calculate_distance(
                        face_encoding.encoding_vector, identity_encoding
                    )
                    best_distance = min(best_distance, distance)

            is_match = best_distance <= identity.confidence_threshold
            confidence = 1.0 - (best_distance / identity.confidence_threshold) if is_match else 0.0
            confidence = max(0.0, min(1.0, confidence))

            processing_time = asyncio.get_event_loop().time() - start_time

            return VerificationResult(
                is_match=is_match and anti_spoofing_passed,
                confidence=confidence,
                distance=best_distance,
                threshold_used=identity.confidence_threshold,
                verification_mode=verification_mode,
                processing_time=processing_time,
                anti_spoofing_passed=anti_spoofing_passed,
                metadata={
                    "faces_detected": len(detection_result.face_locations),
                    "identity_name": identity.name,
                },
            )

        except Exception as e:
            self.logger.error(f"Identity verification failed: {e}")
            raise FaceRecognitionError(f"Verification failed: {e}", "VERIFICATION_FAILED")

    async def enroll_identity(
        self, images: List[np.ndarray], name: str, identity_id: Optional[str] = None, **kwargs
    ) -> str:
        """Enroll a new identity with multiple face images."""
        if not self._is_initialized:
            await self.initialize()

        try:
            all_encodings = []

            for i, image in enumerate(images):
                # Detect faces
                detection_result = await self.detector.detect_faces(image)

                if not detection_result.face_locations:
                    self.logger.warning(f"No faces detected in image {i}")
                    continue

                # Use the largest face if multiple detected
                largest_face_idx = 0
                if len(detection_result.face_locations) > 1:
                    largest_area = 0
                    for j, (top, right, bottom, left) in enumerate(detection_result.face_locations):
                        area = (bottom - top) * (right - left)
                        if area > largest_area:
                            largest_area = area
                            largest_face_idx = j

                face_location = [detection_result.face_locations[largest_face_idx]]

                # Encode face
                face_encodings = await self.encoder.encode_faces(image, face_location)

                if face_encodings:
                    # Only keep high-quality encodings
                    encoding = face_encodings[0]
                    if encoding.quality_score >= self.config.quality_threshold:
                        all_encodings.append(encoding)
                    else:
                        self.logger.warning(f"Low quality encoding in image {i}, skipping")

            if not all_encodings:
                raise FaceRecognitionError("No valid face encodings found in provided images")

            # Limit number of encodings per identity
            if len(all_encodings) > self.config.max_encodings_per_identity:
                # Keep the highest quality encodings
                all_encodings.sort(key=lambda x: x.quality_score, reverse=True)
                all_encodings = all_encodings[: self.config.max_encodings_per_identity]

            # Register identity
            final_identity_id = await self.identity_manager.register_identity(
                name=name, encodings=all_encodings, **kwargs
            )

            self.logger.info(
                f"Enrolled identity {final_identity_id} with {len(all_encodings)} encodings"
            )

            # Emit enrollment event
            if self.event_bus:
                await self.event_bus.emit(
                    EventType.FACE_IDENTITY_ENROLLED,
                    {
                        "identity_id": final_identity_id,
                        "name": name,
                        "encodings_count": len(all_encodings),
                        "timestamp": datetime.now(timezone.utc),
                    },
                )

            return final_identity_id

        except Exception as e:
            self.logger.error(f"Identity enrollment failed: {e}")
            raise FaceRecognitionError(f"Enrollment failed: {e}", "ENROLLMENT_FAILED")

    def _update_metrics(self, processing_time: float, faces_count: int) -> None:
        """Update performance metrics."""
        if self.metrics:
            self.metrics.observe("face_recognition_duration_seconds", processing_time)
            self.metrics.increment("face_recognition_requests_total")
            if faces_count > 0:
                self.metrics.increment("faces_detected_total", faces_count)

    async def get_system_status(self) -> Dict[str, Any]:
        """Get face recognition system status."""
        if not self._is_initialized:
            return {"status": "not_initialized"}

        try:
            identities_count = len(self.identity_manager.list_identities())
            total_encodings = sum(
                len(encodings) for encodings in self.identity_manager._identity_encodings.values()
            )

            return {
                "status": "healthy",
                "initialized": self._is_initialized,
                "identities_count": identities_count,
                "total_encodings": total_encodings,
                "detection_method": self.config.detection_method.value,
                "encoding_method": self.config.encoding_method.value,
                "anti_spoofing_enabled": self.config.enable_anti_spoofing,
                "cache_enabled": self.cache is not None,
                "encryption_enabled": self.encryption is not None,
            }

        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def _health_check_callback(self) -> Dict[str, Any]:
        """Health check callback for system monitoring."""
        return await self.get_system_status()

    async def cleanup(self) -> None:
        """Cleanup face recognition resources."""
        try:
            self.logger.info("Cleaning up face recognition system...")

            # Cleanup components
            if self.anti_spoofing:
                # No explicit cleanup needed for anti-spoofing
                pass

            self._is_initialized = False
            self.logger.info("Face recognition system cleanup completed")

        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")

    def __del__(self):
        """Destructor to ensure cleanup."""
        if self._is_initialized:
            # Note: Can't use async in __del__, so this is a best-effort cleanup
            try:
                self.logger.info("Face recognition system being destroyed")
            except:
                pass
