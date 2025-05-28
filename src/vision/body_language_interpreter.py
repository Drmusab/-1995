"""
Body Language Interpreter

This module combines multiple vision analysis components to provide comprehensive
body language interpretation and non-verbal communication analysis.
"""

import asyncio
import logging
import numpy as np
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Union, Set
from enum import Enum
from abc import ABC, abstractmethod

# Core system imports
from src.core.dependency_injection import Container
from src.core.events.event_bus import EventBus
from src.observability.monitoring.metrics import MetricsCollector
from src.observability.logging.config import get_logger
from src.integrations.cache.redis_cache import RedisCache
from src.integrations.storage.database import DatabaseManager
from src.core.security.encryption import EncryptionManager

# Vision processing imports
from src.vision.pose_estimator import (
    EnhancedPoseEstimator, PoseEstimationResult, PosePerson,
    PoseKeypoint, PoseConfiguration, PoseEstimationMethod
)
from src.vision.face_recognition import (
    EnhancedFaceRecognition, FaceRecognitionResult
)
from src.vision.gesture_recognizer import (
    EnhancedGestureRecognizer, GestureRecognitionResult, RecognizedGesture,
    GestureType, HandSide
)
from src.vision.expression_analyzer import (
    EnhancedExpressionAnalyzer, ExpressionAnalysisResult, ExpressionVector,
    EmotionalState, ExpressionType
)

# Component imports
from src.assistant.component_manager import ComponentInterface, ComponentState

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
except ImportError:
    StandardScaler = None
    KMeans = None


class BodyLanguageCategory(Enum):
    """Categories of body language interpretations."""
    ENGAGEMENT = "engagement"
    CONFIDENCE = "confidence"
    STRESS = "stress"
    DECEPTION = "deception"
    DOMINANCE = "dominance"
    SUBMISSION = "submission"
    ATTRACTION = "attraction"
    DEFENSIVENESS = "defensiveness"
    COMFORT = "comfort"
    DISCOMFORT = "discomfort"
    FOCUS = "focus"
    DISTRACTION = "distraction"
    OPENNESS = "openness"
    CLOSED_OFF = "closed_off"
    AGREEMENT = "agreement"
    DISAGREEMENT = "disagreement"


class InterpretationConfidence(Enum):
    """Confidence levels for body language interpretation."""
    VERY_LOW = "very_low"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"


class CulturalContext(Enum):
    """Cultural contexts that affect body language interpretation."""
    WESTERN = "western"
    EASTERN = "eastern"
    MIDDLE_EASTERN = "middle_eastern"
    AFRICAN = "african"
    LATIN_AMERICAN = "latin_american"
    UNIVERSAL = "universal"
    UNKNOWN = "unknown"


class SocialContext(Enum):
    """Social contexts for body language interpretation."""
    FORMAL = "formal"
    INFORMAL = "informal"
    PROFESSIONAL = "professional"
    CASUAL = "casual"
    INTIMATE = "intimate"
    PUBLIC = "public"
    PRESENTATION = "presentation"
    NEGOTIATION = "negotiation"
    INTERVIEW = "interview"
    UNKNOWN = "unknown"


@dataclass
class BodyLanguageFeature:
    """Individual body language feature."""
    feature_name: str
    value: float
    confidence: float
    source_component: str  # pose, face, gesture, expression
    description: Optional[str] = None
    cultural_variance: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BodyLanguageCluster:
    """Cluster of related body language features."""
    cluster_id: str
    features: List[BodyLanguageFeature]
    cluster_confidence: float
    primary_interpretation: str
    secondary_interpretations: List[str] = field(default_factory=list)
    temporal_consistency: float = 0.0


@dataclass
class NonverbalCue:
    """Individual non-verbal communication cue."""
    cue_type: str
    cue_name: str
    intensity: float
    confidence: float
    duration_ms: float
    frequency: float = 0.0
    associated_emotions: List[str] = field(default_factory=list)
    cultural_significance: Dict[str, float] = field(default_factory=dict)
    contradictory_signals: List[str] = field(default_factory=list)


@dataclass
class BodyLanguageInterpretation:
    """Complete body language interpretation."""
    interpretation_id: str
    category: BodyLanguageCategory
    primary_message: str
    confidence: InterpretationConfidence
    confidence_score: float
    
    # Supporting evidence
    supporting_features: List[BodyLanguageFeature]
    nonverbal_cues: List[NonverbalCue]
    
    # Context factors
    cultural_context: CulturalContext = CulturalContext.UNIVERSAL
    social_context: SocialContext = SocialContext.UNKNOWN
    temporal_context: str = "single_frame"
    
    # Interpretation characteristics
    consistency_score: float = 0.0
    authenticity_score: float = 0.0
    stress_indicators: float = 0.0
    engagement_level: float = 0.0
    emotional_valence: float = 0.0
    emotional_arousal: float = 0.0
    
    # Alternative interpretations
    alternative_interpretations: List[Dict[str, Any]] = field(default_factory=list)
    conflicting_signals: List[str] = field(default_factory=list)
    
    # Metadata
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    processing_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BodyLanguageAnalysisResult:
    """Complete body language analysis result."""
    success: bool
    request_id: str
    processing_time: float
    
    # Core interpretations
    interpretations: List[BodyLanguageInterpretation]
    dominant_interpretation: Optional[BodyLanguageInterpretation] = None
    
    # Aggregate analysis
    overall_engagement: float = 0.0
    overall_confidence_level: float = 0.0
    overall_stress_level: float = 0.0
    overall_authenticity: float = 0.0
    emotional_state_summary: Dict[str, float] = field(default_factory=dict)
    
    # Feature analysis
    all_features: List[BodyLanguageFeature] = field(default_factory=list)
    feature_clusters: List[BodyLanguageCluster] = field(default_factory=list)
    nonverbal_cues: List[NonverbalCue] = field(default_factory=list)
    
    # Component results
    pose_result: Optional[PoseEstimationResult] = None
    face_result: Optional[FaceRecognitionResult] = None
    gesture_result: Optional[GestureRecognitionResult] = None
    expression_result: Optional[ExpressionAnalysisResult] = None
    
    # Quality metrics
    analysis_quality: float = 0.0
    interpretation_confidence: float = 0.0
    temporal_consistency: float = 0.0
    
    # Context information
    detected_persons: int = 0
    scene_context: Dict[str, Any] = field(default_factory=dict)
    
    # Error handling
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Metadata
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BodyLanguageConfiguration:
    """Configuration for body language interpretation."""
    # Component configurations
    enable_pose_analysis: bool = True
    enable_face_analysis: bool = True
    enable_gesture_analysis: bool = True
    enable_expression_analysis: bool = True
    
    # Analysis settings
    interpretation_threshold: float = 0.6
    feature_clustering: bool = True
    temporal_analysis: bool = True
    cultural_adaptation: bool = True
    
    # Context settings
    default_cultural_context: CulturalContext = CulturalContext.UNIVERSAL
    default_social_context: SocialContext = SocialContext.UNKNOWN
    context_window_frames: int = 30
    
    # Feature extraction
    extract_micro_expressions: bool = True
    extract_pose_dynamics: bool = True
    extract_gesture_sequences: bool = True
    extract_gaze_patterns: bool = True
    
    # Filtering and quality
    min_person_confidence: float = 0.7
    min_feature_confidence: float = 0.5
    enable_outlier_detection: bool = True
    enable_contradiction_detection: bool = True
    
    # Performance settings
    max_persons_analyze: int = 5
    enable_parallel_processing: bool = True
    enable_caching: bool = True
    cache_ttl: int = 300
    
    # Cultural models
    cultural_weight_factors: Dict[str, float] = field(default_factory=lambda: {
        "western": 1.0,
        "eastern": 0.8,
        "universal": 0.9
    })
    
    # Privacy and ethics
    anonymize_results: bool = False
    store_analysis_data: bool = False
    respect_cultural_norms: bool = True


class BodyLanguageInterpreterError(Exception):
    """Custom exception for body language interpretation operations."""
    
    def __init__(self, message: str, error_code: Optional[str] = None,
                 component: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.error_code = error_code
        self.component = component
        self.details = details or {}
        self.timestamp = datetime.now(timezone.utc)


class FeatureExtractor(ABC):
    """Abstract base class for body language feature extractors."""
    
    @abstractmethod
    async def extract_features(self, 
                             pose_result: Optional[PoseEstimationResult],
                             face_result: Optional[FaceRecognitionResult],
                             gesture_result: Optional[GestureRecognitionResult],
                             expression_result: Optional[ExpressionAnalysisResult],
                             context: Dict[str, Any]) -> List[BodyLanguageFeature]:
        """Extract body language features from component results."""
        pass


class PostureFeatureExtractor(FeatureExtractor):
    """Extracts posture-related body language features."""
    
    def __init__(self, logger):
        self.logger = logger
        
    async def extract_features(self, 
                             pose_result: Optional[PoseEstimationResult],
                             face_result: Optional[FaceRecognitionResult],
                             gesture_result: Optional[GestureRecognitionResult],
                             expression_result: Optional[ExpressionAnalysisResult],
                             context: Dict[str, Any]) -> List[BodyLanguageFeature]:
        """Extract posture features."""
        features = []
        
        if not pose_result or not pose_result.detected_persons:
            return features
            
        for person in pose_result.detected_persons:
            # Shoulder alignment
            shoulder_alignment = self._analyze_shoulder_alignment(person.keypoints)
            if shoulder_alignment is not None:
                features.append(BodyLanguageFeature(
                    feature_name="shoulder_alignment",
                    value=shoulder_alignment,
                    confidence=person.confidence,
                    source_component="pose",
                    description="Level of shoulder alignment indicating posture confidence"
                ))
            
            # Back straightness
            back_straightness = self._analyze_back_straightness(person.keypoints)
            if back_straightness is not None:
                features.append(BodyLanguageFeature(
                    feature_name="back_straightness",
                    value=back_straightness,
                    confidence=person.confidence,
                    source_component="pose",
                    description="Straightness of back indicating confidence/alertness"
                ))
            
            # Head tilt
            head_tilt = self._analyze_head_tilt(person.keypoints)
            if head_tilt is not None:
                features.append(BodyLanguageFeature(
                    feature_name="head_tilt",
                    value=head_tilt,
                    confidence=person.confidence,
                    source_component="pose",
                    description="Head tilt angle indicating interest/confusion"
                ))
            
            # Body openness
            body_openness = self._analyze_body_openness(person.keypoints)
            if body_openness is not None:
                features.append(BodyLanguageFeature(
                    feature_name="body_openness",
                    value=body_openness,
                    confidence=person.confidence,
                    source_component="pose",
                    description="Openness of body posture indicating receptiveness"
                ))
                
        return features
    
    def _analyze_shoulder_alignment(self, keypoints: List[PoseKeypoint]) -> Optional[float]:
        """Analyze shoulder alignment."""
        try:
            # Find shoulder keypoints (adjust indices based on pose format)
            left_shoulder = None
            right_shoulder = None
            
            for kp in keypoints:
                if kp.name and "left_shoulder" in kp.name.lower():
                    left_shoulder = kp
                elif kp.name and "right_shoulder" in kp.name.lower():
                    right_shoulder = kp
            
            if not left_shoulder or not right_shoulder:
                return None
                
            # Calculate shoulder angle
            dy = right_shoulder.y - left_shoulder.y
            dx = right_shoulder.x - left_shoulder.x
            angle = abs(np.arctan2(dy, dx))
            
            # Normalize to 0-1 (0 = tilted, 1 = aligned)
            alignment = 1.0 - min(angle / (np.pi / 6), 1.0)
            return float(alignment)
            
        except Exception as e:
            self.logger.warning(f"Error analyzing shoulder alignment: {e}")
            return None
    
    def _analyze_back_straightness(self, keypoints: List[PoseKeypoint]) -> Optional[float]:
        """Analyze back straightness."""
        try:
            # Find spine-related keypoints
            neck = None
            hip_center = None
            
            for kp in keypoints:
                if kp.name and "neck" in kp.name.lower():
                    neck = kp
                elif kp.name and "hip" in kp.name.lower() and "center" in kp.name.lower():
                    hip_center = kp
            
            if not neck or not hip_center:
                return None
                
            # Calculate spine angle from vertical
            dx = neck.x - hip_center.x
            dy = neck.y - hip_center.y
            angle = abs(np.arctan2(dx, dy))
            
            # Normalize to 0-1 (0 = bent, 1 = straight)
            straightness = 1.0 - min(angle / (np.pi / 4), 1.0)
            return float(straightness)
            
        except Exception as e:
            self.logger.warning(f"Error analyzing back straightness: {e}")
            return None
    
    def _analyze_head_tilt(self, keypoints: List[PoseKeypoint]) -> Optional[float]:
        """Analyze head tilt."""
        try:
            # Find head-related keypoints
            left_ear = None
            right_ear = None
            
            for kp in keypoints:
                if kp.name and "left_ear" in kp.name.lower():
                    left_ear = kp
                elif kp.name and "right_ear" in kp.name.lower():
                    right_ear = kp
            
            if not left_ear or not right_ear:
                return None
                
            # Calculate head tilt angle
            dy = right_ear.y - left_ear.y
            dx = right_ear.x - left_ear.x
            angle = np.arctan2(dy, dx)
            
            # Normalize to -1 to 1 (negative = left tilt, positive = right tilt)
            tilt = np.clip(angle / (np.pi / 4), -1.0, 1.0)
            return float(tilt)
            
        except Exception as e:
            self.logger.warning(f"Error analyzing head tilt: {e}")
            return None
    
    def _analyze_body_openness(self, keypoints: List[PoseKeypoint]) -> Optional[float]:
        """Analyze body openness."""
        try:
            # Find arm keypoints
            left_elbow = None
            right_elbow = None
            torso_center = None
            
            for kp in keypoints:
                if kp.name and "left_elbow" in kp.name.lower():
                    left_elbow = kp
                elif kp.name and "right_elbow" in kp.name.lower():
                    right_elbow = kp
                elif kp.name and ("chest" in kp.name.lower() or "torso" in kp.name.lower()):
                    torso_center = kp
            
            if not left_elbow or not right_elbow or not torso_center:
                return None
                
            # Calculate distance of elbows from torso center
            left_distance = np.sqrt((left_elbow.x - torso_center.x)**2 + 
                                  (left_elbow.y - torso_center.y)**2)
            right_distance = np.sqrt((right_elbow.x - torso_center.x)**2 + 
                                   (right_elbow.y - torso_center.y)**2)
            
            # Average distance as openness indicator
            openness = (left_distance + right_distance) / 2.0
            
            # Normalize (this would need calibration based on image size)
            normalized_openness = np.clip(openness / 100.0, 0.0, 1.0)
            return float(normalized_openness)
            
        except Exception as e:
            self.logger.warning(f"Error analyzing body openness: {e}")
            return None


class FacialFeatureExtractor(FeatureExtractor):
    """Extracts facial expression and gaze-related features."""
    
    def __init__(self, logger):
        self.logger = logger
        
    async def extract_features(self, 
                             pose_result: Optional[PoseEstimationResult],
                             face_result: Optional[FaceRecognitionResult],
                             gesture_result: Optional[GestureRecognitionResult],
                             expression_result: Optional[ExpressionAnalysisResult],
                             context: Dict[str, Any]) -> List[BodyLanguageFeature]:
        """Extract facial features."""
        features = []
        
        if expression_result and expression_result.expression_vectors:
            for expr_vector in expression_result.expression_vectors:
                # Eye contact estimation
                eye_contact = self._estimate_eye_contact(expr_vector)
                if eye_contact is not None:
                    features.append(BodyLanguageFeature(
                        feature_name="eye_contact",
                        value=eye_contact,
                        confidence=expr_vector.confidence,
                        source_component="expression",
                        description="Level of direct eye contact"
                    ))
                
                # Smile authenticity
                smile_auth = self._analyze_smile_authenticity(expr_vector)
                if smile_auth is not None:
                    features.append(BodyLanguageFeature(
                        feature_name="smile_authenticity",
                        value=smile_auth,
                        confidence=expr_vector.confidence,
                        source_component="expression",
                        description="Authenticity of smile expression"
                    ))
                
                # Facial tension
                tension = self._analyze_facial_tension(expr_vector)
                if tension is not None:
                    features.append(BodyLanguageFeature(
                        feature_name="facial_tension",
                        value=tension,
                        confidence=expr_vector.confidence,
                        source_component="expression",
                        description="Level of facial muscle tension"
                    ))
        
        return features
    
    def _estimate_eye_contact(self, expr_vector: ExpressionVector) -> Optional[float]:
        """Estimate eye contact level."""
        try:
            # This is a simplified estimation - in practice, you'd need
            # gaze direction analysis
            if "neutral" in expr_vector.expression_scores:
                neutral_score = expr_vector.expression_scores["neutral"]
                # Higher neutral score might indicate direct gaze
                return float(neutral_score)
            return None
        except Exception as e:
            self.logger.warning(f"Error estimating eye contact: {e}")
            return None
    
    def _analyze_smile_authenticity(self, expr_vector: ExpressionVector) -> Optional[float]:
        """Analyze smile authenticity (Duchenne vs fake smile)."""
        try:
            happiness_score = expr_vector.expression_scores.get("happiness", 0.0)
            if happiness_score > 0.3:
                # Use authenticity score if available
                return float(expr_vector.authenticity)
            return None
        except Exception as e:
            self.logger.warning(f"Error analyzing smile authenticity: {e}")
            return None
    
    def _analyze_facial_tension(self, expr_vector: ExpressionVector) -> Optional[float]:
        """Analyze facial tension indicators."""
        try:
            # Combine stress-related expressions
            stress_expressions = ["anger", "fear", "surprise", "disgust"]
            tension_score = sum(expr_vector.expression_scores.get(expr, 0.0) 
                              for expr in stress_expressions)
            return float(np.clip(tension_score, 0.0, 1.0))
        except Exception as e:
            self.logger.warning(f"Error analyzing facial tension: {e}")
            return None


class GestureFeatureExtractor(FeatureExtractor):
    """Extracts gesture-related body language features."""
    
    def __init__(self, logger):
        self.logger = logger
        
    async def extract_features(self, 
                             pose_result: Optional[PoseEstimationResult],
                             face_result: Optional[FaceRecognitionResult],
                             gesture_result: Optional[GestureRecognitionResult],
                             expression_result: Optional[ExpressionAnalysisResult],
                             context: Dict[str, Any]) -> List[BodyLanguageFeature]:
        """Extract gesture features."""
        features = []
        
        if gesture_result and gesture_result.recognized_gestures:
            # Gesture confidence aggregate
            gesture_confidence = np.mean([g.confidence for g in gesture_result.recognized_gestures])
            features.append(BodyLanguageFeature(
                feature_name="gesture_confidence",
                value=float(gesture_confidence),
                confidence=gesture_confidence,
                source_component="gesture",
                description="Overall confidence in gesture recognition"
            ))
            
            # Defensive gestures
            defensive_score = self._analyze_defensive_gestures(gesture_result.recognized_gestures)
            if defensive_score > 0:
                features.append(BodyLanguageFeature(
                    feature_name="defensive_gestures",
                    value=defensive_score,
                    confidence=gesture_confidence,
                    source_component="gesture",
                    description="Presence of defensive body language"
                ))
            
            # Open gestures
            open_score = self._analyze_open_gestures(gesture_result.recognized_gestures)
            if open_score > 0:
                features.append(BodyLanguageFeature(
                    feature_name="open_gestures",
                    value=open_score,
                    confidence=gesture_confidence,
                    source_component="gesture",
                    description="Presence of open, welcoming gestures"
                ))
        
        return features
    
    def _analyze_defensive_gestures(self, gestures: List[RecognizedGesture]) -> float:
        """Analyze defensive gesture patterns."""
        defensive_types = {
            GestureType.ARMS_CROSSED: 0.8,
            GestureType.FIST: 0.6,
            GestureType.STOP_GESTURE: 0.7
        }
        
        total_score = 0.0
        for gesture in gestures:
            if gesture.gesture_type in defensive_types:
                total_score += defensive_types[gesture.gesture_type] * gesture.confidence
        
        return min(total_score, 1.0)
    
    def _analyze_open_gestures(self, gestures: List[RecognizedGesture]) -> float:
        """Analyze open gesture patterns."""
        open_types = {
            GestureType.OPEN_PALM: 0.8,
            GestureType.HAND_WAVE: 0.6,
            GestureType.THUMBS_UP: 0.7,
            GestureType.PEACE_SIGN: 0.5
        }
        
        total_score = 0.0
        for gesture in gestures:
            if gesture.gesture_type in open_types:
                total_score += open_types[gesture.gesture_type] * gesture.confidence
        
        return min(total_score, 1.0)


class BodyLanguageInterpreter:
    """Core body language interpretation engine."""
    
    def __init__(self, logger, config: BodyLanguageConfiguration):
        self.logger = logger
        self.config = config
        self.feature_extractors = []
        self.interpretation_rules = {}
        self._setup_extractors()
        self._setup_interpretation_rules()
    
    def _setup_extractors(self) -> None:
        """Setup feature extractors."""
        self.feature_extractors = [
            PostureFeatureExtractor(self.logger),
            FacialFeatureExtractor(self.logger),
            GestureFeatureExtractor(self.logger)
        ]
    
    def _setup_interpretation_rules(self) -> None:
        """Setup interpretation rules for different body language categories."""
        self.interpretation_rules = {
            BodyLanguageCategory.CONFIDENCE: {
                "required_features": ["back_straightness", "shoulder_alignment"],
                "optional_features": ["eye_contact", "open_gestures"],
                "weights": {"back_straightness": 0.4, "shoulder_alignment": 0.3, 
                           "eye_contact": 0.2, "open_gestures": 0.1},
                "threshold": 0.6
            },
            BodyLanguageCategory.STRESS: {
                "required_features": ["facial_tension"],
                "optional_features": ["defensive_gestures", "body_openness"],
                "weights": {"facial_tension": 0.5, "defensive_gestures": 0.3, 
                           "body_openness": -0.2},  # negative weight for inverse correlation
                "threshold": 0.4
            },
            BodyLanguageCategory.ENGAGEMENT: {
                "required_features": ["eye_contact"],
                "optional_features": ["head_tilt", "body_openness", "gesture_confidence"],
                "weights": {"eye_contact": 0.4, "head_tilt": 0.2, 
                           "body_openness": 0.2, "gesture_confidence": 0.2},
                "threshold": 0.5
            },
            BodyLanguageCategory.OPENNESS: {
                "required_features": ["body_openness"],
                "optional_features": ["open_gestures", "smile_authenticity"],
                "weights": {"body_openness": 0.5, "open_gestures": 0.3, 
                           "smile_authenticity": 0.2},
                "threshold": 0.6
            },
            BodyLanguageCategory.DEFENSIVENESS: {
                "required_features": ["defensive_gestures"],
                "optional_features": ["body_openness", "facial_tension"],
                "weights": {"defensive_gestures": 0.6, "body_openness": -0.2, 
                           "facial_tension": 0.2},
                "threshold": 0.5
            }
        }
    
    async def interpret(self, 
                       pose_result: Optional[PoseEstimationResult],
                       face_result: Optional[FaceRecognitionResult],
                       gesture_result: Optional[GestureRecognitionResult],
                       expression_result: Optional[ExpressionAnalysisResult],
                       context: Dict[str, Any]) -> List[BodyLanguageInterpretation]:
        """Perform body language interpretation."""
        interpretations = []
        
        try:
            # Extract features from all components
            all_features = []
            for extractor in self.feature_extractors:
                features = await extractor.extract_features(
                    pose_result, face_result, gesture_result, expression_result, context
                )
                all_features.extend(features)
            
            if not all_features:
                return interpretations
            
            # Create feature lookup
            feature_dict = {f.feature_name: f for f in all_features}
            
            # Apply interpretation rules
            for category, rules in self.interpretation_rules.items():
                interpretation = self._apply_interpretation_rule(
                    category, rules, feature_dict, context
                )
                if interpretation:
                    interpretations.append(interpretation)
            
            # Sort by confidence
            interpretations.sort(key=lambda x: x.confidence_score, reverse=True)
            
        except Exception as e:
            self.logger.error(f"Error in body language interpretation: {e}")
            raise BodyLanguageInterpreterError(
                f"Interpretation failed: {e}",
                error_code="INTERPRETATION_ERROR",
                component="interpreter"
            )
        
        return interpretations
    
    def _apply_interpretation_rule(self, 
                                 category: BodyLanguageCategory,
                                 rules: Dict[str, Any],
                                 feature_dict: Dict[str, BodyLanguageFeature],
                                 context: Dict[str, Any]) -> Optional[BodyLanguageInterpretation]:
        """Apply a specific interpretation rule."""
        try:
            # Check if required features are present
            required_features = rules.get("required_features", [])
            if not all(feat in feature_dict for feat in required_features):
                return None
            
            # Calculate weighted score
            weights = rules.get("weights", {})
            threshold = rules.get("threshold", 0.5)
            
            total_score = 0.0
            total_weight = 0.0
            supporting_features = []
            
            for feature_name, weight in weights.items():
                if feature_name in feature_dict:
                    feature = feature_dict[feature_name]
                    score_contribution = feature.value * weight * feature.confidence
                    total_score += score_contribution
                    total_weight += abs(weight)
                    supporting_features.append(feature)
            
            if total_weight == 0:
                return None
            
            # Normalize score
            normalized_score = total_score / total_weight
            
            # Check threshold
            if abs(normalized_score) < threshold:
                return None
            
            # Determine confidence level
            confidence_score = abs(normalized_score)
            if confidence_score >= 0.9:
                confidence = InterpretationConfidence.VERY_HIGH
            elif confidence_score >= 0.75:
                confidence = InterpretationConfidence.HIGH
            elif confidence_score >= 0.6:
                confidence = InterpretationConfidence.MODERATE
            elif confidence_score >= 0.4:
                confidence = InterpretationConfidence.LOW
            else:
                confidence = InterpretationConfidence.VERY_LOW
            
            # Create interpretation
            interpretation = BodyLanguageInterpretation(
                interpretation_id=f"{category.value}_{datetime.now().timestamp()}",
                category=category,
                primary_message=self._generate_message(category, normalized_score),
                confidence=confidence,
                confidence_score=confidence_score,
                supporting_features=supporting_features,
                nonverbal_cues=self._extract_nonverbal_cues(category, supporting_features),
                cultural_context=context.get("cultural_context", CulturalContext.UNIVERSAL),
                social_context=context.get("social_context", SocialContext.UNKNOWN)
            )
            
            return interpretation
            
        except Exception as e:
            self.logger.warning(f"Error applying interpretation rule for {category}: {e}")
            return None
    
    def _generate_message(self, category: BodyLanguageCategory, score: float) -> str:
        """Generate interpretation message based on category and score."""
        intensity = "high" if abs(score) > 0.8 else "moderate" if abs(score) > 0.6 else "low"
        
        messages = {
            BodyLanguageCategory.CONFIDENCE: f"Shows {intensity} confidence in posture and demeanor",
            BodyLanguageCategory.STRESS: f"Displays {intensity} stress indicators",
            BodyLanguageCategory.ENGAGEMENT: f"Demonstrates {intensity} engagement level",
            BodyLanguageCategory.OPENNESS: f"Exhibits {intensity} openness to interaction",
            BodyLanguageCategory.DEFENSIVENESS: f"Shows {intensity} defensive body language"
        }
        
        return messages.get(category, f"Shows {intensity} {category.value}")
    
    def _extract_nonverbal_cues(self, 
                               category: BodyLanguageCategory,
                               features: List[BodyLanguageFeature]) -> List[NonverbalCue]:
        """Extract specific nonverbal cues from features."""
        cues = []
        
        for feature in features:
            cue = NonverbalCue(
                cue_type=feature.source_component,
                cue_name=feature.feature_name,
                intensity=feature.value,
                confidence=feature.confidence,
                duration_ms=0.0,  # Would need temporal analysis
                associated_emotions=[category.value]
            )
            cues.append(cue)
        
        return cues


class EnhancedBodyLanguageInterpreter(ComponentInterface):
    """Enhanced body language interpreter with full system integration."""
    
    def __init__(self, container: Container):
        self.container = container
        self.logger = get_logger(__name__)
        self.config = BodyLanguageConfiguration()
        
        # Core components
        self.pose_estimator: Optional[EnhancedPoseEstimator] = None
        self.face_recognition: Optional[EnhancedFaceRecognition] = None
        self.gesture_recognizer: Optional[EnhancedGestureRecognizer] = None
        self.expression_analyzer: Optional[EnhancedExpressionAnalyzer] = None
        
        # Processing components
        self.interpreter: Optional[BodyLanguageInterpreter] = None
        
        # Infrastructure
        self.event_bus: Optional[EventBus] = None
        self.metrics: Optional[MetricsCollector] = None
        self.cache: Optional[RedisCache] = None
        self.database: Optional[DatabaseManager] = None
        
        # State management
        self.is_initialized = False
        self.processing_stats = {
            "total_analyses": 0,
            "successful_analyses": 0,
            "average_processing_time": 0.0,
            "interpretation_accuracy": 0.0
        }
        
        self._load_configuration()
        self._setup_core_components()
        self._setup_processing_components()
        self._setup_monitoring()
        self._setup_caching()
    
    def _load_configuration(self) -> None:
        """Load configuration from container."""
        try:
            config_manager = self.container.get("config_manager")
            if config_manager:
                body_lang_config = config_manager.get("body_language_interpreter", {})
                
                # Update configuration with loaded values
                for key, value in body_lang_config.items():
                    if hasattr(self.config, key):
                        setattr(self.config, key, value)
                        
        except Exception as e:
            self.logger.warning(f"Could not load configuration: {e}")
    
    def _setup_core_components(self) -> None:
        """Setup core vision processing components."""
        try:
            self.pose_estimator = self.container.get("pose_estimator")
            self.face_recognition = self.container.get("face_recognition")
            self.gesture_recognizer = self.container.get("gesture_recognizer")
            self.expression_analyzer = self.container.get("expression_analyzer")
            
        except Exception as e:
            self.logger.warning(f"Could not setup core components: {e}")
    
    def _setup_processing_components(self) -> None:
        """Setup processing components."""
        try:
            self.interpreter = BodyLanguageInterpreter(self.logger, self.config)
            
        except Exception as e:
            self.logger.error(f"Error setting up processing components: {e}")
    
    def _setup_monitoring(self) -> None:
        """Setup monitoring components."""
        try:
            self.metrics = self.container.get("metrics_collector")
            self.event_bus = self.container.get("event_bus")
            
        except Exception as e:
            self.logger.warning(f"Could not setup monitoring: {e}")
    
    def _setup_caching(self) -> None:
        """Setup caching components."""
        try:
            self.cache = self.container.get("redis_cache")
            self.database = self.container.get("database_manager")
            
        except Exception as e:
            self.logger.warning(f"Could not setup caching: {e}")
    
    async def initialize(self) -> None:
        """Initialize the body language interpreter."""
        try:
            if self.is_initialized:
                return
            
            self.logger.info("Initializing Body Language Interpreter...")
            
            # Initialize component dependencies
            await self._initialize_dependencies()
            
            # Initialize interpreter
            if not self.interpreter:
                self.interpreter = BodyLanguageInterpreter(self.logger, self.config)
            
            # Register event handlers
            await self._register_event_handlers()
            
            self.is_initialized = True
            self.logger.info("Body Language Interpreter initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize body language interpreter: {e}")
            raise BodyLanguageInterpreterError(
                f"Initialization failed: {e}",
                error_code="INIT_ERROR",
                component="body_language_interpreter"
            )
    
    async def _initialize_dependencies(self) -> None:
        """Initialize component dependencies."""
        dependencies = [
            ("pose_estimator", self.pose_estimator),
            ("face_recognition", self.face_recognition),
            ("gesture_recognizer", self.gesture_recognizer),
            ("expression_analyzer", self.expression_analyzer)
        ]
        
        for name, component in dependencies:
            if component and hasattr(component, 'initialize'):
                try:
                    await component.initialize()
                    self.logger.debug(f"Initialized {name}")
                except Exception as e:
                    self.logger.warning(f"Could not initialize {name}: {e}")
    
    async def _register_event_handlers(self) -> None:
        """Register event handlers."""
        if self.event_bus:
            await self.event_bus.subscribe("vision_analysis_requested", self._handle_vision_analysis)
            await self.event_bus.subscribe("component_health_check", self._handle_health_check)
    
    async def analyze_body_language(self, 
                                  image: np.ndarray,
                                  context: Optional[Dict[str, Any]] = None,
                                  cultural_context: CulturalContext = CulturalContext.UNIVERSAL,
                                  social_context: SocialContext = SocialContext.UNKNOWN) -> BodyLanguageAnalysisResult:
        """Analyze body language from image."""
        start_time = datetime.now()
        request_id = f"bl_{start_time.timestamp()}"
        
        if context is None:
            context = {}
        
        context.update({
            "cultural_context": cultural_context,
            "social_context": social_context
        })
        
        try:
            if not self.is_initialized:
                await self.initialize()
            
            # Check cache first
            if self.config.enable_caching:
                cached_result = await self._get_cached_result(image, context)
                if cached_result:
                    return cached_result
            
            # Run vision analysis components in parallel
            pose_result, face_result, gesture_result, expression_result = await asyncio.gather(
                self._analyze_pose(image) if self.config.enable_pose_analysis else None,
                self._analyze_face(image) if self.config.enable_face_analysis else None,
                self._analyze_gestures(image) if self.config.enable_gesture_analysis else None,
                self._analyze_expressions(image) if self.config.enable_expression_analysis else None,
                return_exceptions=True
            )
            
            # Handle any exceptions from parallel execution
            results = [pose_result, face_result, gesture_result, expression_result]
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    component_names = ["pose", "face", "gesture", "expression"]
                    self.logger.warning(f"Error in {component_names[i]} analysis: {result}")
                    results[i] = None
            
            pose_result, face_result, gesture_result, expression_result = results
            
            # Perform body language interpretation
            interpretations = []
            if self.interpreter:
                interpretations = await self.interpreter.interpret(
                    pose_result, face_result, gesture_result, expression_result, context
                )
            
            # Create analysis result
            processing_time = (datetime.now() - start_time).total_seconds()
            result = self._create_analysis_result(
                request_id, processing_time, interpretations,
                pose_result, face_result, gesture_result, expression_result
            )
            
            # Cache result
            if self.config.enable_caching and result.success:
                await self._cache_result(image, context, result)
            
            # Update metrics
            self._update_metrics(processing_time, len(interpretations))
            
            # Store for learning
            if self.config.store_analysis_data:
                await self._store_analysis_for_learning(result, image, context)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Body language analysis failed: {e}")
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return BodyLanguageAnalysisResult(
                success=False,
                request_id=request_id,
                processing_time=processing_time,
                interpretations=[],
                errors=[str(e)]
            )
    
    async def _analyze_pose(self, image: np.ndarray) -> Optional[PoseEstimationResult]:
        """Analyze pose from image."""
        if not self.pose_estimator:
            return None
        
        try:
            return await self.pose_estimator.estimate_pose(image)
        except Exception as e:
            self.logger.warning(f"Pose analysis failed: {e}")
            return None
    
    async def _analyze_face(self, image: np.ndarray) -> Optional[FaceRecognitionResult]:
        """Analyze face from image."""
        if not self.face_recognition:
            return None
        
        try:
            return await self.face_recognition.detect_and_recognize_faces(image)
        except Exception as e:
            self.logger.warning(f"Face analysis failed: {e}")
            return None
    
    async def _analyze_gestures(self, image: np.ndarray) -> Optional[GestureRecognitionResult]:
        """Analyze gestures from image."""
        if not self.gesture_recognizer:
            return None
        
        try:
            return await self.gesture_recognizer.recognize_gestures(image)
        except Exception as e:
            self.logger.warning(f"Gesture analysis failed: {e}")
            return None
    
    async def _analyze_expressions(self, image: np.ndarray) -> Optional[ExpressionAnalysisResult]:
        """Analyze expressions from image."""
        if not self.expression_analyzer:
            return None
        
        try:
            return await self.expression_analyzer.analyze_expressions(image)
        except Exception as e:
            self.logger.warning(f"Expression analysis failed: {e}")
            return None
    
    def _create_analysis_result(self, 
                               request_id: str,
                               processing_time: float,
                               interpretations: List[BodyLanguageInterpretation],
                               pose_result: Optional[PoseEstimationResult],
                               face_result: Optional[FaceRecognitionResult],
                               gesture_result: Optional[GestureRecognitionResult],
                               expression_result: Optional[ExpressionAnalysisResult]) -> BodyLanguageAnalysisResult:
        """Create comprehensive analysis result."""
        
        # Extract all features
        all_features = []
        for interpretation in interpretations:
            all_features.extend(interpretation.supporting_features)
        
        # Calculate aggregate metrics
        overall_engagement = self._calculate_overall_engagement(interpretations)
        overall_confidence = self._calculate_overall_confidence(interpretations)
        overall_stress = self._calculate_overall_stress(interpretations)
        overall_authenticity = self._calculate_overall_authenticity(interpretations)
        
        # Determine dominant interpretation
        dominant_interpretation = interpretations[0] if interpretations else None
        
        # Count detected persons
        detected_persons = 0
        if pose_result:
            detected_persons = max(detected_persons, len(pose_result.detected_persons))
        if expression_result:
            detected_persons = max(detected_persons, len(expression_result.detected_faces))
        
        return BodyLanguageAnalysisResult(
            success=True,
            request_id=request_id,
            processing_time=processing_time,
            interpretations=interpretations,
            dominant_interpretation=dominant_interpretation,
            overall_engagement=overall_engagement,
            overall_confidence_level=overall_confidence,
            overall_stress_level=overall_stress,
            overall_authenticity=overall_authenticity,
            all_features=all_features,
            pose_result=pose_result,
            face_result=face_result,
            gesture_result=gesture_result,
            expression_result=expression_result,
            detected_persons=detected_persons,
            analysis_quality=self._calculate_analysis_quality(interpretations),
            interpretation_confidence=np.mean([i.confidence_score for i in interpretations]) if interpretations else 0.0
        )
    
    def _calculate_overall_engagement(self, interpretations: List[BodyLanguageInterpretation]) -> float:
        """Calculate overall engagement level."""
        engagement_scores = []
        for interpretation in interpretations:
            if interpretation.category == BodyLanguageCategory.ENGAGEMENT:
                engagement_scores.append(interpretation.confidence_score)
        
        return np.mean(engagement_scores) if engagement_scores else 0.0
    
    def _calculate_overall_confidence(self, interpretations: List[BodyLanguageInterpretation]) -> float:
        """Calculate overall confidence level."""
        confidence_scores = []
        for interpretation in interpretations:
            if interpretation.category == BodyLanguageCategory.CONFIDENCE:
                confidence_scores.append(interpretation.confidence_score)
        
        return np.mean(confidence_scores) if confidence_scores else 0.0
    
    def _calculate_overall_stress(self, interpretations: List[BodyLanguageInterpretation]) -> float:
        """Calculate overall stress level."""
        stress_scores = []
        for interpretation in interpretations:
            if interpretation.category == BodyLanguageCategory.STRESS:
                stress_scores.append(interpretation.confidence_score)
        
        return np.mean(stress_scores) if stress_scores else 0.0
    
    def _calculate_overall_authenticity(self, interpretations: List[BodyLanguageInterpretation]) -> float:
        """Calculate overall authenticity."""
        authenticity_scores = []
        for interpretation in interpretations:
            authenticity_scores.append(interpretation.authenticity_score)
        
        return np.mean(authenticity_scores) if authenticity_scores else 1.0
    
    def _calculate_analysis_quality(self, interpretations: List[BodyLanguageInterpretation]) -> float:
        """Calculate overall analysis quality."""
        if not interpretations:
            return 0.0
        
        # Quality based on number of interpretations and their confidence
        confidence_scores = [i.confidence_score for i in interpretations]
        avg_confidence = np.mean(confidence_scores)
        interpretation_count_factor = min(len(interpretations) / 5.0, 1.0)  # Normalize to max 5 interpretations
        
        return avg_confidence * interpretation_count_factor
    
    async def _get_cached_result(self, image: np.ndarray, context: Dict[str, Any]) -> Optional[BodyLanguageAnalysisResult]:
        """Get cached analysis result."""
        if not self.cache:
            return None
        
        try:
            cache_key = self._generate_cache_key(image, context)
            cached_data = await self.cache.get(cache_key)
            if cached_data:
                # Deserialize cached result
                return self._deserialize_result(cached_data)
        except Exception as e:
            self.logger.warning(f"Error retrieving cached result: {e}")
        
        return None
    
    async def _cache_result(self, image: np.ndarray, context: Dict[str, Any], result: BodyLanguageAnalysisResult) -> None:
        """Cache analysis result."""
        if not self.cache:
            return
        
        try:
            cache_key = self._generate_cache_key(image, context)
            serialized_result = self._serialize_result(result)
            await self.cache.set(cache_key, serialized_result, ttl=self.config.cache_ttl)
        except Exception as e:
            self.logger.warning(f"Error caching result: {e}")
    
    def _generate_cache_key(self, image: np.ndarray, context: Dict[str, Any]) -> str:
        """Generate cache key for image and context."""
        import hashlib
        
        # Create hash of image
        image_hash = hashlib.md5(image.tobytes()).hexdigest()
        
        # Create hash of relevant context
        context_str = f"{context.get('cultural_context', '')}{context.get('social_context', '')}"
        context_hash = hashlib.md5(context_str.encode()).hexdigest()
        
        return f"body_lang:{image_hash}:{context_hash}"
    
    def _serialize_result(self, result: BodyLanguageAnalysisResult) -> Dict[str, Any]:
        """Serialize result for caching."""
        # Simplified serialization - in practice, you'd want more robust serialization
        return {
            "success": result.success,
            "request_id": result.request_id,
            "processing_time": result.processing_time,
            "overall_engagement": result.overall_engagement,
            "overall_confidence_level": result.overall_confidence_level,
            "overall_stress_level": result.overall_stress_level,
            "interpretation_count": len(result.interpretations),
            "timestamp": result.timestamp.isoformat()
        }
    
    def _deserialize_result(self, data: Dict[str, Any]) -> BodyLanguageAnalysisResult:
        """Deserialize cached result."""
        # Simplified deserialization
        return BodyLanguageAnalysisResult(
            success=data["success"],
            request_id=data["request_id"],
            processing_time=data["processing_time"],
            interpretations=[],  # Would need full deserialization
            overall_engagement=data["overall_engagement"],
            overall_confidence_level=data["overall_confidence_level"],
            overall_stress_level=data["overall_stress_level"]
        )
    
    def _update_metrics(self, processing_time: float, interpretation_count: int) -> None:
        """Update processing metrics."""
        try:
            self.processing_stats["total_analyses"] += 1
            if interpretation_count > 0:
                self.processing_stats["successful_analyses"] += 1
            
            # Update average processing time
            current_avg = self.processing_stats["average_processing_time"]
            total_analyses = self.processing_stats["total_analyses"]
            self.processing_stats["average_processing_time"] = (
                (current_avg * (total_analyses - 1) + processing_time) / total_analyses
            )
            
            # Send metrics
            if self.metrics:
                self.metrics.record_counter("body_language_analyses_total", 1)
                self.metrics.record_histogram("body_language_processing_time", processing_time)
                self.metrics.record_gauge("body_language_interpretations_count", interpretation_count)
                
        except Exception as e:
            self.logger.warning(f"Error updating metrics: {e}")
    
    async def _store_analysis_for_learning(self, result: BodyLanguageAnalysisResult, 
                                         image: np.ndarray, context: Dict[str, Any]) -> None:
        """Store analysis result for learning and improvement."""
        if not self.database:
            return
        
        try:
            # Store analysis metadata (not the actual image for privacy)
            analysis_data = {
                "request_id": result.request_id,
                "timestamp": result.timestamp.isoformat(),
                "processing_time": result.processing_time,
                "interpretation_count": len(result.interpretations),
                "overall_engagement": result.overall_engagement,
                "overall_confidence": result.overall_confidence_level,
                "overall_stress": result.overall_stress_level,
                "cultural_context": context.get("cultural_context", "universal"),
                "social_context": context.get("social_context", "unknown"),
                "analysis_quality": result.analysis_quality
            }
            
            await self.database.store_analysis_result("body_language_analyses", analysis_data)
            
        except Exception as e:
            self.logger.warning(f"Error storing analysis for learning: {e}")
    
    async def _handle_vision_analysis(self, event) -> None:
        """Handle vision analysis events."""
        try:
            if event.data.get("request_body_language", False):
                image = event.data.get("image")
                context = event.data.get("context", {})
                
                if image is not None:
                    result = await self.analyze_body_language(image, context)
                    
                    # Publish result
                    if self.event_bus:
                        await self.event_bus.publish("body_language_analysis_completed", {
                            "request_id": event.data.get("request_id"),
                            "result": result
                        })
                        
        except Exception as e:
            self.logger.error(f"Error handling vision analysis event: {e}")
    
    async def _handle_health_check(self, event) -> None:
        """Handle health check events."""
        try:
            health_data = await self._health_check_callback()
            
            if self.event_bus:
                await self.event_bus.publish("component_health_status", {
                    "component": "body_language_interpreter",
                    "health_data": health_data
                })
                
        except Exception as e:
            self.logger.error(f"Error handling health check event: {e}")
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get system status information."""
        return {
            "component": "body_language_interpreter",
            "initialized": self.is_initialized,
            "processing_stats": self.processing_stats.copy(),
            "configuration": {
                "enable_pose_analysis": self.config.enable_pose_analysis,
                "enable_face_analysis": self.config.enable_face_analysis,
                "enable_gesture_analysis": self.config.enable_gesture_analysis,
                "enable_expression_analysis": self.config.enable_expression_analysis,
                "cultural_adaptation": self.config.cultural_adaptation,
                "temporal_analysis": self.config.temporal_analysis
            },
            "dependencies": {
                "pose_estimator": self.pose_estimator is not None,
                "face_recognition": self.face_recognition is not None,
                "gesture_recognizer": self.gesture_recognizer is not None,
                "expression_analyzer": self.expression_analyzer is not None
            }
        }
    
    async def _health_check_callback(self) -> Dict[str, Any]:
        """Health check callback for monitoring."""
        try:
            health_data = {
                "status": "healthy" if self.is_initialized else "unhealthy",
                "component": "body_language_interpreter",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "metrics": self.processing_stats.copy(),
                "dependencies_available": {
                    "pose_estimator": self.pose_estimator is not None,
                    "face_recognition": self.face_recognition is not None,
                    "gesture_recognizer": self.gesture_recognizer is not None,
                    "expression_analyzer": self.expression_analyzer is not None
                }
            }
            
            return health_data
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "component": "body_language_interpreter",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
    
    async def cleanup(self) -> None:
        """Cleanup resources."""
        try:
            self.logger.info("Cleaning up Body Language Interpreter...")
            
            # Cleanup dependencies
            dependencies = [
                self.pose_estimator,
                self.face_recognition,
                self.gesture_recognizer,
                self.expression_analyzer
            ]
            
            for component in dependencies:
                if component and hasattr(component, 'cleanup'):
                    try:
                        await component.cleanup()
                    except Exception as e:
                        self.logger.warning(f"Error cleaning up component: {e}")
            
            self.is_initialized = False
            self.logger.info("Body Language Interpreter cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
    
    def __del__(self):
        """Destructor."""
        if hasattr(self, 'is_initialized') and self.is_initialized:
            try:
                asyncio.create_task(self.cleanup())
            except Exception:
                pass
