"""
Advanced Sentiment Analysis System
Author: Drmusab
Last Modified: 2025-05-27 14:22:08 UTC

This module provides comprehensive sentiment analysis capabilities for the AI assistant,
including emotion detection, polarity analysis, intensity scoring, and contextual understanding.
"""

from pathlib import Path
from typing import Optional, Dict, Any, List, Set, Union, Tuple, AsyncGenerator
import asyncio
import numpy as np
import threading
import time
from datetime import datetime, timezone
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
import json
import re
import logging
from collections import defaultdict, deque
import weakref

# Core imports
from src.core.config.loader import ConfigLoader
from src.core.events.event_bus import EventBus
from src.core.events.event_types import (
    ComponentInitialized,
    ComponentStarted,
    ComponentStopped,
    ComponentFailed,
    SentimentAnalyzed,
    EmotionDetected,
    MoodChanged
)
from src.core.error_handling import ErrorHandler, handle_exceptions
from src.core.dependency_injection import Container
from src.core.health_check import HealthCheck

# Integration imports
from src.integrations.llm.model_router import ModelRouter
from src.integrations.cache.cache_strategy import CacheStrategy
from src.processing.natural_language.tokenizer import EnhancedTokenizer, TokenizationResult
from src.memory.core_memory.memory_manager import MemoryManager
from src.learning.feedback_processor import FeedbackProcessor

# Observability
from src.observability.monitoring.metrics import MetricsCollector
from src.observability.monitoring.tracing import TraceManager
from src.observability.logging.config import get_logger


class SentimentPolarity(Enum):
    """Sentiment polarity categories."""
    VERY_NEGATIVE = "very_negative"    # -1.0 to -0.6
    NEGATIVE = "negative"              # -0.6 to -0.2
    NEUTRAL = "neutral"                # -0.2 to 0.2
    POSITIVE = "positive"              # 0.2 to 0.6
    VERY_POSITIVE = "very_positive"    # 0.6 to 1.0


class EmotionCategory(Enum):
    """Primary emotion categories based on Plutchik's model."""
    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    TRUST = "trust"
    ANTICIPATION = "anticipation"
    
    # Extended emotions
    LOVE = "love"
    CONTEMPT = "contempt"
    PRIDE = "pride"
    SHAME = "shame"
    GUILT = "guilt"
    ENVY = "envy"
    GRATITUDE = "gratitude"
    CURIOSITY = "curiosity"
    CONFUSION = "confusion"
    FRUSTRATION = "frustration"


class AnalysisMode(Enum):
    """Sentiment analysis processing modes."""
    FAST = "fast"                      # Quick rule-based analysis
    BALANCED = "balanced"              # Hybrid approach
    COMPREHENSIVE = "comprehensive"    # Deep ML-based analysis
    REAL_TIME = "real_time"           # Optimized for real-time
    CONTEXTUAL = "contextual"         # Context-aware analysis
    TEMPORAL = "temporal"             # Time-series analysis


class ContextType(Enum):
    """Types of context for sentiment analysis."""
    CONVERSATIONAL = "conversational"
    CUSTOMER_SERVICE = "customer_service"
    SOCIAL_MEDIA = "social_media"
    FORMAL_COMMUNICATION = "formal_communication"
    CREATIVE_WRITING = "creative_writing"
    EDUCATIONAL = "educational"
    PROFESSIONAL = "professional"
    PERSONAL = "personal"


class IntensityLevel(Enum):
    """Sentiment intensity levels."""
    VERY_LOW = "very_low"      # 0.0 - 0.2
    LOW = "low"                # 0.2 - 0.4
    MODERATE = "moderate"      # 0.4 - 0.6
    HIGH = "high"              # 0.6 - 0.8
    VERY_HIGH = "very_high"    # 0.8 - 1.0


@dataclass
class EmotionScore:
    """Represents an emotion with its confidence score."""
    emotion: EmotionCategory
    confidence: float
    intensity: float
    context_relevance: float = 1.0
    temporal_weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SentimentFeatures:
    """Linguistic and contextual features for sentiment analysis."""
    # Lexical features
    positive_words: List[str] = field(default_factory=list)
    negative_words: List[str] = field(default_factory=list)
    emotion_words: List[str] = field(default_factory=list)
    intensifiers: List[str] = field(default_factory=list)
    negations: List[str] = field(default_factory=list)
    
    # Syntactic features
    exclamation_count: int = 0
    question_count: int = 0
    capitalization_ratio: float = 0.0
    punctuation_density: float = 0.0
    
    # Semantic features
    metaphors: List[str] = field(default_factory=list)
    irony_indicators: List[str] = field(default_factory=list)
    sarcasm_markers: List[str] = field(default_factory=list)
    
    # Contextual features
    named_entities: List[Dict[str, Any]] = field(default_factory=list)
    topics: List[str] = field(default_factory=list)
    discourse_markers: List[str] = field(default_factory=list)
    
    # Temporal features
    tense_distribution: Dict[str, float] = field(default_factory=dict)
    time_references: List[str] = field(default_factory=list)


@dataclass
class SentimentRequest:
    """Request configuration for sentiment analysis."""
    text: str
    mode: AnalysisMode = AnalysisMode.BALANCED
    context_type: ContextType = ContextType.CONVERSATIONAL
    
    # Session and user context
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    conversation_id: Optional[str] = None
    interaction_context: Optional[Dict[str, Any]] = None
    
    # Analysis options
    enable_emotion_detection: bool = True
    enable_mood_tracking: bool = True
    enable_temporal_analysis: bool = False
    enable_comparative_analysis: bool = False
    
    # Personalization
    user_profile: Optional[Dict[str, Any]] = None
    cultural_context: Optional[str] = None
    language: str = "en"
    
    # Quality and performance
    confidence_threshold: float = 0.6
    max_processing_time: float = 5.0
    cache_result: bool = True
    
    # Advanced options
    historical_context_window: int = 10  # Number of previous messages
    enable_cross_modal_analysis: bool = False  # For multimodal input
    custom_lexicons: List[str] = field(default_factory=list)
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: Set[str] = field(default_factory=set)


@dataclass
class SentimentResult:
    """Comprehensive sentiment analysis result."""
    # Primary sentiment scores
    polarity_score: float  # -1.0 to 1.0
    polarity_category: SentimentPolarity
    confidence: float
    intensity: IntensityLevel
    
    # Emotion analysis
    primary_emotion: Optional[EmotionCategory] = None
    emotion_scores: List[EmotionScore] = field(default_factory=list)
    emotion_confidence: float = 0.0
    
    # Detailed analysis
    subjectivity_score: float = 0.0  # 0.0 (objective) to 1.0 (subjective)
    arousal_score: float = 0.0       # 0.0 (calm) to 1.0 (excited)
    valence_score: float = 0.0       # -1.0 (negative) to 1.0 (positive)
    dominance_score: float = 0.0     # 0.0 (submissive) to 1.0 (dominant)
    
    # Linguistic features
    features: Optional[SentimentFeatures] = None
    
    # Contextual analysis
    context_appropriateness: float = 1.0
    mood_indicators: List[str] = field(default_factory=list)
    temporal_trends: Dict[str, float] = field(default_factory=dict)
    
    # Comparative analysis
    sentiment_shift: Optional[float] = None  # Compared to baseline
    emotional_volatility: float = 0.0
    
    # Quality metrics
    analysis_quality_score: float = 0.0
    feature_coverage: float = 0.0
    model_agreement_score: float = 0.0
    
    # Processing information
    processing_time: float = 0.0
    models_used: List[str] = field(default_factory=list)
    analysis_mode: AnalysisMode = AnalysisMode.BALANCED
    
    # Session context
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    conversation_id: Optional[str] = None
    
    # Metadata
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    language: str = "en"
    warnings: List[str] = field(default_factory=list)
    debug_info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MoodProfile:
    """User mood profile over time."""
    user_id: str
    current_mood: EmotionCategory
    mood_stability: float  # 0.0 (volatile) to 1.0 (stable)
    
    # Temporal mood data
    recent_moods: List[Tuple[datetime, EmotionCategory, float]] = field(default_factory=list)
    mood_patterns: Dict[str, float] = field(default_factory=dict)
    
    # Contextual mood variations
    context_moods: Dict[ContextType, EmotionCategory] = field(default_factory=dict)
    
    # Statistical information
    baseline_sentiment: float = 0.0
    sentiment_variance: float = 0.0
    emotional_range: float = 0.0
    
    # Update tracking
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    update_count: int = 0


class SentimentAnalyzerError(Exception):
    """Custom exception for sentiment analysis operations."""
    
    def __init__(self, message: str, error_code: Optional[str] = None, 
                 analysis_mode: Optional[AnalysisMode] = None):
        super().__init__(message)
        self.error_code = error_code
        self.analysis_mode = analysis_mode
        self.timestamp = datetime.now(timezone.utc)


class SentimentAnalyzer(ABC):
    """Abstract base class for sentiment analyzers."""
    
    @abstractmethod
    async def analyze(self, request: SentimentRequest) -> SentimentResult:
        """Analyze sentiment in the given text."""
        pass
    
    @abstractmethod
    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages."""
        pass
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the analyzer."""
        pass


class LexiconBasedAnalyzer(SentimentAnalyzer):
    """Rule-based sentiment analyzer using lexicons and patterns."""
    
    def __init__(self, logger, config: Dict[str, Any]):
        self.logger = logger
        self.config = config
        self.lexicons: Dict[str, Dict[str, float]] = {}
        self.patterns: Dict[str, List[str]] = {}
        self.negation_words = set()
        self.intensifiers: Dict[str, float] = {}
        self.initialized = False
    
    async def initialize(self) -> None:
        """Initialize lexicons and patterns."""
        try:
            await self._load_lexicons()
            await self._load_patterns()
            await self._load_linguistic_resources()
            self.initialized = True
            self.logger.info("LexiconBasedAnalyzer initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize LexiconBasedAnalyzer: {str(e)}")
            raise SentimentAnalyzerError(f"Initialization failed: {str(e)}")
    
    async def _load_lexicons(self) -> None:
        """Load sentiment lexicons."""
        # Load default lexicons (VADER, TextBlob, etc.)
        default_lexicons = {
            "positive": {
                "good": 0.7, "great": 0.8, "excellent": 0.9, "amazing": 0.9,
                "wonderful": 0.8, "fantastic": 0.9, "awesome": 0.8, "love": 0.8,
                "like": 0.6, "enjoy": 0.7, "happy": 0.8, "joy": 0.8, "pleased": 0.7
            },
            "negative": {
                "bad": -0.7, "terrible": -0.9, "awful": -0.8, "horrible": -0.9,
                "hate": -0.8, "dislike": -0.6, "sad": -0.7, "angry": -0.8,
                "frustrated": -0.7, "disappointed": -0.6, "upset": -0.7
            }
        }
        
        self.lexicons.update(default_lexicons)
        
        # Load custom lexicons from config
        lexicon_paths = self.config.get("lexicon_paths", {})
        for lexicon_name, path in lexicon_paths.items():
            try:
                lexicon_data = await self._load_lexicon_file(Path(path))
                self.lexicons[lexicon_name] = lexicon_data
            except Exception as e:
                self.logger.warning(f"Failed to load lexicon {lexicon_name}: {str(e)}")
    
    async def _load_lexicon_file(self, path: Path) -> Dict[str, float]:
        """Load lexicon from file."""
        lexicon = {}
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        word = parts[0].lower()
                        try:
                            score = float(parts[1])
                            lexicon[word] = score
                        except ValueError:
                            continue
        return lexicon
    
    async def _load_patterns(self) -> None:
        """Load linguistic patterns."""
        self.patterns = {
            "intensifiers": [
                r"\b(very|extremely|incredibly|really|quite|rather|fairly)\b",
                r"\b(absolutely|completely|totally|utterly)\b",
                r"\b(so|such|too|more|most)\b"
            ],
            "negations": [
                r"\b(not|no|never|nothing|nowhere|nobody|none)\b",
                r"\b(isn't|aren't|wasn't|weren't|can't|couldn't|won't|wouldn't)\b",
                r"\b(don't|doesn't|didn't|haven't|hasn't|hadn't)\b"
            ],
            "questions": [r"\?"],
            "exclamations": [r"!+"],
            "emphasis": [r"[A-Z]{2,}", r"(.)\1{2,}"]  # ALL CAPS, repeated chars
        }
    
    async def _load_linguistic_resources(self) -> None:
        """Load additional linguistic resources."""
        self.negation_words = {
            "not", "no", "never", "nothing", "nobody", "none", "neither",
            "isn't", "aren't", "wasn't", "weren't", "can't", "couldn't",
            "won't", "wouldn't", "don't", "doesn't", "didn't", "haven't",
            "hasn't", "hadn't", "shouldn't", "mustn't"
        }
        
        self.intensifiers = {
            "very": 1.3, "extremely": 1.5, "incredibly": 1.4, "really": 1.2,
            "quite": 1.1, "rather": 1.1, "fairly": 1.1, "absolutely": 1.5,
            "completely": 1.4, "totally": 1.4, "utterly": 1.5, "so": 1.2,
            "such": 1.2, "too": 1.3, "more": 1.1, "most": 1.2
        }
    
    async def analyze(self, request: SentimentRequest) -> SentimentResult:
        """Analyze sentiment using lexicon-based approach."""
        if not self.initialized:
            await self.initialize()
        
        start_time = time.time()
        
        try:
            # Extract features
            features = await self._extract_features(request.text)
            
            # Calculate sentiment scores
            polarity_score = await self._calculate_polarity(request.text, features)
            emotion_scores = await self._detect_emotions(request.text, features)
            
            # Determine primary emotion
            primary_emotion = None
            emotion_confidence = 0.0
            if emotion_scores:
                top_emotion = max(emotion_scores, key=lambda x: x.confidence)
                primary_emotion = top_emotion.emotion
                emotion_confidence = top_emotion.confidence
            
            # Create result
            result = SentimentResult(
                polarity_score=polarity_score,
                polarity_category=self._score_to_polarity(polarity_score),
                confidence=min(abs(polarity_score) + 0.5, 1.0),
                intensity=self._score_to_intensity(abs(polarity_score)),
                primary_emotion=primary_emotion,
                emotion_scores=emotion_scores,
                emotion_confidence=emotion_confidence,
                features=features,
                processing_time=time.time() - start_time,
                models_used=["lexicon_based"],
                analysis_mode=request.mode,
                session_id=request.session_id,
                user_id=request.user_id,
                conversation_id=request.conversation_id,
                language=request.language
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Lexicon-based analysis failed: {str(e)}")
            raise SentimentAnalyzerError(f"Analysis failed: {str(e)}", "LEXICON_ERROR")
    
    async def _extract_features(self, text: str) -> SentimentFeatures:
        """Extract sentiment features from text."""
        features = SentimentFeatures()
        words = text.lower().split()
        
        # Extract sentiment words
        for word in words:
            if word in self.lexicons.get("positive", {}):
                features.positive_words.append(word)
            elif word in self.lexicons.get("negative", {}):
                features.negative_words.append(word)
        
        # Extract patterns
        features.exclamation_count = len(re.findall(r"!", text))
        features.question_count = len(re.findall(r"\?", text))
        features.capitalization_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        
        # Extract negations and intensifiers
        for word in words:
            if word in self.negation_words:
                features.negations.append(word)
            elif word in self.intensifiers:
                features.intensifiers.append(word)
        
        return features
    
    async def _calculate_polarity(self, text: str, features: SentimentFeatures) -> float:
        """Calculate polarity score."""
        words = text.lower().split()
        total_score = 0.0
        word_count = 0
        
        i = 0
        while i < len(words):
            word = words[i]
            score = 0.0
            
            # Get base sentiment score
            if word in self.lexicons.get("positive", {}):
                score = self.lexicons["positive"][word]
            elif word in self.lexicons.get("negative", {}):
                score = self.lexicons["negative"][word]
            
            if score != 0.0:
                # Apply intensifiers
                if i > 0 and words[i-1] in self.intensifiers:
                    score *= self.intensifiers[words[i-1]]
                
                # Apply negations (within 3 words)
                negated = False
                for j in range(max(0, i-3), i):
                    if words[j] in self.negation_words:
                        negated = True
                        break
                
                if negated:
                    score *= -0.7  # Reduce and flip
                
                total_score += score
                word_count += 1
            
            i += 1
        
        # Normalize score
        if word_count > 0:
            avg_score = total_score / word_count
        else:
            avg_score = 0.0
        
        # Apply feature adjustments
        if features.exclamation_count > 0:
            avg_score *= (1 + 0.1 * features.exclamation_count)
        
        if features.capitalization_ratio > 0.3:
            avg_score *= 1.2
        
        return max(-1.0, min(1.0, avg_score))
    
    async def _detect_emotions(self, text: str, features: SentimentFeatures) -> List[EmotionScore]:
        """Detect emotions using pattern matching."""
        emotion_scores = []
        
        # Simple emotion detection based on keywords
        emotion_keywords = {
            EmotionCategory.JOY: ["happy", "joy", "excited", "cheerful", "delighted"],
            EmotionCategory.SADNESS: ["sad", "depressed", "down", "blue", "melancholy"],
            EmotionCategory.ANGER: ["angry", "mad", "furious", "rage", "annoyed"],
            EmotionCategory.FEAR: ["afraid", "scared", "terrified", "worried", "anxious"],
            EmotionCategory.SURPRISE: ["surprised", "amazed", "shocked", "stunned"],
            EmotionCategory.TRUST: ["trust", "confident", "secure", "certain"],
            EmotionCategory.DISGUST: ["disgusted", "revolted", "sickened", "appalled"],
            EmotionCategory.ANTICIPATION: ["excited", "eager", "looking forward", "anticipating"]
        }
        
        words = text.lower().split()
        for emotion, keywords in emotion_keywords.items():
            matches = sum(1 for word in words if any(kw in word for kw in keywords))
            if matches > 0:
                confidence = min(matches * 0.3, 1.0)
                emotion_scores.append(EmotionScore(
                    emotion=emotion,
                    confidence=confidence,
                    intensity=confidence,
                    metadata={"keyword_matches": matches}
                ))
        
        return sorted(emotion_scores, key=lambda x: x.confidence, reverse=True)
    
    def _score_to_polarity(self, score: float) -> SentimentPolarity:
        """Convert numeric score to polarity category."""
        if score <= -0.6:
            return SentimentPolarity.VERY_NEGATIVE
        elif score <= -0.2:
            return SentimentPolarity.NEGATIVE
        elif score <= 0.2:
            return SentimentPolarity.NEUTRAL
        elif score <= 0.6:
            return SentimentPolarity.POSITIVE
        else:
            return SentimentPolarity.VERY_POSITIVE
    
    def _score_to_intensity(self, abs_score: float) -> IntensityLevel:
        """Convert absolute score to intensity level."""
        if abs_score <= 0.2:
            return IntensityLevel.VERY_LOW
        elif abs_score <= 0.4:
            return IntensityLevel.LOW
        elif abs_score <= 0.6:
            return IntensityLevel.MODERATE
        elif abs_score <= 0.8:
            return IntensityLevel.HIGH
        else:
            return IntensityLevel.VERY_HIGH
    
    def get_supported_languages(self) -> List[str]:
        """Get supported languages."""
        return ["en"]  # Extend based on available lexicons


class MLBasedAnalyzer(SentimentAnalyzer):
    """Machine learning-based sentiment analyzer."""
    
    def __init__(self, model_router: ModelRouter, tokenizer: EnhancedTokenizer, logger):
        self.model_router = model_router
        self.tokenizer = tokenizer
        self.logger = logger
        self.initialized = False
    
    async def initialize(self) -> None:
        """Initialize ML models."""
        try:
            # Models would be loaded here
            self.initialized = True
            self.logger.info("MLBasedAnalyzer initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize MLBasedAnalyzer: {str(e)}")
            raise SentimentAnalyzerError(f"ML initialization failed: {str(e)}")
    
    async def analyze(self, request: SentimentRequest) -> SentimentResult:
        """Analyze sentiment using ML models."""
        if not self.initialized:
            await self.initialize()
        
        start_time = time.time()
        
        try:
            # This would use actual ML models
            # For now, return a placeholder result
            result = SentimentResult(
                polarity_score=0.0,
                polarity_category=SentimentPolarity.NEUTRAL,
                confidence=0.8,
                intensity=IntensityLevel.MODERATE,
                processing_time=time.time() - start_time,
                models_used=["transformer_based"],
                analysis_mode=request.mode,
                session_id=request.session_id,
                user_id=request.user_id,
                conversation_id=request.conversation_id,
                language=request.language
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"ML-based analysis failed: {str(e)}")
            raise SentimentAnalyzerError(f"ML analysis failed: {str(e)}", "ML_ERROR")
    
    def get_supported_languages(self) -> List[str]:
        """Get supported languages."""
        return ["en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko"]


class ContextualAnalyzer:
    """Provides contextual analysis capabilities."""
    
    def __init__(self, memory_manager: MemoryManager, logger):
        self.memory_manager = memory_manager
        self.logger = logger
    
    async def analyze_context_shift(self, current_result: SentimentResult, 
                                  session_id: str) -> Dict[str, Any]:
        """Analyze sentiment shift in context."""
        try:
            # Get recent sentiment history
            history = await self.memory_manager.retrieve_memories(
                session_id=session_id,
                memory_type="sentiment_history",
                limit=10
            )
            
            if not history:
                return {"shift": 0.0, "trend": "stable", "volatility": 0.0}
            
            # Calculate sentiment shift
            recent_scores = [h.get("polarity_score", 0.0) for h in history[-5:]]
            if recent_scores:
                avg_recent = sum(recent_scores) / len(recent_scores)
                shift = current_result.polarity_score - avg_recent
                
                # Calculate volatility
                volatility = np.std(recent_scores) if len(recent_scores) > 1 else 0.0
                
                # Determine trend
                if shift > 0.2:
                    trend = "improving"
                elif shift < -0.2:
                    trend = "declining"
                else:
                    trend = "stable"
                
                return {
                    "shift": shift,
                    "trend": trend,
                    "volatility": volatility,
                    "baseline": avg_recent
                }
            
            return {"shift": 0.0, "trend": "stable", "volatility": 0.0}
            
        except Exception as e:
            self.logger.error(f"Context analysis failed: {str(e)}")
            return {"shift": 0.0, "trend": "stable", "volatility": 0.0, "error": str(e)}


class MoodTracker:
    """Tracks user mood over time."""
    
    def __init__(self, memory_manager: MemoryManager, logger):
        self.memory_manager = memory_manager
        self.logger = logger
        self.mood_profiles: Dict[str, MoodProfile] = {}
    
    async def update_mood(self, user_id: str, sentiment_result: SentimentResult) -> MoodProfile:
        """Update user's mood profile."""
        try:
            # Get or create mood profile
            if user_id not in self.mood_profiles:
                profile = await self._load_mood_profile(user_id)
                if not profile:
                    profile = MoodProfile(
                        user_id=user_id,
                        current_mood=sentiment_result.primary_emotion or EmotionCategory.NEUTRAL,
                        mood_stability=0.5
                    )
                self.mood_profiles[user_id] = profile
            
            profile = self.mood_profiles[user_id]
            
            # Update current mood
            if sentiment_result.primary_emotion:
                profile.current_mood = sentiment_result.primary_emotion
            
            # Add to recent moods
            profile.recent_moods.append((
                datetime.now(timezone.utc),
                sentiment_result.primary_emotion or EmotionCategory.NEUTRAL,
                sentiment_result.confidence
            ))
            
            # Keep only recent moods (last 50)
            if len(profile.recent_moods) > 50:
                profile.recent_moods = profile.recent_moods[-50:]
            
            # Update statistics
            await self._update_mood_statistics(profile, sentiment_result)
            
            # Save profile
            await self._save_mood_profile(profile)
            
            return profile
            
        except Exception as e:
            self.logger.error(f"Mood update failed for user {user_id}: {str(e)}")
            # Return default profile on error
            return MoodProfile(
                user_id=user_id,
                current_mood=EmotionCategory.NEUTRAL,
                mood_stability=0.5
            )
    
    async def _load_mood_profile(self, user_id: str) -> Optional[MoodProfile]:
        """Load mood profile from memory."""
        try:
            profiles = await self.memory_manager.retrieve_memories(
                user_id=user_id,
                memory_type="mood_profile",
                limit=1
            )
            
            if profiles:
                profile_data = profiles[0]
                # Convert data back to MoodProfile object
                # This would need proper deserialization
                return None  # Placeholder
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to load mood profile for {user_id}: {str(e)}")
            return None
    
    async def _save_mood_profile(self, profile: MoodProfile) -> None:
        """Save mood profile to memory."""
        try:
            # Convert profile to dictionary for storage
            profile_data = {
                "user_id": profile.user_id,
                "current_mood": profile.current_mood.value,
                "mood_stability": profile.mood_stability,
                "recent_moods": [(dt.isoformat(), mood.value, conf) 
                               for dt, mood, conf in profile.recent_moods[-20:]],
                "mood_patterns": profile.mood_patterns,
                "baseline_sentiment": profile.baseline_sentiment,
                "sentiment_variance": profile.sentiment_variance,
                "last_updated": profile.last_updated.isoformat(),
                "update_count": profile.update_count
            }
            
            await self.memory_manager.store_memory(
                content=json.dumps(profile_data),
                memory_type="mood_profile",
                user_id=profile.user_id,
                metadata={"profile_type": "mood", "version": "1.0"}
            )
            
        except Exception as e:
            self.logger.error(f"Failed to save mood profile: {str(e)}")
    
    async def _update_mood_statistics(self, profile: MoodProfile, 
                                    sentiment_result: SentimentResult) -> None:
        """Update mood profile statistics."""
        try:
            # Update baseline sentiment
            recent_scores = [sentiment_result.polarity_score]
            if profile.recent_moods:
                # This would calculate from actual sentiment scores
                pass
            
            profile.baseline_sentiment = sentiment_result.polarity_score
            profile.last_updated = datetime.now(timezone.utc)
            profile.update_count += 1
            
            # Calculate mood stability (simplified)
            if len(profile.recent_moods) >= 5:
                recent_moods = [mood for _, mood, _ in profile.recent_moods[-5:]]
                unique_moods = len(set(recent_moods))
                profile.mood_stability = 1.0 - (unique_moods - 1) / 4.0  # Normalize
            
        except Exception as e:
            self.logger.error(f"Failed to update mood statistics: {str(e)}")


class EnhancedSentimentAnalyzer:
    """
    Advanced sentiment analysis system for the AI assistant.
    
    Features:
    - Multi-strategy sentiment analysis (lexicon, ML, hybrid)
    - Emotion detection and classification
    - Temporal mood tracking and analysis
    - Contextual sentiment understanding
    - Cross-modal sentiment analysis
    - Personalized sentiment interpretation
    - Real-time sentiment monitoring
    - Sentiment-driven response adaptation
    """
    
    def __init__(self, container: Container):
        """Initialize the enhanced sentiment analyzer."""
        self.container = container
        self.logger = get_logger(__name__)
        self.config = container.get(ConfigLoader)
        self.event_bus = container.get(EventBus)
        self.error_handler = container.get(ErrorHandler)
        self.health_check = container.get(HealthCheck)
        
        # Core components
        self.tokenizer = container.get(EnhancedTokenizer)
        self.memory_manager = container.get(MemoryManager)
        self.feedback_processor = container.get(FeedbackProcessor)
        
        # Analysis components
        self.analyzers: Dict[str, SentimentAnalyzer] = {}
        self.contextual_analyzer: Optional[ContextualAnalyzer] = None
        self.mood_tracker: Optional[MoodTracker] = None
        
        # Setup monitoring and caching
        self._setup_monitoring()
        self._setup_caching()
        
        # Configuration
        self._default_mode = AnalysisMode(self.config.get("sentiment.default_mode", "balanced"))
        self._enable_caching = self.config.get("sentiment.enable_caching", True)
        self._cache_ttl = self.config.get("sentiment.cache_ttl_seconds", 3600)
        self._enable_mood_tracking = self.config.get("sentiment.enable_mood_tracking", True)
        
        # State tracking
        self._analysis_count = 0
        self._error_count = 0
        self._performance_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Register health check
        self.health_check.register_component("sentiment_analyzer", self._health_check_callback)
        
        self.logger.info("EnhancedSentimentAnalyzer initialized")

    def _setup_monitoring(self) -> None:
        """Setup monitoring and metrics collection."""
        try:
            self.metrics = self.container.get(MetricsCollector)
            self.tracer = self.container.get(TraceManager)
            
            # Register sentiment analysis metrics
            self.metrics.register_counter("sentiment_analyses_total")
            self.metrics.register_counter("sentiment_errors_total")
            self.metrics.register_histogram("sentiment_analysis_duration_seconds")
            self.metrics.register_gauge("sentiment_confidence_average")
            self.metrics.register_gauge("mood_tracking_active_users")
            
        except Exception as e:
            self.logger.warning(f"Failed to setup monitoring: {str(e)}")
            self.metrics = None
            self.tracer = None

    def _setup_caching(self) -> None:
        """Setup caching for sentiment analysis results."""
        try:
            self.cache_strategy = self.container.get(CacheStrategy)
        except Exception as e:
            self.logger.warning(f"Failed to setup caching: {str(e)}")
            self.cache_strategy = None

    async def initialize(self) -> None:
        """Initialize the sentiment analyzer."""
        try:
            self.logger.info("Initializing EnhancedSentimentAnalyzer...")
            
            # Initialize analyzers
            await self._setup_analyzers()
            
            # Initialize contextual components
            await self._setup_contextual_components()
            
            # Register event handlers
            await self._register_event_handlers()
            
            # Start background tasks
            await self._start_background_tasks()
            
            # Emit initialization event
            await self.event_bus.emit(ComponentInitialized(
                component_id="sentiment_analyzer",
                component_type="EnhancedSentimentAnalyzer"
            ))
            
            self.logger.info("EnhancedSentimentAnalyzer initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize sentiment analyzer: {str(e)}")
            await self.event_bus.emit(ComponentFailed(
                component_id="sentiment_analyzer",
                error_message=str(e)
            ))
            raise

    async def _setup_analyzers(self) -> None:
        """Setup sentiment analysis components."""
        try:
            # Setup lexicon-based analyzer
            lexicon_config = self.config.get("sentiment.lexicon", {})
            self.analyzers["lexicon"] = LexiconBasedAnalyzer(self.logger, lexicon_config)
            await self.analyzers["lexicon"].initialize()
            
            # Setup ML-based analyzer
            if self.config.get("sentiment.enable_ml", True):
                try:
                    model_router = self.container.get(ModelRouter)
                    self.analyzers["ml"] = MLBasedAnalyzer(model_router, self.tokenizer, self.logger)
                    await self.analyzers["ml"].initialize()
                except Exception as e:
                    self.logger.warning(f"ML analyzer not available: {str(e)}")
            
            self.logger.info(f"Initialized {len(self.analyzers)} sentiment analyzers")
            
        except Exception as e:
            self.logger.error(f"Failed to setup analyzers: {str(e)}")
            raise

    async def _setup_contextual_components(self) -> None:
        """Setup contextual analysis components."""
        try:
            # Setup contextual analyzer
            self.contextual_analyzer = ContextualAnalyzer(self.memory_manager, self.logger)
            
            # Setup mood tracker
            if self._enable_mood_tracking:
                self.mood_tracker = MoodTracker(self.memory_manager, self.logger)
            
            self.logger.info("Contextual components initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to setup contextual components: {str(e)}")
            raise

    async def _register_event_handlers(self) -> None:
        """Register event handlers."""
        try:
            # Register handlers for relevant events
            self.event_bus.subscribe("user_feedback", self._handle_user_feedback)
            self.event_bus.subscribe("session_started", self._handle_session_started)
            self.event_bus.subscribe("session_ended", self._handle_session_ended)
            
        except Exception as e:
            self.logger.error(f"Failed to register event handlers: {str(e)}")

    async def _start_background_tasks(self) -> None:
        """Start background monitoring tasks."""
        try:
            # Start performance monitoring
            asyncio.create_task(self._performance_monitoring_loop())
            
            # Start mood analysis if enabled
            if self._enable_mood_tracking:
                asyncio.create_task(self._mood_analysis_loop())
            
        except Exception as e:
            self.logger.error(f"Failed to start background tasks: {str(e)}")

    @handle_exceptions
    async def analyze_sentiment(self, request: SentimentRequest) -> SentimentResult:
        """
        Analyze sentiment in the given text.
        
        Args:
            request: Sentiment analysis request
            
        Returns:
            Comprehensive sentiment analysis result
        """
        if not self.analyzers:
            raise SentimentAnalyzerError("No analyzers available")
        
        start_time = time.time()
        
        with self.tracer.trace("sentiment_analysis") if self.tracer else None:
            try:
                # Check cache first
                cache_key = None
                if self._enable_caching and self.cache_strategy:
                    cache_key = self._generate_cache_key(request)
                    cached_result = await self._get_cached_result(cache_key)
                    if cached_result:
                        return cached_result
                
                # Perform analysis based on mode
                if request.mode == AnalysisMode.FAST:
                    result = await self._analyze_fast(request)
                elif request.mode == AnalysisMode.COMPREHENSIVE:
                    result = await self._analyze_comprehensive(request)
                elif request.mode == AnalysisMode.CONTEXTUAL:
                    result = await self._analyze_contextual(request)
                else:
                    result = await self._analyze_balanced(request)
                
                # Enhance with contextual analysis
                if self.contextual_analyzer and request.session_id:
                    context_analysis = await self.contextual_analyzer.analyze_context_shift(
                        result, request.session_id
                    )
                    result.sentiment_shift = context_analysis.get("shift")
                    result.emotional_volatility = context_analysis.get("volatility", 0.0)
                    result.temporal_trends = context_analysis
                
                # Update mood tracking
                if self.mood_tracker and request.user_id:
                    mood_profile = await self.mood_tracker.update_mood(request.user_id, result)
                    # Emit mood change event if significant
                    if abs(result.sentiment_shift or 0.0) > 0.3:
                        await self.event_bus.emit(MoodChanged(
                            user_id=request.user_id,
                            previous_mood=mood_profile.mood_patterns.get("previous", "neutral"),
                            current_mood=mood_profile.current_mood.value,
                            confidence=result.confidence
                        ))
                
                # Store result and learn
                await self._store_analysis_result(request, result)
                
                # Cache result
                if cache_key and self.cache_strategy:
                    await self._cache_result(cache_key, result)
                
                # Update metrics
                self._update_metrics(result, time.time() - start_time)
                
                # Emit analysis event
                await self.event_bus.emit(SentimentAnalyzed(
                    session_id=request.session_id,
                    user_id=request.user_id,
                    text=request.text[:100],  # Truncated for privacy
                    polarity=result.polarity_category.value,
                    confidence=result.confidence,
                    primary_emotion=result.primary_emotion.value if result.primary_emotion else None
                ))
                
                return result
                
            except Exception as e:
                self._error_count += 1
                if self.metrics:
                    self.metrics.increment("sentiment_errors_total")
                
                self.logger.error(f"Sentiment analysis failed: {str(e)}")
                raise SentimentAnalyzerError(f"Analysis failed: {str(e)}")

    async def _analyze_fast(self, request: SentimentRequest) -> SentimentResult:
        """Fast sentiment analysis using lexicon-based approach."""
        if "lexicon" not in self.analyzers:
            raise SentimentAnalyzerError("Lexicon analyzer not available")
        
        return await self.analyzers["lexicon"].analyze(request)

    async def _analyze_balanced(self, request: SentimentRequest) -> SentimentResult:
        """Balanced sentiment analysis using available analyzers."""
        results = []
        
        # Try lexicon-based first (fastest)
        if "lexicon" in self.analyzers:
            try:
                lexicon_result = await self.analyzers["lexicon"].analyze(request)
                results.append(("lexicon", lexicon_result))
            except Exception as e:
                self.logger.warning(f"Lexicon analysis failed: {str(e)}")
        
        # Try ML-based if available and needed
        if "ml" in self.analyzers and len(results) == 0:
            try:
                ml_result = await self.analyzers["ml"].analyze(request)
                results.append(("ml", ml_result))
            except Exception as e:
                self.logger.warning(f"ML analysis failed: {str(e)}")
        
        if not results:
            raise SentimentAnalyzerError("No analyzer produced results")
        
        # Return the best result (for now, just the first)
        return results[0][1]

    async def _analyze_comprehensive(self, request: SentimentRequest) -> SentimentResult:
        """Comprehensive sentiment analysis using all available analyzers."""
        results = []
        
        # Run all available analyzers
        for analyzer_name, analyzer in self.analyzers.items():
            try:
                result = await analyzer.analyze(request)
                results.append((analyzer_name, result))
            except Exception as e:
                self.logger.warning(f"{analyzer_name} analysis failed: {str(e)}")
        
        if not results:
            raise SentimentAnalyzerError("No analyzer produced results")
        
        # Ensemble the results
        return await self._ensemble_results(results, request)

    async def _analyze_contextual(self, request: SentimentRequest) -> SentimentResult:
        """Contextual sentiment analysis with enhanced context awareness."""
        # Start with balanced analysis
        result = await self._analyze_balanced(request)
        
        # Enhance with context if available
        if request.interaction_context:
            # Adjust based on conversation context
            context = request.interaction_context
            
            # Consider conversation history
            if context.get("conversation_history"):
                # This would analyze conversation flow and adjust sentiment
                pass
            
            # Consider user emotional state
            if context.get("emotional_state"):
                # This would factor in user's emotional context
                pass
        
        return result

    async def _ensemble_results(self, results: List[Tuple[str, SentimentResult]], 
                               request: SentimentRequest) -> SentimentResult:
        """Ensemble multiple sentiment analysis results."""
        if len(results) == 1:
            return results[0][1]
        
        # Weight the results based on confidence and analyzer type
        weights = {
            "lexicon": 0.3,
            "ml": 0.7
        }
        
        total_weight = 0.0
        weighted_polarity = 0.0
        all_emotions = []
        all_models = []
        
        for analyzer_name, result in results:
            weight = weights.get(analyzer_name, 0.5) * result.confidence
            weighted_polarity += result.polarity_score * weight
            total_weight += weight
            all_emotions.extend(result.emotion_scores)
            all_models.extend(result.models_used)
        
        # Calculate ensemble polarity
        if total_weight > 0:
            ensemble_polarity = weighted_polarity / total_weight
        else:
            ensemble_polarity = 0.0
        
        # Use the highest confidence result as base
        best_result = max(results, key=lambda x: x[1].confidence)[1]
        
        # Create ensemble result
        ensemble_result = SentimentResult(
            polarity_score=ensemble_polarity,
            polarity_category=self._score_to_polarity(ensemble_polarity),
            confidence=min(total_weight, 1.0),
            intensity=self._score_to_intensity(abs(ensemble_polarity)),
            primary_emotion=best_result.primary_emotion,
            emotion_scores=sorted(all_emotions, key=lambda x: x.confidence, reverse=True)[:5],
            emotion_confidence=best_result.emotion_confidence,
            processing_time=max(r[1].processing_time for r in results),
            models_used=list(set(all_models)),
            analysis_mode=request.mode,
            session_id=request.session_id,
            user_id=request.user_id,
            conversation_id=request.conversation_id,
            language=request.language,
            model_agreement_score=self._calculate_agreement_score(results)
        )
        
        return ensemble_result

    def _calculate_agreement_score(self, results: List[Tuple[str, SentimentResult]]) -> float:
        """Calculate agreement score between different analyzers."""
        if len(results) <= 1:
            return 1.0
        
        polarities = [r[1].polarity_score for r in results]
        std_dev = np.std(polarities)
        
        # Convert standard deviation to agreement score (0-1)
        agreement = max(0.0, 1.0 - std_dev)
        return agreement

    def _score_to_polarity(self, score: float) -> SentimentPolarity:
        """Convert numeric score to polarity category."""
        if score <= -0.6:
            return SentimentPolarity.VERY_NEGATIVE
        elif score <= -0.2:
            return SentimentPolarity.NEGATIVE
        elif score <= 0.2:
            return SentimentPolarity.NEUTRAL
        elif score <= 0.6:
            return SentimentPolarity.POSITIVE
        else:
            return SentimentPolarity.VERY_POSITIVE

    def _score_to_intensity(self, abs_score: float) -> IntensityLevel:
        """Convert absolute score to intensity level."""
        if abs_score <= 0.2:
            return IntensityLevel.VERY_LOW
        elif abs_score <= 0.4:
            return IntensityLevel.LOW
        elif abs_score <= 0.6:
            return IntensityLevel.MODERATE
        elif abs_score <= 0.8:
            return IntensityLevel.HIGH
        else:
            return IntensityLevel.VERY_HIGH

    def _generate_cache_key(self, request: SentimentRequest) -> str:
        """Generate cache key for sentiment analysis request."""
        import hashlib
        
        # Create a hash of the request parameters
        key_components = [
            request.text,
            request.mode.value,
            request.context_type.value,
            request.language,
            str(request.enable_emotion_detection),
            str(request.enable_mood_tracking)
        ]
        
        key_string = "|".join(key_components)
        return f"sentiment:{hashlib.md5(key_string.encode()).hexdigest()}"

    async def _get_cached_result(self, cache_key: str) -> Optional[SentimentResult]:
        """Get cached sentiment analysis result."""
        try:
            if self.cache_strategy:
                cached_data = await self.cache_strategy.get(cache_key)
                if cached_data:
                    # Deserialize the cached result
                    # This would need proper deserialization
                    return None  # Placeholder
            return None
        except Exception as e:
            self.logger.warning(f"Cache retrieval failed: {str(e)}")
            return None

    async def _cache_result(self, cache_key: str, result: SentimentResult) -> None:
        """Cache sentiment analysis result."""
        try:
            if self.cache_strategy:
                # Serialize the result for caching
                # This would need proper serialization
                await self.cache_strategy.set(cache_key, {}, ttl=self._cache_ttl)
        except Exception as e:
            self.logger.warning(f"Cache storage failed: {str(e)}")

    async def _store_analysis_result(self, request: SentimentRequest, 
                                   result: SentimentResult) -> None:
        """Store analysis result for learning and context."""
        try:
            # Store in memory for contextual analysis
            if request.session_id:
                memory_data = {
                    "polarity_score": result.polarity_score,
                    "polarity_category": result.polarity_category.value,
                    "confidence": result.confidence,
                    "primary_emotion": result.primary_emotion.value if result.primary_emotion else None,
                    "analysis_timestamp": result.timestamp.isoformat(),
                    "text_sample": request.text[:200]  # Store sample for context
                }
                
                await self.memory_manager.store_memory(
                    content=json.dumps(memory_data),
                    memory_type="sentiment_history",
                    session_id=request.session_id,
                    user_id=request.user_id,
                    metadata={"analysis_mode": request.mode.value}
                )
            
            # Store for learning if feedback processor is available
            if self.feedback_processor:
                await self.feedback_processor.process_interaction_data({
                    "type": "sentiment_analysis",
                    "request": {
                        "text": request.text,
                        "mode": request.mode.value,
                        "context_type": request.context_type.value
                    },
                    "result": {
                        "polarity_score": result.polarity_score,
                        "confidence": result.confidence,
                        "processing_time": result.processing_time
                    }
                })
            
        except Exception as e:
            self.logger.error(f"Failed to store analysis result: {str(e)}")

    def _update_metrics(self, result: SentimentResult, processing_time: float) -> None:
        """Update performance metrics."""
        try:
            self._analysis_count += 1
            
            # Track performance
            self._performance_metrics["processing_time"].append(processing_time)
            self._performance_metrics["confidence"].append(result.confidence)
            
            if self.metrics:
                self.metrics.increment("sentiment_analyses_total")
                self.metrics.record("sentiment_analysis_duration_seconds", processing_time)
                self.metrics.set("sentiment_confidence_average", result.confidence)
            
        except Exception as e:
            self.logger.error(f"Failed to update metrics: {str(e)}")

    async def get_user_mood_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user's mood profile."""
        try:
            if self.mood_tracker and user_id in self.mood_tracker.mood_profiles:
                profile = self.mood_tracker.mood_profiles[user_id]
                return {
                    "current_mood": profile.current_mood.value,
                    "mood_stability": profile.mood_stability,
                    "baseline_sentiment": profile.baseline_sentiment,
                    "recent_moods": [(dt.isoformat(), mood.value, conf) 
                                   for dt, mood, conf in profile.recent_moods[-10:]],
                    "last_updated": profile.last_updated.isoformat()
                }
            return None
        except Exception as e:
            self.logger.error(f"Failed to get mood profile for {user_id}: {str(e)}")
            return None

    def get_analysis_statistics(self) -> Dict[str, Any]:
        """Get sentiment analysis statistics."""
        try:
            stats = {
                "total_analyses": self._analysis_count,
                "error_count": self._error_count,
                "available_analyzers": list(self.analyzers.keys()),
                "supported_languages": list(set().union(*[
                    analyzer.get_supported_languages() 
                    for analyzer in self.analyzers.values()
                ])),
                "mood_tracking_enabled": self._enable_mood_tracking
            }
            
            # Add performance metrics
            if self._performance_metrics["processing_time"]:
                times = list(self._performance_metrics["processing_time"])
                stats["performance"] = {
                    "avg_processing_time": sum(times) / len(times),
                    "min_processing_time": min(times),
                    "max_processing_time": max(times)
                }
            
            if self._performance_metrics["confidence"]:
                confidences = list(self._performance_metrics["confidence"])
                stats["quality"] = {
                    "avg_confidence": sum(confidences) / len(confidences),
                    "min_confidence": min(confidences),
                    "max_confidence": max(confidences)
                }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to get statistics: {str(e)}")
            return {"error": str(e)}

    # Background task methods
    async def _performance_monitoring_loop(self) -> None:
        """Monitor performance and adjust if needed."""
        while True:
            try:
                await asyncio.sleep(300)  # Every 5 minutes
                
                # Check performance metrics
                if self._performance_metrics["processing_time"]:
                    avg_time = sum(self._performance_metrics["processing_time"]) / len(self._performance_metrics["processing_time"])
                    
                    # Log if performance is degrading
                    if avg_time > 5.0:  # 5 seconds threshold
                        self.logger.warning(f"Sentiment analysis performance degrading: {avg_time:.2f}s average")
                
                # Clear old metrics
                for metric_queue in self._performance_metrics.values():
                    if len(metric_queue) > 500:
                        # Keep only recent metrics
                        for _ in range(100):
                            if metric_queue:
                                metric_queue.popleft()
                
            except Exception as e:
                self.logger.error(f"Performance monitoring failed: {str(e)}")

    async def _mood_analysis_loop(self) -> None:
        """Analyze mood patterns and trends."""
        while True:
            try:
                await asyncio.sleep(3600)  # Every hour
                
                if self.mood_tracker:
                    # Analyze mood patterns for insights
                    for user_id, profile in self.mood_tracker.mood_profiles.items():
                        if len(profile.recent_moods) >= 10:
                            # Detect mood patterns
                            moods = [mood for _, mood, _ in profile.recent_moods[-24:]]  # Last 24 entries
                            
                            # Simple pattern detection
                            mood_changes = sum(1 for i in range(1, len(moods)) if moods[i] != moods[i-1])
                            
                            if mood_changes > len(moods) * 0.7:  # High volatility
                                self.logger.info(f"High mood volatility detected for user {user_id}")
                
            except Exception as e:
                self.logger.error(f"Mood analysis failed: {str(e)}")

    # Event handlers
    async def _handle_user_feedback(self, event) -> None:
        """Handle user feedback on sentiment analysis."""
        try:
            feedback_data = event.data
            if feedback_data.get("component") == "sentiment_analyzer":
                # Process feedback for model improvement
                await self.feedback_processor.process_feedback(feedback_data)
                
        except Exception as e:
            self.logger.error(f"Failed to handle user feedback: {str(e)}")

    async def _handle_session_started(self, event) -> None:
        """Handle session start event."""
        try:
            # Initialize session-specific sentiment tracking
            session_id = event.session_id
            if session_id:
                self.logger.debug(f"Starting sentiment tracking for session {session_id}")
                
        except Exception as e:
            self.logger.error(f"Failed to handle session start: {str(e)}")

    async def _handle_session_ended(self, event) -> None:
        """Handle session end event."""
        try:
            # Finalize session sentiment analysis
            session_id = event.session_id
            if session_id:
                # Generate session sentiment summary
                summary = await self._generate_session_sentiment_summary(session_id)
                if summary:
                    self.logger.info(f"Session {session_id} sentiment summary: {summary}")
                
        except Exception as e:
            self.logger.error(f"Failed to handle session end: {str(e)}")

    async def _generate_session_sentiment_summary(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Generate sentiment summary for a session."""
        try:
            # Get session sentiment history
            history = await self.memory_manager.retrieve_memories(
                session_id=session_id,
                memory_type="sentiment_history",
                limit=100
            )
            
            if not history:
                return None
            
            # Calculate summary statistics
            scores = [h.get("polarity_score", 0.0) for h in history]
            emotions = [h.get("primary_emotion") for h in history if h.get("primary_emotion")]
            
            summary = {
                "total_analyses": len(history),
                "avg_sentiment": sum(scores) / len(scores) if scores else 0.0,
                "common_emotions": emotions[:5] if emotions else [],
                "time_range": {
                    "start": history[0].get("timestamp"),
                    "end": history[-1].get("timestamp")
                } if history else {}
            }
