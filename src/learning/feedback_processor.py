"""
Advanced Feedback Processing System for AI Assistant
Author: Drmusab
Last Modified: 2025-06-20 03:14:17 UTC

This module provides comprehensive feedback processing capabilities for the AI assistant,
including real-time feedback analysis, sentiment processing, pattern recognition,
continuous learning, and seamless integration with all core system components.
"""

from pathlib import Path
from typing import Optional, Dict, Any, List, Set, Callable, Type, Union, AsyncGenerator, TypeVar
import asyncio
import threading
import time
import json
import hashlib
import numpy as np
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
from contextlib import asynccontextmanager
import uuid
from collections import defaultdict, deque
import weakref
from abc import ABC, abstractmethod
import logging
import statistics
from concurrent.futures import ThreadPoolExecutor

# Core imports
from src.core.config.loader import ConfigLoader
from src.core.events.event_bus import EventBus
from src.core.events.event_types import (
    FeedbackReceived, FeedbackProcessed, FeedbackAnalyzed, LearningEventOccurred,
    UserPreferenceUpdated, ModelAdaptationStarted, ModelAdaptationCompleted,
    QualityMetricUpdated, BiasDetected, FeedbackAggregated, SystemOptimized,
    ErrorOccurred, ComponentHealthChanged, SessionStarted, SessionEnded,
    WorkflowCompleted, InteractionCompleted, SkillExecutionCompleted
)
from src.core.error_handling import ErrorHandler, handle_exceptions
from src.core.dependency_injection import Container
from src.core.health_check import HealthCheck
from src.core.security.sanitization import InputSanitizer

# Processing components
from src.processing.natural_language.sentiment_analyzer import SentimentAnalyzer
from src.processing.natural_language.entity_extractor import EntityExtractor
from src.processing.speech.emotion_detection import EnhancedEmotionDetector
from src.processing.vision.vision_processor import VisionProcessor

# Memory systems
from src.memory.core_memory.memory_manager import MemoryManager
from src.memory.operations.context_manager import ContextManager
from src.memory.core_memory.memory_types import EpisodicMemory, SemanticMemory
from src.memory.storage.vector_store import VectorStore

# Learning systems
from src.learning.continual_learning import ContinualLearner
from src.learning.preference_learning import PreferenceLearner
from src.learning.model_adaptation import ModelAdapter

# Integrations
from src.integrations.llm.model_router import ModelRouter
from src.integrations.cache.cache_strategy import CacheStrategy
from src.integrations.storage.database import DatabaseManager

# Observability
from src.observability.monitoring.metrics import MetricsCollector
from src.observability.monitoring.tracing import TraceManager
from src.observability.logging.config import get_logger

# Type definitions
T = TypeVar('T')


class FeedbackType(Enum):
    """Types of feedback that can be processed."""
    RATING = "rating"                    # Numerical ratings (1-5, 1-10, etc.)
    TEXT = "text"                       # Textual feedback
    VOICE = "voice"                     # Audio feedback
    GESTURE = "gesture"                 # Gesture-based feedback
    BEHAVIORAL = "behavioral"           # Implicit behavioral feedback
    EXPLICIT = "explicit"               # Explicit user feedback
    IMPLICIT = "implicit"               # Implicit feedback from actions
    CORRECTION = "correction"           # Error corrections
    SUGGESTION = "suggestion"           # User suggestions
    COMPLAINT = "complaint"             # User complaints
    COMPLIMENT = "compliment"           # User compliments
    FEATURE_REQUEST = "feature_request" # Feature requests


class FeedbackCategory(Enum):
    """Categories of feedback for classification."""
    QUALITY = "quality"                 # Overall quality feedback
    ACCURACY = "accuracy"               # Accuracy-related feedback
    RELEVANCE = "relevance"             # Relevance of responses
    SPEED = "speed"                     # Performance speed feedback
    USABILITY = "usability"             # Usability feedback
    CONTENT = "content"                 # Content-related feedback
    INTERFACE = "interface"             # UI/UX feedback
    FUNCTIONALITY = "functionality"     # Feature functionality
    PERSONALIZATION = "personalization" # Personalization quality
    SAFETY = "safety"                   # Safety and appropriateness
    PRIVACY = "privacy"                 # Privacy concerns
    ACCESSIBILITY = "accessibility"     # Accessibility feedback


class FeedbackPriority(Enum):
    """Priority levels for feedback processing."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    URGENT = 3
    CRITICAL = 4


class FeedbackSentiment(Enum):
    """Sentiment classifications for feedback."""
    VERY_NEGATIVE = -2
    NEGATIVE = -1
    NEUTRAL = 0
    POSITIVE = 1
    VERY_POSITIVE = 2


class ProcessingStatus(Enum):
    """Status of feedback processing."""
    RECEIVED = "received"
    VALIDATING = "validating"
    PROCESSING = "processing"
    ANALYZING = "analyzing"
    LEARNING = "learning"
    COMPLETED = "completed"
    FAILED = "failed"
    IGNORED = "ignored"


@dataclass
class FeedbackMetrics:
    """Metrics extracted from feedback."""
    sentiment_score: float = 0.0
    emotion_scores: Dict[str, float] = field(default_factory=dict)
    confidence_score: float = 0.0
    quality_score: float = 0.0
    relevance_score: float = 0.0
    satisfaction_score: float = 0.0
    effort_score: float = 0.0  # How much effort user had to put in
    clarity_score: float = 0.0  # How clear the feedback is
    actionability_score: float = 0.0  # How actionable the feedback is
    bias_indicators: Dict[str, float] = field(default_factory=dict)
    urgency_score: float = 0.0
    impact_score: float = 0.0


@dataclass
class FeedbackContext:
    """Context information for feedback."""
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    interaction_id: Optional[str] = None
    workflow_id: Optional[str] = None
    component_id: Optional[str] = None
    skill_id: Optional[str] = None
    
    # Contextual data
    user_intent: Optional[str] = None
    task_completed: bool = False
    response_time: float = 0.0
    user_effort: str = "low"  # low, medium, high
    system_confidence: float = 0.0
    
    # Environmental context
    device_type: Optional[str] = None
    interface_mode: Optional[str] = None
    time_of_day: Optional[str] = None
    user_state: Optional[str] = None  # busy, relaxed, frustrated, etc.
    
    # Previous interactions
    recent_interactions: List[str] = field(default_factory=list)
    user_history_summary: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FeedbackData:
    """Comprehensive feedback data structure."""
    feedback_id: str
    feedback_type: FeedbackType
    category: FeedbackCategory
    priority: FeedbackPriority = FeedbackPriority.NORMAL
    
    # Content
    content: Any = None  # Can be text, audio, image, etc.
    rating: Optional[float] = None
    scale: Optional[str] = None  # "1-5", "1-10", "thumbs", etc.
    
    # Metadata
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    language: str = "en"
    source: str = "user"  # user, system, automated
    
    # Context
    context: FeedbackContext = field(default_factory=FeedbackContext)
    
    # Processing results
    metrics: FeedbackMetrics = field(default_factory=FeedbackMetrics)
    extracted_entities: List[Dict[str, Any]] = field(default_factory=list)
    processed_content: Dict[str, Any] = field(default_factory=dict)
    
    # Classification
    sentiment: FeedbackSentiment = FeedbackSentiment.NEUTRAL
    emotions: Dict[str, float] = field(default_factory=dict)
    topics: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    
    # Quality assessment
    is_valid: bool = True
    quality_issues: List[str] = field(default_factory=list)
    confidence: float = 0.0
    
    # Processing status
    status: ProcessingStatus = ProcessingStatus.RECEIVED
    processing_time: float = 0.0
    errors: List[str] = field(default_factory=list)


@dataclass
class FeedbackPattern:
    """Identified pattern in feedback data."""
    pattern_id: str
    pattern_type: str  # sentiment, topic, temporal, user_behavior
    description: str
    frequency: int
    confidence: float
    impact_score: float
    
    # Pattern data
    examples: List[str] = field(default_factory=list)
    time_range: Optional[Dict[str, datetime]] = None
    user_segments: List[str] = field(default_factory=list)
    
    # Recommendations
    recommended_actions: List[str] = field(default_factory=list)
    priority: FeedbackPriority = FeedbackPriority.NORMAL


@dataclass
class FeedbackInsights:
    """Insights derived from feedback analysis."""
    insights_id: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Aggregate metrics
    total_feedback_count: int = 0
    satisfaction_trends: Dict[str, float] = field(default_factory=dict)
    sentiment_distribution: Dict[str, int] = field(default_factory=dict)
    category_breakdown: Dict[str, int] = field(default_factory=dict)
    
    # Quality metrics
    response_quality_score: float = 0.0
    user_experience_score: float = 0.0
    system_performance_score: float = 0.0
    
    # Patterns and trends
    identified_patterns: List[FeedbackPattern] = field(default_factory=list)
    trending_topics: List[str] = field(default_factory=list)
    emerging_issues: List[str] = field(default_factory=list)
    
    # Recommendations
    improvement_recommendations: List[str] = field(default_factory=list)
    critical_issues: List[str] = field(default_factory=list)
    quick_wins: List[str] = field(default_factory=list)


class FeedbackError(Exception):
    """Custom exception for feedback processing operations."""
    
    def __init__(self, message: str, feedback_id: Optional[str] = None, 
                 error_code: Optional[str] = None):
        super().__init__(message)
        self.feedback_id = feedback_id
        self.error_code = error_code
        self.timestamp = datetime.now(timezone.utc)


class FeedbackValidator:
    """Validates feedback data quality and authenticity."""
    
    def __init__(self, input_sanitizer: InputSanitizer):
        self.input_sanitizer = input_sanitizer
        self.logger = get_logger(__name__)
    
    async def validate_feedback(self, feedback: FeedbackData) -> bool:
        """Validate feedback data and mark quality issues."""
        try:
            feedback.quality_issues = []
            
            # Content validation
            if not await self._validate_content(feedback):
                return False
            
            # Context validation
            if not self._validate_context(feedback):
                return False
            
            # Rating validation
            if not self._validate_rating(feedback):
                return False
            
            # Spam detection
            if await self._detect_spam(feedback):
                feedback.quality_issues.append("potential_spam")
                return False
            
            # Bias detection
            bias_score = await self._detect_bias(feedback)
            if bias_score > 0.7:
                feedback.quality_issues.append("potential_bias")
                feedback.metrics.bias_indicators["general"] = bias_score
            
            return len(feedback.quality_issues) == 0
            
        except Exception as e:
            self.logger.error(f"Error validating feedback {feedback.feedback_id}: {str(e)}")
            feedback.quality_issues.append(f"validation_error: {str(e)}")
            return False
    
    async def _validate_content(self, feedback: FeedbackData) -> bool:
        """Validate feedback content."""
        if feedback.feedback_type == FeedbackType.TEXT:
            if not feedback.content or not isinstance(feedback.content, str):
                feedback.quality_issues.append("empty_or_invalid_text_content")
                return False
            
            # Sanitize content
            feedback.content = await self.input_sanitizer.sanitize_text(feedback.content)
            
            # Check minimum length
            if len(feedback.content.strip()) < 3:
                feedback.quality_issues.append("content_too_short")
                return False
        
        elif feedback.feedback_type == FeedbackType.RATING:
            if feedback.rating is None:
                feedback.quality_issues.append("missing_rating")
                return False
        
        return True
    
    def _validate_context(self, feedback: FeedbackData) -> bool:
        """Validate context information."""
        # Basic context validation
        if not feedback.context.session_id and not feedback.context.user_id:
            feedback.quality_issues.append("insufficient_context")
            return False
        
        return True
    
    def _validate_rating(self, feedback: FeedbackData) -> bool:
        """Validate rating values."""
        if feedback.rating is not None:
            # Determine scale
            if feedback.scale == "1-5" and not (1 <= feedback.rating <= 5):
                feedback.quality_issues.append("rating_out_of_range")
                return False
            elif feedback.scale == "1-10" and not (1 <= feedback.rating <= 10):
                feedback.quality_issues.append("rating_out_of_range")
                return False
        
        return True
    
    async def _detect_spam(self, feedback: FeedbackData) -> bool:
        """Detect potential spam in feedback."""
        if feedback.feedback_type == FeedbackType.TEXT:
            content = feedback.content.lower()
            
            # Check for repeated characters
            if any(char * 5 in content for char in "abcdefghijklmnopqrstuvwxyz"):
                return True
            
            # Check for promotional content
            spam_indicators = ["buy now", "click here", "free money", "guaranteed"]
            if any(indicator in content for indicator in spam_indicators):
                return True
        
        return False
    
    async def _detect_bias(self, feedback: FeedbackData) -> float:
        """Detect potential bias in feedback."""
        bias_score = 0.0
        
        if feedback.feedback_type == FeedbackType.TEXT:
            content = feedback.content.lower()
            
            # Simple bias detection (would be more sophisticated in practice)
            bias_patterns = [
                "always", "never", "all", "none", "every", "worst", "best", "terrible", "perfect"
            ]
            
            bias_count = sum(1 for pattern in bias_patterns if pattern in content)
            bias_score = min(bias_count / len(bias_patterns), 1.0)
        
        return bias_score


class FeedbackAnalyzer:
    """Analyzes feedback content to extract insights and metrics."""
    
    def __init__(self, sentiment_analyzer: SentimentAnalyzer, 
                 emotion_detector: Optional[EnhancedEmotionDetector] = None,
                 entity_extractor: Optional[EntityExtractor] = None):
        self.sentiment_analyzer = sentiment_analyzer
        self.emotion_detector = emotion_detector
        self.entity_extractor = entity_extractor
        self.logger = get_logger(__name__)
    
    async def analyze_feedback(self, feedback: FeedbackData) -> None:
        """Perform comprehensive analysis of feedback."""
        try:
            feedback.status = ProcessingStatus.ANALYZING
            
            # Sentiment analysis
            await self._analyze_sentiment(feedback)
            
            # Emotion detection
            if self.emotion_detector:
                await self._analyze_emotions(feedback)
            
            # Entity extraction
            if self.entity_extractor:
                await self._extract_entities(feedback)
            
            # Topic extraction
            await self._extract_topics(feedback)
            
            # Quality metrics calculation
            await self._calculate_quality_metrics(feedback)
            
            # Impact assessment
            await self._assess_impact(feedback)
            
        except Exception as e:
            self.logger.error(f"Error analyzing feedback {feedback.feedback_id}: {str(e)}")
            feedback.errors.append(f"analysis_error: {str(e)}")
            feedback.status = ProcessingStatus.FAILED
    
    async def _analyze_sentiment(self, feedback: FeedbackData) -> None:
        """Analyze sentiment of feedback content."""
        if feedback.feedback_type == FeedbackType.TEXT:
            sentiment_result = await self.sentiment_analyzer.analyze(feedback.content)
            
            feedback.metrics.sentiment_score = sentiment_result.get("compound", 0.0)
            
            # Map to sentiment enum
            if feedback.metrics.sentiment_score >= 0.6:
                feedback.sentiment = FeedbackSentiment.VERY_POSITIVE
            elif feedback.metrics.sentiment_score >= 0.2:
                feedback.sentiment = FeedbackSentiment.POSITIVE
            elif feedback.metrics.sentiment_score <= -0.6:
                feedback.sentiment = FeedbackSentiment.VERY_NEGATIVE
            elif feedback.metrics.sentiment_score <= -0.2:
                feedback.sentiment = FeedbackSentiment.NEGATIVE
            else:
                feedback.sentiment = FeedbackSentiment.NEUTRAL
        
        elif feedback.feedback_type == FeedbackType.RATING:
            # Convert rating to sentiment
            if feedback.scale == "1-5":
                normalized_rating = (feedback.rating - 1) / 4  # 0-1 scale
            elif feedback.scale == "1-10":
                normalized_rating = (feedback.rating - 1) / 9  # 0-1 scale
            else:
                normalized_rating = feedback.rating
            
            feedback.metrics.sentiment_score = (normalized_rating - 0.5) * 2  # -1 to 1 scale
            
            if normalized_rating >= 0.8:
                feedback.sentiment = FeedbackSentiment.VERY_POSITIVE
            elif normalized_rating >= 0.6:
                feedback.sentiment = FeedbackSentiment.POSITIVE
            elif normalized_rating <= 0.2:
                feedback.sentiment = FeedbackSentiment.VERY_NEGATIVE
            elif normalized_rating <= 0.4:
                feedback.sentiment = FeedbackSentiment.NEGATIVE
            else:
                feedback.sentiment = FeedbackSentiment.NEUTRAL
    
    async def _analyze_emotions(self, feedback: FeedbackData) -> None:
        """Analyze emotions in feedback content."""
        if feedback.feedback_type == FeedbackType.TEXT:
            # Text-based emotion detection would go here
            # For now, derive basic emotions from sentiment
            if feedback.sentiment == FeedbackSentiment.VERY_POSITIVE:
                feedback.emotions = {"joy": 0.8, "satisfaction": 0.9}
            elif feedback.sentiment == FeedbackSentiment.POSITIVE:
                feedback.emotions = {"joy": 0.6, "satisfaction": 0.7}
            elif feedback.sentiment == FeedbackSentiment.VERY_NEGATIVE:
                feedback.emotions = {"anger": 0.7, "frustration": 0.8}
            elif feedback.sentiment == FeedbackSentiment.NEGATIVE:
                feedback.emotions = {"disappointment": 0.6, "frustration": 0.5}
            else:
                feedback.emotions = {"neutral": 0.8}
            
            feedback.metrics.emotion_scores = feedback.emotions
    
    async def _extract_entities(self, feedback: FeedbackData) -> None:
        """Extract entities from feedback content."""
        if feedback.feedback_type == FeedbackType.TEXT and self.entity_extractor:
            entities = await self.entity_extractor.extract(feedback.content)
            feedback.extracted_entities = entities
            
            # Extract keywords
            feedback.keywords = [entity.get("text", "") for entity in entities 
                               if entity.get("label") in ["PRODUCT", "FEATURE", "ISSUE"]]
    
    async def _extract_topics(self, feedback: FeedbackData) -> None:
        """Extract topics from feedback content."""
        if feedback.feedback_type == FeedbackType.TEXT:
            content = feedback.content.lower()
            
            # Simple topic extraction (would use more sophisticated NLP in practice)
            topic_keywords = {
                "performance": ["slow", "fast", "speed", "lag", "responsive"],
                "usability": ["easy", "difficult", "confusing", "intuitive", "user-friendly"],
                "quality": ["good", "bad", "excellent", "poor", "quality"],
                "accuracy": ["correct", "wrong", "accurate", "mistake", "error"],
                "content": ["information", "content", "data", "knowledge", "helpful"],
                "interface": ["design", "layout", "ui", "interface", "appearance"],
                "functionality": ["feature", "function", "work", "broken", "bug"]
            }
            
            for topic, keywords in topic_keywords.items():
                if any(keyword in content for keyword in keywords):
                    feedback.topics.append(topic)
    
    async def _calculate_quality_metrics(self, feedback: FeedbackData) -> None:
        """Calculate various quality metrics for the feedback."""
        # Confidence score based on content length and specificity
        if feedback.feedback_type == FeedbackType.TEXT:
            content_length = len(feedback.content.split())
            feedback.metrics.confidence_score = min(content_length / 20, 1.0)  # Max at 20 words
            
            # Clarity score based on presence of specific terms
            specific_terms = len(feedback.keywords) + len(feedback.extracted_entities)
            feedback.metrics.clarity_score = min(specific_terms / 5, 1.0)
            
            # Actionability score based on presence of action words
            action_words = ["should", "could", "need", "want", "improve", "fix", "add", "remove"]
            action_count = sum(1 for word in action_words if word in feedback.content.lower())
            feedback.metrics.actionability_score = min(action_count / 3, 1.0)
        
        elif feedback.feedback_type == FeedbackType.RATING:
            feedback.metrics.confidence_score = 0.8  # Ratings are generally reliable
            feedback.metrics.clarity_score = 1.0  # Ratings are clear
            feedback.metrics.actionability_score = 0.5  # Ratings are moderately actionable
        
        # Overall quality score
        feedback.metrics.quality_score = (
            feedback.metrics.confidence_score * 0.3 +
            feedback.metrics.clarity_score * 0.3 +
            feedback.metrics.actionability_score * 0.4
        )
    
    async def _assess_impact(self, feedback: FeedbackData) -> None:
        """Assess the potential impact of the feedback."""
        impact_factors = []
        
        # Sentiment impact
        if abs(feedback.metrics.sentiment_score) > 0.5:
            impact_factors.append(abs(feedback.metrics.sentiment_score))
        
        # Category impact
        high_impact_categories = [FeedbackCategory.SAFETY, FeedbackCategory.PRIVACY, 
                                FeedbackCategory.FUNCTIONALITY]
        if feedback.category in high_impact_categories:
            impact_factors.append(0.8)
        
        # User context impact
        if feedback.context.task_completed is False:
            impact_factors.append(0.7)
        
        # Quality impact
        impact_factors.append(feedback.metrics.quality_score)
        
        feedback.metrics.impact_score = statistics.mean(impact_factors) if impact_factors else 0.5
        
        # Set urgency based on impact and sentiment
        if feedback.metrics.impact_score > 0.7 and feedback.sentiment in [FeedbackSentiment.VERY_NEGATIVE]:
            feedback.metrics.urgency_score = 0.9
            feedback.priority = FeedbackPriority.URGENT
        elif feedback.metrics.impact_score > 0.5:
            feedback.metrics.urgency_score = 0.6
            feedback.priority = FeedbackPriority.HIGH
        else:
            feedback.metrics.urgency_score = 0.3


class FeedbackAggregator:
    """Aggregates feedback data to extract patterns and insights."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.feedback_buffer: deque = deque(maxlen=10000)  # Keep last 10k feedback items
        self.pattern_cache: Dict[str, FeedbackPattern] = {}
    
    def add_feedback(self, feedback: FeedbackData) -> None:
        """Add feedback to the aggregation buffer."""
        self.feedback_buffer.append(feedback)
    
    async def generate_insights(self, time_window: Optional[timedelta] = None) -> FeedbackInsights:
        """Generate insights from aggregated feedback."""
        time_window = time_window or timedelta(days=7)  # Default to last 7 days
        cutoff_time = datetime.now(timezone.utc) - time_window
        
        # Filter feedback by time window
        recent_feedback = [
            fb for fb in self.feedback_buffer 
            if fb.timestamp >= cutoff_time
        ]
        
        if not recent_feedback:
            return FeedbackInsights(insights_id=str(uuid.uuid4()))
        
        insights = FeedbackInsights(insights_id=str(uuid.uuid4()))
        insights.total_feedback_count = len(recent_feedback)
        
        # Calculate aggregate metrics
        await self._calculate_satisfaction_trends(recent_feedback, insights)
        await self._analyze_sentiment_distribution(recent_feedback, insights)
        await self._analyze_category_breakdown(recent_feedback, insights)
        await self._calculate_quality_scores(recent_feedback, insights)
        
        # Identify patterns
        await self._identify_patterns(recent_feedback, insights)
        await self._identify_trending_topics(recent_feedback, insights)
        await self._identify_emerging_issues(recent_feedback, insights)
        
        # Generate recommendations
        await self._generate_recommendations(insights)
        
        return insights
    
    async def _calculate_satisfaction_trends(self, feedback_list: List[FeedbackData], 
                                           insights: FeedbackInsights) -> None:
        """Calculate satisfaction trends over time."""
        daily_satisfaction = defaultdict(list)
        
        for feedback in feedback_list:
            day = feedback.timestamp.date().isoformat()
            if feedback.metrics.sentiment_score != 0:
                daily_satisfaction[day].append(feedback.metrics.sentiment_score)
        
        # Calculate daily averages
        for day, scores in daily_satisfaction.items():
            insights.satisfaction_trends[day] = statistics.mean(scores)
    
    async def _analyze_sentiment_distribution(self, feedback_list: List[FeedbackData],
                                            insights: FeedbackInsights) -> None:
        """Analyze distribution of sentiments."""
        sentiment_counts = defaultdict(int)
        
        for feedback in feedback_list:
            sentiment_counts[feedback.sentiment.name] += 1
        
        insights.sentiment_distribution = dict(sentiment_counts)
    
    async def _analyze_category_breakdown(self, feedback_list: List[FeedbackData],
                                        insights: FeedbackInsights) -> None:
        """Analyze breakdown by feedback categories."""
        category_counts = defaultdict(int)
        
        for feedback in feedback_list:
            category_counts[feedback.category.value] += 1
        
        insights.category_breakdown = dict(category_counts)
    
    async def _calculate_quality_scores(self, feedback_list: List[FeedbackData],
                                      insights: FeedbackInsights) -> None:
        """Calculate overall quality scores."""
        quality_scores = [fb.metrics.quality_score for fb in feedback_list if fb.metrics.quality_score > 0]
        satisfaction_scores = [fb.metrics.satisfaction_score for fb in feedback_list if fb.metrics.satisfaction_score > 0]
        
        if quality_scores:
            insights.response_quality_score = statistics.mean(quality_scores)
        
        if satisfaction_scores:
            insights.user_experience_score = statistics.mean(satisfaction_scores)
        
        # Calculate system performance score based on various factors
        performance_factors = []
        
        # Response time satisfaction
        response_times = [fb.context.response_time for fb in feedback_list if fb.context.response_time > 0]
        if response_times:
            avg_response_time = statistics.mean(response_times)
            # Assume good response time is under 2 seconds
            performance_factors.append(max(0, 1 - (avg_response_time / 5)))
        
        # Task completion rate
        completed_tasks = sum(1 for fb in feedback_list if fb.context.task_completed)
        completion_rate = completed_tasks / len(feedback_list)
        performance_factors.append(completion_rate)
        
        if performance_factors:
            insights.system_performance_score = statistics.mean(performance_factors)
    
    async def _identify_patterns(self, feedback_list: List[FeedbackData],
                                insights: FeedbackInsights) -> None:
        """Identify patterns in feedback data."""
        # Topic patterns
        topic_counts = defaultdict(int)
        for feedback in feedback_list:
            for topic in feedback.topics:
                topic_counts[topic] += 1
        
        # Identify trending topics as patterns
        for topic, count in topic_counts.items():
            if count >= 5:  # Minimum threshold
                pattern = FeedbackPattern(
                    pattern_id=f"topic_{topic}_{int(time.time())}",
                    pattern_type="topic",
                    description=f"Frequent mentions of {topic}",
                    frequency=count,
                    confidence=min(count / 20, 1.0),
                    impact_score=count / len(feedback_list)
                )
                insights.identified_patterns.append(pattern)
        
        # Sentiment patterns by time
        hourly_sentiment = defaultdict(list)
        for feedback in feedback_list:
            hour = feedback.timestamp.hour
            hourly_sentiment[hour].append(feedback.metrics.sentiment_score)
        
        # Identify time-based sentiment patterns
        for hour, scores in hourly_sentiment.items():
            if len(scores) >= 3:
                avg_sentiment = statistics.mean(scores)
                if abs(avg_sentiment) > 0.3:  # Significant sentiment
                    pattern = FeedbackPattern(
                        pattern_id=f"temporal_sentiment_{hour}_{int(time.time())}",
                        pattern_type="temporal",
                        description=f"{'Negative' if avg_sentiment < 0 else 'Positive'} sentiment during hour {hour}",
                        frequency=len(scores),
                        confidence=min(len(scores) / 10, 1.0),
                        impact_score=abs(avg_sentiment)
                    )
                    insights.identified_patterns.append(pattern)
    
    async def _identify_trending_topics(self, feedback_list: List[FeedbackData],
                                       insights: FeedbackInsights) -> None:
        """Identify trending topics in feedback."""
        topic_counts = defaultdict(int)
        
        for feedback in feedback_list:
            for topic in feedback.topics:
                topic_counts[topic] += 1
        
        # Sort by frequency and take top topics
        sorted_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)
        insights.trending_topics = [topic for topic, count in sorted_topics[:5]]
    
    async def _identify_emerging_issues(self, feedback_list: List[FeedbackData],
                                       insights: FeedbackInsights) -> None:
        """Identify emerging issues from feedback."""
        # Look for negative feedback with high impact
        issues = []
        
        for feedback in feedback_list:
            if (feedback.sentiment in [FeedbackSentiment.NEGATIVE, FeedbackSentiment.VERY_NEGATIVE] and
                feedback.metrics.impact_score > 0.6):
                
                for topic in feedback.topics:
                    if topic not in issues:
                        issues.append(f"{topic} issues")
        
        insights.emerging_issues = issues[:5]  # Top 5 issues
    
    async def _generate_recommendations(self, insights: FeedbackInsights) -> None:
        """Generate improvement recommendations based on insights."""
        recommendations = []
        critical_issues = []
        quick_wins = []
        
        # Analyze satisfaction trends
        if insights.satisfaction_trends:
            recent_scores = list(insights.satisfaction_trends.values())[-3:]  # Last 3 days
            if recent_scores and statistics.mean(recent_scores) < -0.2:
                critical_issues.append("Declining user satisfaction trend")
                recommendations.append("Investigate recent changes that may impact user satisfaction")
        
        # Analyze sentiment distribution
        negative_feedback = insights.sentiment_distribution.get("NEGATIVE", 0) + \
                          insights.sentiment_distribution.get("VERY_NEGATIVE", 0)
        total_feedback = sum(insights.sentiment_distribution.values())
        
        if total_feedback > 0 and negative_feedback / total_feedback > 0.3:
            critical_issues.append("High proportion of negative feedback")
            recommendations.append("Focus on addressing common complaints and pain points")
        
        # Analyze quality scores
        if insights.response_quality_score < 0.6:
            recommendations.append("Improve response quality through better training or model tuning")
        
        if insights.user_experience_score < 0.6:
            recommendations.append("Enhance user experience through interface improvements")
        
        if insights.system_performance_score < 0.7:
            critical_issues.append("Poor system performance")
            recommendations.append("Optimize system performance and response times")
        
        # Analyze patterns for quick wins
        for pattern in insights.identified_patterns:
            if pattern.pattern_type == "topic" and pattern.frequency > 10:
                if any(keyword in pattern.description.lower() for keyword in ["easy", "simple", "quick"]):
                    quick_wins.append(f"Address frequent requests about {pattern.description}")
        
        insights.improvement_recommendations = recommendations
        insights.critical_issues = critical_issues
        insights.quick_wins = quick_wins


class EnhancedFeedbackProcessor:
    """
    Advanced Feedback Processing System for the AI Assistant.
    
    This processor provides comprehensive feedback processing capabilities including:
    - Multi-modal feedback collection and validation
    - Real-time sentiment and emotion analysis
    - Pattern recognition and trend analysis
    - Continuous learning and model adaptation
    - Integration with all core system components
    - Automated quality assessment and bias detection
    - Actionable insights generation
    - Performance monitoring and optimization
    """
    
    def __init__(self, container: Container):
        """
        Initialize the enhanced feedback processor.
        
        Args:
            container: Dependency injection container
        """
        self.container = container
        self.logger = get_logger(__name__)
        
        # Core services
        self.config = container.get(ConfigLoader)
        self.event_bus = container.get(EventBus)
        self.error_handler = container.get(ErrorHandler)
        self.health_check = container.get(HealthCheck)
        
        # Processing components
        self.sentiment_analyzer = container.get(SentimentAnalyzer)
        try:
            self.emotion_detector = container.get(EnhancedEmotionDetector)
            self.entity_extractor = container.get(EntityExtractor)
            self.vision_processor = container.get(VisionProcessor)
        except Exception:
            self.emotion_detector = None
            self.entity_extractor = None
            self.vision_processor = None
        
        # Memory systems
        self.memory_manager = container.get(MemoryManager)
        self.context_manager = container.get(ContextManager)
        self.episodic_memory = container.get(EpisodicMemory)
        self.semantic_memory = container.get(SemanticMemory)
        self.vector_store = container.get(VectorStore)
        
        # Learning systems
        self.continual_learner = container.get(ContinualLearner)
        self.preference_learner = container.get(PreferenceLearner)
        self.model_adapter = container.get(ModelAdapter)
        
        # Storage and caching
        try:
            self.database = container.get(DatabaseManager)
            self.cache_strategy = container.get(CacheStrategy)
        except Exception:
            self.database = None
            self.cache_strategy = None
        
        # Security
        try:
            self.input_sanitizer = container.get(InputSanitizer)
        except Exception:
            self.input_sanitizer = None
        
        # Observability
        self.metrics = container.get(MetricsCollector)
        self.tracer = container.get(TraceManager)
        
        # Feedback processing components
        self.validator = FeedbackValidator(self.input_sanitizer) if self.input_sanitizer else None
        self.analyzer = FeedbackAnalyzer(self.sentiment_analyzer, self.emotion_detector, 
                                       self.entity_extractor)
        self.aggregator = FeedbackAggregator()
        
        # State management
        self.active_feedback: Dict[str, FeedbackData] = {}
        self.processing_queue = asyncio.Queue()
        self.feedback_history: deque = deque(maxlen=10000)
        
        # Performance tracking
        self.processing_stats = {
            "total_processed": 0,
            "successful": 0,
            "failed": 0,
            "average_processing_time": 0.0
        }
        
        # Background tasks
        self.background_tasks: List[asyncio.Task] = []
        
        # Configuration
        self.enable_real_time_processing = self.config.get("feedback.enable_real_time", True)
        self.enable_learning = self.config.get("feedback.enable_learning", True)
        self.batch_processing_size = self.config.get("feedback.batch_size", 10)
        self.processing_timeout = self.config.get("feedback.processing_timeout", 30.0)
        self.insights_generation_interval = self.config.get("feedback.insights_interval", 3600)
        
        # Setup monitoring and health checks
        self._setup_monitoring()
        self.health_check.register_component("feedback_processor", self._health_check_callback)
        
        self.logger.info("EnhancedFeedbackProcessor initialized successfully")

    def _setup_monitoring(self) -> None:
        """Setup monitoring and metrics collection."""
        try:
            # Register feedback metrics
            self.metrics.register_counter("feedback_received_total")
            self.metrics.register_counter("feedback_processed_total")
            self.metrics.register_counter("feedback_failed_total")
            self.metrics.register_histogram("feedback_processing_duration_seconds")
            self.metrics.register_gauge("feedback_queue_size")
            self.metrics.register_histogram("feedback_sentiment_score")
            self.metrics.register_counter("feedback_patterns_detected")
            self.metrics.register_gauge("user_satisfaction_score")
            
        except Exception as e:
            self.logger.warning(f"Failed to setup monitoring: {str(e)}")

    async def initialize(self) -> None:
        """Initialize the feedback processor."""
        try:
            # Initialize processing components
            if hasattr(self.analyzer, 'initialize'):
                await self.analyzer.initialize()
            
            # Start background tasks
            if self.enable_real_time_processing:
                self.background_tasks.append(
                    asyncio.create_task(self._real_time_processing_loop())
                )
            
            self.background_tasks.extend([
                asyncio.create_task(self._batch_processing_loop()),
                asyncio.create_task(self._insights_generation_loop()),
                asyncio.create_task(self._learning_update_loop()),
                asyncio.create_task(self._performance_monitoring_loop())
            ])
            
            # Register event handlers
            await self._register_event_handlers()
            
            self.logger.info("FeedbackProcessor initialization completed")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize FeedbackProcessor: {str(e)}")
            raise FeedbackError(f"Initialization failed: {str(e)}")

    async def _register_event_handlers(self) -> None:
        """Register event handlers for system events."""
        # Interaction events
        self.event_bus.subscribe("interaction_completed", self._handle_interaction_completed)
        self.event_bus.subscribe("workflow_completed", self._handle_workflow_completed)
        self.event_bus.subscribe("skill_execution_completed", self._handle_skill_execution_completed)
        
        # Session events
        self.event_bus.subscribe("session_started", self._handle_session_started)
        self.event_bus.subscribe("session_ended", self._handle_session_ended)
        
        # Component health events
        self.event_bus.subscribe("component_health_changed", self._handle_component_health_change)

    @handle_exceptions
    async def process_feedback(
        self,
        feedback_type: FeedbackType,
        content: Any,
        context: Optional[FeedbackContext] = None,
        category: FeedbackCategory = FeedbackCategory.QUALITY,
        priority: FeedbackPriority = FeedbackPriority.NORMAL,
        **kwargs
    ) -> str:
        """
        Process user feedback through the complete pipeline.
        
        Args:
            feedback_type: Type of feedback
            content: Feedback content (text, rating, audio, etc.)
            context: Feedback context information
            category: Feedback category
            priority: Processing priority
            **kwargs: Additional feedback metadata
            
        Returns:
            Feedback ID for tracking
        """
        feedback_id = str(uuid.uuid4())
        start_time = time.time()
        
        # Create feedback data structure
        feedback = FeedbackData(
            feedback_id=feedback_id,
            feedback_type=feedback_type,
            category=category,
            priority=priority,
            content=content,
            context=context or FeedbackContext(),
            **kwargs
        )
        
        try:
            with self.tracer.trace("feedback_processing") as span:
                span.set_attributes({
                    "feedback_id": feedback_id,
                    "feedback_type": feedback_type.value,
                    "category": category.value,
                    "priority": priority.value,
                    "user_id": feedback.context.user_id or "anonymous"
                })
                
                # Store active feedback
                self.active_feedback[feedback_id] = feedback
                
                # Emit feedback received event
                await self.event_bus.emit(FeedbackReceived(
                    feedback_id=feedback_id,
                    feedback_type=feedback_type.value,
                    user_id=feedback.context.user_id,
                    session_id=feedback.context.session_id,
                    category=category.value
                ))
                
                # Process feedback through pipeline
                await self._process_feedback_pipeline(feedback)
                
                # Calculate processing time
                feedback.processing_time = time.time() - start_time
                
                # Update metrics
                self.metrics.increment("feedback_received_total")
                self.metrics.increment("feedback_processed_total")
                self.metrics.record("feedback_processing_duration_seconds", feedback.processing_time)
                self.metrics.record("feedback_sentiment_score", feedback.metrics.sentiment_score)
                
                # Update statistics
                self.processing_stats["total_processed"] += 1
                self.processing_stats["successful"] += 1
                self.processing_stats["average_processing_time"] = (
                    (self.processing_stats["average_processing_time"] * 
                     (self.processing_stats["total_processed"] - 1) + feedback.processing_time) /
                    self.processing_stats["total_processed"]
                )
                
                # Emit processing completed event
                await self.event_bus.emit(FeedbackProcessed(
                    feedback_id=feedback_id,
                    processing_time=feedback.processing_time,
                    sentiment=feedback.sentiment.value,
                    quality_score=feedback.metrics.quality_score,
                    success=feedback.status == ProcessingStatus.COMPLETED
                ))
                
                self.logger.info(
                    f"Processed feedback {feedback_id} in {feedback.processing_time:.2f}s "
                    f"(sentiment: {feedback.sentiment.value}, quality: {feedback.metrics.quality_score:.2f})"
                )
                
                return feedback_id
                
        except Exception as e:
            # Handle processing error
            feedback.status = ProcessingStatus.FAILED
            feedback.errors.append(str(e))
            feedback.processing_time = time.time() - start_time
            
            self.metrics.increment("feedback_failed_total")
            self.processing_stats["failed"] += 1
            
            self.logger.error(f"Failed to process feedback {feedback_id}: {str(e)}")
            raise FeedbackError(f"Feedback processing failed: {str(e)}", feedback_id)
        
        finally:
            # Move to history and cleanup
            self.feedback_history.append(feedback)
            self.active_feedback.pop(feedback_id, None)

    async def _process_feedback_pipeline(self, feedback: FeedbackData) -> None:
        """Process feedback through the complete processing pipeline."""
        try:
            # Step 1: Validation
            feedback.status = ProcessingStatus.VALIDATING
            if self.validator:
                feedback.is_valid = await self.validator.validate_feedback(feedback)
                if not feedback.is_valid:
                    feedback.status = ProcessingStatus.IGNORED
                    return
            
            # Step 2: Content processing
            feedback.status = ProcessingStatus.PROCESSING
            await self._preprocess_content(feedback)
            
            # Step 3: Analysis
            await self.analyzer.analyze_feedback(feedback)
            
            # Step 4: Learning and adaptation
            if self.enable_learning:
                await self._learn_from_feedback(feedback)
            
            # Step 5: Storage
            await self._store_feedback(feedback)
            
            # Step 6: Aggregation
            self.aggregator.add_feedback(feedback)
            
            feedback.status = ProcessingStatus.COMPLETED
            
        except Exception as e:
            feedback.status = ProcessingStatus.FAILED
            feedback.errors.append(str(e))
            raise

    async def _preprocess_content(self, feedback: FeedbackData) -> None:
        """Preprocess feedback content based on type."""
        if feedback.feedback_type == FeedbackType.TEXT:
            # Text preprocessing
            feedback.processed_content["original_length"] = len(feedback.content)
            feedback.processed_content["word_count"] = len(feedback.content.split())
            feedback.processed_content["language"] = feedback.language
        
        elif feedback.feedback_type == FeedbackType.VOICE:
            # Voice preprocessing - transcription would happen here
            if self.emotion_detector:
                # Analyze audio for emotions
                pass
        
        elif feedback.feedback_type == FeedbackType.RATING:
            # Rating preprocessing
            feedback.processed_content["normalized_rating"] = self._normalize_rating(
                feedback.rating, feedback.scale
            )

    def _normalize_rating(self, rating: float, scale: str) -> float:
        """Normalize rating to 0-1 scale."""
        if scale == "1-5":
            return (rating - 1) / 4
        elif scale == "1-10":
            return (rating - 1) / 9
        elif scale == "thumbs":
            return 1.0 if rating > 0 else 0.0
        else:
            return rating

    async def _learn_from_feedback(self, feedback: FeedbackData) -> None:
        """Update learning systems with feedback data."""
        try:
            learning_data = {
                "feedback_id": feedback.feedback_id,
                "feedback_type": feedback.feedback_type.value,
                "sentiment": feedback.sentiment.value,
                "quality_metrics": asdict(feedback.metrics),
                "context": asdict(feedback.context),
                "timestamp": feedback.timestamp
            }
            
            # Update continual learning
            await self.continual_learner.learn_from_feedback(learning_data)
            
            # Update user preferences
            if feedback.context.user_id:
                await self.preference_learner.update_from_feedback(
                    feedback.context.user_id, learning_data
                )
            
            # Model adaptation for negative feedback
            if feedback.sentiment in [FeedbackSentiment.NEGATIVE, FeedbackSentiment.VERY_NEGATIVE]:
                await self.model_adapter.adapt_from_negative_feedback(learning_data)
            
            # Emit learning event
            await self.event_bus.emit(LearningEventOccurred(
                event_type="feedback_learning",
                data=learning_data,
                session_id=feedback.context.session_id
            ))
            
        except Exception as e:
            self.logger.warning(f"Failed to learn from feedback {feedback.feedback_id}: {str(e)}")

    async def _store_feedback(self, feedback: FeedbackData) -> None:
        """Store feedback data in appropriate storage systems."""
        try:
            # Store in episodic memory
            memory_data = {
                "feedback_id": feedback.feedback_id,
                "type": "feedback",
                "content": feedback.content,
                "sentiment": feedback.sentiment.value,
                "quality_score": feedback.metrics.quality_score,
                "context": asdict(feedback.context),
                "timestamp": feedback.timestamp.isoformat()
            }
            
            await self.episodic_memory.store(memory_data)
            
            # Store in vector store for similarity search
            if feedback.feedback_type == FeedbackType.TEXT:
                await self.vector_store.store_text(
                    text=feedback.content,
                    metadata={
                        "feedback_id": feedback.feedback_id,
                        "sentiment": feedback.sentiment.value,
                        "category": feedback.category.value
                    }
                )
            
            # Store in database if available
            if self.database:
                await self.database.execute(
                    """
                    INSERT INTO feedback (
                        feedback_id, feedback_type, category, content, sentiment,
                        quality_score, user_id, session_id, timestamp, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        feedback.feedback_id,
                        feedback.feedback_type.value,
                        feedback.category.value,
                        str(feedback.content),
                        feedback.sentiment.value,
                        feedback.metrics.quality_score,
                        feedback.context.user_id,
                        feedback.context.session_id,
                        feedback.timestamp,
                        json.dumps(asdict(feedback.metrics))
                    )
                )
            
        except Exception as e:
            self.logger.warning(f"Failed to store feedback {feedback.feedback_id}: {str(e)}")

    @handle_exceptions
    async def process_event_feedback(
        self,
        event_type: str,
        event_data: Dict[str, Any],
        implicit_feedback: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Process implicit feedback from system events.
        
        Args:
            event_type: Type of event that generated feedback
            event_data: Event data
            implicit_feedback: Implicit feedback derived from event
        """
        try:
            # Extract context from event
            context = FeedbackContext(
                session_id=event_data.get("session_id"),
                user_id=event_data.get("user_id"),
                interaction_id=event_data.get("interaction_id"),
                workflow_id=event_data.get("workflow_id"),
                component_id=event_data.get("component_id"),
                skill_id=event_data.get("skill_id")
            )
            
            # Determine feedback based on event type
            feedback_content = None
            feedback_type = FeedbackType.BEHAVIORAL
            category = FeedbackCategory.QUALITY
            
            if event_type == "interaction_completed":
                # Derive satisfaction from completion and response time
                response_time = event_data.get("processing_time", 0)
                task_completed = event_data.get("success", False)
                
                context.response_time = response_time
                context.task_completed = task_completed
                
                # Implicit rating based on performance
                if task_completed and response_time < 2.0:
                    feedback_content = 4.0  # Good performance
                elif task_completed:
                    feedback_content = 3.0  # Acceptable performance
                else:
                    feedback_content = 2.0  # Poor performance
                
                feedback_type = FeedbackType.RATING
                category = FeedbackCategory.QUALITY
            
            elif event_type == "workflow_completed":
                # Workflow completion feedback
                execution_time = event_data.get("execution_time", 0)
                success = event_data.get("success", False)
                
                context.task_completed = success
                
                if success and execution_time < 10.0:
                    feedback_content = 4.0
                elif success:
                    feedback_content = 3.0
                else:
                    feedback_content = 1.0
                
                feedback_type = FeedbackType.RATING
                category = FeedbackCategory.FUNCTIONALITY
            
            # Process the implicit feedback
            if feedback_content is not None:
                await self.process_feedback(
                    feedback_type=feedback_type,
                    content=feedback_content,
                    context=context,
                    category=category,
                    priority=FeedbackPriority.LOW,
                    scale="1-5",
                    source="system"
                )
            
        except Exception as e:
            self.logger.error(f"Failed to process event feedback for {event_type}: {str(e)}")

    @handle_exceptions
    async def get_feedback_insights(
        self,
        time_window: Optional[timedelta] = None,
        user_id: Optional[str] = None,
        category: Optional[FeedbackCategory] = None
    ) -> FeedbackInsights:
        """
        Get feedback insights for a specific time window and filters.
        
        Args:
            time_window: Time window for analysis
            user_id: Optional user filter
            category: Optional category filter
            
        Returns:
            Feedback insights
        """
        try:
            # Filter feedback based on criteria
            filtered_feedback = list(self.feedback_history)
            
            if time_window:
                cutoff_time = datetime.now(timezone.utc) - time_window
                filtered_feedback = [
                    fb for fb in filtered_feedback if fb.timestamp >= cutoff_time
                ]
            
            if user_id:
                filtered_feedback = [
                    fb for fb in filtered_feedback if fb.context.user_id == user_id
                ]
            
            if category:
                filtered_feedback = [
                    fb for fb in filtered_feedback if fb.category == category
                ]
            
            # Generate insights from filtered data
            temp_aggregator = FeedbackAggregator()
            for feedback in filtered_feedback:
                temp_aggregator.add_feedback(feedback)
            
            insights = await temp_aggregator.generate_insights(time_window)
            
            # Emit insights generated event
            await self.event_bus.emit(FeedbackAggregated(
                insights_id=insights.insights_id,
                time_window_hours=(time_window.total_seconds() / 3600) if time_window else 168,  # Default 7 days
                total_feedback=insights.total_feedback_count,
                satisfaction_score=insights.user_experience_score
            ))
            
            return insights
            
        except Exception as e:
            self.logger.error(f"Failed to generate feedback insights: {str(e)}")
            raise FeedbackError(f"Insights generation failed: {str(e)}")

    @handle_exceptions
    async def get_user_feedback_summary(self, user_id: str) -> Dict[str, Any]:
        """
        Get feedback summary for a specific user.
        
        Args:
            user_id: User identifier
            
        Returns:
            User feedback summary
        """
        user_feedback = [
            fb for fb in self.feedback_history 
            if fb.context.user_id == user_id
        ]
        
        if not user_feedback:
            return {
                "user_id": user_id,
                "total_feedback": 0,
                "average_sentiment": 0.0,
                "satisfaction_score": 0.0,
                "categories": {},
                "recent_trends": []
            }
        
        # Calculate summary statistics
        sentiment_scores = [fb.metrics.sentiment_score for fb in user_feedback]
        category_counts = defaultdict(int)
        
        for feedback in user_feedback:
            category_counts[feedback.category.value] += 1
        
        # Recent trends (last 30 days)
        recent_cutoff = datetime.now(timezone.utc) - timedelta(days=30)
        recent_feedback = [fb for fb in user_feedback if fb.timestamp >= recent_cutoff]
        
        return {
            "user_id": user_id,
            "total_feedback": len(user_feedback),
            "average_sentiment": statistics.mean(sentiment_scores) if sentiment_scores else 0.0,
            "satisfaction_score": statistics.mean([
                fb.metrics.satisfaction_score for fb in user_feedback 
                if fb.metrics.satisfaction_score > 0
            ]) if any(fb.metrics.satisfaction_score > 0 for fb in user_feedback) else 0.0,
            "categories": dict(category_counts),
            "recent_feedback_count": len(recent_feedback),
            "recent_average_sentiment": statistics.mean([
                fb.metrics.sentiment_score for fb in recent_feedback
            ]) if recent_feedback else 0.0
        }

    async def _real_time_processing_loop(self) -> None:
        """Background task for real-time feedback processing."""
        while True:
            try:
                # Process high-priority feedback immediately
                high_priority_feedback = [
                    fb for fb in self.active_feedback.values()
                    if fb.priority in [FeedbackPriority.URGENT, FeedbackPriority.CRITICAL]
                    and fb.status == ProcessingStatus.RECEIVED
                ]
                
                for feedback in high_priority_feedback:
                    try:
                        await self._process_feedback_pipeline(feedback)
                    except Exception as e:
                        self.logger.error(f"Real-time processing failed for {feedback.feedback_id}: {str(e)}")
                
                await asyncio.sleep(1)  # Check every second
                
            except Exception as e:
                self.logger.error(f"Error in real-time processing loop: {str(e)}")
                await asyncio.sleep(1)

    async def _batch_processing_loop(self) -> None:
        """Background task for batch feedback processing."""
        while True:
            try:
                # Process normal priority feedback in batches
                normal_feedback = [
                    fb for fb in self.active_feedback.values()
                    if fb.priority == FeedbackPriority.NORMAL
                    and fb.status == ProcessingStatus.RECEIVED
                ]
                
                # Process in batches
                for i in range(0, len(normal_feedback), self.batch_processing_size):
                    batch = normal_feedback[i:i + self.batch_processing_size]
                    
                    # Process batch concurrently
                    tasks = [self._process_feedback_pipeline(fb) for fb in batch]
                    await asyncio.gather(*tasks, return_exceptions=True)
                
                await asyncio.sleep(5)  # Process batches every 5 seconds
                
            except Exception as e:
                self.logger.error(f"Error in batch processing loop: {str(e)}")
                await asyncio.sleep(5)

    async def _insights_generation_loop(self) -> None:
        """Background task for generating periodic insights."""
        while True:
            try:
                # Generate insights every hour
                insights = await self.aggregator.generate_insights()
                
                # Store insights
                await self._store_insights(insights)
                
                # Check for critical issues
                if insights.critical_issues:
                    for issue in insights.critical_issues:
                        await self.event_bus.emit(ErrorOccurred(
                            component="feedback_processor",
                            error_type="critical_issue",
                            error_message=issue,
                            severity="high"
                        ))
                
                # Update satisfaction metrics
                if insights.user_experience_score > 0:
                    self.metrics.set("user_satisfaction_score", insights.user_experience_score)
                
                await asyncio.sleep(self.insights_generation_interval)
                
            except Exception as e:
                self.logger.error(f"Error in insights generation loop: {str(e)}")
                await asyncio.sleep(self.insights_generation_interval)

    async def _learning_update_loop(self) -> None:
        """Background task for learning system updates."""
        if not self.enable_learning:
            return
        
        while True:
            try:
                # Periodic learning updates
                recent_feedback = [
                    fb for fb in self.feedback_history
                    if fb.timestamp > (datetime.now(timezone.utc) - timedelta(hours=24))
                ]
                
                if recent_feedback:
                    # Process recent feedback for learning
                    self.logger.info(f"Processing {len(recent_feedback)} recent feedback items for learning")
                    
                await asyncio.sleep(self.learning_update_interval)
                
            except Exception as e:
                self.logger.error(f"Error in learning update loop: {str(e)}")
                await asyncio.sleep(self.learning_update_interval)
