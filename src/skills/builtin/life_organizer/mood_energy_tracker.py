"""
Mood and Energy Tracking System - نظام تتبع المزاج والطاقة
Author: Drmusab
Last Modified: 2025-01-20

Integrates sentiment analysis from voice input and expression analysis from vision input
to track user's mood and energy levels, providing recommendations for optimal task timing.
"""

import asyncio
import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from src.core.config.loader import ConfigLoader
from src.core.dependency_injection import Container
from src.core.error_handling import ErrorHandler, handle_exceptions
from src.core.events.event_bus import EventBus
from src.core.events.event_types import MoodChanged, EmotionDetected
from src.observability.logging.config import get_logger
from src.processing.natural_language.sentiment_analyzer import SentimentAnalyzer
from src.processing.vision.detectors.expression_analyzer import ExpressionAnalyzer


class MoodLevel(Enum):
    """Mood levels in Arabic and English."""
    VERY_LOW = "منخفض جداً"  # Very Low
    LOW = "منخفض"  # Low
    MODERATE = "متوسط"  # Moderate
    HIGH = "عالي"  # High
    VERY_HIGH = "عالي جداً"  # Very High


class EnergyLevel(Enum):
    """Energy levels in Arabic and English."""
    EXHAUSTED = "منهك"  # Exhausted
    LOW = "قليل"  # Low
    MODERATE = "متوسط"  # Moderate
    HIGH = "عالي"  # High
    ENERGETIC = "نشيط"  # Energetic


@dataclass
class MoodEnergyState:
    """Current mood and energy state."""
    mood_level: MoodLevel
    energy_level: EnergyLevel
    confidence: float  # 0-1 confidence score
    timestamp: datetime
    sources: List[str]  # ["voice", "vision", "manual"]
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskRecommendation:
    """Task recommendation based on current mood/energy."""
    task_type: str
    arabic_description: str
    english_description: str
    optimal_duration: int  # minutes
    reason_arabic: str
    reason_english: str


class MoodEnergyTracker:
    """Tracks user's mood and energy through multimodal analysis."""

    def __init__(self, container: Container):
        self.container = container
        self.config = container.resolve(ConfigLoader)
        self.event_bus = container.resolve(EventBus)
        self.logger = get_logger(__name__)
        
        # Initialize analyzers
        self.sentiment_analyzer = container.resolve(SentimentAnalyzer)
        self.expression_analyzer = container.resolve(ExpressionAnalyzer)
        
        # State tracking
        self.current_state: Optional[MoodEnergyState] = None
        self.state_history: List[MoodEnergyState] = []
        self.max_history = 50  # Keep last 50 states
        
        self._initialize_recommendations()

    def _initialize_recommendations(self) -> None:
        """Initialize task recommendations for different mood/energy combinations."""
        self.recommendations = {
            (MoodLevel.VERY_HIGH, EnergyLevel.ENERGETIC): [
                TaskRecommendation(
                    task_type="creative_work",
                    arabic_description="عمل إبداعي ومبتكر",
                    english_description="Creative and innovative work",
                    optimal_duration=90,
                    reason_arabic="مزاجك ممتاز وطاقتك عالية - وقت مثالي للإبداع!",
                    reason_english="Your mood and energy are excellent - perfect time for creativity!"
                ),
                TaskRecommendation(
                    task_type="complex_problem_solving",
                    arabic_description="حل المشاكل المعقدة",
                    english_description="Complex problem solving",
                    optimal_duration=120,
                    reason_arabic="طاقتك العالية تساعدك على التركيز في المهام الصعبة",
                    reason_english="Your high energy helps you focus on difficult tasks"
                )
            ],
            (MoodLevel.HIGH, EnergyLevel.HIGH): [
                TaskRecommendation(
                    task_type="deep_work",
                    arabic_description="عمل عميق ومركز",
                    english_description="Deep focused work",
                    optimal_duration=90,
                    reason_arabic="حالتك جيدة للتركيز العميق",
                    reason_english="You're in good state for deep focus"
                )
            ],
            (MoodLevel.MODERATE, EnergyLevel.MODERATE): [
                TaskRecommendation(
                    task_type="routine_tasks",
                    arabic_description="مهام روتينية",
                    english_description="Routine tasks",
                    optimal_duration=45,
                    reason_arabic="طاقتك متوسطة - ابدأ بالمهام البسيطة",
                    reason_english="Your energy is moderate - start with simple tasks"
                )
            ],
            (MoodLevel.LOW, EnergyLevel.LOW): [
                TaskRecommendation(
                    task_type="light_tasks",
                    arabic_description="مهام خفيفة",
                    english_description="Light tasks",
                    optimal_duration=20,
                    reason_arabic="تبدو متعباً - ركز على المهام السهلة أو خذ استراحة",
                    reason_english="You seem tired - focus on easy tasks or take a break"
                )
            ],
            (MoodLevel.VERY_LOW, EnergyLevel.EXHAUSTED): [
                TaskRecommendation(
                    task_type="rest",
                    arabic_description="راحة واسترخاء",
                    english_description="Rest and relaxation",
                    optimal_duration=30,
                    reason_arabic="تحتاج إلى راحة - خذ استراحة قبل المتابعة",
                    reason_english="You need rest - take a break before continuing"
                )
            ]
        }

    @handle_exceptions()
    async def analyze_voice_input(self, voice_text: str, language: str = "ar") -> Tuple[MoodLevel, EnergyLevel, float]:
        """Analyze mood and energy from voice input."""
        try:
            # Use sentiment analyzer
            sentiment_result = await self.sentiment_analyzer.analyze_sentiment(
                text=voice_text,
                language=language
            )
            
            # Extract mood from sentiment
            mood_score = sentiment_result.get("polarity", 0.5)  # -1 to 1
            energy_score = sentiment_result.get("arousal", 0.5)  # 0 to 1
            confidence = sentiment_result.get("confidence", 0.7)
            
            # Map sentiment scores to mood/energy levels
            mood_level = self._map_score_to_mood(mood_score)
            energy_level = self._map_score_to_energy(energy_score)
            
            self.logger.info(f"Voice analysis: mood={mood_level.value}, energy={energy_level.value}")
            return mood_level, energy_level, confidence
            
        except Exception as e:
            self.logger.error(f"Error in voice analysis: {e}")
            return MoodLevel.MODERATE, EnergyLevel.MODERATE, 0.3

    @handle_exceptions()
    async def analyze_vision_input(self, image_data: Any) -> Tuple[MoodLevel, EnergyLevel, float]:
        """Analyze mood and energy from facial expressions."""
        try:
            # Use expression analyzer
            expression_result = await self.expression_analyzer.analyze_expression(image_data)
            
            # Extract mood from facial expressions
            emotions = expression_result.get("emotions", {})
            energy_indicators = expression_result.get("energy_indicators", {})
            confidence = expression_result.get("confidence", 0.7)
            
            # Map expressions to mood/energy
            mood_level = self._map_emotions_to_mood(emotions)
            energy_level = self._map_indicators_to_energy(energy_indicators)
            
            self.logger.info(f"Vision analysis: mood={mood_level.value}, energy={energy_level.value}")
            return mood_level, energy_level, confidence
            
        except Exception as e:
            self.logger.error(f"Error in vision analysis: {e}")
            return MoodLevel.MODERATE, EnergyLevel.MODERATE, 0.3

    def _map_score_to_mood(self, score: float) -> MoodLevel:
        """Map sentiment score (-1 to 1) to mood level."""
        if score >= 0.6:
            return MoodLevel.VERY_HIGH
        elif score >= 0.2:
            return MoodLevel.HIGH
        elif score >= -0.2:
            return MoodLevel.MODERATE
        elif score >= -0.6:
            return MoodLevel.LOW
        else:
            return MoodLevel.VERY_LOW

    def _map_score_to_energy(self, score: float) -> EnergyLevel:
        """Map arousal score (0 to 1) to energy level."""
        if score >= 0.8:
            return EnergyLevel.ENERGETIC
        elif score >= 0.6:
            return EnergyLevel.HIGH
        elif score >= 0.4:
            return EnergyLevel.MODERATE
        elif score >= 0.2:
            return EnergyLevel.LOW
        else:
            return EnergyLevel.EXHAUSTED

    def _map_emotions_to_mood(self, emotions: Dict[str, float]) -> MoodLevel:
        """Map facial emotions to mood level."""
        positive_emotions = emotions.get("happy", 0) + emotions.get("surprised", 0)
        negative_emotions = emotions.get("sad", 0) + emotions.get("angry", 0) + emotions.get("fear", 0)
        
        mood_score = positive_emotions - negative_emotions
        return self._map_score_to_mood(mood_score)

    def _map_indicators_to_energy(self, indicators: Dict[str, float]) -> EnergyLevel:
        """Map energy indicators to energy level."""
        alertness = indicators.get("alertness", 0.5)
        engagement = indicators.get("engagement", 0.5)
        
        energy_score = (alertness + engagement) / 2
        return self._map_score_to_energy(energy_score)

    @handle_exceptions()
    async def update_state(
        self, 
        voice_text: Optional[str] = None,
        image_data: Optional[Any] = None,
        manual_mood: Optional[MoodLevel] = None,
        manual_energy: Optional[EnergyLevel] = None
    ) -> MoodEnergyState:
        """Update current mood/energy state from available inputs."""
        
        mood_scores = []
        energy_scores = []
        confidences = []
        sources = []
        
        # Analyze voice input
        if voice_text:
            mood, energy, conf = await self.analyze_voice_input(voice_text)
            mood_scores.append(self._mood_to_score(mood))
            energy_scores.append(self._energy_to_score(energy))
            confidences.append(conf)
            sources.append("voice")
        
        # Analyze vision input
        if image_data:
            mood, energy, conf = await self.analyze_vision_input(image_data)
            mood_scores.append(self._mood_to_score(mood))
            energy_scores.append(self._energy_to_score(energy))
            confidences.append(conf)
            sources.append("vision")
        
        # Use manual input if provided
        if manual_mood is not None:
            mood_scores.append(self._mood_to_score(manual_mood))
            confidences.append(1.0)
            sources.append("manual")
            
        if manual_energy is not None:
            energy_scores.append(self._energy_to_score(manual_energy))
            confidences.append(1.0)
            sources.append("manual")
        
        # Calculate weighted averages
        if mood_scores and energy_scores:
            weighted_mood = sum(m * c for m, c in zip(mood_scores, confidences)) / sum(confidences)
            weighted_energy = sum(e * c for e, c in zip(energy_scores, confidences)) / sum(confidences)
            overall_confidence = sum(confidences) / len(confidences)
            
            final_mood = self._score_to_mood(weighted_mood)
            final_energy = self._score_to_energy(weighted_energy)
        else:
            # Fallback to moderate levels
            final_mood = MoodLevel.MODERATE
            final_energy = EnergyLevel.MODERATE
            overall_confidence = 0.3
        
        # Create new state
        new_state = MoodEnergyState(
            mood_level=final_mood,
            energy_level=final_energy,
            confidence=overall_confidence,
            timestamp=datetime.now(timezone.utc),
            sources=sources,
            details={
                "raw_scores": {
                    "mood_scores": mood_scores,
                    "energy_scores": energy_scores,
                    "confidences": confidences
                }
            }
        )
        
        # Update current state and history
        self.current_state = new_state
        self.state_history.append(new_state)
        
        # Trim history
        if len(self.state_history) > self.max_history:
            self.state_history = self.state_history[-self.max_history:]
        
        # Emit event
        await self.event_bus.emit(MoodChanged(
            user_id="current_user",  # TODO: Get from session
            mood_level=final_mood.value,
            energy_level=final_energy.value,
            confidence=overall_confidence,
            timestamp=new_state.timestamp
        ))
        
        self.logger.info(f"Updated mood/energy state: {final_mood.value}, {final_energy.value}")
        return new_state

    def _mood_to_score(self, mood: MoodLevel) -> float:
        """Convert mood level to numeric score."""
        mapping = {
            MoodLevel.VERY_LOW: -1.0,
            MoodLevel.LOW: -0.5,
            MoodLevel.MODERATE: 0.0,
            MoodLevel.HIGH: 0.5,
            MoodLevel.VERY_HIGH: 1.0
        }
        return mapping[mood]

    def _energy_to_score(self, energy: EnergyLevel) -> float:
        """Convert energy level to numeric score."""
        mapping = {
            EnergyLevel.EXHAUSTED: 0.0,
            EnergyLevel.LOW: 0.25,
            EnergyLevel.MODERATE: 0.5,
            EnergyLevel.HIGH: 0.75,
            EnergyLevel.ENERGETIC: 1.0
        }
        return mapping[energy]

    def _score_to_mood(self, score: float) -> MoodLevel:
        """Convert numeric score to mood level."""
        if score >= 0.75:
            return MoodLevel.VERY_HIGH
        elif score >= 0.25:
            return MoodLevel.HIGH
        elif score >= -0.25:
            return MoodLevel.MODERATE
        elif score >= -0.75:
            return MoodLevel.LOW
        else:
            return MoodLevel.VERY_LOW

    def _score_to_energy(self, score: float) -> EnergyLevel:
        """Convert numeric score to energy level."""
        if score >= 0.9:
            return EnergyLevel.ENERGETIC
        elif score >= 0.7:
            return EnergyLevel.HIGH
        elif score >= 0.4:
            return EnergyLevel.MODERATE
        elif score >= 0.2:
            return EnergyLevel.LOW
        else:
            return EnergyLevel.EXHAUSTED

    def get_current_state(self) -> Optional[MoodEnergyState]:
        """Get current mood/energy state."""
        return self.current_state

    def get_recommendations(self) -> List[TaskRecommendation]:
        """Get task recommendations based on current state."""
        if not self.current_state:
            return []
        
        key = (self.current_state.mood_level, self.current_state.energy_level)
        recommendations = self.recommendations.get(key, [])
        
        # If no exact match, find closest match
        if not recommendations:
            recommendations = self._find_closest_recommendations(key)
        
        return recommendations

    def _find_closest_recommendations(self, target_key: Tuple[MoodLevel, EnergyLevel]) -> List[TaskRecommendation]:
        """Find closest recommendations when exact match not available."""
        target_mood, target_energy = target_key
        target_mood_score = self._mood_to_score(target_mood)
        target_energy_score = self._energy_to_score(target_energy)
        
        best_distance = float('inf')
        best_recommendations = []
        
        for (mood, energy), recommendations in self.recommendations.items():
            mood_score = self._mood_to_score(mood)
            energy_score = self._energy_to_score(energy)
            
            distance = abs(mood_score - target_mood_score) + abs(energy_score - target_energy_score)
            
            if distance < best_distance:
                best_distance = distance
                best_recommendations = recommendations
        
        return best_recommendations

    def get_state_history(self, hours: int = 24) -> List[MoodEnergyState]:
        """Get mood/energy state history for the last N hours."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        return [state for state in self.state_history if state.timestamp >= cutoff_time]

    async def cleanup(self) -> None:
        """Cleanup resources."""
        self.logger.info("MoodEnergyTracker cleanup completed")