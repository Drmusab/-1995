"""
Adaptive Recommendation Engine - محرك التوصيات التكيفي
Author: Drmusab
Last Modified: 2025-01-20

Provides intelligent task and schedule recommendations based on user's current mood, energy levels,
and historical patterns. Adapts advice to optimize productivity and well-being.
"""

import asyncio
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from src.core.config.loader import ConfigLoader
from src.core.dependency_injection import Container
from src.core.error_handling import ErrorHandler, handle_exceptions
from src.core.events.event_bus import EventBus
from src.observability.logging.config import get_logger

from .mood_energy_tracker import MoodEnergyState, MoodLevel, EnergyLevel, TaskRecommendation


class RecommendationType(Enum):
    """Types of recommendations."""
    RESCHEDULE = "إعادة جدولة"  # Reschedule
    TAKE_BREAK = "أخذ استراحة"  # Take Break
    SWITCH_TASK = "تغيير المهمة"  # Switch Task
    OPTIMIZE_TIMING = "تحسين التوقيت"  # Optimize Timing
    ENERGY_BOOST = "رفع الطاقة"  # Energy Boost


@dataclass
class AdaptiveRecommendation:
    """An adaptive recommendation based on current state."""
    id: str
    type: RecommendationType
    title_arabic: str
    title_english: str
    description_arabic: str
    description_english: str
    confidence: float  # 0-1
    suggested_action: str
    reasoning_arabic: str
    reasoning_english: str
    urgency: int  # 1-5, 5 being most urgent
    estimated_benefit: float  # 0-1
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class UserPattern:
    """User behavioral pattern."""
    pattern_id: str
    description: str
    triggers: List[str]  # What triggers this pattern
    outcomes: List[str]  # What typically happens
    confidence: float
    last_observed: datetime


class AdaptiveRecommendationEngine:
    """Generates intelligent recommendations based on user state and patterns."""

    def __init__(self, container: Container):
        self.container = container
        self.config = container.resolve(ConfigLoader)
        self.event_bus = container.resolve(EventBus)
        self.logger = get_logger(__name__)
        
        # Pattern tracking
        self.user_patterns: List[UserPattern] = []
        self.recommendation_history: List[AdaptiveRecommendation] = []
        self.max_history = 100
        
        # Initialize recommendation rules
        self._initialize_recommendation_rules()

    def _initialize_recommendation_rules(self) -> None:
        """Initialize recommendation rules based on mood/energy combinations."""
        self.recommendation_rules = {
            # Very tired or low energy
            (MoodLevel.VERY_LOW, EnergyLevel.EXHAUSTED): [
                {
                    "type": RecommendationType.TAKE_BREAK,
                    "title_ar": "حان وقت الراحة",
                    "title_en": "Time for a break",
                    "desc_ar": "تبدو متعباً جداً. خذ استراحة لمدة 15-30 دقيقة",
                    "desc_en": "You seem very tired. Take a 15-30 minute break",
                    "action": "schedule_break_30min",
                    "reasoning_ar": "طاقتك منخفضة جداً والاستمرار قد يقلل من جودة عملك",
                    "reasoning_en": "Your energy is very low and continuing may reduce work quality",
                    "urgency": 5
                }
            ],
            
            (MoodLevel.LOW, EnergyLevel.LOW): [
                {
                    "type": RecommendationType.SWITCH_TASK,
                    "title_ar": "جرب مهمة أسهل", 
                    "title_en": "Try an easier task",
                    "desc_ar": "انتقل إلى مهمة أسهل أو أكثر متعة",
                    "desc_en": "Switch to an easier or more enjoyable task",
                    "action": "suggest_light_tasks",
                    "reasoning_ar": "مزاجك منخفض وتحتاج شيء يحفزك",
                    "reasoning_en": "Your mood is low and you need something motivating",
                    "urgency": 3
                },
                {
                    "type": RecommendationType.ENERGY_BOOST,
                    "title_ar": "نشط طاقتك",
                    "title_en": "Boost your energy", 
                    "desc_ar": "تحرك قليلاً أو اشرب الماء أو تنفس بعمق",
                    "desc_en": "Move around, drink water, or take deep breaths",
                    "action": "energy_boost_routine",
                    "reasoning_ar": "تحتاج رفع مستوى الطاقة للتحسن",
                    "reasoning_en": "You need an energy boost to improve",
                    "urgency": 4
                }
            ],
            
            # Medium states
            (MoodLevel.MODERATE, EnergyLevel.MODERATE): [
                {
                    "type": RecommendationType.OPTIMIZE_TIMING,
                    "title_ar": "حسن توقيت مهامك",
                    "title_en": "Optimize your task timing",
                    "desc_ar": "ابدأ بالمهام المتوسطة الصعوبة",
                    "desc_en": "Start with medium difficulty tasks",
                    "action": "suggest_routine_tasks",
                    "reasoning_ar": "حالتك متوسطة - مناسبة للمهام العادية",
                    "reasoning_en": "Your state is moderate - suitable for routine tasks",
                    "urgency": 2
                }
            ],
            
            # High energy but low mood
            (MoodLevel.LOW, EnergyLevel.HIGH): [
                {
                    "type": RecommendationType.SWITCH_TASK,
                    "title_ar": "استغل طاقتك العالية",
                    "title_en": "Leverage your high energy",
                    "desc_ar": "رغم المزاج المنخفض، طاقتك عالية - جرب مهمة ممتعة",
                    "desc_en": "Despite low mood, your energy is high - try an enjoyable task",
                    "action": "suggest_engaging_tasks",
                    "reasoning_ar": "الطاقة العالية يمكن أن تحسن مزاجك",
                    "reasoning_en": "High energy can help improve your mood",
                    "urgency": 2
                }
            ],
            
            # Optimal states
            (MoodLevel.HIGH, EnergyLevel.HIGH): [
                {
                    "type": RecommendationType.OPTIMIZE_TIMING,
                    "title_ar": "وقت مثالي للمهام الصعبة",
                    "title_en": "Perfect time for challenging tasks",
                    "desc_ar": "حالتك ممتازة - اغتنم الفرصة للمهام المعقدة",
                    "desc_en": "You're in great state - seize the opportunity for complex tasks",
                    "action": "suggest_challenging_tasks",
                    "reasoning_ar": "مزاجك وطاقتك مثاليين للإنتاجية العالية",
                    "reasoning_en": "Your mood and energy are perfect for high productivity",
                    "urgency": 1
                }
            ],
            
            (MoodLevel.VERY_HIGH, EnergyLevel.ENERGETIC): [
                {
                    "type": RecommendationType.OPTIMIZE_TIMING,
                    "title_ar": "استثمر ذروة طاقتك",
                    "title_en": "Invest your peak energy",
                    "desc_ar": "أنت في أفضل حالاتك - ركز على أهم مشاريعك",
                    "desc_en": "You're at your best - focus on your most important projects",
                    "action": "suggest_priority_tasks",
                    "reasoning_ar": "هذه الحالة نادرة - استغلها للإنجازات الكبيرة",
                    "reasoning_en": "This state is rare - use it for major achievements",
                    "urgency": 1
                }
            ]
        }

    @handle_exceptions()
    async def generate_recommendations(
        self, 
        current_state: MoodEnergyState,
        current_tasks: List[Dict[str, Any]] = None,
        context: Dict[str, Any] = None
    ) -> List[AdaptiveRecommendation]:
        """Generate adaptive recommendations based on current state."""
        
        recommendations = []
        
        # Get rules for current mood/energy combination
        key = (current_state.mood_level, current_state.energy_level)
        rules = self.recommendation_rules.get(key, [])
        
        # If no exact match, find closest rules
        if not rules:
            rules = self._find_closest_rules(current_state)
        
        # Generate recommendations from rules
        for rule in rules:
            recommendation = self._create_recommendation_from_rule(rule, current_state)
            recommendations.append(recommendation)
        
        # Add context-specific recommendations
        if current_tasks:
            context_recs = await self._generate_context_recommendations(
                current_state, current_tasks, context
            )
            recommendations.extend(context_recs)
        
        # Add pattern-based recommendations
        pattern_recs = self._generate_pattern_recommendations(current_state)
        recommendations.extend(pattern_recs)
        
        # Sort by urgency and confidence
        recommendations.sort(key=lambda r: (r.urgency, -r.confidence))
        
        # Store in history
        self.recommendation_history.extend(recommendations)
        if len(self.recommendation_history) > self.max_history:
            self.recommendation_history = self.recommendation_history[-self.max_history:]
        
        self.logger.info(f"Generated {len(recommendations)} recommendations")
        return recommendations[:5]  # Return top 5

    def _find_closest_rules(self, state: MoodEnergyState) -> List[Dict[str, Any]]:
        """Find closest matching rules when exact match not available."""
        target_mood = state.mood_level
        target_energy = state.energy_level
        
        # Calculate distances to all rule combinations
        best_distance = float('inf')
        best_rules = []
        
        for (mood, energy), rules in self.recommendation_rules.items():
            mood_diff = abs(self._mood_to_value(mood) - self._mood_to_value(target_mood))
            energy_diff = abs(self._energy_to_value(energy) - self._energy_to_value(target_energy))
            distance = mood_diff + energy_diff
            
            if distance < best_distance:
                best_distance = distance
                best_rules = rules
        
        return best_rules

    def _mood_to_value(self, mood: MoodLevel) -> int:
        """Convert mood to numeric value for distance calculation."""
        mapping = {
            MoodLevel.VERY_LOW: 1,
            MoodLevel.LOW: 2,
            MoodLevel.MODERATE: 3,
            MoodLevel.HIGH: 4,
            MoodLevel.VERY_HIGH: 5
        }
        return mapping[mood]

    def _energy_to_value(self, energy: EnergyLevel) -> int:
        """Convert energy to numeric value for distance calculation."""
        mapping = {
            EnergyLevel.EXHAUSTED: 1,
            EnergyLevel.LOW: 2,
            EnergyLevel.MODERATE: 3,
            EnergyLevel.HIGH: 4,
            EnergyLevel.ENERGETIC: 5
        }
        return mapping[energy]

    def _create_recommendation_from_rule(
        self, 
        rule: Dict[str, Any], 
        state: MoodEnergyState
    ) -> AdaptiveRecommendation:
        """Create recommendation object from rule."""
        return AdaptiveRecommendation(
            id=f"rule_{hash(str(rule))}_{int(datetime.now().timestamp())}",
            type=rule["type"],
            title_arabic=rule["title_ar"],
            title_english=rule["title_en"],
            description_arabic=rule["desc_ar"],
            description_english=rule["desc_en"],
            confidence=state.confidence,
            suggested_action=rule["action"],
            reasoning_arabic=rule["reasoning_ar"],
            reasoning_english=rule["reasoning_en"],
            urgency=rule["urgency"],
            estimated_benefit=self._calculate_benefit(rule["type"], state)
        )

    async def _generate_context_recommendations(
        self,
        state: MoodEnergyState,
        tasks: List[Dict[str, Any]],
        context: Dict[str, Any]
    ) -> List[AdaptiveRecommendation]:
        """Generate recommendations based on current tasks and context."""
        recommendations = []
        
        # Check if user should reschedule based on current task difficulty and state
        if tasks:
            current_task = tasks[0] if tasks else None
            if current_task and self._should_reschedule(current_task, state):
                recommendations.append(self._create_reschedule_recommendation(current_task, state))
        
        # Check for time-based recommendations
        current_hour = datetime.now().hour
        if self._is_deep_work_time(current_hour) and state.energy_level in [EnergyLevel.HIGH, EnergyLevel.ENERGETIC]:
            recommendations.append(self._create_deep_work_recommendation(state))
        
        return recommendations

    def _should_reschedule(self, task: Dict[str, Any], state: MoodEnergyState) -> bool:
        """Determine if current task should be rescheduled based on state."""
        task_difficulty = task.get("difficulty", "medium")
        
        # High difficulty tasks need good mood/energy
        if task_difficulty == "high" and (
            state.mood_level in [MoodLevel.VERY_LOW, MoodLevel.LOW] or
            state.energy_level in [EnergyLevel.EXHAUSTED, EnergyLevel.LOW]
        ):
            return True
        
        # Creative tasks need good mood
        if task.get("type") == "creative" and state.mood_level in [MoodLevel.VERY_LOW, MoodLevel.LOW]:
            return True
        
        return False

    def _create_reschedule_recommendation(
        self, 
        task: Dict[str, Any], 
        state: MoodEnergyState
    ) -> AdaptiveRecommendation:
        """Create reschedule recommendation."""
        return AdaptiveRecommendation(
            id=f"reschedule_{task.get('id', 'unknown')}_{int(datetime.now().timestamp())}",
            type=RecommendationType.RESCHEDULE,
            title_arabic=f"أعد جدولة '{task.get('title', 'المهمة')}'",
            title_english=f"Reschedule '{task.get('title', 'task')}'",
            description_arabic="هذه المهمة صعبة لحالتك الحالية - أجلها لوقت أفضل",
            description_english="This task is challenging for your current state - postpone to a better time",
            confidence=state.confidence,
            suggested_action="reschedule_task",
            reasoning_arabic="مزاجك أو طاقتك غير مناسبين لهذه المهمة الآن",
            reasoning_english="Your mood or energy isn't suitable for this task right now",
            urgency=3,
            estimated_benefit=0.7
        )

    def _is_deep_work_time(self, hour: int) -> bool:
        """Check if current time is good for deep work."""
        # Typically 9-11 AM and 2-4 PM are good for deep work
        return (9 <= hour <= 11) or (14 <= hour <= 16)

    def _create_deep_work_recommendation(self, state: MoodEnergyState) -> AdaptiveRecommendation:
        """Create deep work recommendation."""
        return AdaptiveRecommendation(
            id=f"deep_work_{int(datetime.now().timestamp())}",
            type=RecommendationType.OPTIMIZE_TIMING,
            title_arabic="وقت مثالي للعمل العميق",
            title_english="Perfect time for deep work",
            description_arabic="طاقتك عالية والوقت مناسب - ركز على المهام المعقدة",
            description_english="Your energy is high and timing is right - focus on complex tasks",
            confidence=state.confidence,
            suggested_action="start_deep_work",
            reasoning_arabic="الوقت الحالي والطاقة العالية مثاليان للتركيز",
            reasoning_english="Current time and high energy are perfect for focus",
            urgency=1,
            estimated_benefit=0.9
        )

    def _generate_pattern_recommendations(self, state: MoodEnergyState) -> List[AdaptiveRecommendation]:
        """Generate recommendations based on learned user patterns."""
        recommendations = []
        
        # For now, return empty list - patterns would be learned over time
        # This is where machine learning models would analyze user behavior
        
        return recommendations

    def _calculate_benefit(self, rec_type: RecommendationType, state: MoodEnergyState) -> float:
        """Calculate estimated benefit of recommendation."""
        # Base benefit calculation
        base_benefit = {
            RecommendationType.TAKE_BREAK: 0.8,
            RecommendationType.SWITCH_TASK: 0.6,
            RecommendationType.RESCHEDULE: 0.7,
            RecommendationType.OPTIMIZE_TIMING: 0.9,
            RecommendationType.ENERGY_BOOST: 0.5
        }.get(rec_type, 0.5)
        
        # Adjust based on state confidence
        return base_benefit * state.confidence

    async def get_recommendation_summary(self, language: str = "ar") -> str:
        """Get a summary of recent recommendations."""
        recent_recs = self.recommendation_history[-5:] if self.recommendation_history else []
        
        if not recent_recs:
            if language == "ar":
                return "لا توجد توصيات حديثة"
            else:
                return "No recent recommendations"
        
        if language == "ar":
            summary = "التوصيات الحديثة:\n\n"
            for i, rec in enumerate(recent_recs, 1):
                summary += f"{i}. {rec.title_arabic}\n   {rec.description_arabic}\n\n"
        else:
            summary = "Recent recommendations:\n\n"
            for i, rec in enumerate(recent_recs, 1):
                summary += f"{i}. {rec.title_english}\n   {rec.description_english}\n\n"
        
        return summary.strip()

    def learn_from_feedback(self, recommendation_id: str, feedback: str, outcome: str) -> None:
        """Learn from user feedback on recommendations."""
        # Find the recommendation
        recommendation = next(
            (r for r in self.recommendation_history if r.id == recommendation_id),
            None
        )
        
        if not recommendation:
            return
        
        # Store feedback for future learning
        # This would feed into a machine learning model
        self.logger.info(f"Feedback on recommendation {recommendation_id}: {feedback} -> {outcome}")

    async def cleanup(self) -> None:
        """Cleanup resources."""
        self.logger.info("AdaptiveRecommendationEngine cleanup completed")