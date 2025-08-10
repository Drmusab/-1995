"""
Assistant personality traits management.

This module defines and manages the AI assistant's personality characteristics,
including trait evolution, adaptation based on user interactions, and
personality-driven behavior modifications.
"""

import asyncio
import logging
import uuid
import json
from datetime import datetime, timezone, timedelta
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np

from src.core.events.event_bus import EventBus
from src.core.events.event_types import Event, EventType
from src.core.error_handling import (
    ValidationError
)
from src.core.config.loader import ConfigLoader
from src.memory.core_memory.memory_manager import MemoryManager
from src.assistant.context_manager import ContextManager, ContextType
from src.learning.preference_learning import PreferenceLearner
from src.learning.feedback_processor import FeedbackProcessor
from src.integrations.cache.cache_strategy import CacheStrategy
from src.processing.natural_language.sentiment_analyzer import SentimentAnalyzer

logger = logging.getLogger(__name__)


class PersonalityDimension(Enum):
    """Core personality dimensions based on psychological models."""
    # Big Five personality traits
    OPENNESS = auto()              # Creativity, curiosity, open to new experiences
    CONSCIENTIOUSNESS = auto()     # Organization, dependability, goal-oriented
    EXTRAVERSION = auto()          # Sociability, assertiveness, energy
    AGREEABLENESS = auto()         # Cooperation, trust, empathy
    NEUROTICISM = auto()           # Emotional stability vs. anxiety
    
    # Additional AI-specific traits
    HUMOR = auto()                 # Sense of humor and playfulness
    FORMALITY = auto()             # Professional vs. casual communication
    VERBOSITY = auto()             # Concise vs. detailed responses
    PROACTIVITY = auto()           # Initiative in offering help
    CREATIVITY = auto()            # Novel and creative solutions
    PATIENCE = auto()              # Tolerance and understanding
    ASSERTIVENESS = auto()         # Confidence in recommendations
    ADAPTABILITY = auto()          # Flexibility in approach
    EMPATHY = auto()               # Understanding and emotional support
    TECHNICAL_DEPTH = auto()       # Level of technical detail


class InteractionStyle(Enum):
    """Communication and interaction styles."""
    TEACHER = auto()               # Educational, explanatory
    COACH = auto()                 # Motivational, supportive
    ASSISTANT = auto()             # Helpful, service-oriented
    COMPANION = auto()             # Friendly, conversational
    ADVISOR = auto()               # Strategic, consultative
    COLLABORATOR = auto()          # Partnership, co-creative
    MENTOR = auto()                # Guiding, wisdom-sharing


class MoodState(Enum):
    """Current mood states that affect personality expression."""
    NEUTRAL = auto()
    CHEERFUL = auto()
    FOCUSED = auto()
    CONTEMPLATIVE = auto()
    ENERGETIC = auto()
    CALM = auto()
    SYMPATHETIC = auto()
    PLAYFUL = auto()


@dataclass
class PersonalityTrait:
    """Individual personality trait with value and metadata."""
    dimension: PersonalityDimension
    value: float  # 0.0 to 1.0
    confidence: float = 1.0
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    adaptation_rate: float = 0.1
    stability: float = 0.8  # How resistant to change
    
    def update(self, new_value: float, confidence: float = 1.0) -> None:
        """Update trait value with stability consideration."""
        # Apply stability factor to limit rapid changes
        change = (new_value - self.value) * self.adaptation_rate * confidence
        self.value = np.clip(self.value + change * (1 - self.stability), 0.0, 1.0)
        self.confidence = confidence
        self.last_updated = datetime.now(timezone.utc)
    
    def decay_confidence(self, decay_rate: float = 0.99) -> None:
        """Decay confidence over time."""
        self.confidence *= decay_rate


@dataclass
class PersonalityProfile:
    """Complete personality profile for a session/user."""
    profile_id: str
    user_id: str
    session_id: str
    traits: Dict[PersonalityDimension, PersonalityTrait] = field(default_factory=dict)
    interaction_style: InteractionStyle = InteractionStyle.ASSISTANT
    mood_state: MoodState = MoodState.NEUTRAL
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_active: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    interaction_count: int = 0
    adaptation_enabled: bool = True
    custom_traits: Dict[str, float] = field(default_factory=dict)
    
    def get_trait_value(self, dimension: PersonalityDimension) -> float:
        """Get value for a personality dimension."""
        if dimension in self.traits:
            return self.traits[dimension].value
        return 0.5  # Default neutral value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert profile to dictionary."""
        return {
            "profile_id": self.profile_id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "traits": {
                dim.name: {
                    "value": trait.value,
                    "confidence": trait.confidence
                }
                for dim, trait in self.traits.items()
            },
            "interaction_style": self.interaction_style.name,
            "mood_state": self.mood_state.name,
            "interaction_count": self.interaction_count,
            "adaptation_enabled": self.adaptation_enabled,
            "custom_traits": self.custom_traits
        }


@dataclass
class PersonalityAdaptation:
    """Record of personality adaptation event."""
    adaptation_id: str
    profile_id: str
    timestamp: datetime
    trigger: str  # What caused the adaptation
    changes: Dict[PersonalityDimension, Tuple[float, float]]  # (old, new) values
    confidence: float
    context: Dict[str, Any] = field(default_factory=dict)


class PersonalityManager:
    """
    Manages AI assistant personality traits and their evolution.
    
    This class handles personality initialization, adaptation based on
    interactions, mood management, and personality-driven behaviors.
    """
    
    def __init__(
        self,
        event_bus: EventBus,
        memory_manager: MemoryManager,
        context_manager: ContextManager,
        preference_learner: PreferenceLearner,
        feedback_processor: FeedbackProcessor,
        sentiment_analyzer: SentimentAnalyzer,
        cache_strategy: CacheStrategy,
        config_loader: ConfigLoader
    ):
        """Initialize the personality manager with dependencies."""
        self.event_bus = event_bus
        self.memory_manager = memory_manager
        self.context_manager = context_manager
        self.preference_learner = preference_learner
        self.feedback_processor = feedback_processor
        self.sentiment_analyzer = sentiment_analyzer
        self.cache = cache_strategy
        self.config = config_loader.get_config("personality")
        
        # Personality profiles storage
        self.profiles: Dict[str, PersonalityProfile] = {}
        self.adaptation_history: Dict[str, List[PersonalityAdaptation]] = defaultdict(list)
        
        # Configuration
        self.default_traits = self._load_default_traits()
        self.adaptation_thresholds = self.config.get("adaptation_thresholds", {
            "interaction_count": 10,
            "confidence_minimum": 0.6,
            "time_window_hours": 24
        })
        
        # Personality archetypes
        self.archetypes = self._load_archetypes()
        
        # Mood influences on traits
        self.mood_modifiers = self._load_mood_modifiers()
        
        # Interaction style rules
        self.style_rules = self._load_style_rules()
        
        # Learning parameters
        self.learning_rate = self.config.get("learning_rate", 0.1)
        self.momentum = self.config.get("momentum", 0.9)
        self.trait_correlations = self._load_trait_correlations()
        
        # Cache configuration
        self.cache_ttl = self.config.get("cache_ttl", 300)
        
        # Subscribe to events
        self._subscribe_to_events()
        
        # Start background tasks
        asyncio.create_task(self._periodic_mood_update())
        asyncio.create_task(self._confidence_decay_task())
        
        logger.info("PersonalityManager initialized successfully")
    
    def _subscribe_to_events(self) -> None:
        """Subscribe to relevant system events."""
        self.event_bus.subscribe(EventType.USER_INPUT, self._handle_user_interaction)
        self.event_bus.subscribe(EventType.FEEDBACK_RECEIVED, self._handle_feedback)
        self.event_bus.subscribe(EventType.EMOTION_DETECTED, self._handle_emotion)
        self.event_bus.subscribe(EventType.CONTEXT_UPDATED, self._handle_context_update)
        self.event_bus.subscribe(EventType.PREFERENCE_LEARNED, self._handle_preference)
    
    def _load_default_traits(self) -> Dict[PersonalityDimension, float]:
        """Load default personality trait values."""
        defaults = self.config.get("default_traits", {})
        
        # Convert string keys to enum and ensure all dimensions have values
        trait_values = {}
        for dimension in PersonalityDimension:
            config_value = defaults.get(dimension.name, 0.5)
            trait_values[dimension] = float(config_value)
        
        return trait_values
    
    def _load_archetypes(self) -> Dict[str, Dict[PersonalityDimension, float]]:
        """Load predefined personality archetypes."""
        return {
            "professional": {
                PersonalityDimension.FORMALITY: 0.8,
                PersonalityDimension.CONSCIENTIOUSNESS: 0.9,
                PersonalityDimension.ASSERTIVENESS: 0.7,
                PersonalityDimension.TECHNICAL_DEPTH: 0.8,
                PersonalityDimension.HUMOR: 0.3
            },
            "friendly": {
                PersonalityDimension.AGREEABLENESS: 0.9,
                PersonalityDimension.EXTRAVERSION: 0.8,
                PersonalityDimension.HUMOR: 0.7,
                PersonalityDimension.FORMALITY: 0.3,
                PersonalityDimension.EMPATHY: 0.8
            },
            "teacher": {
                PersonalityDimension.PATIENCE: 0.9,
                PersonalityDimension.VERBOSITY: 0.7,
                PersonalityDimension.CONSCIENTIOUSNESS: 0.8,
                PersonalityDimension.CREATIVITY: 0.6,
                PersonalityDimension.ADAPTABILITY: 0.8
            },
            "creative": {
                PersonalityDimension.OPENNESS: 0.9,
                PersonalityDimension.CREATIVITY: 0.9,
                PersonalityDimension.HUMOR: 0.6,
                PersonalityDimension.ADAPTABILITY: 0.8,
                PersonalityDimension.FORMALITY: 0.4
            },
            "analytical": {
                PersonalityDimension.CONSCIENTIOUSNESS: 0.9,
                PersonalityDimension.TECHNICAL_DEPTH: 0.9,
                PersonalityDimension.ASSERTIVENESS: 0.6,
                PersonalityDimension.VERBOSITY: 0.6,
                PersonalityDimension.NEUROTICISM: 0.3
            }
        }
    
    def _load_mood_modifiers(self) -> Dict[MoodState, Dict[PersonalityDimension, float]]:
        """Load mood-based personality modifiers."""
        return {
            MoodState.CHEERFUL: {
                PersonalityDimension.EXTRAVERSION: 0.2,
                PersonalityDimension.HUMOR: 0.3,
                PersonalityDimension.AGREEABLENESS: 0.1
            },
            MoodState.FOCUSED: {
                PersonalityDimension.CONSCIENTIOUSNESS: 0.2,
                PersonalityDimension.VERBOSITY: -0.1,
                PersonalityDimension.TECHNICAL_DEPTH: 0.1
            },
            MoodState.SYMPATHETIC: {
                PersonalityDimension.EMPATHY: 0.3,
                PersonalityDimension.AGREEABLENESS: 0.2,
                PersonalityDimension.PATIENCE: 0.2
            },
            MoodState.PLAYFUL: {
                PersonalityDimension.HUMOR: 0.4,
                PersonalityDimension.CREATIVITY: 0.2,
                PersonalityDimension.FORMALITY: -0.3
            },
            MoodState.CONTEMPLATIVE: {
                PersonalityDimension.OPENNESS: 0.2,
                PersonalityDimension.VERBOSITY: 0.1,
                PersonalityDimension.CREATIVITY: 0.1
            }
        }
    
    def _load_style_rules(self) -> Dict[InteractionStyle, Dict[str, Any]]:
        """Load rules for interaction styles."""
        return {
            InteractionStyle.TEACHER: {
                "required_traits": {
                    PersonalityDimension.PATIENCE: 0.7,
                    PersonalityDimension.VERBOSITY: 0.6
                },
                "behaviors": ["explain_concepts", "provide_examples", "check_understanding"]
            },
            InteractionStyle.COACH: {
                "required_traits": {
                    PersonalityDimension.ASSERTIVENESS: 0.6,
                    PersonalityDimension.EMPATHY: 0.7
                },
                "behaviors": ["motivate", "set_goals", "track_progress"]
            },
            InteractionStyle.COMPANION: {
                "required_traits": {
                    PersonalityDimension.AGREEABLENESS: 0.7,
                    PersonalityDimension.HUMOR: 0.5
                },
                "behaviors": ["share_stories", "show_interest", "be_supportive"]
            },
            InteractionStyle.ADVISOR: {
                "required_traits": {
                    PersonalityDimension.CONSCIENTIOUSNESS: 0.7,
                    PersonalityDimension.TECHNICAL_DEPTH: 0.6
                },
                "behaviors": ["analyze_options", "provide_recommendations", "strategic_thinking"]
            }
        }
    
    def _load_trait_correlations(self) -> Dict[Tuple[PersonalityDimension, PersonalityDimension], float]:
        """Load correlations between personality traits."""
        return {
            (PersonalityDimension.OPENNESS, PersonalityDimension.CREATIVITY): 0.8,
            (PersonalityDimension.CONSCIENTIOUSNESS, PersonalityDimension.PATIENCE): 0.6,
            (PersonalityDimension.EXTRAVERSION, PersonalityDimension.HUMOR): 0.5,
            (PersonalityDimension.AGREEABLENESS, PersonalityDimension.EMPATHY): 0.7,
            (PersonalityDimension.NEUROTICISM, PersonalityDimension.PATIENCE): -0.4,
            (PersonalityDimension.FORMALITY, PersonalityDimension.HUMOR): -0.3,
            (PersonalityDimension.TECHNICAL_DEPTH, PersonalityDimension.VERBOSITY): 0.4
        }
    
    async def initialize_profile(
        self,
        user_id: str,
        session_id: str,
        archetype: Optional[str] = None,
        custom_traits: Optional[Dict[str, float]] = None
    ) -> PersonalityProfile:
        """
        Initialize a personality profile for a user/session.
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            archetype: Optional archetype to base personality on
            custom_traits: Optional custom trait values
            
        Returns:
            Initialized personality profile
        """
        try:
            # Check if profile already exists
            profile_key = f"{user_id}:{session_id}"
            if profile_key in self.profiles:
                return self.profiles[profile_key]
            
            # Create new profile
            profile = PersonalityProfile(
                profile_id=str(uuid.uuid4()),
                user_id=user_id,
                session_id=session_id
            )
            
            # Initialize traits
            base_traits = self.default_traits.copy()
            
            # Apply archetype if specified
            if archetype and archetype in self.archetypes:
                archetype_traits = self.archetypes[archetype]
                for dimension, value in archetype_traits.items():
                    base_traits[dimension] = value
            
            # Apply custom traits
            if custom_traits:
                for dimension_name, value in custom_traits.items():
                    try:
                        dimension = PersonalityDimension[dimension_name]
                        base_traits[dimension] = np.clip(float(value), 0.0, 1.0)
                    except (KeyError, ValueError):
                        profile.custom_traits[dimension_name] = float(value)
            
            # Load user's historical preferences
            user_preferences = await self._load_user_personality_preferences(user_id)
            if user_preferences:
                for dimension, value in user_preferences.items():
                    if dimension in base_traits:
                        # Blend with historical preferences
                        base_traits[dimension] = (
                            base_traits[dimension] * 0.3 + value * 0.7
                        )
            
            # Create trait objects
            for dimension, value in base_traits.items():
                profile.traits[dimension] = PersonalityTrait(
                    dimension=dimension,
                    value=value,
                    confidence=1.0 if not user_preferences else 0.8
                )
            
            # Determine initial interaction style
            profile.interaction_style = self._determine_interaction_style(profile)
            
            # Store profile
            self.profiles[profile_key] = profile
            
            # Cache profile
            await self._cache_profile(profile)
            
            # Emit initialization event
            await self.event_bus.emit(Event(
                type=EventType.PERSONALITY_INITIALIZED,
                data={
                    "user_id": user_id,
                    "session_id": session_id,
                    "profile_id": profile.profile_id,
                    "archetype": archetype
                }
            ))
            
            logger.info(f"Initialized personality profile {profile.profile_id}")
            return profile
            
        except Exception as e:
            logger.error(f"Failed to initialize personality profile: {str(e)}")
            raise
    
    async def get_current_traits(
        self,
        session_id: str,
        include_mood_modifiers: bool = True
    ) -> Dict[str, float]:
        """
        Get current personality traits for a session.
        
        Args:
            session_id: Session identifier
            include_mood_modifiers: Whether to include mood-based modifications
            
        Returns:
            Dictionary of trait names to values
        """
        # Find profile by session
        profile = None
        for key, p in self.profiles.items():
            if p.session_id == session_id:
                profile = p
                break
        
        if not profile:
            # Return default traits if no profile
            return {
                dim.name.lower(): value 
                for dim, value in self.default_traits.items()
            }
        
        # Get base trait values
        traits = {}
        for dimension, trait in profile.traits.items():
            value = trait.value
            
            # Apply mood modifiers if requested
            if include_mood_modifiers and profile.mood_state in self.mood_modifiers:
                modifiers = self.mood_modifiers[profile.mood_state]
                if dimension in modifiers:
                    value = np.clip(value + modifiers[dimension], 0.0, 1.0)
            
            traits[dimension.name.lower()] = value
        
        # Add custom traits
        traits.update(profile.custom_traits)
        
        return traits
    
    async def adapt_personality(
        self,
        session_id: str,
        interaction_data: Dict[str, Any],
        feedback: Optional[Dict[str, Any]] = None
    ) -> Optional[PersonalityAdaptation]:
        """
        Adapt personality based on interaction and feedback.
        
        Args:
            session_id: Session identifier
            interaction_data: Data about the interaction
            feedback: Optional user feedback
            
        Returns:
            Adaptation record if adaptation occurred
        """
        # Find profile
        profile = None
        for key, p in self.profiles.items():
            if p.session_id == session_id:
                profile = p
                break
        
        if not profile or not profile.adaptation_enabled:
            return None
        
        # Check if adaptation threshold is met
        if profile.interaction_count < self.adaptation_thresholds["interaction_count"]:
            profile.interaction_count += 1
            return None
        
        # Analyze interaction for trait indicators
        trait_signals = await self._analyze_interaction_signals(
            interaction_data, feedback
        )
        
        if not trait_signals:
            return None
        
        # Create adaptation record
        adaptation = PersonalityAdaptation(
            adaptation_id=str(uuid.uuid4()),
            profile_id=profile.profile_id,
            timestamp=datetime.now(timezone.utc),
            trigger="interaction_feedback",
            changes={},
            confidence=trait_signals.get("confidence", 0.5),
            context=interaction_data
        )
        
        # Apply adaptations
        for dimension, target_value in trait_signals.get("traits", {}).items():
            if dimension in profile.traits:
                old_value = profile.traits[dimension].value
                
                # Apply correlated trait updates
                correlated_updates = self._calculate_correlated_updates(
                    dimension, target_value, profile
                )
                
                # Update main trait
                profile.traits[dimension].update(
                    target_value,
                    confidence=trait_signals["confidence"]
                )
                
                new_value = profile.traits[dimension].value
                adaptation.changes[dimension] = (old_value, new_value)
                
                # Update correlated traits
                for corr_dim, corr_value in correlated_updates.items():
                    if corr_dim in profile.traits:
                        old_corr = profile.traits[corr_dim].value
                        profile.traits[corr_dim].update(
                            corr_value,
                            confidence=trait_signals["confidence"] * 0.5
                        )
                        adaptation.changes[corr_dim] = (
                            old_corr, profile.traits[corr_dim].value
                        )
        
        # Update mood if indicated
        if "mood" in trait_signals:
            profile.mood_state = trait_signals["mood"]
        
        # Update interaction style if needed
        new_style = self._determine_interaction_style(profile)
        if new_style != profile.interaction_style:
            profile.interaction_style = new_style
            adaptation.context["style_change"] = {
                "from": profile.interaction_style.name,
                "to": new_style.name
            }
        
        # Store adaptation history
        self.adaptation_history[profile.profile_id].append(adaptation)
        
        # Update profile timestamp
        profile.last_active = datetime.now(timezone.utc)
        profile.interaction_count += 1
        
        # Emit adaptation event
        await self.event_bus.emit(Event(
            type=EventType.PERSONALITY_ADJUSTED,
            data={
                "session_id": session_id,
                "adaptation_id": adaptation.adaptation_id,
                "changes": {
                    dim.name: {"from": old, "to": new}
                    for dim, (old, new) in adaptation.changes.items()
                }
            }
        ))
        
        return adaptation
    
    async def set_mood(
        self,
        session_id: str,
        mood: MoodState,
        reason: Optional[str] = None
    ) -> None:
        """
        Set the current mood state.
        
        Args:
            session_id: Session identifier
            mood: New mood state
            reason: Optional reason for mood change
        """
        profile = None
        for key, p in self.profiles.items():
            if p.session_id == session_id:
                profile = p
                break
        
        if not profile:
            return
        
        old_mood = profile.mood_state
        profile.mood_state = mood
        
        logger.info(f"Mood changed from {old_mood.name} to {mood.name} for session {session_id}")
        
        # Emit mood change event
        await self.event_bus.emit(Event(
            type=EventType.MOOD_CHANGED,
            data={
                "session_id": session_id,
                "old_mood": old_mood.name,
                "new_mood": mood.name,
                "reason": reason
            }
        ))
    
    async def get_interaction_style(self, session_id: str) -> InteractionStyle:
        """Get current interaction style for a session."""
        profile = None
        for key, p in self.profiles.items():
            if p.session_id == session_id:
                profile = p
                break
        
        if not profile:
            return InteractionStyle.ASSISTANT
        
        return profile.interaction_style
    
    async def suggest_response_modifications(
        self,
        session_id: str,
        base_response: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Suggest personality-based modifications to a response.
        
        Args:
            session_id: Session identifier
            base_response: Original response text
            context: Response context
            
        Returns:
            Suggested modifications
        """
        profile = None
        for key, p in self.profiles.items():
            if p.session_id == session_id:
                profile = p
                break
        
        if not profile:
            return {"modifications": []}
        
        # Analyze base response for personality-driven modifications
        response_length = len(base_response.split())
        
        modifications = []
        
        # Get current traits with mood modifiers
        current_traits = await self.get_current_traits(session_id, include_mood_modifiers=True)
        
        # Humor modifications
        if current_traits.get("humor", 0) > 0.7:
            modifications.append({
                "type": "add_humor",
                "suggestion": "Consider adding a light joke or pun",
                "confidence": current_traits["humor"]
            })
        
        # Verbosity modifications
        verbosity = current_traits.get("verbosity", 0.5)
        if verbosity < 0.3:
            modifications.append({
                "type": "make_concise",
                "suggestion": "Shorten response for brevity",
                "confidence": 1.0 - verbosity
            })
        elif verbosity > 0.7:
            modifications.append({
                "type": "add_detail",
                "suggestion": "Expand with more examples or explanations",
                "confidence": verbosity
            })
        
        # Formality modifications
        formality = current_traits.get("formality", 0.5)
        if formality < 0.3:
            modifications.append({
                "type": "make_casual",
                "suggestion": "Use more casual language and contractions",
                "confidence": 1.0 - formality
            })
        elif formality > 0.7:
            modifications.append({
                "type": "make_formal",
                "suggestion": "Use more professional language",
                "confidence": formality
            })
        
        # Empathy modifications
        if current_traits.get("empathy", 0) > 0.7 and context.get("user_emotion") == "negative":
            modifications.append({
                "type": "add_empathy",
                "suggestion": "Add empathetic acknowledgment of user's feelings",
                "confidence": current_traits["empathy"]
            })
        
        # Creativity modifications
        if current_traits.get("creativity", 0) > 0.6:
            modifications.append({
                "type": "add_creativity",
                "suggestion": "Consider a more creative or novel approach",
                "confidence": current_traits["creativity"]
            })
        
        # Style-specific modifications
        style_mods = self._get_style_modifications(profile.interaction_style, context)
        modifications.extend(style_mods)
        
        return {
            "modifications": modifications,
            "primary_traits": {
                k: v for k, v in current_traits.items()
                if k in ["humor", "formality", "verbosity", "empathy"]
            },
            "interaction_style": profile.interaction_style.name,
            "mood": profile.mood_state.name
        }
    
    def _determine_interaction_style(self, profile: PersonalityProfile) -> InteractionStyle:
        """Determine appropriate interaction style based on traits."""
        # Score each style based on trait match
        style_scores = {}
        
        for style, rules in self.style_rules.items():
            score = 0.0
            required_traits = rules.get("required_traits", {})
            
            for dimension, min_value in required_traits.items():
                if dimension in profile.traits:
                    trait_value = profile.traits[dimension].value
                    if trait_value >= min_value:
                        score += 1.0
                    else:
                        score += trait_value / min_value
            
            style_scores[style] = score / len(required_traits) if required_traits else 0.5
        
        # Return style with highest score
        if style_scores:
            return max(style_scores.items(), key=lambda x: x[1])[0]
        
        return InteractionStyle.ASSISTANT
    
    async def _analyze_interaction_signals(
        self,
        interaction_data: Dict[str, Any],
        feedback: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Analyze interaction for personality trait signals."""
        signals = {
            "traits": {},
            "confidence": 0.5
        }
        
        # Analyze user sentiment
        if "user_input" in interaction_data:
            sentiment = await self.sentiment_analyzer.analyze(
                interaction_data["user_input"]
            )
            
            # Adjust agreeableness based on user sentiment
            if sentiment.get("sentiment") == "negative":
                signals["traits"][PersonalityDimension.AGREEABLENESS] = 0.8
                signals["traits"][PersonalityDimension.EMPATHY] = 0.7
            
        # Analyze feedback if provided
        if feedback:
            rating = feedback.get("rating", 3)
            
            if rating <= 2:
                # Poor rating - adjust traits
                if feedback.get("too_verbose"):
                    signals["traits"][PersonalityDimension.VERBOSITY] = 0.3
                if feedback.get("too_formal"):
                    signals["traits"][PersonalityDimension.FORMALITY] = 0.3
                if feedback.get("not_helpful"):
                    signals["traits"][PersonalityDimension.PROACTIVITY] = 0.7
                    
            elif rating >= 4:
                # Good rating - reinforce current traits
                signals["confidence"] = 0.8
        
        # Analyze interaction patterns
        if "response_time" in interaction_data:
            # Quick responses might indicate need for more patience
            if interaction_data["response_time"] < 2.0:
                signals["traits"][PersonalityDimension.PATIENCE] = 0.8
        
        # Analyze topic for trait adjustments
        if "topic" in interaction_data:
            topic = interaction_data["topic"].lower()
            if "technical" in topic or "code" in topic:
                signals["traits"][PersonalityDimension.TECHNICAL_DEPTH] = 0.8
            elif "creative" in topic or "idea" in topic:
                signals["traits"][PersonalityDimension.CREATIVITY] = 0.7
        
        # Determine mood from context
        if "emotion" in interaction_data:
            emotion = interaction_data["emotion"]
            mood_map = {
                "happy": MoodState.CHEERFUL,
                "focused": MoodState.FOCUSED,
                "sad": MoodState.SYMPATHETIC,
                "playful": MoodState.PLAYFUL,
                "thoughtful": MoodState.CONTEMPLATIVE
            }
            if emotion in mood_map:
                signals["mood"] = mood_map[emotion]
        
        return signals if signals["traits"] else None
    
    def _calculate_correlated_updates(
        self,
        dimension: PersonalityDimension,
        target_value: float,
        profile: PersonalityProfile
    ) -> Dict[PersonalityDimension, float]:
        """Calculate updates for correlated traits."""
        correlated_updates = {}
        
        for (dim1, dim2), correlation in self.trait_correlations.items():
            if dim1 == dimension:
                # Calculate correlated change
                current_value = profile.get_trait_value(dim2)
                change = (target_value - profile.get_trait_value(dimension)) * correlation
                new_value = np.clip(current_value + change * 0.5, 0.0, 1.0)
                correlated_updates[dim2] = new_value
                
            elif dim2 == dimension:
                # Reverse correlation
                current_value = profile.get_trait_value(dim1)
                change = (target_value - profile.get_trait_value(dimension)) * correlation
                new_value = np.clip(current_value + change * 0.5, 0.0, 1.0)
                correlated_updates[dim1] = new_value
        
        return correlated_updates
    
    def _get_style_modifications(
        self,
        style: InteractionStyle,
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Get style-specific response modifications."""
        modifications = []
        
        style_behaviors = self.style_rules.get(style, {}).get("behaviors", [])
        
        if style == InteractionStyle.TEACHER and "explain_concepts" in style_behaviors:
            modifications.append({
                "type": "add_explanation",
                "suggestion": "Add step-by-step explanation",
                "confidence": 0.8
            })
            
        elif style == InteractionStyle.COACH and "motivate" in style_behaviors:
            modifications.append({
                "type": "add_motivation",
                "suggestion": "Include encouraging statement",
                "confidence": 0.7
            })
            
        elif style == InteractionStyle.COMPANION and "share_stories" in style_behaviors:
            modifications.append({
                "type": "add_anecdote",
                "suggestion": "Share relevant anecdote or example",
                "confidence": 0.6
            })
            
        elif style == InteractionStyle.ADVISOR and "analyze_options" in style_behaviors:
            modifications.append({
                "type": "add_analysis",
                "suggestion": "Provide pros and cons analysis",
                "confidence": 0.8
            })
        
        return modifications
    
    async def _load_user_personality_preferences(
        self,
        user_id: str
    ) -> Optional[Dict[PersonalityDimension, float]]:
        """Load user's historical personality preferences."""
        try:
            # Get from preference learner
            preferences = await self.preference_learner.get_preferences(user_id)
            
            if preferences and "personality_traits" in preferences:
                trait_prefs = {}
                for trait_name, value in preferences["personality_traits"].items():
                    try:
                        dimension = PersonalityDimension[trait_name.upper()]
                        trait_prefs[dimension] = float(value)
                    except (KeyError, ValueError):
                        continue
                
                return trait_prefs
                
        except Exception as e:
            logger.warning(f"Failed to load user personality preferences: {str(e)}")
        
        return None
    
    async def _cache_profile(self, profile: PersonalityProfile) -> None:
        """Cache personality profile."""
        cache_key = f"personality:{profile.user_id}:{profile.session_id}"
        await self.cache.set(
            cache_key,
            profile.to_dict(),
            ttl=self.cache_ttl
        )
    
    async def _periodic_mood_update(self) -> None:
        """Periodically update mood based on context."""
        while True:
            try:
                await asyncio.sleep(300)  # Every 5 minutes
                
                for profile_key, profile in self.profiles.items():
                    # Skip inactive profiles
                    if (datetime.now(timezone.utc) - profile.last_active).total_seconds() > 3600:
                        continue
                    
                    # Get current context
                    context = await self.context_manager.get_current_context(
                        profile.session_id
                    )
                    
                    # Determine mood from context
                    new_mood = await self._infer_mood_from_context(context)
                    
                    if new_mood and new_mood != profile.mood_state:
                        await self.set_mood(
                            profile.session_id,
                            new_mood,
                            reason="periodic_context_update"
                        )
                
            except Exception as e:
                logger.error(f"Error in periodic mood update: {str(e)}")
    
    async def _confidence_decay_task(self) -> None:
        """Periodically decay trait confidence."""
        while True:
            try:
                await asyncio.sleep(3600)  # Every hour
                
                for profile in self.profiles.values():
                    for trait in profile.traits.values():
                        # Decay confidence for traits not recently updated
                        time_since_update = (
                            datetime.now(timezone.utc) - trait.last_updated
                        ).total_seconds() / 3600  # Hours
                        
                        if time_since_update > 24:  # More than 24 hours
                            trait.decay_confidence(decay_rate=0.95)
                
            except Exception as e:
                logger.error(f"Error in confidence decay task: {str(e)}")
    
    async def _infer_mood_from_context(
        self,
        context: Dict[str, Any]
    ) -> Optional[MoodState]:
        """Infer mood from current context."""
        # Check emotional context
        if ContextType.EMOTIONAL.name in context:
            emotion_data = context[ContextType.EMOTIONAL.name].get("latest", {})
            emotion = emotion_data.get("overall_emotion")
            
            if emotion == "happy":
                return MoodState.CHEERFUL
            elif emotion == "sad":
                return MoodState.SYMPATHETIC
            elif emotion == "focused":
                return MoodState.FOCUSED
        
        # Check task context
        if ContextType.TASK.name in context:
            task_data = context[ContextType.TASK.name].get("latest", {})
            task_type = task_data.get("task_type")
            
            if task_type == "creative":
                return MoodState.PLAYFUL
            elif task_type == "analytical":
                return MoodState.FOCUSED
        
        # Check temporal context
        if ContextType.TEMPORAL.name in context:
            temporal_data = context[ContextType.TEMPORAL.name].get("latest", {})
            time_of_day = temporal_data.get("time_of_day")
            
            if time_of_day == "morning":
                return MoodState.ENERGETIC
            elif time_of_day == "evening":
                return MoodState.CALM
        
        return None
    
    async def _handle_user_interaction(self, event: Event) -> None:
        """Handle user interaction events."""
        try:
            session_id = event.data.get("session_id")
            if session_id:
                # Update last active timestamp
                for profile in self.profiles.values():
                    if profile.session_id == session_id:
                        profile.last_active = datetime.now(timezone.utc)
                        break
                        
        except Exception as e:
            logger.error(f"Error handling user interaction: {str(e)}")
    
    async def _handle_feedback(self, event: Event) -> None:
        """Handle feedback events."""
        try:
            session_id = event.data.get("session_id")
            feedback = event.data.get("feedback")
            
            if session_id and feedback:
                await self.adapt_personality(
                    session_id,
                    {"source": "feedback_event"},
                    feedback
                )
                
        except Exception as e:
            logger.error(f"Error handling feedback: {str(e)}")
    
    async def _handle_emotion(self, event: Event) -> None:
        """Handle emotion detection events."""
        try:
            session_id = event.data.get("session_id")
            emotion = event.data.get("emotion")
            
            if session_id and emotion:
                # Map emotion to mood
                emotion_mood_map = {
                    "happy": MoodState.CHEERFUL,
                    "sad": MoodState.SYMPATHETIC,
                    "neutral": MoodState.NEUTRAL,
                    "focused": MoodState.FOCUSED,
                    "playful": MoodState.PLAYFUL
                }
                
                if emotion in emotion_mood_map:
                    await self.set_mood(
                        session_id,
                        emotion_mood_map[emotion],
                        reason="emotion_detection"
                    )
                    
        except Exception as e:
            logger.error(f"Error handling emotion event: {str(e)}")
    
    async def _handle_context_update(self, event: Event) -> None:
        """Handle context update events."""
        # Context updates might trigger mood changes
        # Handled by periodic mood update task
        pass
    
    async def _handle_preference(self, event: Event) -> None:
        """Handle preference learning events."""
        try:
            user_id = event.data.get("user_id")
            preferences = event.data.get("preferences", {})
            
            if user_id and "personality_traits" in preferences:
                # Update profiles for this user
                for profile in self.profiles.values():
                    if profile.user_id == user_id:
                        # Adapt traits based on learned preferences
                        for trait_name, target_value in preferences["personality_traits"].items():
                            try:
                                dimension = PersonalityDimension[trait_name.upper()]
                                if dimension in profile.traits:
                                    profile.traits[dimension].update(
                                        float(target_value),
                                        confidence=0.7
                                    )
                            except (KeyError, ValueError):
                                continue
                                
        except Exception as e:
            logger.error(f"Error handling preference event: {str(e)}")
    
    async def export_profile(self, profile_id: str) -> Dict[str, Any]:
        """Export personality profile for persistence."""
        profile = None
        for p in self.profiles.values():
            if p.profile_id == profile_id:
                profile = p
                break
        
        if not profile:
            raise ValueError(f"Profile {profile_id} not found")
        
        return {
            "profile": profile.to_dict(),
            "adaptation_history": [
                {
                    "adaptation_id": adapt.adaptation_id,
                    "timestamp": adapt.timestamp.isoformat(),
                    "trigger": adapt.trigger,
                    "changes": {
                        dim.name: {"from": old, "to": new}
                        for dim, (old, new) in adapt.changes.items()
                    },
                    "confidence": adapt.confidence
                }
                for adapt in self.adaptation_history.get(profile_id, [])
            ]
        }
    
    async def import_profile(
        self,
        user_id: str,
        session_id: str,
        profile_data: Dict[str, Any]
    ) -> PersonalityProfile:
        """Import personality profile from exported data."""
        # Create profile from data
        profile = PersonalityProfile(
            profile_id=profile_data["profile"]["profile_id"],
            user_id=user_id,
            session_id=session_id,
            interaction_style=InteractionStyle[profile_data["profile"]["interaction_style"]],
            mood_state=MoodState[profile_data["profile"]["mood_state"]],
            interaction_count=profile_data["profile"]["interaction_count"],
            adaptation_enabled=profile_data["profile"]["adaptation_enabled"],
            custom_traits=profile_data["profile"]["custom_traits"]
        )
        
        # Import traits
        for trait_name, trait_data in profile_data["profile"]["traits"].items():
            dimension = PersonalityDimension[trait_name]
            profile.traits[dimension] = PersonalityTrait(
                dimension=dimension,
                value=trait_data["value"],
                confidence=trait_data["confidence"]
            )
        
        # Store profile
        profile_key = f"{user_id}:{session_id}"
        self.profiles[profile_key] = profile
        
        return profile
    
    async def get_personality_insights(
        self,
        session_id: str
    ) -> Dict[str, Any]:
        """Get insights about current personality state."""
        profile = None
        for p in self.profiles.values():
            if p.session_id == session_id:
                profile = p
                break
        
        if not profile:
            return {"error": "No profile found"}
        
        # Calculate trait statistics
        trait_values = [t.value for t in profile.traits.values()]
        trait_confidences = [t.confidence for t in profile.traits.values()]
        
        # Find dominant traits
        dominant_traits = sorted(
            profile.traits.items(),
            key=lambda x: x[1].value,
            reverse=True
        )[:3]
        
        # Find recent adaptations
        recent_adaptations = self.adaptation_history.get(profile.profile_id, [])[-5:]
        
        return {
            "profile_id": profile.profile_id,
            "interaction_style": profile.interaction_style.name,
            "mood": profile.mood_state.name,
            "statistics": {
                "mean_trait_value": np.mean(trait_values),
                "trait_variance": np.var(trait_values),
                "mean_confidence": np.mean(trait_confidences),
                "interaction_count": profile.interaction_count
            },
            "dominant_traits": [
                {
                    "dimension": dim.name,
                    "value": trait.value,
                    "confidence": trait.confidence
                }
                for dim, trait in dominant_traits
            ],
            "recent_adaptations": len(recent_adaptations),
            "adaptation_enabled": profile.adaptation_enabled,
            "profile_age_hours": (
                datetime.now(timezone.utc) - profile.created_at
            ).total_seconds() / 3600
        }
