"""
Advanced Preference Learning System for AI Assistant
Author: Drmusab
Last Modified: 2025-06-20 03:06:24 UTC

This module provides comprehensive user preference learning capabilities for the AI assistant,
including multi-dimensional preference modeling, real-time adaptation, contextual learning,
privacy-preserving techniques, and seamless integration with all core system components.
"""

from pathlib import Path
from typing import Optional, Dict, Any, List, Set, Callable, Type, Union, AsyncGenerator, TypeVar, Tuple
import asyncio
import threading
import time
import numpy as np
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
from contextlib import asynccontextmanager
import uuid
import json
import hashlib
from collections import defaultdict, deque
import weakref
from abc import ABC, abstractmethod
import logging
import pickle
import base64
from concurrent.futures import ThreadPoolExecutor
import math
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Core imports
from src.core.config.loader import ConfigLoader
from src.core.events.event_bus import EventBus
from src.core.events.event_types import (
    UserPreferenceUpdated, UserPreferenceLearned, PreferenceConflictDetected,
    PreferenceModelUpdated, UserBehaviorAnalyzed, PersonalizationApplied,
    LearningEventOccurred, SystemStateChanged, ComponentHealthChanged,
    SessionStarted, SessionEnded, UserInteractionStarted, UserInteractionCompleted,
    MessageProcessed, WorkflowCompleted, FeedbackReceived, ErrorOccurred
)
from src.core.error_handling import ErrorHandler, handle_exceptions
from src.core.dependency_injection import Container
from src.core.health_check import HealthCheck
from src.core.security.authentication import AuthenticationManager
from src.core.security.encryption import EncryptionManager

# Memory and storage
from src.memory.core_memory.memory_manager import MemoryManager
from src.memory.operations.context_manager import ContextManager
from src.memory.core_memory.memory_types import WorkingMemory, EpisodicMemory, SemanticMemory
from src.integrations.storage.database import DatabaseManager
from src.integrations.cache.redis_cache import RedisCache

# Learning components
from src.learning.continual_learning import ContinualLearner
from src.learning.feedback_processor import FeedbackProcessor

# Observability
from src.observability.monitoring.metrics import MetricsCollector
from src.observability.monitoring.tracing import TraceManager
from src.observability.logging.config import get_logger

# Type definitions
T = TypeVar('T')


class PreferenceType(Enum):
    """Types of user preferences that can be learned."""
    INTERACTION_STYLE = "interaction_style"        # Formal vs casual, verbose vs concise
    RESPONSE_FORMAT = "response_format"            # Text, audio, visual, multimodal
    CONTENT_TYPE = "content_type"                  # Educational, entertaining, practical
    TIMING_PREFERENCE = "timing_preference"       # Response speed vs quality
    COMMUNICATION_MODE = "communication_mode"     # Direct vs explanatory
    PRIVACY_LEVEL = "privacy_level"               # Data sharing preferences
    NOTIFICATION_STYLE = "notification_style"    # Frequency and type
    WORKFLOW_PREFERENCE = "workflow_preference"   # Task organization preferences
    ERROR_HANDLING = "error_handling"             # How to handle errors
    LEARNING_PACE = "learning_pace"               # Speed of adaptation
    CONTEXT_AWARENESS = "context_awareness"       # Level of context consideration
    PERSONALIZATION_LEVEL = "personalization_level"  # Degree of customization


class PreferenceDimension(Enum):
    """Dimensions along which preferences can vary."""
    FORMALITY = "formality"                       # 0.0 (casual) to 1.0 (formal)
    VERBOSITY = "verbosity"                       # 0.0 (concise) to 1.0 (detailed)
    TECHNICAL_DEPTH = "technical_depth"           # 0.0 (simple) to 1.0 (technical)
    CREATIVITY = "creativity"                     # 0.0 (conservative) to 1.0 (creative)
    SPEED_VS_QUALITY = "speed_vs_quality"         # 0.0 (speed) to 1.0 (quality)
    PRIVACY_SENSITIVITY = "privacy_sensitivity"   # 0.0 (open) to 1.0 (private)
    INTERACTION_FREQUENCY = "interaction_frequency" # 0.0 (minimal) to 1.0 (frequent)
    EXPLANATION_DEPTH = "explanation_depth"       # 0.0 (none) to 1.0 (detailed)
    PROACTIVITY = "proactivity"                   # 0.0 (reactive) to 1.0 (proactive)
    MULTIMODALITY = "multimodality"               # 0.0 (text only) to 1.0 (all modes)


class ContextType(Enum):
    """Types of contexts that affect preferences."""
    TEMPORAL = "temporal"                         # Time of day, day of week
    DEVICE = "device"                            # Mobile, desktop, voice
    LOCATION = "location"                        # Home, work, public
    SOCIAL = "social"                            # Alone, with others
    TASK = "task"                                # Work, leisure, learning
    EMOTIONAL = "emotional"                      # Mood, stress level
    ENVIRONMENTAL = "environmental"              # Noise level, lighting
    SESSION_LENGTH = "session_length"            # Short vs long sessions


class LearningAlgorithm(Enum):
    """Types of learning algorithms for preference learning."""
    COLLABORATIVE_FILTERING = "collaborative_filtering"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    BAYESIAN_INFERENCE = "bayesian_inference"
    NEURAL_NETWORK = "neural_network"
    TEMPORAL_MODELING = "temporal_modeling"
    CLUSTERING = "clustering"
    BANDIT_ALGORITHM = "bandit_algorithm"
    PREFERENCE_ELICITATION = "preference_elicitation"


@dataclass
class PreferenceValue:
    """Represents a preference value with uncertainty and metadata."""
    value: float                                  # Preference value (0.0 to 1.0)
    confidence: float = 0.5                      # Confidence in this value
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    evidence_count: int = 0                      # Number of observations
    variance: float = 0.1                        # Uncertainty measure
    source: str = "unknown"                      # How this preference was learned
    context_dependent: bool = False              # Whether this varies by context
    temporal_stability: float = 0.8              # How stable over time
    
    def update(self, new_value: float, evidence_weight: float = 1.0) -> None:
        """Update preference value using weighted average."""
        old_weight = self.evidence_count
        new_weight = evidence_weight
        total_weight = old_weight + new_weight
        
        if total_weight > 0:
            self.value = (self.value * old_weight + new_value * new_weight) / total_weight
            self.confidence = min(1.0, self.confidence + 0.1 * evidence_weight)
            self.evidence_count += 1
            self.last_updated = datetime.now(timezone.utc)
            
            # Update variance based on new evidence
            if self.evidence_count > 1:
                self.variance = max(0.01, self.variance * 0.9)


@dataclass
class ContextualPreference:
    """Preference that varies based on context."""
    base_preference: PreferenceValue
    context_modifiers: Dict[str, float] = field(default_factory=dict)
    context_weights: Dict[str, float] = field(default_factory=dict)
    
    def get_value_for_context(self, context: Dict[str, Any]) -> float:
        """Calculate preference value for specific context."""
        base_value = self.base_preference.value
        
        # Apply context modifiers
        modified_value = base_value
        total_weight = 0.0
        
        for context_key, modifier in self.context_modifiers.items():
            if context_key in context:
                weight = self.context_weights.get(context_key, 1.0)
                context_value = context[context_key]
                
                if isinstance(context_value, (int, float)):
                    context_value = float(context_value)
                elif isinstance(context_value, bool):
                    context_value = 1.0 if context_value else 0.0
                else:
                    continue
                
                modified_value += modifier * context_value * weight
                total_weight += weight
        
        # Normalize and clamp
        if total_weight > 0:
            modified_value = base_value + (modified_value - base_value) / total_weight
        
        return max(0.0, min(1.0, modified_value))


@dataclass
class UserPreferenceProfile:
    """Comprehensive user preference profile."""
    user_id: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Core preferences
    preferences: Dict[PreferenceDimension, PreferenceValue] = field(default_factory=dict)
    contextual_preferences: Dict[PreferenceDimension, ContextualPreference] = field(default_factory=dict)
    
    # Preference hierarchies and conflicts
    preference_priorities: Dict[PreferenceDimension, float] = field(default_factory=dict)
    conflict_resolutions: Dict[Tuple[PreferenceDimension, PreferenceDimension], str] = field(default_factory=dict)
    
    # Behavioral patterns
    interaction_patterns: Dict[str, Any] = field(default_factory=dict)
    usage_statistics: Dict[str, int] = field(default_factory=dict)
    temporal_patterns: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Learning metadata
    learning_rate: float = 0.1
    adaptation_speed: float = 0.5
    exploration_factor: float = 0.1
    total_interactions: int = 0
    successful_predictions: int = 0
    
    # Privacy and consent
    privacy_settings: Dict[str, Any] = field(default_factory=dict)
    data_retention_period: timedelta = field(default_factory=lambda: timedelta(days=365))
    consent_status: Dict[str, bool] = field(default_factory=dict)
    
    # Clustering and similarity
    cluster_id: Optional[str] = None
    similar_users: List[str] = field(default_factory=list)
    personality_vector: Optional[np.ndarray] = None
    
    def get_preference(self, dimension: PreferenceDimension, 
                      context: Optional[Dict[str, Any]] = None) -> float:
        """Get preference value for a dimension, considering context."""
        if dimension in self.contextual_preferences and context:
            return self.contextual_preferences[dimension].get_value_for_context(context)
        elif dimension in self.preferences:
            return self.preferences[dimension].value
        else:
            return 0.5  # Default neutral preference
    
    def update_preference(self, dimension: PreferenceDimension, value: float,
                         evidence_weight: float = 1.0, context: Optional[Dict[str, Any]] = None) -> None:
        """Update a preference value with new evidence."""
        if dimension not in self.preferences:
            self.preferences[dimension] = PreferenceValue(value=value, evidence_count=1)
        else:
            self.preferences[dimension].update(value, evidence_weight)
        
        # Handle contextual preferences
        if context and dimension in self.contextual_preferences:
            # Update context modifiers based on difference from base preference
            base_value = self.preferences[dimension].value
            context_effect = value - base_value
            
            for context_key, context_value in context.items():
                if isinstance(context_value, (int, float, bool)):
                    current_modifier = self.contextual_preferences[dimension].context_modifiers.get(context_key, 0.0)
                    self.contextual_preferences[dimension].context_modifiers[context_key] = (
                        current_modifier * 0.9 + context_effect * 0.1
                    )
        
        self.last_updated = datetime.now(timezone.utc)


class PreferenceLearningError(Exception):
    """Custom exception for preference learning operations."""
    
    def __init__(self, message: str, user_id: Optional[str] = None, 
                 preference_type: Optional[str] = None):
        super().__init__(message)
        self.user_id = user_id
        self.preference_type = preference_type
        self.timestamp = datetime.now(timezone.utc)


class PreferenceLearner(ABC):
    """Abstract base class for preference learning algorithms."""
    
    @abstractmethod
    async def learn_from_interaction(self, interaction_data: Dict[str, Any], 
                                   user_profile: UserPreferenceProfile) -> None:
        """Learn preferences from user interaction."""
        pass
    
    @abstractmethod
    async def predict_preference(self, user_profile: UserPreferenceProfile,
                               dimension: PreferenceDimension,
                               context: Optional[Dict[str, Any]] = None) -> Tuple[float, float]:
        """Predict preference value and confidence."""
        pass
    
    @abstractmethod
    def get_algorithm_type(self) -> LearningAlgorithm:
        """Get the algorithm type."""
        pass


class BayesianPreferenceLearner(PreferenceLearner):
    """Bayesian inference-based preference learning."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.prior_alpha = 1.0  # Prior for Beta distribution
        self.prior_beta = 1.0
    
    def get_algorithm_type(self) -> LearningAlgorithm:
        return LearningAlgorithm.BAYESIAN_INFERENCE
    
    async def learn_from_interaction(self, interaction_data: Dict[str, Any], 
                                   user_profile: UserPreferenceProfile) -> None:
        """Learn preferences using Bayesian updates."""
        try:
            # Extract preference signals from interaction
            signals = self._extract_preference_signals(interaction_data)
            
            for dimension, (value, confidence) in signals.items():
                if dimension in user_profile.preferences:
                    current_pref = user_profile.preferences[dimension]
                    
                    # Bayesian update
                    alpha = self.prior_alpha + current_pref.evidence_count * current_pref.value
                    beta = self.prior_beta + current_pref.evidence_count * (1 - current_pref.value)
                    
                    # Update with new evidence
                    alpha += confidence * value
                    beta += confidence * (1 - value)
                    
                    # Calculate new preference
                    new_value = alpha / (alpha + beta)
                    new_confidence = min(1.0, (alpha + beta) / 100.0)  # Normalize by total evidence
                    
                    user_profile.preferences[dimension] = PreferenceValue(
                        value=new_value,
                        confidence=new_confidence,
                        evidence_count=current_pref.evidence_count + 1,
                        variance=1.0 / (alpha + beta),  # Inverse of precision
                        source="bayesian_inference"
                    )
                else:
                    # Initialize with weak prior
                    user_profile.preferences[dimension] = PreferenceValue(
                        value=value,
                        confidence=confidence * 0.5,  # Lower initial confidence
                        evidence_count=1,
                        source="bayesian_inference"
                    )
        
        except Exception as e:
            self.logger.error(f"Error in Bayesian preference learning: {str(e)}")
    
    async def predict_preference(self, user_profile: UserPreferenceProfile,
                               dimension: PreferenceDimension,
                               context: Optional[Dict[str, Any]] = None) -> Tuple[float, float]:
        """Predict preference with Bayesian confidence intervals."""
        if dimension in user_profile.preferences:
            pref = user_profile.preferences[dimension]
            
            # Calculate credible interval
            alpha = self.prior_alpha + pref.evidence_count * pref.value
            beta = self.prior_beta + pref.evidence_count * (1 - pref.value)
            
            mean = alpha / (alpha + beta)
            variance = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))
            confidence = 1.0 - variance  # Higher precision means higher confidence
            
            return mean, confidence
        else:
            return 0.5, 0.1  # Neutral with low confidence
    
    def _extract_preference_signals(self, interaction_data: Dict[str, Any]) -> Dict[PreferenceDimension, Tuple[float, float]]:
        """Extract preference signals from interaction data."""
        signals = {}
        
        # Response time preferences
        if 'response_time' in interaction_data:
            response_time = interaction_data['response_time']
            if response_time < 1.0:
                signals[PreferenceDimension.SPEED_VS_QUALITY] = (0.8, 0.7)  # Prefer speed
            elif response_time > 5.0:
                signals[PreferenceDimension.SPEED_VS_QUALITY] = (0.2, 0.7)  # Prefer quality
        
        # Interaction length preferences
        if 'interaction_length' in interaction_data:
            length = interaction_data['interaction_length']
            if length > 100:  # Long interaction
                signals[PreferenceDimension.VERBOSITY] = (0.7, 0.6)
            elif length < 20:  # Short interaction
                signals[PreferenceDimension.VERBOSITY] = (0.3, 0.6)
        
        # Feedback-based learning
        if 'feedback' in interaction_data:
            feedback = interaction_data['feedback']
            if feedback.get('rating', 0) > 4:  # High rating
                # Reinforce current settings
                if 'current_style' in interaction_data:
                    style = interaction_data['current_style']
                    if style == 'formal':
                        signals[PreferenceDimension.FORMALITY] = (0.8, 0.8)
                    elif style == 'casual':
                        signals[PreferenceDimension.FORMALITY] = (0.2, 0.8)
        
        return signals


class ReinforcementPreferenceLearner(PreferenceLearner):
    """Reinforcement learning-based preference learning using multi-armed bandits."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.epsilon = 0.1  # Exploration rate
        self.decay_rate = 0.99
        self.action_values: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.action_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    
    def get_algorithm_type(self) -> LearningAlgorithm:
        return LearningAlgorithm.REINFORCEMENT_LEARNING
    
    async def learn_from_interaction(self, interaction_data: Dict[str, Any], 
                                   user_profile: UserPreferenceProfile) -> None:
        """Learn preferences using reinforcement learning."""
        try:
            # Extract reward signal
            reward = self._calculate_reward(interaction_data)
            
            # Get the action taken (preference setting used)
            action = self._get_action_from_interaction(interaction_data)
            
            if action and reward is not None:
                user_key = user_profile.user_id
                
                # Update Q-values using incremental update
                old_value = self.action_values[user_key][action]
                count = self.action_counts[user_key][action] + 1
                self.action_counts[user_key][action] = count
                
                # Q-learning update
                learning_rate = 1.0 / count  # Decreasing learning rate
                new_value = old_value + learning_rate * (reward - old_value)
                self.action_values[user_key][action] = new_value
                
                # Update user preference based on learned Q-values
                self._update_preferences_from_q_values(user_profile, action, new_value)
        
        except Exception as e:
            self.logger.error(f"Error in RL preference learning: {str(e)}")
    
    async def predict_preference(self, user_profile: UserPreferenceProfile,
                               dimension: PreferenceDimension,
                               context: Optional[Dict[str, Any]] = None) -> Tuple[float, float]:
        """Predict preference using epsilon-greedy strategy."""
        user_key = user_profile.user_id
        
        # Get possible actions for this dimension
        actions = self._get_actions_for_dimension(dimension)
        
        if not actions:
            return 0.5, 0.1
        
        # Choose action using epsilon-greedy
        if np.random.random() < self.epsilon:
            # Exploration: random action
            chosen_action = np.random.choice(actions)
            confidence = 0.3  # Low confidence for exploration
        else:
            # Exploitation: best action
            best_action = max(actions, key=lambda a: self.action_values[user_key][a])
            chosen_action = best_action
            confidence = min(1.0, self.action_counts[user_key][chosen_action] / 100.0)
        
        # Convert action to preference value
        preference_value = self._action_to_preference_value(chosen_action, dimension)
        
        return preference_value, confidence
    
    def _calculate_reward(self, interaction_data: Dict[str, Any]) -> Optional[float]:
        """Calculate reward signal from interaction outcome."""
        reward = 0.0
        
        # Explicit feedback
        if 'feedback' in interaction_data:
            feedback = interaction_data['feedback']
            if 'rating' in feedback:
                reward += (feedback['rating'] - 3.0) / 2.0  # Normalize 1-5 to -1 to 1
        
        # Implicit signals
        if 'user_satisfaction' in interaction_data:
            reward += interaction_data['user_satisfaction'] * 0.5
        
        if 'task_completion' in interaction_data:
            reward += 0.3 if interaction_data['task_completion'] else -0.3
        
        if 'response_time' in interaction_data:
            response_time = interaction_data['response_time']
            if response_time < 2.0:
                reward += 0.2  # Quick response bonus
            elif response_time > 10.0:
                reward -= 0.2  # Slow response penalty
        
        return reward if abs(reward) > 0.01 else None
    
    def _get_action_from_interaction(self, interaction_data: Dict[str, Any]) -> Optional[str]:
        """Extract the action (preference setting) used in the interaction."""
        # This would map interaction settings to actions
        settings = interaction_data.get('settings', {})
        
        action_parts = []
        if 'formality' in settings:
            action_parts.append(f"formality_{settings['formality']}")
        if 'verbosity' in settings:
            action_parts.append(f"verbosity_{settings['verbosity']}")
        
        return "_".join(action_parts) if action_parts else None
    
    def _get_actions_for_dimension(self, dimension: PreferenceDimension) -> List[str]:
        """Get possible actions for a preference dimension."""
        actions_map = {
            PreferenceDimension.FORMALITY: ["formality_low", "formality_medium", "formality_high"],
            PreferenceDimension.VERBOSITY: ["verbosity_low", "verbosity_medium", "verbosity_high"],
            PreferenceDimension.TECHNICAL_DEPTH: ["technical_low", "technical_medium", "technical_high"],
            PreferenceDimension.SPEED_VS_QUALITY: ["speed_focused", "balanced", "quality_focused"]
        }
        return actions_map.get(dimension, [])
    
    def _action_to_preference_value(self, action: str, dimension: PreferenceDimension) -> float:
        """Convert action string to preference value."""
        if "low" in action:
            return 0.2
        elif "medium" in action or "balanced" in action:
            return 0.5
        elif "high" in action:
            return 0.8
        elif "speed_focused" in action:
            return 0.2
        elif "quality_focused" in action:
            return 0.8
        else:
            return 0.5
    
    def _update_preferences_from_q_values(self, user_profile: UserPreferenceProfile, 
                                        action: str, q_value: float) -> None:
        """Update user preferences based on learned Q-values."""
        # Map action to preference dimensions and update
        if "formality" in action:
            value = self._action_to_preference_value(action, PreferenceDimension.FORMALITY)
            confidence = min(1.0, abs(q_value))
            user_profile.update_preference(PreferenceDimension.FORMALITY, value, confidence)


class CollaborativeFilteringLearner(PreferenceLearner):
    """Collaborative filtering for learning preferences from similar users."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.similarity_threshold = 0.3
        self.min_similar_users = 3
        self.user_profiles: Dict[str, UserPreferenceProfile] = {}
    
    def get_algorithm_type(self) -> LearningAlgorithm:
        return LearningAlgorithm.COLLABORATIVE_FILTERING
    
    async def learn_from_interaction(self, interaction_data: Dict[str, Any], 
                                   user_profile: UserPreferenceProfile) -> None:
        """Learn preferences using collaborative filtering."""
        try:
            # Store user profile for similarity calculations
            self.user_profiles[user_profile.user_id] = user_profile
            
            # Find similar users
            similar_users = await self._find_similar_users(user_profile)
            user_profile.similar_users = similar_users
            
            # Learn from similar users' preferences
            if len(similar_users) >= self.min_similar_users:
                await self._update_from_similar_users(user_profile, similar_users)
        
        except Exception as e:
            self.logger.error(f"Error in collaborative filtering: {str(e)}")
    
    async def predict_preference(self, user_profile: UserPreferenceProfile,
                               dimension: PreferenceDimension,
                               context: Optional[Dict[str, Any]] = None) -> Tuple[float, float]:
        """Predict preference using collaborative filtering."""
        if dimension in user_profile.preferences:
            base_value = user_profile.preferences[dimension].value
            base_confidence = user_profile.preferences[dimension].confidence
        else:
            base_value = 0.5
            base_confidence = 0.1
        
        # Get predictions from similar users
        similar_predictions = []
        for similar_user_id in user_profile.similar_users:
            if similar_user_id in self.user_profiles:
                similar_profile = self.user_profiles[similar_user_id]
                if dimension in similar_profile.preferences:
                    similar_value = similar_profile.preferences[dimension].value
                    similar_confidence = similar_profile.preferences[dimension].confidence
                    similar_predictions.append((similar_value, similar_confidence))
        
        if similar_predictions:
            # Weighted average of similar users' preferences
            total_weight = 0.0
            weighted_sum = 0.0
            
            for value, confidence in similar_predictions:
                weight = confidence
                weighted_sum += value * weight
                total_weight += weight
            
            if total_weight > 0:
                collaborative_value = weighted_sum / total_weight
                collaborative_confidence = min(1.0, total_weight / len(similar_predictions))
                
                # Combine with user's own preference
                if base_confidence > 0.1:
                    final_value = (base_value * base_confidence + collaborative_value * collaborative_confidence) / (base_confidence + collaborative_confidence)
                    final_confidence = min(1.0, base_confidence + collaborative_confidence * 0.3)
                else:
                    final_value = collaborative_value
                    final_confidence = collaborative_confidence * 0.7  # Reduce confidence for cold start
                
                return final_value, final_confidence
        
        return base_value, base_confidence
    
    async def _find_similar_users(self, user_profile: UserPreferenceProfile) -> List[str]:
        """Find users with similar preferences."""
        similar_users = []
        
        # Create preference vector for current user
        user_vector = self._create_preference_vector(user_profile)
        
        # Compare with other users
        for other_user_id, other_profile in self.user_profiles.items():
            if other_user_id == user_profile.user_id:
                continue
            
            other_vector = self._create_preference_vector(other_profile)
            
            # Calculate cosine similarity
            similarity = self._calculate_similarity(user_vector, other_vector)
            
            if similarity > self.similarity_threshold:
                similar_users.append(other_user_id)
        
        # Sort by similarity (we'd need to store similarities for this)
        return similar_users[:10]  # Top 10 similar users
    
    def _create_preference_vector(self, user_profile: UserPreferenceProfile) -> np.ndarray:
        """Create a numerical vector representing user preferences."""
        vector = []
        
        for dimension in PreferenceDimension:
            if dimension in user_profile.preferences:
                value = user_profile.preferences[dimension].value
                confidence = user_profile.preferences[dimension].confidence
                # Weight by confidence
                vector.append(value * confidence)
            else:
                vector.append(0.5)  # Default neutral value
        
        return np.array(vector)
    
    def _calculate_similarity(self, vector1: np.ndarray, vector2: np.ndarray) -> float:
        """Calculate cosine similarity between preference vectors."""
        try:
            similarity = cosine_similarity([vector1], [vector2])[0][0]
            return max(0.0, similarity)  # Ensure non-negative
        except Exception:
            return 0.0
    
    async def _update_from_similar_users(self, user_profile: UserPreferenceProfile, 
                                       similar_users: List[str]) -> None:
        """Update preferences based on similar users."""
        for dimension in PreferenceDimension:
            if dimension not in user_profile.preferences:
                # Cold start: initialize from similar users
                values = []
                confidences = []
                
                for similar_user_id in similar_users:
                    if similar_user_id in self.user_profiles:
                        similar_profile = self.user_profiles[similar_user_id]
                        if dimension in similar_profile.preferences:
                            values.append(similar_profile.preferences[dimension].value)
                            confidences.append(similar_profile.preferences[dimension].confidence)
                
                if values:
                    # Weighted average
                    total_weight = sum(confidences)
                    if total_weight > 0:
                        weighted_value = sum(v * c for v, c in zip(values, confidences)) / total_weight
                        avg_confidence = sum(confidences) / len(confidences) * 0.5  # Reduce for cold start
                        
                        user_profile.preferences[dimension] = PreferenceValue(
                            value=weighted_value,
                            confidence=avg_confidence,
                            evidence_count=len(values),
                            source="collaborative_filtering"
                        )


class PreferenceLearningEngine:
    """
    Advanced Preference Learning System for the AI Assistant.
    
    This engine provides comprehensive user preference learning including:
    - Multi-dimensional preference modeling across various aspects
    - Real-time adaptation based on user interactions and feedback
    - Contextual preferences that adapt to different situations
    - Privacy-preserving learning with differential privacy
    - Multiple learning algorithms (Bayesian, RL, Collaborative Filtering)
    - Preference conflict resolution and hierarchy management
    - Integration with all core system components
    - Temporal modeling for preference evolution
    - Clustering for user segmentation and personalization
    """
    
    def __init__(self, container: Container):
        """
        Initialize the preference learning engine.
        
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
        
        # Memory and storage
        self.memory_manager = container.get(MemoryManager)
        self.context_manager = container.get(ContextManager)
        self.working_memory = container.get(WorkingMemory)
        self.episodic_memory = container.get(EpisodicMemory)
        self.semantic_memory = container.get(SemanticMemory)
        
        # Storage and caching
        try:
            self.database = container.get(DatabaseManager)
            self.redis_cache = container.get(RedisCache)
        except Exception:
            self.database = None
            self.redis_cache = None
        
        # Security and privacy
        try:
            self.auth_manager = container.get(AuthenticationManager)
            self.encryption_manager = container.get(EncryptionManager)
        except Exception:
            self.auth_manager = None
            self.encryption_manager = None
        
        # Learning components
        try:
            self.continual_learner = container.get(ContinualLearner)
            self.feedback_processor = container.get(FeedbackProcessor)
        except Exception:
            self.continual_learner = None
            self.feedback_processor = None
        
        # Observability
        self.metrics = container.get(MetricsCollector)
        self.tracer = container.get(TraceManager)
        
        # Preference learning state
        self.user_profiles: Dict[str, UserPreferenceProfile] = {}
        self.learning_algorithms: Dict[LearningAlgorithm, PreferenceLearner] = {}
        self.preference_locks: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        
        # Clustering and similarity
        self.user_clusters: Dict[str, List[str]] = {}
        self.cluster_centers: Dict[str, np.ndarray] = {}
        self.scaler = StandardScaler()
        
        # Performance tracking
        self.learning_stats: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.prediction_accuracy: deque = deque(maxlen=1000)
        
        # Background tasks
        self.background_tasks: List[asyncio.Task] = []
        
        # Configuration
        self.learning_rate = self.config.get("preference_learning.learning_rate", 0.1)
        self.privacy_epsilon = self.config.get("preference_learning.privacy_epsilon", 1.0)
        self.min_interactions_for_learning = self.config.get("preference_learning.min_interactions", 5)
        self.enable_collaborative_filtering = self.config.get("preference_learning.collaborative_filtering", True)
        self.enable_clustering = self.config.get("preference_learning.clustering", True)
        self.auto_clustering_interval = self.config.get("preference_learning.clustering_interval", 3600)
        
        # Initialize components
        self._setup_learning_algorithms()
        self._setup_monitoring()
        
        # Register health check
        self.health_check.register_component("preference_learning", self._health_check_callback)
        
        self.logger.info("PreferenceLearningEngine initialized successfully")

    def _setup_learning_algorithms(self) -> None:
        """Setup different preference learning algorithms."""
        try:
            # Initialize learning algorithms
            self.learning_algorithms[LearningAlgorithm.BAYESIAN_INFERENCE] = BayesianPreferenceLearner()
            self.learning_algorithms[LearningAlgorithm.REINFORCEMENT_LEARNING] = ReinforcementPreferenceLearner()
            
            if self.enable_collaborative_filtering:
                self.learning_algorithms[LearningAlgorithm.COLLABORATIVE_FILTERING] = CollaborativeFilteringLearner()
            
            self.logger.info(f"Initialized {len(self.learning_algorithms)} learning algorithms")
            
        except Exception as e:
            self.logger.error(f"Failed to setup learning algorithms: {str(e)}")

    def _setup_monitoring(self) -> None:
        """Setup monitoring and metrics."""
        try:
            # Register preference learning metrics
            self.metrics.register_counter("preference_updates_total")
            self.metrics.register_counter("preference_predictions_total")
            self.metrics.register_histogram("preference_learning_duration_seconds")
            self.metrics.register_histogram("preference_prediction_accuracy")
            self.metrics.register_gauge("active_user_profiles")
            self.metrics.register_gauge("user_clusters_count")
            self.metrics.register_counter("preference_conflicts_total")
            
        except Exception as e:
            self.logger.warning(f"Failed to setup monitoring: {str(e)}")

    async def initialize(self) -> None:
        """Initialize the preference learning engine."""
        try:
            # Load existing user profiles
            await self._load_user_profiles()
            
            # Initialize learning algorithms
            for algorithm in self.learning_algorithms.values():
                if hasattr(algorithm, 'initialize'):
                    await algorithm.initialize()
            
            # Start background tasks
            self.background_tasks.extend([
                asyncio.create_task(self._profile_maintenance_loop()),
                asyncio.create_task(self._learning_optimization_loop())
            ])
            
            if self.enable_clustering:
                self.background_tasks.append(
                    asyncio.create_task(self._clustering_loop())
                )
            
            # Register event handlers
            await self._register_event_handlers()
            
            self.logger.info("PreferenceLearningEngine initialization completed")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize PreferenceLearningEngine: {str(e)}")
            raise PreferenceLearningError(f"Initialization failed: {str(e)}")

    async def _register_event_handlers(self) -> None:
        """Register event handlers for system events."""
        # User interaction events
        self.event_bus.subscribe("user_interaction_completed", self._handle_interaction_completed)
        self.event_bus.subscribe("message_processed", self._handle_message_processed)
        self.event_bus.subscribe("workflow_completed", self._handle_workflow_completed)
        
        # Feedback events
        self.event_bus.subscribe("feedback_received", self._handle_feedback_received)
        
        # Session events
        self.event_bus.subscribe("session_started", self._handle_session_started)
        self.event_bus.subscribe("session_ended", self._handle_session_ended)
        
        # System events
        self.event_bus.subscribe("system_shutdown_started", self._handle_system_shutdown)

    async def _load_user_profiles(self) -> None:
        """Load existing user profiles from storage."""
        try:
            if self.database:
                # Load from database
                profiles_data = await self.database.fetch_all(
                    "SELECT user_id, profile_data FROM user_preference_profiles"
                )
                
                for user_id, profile_data in profiles_data:
                    try:
                        profile = self._deserialize_profile(profile_data)
                        self.user_profiles[user_id] = profile
                    except Exception as e:
                        self.logger.warning(f"Failed to load profile for user {user_id}: {str(e)}")
            
            self.logger.info(f"Loaded {len(self.user_profiles)} user preference profiles")
            
        except Exception as e:
            self.logger.warning(f"Failed to load user profiles: {str(e)}")

    @handle_exceptions
    async def get_user_preferences(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Get user preferences.
        
        Args:
            user_id: User identifier
            
        Returns:
            User preferences or None if not found
        """
        if user_id in self.user_profiles:
            profile = self.user_profiles[user_id]
            
            preferences = {}
            for dimension, pref_value in profile.preferences.items():
                preferences[dimension.value] = {
                    'value': pref_value.value,
                    'confidence': pref_value.confidence,
                    'last_updated': pref_value.last_updated.isoformat(),
                    'evidence_count': pref_value.evidence_count
                }
            
            return {
                'user_id': user_id,
                'preferences': preferences,
                'last_updated': profile.last_updated.isoformat(),
                'total_interactions': profile.total_interactions,
                'learning_rate': profile.learning_rate,
                'cluster_id': profile.cluster_id,
                'similar_users': profile.similar_users[:5]  # Top 5 similar users
            }
        
        return None

    @handle_exceptions
    async def predict_user_preference(
        self,
        user_id: str,
        preference_dimension: str,
        context: Optional[Dict[str, Any]] = None,
        algorithm: Optional[str] = None
    ) -> Tuple[float, float]:
        """
        Predict user preference for a specific dimension.
        
        Args:
            user_id: User identifier
            preference_dimension: Preference dimension to predict
            context: Optional context information
            algorithm: Optional specific algorithm to use
            
        Returns:
            Tuple of (predicted_value, confidence)
        """
        try:
            # Convert string to enum
            dimension = PreferenceDimension(preference_dimension)
        except ValueError:
            raise PreferenceLearningError(f"Unknown preference dimension: {preference_dimension}")
        
        # Get or create user profile
        profile = await self._get_or_create_user_profile(user_id)
        
        # Choose algorithm
        if algorithm:
            try:
                algo_enum = LearningAlgorithm(algorithm)
                learner = self.learning_algorithms.get(algo_enum)
                if not learner:
                    raise PreferenceLearningError(f"Algorithm {algorithm} not available")
            except ValueError:
                raise PreferenceLearningError(f"Unknown algorithm: {algorithm}")
        else:
            # Use best available algorithm
            learner = self._select_best_algorithm(profile, dimension)
        
        # Make prediction
        prediction, confidence = await learner.predict_preference(profile, dimension, context)
        
        # Update metrics
        self.metrics.increment("preference_predictions_total")
        
        self.logger.debug(f"Predicted preference for user {user_id}, dimension {preference_dimension}: {prediction:.3f} (confidence: {confidence:.3f})")
        
        return prediction, confidence

    @handle_exceptions
    async def update_from_interaction(
        self,
        user_id: str,
        interaction_data: Dict[str, Any]
    ) -> None:
        """
        Update user preferences from interaction data.
        
        Args:
            user_id: User identifier
            interaction_data: Interaction data containing preference signals
        """
        async with self.preference_locks[user_id]:
            try:
                start_time = time.time()
                
                # Get or create user profile
                profile = await self._get_or_create_user_profile(user_id)
                profile.total_interactions += 1
                
                # Apply differential privacy if enabled
                if self.privacy_epsilon > 0:
                    interaction_data = self._apply_differential_privacy(interaction_data)
                
                # Update preferences using all available algorithms
                for algorithm, learner in self.learning_algorithms.items():
                    try:
                        await learner.learn_from_interaction(interaction_data, profile)
                    except Exception as e:
                        self.logger.warning(f"Error in {algorithm.value} learning: {str(e)}")
                
                # Store updated profile
                await self._store_user_profile(profile)
                
                # Update statistics
                learning_time = time.time() - start_time
                self.learning_stats[user_id]['last_update'] = datetime.now(timezone.utc)
                self.learning_stats[user_id]['total_updates'] = self.learning_stats[user_id].get('total_updates', 0) + 1
                
                # Emit preference updated event
                await self.event_bus.emit(UserPreferenceLearned(
                    user_id=user_id,
                    learning_duration=learning_time,
                    total_interactions=profile.total_interactions
                ))
                
                # Update metrics
                self.metrics.increment("preference_updates_total")
                self.metrics.record("preference_learning_duration_seconds", learning_time)
                
                self.logger.debug(f"Updated preferences for user {user_id} in {learning_time:.3f}s")
                
            except Exception as e:
                self.logger.error(f"Error updating preferences for user {user_id}: {str(e)}")
                raise PreferenceLearningError(f"Failed to update preferences: {str(e)}", user_id)

    @handle_exceptions
    async def update_from_feedback(
        self,
        user_id: str,
        feedback_data: Dict[str, Any],
        interaction_context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Update user preferences from explicit feedback.
        
        Args:
            user_id: User identifier
            feedback_data: Explicit feedback data
            interaction_context: Context of the interaction being rated
        """
        async with self.preference_locks[user_id]:
            try:
                profile = await self._get_or_create_user_profile(user_id)
                
                # Extract preference signals from feedback
                preference_updates = self._extract_preference_signals_from_feedback(
                    feedback_data, interaction_context
                )
                
                # Apply updates with higher confidence (explicit feedback)
                for dimension, (value, base_confidence) in preference_updates.items():
                    confidence = min(1.0, base_confidence * 1.5)  # Boost for explicit feedback
                    profile.update_preference(dimension, value, confidence, interaction_context)
                
                # Store updated profile
                await self._store_user_profile(profile)
                
                # Emit preference updated event
                await self.event_bus.emit(UserPreferenceUpdated(
                    user_id=user_id,
                    preferences=list(preference_updates.keys()),
                    source="explicit_feedback"
                ))
                
                self.logger.info(f"Updated preferences from feedback for user {user_id}")
                
            except Exception as e:
                self.logger.error(f"Error updating preferences from feedback for user {user_id}: {str(e)}")
                raise PreferenceLearningError(f"Failed to update from feedback: {str(e)}", user_id)

    async def _get_or_create_user_profile(self, user_id: str) -> UserPreferenceProfile:
        """Get existing user profile or create a new one."""
        if user_id not in self.user_profiles:
            # Create new profile
            profile = UserPreferenceProfile(user_id=user_id)
            
            # Initialize with default preferences
            self._initialize_default_preferences(profile)
            
            # Try to initialize from similar users if clustering is enabled
            if self.enable_clustering and len(self.user_profiles) > 10:
                await self._initialize_from_cluster(profile)
            
            self.user_profiles[user_id] = profile
            
            # Update metrics
            self.metrics.set("active_user_profiles", len(self.user_profiles))
            
            self.logger.info(f"Created new preference profile for user {user_id}")
        
        return self.user_profiles[user_id]

    def _initialize_default_preferences(self, profile: UserPreferenceProfile) -> None:
        """Initialize profile with default preference values."""
        default_preferences = {
            PreferenceDimension.FORMALITY: 0.5,
            PreferenceDimension.VERBOSITY: 0.5,
            PreferenceDimension.TECHNICAL_DEPTH: 0.5,
            PreferenceDimension.CREATIVITY: 0.5,
            PreferenceDimension.SPEED_VS_QUALITY: 0.6,
            PreferenceDimension.PRIVACY_SENSITIVITY: 0.7,
            PreferenceDimension.INTERACTION_FREQUENCY: 0.5,
            PreferenceDimension.EXPLANATION_DEPTH: 0.5,
            PreferenceDimension.PROACTIVITY: 0.4,
            PreferenceDimension.MULTIMODALITY: 0.3
        }
        
        for dimension, value in default_preferences.items():
            profile.preferences[dimension] = PreferenceValue(
                value=value,
                confidence=0.1,  # Low initial confidence
                evidence_count=0,
                source="default_initialization"
            )

    async def _initialize_from_cluster(self, profile: UserPreferenceProfile) -> None:
        """Initialize new user profile from cluster center."""
        try:
            # Find the most suitable cluster based on initial signals
            # For now, use the largest cluster
            if self.user_clusters:
                largest_cluster = max(self.user_clusters.values(), key=len)
                
                if largest_cluster:
                    # Get average preferences from cluster members
                    cluster_preferences = defaultdict(list)
                    
                    for user_id in largest_cluster[:10]:  # Sample from cluster
                        if user_id in self.user_profiles:
                            user_profile = self.user_profiles[user_id]
                            for dimension, pref_value in user_profile.preferences.items():
                                cluster_preferences[dimension].append(pref_value.value)
                    
                    # Update profile with cluster averages
                    for dimension, values in cluster_preferences.items():
                        if values:
                            avg_value = sum(values) / len(values)
                            profile.preferences[dimension] = PreferenceValue(
                                value=avg_value,
                                confidence=0.3,  # Medium confidence for cluster initialization
                                evidence_count=len(values),
                                source="cluster_initialization"
                            )
        
        except Exception as e:
            self.logger.warning(f"Failed to initialize from cluster: {str(e)}")

    def _select_best_algorithm(self, profile: UserPreferenceProfile, 
                             dimension: PreferenceDimension) -> PreferenceLearner:
        """Select the best learning algorithm for a user and dimension."""
        # Simple heuristics for algorithm selection
        if profile.total_interactions < 10:
            # Use Bayesian for cold start
            return self.learning_algorithms[LearningAlgorithm.BAYESIAN_INFERENCE]
        elif profile.total_interactions < 100:
            # Use RL for moderate experience
            return self.learning_algorithms.get(
                LearningAlgorithm.REINFORCEMENT_LEARNING,
                self.learning_algorithms[LearningAlgorithm.BAYESIAN_INFERENCE]
            )
        else:
            # Use collaborative filtering for experienced users
            return self.learning_algorithms.get(
                LearningAlgorithm.COLLABORATIVE_FILTERING,
                self.learning_algorithms[LearningAlgorithm.BAYESIAN_INFERENCE]
            )

    def _apply_differential_privacy(self, interaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply differential privacy to interaction data."""
        try:
            # Add noise to numerical values
            private_data = interaction_data.copy()
            
            for key, value in private_data.items():
                if isinstance(value, (int, float)):
                    # Add Laplacian noise
                    sensitivity = 1.0  # Assume unit sensitivity
                    noise_scale = sensitivity / self.privacy_epsilon
                    noise = np.random.laplace(0, noise_scale)
                    private_data[key] = max(0, value + noise)
            
            return private_data
            
        except Exception as e:
            self.logger.warning(f"Failed to apply differential privacy: {str(e)}")
            return interaction_data

    def _extract_preference_signals_from_feedback(
        self,
        feedback_data: Dict[str, Any],
        interaction_context: Optional[Dict[str, Any]] = None
    ) -> Dict[PreferenceDimension, Tuple[float, float]]:
        """Extract preference signals from explicit feedback."""
        signals = {}
        
        rating = feedback_data.get('rating', 3)
        
        # Extract preference signals based on feedback and context
        if interaction_context:
            # Formality feedback
            if 'response_style' in interaction_context:
                style = interaction_context['response_style']
                if rating >= 4:  # Positive feedback
                    if style == 'formal':
                        signals[PreferenceDimension.FORMALITY] = (0.8, 0.8)
                    elif style == 'casual':
                        signals[PreferenceDimension.FORMALITY] = (0.2, 0.8)
                elif rating <= 2:  # Negative feedback
                    if style == 'formal':
                        signals[PreferenceDimension.FORMALITY] = (0.2, 0.8)
                    elif style == 'casual':
                        signals[PreferenceDimension.FORMALITY] = (0.8, 0.8)
            
            # Length feedback
            if 'response_length' in interaction_context:
                length = interaction_context['response_length']
                if rating >= 4:
                    if length > 200:
                        signals[PreferenceDimension.VERBOSITY] = (0.8, 0.7)
                    elif length < 50:
                        signals[PreferenceDimension.VERBOSITY] = (0.2, 0.7)
                elif rating <= 2:
                    if length > 200:
                        signals[PreferenceDimension.VERBOSITY] = (0.2, 0.7)
                    elif length < 50:
                        signals[PreferenceDimension.VERBOSITY] = (0.8, 0.7)
        
        # Direct preference feedback
        for pref_key, pref_value in feedback_data.items():
            if pref_key.startswith('prefer_'):
                dimension_name = pref_key.replace('prefer_', '')
                try:
                    dimension = PreferenceDimension(dimension_name)
                    if isinstance(pref_value, (int, float)):
                        normalized_value = max(0.0, min(1.0, float(pref_value)))
                        signals[dimension] = (normalized_value, 0.9)  # High confidence for direct feedback
                except ValueError:
                    continue
        
        return signals

    async def _store_user_profile(self, profile: UserPreferenceProfile) -> None:
        """Store user profile to persistent storage."""
        try:
            if self.database:
                # Serialize profile
                profile_data = self._serialize_profile(profile)
                
                # Encrypt if encryption is available
                if self.encryption_manager:
                    profile_data = await self.encryption_manager.encrypt(profile_data)
                
                # Store in database
                await self.database.execute(
                    """
                    INSERT OR REPLACE INTO user_preference_profiles (user_id, profile_data, updated_at)
                    VALUES (?, ?, ?)
                    """,
                    (profile.user_id, profile_data, profile.last_updated)
                )
            
            # Cache in Redis if available
            if self.redis_cache:
                cache_data = {
                    'preferences': {dim.value: pref.value for dim, pref in profile.preferences.items()},
                    'last_updated': profile.last_updated.isoformat(),
                    'total_interactions': profile.total_interactions
                }
                await self.redis_cache.set(f"preferences:{profile.user_id}", cache_data, ttl=3600)
        
        except Exception as e:
            self.logger.error(f"Failed to store user profile {profile.user_id}: {str(e)}")

    def _serialize_profile(self, profile: UserPreferenceProfile) -> str:
        """Serialize user profile to string."""
        try:
            # Convert to dictionary
            data = asdict(profile)
            
            # Handle special types
            for key, value in data.items():
                if isinstance(value, datetime):
                    data[key] = value.isoformat()
                elif isinstance(value, dict):
                    # Handle enum keys
                    if any(isinstance(k, Enum) for k in value.keys()):
                        data[key] = {(k.value if isinstance(k, Enum) else k): v for k, v in value.items()}
            
            # Handle numpy arrays
            if profile.personality_vector is not None:
                data['personality_vector'] = profile.personality_vector.tolist()
            
            return json.dumps(data, default=str)
            
        except Exception as e:
            raise PreferenceLearningError(f"Failed to serialize profile: {str(e)}")

    def _deserialize_profile(self, profile_data: str) -> UserPreferenceProfile:
        """Deserialize user profile from string."""
        try:
            data = json.loads(profile_data)
            
            # Convert datetime strings back
            datetime_fields = ['created_at', 'last_updated']
            for field in datetime_fields:
                if data.get(field):
                    data[field] = datetime.fromisoformat(data[field])
            
            # Reconstruct preferences
            preferences = {}
            for dim_str, pref_data in data.get('preferences', {}).items():
                try:
                    dimension = PreferenceDimension(dim_str)
                    if isinstance(pref_data, dict):
                        pref_value = PreferenceValue(**pref_data)
                        if 'last_updated' in pref_data:
                            pref_value.last_updated = datetime.fromisoformat(pref_data['last_updated'])
                        preferences[dimension] = pref_value
                    else:
                        # Legacy format
                        preferences[dimension] = PreferenceValue(value=float(pref_data))
                except ValueError:
                    continue
            
            data['preferences'] = preferences
            
            # Handle numpy arrays
            if 'personality_vector' in data and data['personality_vector']:
                data['personality_vector'] = np.array(data['personality_vector'])
            
            return UserPreferenceProfile(**data)
            
        except Exception as e:
            raise PreferenceLearningError(f"Failed to deserialize profile: {str(e)}")

    async def _clustering_loop(self) -> None:
        """Background task for user clustering."""
        while True:
            try:
                if len(self.user_profiles) >= 10:  # Minimum users for clustering
                    await self._perform_user_clustering()
                
                await asyncio.sleep(self.auto_clustering_interval)
                
            except Exception as e:
                self.logger.error(f"Error in clustering loop: {str(e)}")
                await asyncio.sleep(self.auto_clustering_interval)

    async def _perform_user_clustering(self) -> None:
        """Perform user clustering based on preference patterns."""
        try:
            # Create preference matrix
            user_vectors = []
            user_ids = []
            
            for user_id, profile in self.user_profiles.items():
                if profile.total_interactions >= self.min_interactions_for_learning:
                    vector = []
                    for dimension in PreferenceDimension:
                        if dimension in profile.preferences:
                            value = profile.preferences[dimension
