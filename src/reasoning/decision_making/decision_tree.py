"""
Advanced Decision Tree Engine for AI Assistant
Author: Drmusab
Last Modified: 2025-06-13 08:17:33 UTC

This module provides sophisticated decision-making capabilities for the AI assistant,
supporting multi-criteria decisions, probabilistic reasoning, context-aware choices,
and dynamic adaptation based on learning and feedback.
"""

import hashlib
import inspect
import json
import logging
import math
import statistics
import threading
import time
import uuid
import weakref
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    AsyncGenerator,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Set,
    Type,
    TypeVar,
    Union,
)

import asyncio
import numpy as np

# Assistant components (for integration)
from src.assistant.core import SessionManager

# Core imports
from src.core.config.loader import ConfigLoader
from src.core.dependency_injection import Container
from src.core.error_handling import ErrorHandler, handle_exceptions
from src.core.events.event_bus import EventBus
from src.core.events.event_types import (
    ComponentHealthChanged,
    DecisionCompleted,
    DecisionCriteriaUpdated,
    DecisionFailed,
    DecisionLearned,
    DecisionOutcomeRecorded,
    DecisionStarted,
    DecisionTreeAdapted,
    ErrorOccurred,
    SystemDecisionMade,
    SystemStateChanged,
    UserPreferenceInfluencedDecision,
    WorkflowDecisionMade,
)
from src.core.health_check import HealthCheck

# Learning systems
from src.learning.continual_learning import ContinualLearner
from src.learning.feedback_processor import FeedbackProcessor
from src.learning.preference_learning import PreferenceLearner

# Memory and context
from src.memory.core_memory.memory_manager import MemoryManager
from src.memory.core_memory.memory_types import EpisodicMemory, SemanticMemory, WorkingMemory
from src.memory.operations.context_manager import ContextManager
from src.observability.logging.config import get_logger

# Observability
from src.observability.monitoring.metrics import MetricsCollector
from src.observability.monitoring.tracing import TraceManager

# Type definitions
T = TypeVar("T")


class DecisionType(Enum):
    """Types of decisions the engine can make."""

    BINARY = "binary"  # Yes/No, True/False decisions
    MULTIPLE_CHOICE = "multiple_choice"  # Select one from many options
    RANKING = "ranking"  # Order items by preference
    SCORING = "scoring"  # Assign scores to options
    CLASSIFICATION = "classification"  # Categorize input
    OPTIMIZATION = "optimization"  # Find optimal solution
    PROBABILISTIC = "probabilistic"  # Decisions with uncertainty
    CONDITIONAL = "conditional"  # Decisions based on conditions
    SEQUENTIAL = "sequential"  # Series of dependent decisions
    MULTI_CRITERIA = "multi_criteria"  # Multiple factors considered


class DecisionStrategy(Enum):
    """Strategies for making decisions."""

    RULE_BASED = "rule_based"  # Fixed rules and logic
    PROBABILITY_BASED = "probability_based"  # Probabilistic reasoning
    UTILITY_BASED = "utility_based"  # Utility maximization
    MACHINE_LEARNING = "machine_learning"  # ML-based decisions
    HYBRID = "hybrid"  # Combination of strategies
    CONSENSUS = "consensus"  # Multiple strategy consensus
    ADAPTIVE = "adaptive"  # Strategy adapts over time
    HEURISTIC = "heuristic"  # Simple heuristic rules


class DecisionContext(Enum):
    """Context in which decisions are made."""

    WORKFLOW = "workflow"  # Workflow orchestration
    USER_INTERACTION = "user_interaction"  # User interface decisions
    SYSTEM_OPERATION = "system_operation"  # System management
    RESOURCE_ALLOCATION = "resource_allocation"  # Resource management
    LEARNING = "learning"  # Learning system decisions
    SECURITY = "security"  # Security-related decisions
    PERFORMANCE = "performance"  # Performance optimization
    EMERGENCY = "emergency"  # Emergency response


class ConfidenceLevel(Enum):
    """Confidence levels for decisions."""

    VERY_LOW = 0.2
    LOW = 0.4
    MEDIUM = 0.6
    HIGH = 0.8
    VERY_HIGH = 0.95


@dataclass
class DecisionCriterion:
    """Individual criterion for decision making."""

    criterion_id: str
    name: str
    description: Optional[str] = None
    weight: float = 1.0  # Importance weight (0-1)
    threshold: Optional[float] = None  # Minimum threshold for consideration
    value_type: str = "numeric"  # numeric, boolean, categorical, text
    evaluation_function: Optional[Callable[[Any], float]] = None
    optimization_direction: str = "maximize"  # maximize, minimize
    required: bool = False  # Must be satisfied
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DecisionOption:
    """Represents a possible decision option."""

    option_id: str
    name: str
    description: Optional[str] = None
    value: Any = None
    utility_score: float = 0.0
    probability: float = 0.0
    confidence: float = 0.0
    criteria_scores: Dict[str, float] = field(default_factory=dict)
    constraints: Dict[str, Any] = field(default_factory=dict)
    consequences: Dict[str, Any] = field(default_factory=dict)
    risk_assessment: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DecisionNode:
    """Node in a decision tree structure."""

    node_id: str
    node_type: str = "decision"  # decision, condition, action, outcome
    name: Optional[str] = None
    description: Optional[str] = None

    # Decision logic
    criteria: List[DecisionCriterion] = field(default_factory=list)
    options: List[DecisionOption] = field(default_factory=list)
    condition: Optional[str] = None  # Python expression for evaluation

    # Tree structure
    parent_id: Optional[str] = None
    children: Dict[str, str] = field(default_factory=dict)  # condition -> child_node_id

    # Decision parameters
    decision_type: DecisionType = DecisionType.BINARY
    strategy: DecisionStrategy = DecisionStrategy.RULE_BASED
    threshold: float = 0.5
    require_confidence: float = 0.5

    # Execution tracking
    execution_count: int = 0
    success_count: int = 0
    average_confidence: float = 0.0
    last_execution: Optional[datetime] = None

    # Learning data
    adaptation_rate: float = 0.1
    performance_history: List[float] = field(default_factory=list)

    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    tags: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DecisionTreeDefinition:
    """Complete decision tree definition."""

    tree_id: str
    name: str
    version: str = "1.0.0"
    description: Optional[str] = None

    # Tree structure
    nodes: Dict[str, DecisionNode] = field(default_factory=dict)
    root_node_id: Optional[str] = None

    # Tree characteristics
    decision_context: DecisionContext = DecisionContext.SYSTEM_OPERATION
    max_depth: int = 10
    default_strategy: DecisionStrategy = DecisionStrategy.RULE_BASED

    # Learning configuration
    enable_learning: bool = True
    adaptation_threshold: float = 0.1
    min_samples_for_adaptation: int = 10

    # Performance tracking
    total_decisions: int = 0
    successful_decisions: int = 0
    average_decision_time: float = 0.0
    accuracy_score: float = 0.0

    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: Optional[str] = None
    tags: Set[str] = field(default_factory=set)


@dataclass
class DecisionResult:
    """Result of a decision-making process."""

    decision_id: str
    tree_id: str
    node_id: str

    # Decision outcome
    selected_option: Optional[DecisionOption] = None
    selected_value: Any = None
    confidence: float = 0.0
    utility_score: float = 0.0

    # Decision process
    evaluated_options: List[DecisionOption] = field(default_factory=list)
    criteria_evaluations: Dict[str, float] = field(default_factory=dict)
    decision_path: List[str] = field(default_factory=list)

    # Context and reasoning
    input_context: Dict[str, Any] = field(default_factory=dict)
    reasoning_trace: List[str] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)

    # Performance metrics
    decision_time: float = 0.0
    computation_cost: float = 0.0
    memory_usage: float = 0.0

    # Quality indicators
    uncertainty: float = 0.0
    risk_level: float = 0.0
    completeness: float = 1.0

    # Metadata
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    workflow_id: Optional[str] = None

    # Learning data
    feedback_received: bool = False
    outcome_quality: Optional[float] = None
    lesson_learned: Optional[str] = None


class DecisionError(Exception):
    """Custom exception for decision-making operations."""

    def __init__(
        self,
        message: str,
        decision_id: Optional[str] = None,
        tree_id: Optional[str] = None,
        node_id: Optional[str] = None,
    ):
        super().__init__(message)
        self.decision_id = decision_id
        self.tree_id = tree_id
        self.node_id = node_id
        self.timestamp = datetime.now(timezone.utc)


class DecisionEvaluator(ABC):
    """Abstract base class for decision evaluation strategies."""

    @abstractmethod
    async def evaluate_options(
        self,
        options: List[DecisionOption],
        criteria: List[DecisionCriterion],
        context: Dict[str, Any],
    ) -> List[DecisionOption]:
        """Evaluate and score decision options."""
        pass

    @abstractmethod
    def can_evaluate(self, decision_type: DecisionType, strategy: DecisionStrategy) -> bool:
        """Check if this evaluator can handle the decision type and strategy."""
        pass


class RuleBasedEvaluator(DecisionEvaluator):
    """Rule-based decision evaluation."""

    def __init__(self, logger):
        self.logger = logger

    def can_evaluate(self, decision_type: DecisionType, strategy: DecisionStrategy) -> bool:
        """Check if this evaluator supports rule-based strategies."""
        return strategy in [DecisionStrategy.RULE_BASED, DecisionStrategy.HEURISTIC]

    async def evaluate_options(
        self,
        options: List[DecisionOption],
        criteria: List[DecisionCriterion],
        context: Dict[str, Any],
    ) -> List[DecisionOption]:
        """Evaluate options using rule-based logic."""
        evaluated_options = []

        for option in options:
            total_score = 0.0
            total_weight = 0.0
            criteria_met = 0

            for criterion in criteria:
                # Get value for this criterion from context or option
                value = context.get(
                    f"{criterion.criterion_id}_value", option.metadata.get(criterion.criterion_id)
                )

                if value is None:
                    continue

                # Evaluate criterion
                score = await self._evaluate_criterion(criterion, value, context)

                # Check threshold
                if criterion.threshold and score < criterion.threshold:
                    if criterion.required:
                        # Required criterion not met, skip this option
                        score = 0.0
                        break
                    continue

                # Weight the score
                weighted_score = score * criterion.weight
                total_score += weighted_score
                total_weight += criterion.weight
                criteria_met += 1

                # Store individual criterion score
                option.criteria_scores[criterion.criterion_id] = score

            # Calculate final utility score
            if total_weight > 0:
                option.utility_score = total_score / total_weight
            else:
                option.utility_score = 0.0

            # Calculate confidence based on criteria coverage
            if criteria:
                option.confidence = criteria_met / len(criteria)
            else:
                option.confidence = 1.0

            evaluated_options.append(option)

        # Sort by utility score
        evaluated_options.sort(key=lambda x: x.utility_score, reverse=True)

        return evaluated_options

    async def _evaluate_criterion(
        self, criterion: DecisionCriterion, value: Any, context: Dict[str, Any]
    ) -> float:
        """Evaluate a single criterion."""
        try:
            if criterion.evaluation_function:
                # Use custom evaluation function
                return criterion.evaluation_function(value)

            # Default evaluation based on value type
            if criterion.value_type == "boolean":
                return 1.0 if value else 0.0
            elif criterion.value_type == "numeric":
                # Normalize numeric values (simple linear scaling)
                if isinstance(value, (int, float)):
                    # Simple normalization between 0 and 1
                    max_val = context.get(f"{criterion.criterion_id}_max", 100)
                    min_val = context.get(f"{criterion.criterion_id}_min", 0)
                    if max_val > min_val:
                        normalized = (value - min_val) / (max_val - min_val)
                        return max(0.0, min(1.0, normalized))
                return 0.5  # Default middle score
            elif criterion.value_type == "categorical":
                # For categorical values, use preference mapping
                preferences = context.get(f"{criterion.criterion_id}_preferences", {})
                return preferences.get(value, 0.5)
            else:
                # Text or other types
                return 0.5  # Neutral score

        except Exception as e:
            self.logger.warning(f"Error evaluating criterion {criterion.criterion_id}: {str(e)}")
            return 0.0


class ProbabilisticEvaluator(DecisionEvaluator):
    """Probabilistic decision evaluation using Bayesian reasoning."""

    def __init__(self, logger):
        self.logger = logger

    def can_evaluate(self, decision_type: DecisionType, strategy: DecisionStrategy) -> bool:
        """Check if this evaluator supports probabilistic strategies."""
        return strategy in [DecisionStrategy.PROBABILITY_BASED, DecisionStrategy.UTILITY_BASED]

    async def evaluate_options(
        self,
        options: List[DecisionOption],
        criteria: List[DecisionCriterion],
        context: Dict[str, Any],
    ) -> List[DecisionOption]:
        """Evaluate options using probabilistic reasoning."""
        evaluated_options = []

        # Get prior probabilities from context or use uniform distribution
        priors = context.get("prior_probabilities", {})

        for option in options:
            # Calculate likelihood for each criterion
            likelihood_product = 1.0
            evidence_strength = 0.0

            for criterion in criteria:
                value = context.get(
                    f"{criterion.criterion_id}_value", option.metadata.get(criterion.criterion_id)
                )

                if value is None:
                    continue

                # Calculate likelihood of this value given the option
                likelihood = await self._calculate_likelihood(criterion, value, option, context)
                likelihood_product *= likelihood
                evidence_strength += criterion.weight

                option.criteria_scores[criterion.criterion_id] = likelihood

            # Calculate posterior probability using Bayes' theorem
            prior = priors.get(option.option_id, 1.0 / len(options))  # Uniform prior
            posterior = likelihood_product * prior

            option.probability = posterior
            option.utility_score = posterior
            option.confidence = min(evidence_strength / len(criteria) if criteria else 0.0, 1.0)

            evaluated_options.append(option)

        # Normalize probabilities
        total_probability = sum(opt.probability for opt in evaluated_options)
        if total_probability > 0:
            for option in evaluated_options:
                option.probability /= total_probability
                option.utility_score = option.probability

        # Sort by probability
        evaluated_options.sort(key=lambda x: x.probability, reverse=True)

        return evaluated_options

    async def _calculate_likelihood(
        self,
        criterion: DecisionCriterion,
        value: Any,
        option: DecisionOption,
        context: Dict[str, Any],
    ) -> float:
        """Calculate likelihood of observing value given the option."""
        try:
            if criterion.evaluation_function:
                return criterion.evaluation_function(value)

            # Simple likelihood calculation based on value type
            if criterion.value_type == "numeric":
                # Use Gaussian likelihood
                expected_value = option.metadata.get(f"{criterion.criterion_id}_expected", value)
                variance = context.get(f"{criterion.criterion_id}_variance", 1.0)

                # Gaussian probability density (simplified)
                diff = abs(value - expected_value)
                likelihood = math.exp(-0.5 * (diff**2) / variance)
                return likelihood

            elif criterion.value_type == "boolean":
                expected = option.metadata.get(f"{criterion.criterion_id}_expected", True)
                return 0.9 if value == expected else 0.1

            else:
                # Default uniform likelihood
                return 0.5

        except Exception as e:
            self.logger.warning(
                f"Error calculating likelihood for {criterion.criterion_id}: {str(e)}"
            )
            return 0.1


class MLBasedEvaluator(DecisionEvaluator):
    """Machine learning-based decision evaluation."""

    def __init__(self, logger, learning_system=None):
        self.logger = logger
        self.learning_system = learning_system
        self.models = {}  # Store trained models per decision type

    def can_evaluate(self, decision_type: DecisionType, strategy: DecisionStrategy) -> bool:
        """Check if this evaluator supports ML strategies."""
        return strategy in [DecisionStrategy.MACHINE_LEARNING, DecisionStrategy.ADAPTIVE]

    async def evaluate_options(
        self,
        options: List[DecisionOption],
        criteria: List[DecisionCriterion],
        context: Dict[str, Any],
    ) -> List[DecisionOption]:
        """Evaluate options using machine learning models."""
        evaluated_options = []

        # Prepare feature vector from context and criteria
        features = self._extract_features(criteria, context)

        for option in options:
            # Include option-specific features
            option_features = self._extract_option_features(option, criteria)
            combined_features = {**features, **option_features}

            # Use ML model to predict utility/probability
            score = await self._predict_utility(combined_features, context)

            option.utility_score = score
            option.confidence = min(score, 1.0)  # Use score as confidence

            evaluated_options.append(option)

        # Sort by predicted utility
        evaluated_options.sort(key=lambda x: x.utility_score, reverse=True)

        return evaluated_options

    def _extract_features(
        self, criteria: List[DecisionCriterion], context: Dict[str, Any]
    ) -> Dict[str, float]:
        """Extract numerical features from criteria and context."""
        features = {}

        for criterion in criteria:
            value = context.get(f"{criterion.criterion_id}_value")
            if value is not None:
                # Convert to numerical feature
                if isinstance(value, (int, float)):
                    features[criterion.criterion_id] = float(value)
                elif isinstance(value, bool):
                    features[criterion.criterion_id] = 1.0 if value else 0.0
                elif isinstance(value, str):
                    # Simple hash-based encoding for categorical values
                    features[criterion.criterion_id] = float(hash(value) % 1000) / 1000.0

        return features

    def _extract_option_features(
        self, option: DecisionOption, criteria: List[DecisionCriterion]
    ) -> Dict[str, float]:
        """Extract features specific to an option."""
        features = {}

        for criterion in criteria:
            option_value = option.metadata.get(criterion.criterion_id)
            if option_value is not None:
                feature_name = f"option_{criterion.criterion_id}"
                if isinstance(option_value, (int, float)):
                    features[feature_name] = float(option_value)
                elif isinstance(option_value, bool):
                    features[feature_name] = 1.0 if option_value else 0.0

        return features

    async def _predict_utility(self, features: Dict[str, float], context: Dict[str, Any]) -> float:
        """Predict utility score using ML model."""
        try:
            # This would use actual ML models in a real implementation
            # For now, implement a simple weighted average
            if not features:
                return 0.5

            # Simple linear combination as placeholder
            total = sum(features.values())
            count = len(features)

            if count > 0:
                normalized_score = min(total / count, 1.0)
                return max(normalized_score, 0.0)

            return 0.5

        except Exception as e:
            self.logger.warning(f"Error in ML prediction: {str(e)}")
            return 0.5


class EnhancedDecisionTree:
    """
    Advanced Decision Tree Engine for the AI Assistant.

    This engine provides sophisticated decision-making capabilities including:
    - Multi-criteria decision analysis
    - Probabilistic reasoning with uncertainty handling
    - Context-aware decisions using session and user data
    - Learning-based adaptation of decision trees
    - Integration with workflow orchestration
    - Real-time decision making with performance optimization
    - Memory integration for historical decision analysis
    - Event-driven decision tracking and analytics
    """

    def __init__(self, container: Container):
        """
        Initialize the enhanced decision tree engine.

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

        # Memory and context systems
        self.memory_manager = container.get(MemoryManager)
        self.context_manager = container.get(ContextManager)
        self.working_memory = container.get(WorkingMemory)
        self.episodic_memory = container.get(EpisodicMemory)
        self.semantic_memory = container.get(SemanticMemory)

        # Learning systems
        self.continual_learner = container.get(ContinualLearner)
        self.preference_learner = container.get(PreferenceLearner)
        self.feedback_processor = container.get(FeedbackProcessor)

        # Assistant components
        self.session_manager = container.get(SessionManager)

        # Observability
        self.metrics = container.get(MetricsCollector)
        self.tracer = container.get(TraceManager)

        # Decision management
        self.decision_trees: Dict[str, DecisionTreeDefinition] = {}
        self.active_decisions: Dict[str, DecisionResult] = {}
        self.decision_history: deque = deque(maxlen=10000)

        # Evaluation engines
        self.evaluators: List[DecisionEvaluator] = []
        self.default_evaluator: Optional[DecisionEvaluator] = None

        # Performance tracking
        self.decision_performance: Dict[str, List[float]] = defaultdict(list)
        self.accuracy_tracking: Dict[str, List[float]] = defaultdict(list)

        # Configuration
        self.enable_learning = self.config.get("decision_tree.enable_learning", True)
        self.max_decision_time = self.config.get("decision_tree.max_decision_time", 5.0)
        self.min_confidence_threshold = self.config.get("decision_tree.min_confidence", 0.3)
        self.adaptation_rate = self.config.get("decision_tree.adaptation_rate", 0.1)

        # Threading
        self.decision_semaphore = asyncio.Semaphore(50)  # Max concurrent decisions
        self.thread_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="decision_tree")

        # Initialize components
        self._setup_evaluators()
        self._setup_monitoring()
        self._load_default_trees()

        # Register health check
        self.health_check.register_component("decision_tree", self._health_check_callback)

        self.logger.info("EnhancedDecisionTree initialized successfully")

    def _setup_evaluators(self) -> None:
        """Setup decision evaluation engines."""
        try:
            # Rule-based evaluator
            rule_evaluator = RuleBasedEvaluator(self.logger)
            self.evaluators.append(rule_evaluator)
            self.default_evaluator = rule_evaluator

            # Probabilistic evaluator
            prob_evaluator = ProbabilisticEvaluator(self.logger)
            self.evaluators.append(prob_evaluator)

            # ML-based evaluator
            ml_evaluator = MLBasedEvaluator(self.logger, self.continual_learner)
            self.evaluators.append(ml_evaluator)

            self.logger.info(f"Initialized {len(self.evaluators)} decision evaluators")

        except Exception as e:
            self.logger.error(f"Failed to setup evaluators: {str(e)}")

    def _setup_monitoring(self) -> None:
        """Setup monitoring and metrics."""
        try:
            # Register decision metrics
            self.metrics.register_counter("decisions_made_total")
            self.metrics.register_counter("decisions_successful")
            self.metrics.register_counter("decisions_failed")
            self.metrics.register_histogram("decision_duration_seconds")
            self.metrics.register_histogram("decision_confidence_score")
            self.metrics.register_gauge("active_decisions")
            self.metrics.register_counter("decision_trees_adapted")

        except Exception as e:
            self.logger.warning(f"Failed to setup monitoring: {str(e)}")

    def _load_default_trees(self) -> None:
        """Load default decision trees."""
        try:
            # Create basic workflow decision tree
            workflow_tree = self._create_workflow_decision_tree()
            self.decision_trees[workflow_tree.tree_id] = workflow_tree

            # Create user interaction decision tree
            interaction_tree = self._create_interaction_decision_tree()
            self.decision_trees[interaction_tree.tree_id] = interaction_tree

            # Create system operation decision tree
            system_tree = self._create_system_decision_tree()
            self.decision_trees[system_tree.tree_id] = system_tree

            self.logger.info(f"Loaded {len(self.decision_trees)} default decision trees")

        except Exception as e:
            self.logger.warning(f"Failed to load default trees: {str(e)}")

    def _create_workflow_decision_tree(self) -> DecisionTreeDefinition:
        """Create a decision tree for workflow routing decisions."""
        tree = DecisionTreeDefinition(
            tree_id="default_workflow",
            name="Workflow Decision Tree",
            description="Decides on workflow routing and execution strategies",
            decision_context=DecisionContext.WORKFLOW,
        )

        # Root decision node
        root_node = DecisionNode(
            node_id="workflow_root",
            name="Workflow Strategy Selection",
            decision_type=DecisionType.MULTIPLE_CHOICE,
            strategy=DecisionStrategy.RULE_BASED,
            criteria=[
                DecisionCriterion(
                    criterion_id="complexity",
                    name="Task Complexity",
                    weight=0.4,
                    value_type="numeric",
                ),
                DecisionCriterion(
                    criterion_id="urgency", name="Task Urgency", weight=0.3, value_type="numeric"
                ),
                DecisionCriterion(
                    criterion_id="resources",
                    name="Available Resources",
                    weight=0.3,
                    value_type="numeric",
                ),
            ],
            options=[
                DecisionOption(
                    option_id="sequential",
                    name="Sequential Execution",
                    description="Execute workflow steps sequentially",
                ),
                DecisionOption(
                    option_id="parallel",
                    name="Parallel Execution",
                    description="Execute workflow steps in parallel",
                ),
                DecisionOption(
                    option_id="adaptive",
                    name="Adaptive Execution",
                    description="Adapt execution strategy dynamically",
                ),
            ],
        )

        tree.nodes[root_node.node_id] = root_node
        tree.root_node_id = root_node.node_id

        return tree

    def _create_interaction_decision_tree(self) -> DecisionTreeDefinition:
        """Create a decision tree for user interaction decisions."""
        tree = DecisionTreeDefinition(
            tree_id="default_interaction",
            name="User Interaction Decision Tree",
            description="Decides on user interaction strategies and responses",
            decision_context=DecisionContext.USER_INTERACTION,
        )

        # Root decision node
        root_node = DecisionNode(
            node_id="interaction_root",
            name="Interaction Strategy Selection",
            decision_type=DecisionType.MULTIPLE_CHOICE,
            strategy=DecisionStrategy.UTILITY_BASED,
            criteria=[
                DecisionCriterion(
                    criterion_id="user_expertise",
                    name="User Expertise Level",
                    weight=0.3,
                    value_type="categorical",
                ),
                DecisionCriterion(
                    criterion_id="context_complexity",
                    name="Context Complexity",
                    weight=0.25,
                    value_type="numeric",
                ),
                DecisionCriterion(
                    criterion_id="time_pressure",
                    name="Time Pressure",
                    weight=0.25,
                    value_type="numeric",
                ),
                DecisionCriterion(
                    criterion_id="user_preference",
                    name="User Preference",
                    weight=0.2,
                    value_type="categorical",
                ),
            ],
            options=[
                DecisionOption(
                    option_id="detailed",
                    name="Detailed Response",
                    description="Provide comprehensive, detailed response",
                ),
                DecisionOption(
                    option_id="concise",
                    name="Concise Response",
                    description="Provide brief, focused response",
                ),
                DecisionOption(
                    option_id="interactive",
                    name="Interactive Response",
                    description="Engage in interactive dialogue",
                ),
                DecisionOption(
                    option_id="guided",
                    name="Guided Response",
                    description="Provide step-by-step guidance",
                ),
            ],
        )

        tree.nodes[root_node.node_id] = root_node
        tree.root_node_id = root_node.node_id

        return tree

    def _create_system_decision_tree(self) -> DecisionTreeDefinition:
        """Create a decision tree for system operation decisions."""
        tree = DecisionTreeDefinition(
            tree_id="default_system",
            name="System Operation Decision Tree",
            description="Decides on system operations and resource allocation",
            decision_context=DecisionContext.SYSTEM_OPERATION,
        )

        # Root decision node
        root_node = DecisionNode(
            node_id="system_root",
            name="System Operation Strategy",
            decision_type=DecisionType.OPTIMIZATION,
            strategy=DecisionStrategy.UTILITY_BASED,
            criteria=[
                DecisionCriterion(
                    criterion_id="cpu_usage",
                    name="CPU Usage",
                    weight=0.3,
                    value_type="numeric",
                    optimization_direction="minimize",
                ),
                DecisionCriterion(
                    criterion_id="memory_usage",
                    name="Memory Usage",
                    weight=0.3,
                    value_type="numeric",
                    optimization_direction="minimize",
                ),
                DecisionCriterion(
                    criterion_id="response_time",
                    name="Response Time",
                    weight=0.2,
                    value_type="numeric",
                    optimization_direction="minimize",
                ),
                DecisionCriterion(
                    criterion_id="throughput",
                    name="Throughput",
                    weight=0.2,
                    value_type="numeric",
                    optimization_direction="maximize",
                ),
            ],
            options=[
                DecisionOption(
                    option_id="optimize_speed",
                    name="Optimize for Speed",
                    description="Prioritize fast response times",
                ),
                DecisionOption(
                    option_id="optimize_resources",
                    name="Optimize for Resources",
                    description="Prioritize resource conservation",
                ),
                DecisionOption(
                    option_id="balanced",
                    name="Balanced Optimization",
                    description="Balance speed and resource usage",
                ),
            ],
        )

        tree.nodes[root_node.node_id] = root_node
        tree.root_node_id = root_node.node_id

        return tree

    async def initialize(self) -> None:
        """Initialize the decision tree engine."""
        try:
            # Start background tasks
            asyncio.create_task(self._decision_monitoring_loop())
            asyncio.create_task(self._adaptation_loop())
            asyncio.create_task(self._performance_optimization_loop())

            # Register event handlers
            await self._register_event_handlers()

            self.logger.info("DecisionTree initialization completed")

        except Exception as e:
            self.logger.error(f"Failed to initialize DecisionTree: {str(e)}")
            raise DecisionError(f"Initialization failed: {str(e)}")

    async def _register_event_handlers(self) -> None:
        """Register event handlers for system events."""
        # Workflow events
        self.event_bus.subscribe("workflow_step_started", self._handle_workflow_step_started)
        self.event_bus.subscribe("workflow_completed", self._handle_workflow_completed)

        # User interaction events
        self.event_bus.subscribe("user_interaction_started", self._handle_user_interaction_started)
        self.event_bus.subscribe("feedback_received", self._handle_feedback_received)

        # Learning events
        self.event_bus.subscribe("learning_event_occurred", self._handle_learning_event)

    @handle_exceptions
    async def make_decision(
        self,
        tree_id: str,
        context: Dict[str, Any],
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        node_id: Optional[str] = None,
    ) -> DecisionResult:
        """
        Make a decision using the specified decision tree.

        Args:
            tree_id: ID of the decision tree to use
            context: Context information for the decision
            session_id: Optional session ID
            user_id: Optional user ID
            node_id: Optional specific node to start from

        Returns:
            Decision result with selected option and reasoning
        """
        async with self.decision_semaphore:
            start_time = time.time()
            decision_id = str(uuid.uuid4())

            # Get decision tree
            tree = self.decision_trees.get(tree_id)
            if not tree:
                raise DecisionError(f"Decision tree {tree_id} not found", decision_id, tree_id)

            # Determine starting node
            start_node_id = node_id or tree.root_node_id
            if not start_node_id or start_node_id not in tree.nodes:
                raise DecisionError(f"Invalid start node {start_node_id}", decision_id, tree_id)

            try:
                with self.tracer.trace("decision_making") as span:
                    span.set_attributes(
                        {
                            "decision_id": decision_id,
                            "tree_id": tree_id,
                            "session_id": session_id or "unknown",
                            "user_id": user_id or "anonymous",
                            "start_node": start_node_id,
                        }
                    )

                    # Emit decision started event
                    await self.event_bus.emit(
                        DecisionStarted(
                            decision_id=decision_id,
                            tree_id=tree_id,
                            node_id=start_node_id,
                            context=context,
                            session_id=session_id,
                            user_id=user_id,
                        )
                    )

                    # Enhance context with session and user data
                    enhanced_context = await self._enhance_context(context, session_id, user_id)

                    # Navigate decision tree and make decision
                    result = await self._navigate_tree(
                        tree, start_node_id, enhanced_context, decision_id, session_id, user_id
                    )

                    # Calculate final metrics
                    decision_time = time.time() - start_time
                    result.decision_time = decision_time

                    # Store decision
                    self.active_decisions[decision_id] = result
                    self.decision_history.append(result)

                    # Update tree statistics
                    tree.total_decisions += 1
                    if result.confidence >= self.min_confidence_threshold:
                        tree.successful_decisions += 1

                    # Update performance tracking
                    self.decision_performance[tree_id].append(decision_time)

                    # Store decision for learning
                    if self.enable_learning:
                        await self._store_decision_for_learning(result)

                    # Emit decision completed event
                    await self.event_bus.emit(
                        DecisionCompleted(
                            decision_id=decision_id,
                            tree_id=tree_id,
                            selected_option=(
                                result.selected_option.option_id if result.selected_option else None
                            ),
                            confidence=result.confidence,
                            decision_time=decision_time,
                            session_id=session_id,
                            user_id=user_id,
                        )
                    )

                    # Update metrics
                    self.metrics.increment("decisions_made_total")
                    self.metrics.increment("decisions_successful")
                    self.metrics.record("decision_duration_seconds", decision_time)
                    self.metrics.record("decision_confidence_score", result.confidence)

                    self.logger.info(
                        f"Decision {decision_id} completed: {result.selected_option.option_id if result.selected_option else 'none'} "
                        f"(confidence: {result.confidence:.2f}, time: {decision_time:.3f}s)"
                    )

                    return result

            except Exception as e:
                # Handle decision failure
                decision_time = time.time() - start_time

                error_result = DecisionResult(
                    decision_id=decision_id,
                    tree_id=tree_id,
                    node_id=start_node_id,
                    input_context=enhanced_context if "enhanced_context" in locals() else context,
                    decision_time=decision_time,
                    session_id=session_id,
                    user_id=user_id,
                )
                error_result.reasoning_trace.append(f"Decision failed: {str(e)}")

                await self.event_bus.emit(
                    DecisionFailed(
                        decision_id=decision_id,
                        tree_id=tree_id,
                        error_message=str(e),
                        error_type=type(e).__name__,
                        session_id=session_id,
                        user_id=user_id,
                    )
                )

                self.metrics.increment("decisions_failed")
                self.logger.error(f"Decision {decision_id} failed: {str(e)}")

                return error_result

    async def _enhance_context(
        self, context: Dict[str, Any], session_id: Optional[str], user_id: Optional[str]
    ) -> Dict[str, Any]:
        """Enhance context with session and user data."""
        enhanced_context = dict(context)

        try:
            # Add session context
            if session_id:
                session_context = await self.context_manager.get_session_context(session_id)
                enhanced_context.update(session_context)

                # Add working memory data
                working_memory_data = await self.working_memory.get_session_data(session_id)
                enhanced_context["working_memory"] = working_memory_data

            # Add user preferences
            if user_id and hasattr(self.preference_learner, "get_user_preferences"):
                user_preferences = await self.preference_learner.get_user_preferences(user_id)
                enhanced_context["user_preferences"] = user_preferences

                # Add user decision history
                user_decisions = [
                    decision for decision in self.decision_history if decision.user_id == user_id
                ][
                    -10:
                ]  # Last 10 decisions
                enhanced_context["user_decision_history"] = user_decisions

            # Add current timestamp and time-based context
            current_time = datetime.now(timezone.utc)
            enhanced_context.update(
                {
                    "current_timestamp": current_time,
                    "hour_of_day": current_time.hour,
                    "day_of_week": current_time.weekday(),
                    "is_weekend": current_time.weekday() >= 5,
                }
            )

        except Exception as e:
            self.logger.warning(f"Failed to enhance context: {str(e)}")

        return enhanced_context

    async def _navigate_tree(
        self,
        tree: DecisionTreeDefinition,
        node_id: str,
        context: Dict[str, Any],
        decision_id: str,
        session_id: Optional[str],
        user_id: Optional[str],
    ) -> DecisionResult:
        """Navigate the decision tree and make decisions."""
        decision_path = []
        current_node_id = node_id
        max_depth = tree.max_depth
        depth = 0

        while current_node_id and depth < max_depth:
            node = tree.nodes.get(current_node_id)
            if not node:
                break

            decision_path.append(current_node_id)

            # Update node execution statistics
            node.execution_count += 1
            node.last_execution = datetime.now(timezone.utc)

            # Evaluate current node
            if node.node_type == "decision":
                # Make decision at this node
                result = await self._evaluate_decision_node(
                    node, context, decision_id, tree.tree_id
                )

                # Create final result
                final_result = DecisionResult(
                    decision_id=decision_id,
                    tree_id=tree.tree_id,
                    node_id=current_node_id,
                    selected_option=result.selected_option,
                    selected_value=result.selected_value,
                    confidence=result.confidence,
                    utility_score=result.utility_score,
                    evaluated_options=result.evaluated_options,
                    criteria_evaluations=result.criteria_evaluations,
                    decision_path=decision_path,
                    input_context=context,
                    reasoning_trace=result.reasoning_trace,
                    session_id=session_id,
                    user_id=user_id,
                )

                return final_result

            elif node.node_type == "condition":
                # Evaluate condition and move to next node
                next_node_id = await self._evaluate_condition_node(node, context)
                current_node_id = next_node_id

            else:
                # Action or outcome node - end navigation
                break

            depth += 1

        # If we get here, create a default result
        return DecisionResult(
            decision_id=decision_id,
            tree_id=tree.tree_id,
            node_id=current_node_id,
            decision_path=decision_path,
            input_context=context,
            reasoning_trace=["Navigation completed without decision"],
            session_id=session_id,
            user_id=user_id,
        )

    async def _evaluate_decision_node(
        self, node: DecisionNode, context: Dict[str, Any], decision_id: str, tree_id: str
    ) -> DecisionResult:
        """Evaluate a decision node and select the best option."""
        start_time = time.time()

        # Find appropriate evaluator
        evaluator = self._find_evaluator(node.decision_type, node.strategy)
        if not evaluator:
            evaluator = self.default_evaluator

        if not evaluator:
            raise DecisionError(f"No evaluator available for decision type {node.decision_type}")

        try:
            # Evaluate options
            evaluated_options = await evaluator.evaluate_options(
                node.options.copy(), node.criteria, context
            )

            # Select best option
            best_option = None
            confidence = 0.0

            if evaluated_options:
                best_option = evaluated_options[0]  # Assuming sorted by score
                confidence = best_option.confidence

                # Check if confidence meets threshold
                if confidence < node.require_confidence:
                    # Low confidence decision
                    confidence *= 0.8  # Reduce confidence for threshold miss

            # Calculate criteria evaluations
            criteria_evaluations = {}
            if best_option:
                criteria_evaluations = best_option.criteria_scores

            # Update node performance
            node.success_count += 1 if confidence >= node.require_confidence else 0
            node.performance_history.append(confidence)
            node.average_confidence = statistics.mean(node.performance_history[-100:])  # Last 100

            # Create result
            result = DecisionResult(
                decision_id=decision_id,
                tree_id=tree_id,
                node_id=node.node_id,
                selected_option=best_option,
                selected_value=best_option.value if best_option else None,
                confidence=confidence,
                utility_score=best_option.utility_score if best_option else 0.0,
                evaluated_options=evaluated_options,
                criteria_evaluations=criteria_evaluations,
                decision_time=time.time() - start_time,
            )

            # Add reasoning trace
            if best_option:
                result.reasoning_trace.append(
                    f"Selected option '{best_option.name}' with confidence {confidence:.2f}"
                )
                result.reasoning_trace.append(f"Utility score: {best_option.utility_score:.2f}")

                for criterion_id, score in criteria_evaluations.items():
                    result.reasoning_trace.append(f"Criterion '{criterion_id}': {score:.2f}")

            return result

        except Exception as e:
            raise DecisionError(f"Failed to evaluate decision node {node.node_id}: {str(e)}")

    async def _evaluate_condition_node(
        self, node: DecisionNode, context: Dict[str, Any]
    ) -> Optional[str]:
        """Evaluate a condition node and return next node ID."""
        try:
            if node.condition:
                # Evaluate condition expression
                # This would need proper sandboxing in production
                condition_result = eval(node.condition, {"__builtins__": {}}, context)

                if condition_result:
                    return node.children.get("true")
                else:
                    return node.children.get("false")

            return None

        except Exception as e:
            self.logger.warning(f"Failed to evaluate condition in node {node.node_id}: {str(e)}")
            return node.children.get("default")

    def _find_evaluator(
        self, decision_type: DecisionType, strategy: DecisionStrategy
    ) -> Optional[DecisionEvaluator]:
        """Find appropriate evaluator for decision type and strategy."""
        for evaluator in self.evaluators:
            if evaluator.can_evaluate(decision_type, strategy):
                return evaluator
        return None

    async def _store_decision_for_learning(self, result: DecisionResult) -> None:
        """Store decision result for learning and adaptation."""
        try:
            learning_data = {
                "decision_id": result.decision_id,
                "tree_id": result.tree_id,
                "node_id": result.node_id,
                "selected_option": (
                    result.selected_option.option_id if result.selected_option else None
                ),
                "confidence": result.confidence,
                "utility_score": result.utility_score,
                "decision_time": result.decision_time,
                "context": result.input_context,
                "criteria_evaluations": result.criteria_evaluations,
                "timestamp": result.timestamp,
                "session_id": result.session_id,
                "user_id": result.user_id,
            }

            # Store in episodic memory
            await self.episodic_memory.store(
                {
                    "event_type": "decision_made",
                    "data": learning_data,
                    "session_id": result.session_id,
                }
            )

            # Update continual learning
            if self.continual_learner:
                await self.continual_learner.learn_from_decision(learning_data)

        except Exception as e:
            self.logger.warning(f"Failed to store decision for learning: {str(e)}")

    @handle_exceptions
    async def provide_feedback(
        self, decision_id: str, feedback: Dict[str, Any], user_id: Optional[str] = None
    ) -> None:
        """
        Provide feedback on a decision for learning and adaptation.

        Args:
            decision_id: ID of the decision to provide feedback on
            feedback: Feedback data including quality score, correctness, etc.
            user_id: Optional user ID providing feedback
        """
        # Find decision result
        decision_result = None
        for decision in self.decision_history:
            if decision.decision_id == decision_id:
                decision_result = decision
                break

        if not decision_result:
            raise DecisionError(f"Decision {decision_id} not found for feedback")

        try:
            # Update decision result with feedback
            decision_result.feedback_received = True
            decision_result.outcome_quality = feedback.get("quality_score", 0.5)
            decision_result.lesson_learned = feedback.get("lesson", "")

            # Process feedback for learning
            if self.feedback_processor:
                await self.feedback_processor.process_decision_feedback(
                    decision_result, feedback, user_id
                )

            # Adapt decision tree if needed
            if self.enable_learning:
                await self._adapt_tree_from_feedback(decision_result, feedback)

            # Emit feedback event
            await self.event_bus.emit(
                DecisionLearned(
                    decision_id=decision_id,
                    tree_id=decision_result.tree_id,
                    feedback_quality=decision_result.outcome_quality,
                    user_id=user_id,
                )
            )

            self.logger.info(f"Feedback processed for decision {decision_id}")

        except Exception as e:
            self.logger.error(f"Failed to process feedback for decision {decision_id}: {str(e)}")
            raise DecisionError(f"Failed to process feedback: {str(e)}")

    async def _adapt_tree_from_feedback(
        self, decision_result: DecisionResult, feedback: Dict[str, Any]
    ) -> None:
        """Adapt decision tree based on feedback."""
        try:
            tree = self.decision_trees.get(decision_result.tree_id)
            if not tree or not tree.enable_learning:
                return

            node = tree.nodes.get(decision_result.node_id)
            if not node:
                return

            # Extract feedback metrics
            quality_score = feedback.get("quality_score", 0.5)
            correctness = feedback.get("correctness", True)

            # Update node performance tracking
            if correctness and quality_score > 0.7:
                # Good decision - reinforce
                for criterion in node.criteria:
                    if criterion.criterion_id in decision_result.criteria_evaluations:
                        # Slightly increase weight of successful criteria
                        criterion.weight = min(
                            1.0, criterion.weight * (1 + tree.adaptation_threshold)
                        )

            elif not correctness or quality_score < 0.3:
                # Poor decision - adjust
                for criterion in node.criteria:
                    if criterion.criterion_id in decision_result.criteria_evaluations:
                        # Slightly decrease weight of unsuccessful criteria
                        criterion.weight = max(
                            0.1, criterion.weight * (1 - tree.adaptation_threshold)
                        )

            # Update tree metadata
            tree.updated_at = datetime.now(timezone.utc)

            # Emit adaptation event
            await self.event_bus.emit(
                DecisionTreeAdapted(
                    tree_id=tree.tree_id,
                    node_id=node.node_id,
                    adaptation_type="weight_adjustment",
                    feedback_quality=quality_score,
                )
            )

            self.metrics.increment("decision_trees_adapted")

        except Exception as e:
            self.logger.warning(f"Failed to adapt tree from feedback: {str(e)}")

    def register_tree(self, tree: DecisionTreeDefinition) -> None:
        """
        Register a new decision tree.

        Args:
            tree: Decision tree definition to register
        """
        self.decision_trees[tree.tree_id] = tree
        self.logger.info(f"Registered decision tree: {tree.tree_id}")

    def get_tree(self, tree_id: str) -> Optional[DecisionTreeDefinition]:
        """Get a decision tree by ID."""
        return self.decision_trees.get(tree_id)

    def list_trees(self) -> List[Dict[str, Any]]:
        """List all registered decision trees."""
        return [
            {
                "tree_id": tree.tree_id,
                "name": tree.name,
                "version": tree.version,
                "description": tree.description,
                "context": tree.decision_context.value,
                "total_decisions": tree.total_decisions,
                "successful_decisions": tree.successful_decisions,
                "accuracy": tree.successful_decisions / max(tree.total_decisions, 1),
                "node_count": len(tree.nodes),
                "created_at": tree.created_at.isoformat(),
                "updated_at": tree.updated_at.isoformat(),
            }
            for tree in self.decision_trees.values()
        ]

    def get_decision_history(
        self, user_id: Optional[str] = None, session_id: Optional[str] = None, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get decision history with optional filtering."""
        filtered_decisions = []

        for decision in reversed(self.decision_history):
            if user_id and decision.user_id != user_id:
                continue
            if session_id and decision.session_id != session_id:
                continue

            filtered_decisions.append(
                {
                    "decision_id": decision.decision_id,
                    "tree_id": decision.tree_id,
                    "node_id": decision.node_id,
                    "selected_option": (
                        decision.selected_option.option_id if decision.selected_option else None
                    ),
                    "confidence": decision.confidence,
                    "decision_time": decision.decision_time,
                    "timestamp": decision.timestamp.isoformat(),
                    "feedback_received": decision.feedback_received,
                    "outcome_quality": decision.outcome_quality,
                }
            )

            if len(filtered_decisions) >= limit:
                break

        return filtered_decisions

    async def decide(
        self,
        context: Dict[str, Any],
        conditions: Optional[List[Any]] = None,
        tree_id: str = "default_workflow",
    ) -> List[str]:
        """
        Simple decision interface for workflow integration.

        Args:
            context: Decision context
            conditions: Decision conditions (for compatibility)
            tree_id: Decision tree to use

        Returns:
            List of next steps or options
        """
        try:
            result = await self.make_decision(tree_id, context)

            if result.selected_option:
                return [result.selected_option.option_id]
            else:
                return []

        except Exception as e:
            self.logger.error(f"Simple decision failed: {str(e)}")
            return ["default"]

    async def _decision_monitoring_loop(self) -> None:
        """Background task for monitoring active decisions."""
        while True:
            try:
                # Update active decisions metric
                self.metrics.set("active_decisions", len(self.active_decisions))

                # Clean up old active decisions
                current_time = datetime.now(timezone.utc)
                expired_decisions = []

                for decision_id, decision in self.active_decisions.items():
                    age = (current_time - decision.timestamp).total_seconds()
                    if age > 3600:  # 1 hour
                        expired_decisions.append(decision_id)

                for decision_id in expired_decisions:
                    self.active_decisions.pop(decision_id, None)

                await asyncio.sleep(60)  # Monitor every minute

            except Exception as e:
                self.logger.error(f"Error in decision monitoring: {str(e)}")
                await asyncio.sleep(60)

    async def _adaptation_loop(self) -> None:
        """Background task for decision tree adaptation."""
        while True:
            try:
                if not self.enable_learning:
                    await asyncio.sleep(300)
                    continue

                # Perform periodic adaptation
                for tree in self.decision_trees.values():
                    if (
                        tree.enable_learning
                        and tree.total_decisions >= tree.min_samples_for_adaptation
                    ):
                        await self._perform_tree_adaptation(tree)

                await asyncio.sleep(300)  # Sleep for 5 minutes between adaptations

            except Exception as e:
                self.logger.error(f"Error in decision tree adaptation loop: {str(e)}")
                await asyncio.sleep(300)
