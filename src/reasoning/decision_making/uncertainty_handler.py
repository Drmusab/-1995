"""
Advanced Uncertainty Handling System
Author: Drmusab
Last Modified: 2025-01-13 08:25:46 UTC

This module provides comprehensive uncertainty quantification and handling for the AI assistant,
enabling probabilistic reasoning, confidence scoring, and uncertainty-aware decision making
with integration to all core system components.
"""

import hashlib
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
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import asyncio
import numpy as np
import scipy.stats as stats

# Core imports
from src.core.config.loader import ConfigLoader
from src.core.dependency_injection import Container
from src.core.error_handling import ErrorHandler, handle_exceptions
from src.core.events.event_bus import EventBus
from src.core.events.event_types import (
    ComponentHealthChanged,
    ConfidenceScoreCalculated,
    ErrorOccurred,
    LearningEventOccurred,
    ProbabilisticDecisionMade,
    SystemStateChanged,
    UncertaintyAnomalyDetected,
    UncertaintyCalibrated,
    UncertaintyLearningOccurred,
    UncertaintyModelUpdated,
    UncertaintyQuantified,
    UncertaintyThresholdExceeded,
)
from src.core.health_check import HealthCheck

# Learning and adaptation
from src.learning.continual_learning import ContinualLearner
from src.learning.feedback_processor import FeedbackProcessor
from src.learning.model_adaptation import ModelAdapter
from src.learning.preference_learning import PreferenceLearner

# Memory systems
from src.memory.core_memory.memory_manager import MemoryManager
from src.memory.core_memory.memory_types import EpisodicMemory, SemanticMemory, WorkingMemory
from src.memory.operations.context_manager import ContextManager
from src.observability.logging.config import get_logger

# Observability
from src.observability.monitoring.metrics import MetricsCollector
from src.observability.monitoring.tracing import TraceManager

# Type definitions
T = TypeVar("T")


class UncertaintyType(Enum):
    """Types of uncertainty in the system."""

    EPISTEMIC = "epistemic"  # Knowledge uncertainty
    ALEATORIC = "aleatoric"  # Inherent randomness
    MODEL = "model"  # Model uncertainty
    DATA = "data"  # Data uncertainty
    MEASUREMENT = "measurement"  # Measurement uncertainty
    TEMPORAL = "temporal"  # Time-dependent uncertainty
    CONTEXTUAL = "contextual"  # Context-dependent uncertainty
    ADVERSARIAL = "adversarial"  # Adversarial uncertainty


class ConfidenceLevel(Enum):
    """Confidence levels for uncertainty quantification."""

    VERY_LOW = "very_low"  # 0-20%
    LOW = "low"  # 20-40%
    MEDIUM = "medium"  # 40-60%
    HIGH = "high"  # 60-80%
    VERY_HIGH = "very_high"  # 80-100%


class UncertaintyMethod(Enum):
    """Methods for uncertainty quantification."""

    BAYESIAN = "bayesian"  # Bayesian uncertainty
    ENSEMBLE = "ensemble"  # Ensemble-based uncertainty
    DROPOUT = "dropout"  # Monte Carlo dropout
    BOOTSTRAP = "bootstrap"  # Bootstrap sampling
    CONFORMAL = "conformal"  # Conformal prediction
    INFORMATION_THEORY = "information_theory"  # Information-theoretic
    DISTRIBUTION = "distribution"  # Distributional uncertainty
    EVIDENTIAL = "evidential"  # Evidential deep learning


@dataclass
class UncertaintyEstimate:
    """Container for uncertainty estimation results."""

    estimate_id: str
    uncertainty_type: UncertaintyType
    method: UncertaintyMethod

    # Core uncertainty metrics
    mean_estimate: float
    variance: float
    std_deviation: float
    confidence_interval: Tuple[float, float]
    confidence_level: float = 0.95

    # Probabilistic measures
    entropy: Optional[float] = None
    mutual_information: Optional[float] = None
    expected_information_gain: Optional[float] = None

    # Quality metrics
    calibration_score: Optional[float] = None
    sharpness: Optional[float] = None
    reliability: Optional[float] = None

    # Context information
    input_features: Optional[Dict[str, Any]] = None
    model_state: Optional[Dict[str, Any]] = None
    environmental_factors: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    computation_time: float = 0.0
    sample_size: Optional[int] = None

    # Quality assessment
    is_calibrated: bool = False
    is_reliable: bool = True
    anomaly_score: float = 0.0


@dataclass
class UncertaintyContext:
    """Context for uncertainty quantification."""

    context_id: str
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    task_type: Optional[str] = None

    # Decision context
    decision_importance: float = 0.5  # 0-1 scale
    time_pressure: float = 0.5  # 0-1 scale
    risk_tolerance: float = 0.5  # 0-1 scale

    # Historical context
    previous_estimates: List[str] = field(default_factory=list)
    feedback_history: List[Dict[str, Any]] = field(default_factory=list)
    performance_history: List[float] = field(default_factory=list)

    # Environmental context
    system_load: float = 0.5
    data_quality: float = 1.0
    model_freshness: float = 1.0

    # Preferences
    preferred_methods: List[UncertaintyMethod] = field(default_factory=list)
    uncertainty_tolerance: float = 0.5

    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    tags: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)


class UncertaintyError(Exception):
    """Custom exception for uncertainty handling operations."""

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        uncertainty_type: Optional[UncertaintyType] = None,
    ):
        super().__init__(message)
        self.error_code = error_code
        self.uncertainty_type = uncertainty_type
        self.timestamp = datetime.now(timezone.utc)


class UncertaintyQuantifier(ABC):
    """Abstract base class for uncertainty quantification methods."""

    @abstractmethod
    def get_method_type(self) -> UncertaintyMethod:
        """Get the uncertainty method type."""
        pass

    @abstractmethod
    async def quantify(
        self, input_data: Any, model_predictions: Any, context: UncertaintyContext
    ) -> UncertaintyEstimate:
        """Quantify uncertainty for given input and predictions."""
        pass

    @abstractmethod
    def supports_uncertainty_type(self, uncertainty_type: UncertaintyType) -> bool:
        """Check if this quantifier supports the given uncertainty type."""
        pass

    async def calibrate(self, validation_data: List[Tuple[Any, Any, float]]) -> Dict[str, float]:
        """Calibrate the uncertainty quantifier using validation data."""
        return {"calibration_score": 1.0}


class BayesianUncertaintyQuantifier(UncertaintyQuantifier):
    """Bayesian uncertainty quantification."""

    def __init__(self, prior_params: Optional[Dict[str, float]] = None):
        self.prior_params = prior_params or {"alpha": 1.0, "beta": 1.0}
        self.posterior_params = self.prior_params.copy()
        self.logger = get_logger(__name__)

    def get_method_type(self) -> UncertaintyMethod:
        return UncertaintyMethod.BAYESIAN

    def supports_uncertainty_type(self, uncertainty_type: UncertaintyType) -> bool:
        return uncertainty_type in [UncertaintyType.EPISTEMIC, UncertaintyType.MODEL]

    async def quantify(
        self, input_data: Any, model_predictions: Any, context: UncertaintyContext
    ) -> UncertaintyEstimate:
        """Quantify Bayesian uncertainty."""
        start_time = time.time()

        try:
            # Extract prediction statistics
            if isinstance(model_predictions, (list, np.ndarray)):
                predictions = np.array(model_predictions)
                mean_pred = np.mean(predictions)
                var_pred = np.var(predictions)
            else:
                mean_pred = float(model_predictions)
                var_pred = 0.1  # Default uncertainty

            # Bayesian update
            alpha = self.posterior_params["alpha"]
            beta = self.posterior_params["beta"]

            # Calculate posterior statistics
            posterior_mean = alpha / (alpha + beta)
            posterior_var = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))
            posterior_std = math.sqrt(posterior_var)

            # Confidence interval using Beta distribution
            conf_level = context.metadata.get("confidence_level", 0.95)
            alpha_level = (1 - conf_level) / 2

            ci_lower = stats.beta.ppf(alpha_level, alpha, beta)
            ci_upper = stats.beta.ppf(1 - alpha_level, alpha, beta)

            # Calculate entropy (for Beta distribution)
            entropy = self._calculate_beta_entropy(alpha, beta)

            # Create uncertainty estimate
            estimate = UncertaintyEstimate(
                estimate_id=str(uuid.uuid4()),
                uncertainty_type=UncertaintyType.EPISTEMIC,
                method=UncertaintyMethod.BAYESIAN,
                mean_estimate=posterior_mean,
                variance=posterior_var,
                std_deviation=posterior_std,
                confidence_interval=(ci_lower, ci_upper),
                confidence_level=conf_level,
                entropy=entropy,
                input_features={"prediction_mean": mean_pred, "prediction_var": var_pred},
                model_state={"alpha": alpha, "beta": beta},
                computation_time=time.time() - start_time,
            )

            return estimate

        except Exception as e:
            raise UncertaintyError(f"Bayesian uncertainty quantification failed: {str(e)}")

    def _calculate_beta_entropy(self, alpha: float, beta: float) -> float:
        """Calculate entropy of Beta distribution."""
        try:
            return stats.beta.entropy(alpha, beta)
        except Exception:
            return 0.0


class EnsembleUncertaintyQuantifier(UncertaintyQuantifier):
    """Ensemble-based uncertainty quantification."""

    def __init__(self, num_models: int = 10):
        self.num_models = num_models
        self.ensemble_predictions: Dict[str, List[float]] = {}
        self.logger = get_logger(__name__)

    def get_method_type(self) -> UncertaintyMethod:
        return UncertaintyMethod.ENSEMBLE

    def supports_uncertainty_type(self, uncertainty_type: UncertaintyType) -> bool:
        return uncertainty_type in [
            UncertaintyType.MODEL,
            UncertaintyType.EPISTEMIC,
            UncertaintyType.ALEATORIC,
        ]

    async def quantify(
        self, input_data: Any, model_predictions: Any, context: UncertaintyContext
    ) -> UncertaintyEstimate:
        """Quantify ensemble uncertainty."""
        start_time = time.time()

        try:
            # Simulate ensemble predictions if not provided
            if not isinstance(model_predictions, list):
                # Generate ensemble predictions with noise
                base_pred = float(model_predictions)
                ensemble_preds = [
                    base_pred + np.random.normal(0, 0.1) for _ in range(self.num_models)
                ]
            else:
                ensemble_preds = list(model_predictions)

            # Calculate ensemble statistics
            mean_pred = statistics.mean(ensemble_preds)
            var_pred = statistics.variance(ensemble_preds) if len(ensemble_preds) > 1 else 0.0
            std_pred = math.sqrt(var_pred)

            # Confidence interval using t-distribution
            conf_level = context.metadata.get("confidence_level", 0.95)
            alpha_level = (1 - conf_level) / 2

            if len(ensemble_preds) > 1:
                t_val = stats.t.ppf(1 - alpha_level, df=len(ensemble_preds) - 1)
                margin = t_val * std_pred / math.sqrt(len(ensemble_preds))
                ci_lower = mean_pred - margin
                ci_upper = mean_pred + margin
            else:
                ci_lower = ci_upper = mean_pred

            # Calculate entropy based on prediction variance
            entropy = 0.5 * math.log(2 * math.pi * math.e * var_pred) if var_pred > 0 else 0.0

            # Calculate sharpness (confidence)
            sharpness = 1.0 / (1.0 + std_pred) if std_pred > 0 else 1.0

            estimate = UncertaintyEstimate(
                estimate_id=str(uuid.uuid4()),
                uncertainty_type=UncertaintyType.MODEL,
                method=UncertaintyMethod.ENSEMBLE,
                mean_estimate=mean_pred,
                variance=var_pred,
                std_deviation=std_pred,
                confidence_interval=(ci_lower, ci_upper),
                confidence_level=conf_level,
                entropy=entropy,
                sharpness=sharpness,
                input_features={"ensemble_size": len(ensemble_preds)},
                model_state={"predictions": ensemble_preds},
                computation_time=time.time() - start_time,
                sample_size=len(ensemble_preds),
            )

            return estimate

        except Exception as e:
            raise UncertaintyError(f"Ensemble uncertainty quantification failed: {str(e)}")


class InformationTheoreticQuantifier(UncertaintyQuantifier):
    """Information-theoretic uncertainty quantification."""

    def __init__(self):
        self.logger = get_logger(__name__)

    def get_method_type(self) -> UncertaintyMethod:
        return UncertaintyMethod.INFORMATION_THEORY

    def supports_uncertainty_type(self, uncertainty_type: UncertaintyType) -> bool:
        return uncertainty_type in [
            UncertaintyType.EPISTEMIC,
            UncertaintyType.DATA,
            UncertaintyType.CONTEXTUAL,
        ]

    async def quantify(
        self, input_data: Any, model_predictions: Any, context: UncertaintyContext
    ) -> UncertaintyEstimate:
        """Quantify information-theoretic uncertainty."""
        start_time = time.time()

        try:
            # Convert predictions to probability distribution
            if isinstance(model_predictions, (list, np.ndarray)):
                probs = np.array(model_predictions)
                if np.sum(probs) > 0:
                    probs = probs / np.sum(probs)  # Normalize
                else:
                    probs = np.ones_like(probs) / len(probs)  # Uniform if all zeros
            else:
                # Binary case
                p = float(model_predictions)
                probs = np.array([1 - p, p])

            # Calculate entropy
            entropy = -np.sum(
                probs * np.log(probs + 1e-12)
            )  # Add small epsilon for numerical stability

            # Calculate predictive entropy (uncertainty)
            max_entropy = math.log(len(probs))  # Maximum possible entropy
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

            # Calculate mutual information approximation
            # This would require more sophisticated implementation in practice
            mutual_info = entropy * 0.1  # Simplified approximation

            # Confidence based on entropy
            confidence = 1.0 - normalized_entropy

            # Estimate variance from entropy
            estimated_var = normalized_entropy * 0.25  # Heuristic mapping
            estimated_std = math.sqrt(estimated_var)

            # Confidence interval based on entropy
            conf_level = context.metadata.get("confidence_level", 0.95)
            margin = stats.norm.ppf((1 + conf_level) / 2) * estimated_std

            mean_pred = np.dot(probs, range(len(probs))) / len(probs)  # Weighted average
            ci_lower = max(0, mean_pred - margin)
            ci_upper = min(1, mean_pred + margin)

            estimate = UncertaintyEstimate(
                estimate_id=str(uuid.uuid4()),
                uncertainty_type=UncertaintyType.EPISTEMIC,
                method=UncertaintyMethod.INFORMATION_THEORY,
                mean_estimate=mean_pred,
                variance=estimated_var,
                std_deviation=estimated_std,
                confidence_interval=(ci_lower, ci_upper),
                confidence_level=conf_level,
                entropy=entropy,
                mutual_information=mutual_info,
                input_features={"probability_distribution": probs.tolist()},
                model_state={"normalized_entropy": normalized_entropy},
                computation_time=time.time() - start_time,
            )

            return estimate

        except Exception as e:
            raise UncertaintyError(
                f"Information-theoretic uncertainty quantification failed: {str(e)}"
            )


class UncertaintyCalibrator:
    """Calibrates uncertainty estimates using historical data."""

    def __init__(self):
        self.calibration_data: Dict[str, List[Tuple[float, float, bool]]] = defaultdict(list)
        self.calibration_curves: Dict[str, Callable] = {}
        self.logger = get_logger(__name__)

    async def add_calibration_point(
        self,
        method: UncertaintyMethod,
        confidence: float,
        uncertainty: float,
        outcome_correct: bool,
    ) -> None:
        """Add a calibration data point."""
        self.calibration_data[method.value].append((confidence, uncertainty, outcome_correct))

    async def calibrate_method(self, method: UncertaintyMethod) -> Dict[str, float]:
        """Calibrate a specific uncertainty method."""
        if method.value not in self.calibration_data:
            return {"calibration_error": 0.0, "reliability": 1.0}

        data = self.calibration_data[method.value]
        if len(data) < 10:  # Need sufficient data
            return {"calibration_error": 0.0, "reliability": 1.0}

        # Calculate calibration error
        calibration_error = self._calculate_calibration_error(data)

        # Calculate reliability
        reliability = self._calculate_reliability(data)

        return {
            "calibration_error": calibration_error,
            "reliability": reliability,
            "data_points": len(data),
        }

    def _calculate_calibration_error(self, data: List[Tuple[float, float, bool]]) -> float:
        """Calculate calibration error using reliability diagram."""
        try:
            # Bin the data by confidence levels
            bins = np.linspace(0, 1, 11)  # 10 bins
            bin_indices = np.digitize([d[0] for d in data], bins) - 1

            calibration_error = 0.0
            total_points = len(data)

            for bin_idx in range(len(bins) - 1):
                bin_data = [data[i] for i in range(len(data)) if bin_indices[i] == bin_idx]

                if not bin_data:
                    continue

                # Average confidence in this bin
                avg_confidence = sum(d[0] for d in bin_data) / len(bin_data)

                # Actual accuracy in this bin
                accuracy = sum(1 for d in bin_data if d[2]) / len(bin_data)

                # Weight by bin size
                weight = len(bin_data) / total_points

                # Add to calibration error
                calibration_error += weight * abs(avg_confidence - accuracy)

            return calibration_error

        except Exception as e:
            self.logger.warning(f"Failed to calculate calibration error: {str(e)}")
            return 0.0

    def _calculate_reliability(self, data: List[Tuple[float, float, bool]]) -> float:
        """Calculate reliability score."""
        try:
            if not data:
                return 1.0

            # Simple reliability based on consistency
            correct_predictions = sum(1 for d in data if d[2])
            reliability = correct_predictions / len(data)

            return reliability

        except Exception as e:
            self.logger.warning(f"Failed to calculate reliability: {str(e)}")
            return 1.0


class UncertaintyAggregator:
    """Aggregates uncertainty estimates from multiple sources."""

    def __init__(self):
        self.aggregation_strategies = {
            "weighted_average": self._weighted_average,
            "maximum": self._maximum_uncertainty,
            "minimum": self._minimum_uncertainty,
            "bayesian_fusion": self._bayesian_fusion,
        }
        self.logger = get_logger(__name__)

    async def aggregate(
        self,
        estimates: List[UncertaintyEstimate],
        strategy: str = "weighted_average",
        weights: Optional[List[float]] = None,
    ) -> UncertaintyEstimate:
        """Aggregate multiple uncertainty estimates."""
        if not estimates:
            raise UncertaintyError("No estimates provided for aggregation")

        if len(estimates) == 1:
            return estimates[0]

        if strategy not in self.aggregation_strategies:
            strategy = "weighted_average"

        aggregator = self.aggregation_strategies[strategy]
        return await aggregator(estimates, weights)

    async def _weighted_average(
        self, estimates: List[UncertaintyEstimate], weights: Optional[List[float]] = None
    ) -> UncertaintyEstimate:
        """Aggregate using weighted average."""
        if weights is None:
            weights = [1.0] * len(estimates)

        if len(weights) != len(estimates):
            weights = [1.0] * len(estimates)

        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            weights = [1.0 / len(estimates)] * len(estimates)

        # Weighted average of estimates
        agg_mean = sum(w * est.mean_estimate for w, est in zip(weights, estimates))
        agg_var = sum(w * est.variance for w, est in zip(weights, estimates))
        agg_std = math.sqrt(agg_var)

        # Aggregate confidence interval
        ci_lower = sum(w * est.confidence_interval[0] for w, est in zip(weights, estimates))
        ci_upper = sum(w * est.confidence_interval[1] for w, est in zip(weights, estimates))

        # Average entropy
        entropies = [est.entropy for est in estimates if est.entropy is not None]
        agg_entropy = sum(entropies) / len(entropies) if entropies else None

        return UncertaintyEstimate(
            estimate_id=str(uuid.uuid4()),
            uncertainty_type=UncertaintyType.MODEL,  # Aggregated type
            method=UncertaintyMethod.ENSEMBLE,  # Aggregation method
            mean_estimate=agg_mean,
            variance=agg_var,
            std_deviation=agg_std,
            confidence_interval=(ci_lower, ci_upper),
            confidence_level=estimates[0].confidence_level,
            entropy=agg_entropy,
            input_features={"aggregated_from": len(estimates)},
            model_state={"weights": weights, "methods": [est.method.value for est in estimates]},
        )

    async def _maximum_uncertainty(
        self, estimates: List[UncertaintyEstimate], weights: Optional[List[float]] = None
    ) -> UncertaintyEstimate:
        """Take maximum uncertainty (most conservative)."""
        max_est = max(estimates, key=lambda x: x.std_deviation)
        max_est.estimate_id = str(uuid.uuid4())
        max_est.model_state = {"aggregation": "maximum", "source_count": len(estimates)}
        return max_est

    async def _minimum_uncertainty(
        self, estimates: List[UncertaintyEstimate], weights: Optional[List[float]] = None
    ) -> UncertaintyEstimate:
        """Take minimum uncertainty (most optimistic)."""
        min_est = min(estimates, key=lambda x: x.std_deviation)
        min_est.estimate_id = str(uuid.uuid4())
        min_est.model_state = {"aggregation": "minimum", "source_count": len(estimates)}
        return min_est

    async def _bayesian_fusion(
        self, estimates: List[UncertaintyEstimate], weights: Optional[List[float]] = None
    ) -> UncertaintyEstimate:
        """Bayesian fusion of uncertainty estimates."""
        # Simplified Bayesian fusion
        # In practice, this would be more sophisticated

        precisions = [1.0 / max(est.variance, 1e-6) for est in estimates]
        weighted_means = [est.mean_estimate * prec for est, prec in zip(estimates, precisions)]

        total_precision = sum(precisions)
        fused_mean = sum(weighted_means) / total_precision if total_precision > 0 else 0.0
        fused_var = 1.0 / total_precision if total_precision > 0 else 1.0
        fused_std = math.sqrt(fused_var)

        # Confidence interval
        margin = stats.norm.ppf(0.975) * fused_std  # 95% CI
        ci_lower = fused_mean - margin
        ci_upper = fused_mean + margin

        return UncertaintyEstimate(
            estimate_id=str(uuid.uuid4()),
            uncertainty_type=UncertaintyType.MODEL,
            method=UncertaintyMethod.BAYESIAN,
            mean_estimate=fused_mean,
            variance=fused_var,
            std_deviation=fused_std,
            confidence_interval=(ci_lower, ci_upper),
            confidence_level=0.95,
            input_features={"bayesian_fusion": True},
            model_state={"precisions": precisions, "source_count": len(estimates)},
        )


class EnhancedUncertaintyHandler:
    """
    Advanced Uncertainty Handling System for the AI Assistant.

    This handler provides comprehensive uncertainty quantification and management including:
    - Multiple uncertainty estimation methods (Bayesian, ensemble, information-theoretic)
    - Uncertainty calibration and validation
    - Context-aware uncertainty assessment
    - Integration with memory, learning, and decision-making systems
    - Real-time uncertainty monitoring and adaptation
    - Probabilistic reasoning and confidence scoring
    - Uncertainty-aware decision support
    - Event-driven uncertainty management
    - Performance optimization and caching
    """

    def __init__(self, container: Container):
        """
        Initialize the enhanced uncertainty handler.

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

        # Memory systems
        self.memory_manager = container.get(MemoryManager)
        self.context_manager = container.get(ContextManager)
        self.working_memory = container.get(WorkingMemory)
        self.episodic_memory = container.get(EpisodicMemory)
        self.semantic_memory = container.get(SemanticMemory)

        # Learning systems
        self.continual_learner = container.get(ContinualLearner)
        self.preference_learner = container.get(PreferenceLearner)
        self.feedback_processor = container.get(FeedbackProcessor)
        self.model_adapter = container.get(ModelAdapter)

        # Observability
        self.metrics = container.get(MetricsCollector)
        self.tracer = container.get(TraceManager)

        # Uncertainty quantification components
        self.quantifiers: Dict[UncertaintyMethod, UncertaintyQuantifier] = {}
        self.calibrator = UncertaintyCalibrator()
        self.aggregator = UncertaintyAggregator()

        # State management
        self.uncertainty_cache: Dict[str, UncertaintyEstimate] = {}
        self.calibration_history: deque = deque(maxlen=10000)
        self.uncertainty_thresholds: Dict[str, float] = {}

        # Performance tracking
        self.estimation_times: deque = deque(maxlen=1000)
        self.accuracy_history: deque = deque(maxlen=1000)

        # Configuration
        self.default_confidence_level = self.config.get(
            "uncertainty.default_confidence_level", 0.95
        )
        self.cache_ttl_seconds = self.config.get("uncertainty.cache_ttl", 300)
        self.enable_calibration = self.config.get("uncertainty.enable_calibration", True)
        self.uncertainty_threshold = self.config.get("uncertainty.default_threshold", 0.5)
        self.enable_learning = self.config.get("uncertainty.enable_learning", True)

        # Initialize components
        self._setup_quantifiers()
        self._setup_monitoring()
        self._setup_thresholds()

        # Register health check
        self.health_check.register_component("uncertainty_handler", self._health_check_callback)

        self.logger.info("EnhancedUncertaintyHandler initialized successfully")

    def _setup_quantifiers(self) -> None:
        """Setup uncertainty quantification methods."""
        try:
            # Initialize quantifiers
            self.quantifiers[UncertaintyMethod.BAYESIAN] = BayesianUncertaintyQuantifier()
            self.quantifiers[UncertaintyMethod.ENSEMBLE] = EnsembleUncertaintyQuantifier()
            self.quantifiers[UncertaintyMethod.INFORMATION_THEORY] = (
                InformationTheoreticQuantifier()
            )

            self.logger.info(f"Initialized {len(self.quantifiers)} uncertainty quantifiers")

        except Exception as e:
            self.logger.error(f"Failed to setup quantifiers: {str(e)}")

    def _setup_monitoring(self) -> None:
        """Setup monitoring and metrics."""
        try:
            # Register uncertainty metrics
            self.metrics.register_counter("uncertainty_estimates_total")
            self.metrics.register_histogram("uncertainty_estimation_duration_seconds")
            self.metrics.register_gauge("uncertainty_accuracy")
            self.metrics.register_counter("uncertainty_threshold_exceeded")
            self.metrics.register_gauge("uncertainty_calibration_error")
            self.metrics.register_histogram("uncertainty_confidence_scores")

        except Exception as e:
            self.logger.warning(f"Failed to setup monitoring: {str(e)}")

    def _setup_thresholds(self) -> None:
        """Setup uncertainty thresholds for different contexts."""
        self.uncertainty_thresholds = {
            "critical_decision": 0.1,  # Very low uncertainty required
            "important_decision": 0.3,  # Low uncertainty required
            "routine_decision": 0.5,  # Medium uncertainty acceptable
            "exploratory": 0.8,  # High uncertainty acceptable
            "learning": 1.0,  # Any uncertainty acceptable
        }

    async def initialize(self) -> None:
        """Initialize the uncertainty handler."""
        try:
            # Initialize quantifiers
            for quantifier in self.quantifiers.values():
                if hasattr(quantifier, "initialize"):
                    await quantifier.initialize()

            # Start background tasks
            asyncio.create_task(self._calibration_update_loop())
            asyncio.create_task(self._cache_cleanup_loop())

            if self.enable_learning:
                asyncio.create_task(self._learning_update_loop())

            # Register event handlers
            await self._register_event_handlers()

            self.logger.info("UncertaintyHandler initialization completed")

        except Exception as e:
            self.logger.error(f"Failed to initialize UncertaintyHandler: {str(e)}")
            raise UncertaintyError(f"Initialization failed: {str(e)}")

    async def _register_event_handlers(self) -> None:
        """Register event handlers for system events."""
        # Learning events
        self.event_bus.subscribe("learning_event_occurred", self._handle_learning_event)

        # Feedback events
        self.event_bus.subscribe("feedback_received", self._handle_feedback)

        # Decision events
        self.event_bus.subscribe("decision_made", self._handle_decision_feedback)

        # Component health events
        self.event_bus.subscribe("component_health_changed", self._handle_component_health_change)

    @handle_exceptions
    async def quantify_uncertainty(
        self,
        input_data: Any,
        model_predictions: Any,
        context: Optional[UncertaintyContext] = None,
        methods: Optional[List[UncertaintyMethod]] = None,
        uncertainty_types: Optional[List[UncertaintyType]] = None,
    ) -> List[UncertaintyEstimate]:
        """
        Quantify uncertainty using specified methods.

        Args:
            input_data: Input data for uncertainty quantification
            model_predictions: Model predictions or outputs
            context: Uncertainty context information
            methods: Specific quantification methods to use
            uncertainty_types: Types of uncertainty to quantify

        Returns:
            List of uncertainty estimates
        """
        start_time = time.time()

        # Create default context if not provided
        if context is None:
            context = UncertaintyContext(context_id=str(uuid.uuid4()))

        # Use all available methods if not specified
        if methods is None:
            methods = list(self.quantifiers.keys())

        # Use all uncertainty types if not specified
        if uncertainty_types is None:
            uncertainty_types = [UncertaintyType.EPISTEMIC, UncertaintyType.MODEL]

        estimates = []

        try:
            with self.tracer.trace("uncertainty_quantification") as span:
                span.set_attributes(
                    {
                        "context_id": context.context_id,
                        "session_id": context.session_id or "unknown",
                        "methods_count": len(methods),
                        "uncertainty_types_count": len(uncertainty_types),
                    }
                )

                # Check cache first
                cache_key = self._generate_cache_key(input_data, model_predictions, methods)
                cached_estimates = self._get_cached_estimates(cache_key)
                if cached_estimates:
                    self.logger.debug(f"Using cached uncertainty estimates for key: {cache_key}")
                    return cached_estimates

                # Quantify uncertainty using each method
                for method in methods:
                    if method not in self.quantifiers:
                        self.logger.warning(f"Quantifier for method {method} not available")
                        continue

                    quantifier = self.quantifiers[method]

                    # Check if quantifier supports any of the requested uncertainty types
                    supported_types = [
                        utype
                        for utype in uncertainty_types
                        if quantifier.supports_uncertainty_type(utype)
                    ]

                    if not supported_types:
                        continue

                    try:
                        # Quantify uncertainty
                        estimate = await quantifier.quantify(input_data, model_predictions, context)

                        # Apply calibration if available
                        if self.enable_calibration:
                            await self._apply_calibration(estimate, method)

                        estimates.append(estimate)

                        # Emit quantification event
                        await self.event_bus.emit(
                            UncertaintyQuantified(
                                estimate_id=estimate.estimate_id,
                                method=method.value,
                                uncertainty_type=estimate.uncertainty_type.value,
                                confidence_score=1.0 - estimate.std_deviation,
                                context_id=context.context_id,
                            )
                        )

                    except Exception as e:
                        self.logger.error(
                            f"Uncertainty quantification failed for method {method}: {str(e)}"
                        )
                        continue

                # Cache results
                if estimates:
                    self._cache_estimates(cache_key, estimates)

                # Update metrics
                processing_time = time.time() - start_time
                self.estimation_times.append(processing_time)

                self.metrics.increment("uncertainty_estimates_total")
                self.metrics.record("uncertainty_estimation_duration_seconds", processing_time)

                # Check uncertainty thresholds
                await self._check_uncertainty_thresholds(estimates, context)

                self.logger.debug(
                    f"Quantified uncertainty using {len(estimates)} methods in {processing_time:.3f}s"
                )

                return estimates

        except Exception as e:
            self.logger.error(f"Uncertainty quantification failed: {str(e)}")
            raise UncertaintyError(f"Uncertainty quantification failed: {str(e)}")

    async def _apply_calibration(
        self, estimate: UncertaintyEstimate, method: UncertaintyMethod
    ) -> None:
        """Apply calibration to uncertainty estimate."""
        try:
            calibration_info = await self.calibrator.calibrate_method(method)

            if "calibration_error" in calibration_info:
                # Adjust confidence based on calibration error
                calibration_error = calibration_info["calibration_error"]
                estimate.calibration_score = 1.0 - calibration_error
                estimate.is_calibrated = calibration_error < 0.1  # Threshold for well-calibrated

                # Adjust variance based on calibration
                if calibration_error > 0.1:
                    # Increase uncertainty if poorly calibrated
                    estimate.variance *= 1.0 + calibration_error
                    estimate.std_deviation = math.sqrt(estimate.variance)

            if "reliability" in calibration_info:
                estimate.reliability = calibration_info["reliability"]
                estimate.is_reliable = calibration_info["reliability"] > 0.7

        except Exception as e:
            self.logger.warning(f"Failed to apply calibration: {str(e)}")

    def _generate_cache_key(
        self, input_data: Any, model_predictions: Any, methods: List[UncertaintyMethod]
    ) -> str:
        """Generate cache key for uncertainty estimates."""
        try:
            # Create a hash of the inputs
            key_data = {
                "input_hash": hashlib.md5(str(input_data).encode()).hexdigest()[:16],
                "prediction_hash": hashlib.md5(str(model_predictions).encode()).hexdigest()[:16],
                "methods": sorted([m.value for m in methods]),
            }

            return hashlib.md5(json.dumps(key_data, sort_keys=True).encode()).hexdigest()

        except Exception:
            return str(uuid.uuid4())

    def _get_cached_estimates(self, cache_key: str) -> Optional[List[UncertaintyEstimate]]:
        """Get cached uncertainty estimates."""
        if cache_key in self.uncertainty_cache:
            estimate = self.uncertainty_cache[cache_key]

            # Check if cache is still valid
            cache_age = (datetime.now(timezone.utc) - estimate.timestamp).total_seconds()
            if cache_age < self.cache_ttl_seconds:
                return [estimate]  # Return as list for consistency
            else:
                del self.uncertainty_cache[cache_key]

        return None

    def _cache_estimates(self, cache_key: str, estimates: List[UncertaintyEstimate]) -> None:
        """Cache uncertainty estimates."""
        if estimates:
            # Cache the first estimate (or aggregate if multiple)
            self.uncertainty_cache[cache_key] = estimates[0]

    async def _check_uncertainty_thresholds(
        self, estimates: List[UncertaintyEstimate], context: UncertaintyContext
    ) -> None:
        """Check if uncertainty exceeds thresholds."""
        for estimate in estimates:
            # Determine appropriate threshold
            decision_importance = context.decision_importance
            if decision_importance > 0.8:
                threshold_key = "critical_decision"
            elif decision_importance > 0.6:
                threshold_key = "important_decision"
            else:
                threshold_key = "routine_decision"

            threshold = self.uncertainty_thresholds.get(threshold_key, self.uncertainty_threshold)

            # Check if uncertainty exceeds threshold
            if estimate.std_deviation > threshold:
                await self.event_bus.emit(
                    UncertaintyThresholdExceeded(
                        estimate_id=estimate.estimate_id,
                        threshold=threshold,
                        actual_uncertainty=estimate.std_deviation,
                        context_id=context.context_id,
                        decision_importance=decision_importance,
                    )
                )

                self.metrics.increment("uncertainty_threshold_exceeded")

    @handle_exceptions
    async def aggregate_uncertainties(
        self,
        estimates: List[UncertaintyEstimate],
        strategy: str = "weighted_average",
        weights: Optional[List[float]] = None,
    ) -> UncertaintyEstimate:
        """
        Aggregate multiple uncertainty estimates.

        Args:
            estimates: List of uncertainty estimates to aggregate
            strategy: Aggregation strategy
            weights: Optional weights for aggregation

        Returns:
            Aggregated uncertainty estimate
        """
        try:
            aggregated = await self.aggregator.aggregate(estimates, strategy, weights)

            # Store aggregation in memory for learning
            if self.enable_learning:
                aggregation_data = {
                    "aggregated_estimate": asdict(aggregated),
                    "source_estimates": [asdict(est) for est in estimates],
                    "strategy": strategy,
                    "weights": weights,
                    "timestamp": datetime.now(timezone.utc),
                }

                await self.episodic_memory.store(aggregation_data)

            return aggregated

        except Exception as e:
            raise UncertaintyError(f"Uncertainty aggregation failed: {str(e)}")

    @handle_exceptions
    async def calculate_confidence_score(
        self, estimate: UncertaintyEstimate, context: Optional[UncertaintyContext] = None
    ) -> float:
        """
        Calculate confidence score from uncertainty estimate.

        Args:
            estimate: Uncertainty estimate
            context: Optional context information

        Returns:
            Confidence score (0-1)
        """
        try:
            # Base confidence from uncertainty
            base_confidence = 1.0 / (1.0 + estimate.std_deviation)

            # Adjust for calibration
            calibration_factor = estimate.calibration_score or 1.0
            confidence = base_confidence * calibration_factor

            # Adjust for reliability
            reliability_factor = estimate.reliability or 1.0
            confidence *= reliability_factor

            # Context-based adjustments
            if context:
                # Adjust for decision importance
                importance_factor = 1.0 - (
                    context.decision_importance * 0.1
                )  # Slight penalty for important decisions
                confidence *= importance_factor

                # Adjust for time pressure
                time_factor = 1.0 - (
                    context.time_pressure * 0.05
                )  # Slight penalty for time pressure
                confidence *= time_factor

            # Ensure confidence is in [0, 1]
            confidence = max(0.0, min(1.0, confidence))

            # Emit confidence calculation event
            await self.event_bus.emit(
                ConfidenceScoreCalculated(
                    estimate_id=estimate.estimate_id,
                    confidence_score=confidence,
                    base_confidence=base_confidence,
                    adjustments={
                        "calibration": calibration_factor,
                        "reliability": reliability_factor,
                    },
                )
            )

            # Record confidence score
            self.metrics.record("uncertainty_confidence_scores", confidence)

            return confidence

        except Exception as e:
            self.logger.error(f"Confidence score calculation failed: {str(e)}")
            return 0.5  # Default moderate confidence

    @handle_exceptions
    async def make_probabilistic_decision(
        self,
        alternatives: List[Dict[str, Any]],
        uncertainty_estimates: List[UncertaintyEstimate],
        context: Optional[UncertaintyContext] = None,
        decision_criteria: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Make probabilistic decision considering uncertainty.

        Args:
            alternatives: List of decision alternatives
            uncertainty_estimates: Uncertainty estimates for each alternative
            context: Decision context
            decision_criteria: Criteria weights for decision making

        Returns:
            Decision result with probability distribution
        """
        try:
            if len(alternatives) != len(uncertainty_estimates):
                raise UncertaintyError(
                    "Number of alternatives must match number of uncertainty estimates"
                )

            # Default decision criteria
            if decision_criteria is None:
                decision_criteria = {
                    "expected_value": 0.6,
                    "uncertainty_penalty": 0.3,
                    "risk_tolerance": 0.1,
                }

            decision_scores = []

            for i, (alternative, estimate) in enumerate(zip(alternatives, uncertainty_estimates)):
                # Expected value score
                expected_value = alternative.get("value", 0.5)

                # Uncertainty penalty
                uncertainty_penalty = (
                    estimate.std_deviation * decision_criteria["uncertainty_penalty"]
                )

                # Risk adjustment
                risk_factor = context.risk_tolerance if context else 0.5
                risk_adjustment = (1.0 - risk_factor) * estimate.variance

                # Calculate total score
                score = (
                    expected_value * decision_criteria["expected_value"]
                    - uncertainty_penalty
                    - risk_adjustment * decision_criteria["risk_tolerance"]
                )

                decision_scores.append(
                    {
                        "alternative_index": i,
                        "alternative": alternative,
                        "score": score,
                        "expected_value": expected_value,
                        "uncertainty": estimate.std_deviation,
                        "confidence": await self.calculate_confidence_score(estimate, context),
                    }
                )

            # Sort by score
            decision_scores.sort(key=lambda x: x["score"], reverse=True)

            # Calculate probability distribution using softmax
            scores = [ds["score"] for ds in decision_scores]
            max_score = max(scores)
            exp_scores = [
                math.exp(score - max_score) for score in scores
            ]  # Subtract max for numerical stability
            sum_exp = sum(exp_scores)
            probabilities = [exp_score / sum_exp for exp_score in exp_scores]

            # Create decision result
            decision_result = {
                "recommended_alternative": decision_scores[0]["alternative"],
                "recommended_index": decision_scores[0]["alternative_index"],
                "confidence": decision_scores[0]["confidence"],
                "probability_distribution": probabilities,
                "alternatives_ranked": decision_scores,
                "decision_criteria": decision_criteria,
                "uncertainty_considered": True,
                "context_id": context.context_id if context else None,
            }

            # Emit probabilistic decision event
            await self.event_bus.emit(
                ProbabilisticDecisionMade(
                    decision_id=str(uuid.uuid4()),
                    recommended_alternative=decision_scores[0]["alternative_index"],
                    confidence=decision_scores[0]["confidence"],
                    probability_distribution=probabilities,
                    context_id=context.context_id if context else None,
                )
            )

            return decision_result

        except Exception as e:
            raise UncertaintyError(f"Probabilistic decision making failed: {str(e)}")

    @handle_exceptions
    async def update_uncertainty_model(
        self, feedback_data: Dict[str, Any], estimate_id: str
    ) -> None:
        """
        Update uncertainty model based on feedback.

        Args:
            feedback_data: Feedback information
            estimate_id: ID of the uncertainty estimate
        """
        try:
            # Find the estimate in cache or history
            estimate = None
            for cached_estimate in self.uncertainty_cache.values():
                if cached_estimate.estimate_id == estimate_id:
                    estimate = cached_estimate
                    break

            if not estimate:
                self.logger.warning(f"Estimate {estimate_id} not found for update")
                return

            # Extract feedback information
            actual_outcome = feedback_data.get("actual_outcome")
            outcome_correct = feedback_data.get("outcome_correct", True)
            user_satisfaction = feedback_data.get("user_satisfaction", 0.5)

            # Update calibration data
            if self.enable_calibration and actual_outcome is not None:
                confidence = 1.0 - estimate.std_deviation
                await self.calibrator.add_calibration_point(
                    estimate.method, confidence, estimate.std_deviation, outcome_correct
                )

            # Store feedback for learning
            feedback_entry = {
                "estimate_id": estimate_id,
                "estimate": asdict(estimate),
                "feedback": feedback_data,
                "timestamp": datetime.now(timezone.utc),
            }

            self.calibration_history.append(feedback_entry)

            # Update accuracy tracking
            if outcome_correct is not None:
                self.accuracy_history.append(1.0 if outcome_correct else 0.0)

            # Emit model update event
            await self.event_bus.emit(
                UncertaintyModelUpdated(
                    estimate_id=estimate_id,
                    method=estimate.method.value,
                    feedback_type="outcome_feedback",
                    accuracy_improvement=self._calculate_recent_accuracy(),
                )
            )

            # Learn from feedback if enabled
            if self.enable_learning:
                await self._learn_from_feedback(estimate, feedback_data)

        except Exception as e:
            self.logger.error(f"Uncertainty model update failed: {str(e)}")

    def _calculate_recent_accuracy(self) -> float:
        """Calculate recent accuracy from accuracy history."""
        if not self.accuracy_history:
            return 0.0

        recent_data = list(self.accuracy_history)[-100:]  # Last 100 points
        return sum(recent_data) / len(recent_data)

    async def _learn_from_feedback(
        self, estimate: UncertaintyEstimate, feedback_data: Dict[str, Any]
    ) -> None:
        """Learn from feedback to improve uncertainty estimation."""
        try:
            learning_data = {
                "uncertainty_estimate": asdict(estimate),
                "feedback": feedback_data,
                "accuracy": self._calculate_recent_accuracy(),
                "context": "uncertainty_feedback",
            }

            # Update continual learning
            if hasattr(self.continual_learner, "learn_from_uncertainty_feedback"):
                await self.continual_learner.learn_from_uncertainty_feedback(learning_data)

            # Update model adaptation
            if hasattr(self.model_adapter, "adapt_uncertainty_model"):
                await self.model_adapter.adapt_uncertainty_model(estimate, feedback_data)

            # Emit learning event
            await self.event_bus.emit(
                UncertaintyLearningOccurred(
                    estimate_id=estimate.estimate_id,
                    learning_type="feedback_based",
                    improvement_metric=self._calculate_recent_accuracy(),
                )
            )

        except Exception as e:
            self.logger.warning(f"Learning from uncertainty feedback failed: {str(e)}")

    async def detect_uncertainty_anomalies(
        self, estimates: List[UncertaintyEstimate], context: Optional[UncertaintyContext] = None
    ) -> List[Dict[str, Any]]:
        """
        Detect anomalies in uncertainty estimates.

        Args:
            estimates: List of uncertainty estimates
            context: Optional context information

        Returns:
            List of detected anomalies
        """
        anomalies = []

        try:
            if len(estimates) < 2:
                return anomalies

            # Calculate statistics
            uncertainties = [est.std_deviation for est in estimates]
            mean_uncertainty = statistics.mean(uncertainties)
            std_uncertainty = statistics.stdev(uncertainties) if len(uncertainties) > 1 else 0.0

            # Z-score threshold for anomaly detection
            z_threshold = 2.5

            for estimate in estimates:
                z_score = 0.0
                if std_uncertainty > 0:
                    z_score = abs(estimate.std_deviation - mean_uncertainty) / std_uncertainty

                # Check for various anomaly conditions
                anomaly_reasons = []

                # High uncertainty anomaly
                if z_score > z_threshold:
                    anomaly_reasons.append("extreme_uncertainty")

                # Calibration anomaly
                if estimate.calibration_score and estimate.calibration_score < 0.3:
                    anomaly_reasons.append("poor_calibration")

                # Reliability anomaly
                if estimate.reliability and estimate.reliability < 0.5:
                    anomaly_reasons.append("low_reliability")

                # Entropy anomaly (if available)
                if estimate.entropy and estimate.entropy > 3.0:  # High entropy threshold
                    anomaly_reasons.append("high_entropy")

                if anomaly_reasons:
                    anomaly = {
                        "estimate_id": estimate.estimate_id,
                        "anomaly_reasons": anomaly_reasons,
                        "z_score": z_score,
                        "uncertainty": estimate.std_deviation,
                        "severity": "high" if z_score > 3.0 else "medium",
                        "timestamp": datetime.now(timezone.utc),
                    }

                    anomalies.append(anomaly)
                    estimate.anomaly_score = z_score

                    # Emit anomaly detection event
                    await self.event_bus.emit(
                        UncertaintyAnomalyDetected(
                            estimate_id=estimate.estimate_id,
                            anomaly_type="|".join(anomaly_reasons),
                            severity=anomaly["severity"],
                            z_score=z_score,
                        )
                    )

            return anomalies

        except Exception as e:
            self.logger.error(f"Uncertainty anomaly detection failed: {str(e)}")
            return []

    def get_uncertainty_summary(self) -> Dict[str, Any]:
        """Get comprehensive uncertainty system summary."""
        try:
            # Calculate recent performance metrics
            recent_accuracy = self._calculate_recent_accuracy()

            avg_estimation_time = (
                sum(self.estimation_times) / len(self.estimation_times)
                if self.estimation_times
                else 0.0
            )

            # Calibration information
            calibration_info = {}
            for method in self.quantifiers.keys():
                try:
                    cal_info = asyncio.create_task(self.calibrator.calibrate_method(method))
                    calibration_info[method.value] = cal_info
                except Exception:
                    calibration_info[method.value] = {"error": "calibration_unavailable"}

            return {
                "system_status": "healthy",
                "quantifiers_available": list(self.quantifiers.keys()),
                "recent_accuracy": recent_accuracy,
                "average_estimation_time": avg_estimation_time,
                "cached_estimates": len(self.uncertainty_cache),
                "calibration_history_size": len(self.calibration_history),
                "uncertainty_thresholds": self.uncertainty_thresholds,
                "calibration_enabled": self.enable_calibration,
                "learning_enabled": self.enable_learning,
                "default_confidence_level": self.default_confidence_level,
            }

        except Exception as e:
            self.logger.error(f"Failed to generate uncertainty summary: {str(e)}")
            return {"system_status": "error", "error": str(e)}

    async def _calibration_update_loop(self) -> None:
        """Background task for calibration updates."""
        while True:
            try:
                if self.enable_calibration and len(self.calibration_history) > 0:
                    # Update calibration for each method
                    for method in self.quantifiers.keys():
                        calibration_result = await self.calibrator.calibrate_method(method)

                        # Emit calibration event
                        await self.event_bus.emit(
                            UncertaintyCalibrated(
                                method=method.value,
                                calibration_error=calibration_result.get("calibration_error", 0.0),
                                reliability=calibration_result.get("reliability", 1.0),
                                data_points=calibration_result.get("data_points", 0),
                            )
                        )

                        # Update metrics
                        self.metrics.set(
                            "uncertainty_calibration_error",
                            calibration_result.get("calibration_error", 0.0),
                            tags={"method": method.value},
                        )

                await asyncio.sleep(300)  # Update every 5 minutes

            except Exception as e:
                self.logger.error(f"Calibration update loop error: {str(e)}")
                await asyncio.sleep(300)

    async def _cache_cleanup_loop(self) -> None:
        """Background task for cache cleanup."""
        while True:
            try:
                current_time = datetime.now(timezone.utc)
                expired_keys = []

                for key, estimate in self.uncertainty_cache.items():
                    age = (current_time - estimate.timestamp).total_seconds()
                    if age > self.cache_ttl_seconds:
                        expired_keys.append(key)

                for key in expired_keys:
                    del self.uncertainty_cache[key]

                if expired_keys:
                    self.logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")

                await asyncio.sleep(60)  # Check every minute

            except Exception as e:
                self.logger.error(f"Cache cleanup loop error: {str(e)}")
                await asyncio.sleep(60)

    async def _learning_update_loop(self) -> None:
        """Background task for learning updates."""
        while True:
            try:
                if self.enable_learning and self.accuracy_history:
                    # Calculate learning metrics
                    recent_accuracy = self._calculate_recent_accuracy()

                    # Update learning systems
                    learning_data = {
                        "uncertainty_accuracy": recent_accuracy,
                        "estimation_times": list(self.estimation_times),
                        "calibration_history": list(self.calibration_history)[
                            -100:
                        ],  # Last 100 entries
                        "timestamp": datetime.now(timezone.utc),
                    }

                    # Update continual learning
                    try:
                        if hasattr(self.continual_learner, "update_uncertainty_knowledge"):
                            await self.continual_learner.update_uncertainty_knowledge(
                                uncertainty_info
                            )
                    except Exception as e:
                        self.logger.error(f"Error updating uncertainty knowledge: {e}")

                # Sleep before next update
                await asyncio.sleep(self.learning_update_interval)

            except Exception as e:
                self.logger.error(f"Learning update loop error: {str(e)}")
                await asyncio.sleep(60)
