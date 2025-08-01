"""
ModelInferenceCoordinator - Bidirectional Integration Component

This component addresses the critical integration flaw identified in the analysis
where model_router and inference_engine have unidirectional integration.

Author: Integration Analysis Response
Created: 2025-01-29
"""

import logging
import statistics
import threading
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import asyncio

from src.core.config.loader import ConfigLoader
from src.core.error_handling import ErrorHandler, handle_exceptions
from src.core.events.event_bus import EventBus
from src.core.events.event_types import (
    ComponentHealthChanged,
    InferenceCompleted,
    InferenceFailed,
    InferenceStarted,
    ModelRequestCompleted,
    ModelRequestFailed,
    ModelRequestStarted,
    ModelRouteSelected,
    PerformanceThresholdExceeded,
)
from src.observability.logging.config import get_logger


class CoordinationStrategy(Enum):
    """Strategies for coordinating model selection and inference."""

    PERFORMANCE_OPTIMIZED = "performance_optimized"
    COST_OPTIMIZED = "cost_optimized"
    BALANCED = "balanced"
    ADAPTIVE = "adaptive"


class InferenceComplexity(Enum):
    """Complexity levels for inference tasks."""

    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"
    HIGHLY_COMPLEX = "highly_complex"


@dataclass
class InferenceMetrics:
    """Metrics from inference engine performance."""

    inference_id: str
    model_id: str
    inference_type: str
    complexity: InferenceComplexity
    latency: float
    success: bool
    confidence_score: float
    memory_usage: float
    cpu_usage: float
    error_message: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class ModelPerformanceProfile:
    """Performance profile for a model on different inference types."""

    model_id: str
    inference_metrics: Dict[str, List[InferenceMetrics]] = field(default_factory=dict)
    success_rates: Dict[str, float] = field(default_factory=dict)
    avg_latencies: Dict[str, float] = field(default_factory=dict)
    complexity_scores: Dict[InferenceComplexity, float] = field(default_factory=dict)
    cost_effectiveness: float = 0.0
    last_updated: float = field(default_factory=time.time)


@dataclass
class InferenceRequest:
    """Request for coordinated inference with model selection."""

    request_id: str
    inference_type: str
    complexity: InferenceComplexity
    input_data: Any
    context: Optional[Dict] = None
    preferences: Optional[Dict] = None  # User preferences for speed vs accuracy
    max_latency: Optional[float] = None
    max_cost: Optional[float] = None


@dataclass
class CoordinatedResponse:
    """Response from coordinated inference."""

    request_id: str
    model_id: str
    inference_result: Any
    metrics: InferenceMetrics
    fallback_used: bool = False
    total_latency: float = 0.0


class ModelInferenceCoordinator:
    """
    Coordinates between ModelRouter and InferenceEngine for optimal performance.

    This component addresses the critical integration flaw where:
    - Model router couldn't receive feedback from inference performance
    - Inference engine couldn't influence model selection
    - No coordinated fallback strategies existed
    - Performance optimization was suboptimal
    """

    def __init__(
        self,
        config: ConfigLoader,
        event_bus: EventBus,
        model_router: Any = None,  # Will be injected
        inference_engine: Any = None,  # Will be injected
    ):
        self.config = config
        self.event_bus = event_bus
        self.logger = get_logger(__name__)

        # Components (will be injected via dependency injection)
        self._model_router = model_router
        self._inference_engine = inference_engine

        # Performance tracking
        self.model_profiles: Dict[str, ModelPerformanceProfile] = {}
        self.recent_requests: List[InferenceRequest] = []
        self.coordination_strategy = CoordinationStrategy(
            config.get("model_inference_coordinator.strategy", "balanced")
        )

        # Configuration
        self.max_profile_age = config.get(
            "model_inference_coordinator.max_profile_age", 3600
        )  # 1 hour
        self.min_samples_for_profile = config.get("model_inference_coordinator.min_samples", 10)
        self.performance_threshold = config.get(
            "model_inference_coordinator.performance_threshold", 0.8
        )
        self.max_retries = config.get("model_inference_coordinator.max_retries", 3)

        # Threading for async coordination
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="coordinator")
        self._lock = threading.RLock()

    def set_model_router(self, model_router: Any) -> None:
        """Inject model router dependency."""
        self._model_router = model_router
        self.logger.info("Model router injected into coordinator")

    def set_inference_engine(self, inference_engine: Any) -> None:
        """Inject inference engine dependency."""
        self._inference_engine = inference_engine
        self.logger.info("Inference engine injected into coordinator")

    async def coordinate_inference(self, request: InferenceRequest) -> CoordinatedResponse:
        """
        Main coordination method that selects optimal model and executes inference.
        """
        start_time = time.time()

        try:
            # 1. Analyze request complexity and requirements
            complexity_analysis = await self._analyze_request_complexity(request)

            # 2. Select optimal model based on performance profiles
            selected_model = await self._select_optimal_model(request, complexity_analysis)

            # 3. Execute inference with monitoring
            inference_result, metrics = await self._execute_monitored_inference(
                request, selected_model
            )

            # 4. Update performance profiles
            await self._update_model_profile(selected_model, metrics)

            # 5. Return coordinated response
            response = CoordinatedResponse(
                request_id=request.request_id,
                model_id=selected_model,
                inference_result=inference_result,
                metrics=metrics,
                total_latency=time.time() - start_time,
            )

            # Emit completion event
            await self.event_bus.emit(
                InferenceCompleted(
                    inference_id=request.request_id,
                    model_id=selected_model,
                    inference_type=request.inference_type,
                    success=True,
                    latency=response.total_latency,
                )
            )

            return response

        except Exception as e:
            self.logger.error(f"Coordination failed for request {request.request_id}: {e}")

            # Attempt fallback
            try:
                fallback_response = await self._execute_fallback(request, str(e))
                fallback_response.total_latency = time.time() - start_time
                return fallback_response
            except Exception as fallback_error:
                self.logger.error(f"Fallback also failed: {fallback_error}")
                raise

    async def _analyze_request_complexity(self, request: InferenceRequest) -> Dict[str, Any]:
        """Analyze the complexity and requirements of an inference request."""
        analysis = {
            "declared_complexity": request.complexity,
            "estimated_tokens": len(str(request.input_data)) // 4,  # Rough token estimate
            "has_context": request.context is not None,
            "context_size": len(str(request.context)) if request.context else 0,
            "requires_reasoning": request.inference_type in ["logical", "causal", "temporal"],
            "is_multimodal": isinstance(request.input_data, dict) and len(request.input_data) > 1,
        }

        # Adjust complexity based on analysis
        if analysis["requires_reasoning"] or analysis["is_multimodal"]:
            if request.complexity == InferenceComplexity.SIMPLE:
                analysis["adjusted_complexity"] = InferenceComplexity.MEDIUM
            elif request.complexity == InferenceComplexity.MEDIUM:
                analysis["adjusted_complexity"] = InferenceComplexity.COMPLEX
            else:
                analysis["adjusted_complexity"] = request.complexity
        else:
            analysis["adjusted_complexity"] = request.complexity

        return analysis

    async def _select_optimal_model(
        self, request: InferenceRequest, complexity_analysis: Dict[str, Any]
    ) -> str:
        """Select the optimal model based on request requirements and performance profiles."""

        if not self._model_router:
            raise RuntimeError("Model router not injected")

        # Get available models from router
        try:
            available_models = await self._get_available_models(request.inference_type)
        except Exception as e:
            self.logger.error(f"Failed to get available models: {e}")
            raise

        if not available_models:
            raise RuntimeError(f"No models available for inference type: {request.inference_type}")

        # Score models based on performance profiles and request requirements
        model_scores = {}
        for model_info in available_models:
            model_id = model_info.get("model_id")
            if not model_id:
                continue

            score = await self._score_model_for_request(
                model_id, model_info, request, complexity_analysis
            )
            model_scores[model_id] = score

        # Select best scoring model
        if not model_scores:
            raise RuntimeError("No suitable models found for request")

        best_model = max(model_scores.items(), key=lambda x: x[1])[0]

        self.logger.info(
            f"Selected model {best_model} for request {request.request_id} "
            f"(score: {model_scores[best_model]:.3f})"
        )

        return best_model

    async def _score_model_for_request(
        self,
        model_id: str,
        model_info: Dict,
        request: InferenceRequest,
        complexity_analysis: Dict[str, Any],
    ) -> float:
        """Score a model's suitability for a specific request."""

        base_score = 0.5  # Base score

        # Get performance profile
        profile = self.model_profiles.get(model_id)

        if profile and self._is_profile_valid(profile):
            # Score based on historical performance
            inference_type = request.inference_type

            if inference_type in profile.success_rates:
                success_rate = profile.success_rates[inference_type]
                base_score += 0.3 * success_rate

            if inference_type in profile.avg_latencies:
                # Lower latency is better (normalized score)
                avg_latency = profile.avg_latencies[inference_type]
                if request.max_latency:
                    latency_score = min(1.0, request.max_latency / max(avg_latency, 0.1))
                    base_score += 0.2 * latency_score
                else:
                    # Prefer faster models
                    latency_score = 1.0 / (1.0 + avg_latency)
                    base_score += 0.1 * latency_score

            # Score based on complexity handling
            complexity = complexity_analysis["adjusted_complexity"]
            if complexity in profile.complexity_scores:
                complexity_score = profile.complexity_scores[complexity]
                base_score += 0.2 * complexity_score

        # Score based on model capabilities from router
        capabilities = model_info.get("capabilities", [])
        if request.inference_type in capabilities:
            base_score += 0.1

        # Apply strategy-specific adjustments
        if self.coordination_strategy == CoordinationStrategy.PERFORMANCE_OPTIMIZED:
            # Prefer models with better success rates
            if profile and request.inference_type in profile.success_rates:
                base_score *= 1.0 + profile.success_rates[request.inference_type]

        elif self.coordination_strategy == CoordinationStrategy.COST_OPTIMIZED:
            # Prefer cheaper models
            cost = model_info.get("avg_cost", 0.0)
            if request.max_cost and cost > 0:
                cost_score = min(1.0, request.max_cost / cost)
                base_score *= 0.5 + 0.5 * cost_score

        elif self.coordination_strategy == CoordinationStrategy.ADAPTIVE:
            # Adapt based on recent performance
            recent_success_rate = self._get_recent_success_rate(model_id, request.inference_type)
            if recent_success_rate is not None:
                base_score *= 0.5 + 0.5 * recent_success_rate

        return min(1.0, base_score)  # Cap at 1.0

    async def _execute_monitored_inference(
        self, request: InferenceRequest, model_id: str
    ) -> Tuple[Any, InferenceMetrics]:
        """Execute inference with comprehensive monitoring."""

        if not self._inference_engine:
            raise RuntimeError("Inference engine not injected")

        start_time = time.time()

        # Create inference context with model selection
        inference_context = {
            "model_id": model_id,
            "inference_type": request.inference_type,
            "complexity": request.complexity,
            "coordinator_request_id": request.request_id,
        }

        # Merge with request context
        if request.context:
            inference_context.update(request.context)

        try:
            # Emit inference started event
            await self.event_bus.emit(
                InferenceStarted(
                    inference_id=request.request_id,
                    inference_type=request.inference_type,
                    model_id=model_id,
                )
            )

            # Execute inference (this would call the actual inference engine)
            result = await self._call_inference_engine(request.input_data, inference_context)

            # Calculate metrics
            latency = time.time() - start_time

            metrics = InferenceMetrics(
                inference_id=request.request_id,
                model_id=model_id,
                inference_type=request.inference_type,
                complexity=request.complexity,
                latency=latency,
                success=True,
                confidence_score=self._extract_confidence(result),
                memory_usage=0.0,  # Would be measured in real implementation
                cpu_usage=0.0,  # Would be measured in real implementation
            )

            return result, metrics

        except Exception as e:
            latency = time.time() - start_time

            metrics = InferenceMetrics(
                inference_id=request.request_id,
                model_id=model_id,
                inference_type=request.inference_type,
                complexity=request.complexity,
                latency=latency,
                success=False,
                confidence_score=0.0,
                memory_usage=0.0,
                cpu_usage=0.0,
                error_message=str(e),
            )

            # Emit inference failed event
            await self.event_bus.emit(
                InferenceFailed(
                    inference_id=request.request_id,
                    inference_type=request.inference_type,
                    model_id=model_id,
                    error_message=str(e),
                )
            )

            raise

    async def _call_inference_engine(self, input_data: Any, context: Dict) -> Any:
        """Call the inference engine with model-specific context."""
        # This would be the actual call to inference engine
        # For now, we'll create a placeholder that would be replaced with real integration

        if hasattr(self._inference_engine, "infer"):
            return await self._inference_engine.infer(input_data, context)
        elif hasattr(self._inference_engine, "process"):
            return await self._inference_engine.process(input_data, context)
        else:
            # Fallback for testing
            await asyncio.sleep(0.1)  # Simulate processing time
            return {"result": "coordinated_inference_result", "context": context}

    def _extract_confidence(self, result: Any) -> float:
        """Extract confidence score from inference result."""
        if isinstance(result, dict):
            return result.get("confidence", 0.5)
        return 0.5  # Default confidence

    async def _update_model_profile(self, model_id: str, metrics: InferenceMetrics) -> None:
        """Update performance profile for a model."""
        with self._lock:
            if model_id not in self.model_profiles:
                self.model_profiles[model_id] = ModelPerformanceProfile(model_id=model_id)

            profile = self.model_profiles[model_id]
            inference_type = metrics.inference_type

            # Add metrics to history
            if inference_type not in profile.inference_metrics:
                profile.inference_metrics[inference_type] = []

            profile.inference_metrics[inference_type].append(metrics)

            # Keep only recent metrics (last 100)
            if len(profile.inference_metrics[inference_type]) > 100:
                profile.inference_metrics[inference_type].pop(0)

            # Update aggregated metrics
            recent_metrics = profile.inference_metrics[inference_type]
            if recent_metrics:
                successes = [m for m in recent_metrics if m.success]
                profile.success_rates[inference_type] = len(successes) / len(recent_metrics)

                if successes:
                    profile.avg_latencies[inference_type] = statistics.mean(
                        m.latency for m in successes
                    )

                # Update complexity scores
                complexity_metrics = [
                    m for m in recent_metrics if m.complexity == metrics.complexity
                ]
                if complexity_metrics:
                    complexity_successes = [m for m in complexity_metrics if m.success]
                    profile.complexity_scores[metrics.complexity] = len(complexity_successes) / len(
                        complexity_metrics
                    )

            profile.last_updated = time.time()

    async def _get_available_models(self, inference_type: str) -> List[Dict]:
        """Get available models from the model router."""
        if hasattr(self._model_router, "get_available_models"):
            return await self._model_router.get_available_models(capabilities=[inference_type])
        else:
            # Fallback for testing
            return [
                {
                    "model_id": "default_model",
                    "capabilities": [inference_type],
                    "avg_cost": 0.01,
                    "success_rate": 0.9,
                }
            ]

    def _is_profile_valid(self, profile: ModelPerformanceProfile) -> bool:
        """Check if a performance profile is valid and recent enough."""
        age = time.time() - profile.last_updated
        return age < self.max_profile_age

    def _get_recent_success_rate(self, model_id: str, inference_type: str) -> Optional[float]:
        """Get recent success rate for adaptive strategy."""
        profile = self.model_profiles.get(model_id)
        if not profile or inference_type not in profile.inference_metrics:
            return None

        # Get last 10 metrics
        recent_metrics = profile.inference_metrics[inference_type][-10:]
        if len(recent_metrics) < 3:  # Need minimum samples
            return None

        successes = sum(1 for m in recent_metrics if m.success)
        return successes / len(recent_metrics)

    async def _execute_fallback(
        self, request: InferenceRequest, error_msg: str
    ) -> CoordinatedResponse:
        """Execute fallback strategy when primary coordination fails."""
        self.logger.warning(f"Executing fallback for request {request.request_id}: {error_msg}")

        try:
            # Try with simplest available model
            fallback_models = await self._get_available_models("basic")
            if fallback_models:
                fallback_model = fallback_models[0]["model_id"]

                # Simplified inference call
                result, metrics = await self._execute_monitored_inference(request, fallback_model)

                return CoordinatedResponse(
                    request_id=request.request_id,
                    model_id=fallback_model,
                    inference_result=result,
                    metrics=metrics,
                    fallback_used=True,
                )
        except Exception as e:
            self.logger.error(f"Fallback execution failed: {e}")

        # Ultimate fallback - return error response
        raise RuntimeError(f"All coordination and fallback strategies failed: {error_msg}")

    async def get_coordination_metrics(self) -> Dict[str, Any]:
        """Get coordination performance metrics."""
        with self._lock:
            total_models = len(self.model_profiles)

            # Calculate overall performance
            total_inferences = 0
            total_successes = 0

            for profile in self.model_profiles.values():
                for metrics_list in profile.inference_metrics.values():
                    total_inferences += len(metrics_list)
                    total_successes += sum(1 for m in metrics_list if m.success)

            overall_success_rate = total_successes / max(total_inferences, 1)

            return {
                "total_models_tracked": total_models,
                "total_inferences_coordinated": total_inferences,
                "overall_success_rate": overall_success_rate,
                "coordination_strategy": self.coordination_strategy.value,
                "active_profiles": len(
                    [p for p in self.model_profiles.values() if self._is_profile_valid(p)]
                ),
            }

    async def shutdown(self) -> None:
        """Shutdown coordinator and cleanup resources."""
        self.logger.info("Shutting down ModelInferenceCoordinator")
        self._executor.shutdown(wait=True)
        self.model_profiles.clear()
        self.recent_requests.clear()
