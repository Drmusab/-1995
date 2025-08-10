"""
Advanced Model Router for AI Assistant
Author: Drmusab
Last Modified: 2025-01-13 18:39:01 UTC

This module provides comprehensive model routing and management for the AI assistant,
including intelligent provider selection, load balancing, failover management,
and cost optimization across multiple LLM providers.
"""

import hashlib
import json
import logging
import random
import threading
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Set, TypeVar, Union

import aiohttp
import asyncio
import numpy as np

# Core imports
from src.core.config.loader import ConfigLoader
from src.core.dependency_injection import Container
from src.core.error_handling import ErrorHandler, handle_exceptions
from src.core.events.event_bus import EventBus
from src.core.events.event_types import (
    ComponentHealthChanged,
    ModelCacheHit,
    ModelCacheMiss,
    ModelRateLimitExceeded,
    ModelRequestCompleted,
    ModelRequestFailed,
    ModelRequestStarted,
    ModelRouteSelected,
)
from src.core.health_check import HealthCheck
from src.core.security.authentication import AuthenticationManager
from src.core.security.authorization import AuthorizationManager
from src.core.security.sanitization import InputSanitizer

# Cache and storage
from src.integrations.cache.cache_strategy import CacheStrategy
from src.integrations.cache.redis_cache import RedisCache

# LLM Provider imports
from src.integrations.llm.base_provider import BaseLLMProvider, ModelResponse
from src.integrations.llm.deepseek import DeepSeekProvider
from src.integrations.llm.ollama import OllamaProvider
from src.integrations.llm.openai import OpenAIProvider
from src.integrations.storage.database import DatabaseManager

# Learning and adaptation
from src.learning.continual_learning import ContinualLearner
from src.learning.feedback_processor import FeedbackProcessor
from src.learning.preference_learning import PreferenceLearner

# Memory systems
from src.memory.core_memory.memory_manager import MemoryManager
from src.memory.operations.context_manager import ContextManager
from src.observability.logging.config import get_logger

# Observability
from src.observability.monitoring.metrics import MetricsCollector
from src.observability.monitoring.tracing import TraceManager

# Type definitions
T = TypeVar("T")


class RoutingStrategy(Enum):
    """Available routing strategies."""

    ROUND_ROBIN = "round_robin"
    LEAST_LATENCY = "least_latency"
    LEAST_COST = "least_cost"
    BEST_PERFORMANCE = "best_performance"
    LOAD_BALANCED = "load_balanced"
    CONTEXT_AWARE = "context_aware"
    CAPABILITY_BASED = "capability_based"
    RANDOM = "random"
    PRIORITY_BASED = "priority_based"
    ADAPTIVE = "adaptive"


class ModelCapability(Enum):
    """Model capabilities for routing decisions."""

    TEXT_GENERATION = "text_generation"
    CODE_GENERATION = "code_generation"
    MATHEMATICAL_REASONING = "mathematical_reasoning"
    CREATIVE_WRITING = "creative_writing"
    QUESTION_ANSWERING = "question_answering"
    SUMMARIZATION = "summarization"
    TRANSLATION = "translation"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    CONVERSATION = "conversation"
    FUNCTION_CALLING = "function_calling"
    MULTIMODAL = "multimodal"
    LONG_CONTEXT = "long_context"


class ProviderStatus(Enum):
    """Provider health status."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    OFFLINE = "offline"
    RATE_LIMITED = "rate_limited"
    MAINTENANCE = "maintenance"


class RequestPriority(Enum):
    """Request priority levels."""

    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3
    EMERGENCY = 4


@dataclass
class ModelConfig:
    """Configuration for a specific model."""

    model_id: str
    provider: str
    model_name: str

    # Capabilities
    capabilities: Set[ModelCapability] = field(default_factory=set)
    max_tokens: int = 4096
    supports_streaming: bool = True
    supports_function_calling: bool = False

    # Performance characteristics
    avg_latency_ms: float = 1000.0
    cost_per_1k_tokens: float = 0.002
    max_requests_per_minute: int = 60
    max_tokens_per_minute: int = 40000

    # Quality metrics
    quality_score: float = 0.8
    reliability_score: float = 0.9
    accuracy_score: float = 0.85

    # Routing preferences
    priority: int = 5
    weight: float = 1.0
    enabled: bool = True

    # Metadata
    description: Optional[str] = None
    tags: Set[str] = field(default_factory=set)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ProviderConfig:
    """Configuration for a model provider."""

    provider_id: str
    provider_class: str
    enabled: bool = True

    # Connection settings
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    timeout_seconds: float = 30.0
    max_retries: int = 3

    # Rate limiting
    requests_per_minute: int = 100
    tokens_per_minute: int = 100000
    concurrent_requests: int = 10

    # Health monitoring
    health_check_interval: float = 60.0
    failure_threshold: int = 3
    recovery_threshold: int = 2

    # Cost management
    daily_cost_limit: Optional[float] = None
    monthly_cost_limit: Optional[float] = None

    # Metadata
    description: Optional[str] = None
    region: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class RoutingRequest:
    """Request for model routing."""

    request_id: str
    session_id: Optional[str] = None
    user_id: Optional[str] = None

    # Request content
    prompt: str
    system_prompt: Optional[str] = None
    messages: Optional[List[Dict[str, Any]]] = None

    # Requirements
    required_capabilities: Set[ModelCapability] = field(default_factory=set)
    max_tokens: Optional[int] = None
    temperature: float = 0.7
    streaming: bool = False

    # Constraints
    max_cost: Optional[float] = None
    max_latency_ms: Optional[float] = None
    priority: RequestPriority = RequestPriority.NORMAL

    # Context
    conversation_context: Dict[str, Any] = field(default_factory=dict)
    user_preferences: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    tags: Set[str] = field(default_factory=set)


@dataclass
class RoutingDecision:
    """Result of routing decision."""

    model_id: str
    provider_id: str
    routing_strategy: RoutingStrategy
    confidence: float

    # Decision factors
    factors: Dict[str, float] = field(default_factory=dict)
    alternatives: List[str] = field(default_factory=list)

    # Performance predictions
    estimated_latency_ms: float = 1000.0
    estimated_cost: float = 0.01
    estimated_quality: float = 0.8

    # Metadata
    decision_time_ms: float = 0.0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class RequestMetrics:
    """Metrics for a model request."""

    request_id: str
    model_id: str
    provider_id: str

    # Timing metrics
    start_time: datetime
    end_time: Optional[datetime] = None
    total_latency_ms: float = 0.0
    processing_latency_ms: float = 0.0
    network_latency_ms: float = 0.0

    # Usage metrics
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost: float = 0.0

    # Quality metrics
    success: bool = False
    error_type: Optional[str] = None
    quality_score: Optional[float] = None
    user_feedback: Optional[float] = None

    # Metadata
    cached: bool = False
    retries: int = 0


class ModelRouterError(Exception):
    """Custom exception for model routing operations."""

    def __init__(
        self,
        message: str,
        request_id: Optional[str] = None,
        model_id: Optional[str] = None,
        error_code: Optional[str] = None,
    ):
        super().__init__(message)
        self.request_id = request_id
        self.model_id = model_id
        self.error_code = error_code
        self.timestamp = datetime.now(timezone.utc)


class RoutingEngine(ABC):
    """Abstract base class for routing engines."""

    @abstractmethod
    async def select_model(
        self,
        request: RoutingRequest,
        available_models: List[ModelConfig],
        provider_status: Dict[str, ProviderStatus],
    ) -> RoutingDecision:
        """Select the best model for the request."""
        pass


class RoundRobinRoutingEngine(RoutingEngine):
    """Simple round-robin routing engine."""

    def __init__(self):
        self.current_index = 0
        self.lock = threading.Lock()

    async def select_model(
        self,
        request: RoutingRequest,
        available_models: List[ModelConfig],
        provider_status: Dict[str, ProviderStatus],
    ) -> RoutingDecision:
        """Select model using round-robin strategy."""
        if not available_models:
            raise ModelRouterError("No available models for routing")

        # Filter healthy models
        healthy_models = [
            model
            for model in available_models
            if provider_status.get(model.provider, ProviderStatus.OFFLINE) == ProviderStatus.HEALTHY
        ]

        if not healthy_models:
            raise ModelRouterError("No healthy models available")

        # Round-robin selection
        with self.lock:
            selected_model = healthy_models[self.current_index % len(healthy_models)]
            self.current_index += 1

        return RoutingDecision(
            model_id=selected_model.model_id,
            provider_id=selected_model.provider,
            routing_strategy=RoutingStrategy.ROUND_ROBIN,
            confidence=0.8,
            estimated_latency_ms=selected_model.avg_latency_ms,
            estimated_cost=selected_model.cost_per_1k_tokens * (request.max_tokens or 1000) / 1000,
            estimated_quality=selected_model.quality_score,
        )


class CapabilityBasedRoutingEngine(RoutingEngine):
    """Capability-based routing engine."""

    async def select_model(
        self,
        request: RoutingRequest,
        available_models: List[ModelConfig],
        provider_status: Dict[str, ProviderStatus],
    ) -> RoutingDecision:
        """Select model based on required capabilities."""
        if not available_models:
            raise ModelRouterError("No available models for routing")

        # Filter models by capabilities and health
        suitable_models = []
        for model in available_models:
            if (
                provider_status.get(model.provider, ProviderStatus.OFFLINE)
                != ProviderStatus.HEALTHY
            ):
                continue

            # Check if model has all required capabilities
            if request.required_capabilities.issubset(model.capabilities):
                suitable_models.append(model)

        if not suitable_models:
            raise ModelRouterError("No models with required capabilities available")

        # Score models based on capability match and performance
        scored_models = []
        for model in suitable_models:
            score = 0.0

            # Capability score (higher for more specific matches)
            capability_overlap = len(request.required_capabilities.intersection(model.capabilities))
            capability_score = capability_overlap / max(len(model.capabilities), 1)
            score += capability_score * 0.4

            # Performance scores
            score += model.quality_score * 0.3
            score += model.reliability_score * 0.2
            score += (1.0 / max(model.avg_latency_ms, 1)) * 0.1  # Lower latency is better

            scored_models.append((model, score))

        # Select best scoring model
        best_model, best_score = max(scored_models, key=lambda x: x[1])

        return RoutingDecision(
            model_id=best_model.model_id,
            provider_id=best_model.provider,
            routing_strategy=RoutingStrategy.CAPABILITY_BASED,
            confidence=best_score,
            factors={
                "capability_match": capability_overlap / max(len(request.required_capabilities), 1),
                "quality_score": best_model.quality_score,
                "reliability_score": best_model.reliability_score,
            },
            estimated_latency_ms=best_model.avg_latency_ms,
            estimated_cost=best_model.cost_per_1k_tokens * (request.max_tokens or 1000) / 1000,
            estimated_quality=best_model.quality_score,
        )


class AdaptiveRoutingEngine(RoutingEngine):
    """Adaptive routing engine that learns from performance."""

    def __init__(self, logger):
        self.logger = logger
        self.performance_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.model_weights: Dict[str, float] = defaultdict(lambda: 1.0)

    async def select_model(
        self,
        request: RoutingRequest,
        available_models: List[ModelConfig],
        provider_status: Dict[str, ProviderStatus],
    ) -> RoutingDecision:
        """Select model using adaptive strategy based on historical performance."""
        if not available_models:
            raise ModelRouterError("No available models for routing")

        # Filter healthy models with required capabilities
        suitable_models = []
        for model in available_models:
            if (
                provider_status.get(model.provider, ProviderStatus.OFFLINE)
                != ProviderStatus.HEALTHY
            ):
                continue

            if request.required_capabilities and not request.required_capabilities.issubset(
                model.capabilities
            ):
                continue

            suitable_models.append(model)

        if not suitable_models:
            raise ModelRouterError("No suitable models available")

        # Calculate adaptive scores
        scored_models = []
        for model in suitable_models:
            score = self._calculate_adaptive_score(model, request)
            scored_models.append((model, score))

        # Select model using weighted random selection
        weights = [score for _, score in scored_models]
        selected_model, selected_score = self._weighted_random_choice(scored_models, weights)

        return RoutingDecision(
            model_id=selected_model.model_id,
            provider_id=selected_model.provider,
            routing_strategy=RoutingStrategy.ADAPTIVE,
            confidence=selected_score,
            factors=self._get_score_factors(selected_model, request),
            estimated_latency_ms=self._estimate_latency(selected_model),
            estimated_cost=selected_model.cost_per_1k_tokens * (request.max_tokens or 1000) / 1000,
            estimated_quality=self._estimate_quality(selected_model),
        )

    def _calculate_adaptive_score(self, model: ModelConfig, request: RoutingRequest) -> float:
        """Calculate adaptive score for model based on historical performance."""
        base_score = model.quality_score * 0.3 + model.reliability_score * 0.2

        # Historical performance
        history = self.performance_history[model.model_id]
        if history:
            recent_success_rate = sum(
                1 for m in list(history)[-10:] if m.get("success", False)
            ) / min(len(history), 10)
            recent_avg_latency = np.mean([m.get("latency_ms", 1000) for m in list(history)[-10:]])
            recent_avg_quality = np.mean(
                [m.get("quality_score", 0.5) for m in list(history)[-10:] if m.get("quality_score")]
            )

            base_score += recent_success_rate * 0.3
            base_score += (1.0 / max(recent_avg_latency, 1)) * 0.1
            base_score += recent_avg_quality * 0.1

        # Apply adaptive weights
        adaptive_weight = self.model_weights[model.model_id]

        return base_score * adaptive_weight

    def _weighted_random_choice(self, choices: List[tuple], weights: List[float]) -> tuple:
        """Select item using weighted random choice."""
        total = sum(weights)
        if total <= 0:
            return random.choice(choices)

        normalized_weights = [w / total for w in weights]
        cumulative = np.cumsum(normalized_weights)
        r = random.random()

        for i, cum_weight in enumerate(cumulative):
            if r <= cum_weight:
                return choices[i]

        return choices[-1]

    def _get_score_factors(self, model: ModelConfig, request: RoutingRequest) -> Dict[str, float]:
        """Get scoring factors for transparency."""
        return {
            "quality_score": model.quality_score,
            "reliability_score": model.reliability_score,
            "adaptive_weight": self.model_weights[model.model_id],
            "estimated_cost": model.cost_per_1k_tokens,
        }

    def _estimate_latency(self, model: ModelConfig) -> float:
        """Estimate latency based on historical data."""
        history = self.performance_history[model.model_id]
        if history:
            recent_latencies = [
                m.get("latency_ms", model.avg_latency_ms) for m in list(history)[-10:]
            ]
            return np.mean(recent_latencies)
        return model.avg_latency_ms

    def _estimate_quality(self, model: ModelConfig) -> float:
        """Estimate quality based on historical data."""
        history = self.performance_history[model.model_id]
        if history:
            recent_quality = [
                m.get("quality_score", model.quality_score)
                for m in list(history)[-10:]
                if m.get("quality_score")
            ]
            if recent_quality:
                return np.mean(recent_quality)
        return model.quality_score

    def update_performance(self, metrics: RequestMetrics) -> None:
        """Update performance history with new metrics."""
        self.performance_history[metrics.model_id].append(
            {
                "success": metrics.success,
                "latency_ms": metrics.total_latency_ms,
                "quality_score": metrics.quality_score,
                "cost": metrics.cost,
                "timestamp": datetime.now(timezone.utc),
            }
        )

        # Update adaptive weights based on performance
        if metrics.success:
            self.model_weights[metrics.model_id] *= 1.01  # Slight increase for success
        else:
            self.model_weights[metrics.model_id] *= 0.95  # Decrease for failure

        # Keep weights in reasonable bounds
        self.model_weights[metrics.model_id] = max(
            0.1, min(2.0, self.model_weights[metrics.model_id])
        )


class RateLimiter:
    """Rate limiter for model providers."""

    def __init__(self, redis_cache: Optional[RedisCache] = None):
        self.redis_cache = redis_cache
        self.local_buckets: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.Lock()

    async def check_rate_limit(
        self,
        provider_id: str,
        requests_per_minute: int,
        tokens_per_minute: int,
        request_tokens: int = 0,
    ) -> bool:
        """Check if request is within rate limits."""
        current_time = time.time()
        minute_window = int(current_time / 60)

        if self.redis_cache:
            return await self._check_redis_rate_limit(
                provider_id, minute_window, requests_per_minute, tokens_per_minute, request_tokens
            )
        else:
            return self._check_local_rate_limit(
                provider_id, minute_window, requests_per_minute, tokens_per_minute, request_tokens
            )

    async def _check_redis_rate_limit(
        self,
        provider_id: str,
        minute_window: int,
        requests_per_minute: int,
        tokens_per_minute: int,
        request_tokens: int,
    ) -> bool:
        """Check rate limit using Redis."""
        request_key = f"rate_limit:{provider_id}:requests:{minute_window}"
        token_key = f"rate_limit:{provider_id}:tokens:{minute_window}"

        # Check current counts
        current_requests = await self.redis_cache.get(request_key) or 0
        current_tokens = await self.redis_cache.get(token_key) or 0

        # Check limits
        if int(current_requests) >= requests_per_minute:
            return False

        if int(current_tokens) + request_tokens > tokens_per_minute:
            return False

        # Increment counters
        pipe = await self.redis_cache.pipeline()
        pipe.incr(request_key)
        pipe.expire(request_key, 60)
        pipe.incrby(token_key, request_tokens)
        pipe.expire(token_key, 60)
        await pipe.execute()

        return True

    def _check_local_rate_limit(
        self,
        provider_id: str,
        minute_window: int,
        requests_per_minute: int,
        tokens_per_minute: int,
        request_tokens: int,
    ) -> bool:
        """Check rate limit using local memory."""
        with self.lock:
            if provider_id not in self.local_buckets:
                self.local_buckets[provider_id] = {}

            bucket = self.local_buckets[provider_id]

            # Clean old windows
            bucket = {k: v for k, v in bucket.items() if k >= minute_window - 1}
            self.local_buckets[provider_id] = bucket

            # Get current window data
            if minute_window not in bucket:
                bucket[minute_window] = {"requests": 0, "tokens": 0}

            current = bucket[minute_window]

            # Check limits
            if current["requests"] >= requests_per_minute:
                return False

            if current["tokens"] + request_tokens > tokens_per_minute:
                return False

            # Increment counters
            current["requests"] += 1
            current["tokens"] += request_tokens

            return True


class ResponseCache:
    """Cache for model responses."""

    def __init__(self, cache_strategy: CacheStrategy, logger):
        self.cache_strategy = cache_strategy
        self.logger = logger

    def _generate_cache_key(self, request: RoutingRequest) -> str:
        """Generate cache key for request."""
        # Create a deterministic hash of the request content
        content = {
            "prompt": request.prompt,
            "system_prompt": request.system_prompt,
            "messages": request.messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }

        content_str = json.dumps(content, sort_keys=True)
        return f"model_response:{hashlib.md5(content_str.encode()).hexdigest()}"

    async def get_cached_response(self, request: RoutingRequest) -> Optional[Dict[str, Any]]:
        """Get cached response if available."""
        try:
            cache_key = self._generate_cache_key(request)
            cached_data = await self.cache_strategy.get(cache_key)

            if cached_data:
                self.logger.debug(f"Cache hit for request {request.request_id}")
                return json.loads(cached_data)

            return None

        except Exception as e:
            self.logger.warning(f"Error getting cached response: {str(e)}")
            return None

    async def cache_response(
        self, request: RoutingRequest, response: Dict[str, Any], ttl_seconds: int = 3600
    ) -> None:
        """Cache model response."""
        try:
            cache_key = self._generate_cache_key(request)
            response_data = json.dumps(response)

            await self.cache_strategy.set(cache_key, response_data, ttl_seconds)
            self.logger.debug(f"Cached response for request {request.request_id}")

        except Exception as e:
            self.logger.warning(f"Error caching response: {str(e)}")


class ProviderHealthMonitor:
    """Monitor health of model providers."""

    def __init__(self, logger):
        self.logger = logger
        self.provider_status: Dict[str, ProviderStatus] = {}
        self.failure_counts: Dict[str, int] = defaultdict(int)
        self.last_health_check: Dict[str, datetime] = {}
        self.health_check_tasks: Dict[str, asyncio.Task] = {}

    async def start_monitoring(
        self, provider_id: str, provider: BaseLLMProvider, check_interval: float = 60.0
    ) -> None:
        """Start monitoring a provider."""
        if provider_id in self.health_check_tasks:
            self.health_check_tasks[provider_id].cancel()

        self.health_check_tasks[provider_id] = asyncio.create_task(
            self._monitor_provider(provider_id, provider, check_interval)
        )

    async def stop_monitoring(self, provider_id: str) -> None:
        """Stop monitoring a provider."""
        if provider_id in self.health_check_tasks:
            self.health_check_tasks[provider_id].cancel()
            del self.health_check_tasks[provider_id]

    async def _monitor_provider(
        self, provider_id: str, provider: BaseLLMProvider, check_interval: float
    ) -> None:
        """Monitor provider health continuously."""
        while True:
            try:
                await asyncio.sleep(check_interval)

                # Perform health check
                start_time = time.time()
                health_status = await provider.health_check()
                response_time = (time.time() - start_time) * 1000

                # Update status based on health check
                if health_status.get("healthy", False):
                    if response_time < 5000:  # Under 5 seconds
                        self.provider_status[provider_id] = ProviderStatus.HEALTHY
                    else:
                        self.provider_status[provider_id] = ProviderStatus.DEGRADED

                    # Reset failure count on success
                    self.failure_counts[provider_id] = 0
                else:
                    self.failure_counts[provider_id] += 1
                    if self.failure_counts[provider_id] >= 3:
                        self.provider_status[provider_id] = ProviderStatus.UNHEALTHY
                    else:
                        self.provider_status[provider_id] = ProviderStatus.DEGRADED

                self.last_health_check[provider_id] = datetime.now(timezone.utc)

            except Exception as e:
                self.logger.error(f"Health check failed for provider {provider_id}: {str(e)}")
                self.failure_counts[provider_id] += 1

                if self.failure_counts[provider_id] >= 5:
                    self.provider_status[provider_id] = ProviderStatus.OFFLINE
                else:
                    self.provider_status[provider_id] = ProviderStatus.UNHEALTHY

    def get_provider_status(self, provider_id: str) -> ProviderStatus:
        """Get current provider status."""
        return self.provider_status.get(provider_id, ProviderStatus.OFFLINE)

    def record_request_result(self, provider_id: str, success: bool) -> None:
        """Record the result of a request for health tracking."""
        if success:
            self.failure_counts[provider_id] = max(0, self.failure_counts[provider_id] - 1)
        else:
            self.failure_counts[provider_id] += 1

            # Update status based on failure count
            if self.failure_counts[provider_id] >= 5:
                self.provider_status[provider_id] = ProviderStatus.UNHEALTHY


class EnhancedModelRouter:
    """
    Advanced Model Router for the AI Assistant.

    This router provides comprehensive model management including:
    - Multi-provider support (OpenAI, DeepSeek, Ollama, etc.)
    - Intelligent routing strategies (capability-based, cost-optimized, adaptive)
    - Load balancing and failover management
    - Rate limiting and quota management
    - Response caching and performance optimization
    - Real-time health monitoring
    - Cost tracking and optimization
    - A/B testing and experimentation support
    - Integration with core system components
    """

    def __init__(self, container: Container):
        """
        Initialize the enhanced model router.

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

        # Security components
        try:
            self.auth_manager = container.get(AuthenticationManager)
            self.authz_manager = container.get(AuthorizationManager)
            self.input_sanitizer = container.get(InputSanitizer)
        except Exception:
            self.auth_manager = None
            self.authz_manager = None
            self.input_sanitizer = None

        # Storage and caching
        try:
            self.database = container.get(DatabaseManager)
            self.redis_cache = container.get(RedisCache)
            self.cache_strategy = container.get(CacheStrategy)
        except Exception:
            self.database = None
            self.redis_cache = None
            self.cache_strategy = None

        # Memory and learning
        try:
            self.memory_manager = container.get(MemoryManager)
            self.context_manager = container.get(ContextManager)
            self.continual_learner = container.get(ContinualLearner)
            self.preference_learner = container.get(PreferenceLearner)
            self.feedback_processor = container.get(FeedbackProcessor)
        except Exception:
            self.memory_manager = None
            self.context_manager = None
            self.continual_learner = None
            self.preference_learner = None
            self.feedback_processor = None

        # Observability
        self.metrics = container.get(MetricsCollector)
        self.tracer = container.get(TraceManager)

        # Model management
        self.providers: Dict[str, BaseLLMProvider] = {}
        self.model_configs: Dict[str, ModelConfig] = {}
        self.provider_configs: Dict[str, ProviderConfig] = {}

        # Routing engines
        self.routing_engines: Dict[RoutingStrategy, RoutingEngine] = {}
        self.default_strategy = RoutingStrategy.CAPABILITY_BASED

        # Monitoring and optimization
        self.health_monitor = ProviderHealthMonitor(self.logger)
        self.rate_limiter = RateLimiter(self.redis_cache)
        if self.cache_strategy:
            self.response_cache = ResponseCache(self.cache_strategy, self.logger)
        else:
            self.response_cache = None

        # Performance tracking
        self.request_metrics: Dict[str, RequestMetrics] = {}
        self.performance_history: deque = deque(maxlen=10000)
        self.cost_tracking: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))

        # Configuration
        self.enable_caching = self.config.get("model_router.enable_caching", True)
        self.enable_rate_limiting = self.config.get("model_router.enable_rate_limiting", True)
        self.enable_cost_tracking = self.config.get("model_router.enable_cost_tracking", True)
        self.cache_ttl_seconds = self.config.get("model_router.cache_ttl_seconds", 3600)
        self.max_retries = self.config.get("model_router.max_retries", 3)
        self.request_timeout = self.config.get("model_router.request_timeout", 30.0)

        # Initialize components
        self._setup_routing_engines()
        self._setup_monitoring()

        # Register health check
        self.health_check.register_component("model_router", self._health_check_callback)

        self.logger.info("EnhancedModelRouter initialized successfully")

    def _setup_routing_engines(self) -> None:
        """Setup routing engines for different strategies."""
        try:
            self.routing_engines[RoutingStrategy.ROUND_ROBIN] = RoundRobinRoutingEngine()
            self.routing_engines[RoutingStrategy.CAPABILITY_BASED] = CapabilityBasedRoutingEngine()
            self.routing_engines[RoutingStrategy.ADAPTIVE] = AdaptiveRoutingEngine(self.logger)

            self.logger.info(f"Initialized {len(self.routing_engines)} routing engines")

        except Exception as e:
            self.logger.error(f"Failed to setup routing engines: {str(e)}")

    def _setup_monitoring(self) -> None:
        """Setup monitoring and metrics."""
        try:
            # Register model router metrics
            self.metrics.register_counter("model_requests_total")
            self.metrics.register_counter("model_requests_successful")
            self.metrics.register_counter("model_requests_failed")
            self.metrics.register_histogram("model_request_duration_seconds")
            self.metrics.register_histogram("model_request_cost_dollars")
            self.metrics.register_gauge("active_providers")
            self.metrics.register_gauge("healthy_providers")
            self.metrics.register_counter("model_cache_hits")
            self.metrics.register_counter("model_cache_misses")
            self.metrics.register_counter("rate_limit_exceeded")

        except Exception as e:
            self.logger.warning(f"Failed to setup monitoring: {str(e)}")

    async def initialize(self) -> None:
        """Initialize the model router."""
        try:
            # Load configurations
            await self._load_configurations()

            # Initialize providers
            await self._initialize_providers()

            # Start health monitoring
            await self._start_health_monitoring()

            # Load model performance history
            await self._load_performance_history()

            # Start background tasks
            asyncio.create_task(self._cost_tracking_loop())
            asyncio.create_task(self._performance_optimization_loop())

            self.logger.info("ModelRouter initialization completed")

        except Exception as e:
            self.logger.error(f"Failed to initialize ModelRouter: {str(e)}")
            raise ModelRouterError(f"Initialization failed: {str(e)}")

    async def _load_configurations(self) -> None:
        """Load model and provider configurations."""
        try:
            # Load provider configurations
            provider_configs = self.config.get("model_router.providers", {})
            for provider_id, config_data in provider_configs.items():
                config = ProviderConfig(provider_id=provider_id, **config_data)
                self.provider_configs[provider_id] = config

            # Load model configurations
            model_configs = self.config.get("model_router.models", {})
            for model_id, config_data in model_configs.items():
                # Convert capabilities from strings to enums
                capabilities = set()
                for cap_str in config_data.get("capabilities", []):
                    try:
                        capabilities.add(ModelCapability(cap_str))
                    except ValueError:
                        self.logger.warning(f"Unknown capability: {cap_str}")

                config = ModelConfig(
                    model_id=model_id,
                    capabilities=capabilities,
                    **{k: v for k, v in config_data.items() if k != "capabilities"},
                )
                self.model_configs[model_id] = config

            # Set default routing strategy
            strategy_name = self.config.get("model_router.default_strategy", "capability_based")
            try:
                self.default_strategy = RoutingStrategy(strategy_name)
            except ValueError:
                self.logger.warning(f"Unknown routing strategy: {strategy_name}, using default")

            self.logger.info(
                f"Loaded {len(self.provider_configs)} providers and {len(self.model_configs)} models"
            )

        except Exception as e:
            self.logger.error(f"Failed to load configurations: {str(e)}")
            raise

    async def _initialize_providers(self) -> None:
        """Initialize LLM providers."""
        try:
            for provider_id, config in self.provider_configs.items():
                if not config.enabled:
                    continue

                # Create provider instance based on class name
                if config.provider_class == "OpenAIProvider":
                    provider = OpenAIProvider(
                        api_key=config.api_key,
                        base_url=config.base_url,
                        timeout=config.timeout_seconds,
                    )
                elif config.provider_class == "DeepSeekProvider":
                    provider = DeepSeekProvider(
                        api_key=config.api_key,
                        base_url=config.base_url,
                        timeout=config.timeout_seconds,
                    )
                elif config.provider_class == "OllamaProvider":
                    provider = OllamaProvider(
                        base_url=config.base_url or "http://localhost:11434",
                        timeout=config.timeout_seconds,
                    )
                else:
                    self.logger.warning(f"Unknown provider class: {config.provider_class}")
                    continue

                # Initialize provider
                await provider.initialize()
                self.providers[provider_id] = provider

                self.logger.info(f"Initialized provider: {provider_id}")

            # Update metrics
            self.metrics.set("active_providers", len(self.providers))

        except Exception as e:
            self.logger.error(f"Failed to initialize providers: {str(e)}")
            raise

    async def _start_health_monitoring(self) -> None:
        """Start health monitoring for all providers."""
        try:
            for provider_id, provider in self.providers.items():
                config = self.provider_configs[provider_id]
                await self.health_monitor.start_monitoring(
                    provider_id, provider, config.health_check_interval
                )

            self.logger.info("Started health monitoring for all providers")

        except Exception as e:
            self.logger.error(f"Failed to start health monitoring: {str(e)}")

    async def _load_performance_history(self) -> None:
        """Load historical performance data."""
        try:
            if self.database:
                # Load from database
                history_data = await self.database.fetch_all(
                    """
                    SELECT * FROM model_performance_history 
                    WHERE created_at > datetime('now', '-30 days')
                    ORDER BY created_at DESC
                    LIMIT 10000
                    """
                )

                for row in history_data:
                    metrics = RequestMetrics(
                        request_id=row["request_id"],
                        model_id=row["model_id"],
                        provider_id=row["provider_id"],
                        start_time=datetime.fromisoformat(row["start_time"]),
                        **{
                            k: v
                            for k, v in row.items()
                            if k not in ["request_id", "model_id", "provider_id", "start_time"]
                        },
                    )
                    self.performance_history.append(metrics)

                # Update adaptive routing engine
                adaptive_engine = self.routing_engines.get(RoutingStrategy.ADAPTIVE)
                if adaptive_engine and hasattr(adaptive_engine, "update_performance"):
                    for metrics in self.performance_history:
                        adaptive_engine.update_performance(metrics)

                self.logger.info(f"Loaded {len(self.performance_history)} performance records")

        except Exception as e:
            self.logger.warning(f"Failed to load performance history: {str(e)}")

    @handle_exceptions
    async def route_request(
        self, request: RoutingRequest, strategy: Optional[RoutingStrategy] = None
    ) -> ModelResponse:
        """
        Route a request to the best available model.

        Args:
            request: Routing request with content and requirements
            strategy: Optional routing strategy override

        Returns:
            Model response
        """
        start_time = time.time()
        strategy = strategy or self.default_strategy

        try:
            with self.tracer.trace("model_request_routing") as span:
                span.set_attributes(
                    {
                        "request_id": request.request_id,
                        "session_id": request.session_id or "anonymous",
                        "user_id": request.user_id or "anonymous",
                        "strategy": strategy.value,
                        "required_capabilities": list(request.required_capabilities),
                        "max_tokens": request.max_tokens or 0,
                        "streaming": request.streaming,
                    }
                )

                # Security validation
                if self.input_sanitizer:
                    request.prompt = await self.input_sanitizer.sanitize_text(request.prompt)
                    if request.system_prompt:
                        request.system_prompt = await self.input_sanitizer.sanitize_text(
                            request.system_prompt
                        )

                # Check cache first
                cached_response = None
                if self.enable_caching and self.response_cache and not request.streaming:
                    cached_response = await self.response_cache.get_cached_response(request)
                    if cached_response:
                        await self.event_bus.emit(
                            ModelCacheHit(
                                request_id=request.request_id,
                                cache_key=self.response_cache._generate_cache_key(request),
                            )
                        )
                        self.metrics.increment("model_cache_hits")

                        # Return cached response
                        return ModelResponse(
                            content=cached_response["content"],
                            model_id=cached_response.get("model_id", "cached"),
                            provider_id=cached_response.get("provider_id", "cache"),
                            tokens_used=cached_response.get("tokens_used", 0),
                            latency_ms=time.time() - start_time,
                            cached=True,
                        )

                if not cached_response:
                    self.metrics.increment("model_cache_misses")
                    await self.event_bus.emit(ModelCacheMiss(request_id=request.request_id))

                # Route to best model
                routing_decision = await self._route_to_model(request, strategy)

                # Execute request
                response = await self._execute_request(request, routing_decision)

                # Cache response if successful
                if (
                    self.enable_caching
                    and self.response_cache
                    and response.success
                    and not request.streaming
                ):
                    cache_data = {
                        "content": response.content,
                        "model_id": response.model_id,
                        "provider_id": response.provider_id,
                        "tokens_used": response.tokens_used,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                    await self.response_cache.cache_response(
                        request, cache_data, self.cache_ttl_seconds
                    )

                # Record metrics
                await self._record_request_metrics(request, routing_decision, response, start_time)

                # Update learning systems
                if self.continual_learner:
                    await self._update_learning_systems(request, routing_decision, response)

                return response

        except Exception as e:
            # Record failure metrics
            await self._record_failure_metrics(request, str(e), start_time)

            self.logger.error(f"Model routing failed for request {request.request_id}: {str(e)}")
            raise ModelRouterError(f"Routing failed: {str(e)}", request.request_id)

    async def _route_to_model(
        self, request: RoutingRequest, strategy: RoutingStrategy
    ) -> RoutingDecision:
        """Route request to the best model using specified strategy."""
        decision_start = time.time()

        # Get available models
        available_models = [
            model
            for model in self.model_configs.values()
            if model.enabled and self._is_model_available(model, request)
        ]

        if not available_models:
            raise ModelRouterError("No available models match the requirements")

        # Get provider status
        provider_status = {
            provider_id: self.health_monitor.get_provider_status(provider_id)
            for provider_id in self.providers.keys()
        }

        # Get routing engine
        routing_engine = self.routing_engines.get(strategy)
        if not routing_engine:
            self.logger.warning(f"Routing engine for {strategy} not found, using default")
            routing_engine = self.routing_engines[RoutingStrategy.CAPABILITY_BASED]

        # Make routing decision
        decision = await routing_engine.select_model(request, available_models, provider_status)
        decision.decision_time_ms = (time.time() - decision_start) * 1000

        # Emit routing event
        await self.event_bus.emit(
            ModelRouteSelected(
                request_id=request.request_id,
                model_id=decision.model_id,
                provider_id=decision.provider_id,
                strategy=strategy.value,
                confidence=decision.confidence,
                decision_time_ms=decision.decision_time_ms,
            )
        )

        return decision

    def _is_model_available(self, model: ModelConfig, request: RoutingRequest) -> bool:
        """Check if model is available for the request."""
        # Check if provider is available
        if model.provider not in self.providers:
            return False

        # Check provider health
        provider_status = self.health_monitor.get_provider_status(model.provider)
        if provider_status not in [ProviderStatus.HEALTHY, ProviderStatus.DEGRADED]:
            return False

        # Check capabilities
        if request.required_capabilities and not request.required_capabilities.issubset(
            model.capabilities
        ):
            return False

        # Check token limits
        if request.max_tokens and request.max_tokens > model.max_tokens:
            return False

        # Check cost constraints
        if request.max_cost:
            estimated_cost = model.cost_per_1k_tokens * (request.max_tokens or 1000) / 1000
            if estimated_cost > request.max_cost:
                return False

        return True

    async def _execute_request(
        self, request: RoutingRequest, decision: RoutingDecision
    ) -> ModelResponse:
        """Execute the routed request."""
        provider = self.providers[decision.provider_id]
        model_config = self.model_configs[decision.model_id]

        # Check rate limits
        if self.enable_rate_limiting:
            provider_config = self.provider_configs[decision.provider_id]
            tokens_estimate = request.max_tokens or 1000

            rate_limit_ok = await self.rate_limiter.check_rate_limit(
                decision.provider_id,
                provider_config.requests_per_minute,
                provider_config.tokens_per_minute,
                tokens_estimate,
            )

            if not rate_limit_ok:
                await self.event_bus.emit(
                    ModelRateLimitExceeded(
                        provider_id=decision.provider_id, request_id=request.request_id
                    )
                )
                self.metrics.increment("rate_limit_exceeded")
                raise ModelRouterError(f"Rate limit exceeded for provider {decision.provider_id}")

        # Emit request started event
        await self.event_bus.emit(
            ModelRequestStarted(
                request_id=request.request_id,
                model_id=decision.model_id,
                provider_id=decision.provider_id,
            )
        )

        # Execute request with retry logic
        for attempt in range(self.max_retries + 1):
            try:
                # Prepare request parameters
                request_params = {
                    "model": decision.model_id,
                    "messages": request.messages or [{"role": "user", "content": request.prompt}],
                    "temperature": request.temperature,
                    "max_tokens": request.max_tokens,
                    "stream": request.streaming,
                }

                if request.system_prompt:
                    request_params["messages"].insert(
                        0, {"role": "system", "content": request.system_prompt}
                    )

                # Execute request
                response = await asyncio.wait_for(
                    provider.generate(request_params), timeout=self.request_timeout
                )

                # Record success
                self.health_monitor.record_request_result(decision.provider_id, True)

                await self.event_bus.emit(
                    ModelRequestCompleted(
                        request_id=request.request_id,
                        model_id=decision.model_id,
                        provider_id=decision.provider_id,
                        tokens_used=response.tokens_used,
                        success=True,
                    )
                )

                return response

            except Exception as e:
                # Record failure
                self.health_monitor.record_request_result(decision.provider_id, False)

                if attempt < self.max_retries:
                    self.logger.warning(f"Request attempt {attempt + 1} failed, retrying: {str(e)}")
                    await asyncio.sleep(min(2**attempt, 10))  # Exponential backoff
                else:
                    await self.event_bus.emit(
                        ModelRequestFailed(
                            request_id=request.request_id,
                            model_id=decision.model_id,
                            provider_id=decision.provider_id,
                            error_message=str(e),
                            error_type=type(e).__name__,
                        )
                    )
                    raise

    async def _record_request_metrics(
        self,
        request: RoutingRequest,
        decision: RoutingDecision,
        response: ModelResponse,
        start_time: float,
    ) -> None:
        """Record comprehensive request metrics."""
        try:
            # Create metrics record
            metrics = RequestMetrics(
                request_id=request.request_id,
                model_id=decision.model_id,
                provider_id=decision.provider_id,
                start_time=datetime.fromtimestamp(start_time, timezone.utc),
                end_time=datetime.now(timezone.utc),
                total_latency_ms=(time.time() - start_time) * 1000,
                processing_latency_ms=response.latency_ms,
                input_tokens=response.tokens_used,  # Simplified
                output_tokens=response.tokens_used,
                total_tokens=response.tokens_used,
                cost=self._calculate_cost(decision.model_id, response.tokens_used),
                success=response.success,
                cached=response.cached,
            )

            # Store metrics
            self.request_metrics[request.request_id] = metrics
            self.performance_history.append(metrics)

            # Update adaptive routing engine
            adaptive_engine = self.routing_engines.get(RoutingStrategy.ADAPTIVE)
            if adaptive_engine and hasattr(adaptive_engine, "update_performance"):
                adaptive_engine.update_performance(metrics)

            # Update Prometheus metrics
            self.metrics.increment("model_requests_total")
            if response.success:
                self.metrics.increment("model_requests_successful")
            else:
                self.metrics.increment("model_requests_failed")

            self.metrics.record("model_request_duration_seconds", metrics.total_latency_ms / 1000)
            self.metrics.record("model_request_cost_dollars", metrics.cost)

            # Track costs
            if self.enable_cost_tracking:
                current_date = datetime.now(timezone.utc).date().isoformat()
                self.cost_tracking[decision.provider_id][current_date] += metrics.cost

            # Store in database if available
            if self.database:
                await self.database.execute(
                    """
                    INSERT INTO model_performance_history 
                    (request_id, model_id, provider_id, start_time, total_latency_ms, 
                     tokens_used, cost, success, cached)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        metrics.request_id,
                        metrics.model_id,
                        metrics.provider_id,
                        metrics.start_time.isoformat(),
                        metrics.total_latency_ms,
                        metrics.total_tokens,
                        metrics.cost,
                        metrics.success,
                        metrics.cached,
                    ),
                )

        except Exception as e:
            self.logger.warning(f"Failed to record metrics: {str(e)}")

    async def _record_failure_metrics(
        self, request: RoutingRequest, error_message: str, start_time: float
    ) -> None:
        """Record metrics for failed requests."""
        try:
            self.metrics.increment("model_requests_total")
            self.metrics.increment("model_requests_failed")
            self.metrics.record("model_request_duration_seconds", time.time() - start_time)

        except Exception as e:
            self.logger.warning(f"Failed to record failure metrics: {str(e)}")

    def _calculate_cost(self, model_id: str, tokens_used: int) -> float:
        """Calculate cost for the request."""
        try:
            model_config = self.model_configs.get(model_id)
            if model_config:
                return model_config.cost_per_1k_tokens * tokens_used / 1000
            return 0.0
        except Exception:
            return 0.0

    async def _update_learning_systems(
        self, request: RoutingRequest, decision: RoutingDecision, response: ModelResponse
    ) -> None:
        """Update learning systems with request data."""
        try:
            learning_data = {
                "request_id": request.request_id,
                "user_id": request.user_id,
                "session_id": request.session_id,
                "model_id": decision.model_id,
                "provider_id": decision.provider_id,
                "routing_strategy": decision.routing_strategy.value,
                "success": response.success,
                "latency_ms": response.latency_ms,
                "cost": self._calculate_cost(decision.model_id, response.tokens_used),
                "quality_score": getattr(response, "quality_score", None),
                "user_feedback": None,  # Will be updated when feedback is received
                "context": request.conversation_context,
                "timestamp": datetime.now(timezone.utc),
            }

            # Update continual learning
            if self.continual_learner:
                await self.continual_learner.learn_from_model_interaction(learning_data)

            # Update user preferences
            if self.preference_learner and request.user_id:
                await self.preference_learner.update_model_preferences(
                    request.user_id, learning_data
                )

        except Exception as e:
            self.logger.warning(f"Failed to update learning systems: {str(e)}")

    async def get_available_models(
        self, capabilities: Optional[Set[ModelCapability]] = None, user_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get list of available models with their capabilities.

        Args:
            capabilities: Filter by required capabilities
            user_id: Optional user ID for personalization

        Returns:
            List of available models with metadata
        """
        available_models = []

        for model_id, model_config in self.model_configs.items():
            if not model_config.enabled:
                continue

            provider_status = self.health_monitor.get_provider_status(model_config.provider)
            if provider_status == ProviderStatus.OFFLINE:
                continue

            # Filter by capabilities
            if capabilities and not capabilities.issubset(model_config.capabilities):
                continue

            # Get performance metrics
            recent_metrics = [
                m
                for m in self.performance_history
                if m.model_id == model_id
                and m.start_time > datetime.now(timezone.utc) - timedelta(hours=24)
            ]

            avg_latency = (
                np.mean([m.total_latency_ms for m in recent_metrics])
                if recent_metrics
                else model_config.avg_latency_ms
            )
            success_rate = np.mean([m.success for m in recent_metrics]) if recent_metrics else 0.9
            avg_cost = (
                np.mean([m.cost for m in recent_metrics])
                if recent_metrics
                else model_config.cost_per_1k_tokens
            )

            available_models.append(
                {
                    "model_id": model_id,
                    "provider": provider_name,
                    "capabilities": model_config.capabilities,
                    "success_rate": success_rate,
                    "avg_latency": avg_latency,
                    "avg_cost": avg_cost,
                    "health_score": health_score,
                    "weighted_score": weighted_score,
                }
            )
