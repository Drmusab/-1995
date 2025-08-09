"""
Advanced Base LLM Provider for AI Assistant
Author: Drmusab
Last Modified: 2025-06-13 17:57:41 UTC

This module provides the foundational base class for all LLM provider integrations
in the AI assistant system, including comprehensive monitoring, caching, security,
and seamless integration with core system components.
"""

import hashlib
import inspect
import json
import logging
import threading
import time
import traceback
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
import tiktoken

from src.assistant.core import (
    ComponentMetadata,
    ComponentPriority,
    EnhancedComponentManager,
)

# Assistant components
from src.assistant.core import ProcessingContext, ProcessingResult
from src.assistant.core import InteractionHandler
from src.assistant.core import EnhancedSessionManager
from src.assistant.core import WorkflowOrchestrator

# Core imports
from src.core.config.loader import ConfigLoader
from src.core.dependency_injection import Container
from src.core.error_handling import ErrorHandler, handle_exceptions
from src.core.events.event_bus import EventBus
from src.core.events.event_types import (
    ComponentHealthChanged,
    ErrorOccurred,
    LLMCacheHit,
    LLMCacheMiss,
    LLMContextWindowExceeded,
    LLMCostThresholdExceeded,
    LLMModelLoaded,
    LLMModelUnloaded,
    LLMPerformanceWarning,
    LLMProviderHealthChanged,
    LLMRequestCompleted,
    LLMRequestFailed,
    LLMRequestStarted,
    LLMSecurityViolation,
    LLMStreamingCompleted,
    LLMStreamingStarted,
    LLMTokenLimitExceeded,
    SystemStateChanged,
)
from src.core.health_check import HealthCheck
from src.core.security.authentication import AuthenticationManager
from src.core.security.authorization import AuthorizationManager
from src.core.security.sanitization import SecuritySanitizer

# Integrations
from src.integrations.cache.cache_strategy import CacheStrategy
from src.integrations.cache.redis_cache import RedisCache
from src.integrations.storage.database import DatabaseManager
from src.learning.continual_learning import ContinualLearner
from src.learning.feedback_processor import FeedbackProcessor
from src.learning.preference_learning import PreferenceLearner

# Memory and learning
from src.memory.core_memory.memory_manager import MemoryManager
from src.memory.operations.context_manager import ContextManager
from src.observability.logging.config import get_logger

# Observability
from src.observability.monitoring.metrics import MetricsCollector
from src.observability.monitoring.tracing import TraceManager

# Type definitions
T = TypeVar("T")


class LLMProviderType(Enum):
    """Types of LLM providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    HUGGINGFACE = "huggingface"
    OLLAMA = "ollama"
    DEEPSEEK = "deepseek"
    AZURE_OPENAI = "azure_openai"
    GOOGLE = "google"
    COHERE = "cohere"
    LOCAL = "local"
    CUSTOM = "custom"


class ModelCapability(Enum):
    """Model capabilities."""

    TEXT_GENERATION = "text_generation"
    TEXT_COMPLETION = "text_completion"
    CHAT_COMPLETION = "chat_completion"
    EMBEDDING = "embedding"
    IMAGE_GENERATION = "image_generation"
    IMAGE_UNDERSTANDING = "image_understanding"
    CODE_GENERATION = "code_generation"
    FUNCTION_CALLING = "function_calling"
    JSON_MODE = "json_mode"
    STREAMING = "streaming"
    FINE_TUNING = "fine_tuning"


class RequestPriority(Enum):
    """Request priority levels."""

    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3
    EMERGENCY = 4


class ProviderStatus(Enum):
    """Provider status states."""

    INITIALIZING = "initializing"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    MAINTENANCE = "maintenance"
    OFFLINE = "offline"
    ERROR = "error"


@dataclass
class ModelInfo:
    """Information about an LLM model."""

    model_id: str
    name: str
    provider_type: LLMProviderType

    # Capabilities
    capabilities: Set[ModelCapability] = field(default_factory=set)
    max_tokens: int = 4096
    context_window: int = 4096
    max_output_tokens: int = 2048

    # Pricing (per 1K tokens)
    input_cost_per_1k: float = 0.0
    output_cost_per_1k: float = 0.0

    # Performance characteristics
    average_latency_ms: float = 1000.0
    throughput_tokens_per_second: float = 100.0

    # Metadata
    version: str = "1.0.0"
    description: Optional[str] = None
    supported_languages: List[str] = field(default_factory=lambda: ["en"])

    # Technical details
    tokenizer_name: Optional[str] = None
    architecture: Optional[str] = None
    training_cutoff: Optional[datetime] = None

    # Usage limits
    rate_limit_requests_per_minute: int = 60
    rate_limit_tokens_per_minute: int = 90000

    # Quality metrics
    quality_score: float = 0.8
    safety_score: float = 0.9
    reliability_score: float = 0.95


@dataclass
class LLMRequest:
    """Request object for LLM operations."""

    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: Optional[str] = None
    user_id: Optional[str] = None

    # Request content
    messages: List[Dict[str, Any]] = field(default_factory=list)
    prompt: Optional[str] = None
    system_prompt: Optional[str] = None

    # Model configuration
    model_id: str = "default"
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    top_p: float = 1.0
    top_k: Optional[int] = None
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop_sequences: List[str] = field(default_factory=list)

    # Advanced options
    streaming: bool = False
    json_mode: bool = False
    function_calling: bool = False
    functions: List[Dict[str, Any]] = field(default_factory=list)
    tools: List[Dict[str, Any]] = field(default_factory=list)

    # Request metadata
    priority: RequestPriority = RequestPriority.NORMAL
    timeout_seconds: float = 30.0
    retry_count: int = 0
    max_retries: int = 3

    # Context and processing
    context: Dict[str, Any] = field(default_factory=dict)
    processing_hints: Dict[str, Any] = field(default_factory=dict)

    # Caching
    cache_enabled: bool = True
    cache_ttl: Optional[int] = None

    # Security and safety
    content_filter: bool = True
    safety_check: bool = True

    # Timestamps
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


@dataclass
class LLMResponse:
    """Response object from LLM operations."""

    request_id: str
    provider_type: LLMProviderType
    model_id: str

    # Response content
    content: str = ""
    finish_reason: str = "stop"

    # Token usage
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    # Metadata
    response_id: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Performance metrics
    latency_ms: float = 0.0
    tokens_per_second: float = 0.0
    time_to_first_token_ms: float = 0.0

    # Cost information
    input_cost: float = 0.0
    output_cost: float = 0.0
    total_cost: float = 0.0

    # Quality and confidence
    confidence_score: float = 0.0
    quality_score: float = 0.0

    # Function calling results
    function_calls: List[Dict[str, Any]] = field(default_factory=list)
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)

    # Safety and filtering
    content_filtered: bool = False
    safety_scores: Dict[str, float] = field(default_factory=dict)

    # Caching information
    from_cache: bool = False
    cache_hit: bool = False

    # Error information
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)


@dataclass
class TokenUsage:
    """Token usage tracking."""

    session_id: str
    user_id: Optional[str] = None
    provider_type: LLMProviderType = LLMProviderType.OPENAI
    model_id: str = "default"

    # Token counts
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    # Cost tracking
    total_cost: float = 0.0
    input_cost: float = 0.0
    output_cost: float = 0.0

    # Time tracking
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    end_time: Optional[datetime] = None

    # Request tracking
    request_count: int = 0
    success_count: int = 0
    error_count: int = 0

    # Performance metrics
    average_latency_ms: float = 0.0
    total_processing_time_ms: float = 0.0


class TokenManager:
    """Advanced token management and tracking."""

    def __init__(self, logger):
        self.logger = logger
        self.session_usage: Dict[str, TokenUsage] = {}
        self.user_usage: Dict[str, TokenUsage] = {}
        self.model_usage: Dict[str, TokenUsage] = defaultdict(lambda: TokenUsage("global"))
        self.usage_lock = threading.Lock()

        # Cost limits and thresholds
        self.daily_cost_limit = 100.0
        self.session_cost_limit = 10.0
        self.user_cost_limit = 50.0

        # Token counting
        self.tokenizers: Dict[str, Any] = {}
        self._load_tokenizers()

    def _load_tokenizers(self) -> None:
        """Load tokenizers for different models."""
        try:
            # Load common tokenizers
            self.tokenizers["gpt-4"] = tiktoken.encoding_for_model("gpt-4")
            self.tokenizers["gpt-3.5-turbo"] = tiktoken.encoding_for_model("gpt-3.5-turbo")
            self.tokenizers["text-embedding-ada-002"] = tiktoken.encoding_for_model(
                "text-embedding-ada-002"
            )
        except Exception as e:
            self.logger.warning(f"Failed to load some tokenizers: {str(e)}")

    def count_tokens(self, text: str, model_id: str = "gpt-4") -> int:
        """Count tokens in text for specific model."""
        try:
            if model_id in self.tokenizers:
                return len(self.tokenizers[model_id].encode(text))
            elif "gpt-4" in self.tokenizers:
                return len(self.tokenizers["gpt-4"].encode(text))
            else:
                # Fallback estimation (rough)
                return len(text.split()) * 1.3
        except Exception as e:
            self.logger.warning(f"Token counting failed: {str(e)}")
            return len(text.split()) * 1.3

    def update_usage(self, request: LLMRequest, response: LLMResponse) -> None:
        """Update token usage statistics."""
        with self.usage_lock:
            # Update session usage
            if request.session_id:
                if request.session_id not in self.session_usage:
                    self.session_usage[request.session_id] = TokenUsage(
                        session_id=request.session_id,
                        user_id=request.user_id,
                        provider_type=response.provider_type,
                        model_id=response.model_id,
                    )

                session_usage = self.session_usage[request.session_id]
                session_usage.total_tokens += response.total_tokens
                session_usage.prompt_tokens += response.prompt_tokens
                session_usage.completion_tokens += response.completion_tokens
                session_usage.total_cost += response.total_cost
                session_usage.input_cost += response.input_cost
                session_usage.output_cost += response.output_cost
                session_usage.request_count += 1
                session_usage.total_processing_time_ms += response.latency_ms

                if response.error:
                    session_usage.error_count += 1
                else:
                    session_usage.success_count += 1

                # Update average latency
                session_usage.average_latency_ms = (
                    session_usage.total_processing_time_ms / session_usage.request_count
                )

            # Update user usage
            if request.user_id:
                if request.user_id not in self.user_usage:
                    self.user_usage[request.user_id] = TokenUsage(
                        session_id=request.user_id,
                        user_id=request.user_id,
                        provider_type=response.provider_type,
                        model_id=response.model_id,
                    )

                user_usage = self.user_usage[request.user_id]
                user_usage.total_tokens += response.total_tokens
                user_usage.prompt_tokens += response.prompt_tokens
                user_usage.completion_tokens += response.completion_tokens
                user_usage.total_cost += response.total_cost
                user_usage.input_cost += response.input_cost
                user_usage.output_cost += response.output_cost
                user_usage.request_count += 1

            # Update model usage
            model_key = f"{response.provider_type.value}:{response.model_id}"
            model_usage = self.model_usage[model_key]
            model_usage.total_tokens += response.total_tokens
            model_usage.prompt_tokens += response.prompt_tokens
            model_usage.completion_tokens += response.completion_tokens
            model_usage.total_cost += response.total_cost
            model_usage.request_count += 1

    def check_limits(self, request: LLMRequest) -> Dict[str, Any]:
        """Check if request would exceed usage limits."""
        violations = []

        # Check session limits
        if request.session_id and request.session_id in self.session_usage:
            session_usage = self.session_usage[request.session_id]
            if session_usage.total_cost >= self.session_cost_limit:
                violations.append(f"Session cost limit exceeded: {session_usage.total_cost}")

        # Check user limits
        if request.user_id and request.user_id in self.user_usage:
            user_usage = self.user_usage[request.user_id]
            if user_usage.total_cost >= self.user_cost_limit:
                violations.append(f"User cost limit exceeded: {user_usage.total_cost}")

        return {"allowed": len(violations) == 0, "violations": violations}

    def get_usage_stats(self, session_id: str = None, user_id: str = None) -> Dict[str, Any]:
        """Get usage statistics."""
        with self.usage_lock:
            stats = {}

            if session_id and session_id in self.session_usage:
                stats["session"] = asdict(self.session_usage[session_id])

            if user_id and user_id in self.user_usage:
                stats["user"] = asdict(self.user_usage[user_id])

            # Global stats
            total_tokens = sum(usage.total_tokens for usage in self.model_usage.values())
            total_cost = sum(usage.total_cost for usage in self.model_usage.values())
            total_requests = sum(usage.request_count for usage in self.model_usage.values())

            stats["global"] = {
                "total_tokens": total_tokens,
                "total_cost": total_cost,
                "total_requests": total_requests,
                "active_sessions": len(self.session_usage),
                "active_users": len(self.user_usage),
            }

            return stats


class LLMError(Exception):
    """Custom exception for LLM provider operations."""

    def __init__(
        self,
        message: str,
        provider_type: Optional[LLMProviderType] = None,
        model_id: Optional[str] = None,
        request_id: Optional[str] = None,
        error_code: Optional[str] = None,
    ):
        super().__init__(message)
        self.provider_type = provider_type
        self.model_id = model_id
        self.request_id = request_id
        self.error_code = error_code
        self.timestamp = datetime.now(timezone.utc)


class BaseLLMProvider(ABC):
    """
    Advanced Base LLM Provider for the AI Assistant System.

    This abstract base class provides a comprehensive foundation for all LLM provider
    integrations, including:

    Features:
    - Seamless integration with core assistant components
    - Advanced token management and cost tracking
    - Comprehensive caching and performance optimization
    - Security and authentication integration
    - Event-driven architecture with real-time monitoring
    - Session and context awareness
    - Multi-model support and intelligent routing
    - Streaming and non-streaming response handling
    - Automatic error recovery and retry mechanisms
    - Performance profiling and optimization
    - Learning and adaptation capabilities
    """

    def __init__(
        self,
        container: Container,
        provider_type: LLMProviderType,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the base LLM provider.

        Args:
            container: Dependency injection container
            provider_type: Type of LLM provider
            config: Provider-specific configuration
        """
        self.container = container
        self.provider_type = provider_type
        self.config = config or {}
        self.logger = get_logger(f"llm_provider_{provider_type.value}")

        # Core services
        self._setup_core_services()

        # Provider state
        self.status = ProviderStatus.INITIALIZING
        self.supported_models: Dict[str, ModelInfo] = {}
        self.active_requests: Dict[str, LLMRequest] = {}
        self.request_history: deque = deque(maxlen=1000)

        # Performance tracking
        self.performance_metrics: Dict[str, List[float]] = defaultdict(list)
        self.error_counts: Dict[str, int] = defaultdict(int)
        self.success_counts: Dict[str, int] = defaultdict(int)

        # Infrastructure
        self._setup_infrastructure()
        self._setup_monitoring()

        # Configuration
        self.max_concurrent_requests = self.config.get("max_concurrent_requests", 10)
        self.default_timeout = self.config.get("default_timeout", 30.0)
        self.retry_enabled = self.config.get("retry_enabled", True)
        self.cache_enabled = self.config.get("cache_enabled", True)
        self.streaming_enabled = self.config.get("streaming_enabled", True)

        # Rate limiting
        self.request_semaphore = asyncio.Semaphore(self.max_concurrent_requests)
        self.rate_limiter = self._setup_rate_limiter()

        self.logger.info(f"Initialized {provider_type.value} LLM provider")

    def _setup_core_services(self) -> None:
        """Setup core assistant services."""
        try:
            self.config_loader = self.container.get(ConfigLoader)
            self.event_bus = self.container.get(EventBus)
            self.error_handler = self.container.get(ErrorHandler)
            self.health_check = self.container.get(HealthCheck)

            # Assistant components
            self.component_manager = self.container.get(EnhancedComponentManager)
            self.workflow_orchestrator = self.container.get(WorkflowOrchestrator)
            self.session_manager = self.container.get(EnhancedSessionManager)
            self.interaction_handler = self.container.get(InteractionHandler)

            # Memory and context
            self.memory_manager = self.container.get(MemoryManager)
            self.context_manager = self.container.get(ContextManager)

            # Learning systems
            self.continual_learner = self.container.get(ContinualLearner)
            self.preference_learner = self.container.get(PreferenceLearner)
            self.feedback_processor = self.container.get(FeedbackProcessor)

            # Security
            try:
                self.auth_manager = self.container.get(AuthenticationManager)
                self.authz_manager = self.container.get(AuthorizationManager)
                self.security_sanitizer = self.container.get(SecuritySanitizer)
            except Exception:
                self.auth_manager = None
                self.authz_manager = None
                self.security_sanitizer = None

        except Exception as e:
            self.logger.error(f"Failed to setup core services: {str(e)}")
            raise LLMError(f"Core services setup failed: {str(e)}", self.provider_type)

    def _setup_infrastructure(self) -> None:
        """Setup provider infrastructure."""
        try:
            # Token management
            self.token_manager = TokenManager(self.logger)

            # Caching
            try:
                self.cache_strategy = self.container.get(CacheStrategy)
                self.redis_cache = self.container.get(RedisCache)
            except Exception:
                self.cache_strategy = None
                self.redis_cache = None

            # Storage
            try:
                self.database = self.container.get(DatabaseManager)
            except Exception:
                self.database = None

            # Thread pool for blocking operations
            self.thread_pool = ThreadPoolExecutor(
                max_workers=4, thread_name_prefix=f"llm_{self.provider_type.value}"
            )

        except Exception as e:
            self.logger.error(f"Failed to setup infrastructure: {str(e)}")

    def _setup_monitoring(self) -> None:
        """Setup monitoring and observability."""
        try:
            self.metrics = self.container.get(MetricsCollector)
            self.tracer = self.container.get(TraceManager)

            # Register provider-specific metrics
            provider_name = self.provider_type.value
            self.metrics.register_counter(f"llm_{provider_name}_requests_total")
            self.metrics.register_counter(f"llm_{provider_name}_requests_successful")
            self.metrics.register_counter(f"llm_{provider_name}_requests_failed")
            self.metrics.register_histogram(f"llm_{provider_name}_latency_seconds")
            self.metrics.register_histogram(f"llm_{provider_name}_tokens_per_request")
            self.metrics.register_gauge(f"llm_{provider_name}_active_requests")
            self.metrics.register_counter(f"llm_{provider_name}_total_cost")
            self.metrics.register_histogram(f"llm_{provider_name}_tokens_per_second")

        except Exception as e:
            self.logger.warning(f"Failed to setup monitoring: {str(e)}")
            self.metrics = None
            self.tracer = None

    def _setup_rate_limiter(self) -> Dict[str, Any]:
        """Setup rate limiting mechanism."""
        return {
            "requests_per_minute": 60,
            "tokens_per_minute": 90000,
            "last_request_time": 0,
            "request_count": 0,
            "token_count": 0,
            "window_start": time.time(),
        }

    async def initialize(self) -> None:
        """Initialize the provider."""
        try:
            self.logger.info(f"Initializing {self.provider_type.value} provider...")

            # Load supported models
            await self._load_supported_models()

            # Initialize provider-specific components
            await self._initialize_provider()

            # Register with component manager
            await self._register_with_component_manager()

            # Setup health monitoring
            self._setup_health_monitoring()

            # Register event handlers
            await self._register_event_handlers()

            # Start background tasks
            await self._start_background_tasks()

            self.status = ProviderStatus.HEALTHY

            # Emit provider initialization event
            await self.event_bus.emit(
                LLMModelLoaded(
                    provider_type=self.provider_type.value,
                    models_loaded=list(self.supported_models.keys()),
                    initialization_time=0.0,
                )
            )

            self.logger.info(f"Successfully initialized {self.provider_type.value} provider")

        except Exception as e:
            self.status = ProviderStatus.ERROR
            self.logger.error(f"Failed to initialize provider: {str(e)}")
            raise LLMError(f"Provider initialization failed: {str(e)}", self.provider_type)

    @abstractmethod
    async def _load_supported_models(self) -> None:
        """Load information about supported models."""
        pass

    @abstractmethod
    async def _initialize_provider(self) -> None:
        """Initialize provider-specific components."""
        pass

    async def _register_with_component_manager(self) -> None:
        """Register provider with component manager."""
        try:
            metadata = ComponentMetadata(
                component_id=f"llm_provider_{self.provider_type.value}",
                component_type=type(self),
                priority=ComponentPriority.HIGH,
                description=f"LLM Provider: {self.provider_type.value}",
                health_check_interval=60.0,
            )

            self.component_manager.register_component(
                f"llm_provider_{self.provider_type.value}", type(self), ComponentPriority.HIGH, []
            )

        except Exception as e:
            self.logger.warning(f"Failed to register with component manager: {str(e)}")

    def _setup_health_monitoring(self) -> None:
        """Setup health monitoring."""
        self.health_check.register_component(
            f"llm_provider_{self.provider_type.value}", self._health_check_callback
        )

    async def _register_event_handlers(self) -> None:
        """Register event handlers."""
        # System events
        self.event_bus.subscribe("system_shutdown_started", self._handle_system_shutdown)

        # Session events
        self.event_bus.subscribe("session_started", self._handle_session_started)
        self.event_bus.subscribe("session_ended", self._handle_session_ended)

        # Learning events
        self.event_bus.subscribe("feedback_received", self._handle_feedback)

    async def _start_background_tasks(self) -> None:
        """Start background maintenance tasks."""
        # Performance monitoring
        asyncio.create_task(self._performance_monitoring_loop())

        # Cache cleanup
        if self.cache_enabled:
            asyncio.create_task(self._cache_cleanup_loop())

        # Usage tracking
        asyncio.create_task(self._usage_tracking_loop())

        # Health monitoring
        asyncio.create_task(self._health_monitoring_loop())

    @handle_exceptions
    async def generate_completion(self, request: LLMRequest) -> LLMResponse:
        """
        Generate text completion from the LLM.

        Args:
            request: LLM request object

        Returns:
            LLM response object
        """
        async with self.request_semaphore:
            request_start_time = time.time()
            request.started_at = datetime.now(timezone.utc)

            # Store active request
            self.active_requests[request.request_id] = request

            try:
                with self.tracer.trace("llm_generation") as span:
                    span.set_attributes(
                        {
                            "provider_type": self.provider_type.value,
                            "model_id": request.model_id,
                            "request_id": request.request_id,
                            "session_id": request.session_id or "unknown",
                            "user_id": request.user_id or "anonymous",
                            "streaming": request.streaming,
                            "temperature": request.temperature,
                        }
                    )

                    # Emit request started event
                    await self.event_bus.emit(
                        LLMRequestStarted(
                            provider_type=self.provider_type.value,
                            model_id=request.model_id,
                            request_id=request.request_id,
                            session_id=request.session_id,
                            user_id=request.user_id,
                            streaming=request.streaming,
                        )
                    )

                    # Pre-processing
                    await self._preprocess_request(request)

                    # Check usage limits
                    limit_check = self.token_manager.check_limits(request)
                    if not limit_check["allowed"]:
                        raise LLMError(
                            f"Usage limits exceeded: {limit_check['violations']}",
                            self.provider_type,
                            request.model_id,
                            request.request_id,
                        )

                    # Check cache
                    cached_response = None
                    if request.cache_enabled and self.cache_enabled:
                        cached_response = await self._check_cache(request)

                    if cached_response:
                        response = cached_response
                        response.from_cache = True
                        response.cache_hit = True

                        await self.event_bus.emit(
                            LLMCacheHit(
                                provider_type=self.provider_type.value,
                                request_id=request.request_id,
                                cache_key=self._generate_cache_key(request),
                            )
                        )
                    else:
                        # Generate response
                        if request.streaming:
                            response = await self._generate_streaming_response(request)
                        else:
                            response = await self._generate_response(request)

                        # Cache response
                        if request.cache_enabled and self.cache_enabled and not response.error:
                            await self._cache_response(request, response)

                        await self.event_bus.emit(
                            LLMCacheMiss(
                                provider_type=self.provider_type.value,
                                request_id=request.request_id,
                                cache_key=self._generate_cache_key(request),
                            )
                        )

                    # Post-processing
                    await self._postprocess_response(request, response)

                    # Update metrics
                    response.latency_ms = (time.time() - request_start_time) * 1000
                    response.tokens_per_second = response.completion_tokens / max(
                        response.latency_ms / 1000, 0.001
                    )

                    # Update token usage
                    self.token_manager.update_usage(request, response)

                    # Update metrics
                    if self.metrics:
                        self.metrics.increment(f"llm_{self.provider_type.value}_requests_total")
                        self.metrics.increment(
                            f"llm_{self.provider_type.value}_requests_successful"
                        )
                        self.metrics.record(
                            f"llm_{self.provider_type.value}_latency_seconds",
                            response.latency_ms / 1000,
                        )
                        self.metrics.record(
                            f"llm_{self.provider_type.value}_tokens_per_request",
                            response.total_tokens,
                        )
                        self.metrics.record(
                            f"llm_{self.provider_type.value}_tokens_per_second",
                            response.tokens_per_second,
                        )

                    # Track performance
                    self.performance_metrics[f"{request.model_id}_latency"].append(
                        response.latency_ms
                    )
                    self.performance_metrics[f"{request.model_id}_tokens_per_second"].append(
                        response.tokens_per_second
                    )
                    self.success_counts[request.model_id] += 1

                    # Emit completion event
                    await self.event_bus.emit(
                        LLMRequestCompleted(
                            provider_type=self.provider_type.value,
                            model_id=request.model_id,
                            request_id=request.request_id,
                            session_id=request.session_id,
                            user_id=request.user_id,
                            tokens_used=response.total_tokens,
                            cost=response.total_cost,
                            latency_ms=response.latency_ms,
                            from_cache=response.from_cache,
                        )
                    )

                    # Store in memory for learning
                    await self._store_interaction_for_learning(request, response)

                    request.completed_at = datetime.now(timezone.utc)
                    return response

            except Exception as e:
                # Handle error
                error_response = LLMResponse(
                    request_id=request.request_id,
                    provider_type=self.provider_type,
                    model_id=request.model_id,
                    error=str(e),
                    latency_ms=(time.time() - request_start_time) * 1000,
                )

                # Update error tracking
                self.error_counts[request.model_id] += 1

                # Update metrics
                if self.metrics:
                    self.metrics.increment(f"llm_{self.provider_type.value}_requests_failed")

                # Emit error event
                await self.event_bus.emit(
                    LLMRequestFailed(
                        provider_type=self.provider_type.value,
                        model_id=request.model_id,
                        request_id=request.request_id,
                        session_id=request.session_id,
                        error_message=str(e),
                        error_type=type(e).__name__,
                    )
                )

                # Retry logic
                if (
                    self.retry_enabled
                    and request.retry_count < request.max_retries
                    and not isinstance(e, (LLMError))
                ):

                    request.retry_count += 1
                    self.logger.warning(
                        f"Retrying request {request.request_id} "
                        f"({request.retry_count}/{request.max_retries}): {str(e)}"
                    )

                    # Exponential backoff
                    await asyncio.sleep(2**request.retry_count)
                    return await self.generate_completion(request)

                self.logger.error(f"Request {request.request_id} failed: {str(e)}")
                raise LLMError(str(e), self.provider_type, request.model_id, request.request_id)

            finally:
                # Cleanup
                self.active_requests.pop(request.request_id, None)
                self.request_history.append(request)

    @abstractmethod
    async def _generate_response(self, request: LLMRequest) -> LLMResponse:
        """Generate non-streaming response (implemented by subclasses)."""
        pass

    @abstractmethod
    async def _generate_streaming_response(self, request: LLMRequest) -> LLMResponse:
        """Generate streaming response (implemented by subclasses)."""
        pass

    async def generate_streaming(self, request: LLMRequest) -> AsyncGenerator[str, None]:
        """
        Generate streaming text completion.

        Args:
            request: LLM request object

        Yields:
            Text chunks as they are generated
        """
        request.streaming = True

        # Emit streaming started event
        await self.event_bus.emit(
            LLMStreamingStarted(
                provider_type=self.provider_type.value,
                model_id=request.model_id,
                request_id=request.request_id,
                session_id=request.session_id,
            )
        )

        try:
            async for chunk in self._stream_response(request):
                yield chunk

            # Emit streaming completed event
            await self.event_bus.emit(
                LLMStreamingCompleted(
                    provider_type=self.provider_type.value,
                    model_id=request.model_id,
                    request_id=request.request_id,
                    session_id=request.session_id,
                )
            )

        except Exception as e:
            self.logger.error(f"Streaming failed for request {request.request_id}: {str(e)}")
            raise LLMError(
                f"Streaming failed: {str(e)}",
                self.provider_type,
                request.model_id,
                request.request_id,
            )

    @abstractmethod
    async def _stream_response(self, request: LLMRequest) -> AsyncGenerator[str, None]:
        """Stream response chunks (implemented by subclasses)."""
        pass

    async def _preprocess_request(self, request: LLMRequest) -> None:
        """Preprocess request before sending to provider."""
        try:
            # Security and content filtering
            if self.security_sanitizer and request.content_filter:
                if request.prompt:
                    request.prompt = await self.security_sanitizer.sanitize_text(request.prompt)

                for message in request.messages:
                    if "content" in message:
                        message["content"] = await self.security_sanitizer.sanitize_text(
                            message["content"]
                        )

            # Context enhancement
            if request.session_id:
                session_context = await self.context_manager.get_session_context(request.session_id)
                request.context.update(session_context)

            # Token counting and validation
            model_info = self.supported_models.get(request.model_id)
            if model_info:
                # Estimate tokens
                total_prompt_text = request.prompt or ""
                for message in request.messages:
                    total_prompt_text += message.get("content", "")

                estimated_tokens = self.token_manager.count_tokens(
                    total_prompt_text, request.model_id
                )

                # Check context window
                if estimated_tokens > model_info.context_window:
                    await self.event_bus.emit(
                        LLMContextWindowExceeded(
                            provider_type=self.provider_type.value,
                            model_id=request.model_id,
                            estimated_tokens=estimated_tokens,
                            context_window=model_info.context_window,
                        )
                    )

                    # Truncate or handle appropriately
                    await self._handle_context_overflow(request, estimated_tokens, model_info)

            # Rate limiting check
            await self._check_rate_limits(request)

        except Exception as e:
            self.logger.error(f"Request preprocessing failed: {str(e)}")
            raise LLMError(f"Request preprocessing failed: {str(e)}", self.provider_type)

    async def _postprocess_response(self, request: LLMRequest, response: LLMResponse) -> None:
        """Postprocess response after receiving from provider."""
        try:
            # Content filtering and safety checks
            if request.safety_check and response.content:
                safety_result = await self._perform_safety_check(response.content)
                response.safety_scores = safety_result.get("scores", {})

                if safety_result.get("filtered", False):
                    response.content_filtered = True
                    response.content = safety_result.get("filtered_content", response.content)

            # Quality assessment
            if response.content:
                response.quality_score = await self._assess_response_quality(request, response)
                response.confidence_score = await self._calculate_confidence(request, response)

            # Cost calculation
            model_info = self.supported_models.get(request.model_id)
            if model_info:
                response.input_cost = (response.prompt_tokens / 1000) * model_info.input_cost_per_1k
                response.output_cost = (
                    response.completion_tokens / 1000
                ) * model_info.output_cost_per_1k
                response.total_cost = response.input_cost + response.output_cost

        except Exception as e:
            self.logger.warning(f"Response postprocessing failed: {str(e)}")

    async def _check_cache(self, request: LLMRequest) -> Optional[LLMResponse]:
        """Check if response is available in cache."""
        if not self.cache_strategy or not self.redis_cache:
            return None

        try:
            cache_key = self._generate_cache_key(request)
            cached_data = await self.redis_cache.get(cache_key)

            if cached_data:
                response_data = json.loads(cached_data)
                response = LLMResponse(**response_data)
                response.from_cache = True
                return response

        except Exception as e:
            self.logger.warning(f"Cache check failed: {str(e)}")

        return None

    async def _cache_response(self, request: LLMRequest, response: LLMResponse) -> None:
        """Cache response for future use."""
        if not self.cache_strategy or not self.redis_cache:
            return

        try:
            cache_key = self._generate_cache_key(request)
            cache_data = asdict(response)

            # Set TTL
            ttl = request.cache_ttl or 3600  # Default 1 hour

            await self.redis_cache.set(cache_key, json.dumps(cache_data, default=str), ttl=ttl)

        except Exception as e:
            self.logger.warning(f"Response caching failed: {str(e)}")

    def _generate_cache_key(self, request: LLMRequest) -> str:
        """Generate cache key for request."""
        # Create deterministic key based on request parameters
        key_data = {
            "provider": self.provider_type.value,
            "model": request.model_id,
            "prompt": request.prompt,
            "messages": request.messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "top_p": request.top_p,
            "frequency_penalty": request.frequency_penalty,
            "presence_penalty": request.presence_penalty,
            "stop_sequences": request.stop_sequences,
        }

        key_string = json.dumps(key_data, sort_keys=True)
        return f"llm_cache:{hashlib.md5(key_string.encode()).hexdigest()}"

    async def _handle_context_overflow(
        self, request: LLMRequest, estimated_tokens: int, model_info: ModelInfo
    ) -> None:
        """Handle context window overflow."""
        # Implement context truncation or summarization
        # This is a simplified version

        if request.messages:
            # Remove oldest messages to fit context
            while estimated_tokens > model_info.context_window and len(request.messages) > 1:
                removed_message = request.messages.pop(0)
                removed_tokens = self.token_manager.count_tokens(
                    removed_message.get("content", ""), request.model_id
                )
                estimated_tokens -= removed_tokens

        elif request.prompt:
            # Truncate prompt
            target_tokens = model_info.context_window - (request.max_tokens or 2048) - 100

            # Simple truncation (could be improved with smarter strategies)
            words = request.prompt.split()
            while estimated_tokens > target_tokens and words:
                words.pop(0)
                request.prompt = " ".join(words)
                estimated_tokens = self.token_manager.count_tokens(request.prompt, request.model_id)

    async def _check_rate_limits(self, request: LLMRequest) -> None:
        """Check and enforce rate limits."""
        current_time = time.time()

        # Reset window if needed
        if current_time - self.rate_limiter["window_start"] >= 60:
            self.rate_limiter["window_start"] = current_time
            self.rate_limiter["request_count"] = 0
            self.rate_limiter["token_count"] = 0

        # Check request rate limit
        model_info = self.supported_models.get(request.model_id)
        if model_info:
            if self.rate_limiter["request_count"] >= model_info.rate_limit_requests_per_minute:
                wait_time = 60 - (current_time - self.rate_limiter["window_start"])
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
                    await self._check_rate_limits(request)  # Retry after wait

        self.rate_limiter["request_count"] += 1
        self.rate_limiter["last_request_time"] = current_time

    async def _perform_safety_check(self, content: str) -> Dict[str, Any]:
        """Perform safety check on content."""
        # Implement content safety checking
        # This is a placeholder implementation
        return {"filtered": False, "scores": {"toxicity": 0.1, "hate": 0.05, "violence": 0.02}}

    async def _assess_response_quality(self, request: LLMRequest, response: LLMResponse) -> float:
        """Assess response quality."""
        # Implement quality assessment logic
        # This is a placeholder implementation
        quality_factors = []

        # Length appropriateness
        if response.content:
            length_ratio = len(response.content) / max(len(request.prompt or ""), 1)
            quality_factors.append(min(length_ratio / 2, 1.0))

        # Response completeness
        if response.finish_reason == "stop":
            quality_factors.append(1.0)
        elif response.finish_reason == "length":
            quality_factors.append(0.7)
        else:
            quality_factors.append(0.5)

        return sum(quality_factors) / len(quality_factors) if quality_factors else 0.5

    async def _calculate_confidence(self, request: LLMRequest, response: LLMResponse) -> float:
        """Calculate confidence score for response."""
        # Implement confidence calculation
        # This is a placeholder implementation
        confidence_factors = []

        # Model capability alignment
        model_info = self.supported_models.get(request.model_id)
        if model_info:
            confidence_factors.append(model_info.quality_score)

        # Token efficiency
        if response.total_tokens > 0:
            efficiency = response.completion_tokens / response.total_tokens
            confidence_factors.append(efficiency)

        # Finish reason
        if response.finish_reason == "stop":
            confidence_factors.append(0.9)
        else:
            confidence_factors.append(0.6)

        return sum(confidence_factors) / len(confidence_factors) if confidence_factors else 0.5

    async def _store_interaction_for_learning(
        self, request: LLMRequest, response: LLMResponse
    ) -> None:
        """Store interaction data for learning systems."""
        try:
            interaction_data = {
                "provider_type": self.provider_type.value,
                "model_id": request.model_id,
                "request_id": request.request_id,
                "session_id": request.session_id,
                "user_id": request.user_id,
                "prompt": request.prompt,
                "messages": request.messages,
                "response": response.content,
                "tokens_used": response.total_tokens,
                "latency_ms": response.latency_ms,
                "cost": response.total_cost,
                "quality_score": response.quality_score,
                "confidence_score": response.confidence_score,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            # Store in memory for learning
            if request.session_id:
                await self.memory_manager.store_interaction(
                    session_id=request.session_id,
                    interaction_type="llm_completion",
                    data=interaction_data,
                )

            # Update learning systems
            if self.continual_learner:
                await self.continual_learner.learn_from_llm_interaction(interaction_data)

            if self.preference_learner and request.user_id:
                await self.preference_learner.update_from_interaction(
                    request.user_id, interaction_data
                )

        except Exception as e:
            self.logger.warning(f"Failed to store interaction for learning: {str(e)}")

    def get_supported_models(self) -> List[ModelInfo]:
        """Get list of supported models."""
        return list(self.supported_models.values())

    def get_model_info(self, model_id: str) -> Optional[ModelInfo]:
        """Get information about a specific model."""
        return self.supported_models.get(model_id)

    def is_model_supported(self, model_id: str) -> bool:
        """Check if a model is supported."""
        return model_id in self.supported_models

    def get_provider_status(self) -> Dict[str, Any]:
        """Get comprehensive provider status."""
        return {
            "provider_type": self.provider_type.value,
            "status": self.status.value,
            "supported_models": len(self.supported_models),
            "active_requests": len(self.active_requests),
            "total_requests": sum(self.success_counts.values()) + sum(self.error_counts.values()),
            "success_rate": self._calculate_success_rate(),
            "average_latency_ms": self._calculate_average_latency(),
            "cache_enabled": self.cache_enabled,
            "streaming_enabled": self.streaming_enabled,
            "rate_limiter": self.rate_limiter,
        }

    def _calculate_success_rate(self) -> float:
        """Calculate overall success rate."""
        total_success = sum(self.success_counts.values())
        total_errors = sum(self.error_counts.values())
        total_requests = total_success + total_errors

        if total_requests == 0:
            return 0.0

        return total_success / total_requests

    def _calculate_average_latency(self) -> float:
        """Calculate average latency across all models."""
        all_latencies = []
        for latencies in self.performance_metrics.values():
            all_latencies.extend(latencies)

        if not all_latencies:
            return 0.0

        return sum(all_latencies) / len(all_latencies)

    def get_usage_statistics(self, session_id: str = None, user_id: str = None) -> Dict[str, Any]:
        """Get usage statistics."""
        return self.token_manager.get_usage_stats(session_id, user_id)

    async def _performance_monitoring_loop(self) -> None:
        """Background task for performance monitoring."""
        while True:
            try:
                # Update metrics
                if self.metrics:
                    self.metrics.set(
                        f"llm_{self.provider_type.value}_active_requests", len(self.active_requests)
                    )

                # Check for performance degradation
                avg_latency = self._calculate_average_latency()
                if avg_latency > 10000:  # 10 seconds
                    await self.event_bus.emit(
                        LLMPerformanceWarning(
                            provider_type=self.provider_type.value,
                            metric="latency",
                            value=avg_latency,
                            threshold=10000,
                        )
                    )

                # Check success rate
                success_rate = self._calculate_success_rate()
                if success_rate < 0.9:  # Below 90%
                    await self.event_bus.emit(
                        LLMPerformanceWarning(
                            provider_type=self.provider_type.value,
                            metric="success_rate",
                            value=success_rate,
                            threshold=0.9,
                        )
                    )

                await asyncio.sleep(30)  # Check every 30 seconds

            except Exception as e:
                self.logger.error(f"Performance monitoring error: {str(e)}")
                await asyncio.sleep(30)

    async def _cache_cleanup_loop(self) -> None:
        """Background task for cache cleanup."""
        while True:
            try:
                # Implement cache cleanup logic
                # This would depend on the specific cache implementation
                await asyncio.sleep(3600)  # Run every hour

            except Exception as e:
                self.logger.error(f"Cache cleanup error: {str(e)}")
                await asyncio.sleep(3600)

    async def _usage_tracking_loop(self) -> None:
        """Background task for usage tracking and cost monitoring."""
        while True:
            try:
                # Check cost thresholds
                stats = self.token_manager.get_usage_stats()
                global_cost = stats.get("global", {}).get("total_cost", 0.0)

                if global_cost > 500.0:  # $500 threshold
                    await self.event_bus.emit(
                        LLMCostThresholdExceeded(
                            provider_type=self.provider_type.value,
                            current_cost=global_cost,
                            threshold=500.0,
                            period="daily",
                        )
                    )

                # Reset daily counters if needed
                # Implement daily reset logic

                await asyncio.sleep(300)  # Check every 5 minutes

            except Exception as e:
                self.logger.error(f"Usage tracking error: {str(e)}")
                await asyncio.sleep(300)

    async def _health_monitoring_loop(self) -> None:
        """Background task for health monitoring."""
        while True:
            try:
                # Perform health checks
                health_result = await self._health_check_callback()

                new_status = ProviderStatus.HEALTHY
                if health_result.get("status") == "unhealthy":
                    new_status = ProviderStatus.UNHEALTHY
                elif health_result.get("status") == "degraded":
                    new_status = ProviderStatus.DEGRADED

                if new_status != self.status:
                    old_status = self.status
                    self.status = new_status

                    await self.event_bus.emit(
                        LLMProviderHealthChanged(
                            provider_type=self.provider_type.value,
                            old_status=old_status.value,
                            new_status=new_status.value,
                            health_details=health_result,
                        )
                    )

                await asyncio.sleep(60)  # Check every minute

            except Exception as e:
                self.logger.error(f"Health monitoring error: {str(e)}")
                await asyncio.sleep(60)

    async def _handle_system_shutdown(self, event) -> None:
        """Handle system shutdown event."""
        try:
            self.logger.info(f"Shutting down {self.provider_type.value} provider...")

            # Cancel active requests
            for request_id in list(self.active_requests.keys()):
                self.active_requests.pop(request_id, None)

            # Cleanup resources
            await self.cleanup()

        except Exception as e:
            self.logger.error(f"Error during shutdown: {str(e)}")

    async def _handle_session_started(self, event) -> None:
        """Handle session started event."""
        # Initialize session-specific tracking if needed
        pass

    async def _handle_session_ended(self, event) -> None:
        """Handle session ended event."""
        # Cleanup session-specific data
        session_id = event.session_id
        if session_id in self.token_manager.session_usage:
            # Archive session usage data
            session_usage = self.token_manager.session_usage.pop(session_id)
            # Store in database for historical analysis if needed

    async def _handle_feedback(self, event) -> None:
        """Handle user feedback event."""
        try:
            # Update learning systems with feedback
            if hasattr(event, "interaction_id") and hasattr(event, "feedback_data"):
                # Find the corresponding request
                for request in self.request_history:
                    if request.request_id == event.interaction_id:
                        # Update learning systems
                        if self.feedback_processor:
                            await self.feedback_processor.process_llm_feedback(
                                request, event.feedback_data
                            )
                        break

        except Exception as e:
            self.logger.error(f"Error handling feedback: {str(e)}")

    async def _health_check_callback(self) -> Dict[str, Any]:
        """Health check callback for the provider."""
        try:
            health_score = 1.0
            issues = []

            # Check active requests
            if len(self.active_requests) > self.max_concurrent_requests * 0.8:
                health_score -= 0.2
                issues.append("High request load")

            # Check success rate
            success_rate = self._calculate_success_rate()
            if success_rate < 0.9:
                health_score -= 0.3
                issues.append(f"Low success rate: {success_rate:.2f}")

            # Check average latency
            avg_latency = self._calculate_average_latency()
            if avg_latency > 5000:  # 5 seconds
                health_score -= 0.2
                issues.append(f"High latency: {avg_latency:.0f}ms")

            # Determine status
            if health_score >= 0.8:
                status = "healthy"
            elif health_score >= 0.5:
                status = "degraded"
            else:
                status = "unhealthy"

            return {
                "status": status,
                "health_score": health_score,
                "active_requests": len(self.active_requests),
                "success_rate": success_rate,
                "average_latency_ms": avg_latency,
                "supported_models": len(self.supported_models),
                "issues": issues,
                "provider_type": self.provider_type.value,
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "provider_type": self.provider_type.value,
            }

    async def cleanup(self) -> None:
        """Cleanup provider resources."""
        try:
            # Close any active connections
            if hasattr(self, "_client"):
                await self._client.close()

            # Clear caches
            self.response_cache.clear()

            # Reset metrics
            self.request_count = 0
            self.error_count = 0

            self.logger.info(f"Provider {self.provider_type} cleanup completed")

        except Exception as e:
            self.logger.error(f"Error during provider cleanup: {str(e)}")
