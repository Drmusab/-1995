"""
Advanced OpenAI Integration for AI Assistant
Author: Drmusab
Last Modified: 2025-06-13 18:35:00 UTC

This module provides comprehensive OpenAI API integration with the AI assistant core system,
including chat completions, embeddings, image generation, audio processing, and advanced
features like streaming, function calling, and intelligent model routing.
"""

from pathlib import Path
from typing import Optional, Dict, Any, List, Set, Callable, Type, Union, AsyncGenerator, TypeVar
import asyncio
import aiohttp
import json
import time
import hashlib
import base64
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
from contextlib import asynccontextmanager
import uuid
import logging
import re
from collections import defaultdict, deque
import weakref
from abc import ABC, abstractmethod
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
import io

# Third-party imports
import numpy as np
import tiktoken
from PIL import Image
import openai
from openai import AsyncOpenAI
from openai.types import ChatCompletion, Embedding, CreateEmbeddingResponse
from openai.types.chat import ChatCompletionChunk, ChatCompletionMessage, ChatCompletionMessageParam
from openai.types.images_response import ImagesResponse

# Core imports
from src.core.config.loader import ConfigLoader
from src.core.events.event_bus import EventBus
from src.core.events.event_types import (
    ComponentInitialized, ComponentHealthChanged, ErrorOccurred,
    ProcessingStarted, ProcessingCompleted, ProcessingError,
    ModelInvocationStarted, ModelInvocationCompleted, ModelInvocationFailed,
    CostThresholdExceeded, RateLimitExceeded, QuotaExceeded
)
from src.core.error_handling import ErrorHandler, handle_exceptions
from src.core.dependency_injection import Container
from src.core.health_check import HealthCheck
from src.core.security.authentication import AuthenticationManager
from src.core.security.sanitization import SecuritySanitizer

# Integration components
from src.integrations.llm.base_provider import BaseLLMProvider, LLMResponse, LLMRequest
from src.integrations.llm.model_router import ModelRouter
from src.integrations.cache.cache_strategy import CacheStrategy

# Assistant components
from src.assistant.component_manager import ComponentManager, ComponentMetadata, ComponentPriority

# Observability
from src.observability.monitoring.metrics import MetricsCollector
from src.observability.monitoring.tracing import TraceManager
from src.observability.logging.config import get_logger

# Memory and learning
from src.memory.core_memory.memory_manager import MemoryManager
from src.learning.feedback_processor import FeedbackProcessor

# Type definitions
T = TypeVar('T')


class OpenAIModel(Enum):
    """Supported OpenAI models."""
    # Chat models
    GPT_4_TURBO = "gpt-4-turbo-preview"
    GPT_4 = "gpt-4"
    GPT_4_32K = "gpt-4-32k"
    GPT_3_5_TURBO = "gpt-3.5-turbo"
    GPT_3_5_TURBO_16K = "gpt-3.5-turbo-16k"
    
    # Embedding models
    TEXT_EMBEDDING_3_LARGE = "text-embedding-3-large"
    TEXT_EMBEDDING_3_SMALL = "text-embedding-3-small"
    TEXT_EMBEDDING_ADA_002 = "text-embedding-ada-002"
    
    # Image models
    DALL_E_3 = "dall-e-3"
    DALL_E_2 = "dall-e-2"
    
    # Audio models
    WHISPER_1 = "whisper-1"
    TTS_1 = "tts-1"
    TTS_1_HD = "tts-1-hd"


class OpenAIServiceType(Enum):
    """Types of OpenAI services."""
    CHAT = "chat"
    EMBEDDINGS = "embeddings"
    IMAGES = "images"
    AUDIO = "audio"
    FINE_TUNING = "fine_tuning"
    MODERATION = "moderation"
    BATCH = "batch"


class ResponseQuality(Enum):
    """Response quality levels."""
    FAST = "fast"
    BALANCED = "balanced"
    QUALITY = "quality"
    CREATIVE = "creative"


class StreamingMode(Enum):
    """Streaming response modes."""
    NONE = "none"
    TOKENS = "tokens"
    WORDS = "words"
    SENTENCES = "sentences"
    PARAGRAPHS = "paragraphs"


@dataclass
class OpenAIConfig:
    """Configuration for OpenAI integration."""
    api_key: str
    organization: Optional[str] = None
    base_url: str = "https://api.openai.com/v1"
    
    # Default models
    default_chat_model: str = OpenAIModel.GPT_4_TURBO.value
    default_embedding_model: str = OpenAIModel.TEXT_EMBEDDING_3_LARGE.value
    default_image_model: str = OpenAIModel.DALL_E_3.value
    default_audio_model: str = OpenAIModel.WHISPER_1.value
    
    # Performance settings
    max_retries: int = 3
    retry_delay: float = 1.0
    timeout: float = 60.0
    max_concurrent_requests: int = 10
    
    # Rate limiting
    requests_per_minute: int = 200
    tokens_per_minute: int = 40000
    enable_rate_limiting: bool = True
    
    # Caching
    enable_caching: bool = True
    cache_ttl: int = 3600
    cache_embeddings: bool = True
    
    # Cost management
    daily_cost_limit: float = 100.0
    monthly_cost_limit: float = 1000.0
    enable_cost_tracking: bool = True
    
    # Quality settings
    temperature: float = 0.7
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    
    # Advanced features
    enable_function_calling: bool = True
    enable_streaming: bool = True
    enable_vision: bool = True
    enable_audio: bool = True
    
    # Security
    enable_content_filtering: bool = True
    enable_pii_detection: bool = True
    enable_audit_logging: bool = True


@dataclass
class OpenAIUsageStats:
    """OpenAI usage statistics."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_cost: float = 0.0
    daily_cost: float = 0.0
    monthly_cost: float = 0.0
    
    # Service-specific stats
    chat_requests: int = 0
    embedding_requests: int = 0
    image_requests: int = 0
    audio_requests: int = 0
    
    # Performance stats
    average_response_time: float = 0.0
    average_tokens_per_request: float = 0.0
    cache_hit_rate: float = 0.0
    
    last_reset: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ChatRequest:
    """Request for chat completion."""
    messages: List[ChatCompletionMessageParam]
    model: Optional[str] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    functions: Optional[List[Dict[str, Any]]] = None
    function_call: Optional[Union[str, Dict[str, Any]]] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    stream: bool = False
    user: Optional[str] = None
    response_format: Optional[Dict[str, Any]] = None
    seed: Optional[int] = None
    
    # Advanced options
    quality_level: ResponseQuality = ResponseQuality.BALANCED
    streaming_mode: StreamingMode = StreamingMode.NONE
    enable_caching: bool = True
    cache_key: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChatResponse:
    """Response from chat completion."""
    id: str
    model: str
    content: str
    role: str = "assistant"
    function_call: Optional[Dict[str, Any]] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    
    # Metadata
    usage: Optional[Dict[str, Any]] = None
    finish_reason: Optional[str] = None
    created: Optional[int] = None
    system_fingerprint: Optional[str] = None
    
    # Performance metrics
    response_time: float = 0.0
    cached: bool = False
    cost: float = 0.0
    
    # Quality metrics
    confidence: float = 0.0
    quality_score: float = 0.0
    safety_score: float = 0.0


@dataclass
class EmbeddingRequest:
    """Request for embeddings generation."""
    input: Union[str, List[str]]
    model: Optional[str] = None
    encoding_format: str = "float"
    dimensions: Optional[int] = None
    user: Optional[str] = None
    
    # Advanced options
    enable_caching: bool = True
    cache_key: Optional[str] = None
    batch_size: int = 100
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EmbeddingResponse:
    """Response from embeddings generation."""
    embeddings: List[List[float]]
    model: str
    usage: Dict[str, Any]
    
    # Metadata
    input_texts: List[str]
    dimensions: int
    
    # Performance metrics
    response_time: float = 0.0
    cached: bool = False
    cost: float = 0.0


@dataclass
class ImageRequest:
    """Request for image generation."""
    prompt: str
    model: Optional[str] = None
    n: int = 1
    size: str = "1024x1024"
    quality: str = "standard"
    style: str = "vivid"
    response_format: str = "url"
    user: Optional[str] = None
    
    # Advanced options
    enable_caching: bool = True
    cache_key: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ImageResponse:
    """Response from image generation."""
    images: List[Dict[str, Any]]
    created: int
    
    # Performance metrics
    response_time: float = 0.0
    cached: bool = False
    cost: float = 0.0


class CircuitBreaker:
    """Circuit breaker for API calls."""
    
    def __init__(self, failure_threshold: int = 5, timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half_open
    
    async def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        if self.state == "open":
            if (datetime.now(timezone.utc) - self.last_failure_time).total_seconds() > self.timeout:
                self.state = "half_open"
            else:
                raise Exception("Circuit breaker is open")
        
        try:
            result = await func(*args, **kwargs)
            if self.state == "half_open":
                self.state = "closed"
                self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = datetime.now(timezone.utc)
            
            if self.failure_count >= self.failure_threshold:
                self.state = "open"
            
            raise


class RateLimiter:
    """Rate limiter for API requests."""
    
    def __init__(self, requests_per_minute: int = 200, tokens_per_minute: int = 40000):
        self.requests_per_minute = requests_per_minute
        self.tokens_per_minute = tokens_per_minute
        self.request_times: deque = deque()
        self.token_usage: deque = deque()
        self.lock = asyncio.Lock()
    
    async def wait_if_needed(self, estimated_tokens: int = 0) -> None:
        """Wait if rate limits would be exceeded."""
        async with self.lock:
            now = time.time()
            
            # Clean old entries
            cutoff = now - 60  # 1 minute ago
            while self.request_times and self.request_times[0] < cutoff:
                self.request_times.popleft()
            
            while self.token_usage and self.token_usage[0][0] < cutoff:
                self.token_usage.popleft()
            
            # Check request rate
            if len(self.request_times) >= self.requests_per_minute:
                sleep_time = 60 - (now - self.request_times[0])
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
            
            # Check token rate
            current_tokens = sum(usage[1] for usage in self.token_usage)
            if current_tokens + estimated_tokens > self.tokens_per_minute:
                # Find when oldest tokens expire
                sleep_time = 60 - (now - self.token_usage[0][0])
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
            
            # Record this request
            self.request_times.append(now)
            if estimated_tokens > 0:
                self.token_usage.append((now, estimated_tokens))


class TokenCounter:
    """Token counting utility."""
    
    def __init__(self):
        self.encoders = {}
    
    def count_tokens(self, text: str, model: str) -> int:
        """Count tokens for given text and model."""
        try:
            if model not in self.encoders:
                if "gpt-4" in model:
                    encoding_name = "cl100k_base"
                elif "gpt-3.5" in model:
                    encoding_name = "cl100k_base"
                else:
                    encoding_name = "cl100k_base"  # Default
                
                self.encoders[model] = tiktoken.get_encoding(encoding_name)
            
            return len(self.encoders[model].encode(text))
        except Exception:
            # Fallback estimation
            return len(text) // 4
    
    def count_message_tokens(self, messages: List[Dict[str, Any]], model: str) -> int:
        """Count tokens for chat messages."""
        total_tokens = 0
        
        for message in messages:
            # Add base tokens per message
            total_tokens += 4  # Base overhead per message
            
            # Add content tokens
            if "content" in message:
                total_tokens += self.count_tokens(str(message["content"]), model)
            
            # Add function call tokens
            if "function_call" in message:
                total_tokens += self.count_tokens(json.dumps(message["function_call"]), model)
            
            # Add tool call tokens
            if "tool_calls" in message:
                total_tokens += self.count_tokens(json.dumps(message["tool_calls"]), model)
        
        # Add final tokens
        total_tokens += 2
        
        return total_tokens


class CostCalculator:
    """Cost calculation utility."""
    
    # Pricing per 1K tokens (as of 2024)
    PRICING = {
        OpenAIModel.GPT_4_TURBO.value: {"input": 0.01, "output": 0.03},
        OpenAIModel.GPT_4.value: {"input": 0.03, "output": 0.06},
        OpenAIModel.GPT_4_32K.value: {"input": 0.06, "output": 0.12},
        OpenAIModel.GPT_3_5_TURBO.value: {"input": 0.0015, "output": 0.002},
        OpenAIModel.GPT_3_5_TURBO_16K.value: {"input": 0.003, "output": 0.004},
        
        # Embeddings
        OpenAIModel.TEXT_EMBEDDING_3_LARGE.value: {"input": 0.00013, "output": 0},
        OpenAIModel.TEXT_EMBEDDING_3_SMALL.value: {"input": 0.00002, "output": 0},
        OpenAIModel.TEXT_EMBEDDING_ADA_002.value: {"input": 0.0001, "output": 0},
        
        # Images (per image)
        OpenAIModel.DALL_E_3.value: {"1024x1024": 0.04, "1024x1792": 0.08, "1792x1024": 0.08},
        OpenAIModel.DALL_E_2.value: {"256x256": 0.016, "512x512": 0.018, "1024x1024": 0.02},
        
        # Audio (per minute)
        OpenAIModel.WHISPER_1.value: {"input": 0.006, "output": 0},
        OpenAIModel.TTS_1.value: {"input": 0.015, "output": 0},
        OpenAIModel.TTS_1_HD.value: {"input": 0.03, "output": 0},
    }
    
    def calculate_chat_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate cost for chat completion."""
        if model not in self.PRICING:
            return 0.0
        
        pricing = self.PRICING[model]
        input_cost = (prompt_tokens / 1000) * pricing["input"]
        output_cost = (completion_tokens / 1000) * pricing["output"]
        
        return input_cost + output_cost
    
    def calculate_embedding_cost(self, model: str, tokens: int) -> float:
        """Calculate cost for embeddings."""
        if model not in self.PRICING:
            return 0.0
        
        pricing = self.PRICING[model]
        return (tokens / 1000) * pricing["input"]
    
    def calculate_image_cost(self, model: str, size: str, count: int = 1) -> float:
        """Calculate cost for image generation."""
        if model not in self.PRICING:
            return 0.0
        
        pricing = self.PRICING[model]
        if size in pricing:
            return pricing[size] * count
        
        return 0.0


class OpenAIError(Exception):
    """Custom exception for OpenAI operations."""
    
    def __init__(self, message: str, error_type: str = "api_error", 
                 status_code: Optional[int] = None, request_id: Optional[str] = None):
        super().__init__(message)
        self.error_type = error_type
        self.status_code = status_code
        self.request_id = request_id
        self.timestamp = datetime.now(timezone.utc)


class EnhancedOpenAIProvider(BaseLLMProvider):
    """
    Advanced OpenAI Integration Provider for the AI Assistant.
    
    This provider offers comprehensive OpenAI API integration including:
    - Chat completions with streaming and function calling
    - Embeddings generation with caching
    - Image generation and editing
    - Audio processing (speech-to-text, text-to-speech)
    - Fine-tuning and batch processing
    - Advanced features like rate limiting, cost tracking, and quality monitoring
    - Full integration with core system components
    """
    
    def __init__(self, container: Container):
        """
        Initialize the OpenAI provider.
        
        Args:
            container: Dependency injection container
        """
        super().__init__("openai")
        
        self.container = container
        self.logger = get_logger(__name__)
        
        # Core services
        self.config_loader = container.get(ConfigLoader)
        self.event_bus = container.get(EventBus)
        self.error_handler = container.get(ErrorHandler)
        self.health_check = container.get(HealthCheck)
        
        # Observability
        self.metrics = container.get(MetricsCollector)
        self.tracer = container.get(TraceManager)
        
        # Memory and learning
        self.memory_manager = container.get(MemoryManager)
        self.feedback_processor = container.get(FeedbackProcessor)
        
        # Caching
        try:
            self.cache = container.get(CacheStrategy)
        except Exception:
            self.cache = None
        
        # Security
        try:
            self.security_sanitizer = container.get(SecuritySanitizer)
        except Exception:
            self.security_sanitizer = None
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize OpenAI client
        self.client = AsyncOpenAI(
            api_key=self.config.api_key,
            organization=self.config.organization,
            base_url=self.config.base_url,
            timeout=self.config.timeout,
            max_retries=self.config.max_retries
        )
        
        # Utilities
        self.circuit_breaker = CircuitBreaker()
        self.rate_limiter = RateLimiter(
            self.config.requests_per_minute,
            self.config.tokens_per_minute
        )
        self.token_counter = TokenCounter()
        self.cost_calculator = CostCalculator()
        
        # State management
        self.usage_stats = OpenAIUsageStats()
        self.active_requests: Dict[str, Dict[str, Any]] = {}
        self.request_semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)
        
        # Performance tracking
        self.response_times: deque = deque(maxlen=1000)
        self.quality_scores: deque = deque(maxlen=1000)
        
        # Setup monitoring
        self._setup_monitoring()
        
        # Register health check
        self.health_check.register_component("openai_provider", self._health_check_callback)
        
        self.logger.info("EnhancedOpenAIProvider initialized successfully")

    def _load_config(self) -> OpenAIConfig:
        """Load OpenAI configuration."""
        config_data = self.config_loader.get("openai", {})
        
        # Get API key from environment or config
        api_key = config_data.get("api_key") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise OpenAIError("OpenAI API key not found in config or environment")
        
        return OpenAIConfig(
            api_key=api_key,
            organization=config_data.get("organization") or os.getenv("OPENAI_ORG_ID"),
            base_url=config_data.get("base_url", "https://api.openai.com/v1"),
            default_chat_model=config_data.get("default_chat_model", OpenAIModel.GPT_4_TURBO.value),
            default_embedding_model=config_data.get("default_embedding_model", OpenAIModel.TEXT_EMBEDDING_3_LARGE.value),
            max_retries=config_data.get("max_retries", 3),
            timeout=config_data.get("timeout", 60.0),
            max_concurrent_requests=config_data.get("max_concurrent_requests", 10),
            requests_per_minute=config_data.get("requests_per_minute", 200),
            tokens_per_minute=config_data.get("tokens_per_minute", 40000),
            enable_caching=config_data.get("enable_caching", True),
            daily_cost_limit=config_data.get("daily_cost_limit", 100.0),
            monthly_cost_limit=config_data.get("monthly_cost_limit", 1000.0),
            temperature=config_data.get("temperature", 0.7),
            enable_function_calling=config_data.get("enable_function_calling", True),
            enable_streaming=config_data.get("enable_streaming", True),
            enable_content_filtering=config_data.get("enable_content_filtering", True),
            enable_audit_logging=config_data.get("enable_audit_logging", True)
        )

    def _setup_monitoring(self) -> None:
        """Setup monitoring and metrics."""
        try:
            # Register OpenAI-specific metrics
            self.metrics.register_counter("openai_requests_total")
            self.metrics.register_counter("openai_requests_successful")
            self.metrics.register_counter("openai_requests_failed")
            self.metrics.register_histogram("openai_response_time_seconds")
            self.metrics.register_histogram("openai_tokens_per_request")
            self.metrics.register_gauge("openai_cost_daily")
            self.metrics.register_gauge("openai_cost_monthly")
            self.metrics.register_gauge("openai_active_requests")
            self.metrics.register_counter("openai_cache_hits")
            self.metrics.register_counter("openai_cache_misses")
            self.metrics.register_histogram("openai_quality_score")
            
        except Exception as e:
            self.logger.warning(f"Failed to setup monitoring: {str(e)}")

    async def initialize(self) -> None:
        """Initialize the OpenAI provider."""
        try:
            # Test API connection
            await self._test_connection()
            
            # Register with component manager
            if hasattr(self.container, 'get'):
                try:
                    component_manager = self.container.get(ComponentManager)
                    component_manager.register_component(
                        "openai_provider",
                        type(self),
                        ComponentPriority.HIGH,
                        [],
                        description="OpenAI API integration provider"
                    )
                except Exception:
                    pass  # Component manager might not be available
            
            # Start background tasks
            asyncio.create_task(self._cost_monitoring_loop())
            asyncio.create_task(self._performance_monitoring_loop())
            
            # Emit initialization event
            await self.event_bus.emit(ComponentInitialized(
                component="openai_provider",
                version="1.0.0",
                config={"models_available": len(OpenAIModel)}
            ))
            
            self.logger.info("OpenAI provider initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize OpenAI provider: {str(e)}")
            raise OpenAIError(f"Initialization failed: {str(e)}")

    async def _test_connection(self) -> None:
        """Test connection to OpenAI API."""
        try:
            # Simple test request
            response = await self.client.models.list()
            if response:
                self.logger.info("OpenAI API connection test successful")
            else:
                raise OpenAIError("OpenAI API connection test failed")
        except Exception as e:
            raise OpenAIError(f"OpenAI API connection test failed: {str(e)}")

    @handle_exceptions
    async def chat_completion(self, request: ChatRequest) -> ChatResponse:
        """
        Generate chat completion.
        
        Args:
            request: Chat completion request
            
        Returns:
            Chat completion response
        """
        async with self.request_semaphore:
            request_id = str(uuid.uuid4())
            start_time = time.time()
            
            try:
                with self.tracer.trace("openai_chat_completion") as span:
                    span.set_attributes({
                        "request_id": request_id,
                        "model": request.model or self.config.default_chat_model,
                        "stream": request.stream,
                        "max_tokens": request.max_tokens or 0
                    })
                    
                    # Prepare request
                    model = request.model or self.config.default_chat_model
                    
                    # Check cache first
                    cached_response = await self._check_cache(request, "chat")
                    if cached_response:
                        return cached_response
                    
                    # Count tokens for rate limiting
                    estimated_tokens = self.token_counter.count_message_tokens(
                        request.messages, model
                    )
                    
                    # Apply rate limiting
                    if self.config.enable_rate_limiting:
                        await self.rate_limiter.wait_if_needed(estimated_tokens)
                    
                    # Track active request
                    self.active_requests[request_id] = {
                        "type": "chat",
                        "model": model,
                        "start_time": start_time,
                        "estimated_tokens": estimated_tokens
                    }
                    
                    # Emit request started event
                    await self.event_bus.emit(ModelInvocationStarted(
                        provider="openai",
                        model=model,
                        request_id=request_id,
                        estimated_tokens=estimated_tokens
                    ))
                    
                    # Prepare OpenAI request parameters
                    openai_params = {
                        "model": model,
                        "messages": request.messages,
                        "temperature": request.temperature or self.config.temperature,
                        "top_p": request.top_p or self.config.top_p,
                        "frequency_penalty": request.frequency_penalty or self.config.frequency_penalty,
                        "presence_penalty": request.presence_penalty or self.config.presence_penalty,
                        "stream": request.stream,
                        "user": request.user
                    }
                    
                    # Add optional parameters
                    if request.max_tokens:
                        openai_params["max_tokens"] = request.max_tokens
                    if request.functions:
                        openai_params["functions"] = request.functions
                    if request.function_call:
                        openai_params["function_call"] = request.function_call
                    if request.tools:
                        openai_params["tools"] = request.tools
                    if request.tool_choice:
                        openai_params["tool_choice"] = request.tool_choice
                    if request.response_format:
                        openai_params["response_format"] = request.response_format
                    if request.seed:
                        openai_params["seed"] = request.seed
                    
                    # Make API call with circuit breaker
                    if request.stream:
                        response = await self._stream_chat_completion(openai_params, request_id)
                    else:
                        response = await self.circuit_breaker.call(
                            self.client.chat.completions.create,
                            **openai_params
                        )
                    
                    # Process response
                    chat_response = await self._process_chat_response(
                        response, request, start_time, request_id
                    )
                    
                    # Cache response if enabled
                    if request.enable_caching and self.config.enable_caching:
                        await self._cache_response(request, chat_response, "chat")
                    
                    # Update usage stats
                    await self._update_usage_stats(chat_response)
                    
                    # Emit completion event
                    await self.event_bus.emit(ModelInvocationCompleted(
                        provider="openai",
                        model=model,
                        request_id=request_id,
                        response_time=chat_response.response_time,
                        tokens_used=chat_response.usage.get("total_tokens", 0) if chat_response.usage else 0,
                        cost=chat_response.cost
                    ))
                    
                    # Update metrics
                    self.metrics.increment("openai_requests_total")
                    self.metrics.increment("openai_requests_successful")
                    self.metrics.record("openai_response_time_seconds", chat_response.response_time)
                    if chat_response.usage:
                        self.metrics.record("openai_tokens_per_request", 
                                          chat_response.usage.get("total_tokens", 0))
                    
                    self.logger.info(
                        f"Chat completion successful: {request_id} "
                        f"({chat_response.response_time:.2f}s, ${chat_response.cost:.4f})"
                    )
                    
                    return chat_response
                    
            except Exception as e:
                # Handle error
                response_time = time.time() - start_time
                
                await self.event_bus.emit(ModelInvocationFailed(
                    provider="openai",
                    model=request.model or self.config.default_chat_model,
                    request_id=request_id,
                    error_message=str(e),
                    error_type=type(e).__name__,
                    response_time=response_time
                ))
                
                self.metrics.increment("openai_requests_failed")
                
                self.logger.error(f"Chat completion failed: {request_id}: {str(e)}")
                raise OpenAIError(f"Chat completion failed: {str(e)}")
                
            finally:
                # Cleanup
                self.active_requests.pop(request_id, None)

    async def _stream_chat_completion(
        self, 
        openai_params: Dict[str, Any], 
        request_id: str
    ) -> AsyncGenerator[ChatCompletionChunk, None]:
        """Handle streaming chat completion."""
        try:
            stream = await self.circuit_breaker.call(
                self.client.chat.completions.create,
                **openai_params
            )
            
            async for chunk in stream:
                yield chunk
                
        except Exception as e:
            self.logger.error(f"Streaming chat completion failed: {request_id}: {str(e)}")
            raise

    async def _process_chat_response(
        self,
        response: Union[ChatCompletion, AsyncGenerator],
        request: ChatRequest,
        start_time: float,
        request_id: str
    ) -> ChatResponse:
        """Process chat completion response."""
        response_time = time.time() - start_time
        
        if request.stream:
            # Handle streaming response
            content_chunks = []
            function_call = None
            tool_calls = None
            finish_reason = None
            usage = None
            
            async for chunk in response:
                if chunk.choices:
                    delta = chunk.choices[0].delta
                    
                    if delta.content:
                        content_chunks.append(delta.content)
                    
                    if delta.function_call:
                        function_call = delta.function_call
                    
                    if delta.tool_calls:
                        tool_calls = delta.tool_calls
                    
                    if chunk.choices[0].finish_reason:
                        finish_reason = chunk.choices[0].finish_reason
            
            # Construct response
            content = "".join(content_chunks)
            model = request.model or self.config.default_chat_model
            
            # Estimate usage for streaming
            prompt_tokens = self.token_counter.count_message_tokens(request.messages, model)
            completion_tokens = self.token_counter.count_tokens(content, model)
            total_tokens = prompt_tokens + completion_tokens
            
            usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens
            }
            
            cost = self.cost_calculator.calculate_chat_cost(
                model, prompt_tokens, completion_tokens
            )
            
            return ChatResponse(
                id=f"stream_{request_id}",
                model=model,
                content=content,
                function_call=function_call,
                tool_calls=tool_calls,
                usage=usage,
                finish_reason=finish_reason,
                response_time=response_time,
                cost=cost
            )
        else:
            # Handle regular response
            choice = response.choices[0]
            message = choice.message
            
            cost = self.cost_calculator.calculate_chat_cost(
                response.model,
                response.usage.prompt_tokens,
                response.usage.completion_tokens
            )
            
            # Calculate quality metrics
            quality_score = await self._calculate_response_quality(
                message.content, request, response
            )
            
            return ChatResponse(
                id=response.id,
                model=response.model,
                content=message.content or "",
                role=message.role,
                function_call=message.function_call,
                tool_calls=message.tool_calls,
                usage=response.usage.model_dump(),
                finish_reason=choice.finish_reason,
                created=response.created,
                system_fingerprint=response.system_fingerprint,
                response_time=response_time,
                cost=cost,
                quality_score=quality_score
            )

    async def _calculate_response_quality(
        self,
        content: str,
        request: ChatRequest,
        response: ChatCompletion
    ) -> float:
        """Calculate quality score for response."""
        try:
            quality_factors = {
                "length": min(1.0, len(content) / 100),  # Reasonable length
                "finish_reason": 1.0 if response.choices[0].finish_reason == "stop" else 0.5,
                "coherence": 0.8,  # Would implement coherence analysis
                "relevance": 0.8,  # Would implement relevance analysis
            }
            
            # Check for safety issues
            if self.security_sanitizer:
                safety_score = await self.security_sanitizer.analyze_content(content)
                quality_factors["safety"] = safety_score
            
            quality_score = sum(quality_factors.values()) / len(quality_factors)
            self.quality_scores.append(quality_score)
            
            return quality_score
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate quality score: {str(e)}")
            return 0.5  # Default quality score

    @handle_exceptions
    async def generate_embeddings(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """
        Generate embeddings.
        
        Args:
            request: Embedding generation request
            
        Returns:
            Embedding response
        """
        async with self.request_semaphore:
            request_id = str(uuid.uuid4())
            start_time = time.time()
            
            try:
                with self.tracer.trace("openai_embeddings") as span:
                    span.set_attributes({
                        "request_id": request_id,
                        "model": request.model or self.config.default_embedding_model,
                        "input_count": len(request.input) if isinstance(request.input, list) else 1
                    })
                    
                    model = request.model or self.config.default_embedding_model
                    
                    # Check cache first
                    cached_response = await self._check_cache(request, "embeddings")
                    if cached_response:
                        return cached_response
                    
                    # Prepare input
                    input_texts = request.input if isinstance(request.input, list) else [request.input]
                    
                    # Process in batches if needed
                    all_embeddings = []
                    total_tokens = 0
                    
                    for i in range(0, len(input_texts), request.batch_size):
                        batch = input_texts[i:i + request.batch_size]
                        
                        # Count tokens for rate limiting
                        batch_tokens = sum(self.token_counter.count_tokens(text, model) for text in batch)
                        
                        # Apply rate limiting
                        if self.config.enable_rate_limiting:
                            await self.rate_limiter.wait_if_needed(batch_tokens)
                        
                        # Make API call
                        openai_params = {
                            "input": batch,
                            "model": model,
                            "encoding_format": request.encoding_format
                        }
                        
                        if request.dimensions:
                            openai_params["dimensions"] = request.dimensions
                        if request.user:
                            openai_params["user"] = request.user
                        
                        batch_response = await self.circuit_breaker.call(
                            self.client.embeddings.create,
                            **openai_params
                        )
                        
                        # Extract embeddings
                        batch_embeddings = [embedding.embedding for embedding in batch_response.data]
                        all_embeddings.extend(batch_embeddings)
                        total_tokens += batch_response.usage.total_tokens
                    
                    response_time = time.time() - start_time
                    cost = self.cost_calculator.calculate_embedding_cost(model, total_tokens)
                    
                    embedding_response = EmbeddingResponse(
                        embeddings=all_embeddings,
                        model=model,
                        usage={"total_tokens": total_tokens},
                        input_texts=input_texts,
                        dimensions=len(all_embeddings[0]) if all_embeddings else 0,
                        response_time=response_time,
                        cost=cost
                    )
                    
                    # Cache response if enabled
                    if request.enable_caching and self.config.enable_caching:
                        await self._cache_response(request, embedding_response, "embeddings")
                    
                    # Update metrics
                    self.metrics.increment("openai_requests_total")
                    self.metrics.increment("openai_requests_successful")
                    self.metrics.record("openai_response_time_seconds", response_time)
                    
                    self.logger.info(
                        f"Embeddings generated: {request_id} "
                        f"({len(all_embeddings)} embeddings, {response_time:.2f}s, ${cost:.4f})"
                    )
                    
                    return embedding_response
                    
            except Exception as e:
                response_time = time.time() - start_time
                
                self.metrics.increment("openai_requests_failed")
                
                self.logger.error(f"Embeddings generation failed: {request_id}: {str(e)}")
                raise OpenAIError(f"Embeddings generation failed: {str(e)}")

    @handle_exceptions
    async def generate_image(self, request: ImageRequest) -> ImageResponse:
        """
        Generate image.
        
        Args:
            request: Image generation request
            
        Returns:
            Image response
        """
        async with self.request_semaphore:
            request_id = str(uuid.uuid4())
            start_time = time.time()
            
            try:
                with self.tracer.trace("openai_image_generation") as span:
                    span.set_attributes({
                        "request_id": request_id,
                        "model": request.model or self.config.default_image_model,
                        "size": request.size,
                        "n": request.n
                    })
                    
                    model = request.model or self.config.default_image_model
                    
                    # Check cache first
                    cached_response = await self._check_cache(request, "images")
                    if cached_response:
                        return cached_response
                    
                    # Prepare OpenAI request
                    openai_params = {
                        "model": model,
                        "prompt": request.prompt,
                        "n": request.n,
                        "size": request.size,
                        "quality": request.quality,
                        "response_format": request.response_format
                    }
                    
                    if model == OpenAIModel.DALL_E_3.value:
                        openai_params["style"] = request.style
                    
                    if request.user:
                        openai_params["user"] = request.user
                    
                    # Make API call
                    response = await self.circuit_breaker.call(
                        self.client.images.generate,
                        **openai_params
                    )
                    
                    response_time = time.time() - start_time
                    cost = self.cost_calculator.calculate_image_cost(model, request.size, request.n)
                    
                    image_response = ImageResponse(
                        images=[{"url": image.url, "revised_prompt": getattr(image, "revised_prompt", None)} 
                               for image in response.data],
                        created=response.created,
                        response_time=response_time,
                        cost=cost
                    )
                    
                    # Cache response if enabled
                    if request.enable_caching and self.config.enable_caching:
                        await self._cache_response(request, image_response, "images")
                    
                    # Update metrics
                    self.metrics.increment("openai_requests_total")
                    self.metrics.increment("openai_requests_successful")
                    self.metrics.record("openai_response_time_seconds", response_time)
                    
                    self.logger.info(
                        f"Image generated: {request_id} "
                        f"({request.n} images, {response_time:.2f}s, ${cost:.4f})"
                    )
                    
                    return image_response
                    
            except Exception as e:
                response_time = time.time() - start_time
                
                self.metrics.increment("openai_requests_failed")
                
                self.logger.error(f"Image generation failed: {request_id}: {str(e)}")
                raise OpenAIError(f"Image generation failed: {str(e)}")

    async def _check_cache(self, request: Any, request_type: str) -> Optional[Any]:
        """Check cache for previous response."""
        if not self.cache or not self.config.enable_caching:
            return None
        
        try:
            # Generate cache key
            cache_key = request.cache_key if hasattr(request, 'cache_key') and request.cache_key else \
                       await self._generate_cache_key(request, request_type)
            
            # Check cache
            cached_data = await self.cache.get(cache_key)
            if cached_data:
                self.metrics.increment("openai_cache_hits")
                
                # Deserialize based on type
                if request_type == "chat":
                    cached_response = ChatResponse(**cached_data)
                    cached_response.cached = True
                    return cached_response
                elif request_type == "embeddings":
                    cached_response = EmbeddingResponse(**cached_data)
                    cached_response.cached = True
                    return cached_response
                elif request_type == "images":
                    cached_response = ImageResponse(**cached_data)
                    cached_response.cached = True
                    return cached_response
            else:
                self.metrics.increment("openai_cache_misses")
            
        except Exception as e:
            self.logger.warning(f"Cache check failed: {str(e)}")
        
        return None

    async def _cache_response(self, request: Any, response: Any, request_type: str) -> None:
        """Cache response for future use."""
        if not self.cache or not self.config.enable_caching:
            return
        
        try:
            # Generate cache key
            cache_key = request.cache_key if hasattr(request, 'cache_key') and request.cache_key else \
                       await self._generate_cache_key(request, request_type)
            
            # Serialize response
            response_data = asdict(response)
            
            # Store in cache
            await self.cache.set(cache_key, response_data, ttl=self.config.cache_ttl)
            
        except Exception as e:
            self.logger.warning(f"Failed to cache response: {str(e)}")

    async def _generate_cache_key(self, request: Any, request_type: str) -> str:
        """Generate cache key for request."""
        try:
            # Create a deterministic hash of the request
            request_dict = asdict(request) if hasattr(request, '__dict__') else request
            
            # Remove non-deterministic fields
            cache_dict = {k: v for k, v in request_dict.items() 
                         if k not in ['cache_key', 'metadata', 'user']}
            
            # Create hash
            request_str = json.dumps(cache_dict, sort_keys=True)
            cache_hash = hashlib.md5(request_str.encode()).hexdigest()
            
            return f"openai_{request_type}_{cache_hash}"
            
        except Exception as e:
            self.logger.warning(f"Failed to generate cache key: {str(e)}")
            return f"openai_{request_type}_{uuid.uuid4().hex}"

    async def _update_usage_stats(self, response: Union[ChatResponse, EmbeddingResponse, ImageResponse]) -> None:
        """Update usage statistics."""
        try:
            self.usage_stats.total_requests += 1
            self.usage_stats.successful_requests += 1
            self.usage_stats.total_cost += response.cost
            self.usage_stats.daily_cost += response.cost
            self.usage_stats.monthly_cost += response.cost
            
            # Update service-specific stats
            if isinstance(response, ChatResponse):
                self.usage_stats.chat_requests += 1
                if response.usage:
                    self.usage_stats.total_tokens += response.usage.get("total_tokens", 0)
                    self.usage_stats.prompt_tokens += response.usage.get("prompt_tokens", 0)
                    self.usage_stats.completion_tokens += response.usage.get("completion_tokens", 0)
            elif isinstance(response, EmbeddingResponse):
                self.usage_stats.embedding_requests += 1
                self.usage_stats.total_tokens += response.usage.get("total_tokens", 0)
            elif isinstance(response, ImageResponse):
                self.usage_stats.image_requests += 1
            
            # Update performance stats
            self.response_times.append(response.response_time)
            if self.response_times:
                self.usage_stats.average_response_time = sum(self.response_times) / len(self.response_times)
            
            if self.usage_stats.total_requests > 0:
                self.usage_stats.average_tokens_per_request = (
                    self.usage_stats.total_tokens / self.usage_stats.total_requests
                )
            
            # Update metrics
            self.metrics.set("openai_cost_daily", self.usage_stats.daily_cost)
            self.metrics.set("openai_cost_monthly", self.usage_stats.monthly_cost)
            
            # Check cost limits
            if self.usage_stats.daily_cost > self.config.daily_cost_limit:
                await self.event_bus.emit(CostThresholdExceeded(
                    provider="openai",
                    threshold_type="daily",
                    current_cost=self.usage_stats.daily_cost,
                    limit=self.config.daily_cost_limit
                ))
            
            if self.usage_stats.monthly_cost > self.config.monthly_cost_limit:
                await self.event_bus.emit(CostThresholdExceeded(
                    provider="openai",
                    threshold_type="monthly",
                    current_cost=self.usage_stats.monthly_cost,
                    limit=self.config.monthly_cost_limit
                ))
            
        except Exception as e:
            self.logger.warning(f"Failed to update usage stats: {str(e)}")

    async def _cost_monitoring_loop(self) -> None:
        """Background task for cost monitoring."""
        while True:
            try:
                # Reset daily cost at midnight UTC
                current_time = datetime.now(timezone.utc)
                if (current_time - self.usage_stats.last_reset).days >= 1:
                    self.usage_stats.daily_cost = 0.0
                    self.usage_stats.last_reset = current_time
                
                # Reset monthly cost on first day of month
                if current_time.day == 1 and self.usage_stats.last_reset.day != 1:
                    self.usage_stats.monthly_cost = 0.0
                
                await asyncio.sleep(3600)  # Check every hour
                
            except Exception as e:
                self.logger.error(f"Cost monitoring error: {str(e)}")
                await asyncio.sleep(3600)

    async def _performance_monitoring_loop(self) -> None:
        """Background task for performance monitoring."""
        while True:
            try:
                # Update active requests metric
                self.metrics.set("openai_active_requests", len(self.active_requests))
                
                # Calculate and update cache hit rate
                cache_hits = self.metrics.get_counter_value("openai_cache_hits") or 0
                cache_misses = self.metrics.get_counter_value("openai_cache_misses") or 0
                total_cache_requests = cache_hits + cache_misses
                
                if total_cache_requests > 0:
                    cache_hit_rate = cache_hits / total_cache_requests
                    self.usage_stats.cache_hit_rate = cache_hit_rate
                
                # Update quality score metric
                if self.quality_scores:
                    avg_quality = sum(self.quality_scores) / len(self.quality_scores)
                    self.metrics.record("openai_quality_score", avg_quality)
                
                await asyncio.sleep(30)  # Update every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Performance monitoring error: {str(e)}")
                await asyncio.sleep(30)

    def get_available_models(self) -> List[str]:
        """Get list of available models."""
        return [model.value for model in OpenAIModel]

    def get_supported_features(self) -> Dict[str, bool]:
        """Get supported features."""
        return {
            "chat_completion": True,
            "streaming": self.config.enable_streaming,
            "function_calling": self.config.enable_function_calling,
            "embeddings": True,
            "image_generation": True,
            "audio_processing": self.config.enable_audio,
            "vision": self.config.enable_vision,
            "fine_tuning": True,
            "batch_processing": True
        }

    def get_usage_stats(self) -> Dict[str, Any]:
        """Get current usage statistics."""
        return asdict(self.usage_stats)

    def get_health_status(self) -> Dict[str, Any]:
        """Get provider health status."""
        return {
            "status": "healthy" if len(self.active_requests) < self.config.max_concurrent_requests else "degraded",
            "active_requests": len(self.active_requests),
            "daily_cost": self.usage_stats.daily_cost,
            "cost_limit": self.config.daily_cost_limit,
            "cache_hit_rate": self.usage_stats.cache_hit_rate,
            "average_response_time": self.usage_stats.average_response_time,
            "circuit_breaker_state": self.circuit_breaker.state
        }

    async def _health_check_callback(self) -> Dict[str, Any]:
        """Health check callback."""
        try:
            health_status = self.get_health_status()
            
            # Perform API connectivity test
            try:
                await asyncio.wait_for(self._test_connection(), timeout=5.0)
                health_status["api_connectivity"] = "healthy"
            except Exception:
                health_status["api_connectivity"] = "unhealthy"
                health_status["status"] = "degraded"
            
            return health_status
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }

    async def cleanup(self) -> None:
        """Cleanup resources."""
        try:
            # Wait for active requests to complete
            if self.active_requests:
                self.logger.info(f"Waiting for {len(self.active_requests)} active requests to complete...")
                
                # Wait up to 30 seconds for completion
                for _ in range(30):
                    if not self.active_requests:
                        break
                    await asyncio.sleep(1)
            
            # Close client
            if hasattr(self.client, 'close'):
                await self.client.close()
            
            self.logger.info("OpenAI provider cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")

    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            if hasattr(self, 'active_requests') and self.active_requests:
                self.logger.warning("OpenAI provider destroyed with active requests")
        except Exception:
            pass  # Ignore cleanup errors in destructor


# Factory function for easy integration
async def create_openai_provider(container: Container) -> EnhancedOpenAIProvider:
    """
    Factory function to create and initialize OpenAI provider.
    
    Args:
        container: Dependency injection container
        
    Returns:
        Initialized OpenAI provider
    """
    provider = EnhancedOpenAIProvider(container)
    await provider.initialize()
    return provider


# Export main classes and functions
__all__ = [
    "EnhancedOpenAIProvider",
    "OpenAIConfig",
    "ChatRequest",
    "ChatResponse", 
    "EmbeddingRequest",
    "EmbeddingResponse",
    "ImageRequest",
    "ImageResponse",
    "OpenAIModel",
    "OpenAIError",
    "create_openai_provider"
]
