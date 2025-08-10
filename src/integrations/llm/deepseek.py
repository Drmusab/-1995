"""
DeepSeek AI Provider Integration
Author: Drmusab
Last Modified: 2025-01-13 18:17:51 UTC

This module provides integration with DeepSeek AI language models,
supporting text generation, reasoning, and code generation capabilities
with comprehensive monitoring and error handling.
"""

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

import aiohttp
import asyncio

# Core imports
from src.core.dependency_injection import Container
from src.core.error_handling import handle_exceptions
from src.core.events.event_types import (
    LLMRateLimitExceeded,
    LLMRequestCompleted,
    LLMRequestFailed,
    LLMRequestStarted,
)

# Base provider and types
from src.integrations.llm.base_provider import (
    BaseLLMProvider,
    LLMError,
    LLMProviderType,
    LLMRequest,
    LLMResponse,
    ModelCapability,
    ModelInfo,
    ProviderStatus,
)


class DeepSeekModel(Enum):
    """DeepSeek model identifiers."""

    DEEPSEEK_CHAT = "deepseek-chat"
    DEEPSEEK_CODER = "deepseek-coder"
    DEEPSEEK_MATH = "deepseek-math"
    DEEPSEEK_REASONING = "deepseek-reasoning"


class DeepSeekError(LLMError):
    """DeepSeek-specific error."""

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        status_code: Optional[int] = None,
        response_data: Optional[Dict] = None,
    ):
        super().__init__(message, LLMProviderType.DEEPSEEK, error_code=error_code)
        self.status_code = status_code
        self.response_data = response_data


@dataclass
class DeepSeekConfig:
    """DeepSeek provider configuration."""

    api_key: str
    base_url: str = "https://api.deepseek.com"
    api_version: str = "v1"
    timeout: float = 30.0
    max_retries: int = 3
    retry_delay: float = 1.0

    # Model configurations
    default_model: str = "deepseek-chat"
    max_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 0.9

    # Performance settings
    concurrent_requests: int = 10
    rate_limit_rpm: int = 60
    rate_limit_tpm: int = 150000

    # Caching
    enable_caching: bool = True
    cache_ttl: int = 3600

    # Security
    content_filtering: bool = True
    safety_checks: bool = True


class DeepSeekProvider(BaseLLMProvider):
    """
    DeepSeek AI Provider for language model integration.

    This provider supports:
    - Text generation and completion
    - Code generation and debugging
    - Mathematical reasoning
    - Complex reasoning tasks
    - Streaming responses
    - Function calling
    - Token usage tracking
    - Error handling and retries
    - Performance monitoring
    """

    def __init__(self, container: Container, config: Optional[DeepSeekConfig] = None):
        """
        Initialize DeepSeek provider.

        Args:
            container: Dependency injection container
            config: Provider configuration
        """
        # Initialize base provider
        super().__init__(provider_type=LLMProviderType.DEEPSEEK, container=container)

        # Configuration
        self.config = config or self._load_config()

        # HTTP session for API calls
        self.session: Optional[aiohttp.ClientSession] = None
        self.connector: Optional[aiohttp.TCPConnector] = None

        # Rate limiting
        self.request_semaphore = asyncio.Semaphore(self.config.concurrent_requests)
        self.rate_limiter = self._setup_rate_limiter()

        # Model registry
        self.model_registry: Dict[str, ModelInfo] = {}

        # Performance tracking
        self.request_count = 0
        self.total_tokens = 0
        self.total_cost = 0.0

        self.logger.info("DeepSeek provider initialized")

    def _load_config(self) -> DeepSeekConfig:
        """Load configuration from config loader."""
        try:
            config_data = self.config_loader.get("integrations.llm.deepseek", {})

            # Validate required fields
            if not config_data.get("api_key"):
                raise DeepSeekError("DeepSeek API key not configured")

            return DeepSeekConfig(**config_data)

        except Exception as e:
            self.logger.error(f"Failed to load DeepSeek configuration: {str(e)}")
            raise DeepSeekError(f"Configuration error: {str(e)}")

    def _setup_rate_limiter(self) -> Dict[str, Any]:
        """Setup rate limiting for API requests."""
        return {
            "requests_per_minute": self.config.rate_limit_rpm,
            "tokens_per_minute": self.config.rate_limit_tpm,
            "request_window": deque(maxlen=self.config.rate_limit_rpm),
            "token_window": deque(maxlen=1000),  # Track recent token usage
            "last_reset": datetime.now(timezone.utc),
        }

    async def _load_supported_models(self) -> None:
        """Load supported models and their capabilities."""
        try:
            # Define DeepSeek models and their capabilities
            models = {
                DeepSeekModel.DEEPSEEK_CHAT.value: ModelInfo(
                    model_id="deepseek-chat",
                    name="DeepSeek Chat",
                    provider_type=LLMProviderType.DEEPSEEK,
                    capabilities={
                        ModelCapability.TEXT_GENERATION,
                        ModelCapability.CONVERSATION,
                        ModelCapability.REASONING,
                        ModelCapability.MULTIMODAL,
                    },
                    max_tokens=4096,
                    context_window=4096,
                    max_output_tokens=4096,
                    input_cost_per_1k=0.14,
                    output_cost_per_1k=0.28,
                    average_latency_ms=1200,
                    throughput_tokens_per_second=50,
                    description="General-purpose conversational AI model",
                    supported_languages=["en", "zh", "ja", "ko"],
                ),
                DeepSeekModel.DEEPSEEK_CODER.value: ModelInfo(
                    model_id="deepseek-coder",
                    name="DeepSeek Coder",
                    provider_type=LLMProviderType.DEEPSEEK,
                    capabilities={
                        ModelCapability.CODE_GENERATION,
                        ModelCapability.CODE_COMPLETION,
                        ModelCapability.TEXT_GENERATION,
                        ModelCapability.REASONING,
                    },
                    max_tokens=8192,
                    context_window=8192,
                    max_output_tokens=4096,
                    input_cost_per_1k=0.14,
                    output_cost_per_1k=0.28,
                    average_latency_ms=1500,
                    throughput_tokens_per_second=45,
                    description="Specialized model for code generation and programming tasks",
                    supported_languages=["en"],
                ),
                DeepSeekModel.DEEPSEEK_MATH.value: ModelInfo(
                    model_id="deepseek-math",
                    name="DeepSeek Math",
                    provider_type=LLMProviderType.DEEPSEEK,
                    capabilities={
                        ModelCapability.REASONING,
                        ModelCapability.MATHEMATICAL,
                        ModelCapability.TEXT_GENERATION,
                    },
                    max_tokens=4096,
                    context_window=4096,
                    max_output_tokens=2048,
                    input_cost_per_1k=0.14,
                    output_cost_per_1k=0.28,
                    average_latency_ms=2000,
                    throughput_tokens_per_second=40,
                    description="Mathematical reasoning and problem-solving model",
                    supported_languages=["en"],
                ),
                DeepSeekModel.DEEPSEEK_REASONING.value: ModelInfo(
                    model_id="deepseek-reasoning",
                    name="DeepSeek Reasoning",
                    provider_type=LLMProviderType.DEEPSEEK,
                    capabilities={
                        ModelCapability.REASONING,
                        ModelCapability.LOGICAL_INFERENCE,
                        ModelCapability.TEXT_GENERATION,
                        ModelCapability.ANALYSIS,
                    },
                    max_tokens=4096,
                    context_window=4096,
                    max_output_tokens=4096,
                    input_cost_per_1k=0.20,
                    output_cost_per_1k=0.40,
                    average_latency_ms=2500,
                    throughput_tokens_per_second=35,
                    description="Advanced reasoning and logical inference model",
                    supported_languages=["en"],
                ),
            }

            # Store in supported models
            self.supported_models.update(models)

            # Also store in model registry for quick access
            self.model_registry.update(models)

            self.logger.info(f"Loaded {len(models)} DeepSeek models")

        except Exception as e:
            self.logger.error(f"Failed to load supported models: {str(e)}")
            raise DeepSeekError(f"Model loading failed: {str(e)}")

    async def _initialize_provider(self) -> None:
        """Initialize provider-specific components."""
        try:
            # Setup HTTP session
            self.connector = aiohttp.TCPConnector(
                limit=self.config.concurrent_requests,
                limit_per_host=self.config.concurrent_requests,
                ttl_dns_cache=300,
                ttl_connection_pool=300,
                enable_cleanup_closed=True,
            )

            timeout = aiohttp.ClientTimeout(total=self.config.timeout)

            self.session = aiohttp.ClientSession(
                connector=self.connector,
                timeout=timeout,
                headers={
                    "Authorization": f"Bearer {self.config.api_key}",
                    "Content-Type": "application/json",
                    "User-Agent": "AI-Assistant/1.0.0",
                },
            )

            # Test API connectivity
            await self._test_connectivity()

            # Load supported models
            await self._load_supported_models()

            self.status = ProviderStatus.READY
            self.logger.info("DeepSeek provider initialized successfully")

        except Exception as e:
            self.status = ProviderStatus.ERROR
            self.logger.error(f"Failed to initialize DeepSeek provider: {str(e)}")
            raise DeepSeekError(f"Provider initialization failed: {str(e)}")

    async def _test_connectivity(self) -> None:
        """Test API connectivity."""
        try:
            url = f"{self.config.base_url}/{self.config.api_version}/models"

            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    self.logger.info(
                        f"API connectivity test successful. Available models: {len(data.get('data', []))}"
                    )
                elif response.status == 401:
                    raise DeepSeekError("Invalid API key", "INVALID_API_KEY", response.status)
                else:
                    raise DeepSeekError(
                        f"API connectivity test failed: {response.status}",
                        "CONNECTIVITY_ERROR",
                        response.status,
                    )

        except aiohttp.ClientError as e:
            raise DeepSeekError(
                f"Network error during connectivity test: {str(e)}", "NETWORK_ERROR"
            )

    async def _check_rate_limits(self, request: LLMRequest) -> None:
        """Check and enforce rate limits."""
        current_time = datetime.now(timezone.utc)

        # Clean old entries from rate limit windows
        minute_ago = current_time.timestamp() - 60

        # Remove old request timestamps
        while (
            self.rate_limiter["request_window"]
            and self.rate_limiter["request_window"][0] < minute_ago
        ):
            self.rate_limiter["request_window"].popleft()

        # Check request rate limit
        if len(self.rate_limiter["request_window"]) >= self.config.rate_limit_rpm:
            await self.event_bus.emit(
                LLMRateLimitExceeded(
                    provider_type=self.provider_type.value,
                    limit_type="requests_per_minute",
                    current_usage=len(self.rate_limiter["request_window"]),
                    limit=self.config.rate_limit_rpm,
                )
            )
            raise DeepSeekError("Rate limit exceeded: requests per minute", "RATE_LIMIT_RPM")

        # Add current request to window
        self.rate_limiter["request_window"].append(current_time.timestamp())

    async def _make_api_request(
        self, endpoint: str, data: Dict[str, Any], stream: bool = False
    ) -> Union[Dict[str, Any], AsyncGenerator[Dict[str, Any], None]]:
        """Make API request to DeepSeek."""
        url = f"{self.config.base_url}/{self.config.api_version}/{endpoint}"

        try:
            if stream:
                return self._stream_api_request(url, data)
            else:
                async with self.session.post(url, json=data) as response:
                    return await self._handle_response(response)

        except aiohttp.ClientError as e:
            raise DeepSeekError(f"Network error: {str(e)}", "NETWORK_ERROR")

    async def _handle_response(self, response: aiohttp.ClientResponse) -> Dict[str, Any]:
        """Handle API response."""
        if response.status == 200:
            return await response.json()

        elif response.status == 401:
            raise DeepSeekError("Invalid API key", "INVALID_API_KEY", response.status)

        elif response.status == 429:
            # Rate limit exceeded
            retry_after = response.headers.get("Retry-After", "60")
            raise DeepSeekError(
                f"Rate limit exceeded. Retry after {retry_after} seconds",
                "RATE_LIMIT_EXCEEDED",
                response.status,
            )

        elif response.status == 400:
            error_data = await response.json()
            raise DeepSeekError(
                f"Bad request: {error_data.get('error', {}).get('message', 'Unknown error')}",
                "BAD_REQUEST",
                response.status,
                error_data,
            )

        elif response.status == 500:
            raise DeepSeekError("Internal server error", "SERVER_ERROR", response.status)

        else:
            error_text = await response.text()
            raise DeepSeekError(
                f"Unexpected status code {response.status}: {error_text}",
                "UNEXPECTED_ERROR",
                response.status,
            )

    async def _stream_api_request(
        self, url: str, data: Dict[str, Any]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream API response for real-time processing."""
        data["stream"] = True

        async with self.session.post(url, json=data) as response:
            if response.status != 200:
                await self._handle_response(response)

            async for line in response.content:
                line = line.decode("utf-8").strip()

                if line.startswith("data: "):
                    data_str = line[6:]

                    if data_str == "[DONE]":
                        break

                    try:
                        chunk_data = json.loads(data_str)
                        yield chunk_data
                    except json.JSONDecodeError:
                        continue

    @handle_exceptions
    async def generate_text(
        self,
        prompt: str,
        model_id: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stream: bool = False,
        **kwargs,
    ) -> Union[LLMResponse, AsyncGenerator[str, None]]:
        """
        Generate text using DeepSeek models.

        Args:
            prompt: Input prompt for generation
            model_id: Model identifier to use
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            stream: Enable streaming response
            **kwargs: Additional parameters

        Returns:
            LLM response or async generator for streaming
        """
        async with self.request_semaphore:
            request_id = kwargs.get("request_id", str(uuid.uuid4()))
            start_time = time.time()

            # Prepare request
            model_id = model_id or self.config.default_model

            if model_id not in self.supported_models:
                raise DeepSeekError(f"Unsupported model: {model_id}", "UNSUPPORTED_MODEL")

            # Check rate limits
            await self._check_rate_limits(None)  # Simplified for this method

            # Prepare API data
            api_data = {
                "model": model_id,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens or self.config.max_tokens,
                "temperature": temperature if temperature is not None else self.config.temperature,
                "top_p": top_p if top_p is not None else self.config.top_p,
                "stream": stream,
            }

            # Add additional parameters
            for key, value in kwargs.items():
                if key not in ["request_id"] and value is not None:
                    api_data[key] = value

            try:
                # Emit request started event
                await self.event_bus.emit(
                    LLMRequestStarted(
                        provider_type=self.provider_type.value,
                        model_id=model_id,
                        request_id=request_id,
                        streaming=stream,
                    )
                )

                if stream:
                    return self._handle_streaming_response(api_data, request_id, start_time)
                else:
                    response_data = await self._make_api_request("chat/completions", api_data)
                    return await self._create_response(
                        response_data, request_id, start_time, model_id
                    )

            except Exception as e:
                # Emit error event
                await self.event_bus.emit(
                    LLMRequestFailed(
                        provider_type=self.provider_type.value,
                        model_id=model_id,
                        request_id=request_id,
                        error_message=str(e),
                        error_type=type(e).__name__,
                    )
                )
                raise

    async def _handle_streaming_response(
        self, api_data: Dict[str, Any], request_id: str, start_time: float
    ) -> AsyncGenerator[str, None]:
        """Handle streaming response generation."""
        try:
            async for chunk in self._make_api_request("chat/completions", api_data, stream=True):
                if "choices" in chunk and chunk["choices"]:
                    choice = chunk["choices"][0]
                    if "delta" in choice and "content" in choice["delta"]:
                        content = choice["delta"]["content"]
                        if content:
                            yield content

        except Exception as e:
            self.logger.error(f"Streaming error: {str(e)}")
            raise DeepSeekError(f"Streaming failed: {str(e)}")

    async def _create_response(
        self, response_data: Dict[str, Any], request_id: str, start_time: float, model_id: str
    ) -> LLMResponse:
        """Create LLM response from API response data."""
        try:
            processing_time = time.time() - start_time

            # Extract response content
            choice = response_data["choices"][0]
            text = choice["message"]["content"]
            finish_reason = choice.get("finish_reason", "stop")

            # Extract usage information
            usage = response_data.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)

            # Calculate costs
            model_info = self.supported_models.get(model_id)
            input_cost = 0.0
            output_cost = 0.0

            if model_info:
                input_cost = (prompt_tokens / 1000) * model_info.input_cost_per_1k
                output_cost = (completion_tokens / 1000) * model_info.output_cost_per_1k

            total_cost = input_cost + output_cost

            # Create response
            response = LLMResponse(
                request_id=request_id,
                provider_type=self.provider_type,
                model_id=model_id,
                text=text,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                input_cost=input_cost,
                output_cost=output_cost,
                total_cost=total_cost,
                latency_ms=processing_time * 1000,
                finish_reason=finish_reason,
                tokens_per_second=completion_tokens / max(processing_time, 0.001),
                metadata={
                    "model": model_id,
                    "api_version": self.config.api_version,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )

            # Update metrics
            self.request_count += 1
            self.total_tokens += total_tokens
            self.total_cost += total_cost

            # Emit completion event
            await self.event_bus.emit(
                LLMRequestCompleted(
                    provider_type=self.provider_type.value,
                    model_id=model_id,
                    request_id=request_id,
                    tokens_used=total_tokens,
                    cost=total_cost,
                    latency_ms=processing_time * 1000,
                    from_cache=False,
                )
            )

            return response

        except Exception as e:
            raise DeepSeekError(f"Failed to create response: {str(e)}")

    async def _generate_completion_impl(self, request: LLMRequest) -> LLMResponse:
        """Implementation of completion generation."""
        # Prepare messages
        messages = []

        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})

        if request.messages:
            messages.extend(request.messages)
        elif request.prompt:
            messages.append({"role": "user", "content": request.prompt})

        # Prepare API data
        api_data = {
            "model": request.model_id,
            "messages": messages,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "stream": request.streaming,
        }

        # Add function calling if supported
        if request.functions:
            api_data["functions"] = request.functions
            if request.function_call:
                api_data["function_call"] = request.function_call

        # Add tools if supported
        if request.tools:
            api_data["tools"] = request.tools
            if request.tool_choice:
                api_data["tool_choice"] = request.tool_choice

        if request.streaming:
            return await self._handle_streaming_completion(api_data, request)
        else:
            response_data = await self._make_api_request("chat/completions", api_data)
            return await self._create_response(
                response_data,
                request.request_id,
                time.time() - request.created_at.timestamp(),
                request.model_id,
            )

    async def _handle_streaming_completion(
        self, api_data: Dict[str, Any], request: LLMRequest
    ) -> LLMResponse:
        """Handle streaming completion response."""
        # For streaming, we'll collect all chunks and return a complete response
        # In a real implementation, you might want to yield partial responses

        full_content = ""
        function_call_data = None
        tool_calls = []

        async for chunk in self._make_api_request("chat/completions", api_data, stream=True):
            if "choices" in chunk and chunk["choices"]:
                choice = chunk["choices"][0]
                delta = choice.get("delta", {})

                if "content" in delta and delta["content"]:
                    full_content += delta["content"]

                # Handle function calls in streaming
                if "function_call" in delta:
                    if not function_call_data:
                        function_call_data = {"name": "", "arguments": ""}

                    if "name" in delta["function_call"]:
                        function_call_data["name"] += delta["function_call"]["name"]
                    if "arguments" in delta["function_call"]:
                        function_call_data["arguments"] += delta["function_call"]["arguments"]

                # Handle tool calls in streaming
                if "tool_calls" in delta:
                    # Simplified tool call handling
                    tool_calls.extend(delta["tool_calls"])

        # Create final response
        response_data = {
            "choices": [
                {"message": {"content": full_content, "role": "assistant"}, "finish_reason": "stop"}
            ],
            "usage": {
                "prompt_tokens": 0,  # Would need to calculate
                "completion_tokens": len(full_content.split()),  # Rough estimate
                "total_tokens": 0,
            },
        }

        if function_call_data:
            response_data["choices"][0]["message"]["function_call"] = function_call_data

        if tool_calls:
            response_data["choices"][0]["message"]["tool_calls"] = tool_calls

        return await self._create_response(
            response_data,
            request.request_id,
            time.time() - request.created_at.timestamp(),
            request.model_id,
        )

    async def get_available_models(self) -> List[ModelInfo]:
        """Get list of available models."""
        try:
            response_data = await self._make_api_request("models", {})
            models = []

            for model_data in response_data.get("data", []):
                model_id = model_data.get("id", "")

                # Check if we have info for this model
                if model_id in self.supported_models:
                    models.append(self.supported_models[model_id])
                else:
                    # Create basic model info
                    model_info = ModelInfo(
                        model_id=model_id,
                        name=model_data.get("name", model_id),
                        provider_type=self.provider_type,
                        description=f"DeepSeek model: {model_id}",
                    )
                    models.append(model_info)

            return models

        except Exception as e:
            self.logger.error(f"Failed to get available models: {str(e)}")
            return list(self.supported_models.values())

    async def _health_check_callback(self) -> Dict[str, Any]:
        """Health check callback for monitoring."""
        try:
            # Test basic connectivity
            start_time = time.time()
            await self._test_connectivity()
            response_time = (time.time() - start_time) * 1000

            return {
                "status": "healthy",
                "provider_type": self.provider_type.value,
                "api_response_time_ms": response_time,
                "total_requests": self.request_count,
                "total_tokens": self.total_tokens,
                "total_cost": self.total_cost,
                "supported_models": len(self.supported_models),
                "concurrent_limit": self.config.concurrent_requests,
                "rate_limit_rpm": self.config.rate_limit_rpm,
                "cache_enabled": self.config.enable_caching,
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
            if self.session:
                await self.session.close()

            if self.connector:
                await self.connector.close()

            self.status = ProviderStatus.STOPPED
            self.logger.info("DeepSeek provider cleaned up successfully")

        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")

    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            if hasattr(self, "session") and self.session and not self.session.closed:
                asyncio.create_task(self.session.close())
        except Exception:
            pass  # Ignore cleanup errors in destructor


# Factory function for easy integration
async def create_deepseek_provider(
    container: Container, config: Optional[Dict[str, Any]] = None
) -> DeepSeekProvider:
    """
    Factory function to create and initialize DeepSeek provider.

    Args:
        container: Dependency injection container
        config: Optional configuration dictionary

    Returns:
        Initialized DeepSeek provider
    """
    # Convert config dict to DeepSeekConfig if provided
    deepseek_config = DeepSeekConfig(**config) if config else None

    # Create provider
    provider = DeepSeekProvider(container, deepseek_config)

    # Initialize provider
    await provider.initialize()

    return provider
