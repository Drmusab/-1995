"""
Advanced Error Handling System for AI Assistant
Author: Drmusab
Last Modified: 2025-01-13 12:43:56 UTC

This module provides a comprehensive error handling framework that integrates
with all core system components, offering hierarchical error types, context-aware
recovery strategies, circuit breakers, retry mechanisms, and observability.
"""

import functools
import hashlib
import inspect
import json
import logging
import sys
import time
import traceback
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    AsyncGenerator,
    Callable,
    Coroutine,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    Type,
    TypeVar,
    Union,
)

import aiohttp
import asyncio
import asyncpg
import redis.exceptions
from pydantic import BaseModel, ValidationError

from src.core.dependency_injection import Container

# Core imports - delayed to avoid circular imports
# from src.core.config.loader import ConfigLoader
from src.core.events.event_bus import EnhancedEventBus
from src.core.events.event_types import (
    ComponentHealthChanged,
    ErrorOccurred,
    ErrorRecoveryCompleted,
    ErrorRecoveryStarted,
    HealthCheckCompleted,
    HealthCheckFailed,
    HealthCheckStarted,
    IntegrationError,
    ProcessingError,
    SystemStateChanged,
)

# Observability - delayed imports to avoid circular dependencies
# from src.observability.monitoring.metrics import MetricsCollector
# from src.observability.monitoring.tracing import TraceManager
from src.observability.logging.config import get_logger

# Type definitions
T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])


class ErrorSeverity(Enum):
    """Error severity levels."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    FATAL = "fatal"


class ErrorCategory(Enum):
    """Error categories for classification."""

    SYSTEM = "system"  # System-level errors
    COMPONENT = "component"  # Component failures
    WORKFLOW = "workflow"  # Workflow execution errors
    SESSION = "session"  # Session management errors
    PLUGIN = "plugin"  # Plugin-related errors
    MEMORY = "memory"  # Memory system errors
    PROCESSING = "processing"  # Data processing errors
    INTEGRATION = "integration"  # External integration errors
    SECURITY = "security"  # Security-related errors
    VALIDATION = "validation"  # Data validation errors
    NETWORK = "network"  # Network-related errors
    STORAGE = "storage"  # Storage-related errors
    AUTHENTICATION = "authentication"  # Authentication errors
    AUTHORIZATION = "authorization"  # Authorization errors
    CONFIGURATION = "configuration"  # Configuration errors
    BUSINESS_LOGIC = "business_logic"  # Business logic errors


class RecoveryStrategy(Enum):
    """Error recovery strategies."""

    NONE = "none"  # No automatic recovery
    RETRY = "retry"  # Retry the operation
    FALLBACK = "fallback"  # Use fallback mechanism
    CIRCUIT_BREAKER = "circuit_breaker"  # Open circuit breaker
    RESTART_COMPONENT = "restart_component"  # Restart failed component
    DEGRADE_SERVICE = "degrade_service"  # Graceful service degradation
    ESCALATE = "escalate"  # Escalate to human intervention
    SHUTDOWN = "shutdown"  # Shutdown system/component


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Circuit is open, calls fail fast
    HALF_OPEN = "half_open"  # Testing if service is back up


@dataclass
class ErrorContext:
    """Context information for errors."""

    error_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    correlation_id: Optional[str] = None
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    component: Optional[str] = None
    operation: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Request context
    request_id: Optional[str] = None
    trace_id: Optional[str] = None
    span_id: Optional[str] = None

    # System context
    hostname: Optional[str] = None
    process_id: Optional[int] = None
    thread_id: Optional[int] = None

    # Custom context
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: Set[str] = field(default_factory=set)


@dataclass
class RetryConfig:
    """Configuration for retry mechanisms."""

    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    backoff_strategy: str = "exponential"  # exponential, linear, fixed
    retry_on: Set[Type[Exception]] = field(default_factory=set)
    stop_on: Set[Type[Exception]] = field(default_factory=set)


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breakers."""

    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    expected_exception: Type[Exception] = Exception
    half_open_max_calls: int = 3
    success_threshold: int = 2  # For half-open to closed transition


class BaseAssistantError(Exception):
    """Base exception class for all assistant errors."""

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        category: ErrorCategory = ErrorCategory.SYSTEM,
        context: Optional[ErrorContext] = None,
        recoverable: bool = True,
        recovery_strategy: RecoveryStrategy = RecoveryStrategy.NONE,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.severity = severity
        self.category = category
        self.context = context or ErrorContext()
        self.recoverable = recoverable
        self.recovery_strategy = recovery_strategy
        self.metadata = metadata or {}
        self.timestamp = datetime.now(timezone.utc)

        # Automatically set error ID if not provided
        if not self.context.error_id:
            self.context.error_id = str(uuid.uuid4())

    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary."""
        return {
            "error_id": self.context.error_id,
            "error_code": self.error_code,
            "message": self.message,
            "severity": self.severity.value,
            "category": self.category.value,
            "recoverable": self.recoverable,
            "recovery_strategy": self.recovery_strategy.value,
            "timestamp": self.timestamp.isoformat(),
            "context": {
                "correlation_id": self.context.correlation_id,
                "session_id": self.context.session_id,
                "user_id": self.context.user_id,
                "component": self.context.component,
                "operation": self.context.operation,
                "metadata": self.context.metadata,
                "tags": list(self.context.tags),
            },
            "metadata": self.metadata,
        }


# Component-specific error classes
class ComponentError(BaseAssistantError):
    """Component-related errors."""

    def __init__(self, message: str, component_id: Optional[str] = None, **kwargs):
        super().__init__(message, category=ErrorCategory.COMPONENT, **kwargs)
        if component_id:
            self.context.component = component_id


class WorkflowError(BaseAssistantError):
    """Workflow execution errors."""

    def __init__(
        self,
        message: str,
        workflow_id: Optional[str] = None,
        execution_id: Optional[str] = None,
        step_id: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(message, category=ErrorCategory.WORKFLOW, **kwargs)
        self.workflow_id = workflow_id
        self.execution_id = execution_id
        self.step_id = step_id


class SessionError(BaseAssistantError):
    """Session management errors."""

    def __init__(self, message: str, session_id: Optional[str] = None, **kwargs):
        super().__init__(message, category=ErrorCategory.SESSION, **kwargs)
        if session_id:
            self.context.session_id = session_id


class PluginError(BaseAssistantError):
    """Plugin-related errors."""

    def __init__(self, message: str, plugin_id: Optional[str] = None, **kwargs):
        super().__init__(message, category=ErrorCategory.PLUGIN, **kwargs)
        self.plugin_id = plugin_id


class ProcessingError(BaseAssistantError):
    """Data processing errors."""

    def __init__(self, message: str, processor: Optional[str] = None, **kwargs):
        super().__init__(message, category=ErrorCategory.PROCESSING, **kwargs)
        self.processor = processor


class IntegrationError(BaseAssistantError):
    """External integration errors."""

    def __init__(
        self,
        message: str,
        integration: Optional[str] = None,
        external_error_code: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            message,
            category=ErrorCategory.INTEGRATION,
            recovery_strategy=RecoveryStrategy.CIRCUIT_BREAKER,
            **kwargs,
        )
        self.integration = integration
        self.external_error_code = external_error_code


class SecurityError(BaseAssistantError):
    """Security-related errors."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.SECURITY,
            severity=ErrorSeverity.CRITICAL,
            recoverable=False,
            **kwargs,
        )


class ValidationError(BaseAssistantError):
    """Data validation errors."""

    def __init__(self, message: str, field: Optional[str] = None, **kwargs):
        super().__init__(
            message, category=ErrorCategory.VALIDATION, severity=ErrorSeverity.WARNING, **kwargs
        )
        self.field = field


class NetworkError(BaseAssistantError):
    """Network-related errors."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.NETWORK,
            recovery_strategy=RecoveryStrategy.RETRY,
            **kwargs,
        )


class StorageError(BaseAssistantError):
    """Storage-related errors."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.STORAGE,
            recovery_strategy=RecoveryStrategy.RETRY,
            **kwargs,
        )


class MemoryError(BaseAssistantError):
    """Memory system errors."""

    def __init__(self, message: str, memory_type: Optional[str] = None, **kwargs):
        super().__init__(message, category=ErrorCategory.MEMORY, **kwargs)
        self.memory_type = memory_type


class ConfigurationError(BaseAssistantError):
    """Configuration errors."""

    def __init__(self, message: str, config_key: Optional[str] = None, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.CONFIGURATION,
            severity=ErrorSeverity.CRITICAL,
            recoverable=False,
            **kwargs,
        )
        self.config_key = config_key


class AuthenticationError(BaseAssistantError):
    """Authentication errors."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message, category=ErrorCategory.AUTHENTICATION, severity=ErrorSeverity.WARNING, **kwargs
        )


class AuthorizationError(BaseAssistantError):
    """Authorization errors."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message, category=ErrorCategory.AUTHORIZATION, severity=ErrorSeverity.WARNING, **kwargs
        )


class BusinessLogicError(BaseAssistantError):
    """Business logic errors."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, category=ErrorCategory.BUSINESS_LOGIC, **kwargs)


class CircuitBreaker:
    """Circuit breaker implementation for fault tolerance."""

    def __init__(
        self,
        name: str,
        config: CircuitBreakerConfig,
        event_bus: Optional[Any] = None,  # EnhancedEventBus
        metrics: Optional[Any] = None,  # MetricsCollector
    ):
        self.name = name
        self.config = config
        self.event_bus = event_bus
        self.metrics = metrics

        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.half_open_calls = 0

        self.logger = get_logger(f"circuit_breaker_{name}")

    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                self.half_open_calls = 0
                self.logger.info(f"Circuit breaker {self.name} transitioning to half-open")
            else:
                raise IntegrationError(
                    f"Circuit breaker {self.name} is open", error_code="CIRCUIT_BREAKER_OPEN"
                )

        try:
            if self.state == CircuitState.HALF_OPEN:
                if self.half_open_calls >= self.config.half_open_max_calls:
                    raise IntegrationError(
                        f"Circuit breaker {self.name} half-open call limit exceeded",
                        error_code="HALF_OPEN_LIMIT_EXCEEDED",
                    )
                self.half_open_calls += 1

            # Execute the function
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)

            # Success handling
            await self._on_success()
            return result

        except Exception as e:
            await self._on_failure(e)
            raise

    def _should_attempt_reset(self) -> bool:
        """Check if we should attempt to reset the circuit breaker."""
        if not self.last_failure_time:
            return True

        time_since_failure = datetime.now(timezone.utc) - self.last_failure_time
        return time_since_failure.total_seconds() >= self.config.recovery_timeout

    async def _on_success(self) -> None:
        """Handle successful call."""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                await self._close_circuit()
        elif self.state == CircuitState.CLOSED:
            self.failure_count = 0

    async def _on_failure(self, error: Exception) -> None:
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = datetime.now(timezone.utc)

        if self.state == CircuitState.HALF_OPEN:
            await self._open_circuit()
        elif (
            self.state == CircuitState.CLOSED
            and self.failure_count >= self.config.failure_threshold
        ):
            await self._open_circuit()

        if self.metrics:
            self.metrics.increment(
                "circuit_breaker_failures_total", tags={"circuit_breaker": self.name}
            )

    async def _open_circuit(self) -> None:
        """Open the circuit breaker."""
        self.state = CircuitState.OPEN
        self.success_count = 0
        self.half_open_calls = 0

        if self.event_bus:
            await self.event_bus.emit(
                CircuitBreakerOpened(
                    circuit_breaker_name=self.name, failure_count=self.failure_count
                )
            )

        if self.metrics:
            self.metrics.set(
                "circuit_breaker_state", 1, tags={"circuit_breaker": self.name}  # 1 for open
            )

        self.logger.warning(
            f"Circuit breaker {self.name} opened after {self.failure_count} failures"
        )

    async def _close_circuit(self) -> None:
        """Close the circuit breaker."""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.half_open_calls = 0

        if self.event_bus:
            await self.event_bus.emit(CircuitBreakerClosed(circuit_breaker_name=self.name))

        if self.metrics:
            self.metrics.set(
                "circuit_breaker_state", 0, tags={"circuit_breaker": self.name}  # 0 for closed
            )

        self.logger.info(f"Circuit breaker {self.name} closed")


class RetryHandler:
    """Handles retry logic with various backoff strategies."""

    def __init__(
        self,
        config: RetryConfig,
        metrics: Optional[Any] = None,  # MetricsCollector
        event_bus: Optional[Any] = None,  # EnhancedEventBus
    ):
        self.config = config
        self.metrics = metrics
        self.event_bus = event_bus
        self.logger = get_logger("retry_handler")

    async def execute_with_retry(
        self, func: Callable, *args, context: Optional[ErrorContext] = None, **kwargs
    ) -> Any:
        """Execute function with retry logic."""
        last_exception = None

        for attempt in range(self.config.max_attempts):
            try:
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)

                if attempt > 0 and self.event_bus:
                    await self.event_bus.emit(
                        ErrorRecovered(
                            error_id=context.error_id if context else str(uuid.uuid4()),
                            component=context.component if context else "unknown",
                            recovery_method="retry",
                            attempts=attempt + 1,
                        )
                    )

                return result

            except Exception as e:
                last_exception = e

                # Check if we should stop retrying
                if type(e) in self.config.stop_on:
                    break

                # Check if we should retry this exception
                if self.config.retry_on and type(e) not in self.config.retry_on:
                    break

                # Don't retry on the last attempt
                if attempt == self.config.max_attempts - 1:
                    break

                # Calculate delay
                delay = self._calculate_delay(attempt)

                # Emit retry event
                if self.event_bus:
                    await self.event_bus.emit(
                        RetryAttempted(
                            error_id=context.error_id if context else str(uuid.uuid4()),
                            component=context.component if context else "unknown",
                            attempt=attempt + 1,
                            max_attempts=self.config.max_attempts,
                            delay=delay,
                            error_type=type(e).__name__,
                        )
                    )

                if self.metrics:
                    self.metrics.increment(
                        "retry_attempts_total",
                        tags={
                            "component": context.component if context else "unknown",
                            "error_type": type(e).__name__,
                        },
                    )

                self.logger.warning(
                    f"Attempt {attempt + 1}/{self.config.max_attempts} failed: {str(e)}. "
                    f"Retrying in {delay:.2f} seconds."
                )

                await asyncio.sleep(delay)

        # All retries exhausted
        if self.metrics:
            self.metrics.increment(
                "retry_failures_total",
                tags={
                    "component": context.component if context else "unknown",
                    "error_type": type(last_exception).__name__ if last_exception else "unknown",
                },
            )

        raise last_exception

    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for retry attempt."""
        if self.config.backoff_strategy == "exponential":
            delay = self.config.base_delay * (self.config.exponential_base**attempt)
        elif self.config.backoff_strategy == "linear":
            delay = self.config.base_delay * (attempt + 1)
        else:  # fixed
            delay = self.config.base_delay

        # Apply maximum delay
        delay = min(delay, self.config.max_delay)

        # Add jitter if enabled
        if self.config.jitter:
            import random

            delay = delay * (0.5 + random.random() * 0.5)

        return delay


class ErrorAggregator:
    """Aggregates and analyzes error patterns."""

    def __init__(
        self,
        window_size: int = 100,
        time_window_seconds: float = 300.0,
        event_bus: Optional[Any] = None,  # EnhancedEventBus
        metrics: Optional[Any] = None,  # MetricsCollector
    ):
        self.window_size = window_size
        self.time_window_seconds = time_window_seconds
        self.event_bus = event_bus
        self.metrics = metrics

        self.error_history: deque = deque(maxlen=window_size)
        self.error_counts: Dict[str, int] = defaultdict(int)
        self.error_patterns: Dict[str, List[datetime]] = defaultdict(list)

        self.logger = get_logger("error_aggregator")

    async def record_error(self, error: BaseAssistantError) -> None:
        """Record an error for analysis."""
        current_time = datetime.now(timezone.utc)

        error_record = {
            "error": error,
            "timestamp": current_time,
            "error_hash": self._hash_error(error),
        }

        self.error_history.append(error_record)

        # Update counts
        error_key = f"{error.category.value}:{error.error_code}"
        self.error_counts[error_key] += 1
        self.error_patterns[error_key].append(current_time)

        # Clean old patterns
        self._clean_old_patterns(current_time)

        # Analyze patterns
        await self._analyze_patterns(error_key, current_time)

        # Update metrics
        if self.metrics:
            self.metrics.increment(
                "errors_total",
                tags={
                    "category": error.category.value,
                    "severity": error.severity.value,
                    "error_code": error.error_code,
                    "component": error.context.component or "unknown",
                },
            )

    def _hash_error(self, error: BaseAssistantError) -> str:
        """Generate hash for error deduplication."""
        hash_data = f"{error.category.value}:{error.error_code}:{error.message}"
        return hashlib.md5(hash_data.encode()).hexdigest()[:8]

    def _clean_old_patterns(self, current_time: datetime) -> None:
        """Remove old error patterns outside the time window."""
        cutoff_time = current_time - timedelta(seconds=self.time_window_seconds)

        for error_key in list(self.error_patterns.keys()):
            self.error_patterns[error_key] = [
                ts for ts in self.error_patterns[error_key] if ts > cutoff_time
            ]

            if not self.error_patterns[error_key]:
                del self.error_patterns[error_key]

    async def _analyze_patterns(self, error_key: str, current_time: datetime) -> None:
        """Analyze error patterns for anomalies."""
        if error_key not in self.error_patterns:
            return

        recent_errors = self.error_patterns[error_key]

        # Check for error spikes
        if len(recent_errors) >= 5:  # Threshold for pattern detection
            # Calculate error rate
            time_span = (recent_errors[-1] - recent_errors[0]).total_seconds()
            if time_span > 0:
                error_rate = len(recent_errors) / time_span

                # Emit pattern detection event if rate is high
                if error_rate > 0.1:  # More than 0.1 errors per second
                    if self.event_bus:
                        await self.event_bus.emit(
                            ErrorPatternDetected(
                                pattern_type="high_frequency",
                                error_key=error_key,
                                error_count=len(recent_errors),
                                time_window=self.time_window_seconds,
                                error_rate=error_rate,
                            )
                        )

    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of error patterns."""
        return {
            "total_errors_recorded": len(self.error_history),
            "unique_error_types": len(self.error_counts),
            "top_errors": dict(
                sorted(self.error_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            ),
            "recent_error_rate": self._calculate_recent_error_rate(),
        }

    def _calculate_recent_error_rate(self) -> float:
        """Calculate recent error rate."""
        if not self.error_history:
            return 0.0

        current_time = datetime.now(timezone.utc)
        cutoff_time = current_time - timedelta(seconds=60)  # Last minute

        recent_errors = [
            record for record in self.error_history if record["timestamp"] > cutoff_time
        ]

        return len(recent_errors) / 60.0  # Errors per second


class ErrorHandler:
    """
    Comprehensive error handling system for the AI Assistant.

    Features:
    - Hierarchical error classification
    - Context-aware error handling
    - Circuit breaker pattern for external dependencies
    - Intelligent retry mechanisms with exponential backoff
    - Error aggregation and pattern detection
    - Integration with monitoring and alerting
    - Security-aware error handling
    - Performance-optimized error processing
    """

    def __init__(self, container: Container):
        """Initialize the error handler."""
        self.container = container
        self.logger = get_logger(__name__)

        # Core services - use dynamic imports to avoid circular dependencies
        try:
            # Import ConfigLoader dynamically
            from src.core.config.loader import ConfigLoader

            self.config = container.get(ConfigLoader)
        except Exception:
            self.config = None

        self.event_bus = container.get(EnhancedEventBus)
        try:
            # Import metrics and tracing dynamically
            from src.observability.monitoring.metrics import MetricsCollector
            from src.observability.monitoring.tracing import TraceManager

            self.metrics = container.get(MetricsCollector)
            self.tracer = container.get(TraceManager)
        except Exception:
            self.metrics = None
            self.tracer = None

        # Error handling components
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.retry_handlers: Dict[str, RetryHandler] = {}
        self.error_aggregator = ErrorAggregator(event_bus=self.event_bus, metrics=self.metrics)

        # Configuration
        self._load_configurations()

        # Error context stack for nested operations
        self._context_stack: List[ErrorContext] = []

        # Security settings
        self.sanitize_errors = self.config.get("error_handling.sanitize_errors", True)
        self.log_sensitive_data = self.config.get("error_handling.log_sensitive_data", False)

        # Performance settings
        self.enable_error_aggregation = self.config.get("error_handling.enable_aggregation", True)
        self.max_stack_trace_depth = self.config.get("error_handling.max_stack_trace_depth", 10)

        self.logger.info("ErrorHandler initialized successfully")

    def _load_configurations(self) -> None:
        """Load error handling configurations."""
        # Default circuit breaker configs
        circuit_configs = self.config.get("error_handling.circuit_breakers", {})
        for name, config_dict in circuit_configs.items():
            config = CircuitBreakerConfig(**config_dict)
            self.circuit_breakers[name] = CircuitBreaker(
                name=name, config=config, event_bus=self.event_bus, metrics=self.metrics
            )

        # Default retry configs
        retry_configs = self.config.get("error_handling.retry_handlers", {})
        for name, config_dict in retry_configs.items():
            config = RetryConfig(**config_dict)
            self.retry_handlers[name] = RetryHandler(
                config=config, metrics=self.metrics, event_bus=self.event_bus
            )

    @contextmanager
    def error_context(
        self,
        component: Optional[str] = None,
        operation: Optional[str] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        **metadata,
    ):
        """Context manager for error handling context."""
        context = ErrorContext(
            component=component,
            operation=operation,
            session_id=session_id,
            user_id=user_id,
            metadata=metadata,
        )

        self._context_stack.append(context)
        try:
            yield context
        finally:
            self._context_stack.pop()

    def get_current_context(self) -> Optional[ErrorContext]:
        """Get the current error context."""
        return self._context_stack[-1] if self._context_stack else None

    async def handle_error(
        self, error: Exception, context: Optional[ErrorContext] = None, recover: bool = True
    ) -> Optional[Any]:
        """
        Handle an error with full error processing pipeline.

        Args:
            error: The exception to handle
            context: Error context (uses current context if None)
            recover: Whether to attempt recovery

        Returns:
            Recovery result if successful, None if no recovery attempted
        """
        # Use current context if none provided
        if context is None:
            context = self.get_current_context() or ErrorContext()

        # Convert to assistant error if needed
        if not isinstance(error, BaseAssistantError):
            assistant_error = self._convert_to_assistant_error(error, context)
        else:
            assistant_error = error
            if not assistant_error.context.component and context.component:
                assistant_error.context = context

        # Add tracing information
        if self.tracer:
            span = self.tracer.get_current_span()
            if span:
                assistant_error.context.trace_id = span.trace_id
                assistant_error.context.span_id = span.span_id

        # Record error for aggregation
        if self.enable_error_aggregation:
            await self.error_aggregator.record_error(assistant_error)

        # Emit error event
        await self.event_bus.emit(
            ErrorOccurred(
                error_id=assistant_error.context.error_id,
                error_code=assistant_error.error_code,
                error_message=self._sanitize_error_message(assistant_error.message),
                severity=assistant_error.severity.value,
                category=assistant_error.category.value,
                component=assistant_error.context.component or "unknown",
                session_id=assistant_error.context.session_id,
                user_id=assistant_error.context.user_id,
                recoverable=assistant_error.recoverable,
                recovery_strategy=assistant_error.recovery_strategy.value,
            )
        )

        # Log error
        self._log_error(assistant_error)

        # Attempt recovery if enabled and error is recoverable
        if recover and assistant_error.recoverable:
            return await self._attempt_recovery(assistant_error)

        return None

    def _convert_to_assistant_error(
        self, error: Exception, context: ErrorContext
    ) -> BaseAssistantError:
        """Convert standard exception to assistant error."""
        error_mappings = {
            ValueError: ValidationError,
            TypeError: ValidationError,
            KeyError: ConfigurationError,
            FileNotFoundError: StorageError,
            PermissionError: AuthorizationError,
            ConnectionError: NetworkError,
            TimeoutError: NetworkError,
            aiohttp.ClientError: NetworkError,
            asyncpg.PostgresError: StorageError,
            redis.exceptions.RedisError: StorageError,
        }

        # Find appropriate error class
        error_class = BaseAssistantError
        for exception_type, assistant_error_type in error_mappings.items():
            if isinstance(error, exception_type):
                error_class = assistant_error_type
                break

        # Create assistant error
        return error_class(
            message=str(error),
            context=context,
            metadata={
                "original_exception": type(error).__name__,
                "traceback": traceback.format_exc() if self.log_sensitive_data else None,
            },
        )

    async def _attempt_recovery(self, error: BaseAssistantError) -> Optional[Any]:
        """Attempt to recover from an error."""
        try:
            if error.recovery_strategy == RecoveryStrategy.RETRY:
                return await self._retry_recovery(error)
            elif error.recovery_strategy == RecoveryStrategy.FALLBACK:
                return await self._fallback_recovery(error)
            elif error.recovery_strategy == RecoveryStrategy.CIRCUIT_BREAKER:
                return await self._circuit_breaker_recovery(error)
            elif error.recovery_strategy == RecoveryStrategy.RESTART_COMPONENT:
                return await self._restart_component_recovery(error)
            elif error.recovery_strategy == RecoveryStrategy.DEGRADE_SERVICE:
                return await self._degrade_service_recovery(error)
            elif error.recovery_strategy == RecoveryStrategy.ESCALATE:
                return await self._escalate_recovery(error)

        except Exception as recovery_error:
            self.logger.error(
                f"Recovery failed for error {error.context.error_id}: {recovery_error}"
            )

            # Emit recovery failure
            await self.event_bus.emit(
                ErrorOccurred(
                    error_id=str(uuid.uuid4()),
                    error_code="RECOVERY_FAILED",
                    error_message=f"Recovery failed: {str(recovery_error)}",
                    severity="critical",
                    category="system",
                    component=error.context.component or "unknown",
                    session_id=error.context.session_id,
                    user_id=error.context.user_id,
                    recoverable=False,
                )
            )

        return None

    async def _retry_recovery(self, error: BaseAssistantError) -> Optional[Any]:
        """Retry-based recovery."""
        # This would need the original function to retry
        # For now, just log the attempt
        self.logger.info(f"Retry recovery attempted for error {error.context.error_id}")
        return None

    async def _fallback_recovery(self, error: BaseAssistantError) -> Optional[Any]:
        """Fallback-based recovery."""
        self.logger.info(f"Fallback recovery attempted for error {error.context.error_id}")
        # Implement fallback logic based on error type
        return None

    async def _circuit_breaker_recovery(self, error: BaseAssistantError) -> Optional[Any]:
        """Circuit breaker recovery."""
        component = error.context.component
        if component and component in self.circuit_breakers:
            circuit_breaker = self.circuit_breakers[component]
            await circuit_breaker._open_circuit()
        return None

    async def _restart_component_recovery(self, error: BaseAssistantError) -> Optional[Any]:
        """Component restart recovery."""
        component = error.context.component
        if component:
            # Emit component restart request
            await self.event_bus.emit(
                ComponentHealthChanged(
                    component=component,
                    healthy=False,
                    details={"restart_requested": True, "error_id": error.context.error_id},
                )
            )
        return None

    async def _degrade_service_recovery(self, error: BaseAssistantError) -> Optional[Any]:
        """Service degradation recovery."""
        self.logger.warning(f"Service degradation initiated for error {error.context.error_id}")
        # Implement service degradation logic
        return None

    async def _escalate_recovery(self, error: BaseAssistantError) -> Optional[Any]:
        """Escalate error for human intervention."""
        await self.event_bus.emit(
            CriticalSystemError(
                error_id=error.context.error_id,
                error_message=error.message,
                component=error.context.component or "unknown",
                requires_intervention=True,
            )
        )
        return None

    def _sanitize_error_message(self, message: str) -> str:
        """Sanitize error message to remove sensitive information."""
        if not self.sanitize_errors:
            return message

        # Remove potential sensitive patterns
        import re

        # Remove API keys, tokens, passwords
        patterns = [
            r'(?i)(api[_-]?key|token|password|secret)["\s]*[:=]["\s]*[^\s"]+',
            r"(?i)(bearer\s+)[a-zA-Z0-9._-]+",
            r"(?i)(basic\s+)[a-zA-Z0-9._-]+",
            r"\b\d{16}\b",  # Credit card-like numbers
            r"\b\d{3}-\d{2}-\d{4}\b",  # SSN-like patterns
        ]

        sanitized = message
        for pattern in patterns:
            sanitized = re.sub(pattern, "[REDACTED]", sanitized)

        return sanitized

    def _log_error(self, error: BaseAssistantError) -> None:
        """Log error with appropriate level and format."""
        log_data = {
            "error_id": error.context.error_id,
            "error_code": error.error_code,
            "category": error.category.value,
            "component": error.context.component,
            "operation": error.context.operation,
            "session_id": error.context.session_id,
            "user_id": error.context.user_id,
            "recoverable": error.recoverable,
            "recovery_strategy": error.recovery_strategy.value,
        }

        if self.log_sensitive_data:
            log_data["metadata"] = error.metadata
            log_data["context_metadata"] = error.context.metadata

        log_message = f"Error {error.context.error_id}: {error.message}"

        if error.severity == ErrorSeverity.DEBUG:
            self.logger.debug(log_message, extra=log_data)
        elif error.severity == ErrorSeverity.INFO:
            self.logger.info(log_message, extra=log_data)
        elif error.severity == ErrorSeverity.WARNING:
            self.logger.warning(log_message, extra=log_data)
        elif error.severity == ErrorSeverity.ERROR:
            self.logger.error(log_message, extra=log_data)
        elif error.severity in [ErrorSeverity.CRITICAL, ErrorSeverity.FATAL]:
            self.logger.critical(log_message, extra=log_data)

    def get_circuit_breaker(self, name: str) -> Optional[CircuitBreaker]:
        """Get circuit breaker by name."""
        return self.circuit_breakers.get(name)

    def get_retry_handler(self, name: str) -> Optional[RetryHandler]:
        """Get retry handler by name."""
        return self.retry_handlers.get(name)

    def add_circuit_breaker(self, name: str, config: CircuitBreakerConfig) -> CircuitBreaker:
        """Add a new circuit breaker."""
        circuit_breaker = CircuitBreaker(
            name=name, config=config, event_bus=self.event_bus, metrics=self.metrics
        )
        self.circuit_breakers[name] = circuit_breaker
        return circuit_breaker

    def add_retry_handler(self, name: str, config: RetryConfig) -> RetryHandler:
        """Add a new retry handler."""
        retry_handler = RetryHandler(config=config, metrics=self.metrics, event_bus=self.event_bus)
        self.retry_handlers[name] = retry_handler
        return retry_handler

    async def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health based on errors."""
        error_summary = self.error_aggregator.get_error_summary()

        # Calculate health score
        recent_error_rate = error_summary.get("recent_error_rate", 0)
        if recent_error_rate == 0:
            health_score = 1.0
        elif recent_error_rate < 0.01:  # Less than 1 error per 100 seconds
            health_score = 0.9
        elif recent_error_rate < 0.1:  # Less than 1 error per 10 seconds
            health_score = 0.7
        else:
            health_score = 0.3

        # Circuit breaker status
        circuit_status = {name: cb.state.value for name, cb in self.circuit_breakers.items()}

        return {
            "overall_health_score": health_score,
            "error_summary": error_summary,
            "circuit_breakers": circuit_status,
            "active_error_contexts": len(self._context_stack),
            "system_status": (
                "healthy"
                if health_score > 0.8
                else "degraded" if health_score > 0.5 else "unhealthy"
            ),
        }


# Decorator functions for easy error handling
def handle_exceptions(
    recovery_strategy: RecoveryStrategy = RecoveryStrategy.NONE,
    retry_config: Optional[RetryConfig] = None,
    circuit_breaker: Optional[str] = None,
    reraise: bool = True,
    default_return: Any = None,
):
    """
    Decorator for automatic error handling.

    Args:
        recovery_strategy: Strategy to use for error recovery
        retry_config: Retry configuration for retry strategy
        circuit_breaker: Name of circuit breaker to use
        reraise: Whether to reraise the exception after handling
        default_return: Default value to return on error
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Get error handler from container if available
            error_handler = None
            if hasattr(args[0], "container"):
                try:
                    error_handler = args[0].container.get(ErrorHandler)
                except Exception:
                    pass

            # Set up error context
            component = getattr(args[0], "__class__", {}).get("__name__", "unknown")
            operation = func.__name__

            context = ErrorContext(component=component, operation=operation)

            try:
                # Apply circuit breaker if specified
                if circuit_breaker and error_handler:
                    cb = error_handler.get_circuit_breaker(circuit_breaker)
                    if cb:
                        return await cb.call(func, *args, **kwargs)

                # Apply retry if specified
                if retry_config and error_handler:
                    retry_handler = RetryHandler(retry_config)
                    return await retry_handler.execute_with_retry(
                        func, *args, context=context, **kwargs
                    )

                # Normal execution
                return await func(*args, **kwargs)

            except Exception as e:
                # Handle error if handler available
                if error_handler:
                    recovery_result = await error_handler.handle_error(
                        e, context, recover=(recovery_strategy != RecoveryStrategy.NONE)
                    )
                    if recovery_result is not None:
                        return recovery_result

                if reraise:
                    raise
                else:
                    return default_return

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            # For synchronous functions, create async wrapper
            if asyncio.iscoroutinefunction(func):
                return async_wrapper(*args, **kwargs)

            # Handle synchronous functions
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if reraise:
                    raise
                else:
                    return default_return

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator


def with_circuit_breaker(circuit_breaker_name: str):
    """Decorator for circuit breaker protection."""

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Get error handler from container
            error_handler = None
            if hasattr(args[0], "container"):
                try:
                    error_handler = args[0].container.get(ErrorHandler)
                except Exception:
                    pass

            if error_handler:
                circuit_breaker = error_handler.get_circuit_breaker(circuit_breaker_name)
                if circuit_breaker:
                    return await circuit_breaker.call(func, *args, **kwargs)

            # Fallback to normal execution
            return await func(*args, **kwargs)

        return wrapper

    return decorator


def with_retry(retry_config: RetryConfig):
    """Decorator for retry functionality."""

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            retry_handler = RetryHandler(retry_config)
            return await retry_handler.execute_with_retry(func, *args, **kwargs)

        return wrapper

    return decorator


# Context managers for error handling
@asynccontextmanager
async def error_boundary(
    error_handler: ErrorHandler,
    component: str,
    operation: str,
    recovery_strategy: RecoveryStrategy = RecoveryStrategy.NONE,
    **context_kwargs,
):
    """Async context manager for error boundary."""
    with error_handler.error_context(
        component=component, operation=operation, **context_kwargs
    ) as context:
        try:
            yield context
        except Exception as e:
            await error_handler.handle_error(
                e, context, recover=(recovery_strategy != RecoveryStrategy.NONE)
            )
            raise


# Utility functions
def create_error_context(
    component: Optional[str] = None,
    operation: Optional[str] = None,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    **metadata,
) -> ErrorContext:
    """Create an error context."""
    return ErrorContext(
        component=component,
        operation=operation,
        session_id=session_id,
        user_id=user_id,
        metadata=metadata,
    )


def is_retryable_error(error: Exception) -> bool:
    """Check if an error is retryable."""
    retryable_types = (NetworkError, StorageError, IntegrationError, ConnectionError, TimeoutError)
    return isinstance(error, retryable_types)


def get_error_severity(error: Exception) -> ErrorSeverity:
    """Get appropriate severity for an error."""
    if isinstance(error, BaseAssistantError):
        return error.severity

    severity_mapping = {
        ValueError: ErrorSeverity.WARNING,
        TypeError: ErrorSeverity.WARNING,
        KeyError: ErrorSeverity.ERROR,
        FileNotFoundError: ErrorSeverity.ERROR,
        PermissionError: ErrorSeverity.ERROR,
        ConnectionError: ErrorSeverity.ERROR,
        TimeoutError: ErrorSeverity.WARNING,
        SystemError: ErrorSeverity.CRITICAL,
        MemoryError: ErrorSeverity.CRITICAL,
    }

    return severity_mapping.get(type(error), ErrorSeverity.ERROR)


def sanitize_error_for_user(error: Exception) -> str:
    """Sanitize error message for user display."""
    if isinstance(error, SecurityError):
        return "A security error occurred. Please contact support."
    elif isinstance(error, AuthenticationError):
        return "Authentication failed. Please log in again."
    elif isinstance(error, AuthorizationError):
        return "You don't have permission to perform this action."
    elif isinstance(error, ValidationError):
        return f"Invalid input: {error.message}"
    elif isinstance(error, NetworkError):
        return "Network error occurred. Please try again."
    elif isinstance(error, BaseAssistantError):
        return error.message
    else:
        return "An unexpected error occurred. Please try again."


# Export main classes and functions
__all__ = [
    # Error classes
    "BaseAssistantError",
    "ComponentError",
    "WorkflowError",
    "SessionError",
    "PluginError",
    "ProcessingError",
    "IntegrationError",
    "SecurityError",
    "ValidationError",
    "NetworkError",
    "StorageError",
    "MemoryError",
    "ConfigurationError",
    "AuthenticationError",
    "AuthorizationError",
    "BusinessLogicError",
    # Enums
    "ErrorSeverity",
    "ErrorCategory",
    "RecoveryStrategy",
    "CircuitState",
    # Data classes
    "ErrorContext",
    "RetryConfig",
    "CircuitBreakerConfig",
    # Main classes
    "ErrorHandler",
    "CircuitBreaker",
    "RetryHandler",
    "ErrorAggregator",
    # Decorators
    "handle_exceptions",
    "with_circuit_breaker",
    "with_retry",
    # Context managers
    "error_boundary",
    # Utility functions
    "create_error_context",
    "is_retryable_error",
    "get_error_severity",
    "sanitize_error_for_user",
]
