"""
Advanced Distributed Tracing System for AI Assistant
Author: Drmusab
Last Modified: 2025-05-26 16:45:12 UTC

This module provides comprehensive distributed tracing capabilities for the AI assistant,
including request correlation, performance monitoring, error tracking, multimodal
processing traces, and integration with external tracing systems.
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
from contextlib import asynccontextmanager, contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    AsyncGenerator,
    Callable,
    ContextManager,
    Dict,
    List,
    Optional,
    Set,
    Type,
    Union,
)

import asyncio

# Third-party tracing libraries
try:
    import opentelemetry
    from opentelemetry import baggage, propagate, trace
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
    from opentelemetry.exporter.zipkin.json import ZipkinExporter
    from opentelemetry.instrumentation.auto_instrumentation import sitecustomize
    from opentelemetry.propagators.b3 import B3MultiFormat
    from opentelemetry.propagators.composite import CompositePropagator
    from opentelemetry.propagators.jaeger import JaegerPropagator
    from opentelemetry.sdk.trace import Span, TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    from opentelemetry.trace import SpanKind, Status, StatusCode

    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False

# Core imports
from src.core.config.loader import ConfigLoader
from src.core.dependency_injection import Container
from src.core.error_handling import ErrorHandler, handle_exceptions
from src.core.events.event_bus import EventBus
from src.core.events.event_types import (
    ComponentPerformanceAlert,
    PerformanceThresholdExceeded,
    SpanCompleted,
    SpanError,
    SpanStarted,
    TraceCompleted,
    TraceCorrelationEstablished,
    TraceFailed,
    TraceStarted,
    TracingConfigurationChanged,
)
from src.core.health_check import HealthCheck
from src.observability.logging.config import get_logger

# Observability
from src.observability.monitoring.metrics import MetricsCollector

# Type definitions
TraceId = str
SpanId = str
CorrelationId = str


class SpanType(Enum):
    """Types of spans for different operations."""

    HTTP_REQUEST = "http_request"
    WEBSOCKET = "websocket"
    GRPC = "grpc"
    DATABASE = "database"
    CACHE = "cache"
    EXTERNAL_API = "external_api"
    SPEECH_PROCESSING = "speech_processing"
    VISION_PROCESSING = "vision_processing"
    NLP_PROCESSING = "nlp_processing"
    MULTIMODAL_FUSION = "multimodal_fusion"
    MEMORY_OPERATION = "memory_operation"
    SKILL_EXECUTION = "skill_execution"
    WORKFLOW_EXECUTION = "workflow_execution"
    REASONING = "reasoning"
    LEARNING = "learning"
    COMPONENT_INITIALIZATION = "component_initialization"
    BACKGROUND_TASK = "background_task"
    USER_INTERACTION = "user_interaction"
    SESSION_MANAGEMENT = "session_management"
    PLUGIN_OPERATION = "plugin_operation"


class TracingBackend(Enum):
    """Supported tracing backends."""

    JAEGER = "jaeger"
    ZIPKIN = "zipkin"
    DATADOG = "datadog"
    NEWRELIC = "newrelic"
    CONSOLE = "console"
    CUSTOM = "custom"


class SamplingStrategy(Enum):
    """Sampling strategies for trace collection."""

    ALWAYS = "always"
    NEVER = "never"
    PROBABILISTIC = "probabilistic"
    RATE_LIMITED = "rate_limited"
    ADAPTIVE = "adaptive"
    ERROR_BASED = "error_based"


@dataclass
class TraceContext:
    """Context information for distributed tracing."""

    trace_id: TraceId
    span_id: SpanId
    parent_span_id: Optional[SpanId] = None
    correlation_id: Optional[CorrelationId] = None

    # Context metadata
    service_name: str = "ai_assistant"
    operation_name: str = "unknown"
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None

    # Timing
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Baggage (cross-cutting concerns)
    baggage: Dict[str, str] = field(default_factory=dict)

    # Sampling
    is_sampled: bool = True
    sampling_priority: float = 1.0

    # Error tracking
    has_error: bool = False
    error_type: Optional[str] = None
    error_message: Optional[str] = None


@dataclass
class SpanData:
    """Data structure for span information."""

    span_id: SpanId
    trace_id: TraceId
    parent_span_id: Optional[SpanId] = None

    # Span metadata
    operation_name: str = "unknown"
    span_type: SpanType = SpanType.COMPONENT_INITIALIZATION
    service_name: str = "ai_assistant"

    # Timing
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    end_time: Optional[datetime] = None
    duration: Optional[float] = None

    # Status
    status_code: str = "OK"
    status_message: Optional[str] = None
    is_error: bool = False

    # Attributes
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)
    links: List[Dict[str, Any]] = field(default_factory=list)

    # Resource information
    resource_attributes: Dict[str, str] = field(default_factory=dict)

    # Custom metadata
    tags: Dict[str, str] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class TracingConfiguration:
    """Configuration for the tracing system."""

    # Backend configuration
    backend: TracingBackend = TracingBackend.JAEGER
    endpoint: Optional[str] = None
    service_name: str = "ai_assistant"
    service_version: str = "1.0.0"

    # Sampling configuration
    sampling_strategy: SamplingStrategy = SamplingStrategy.PROBABILISTIC
    sampling_rate: float = 0.1
    max_traces_per_second: int = 100

    # Export configuration
    export_timeout: float = 10.0
    max_export_batch_size: int = 512
    export_interval: float = 5.0

    # Resource attributes
    resource_attributes: Dict[str, str] = field(default_factory=dict)

    # Feature flags
    enable_automatic_instrumentation: bool = True
    enable_async_export: bool = True
    enable_compression: bool = True
    enable_correlation: bool = True

    # Performance settings
    max_span_attributes: int = 128
    max_span_events: int = 128
    max_span_links: int = 128
    max_attribute_length: int = 1024

    # Component-specific settings
    component_sampling: Dict[str, float] = field(default_factory=dict)
    excluded_operations: Set[str] = field(default_factory=set)
    sensitive_attributes: Set[str] = field(default_factory=set)


class TracingError(Exception):
    """Custom exception for tracing operations."""

    def __init__(
        self, message: str, trace_id: Optional[TraceId] = None, span_id: Optional[SpanId] = None
    ):
        super().__init__(message)
        self.trace_id = trace_id
        self.span_id = span_id
        self.timestamp = datetime.now(timezone.utc)


class SpanProcessor(ABC):
    """Abstract base class for span processors."""

    @abstractmethod
    async def on_start(self, span: SpanData, parent_context: Optional[TraceContext]) -> None:
        """Called when a span starts."""
        pass

    @abstractmethod
    async def on_end(self, span: SpanData) -> None:
        """Called when a span ends."""
        pass

    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown the processor."""
        pass


class SpanExporter(ABC):
    """Abstract base class for span exporters."""

    @abstractmethod
    async def export(self, spans: List[SpanData]) -> None:
        """Export spans to the backend."""
        pass

    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown the exporter."""
        pass


class JaegerSpanExporter(SpanExporter):
    """Jaeger span exporter."""

    def __init__(self, endpoint: str, service_name: str):
        self.endpoint = endpoint
        self.service_name = service_name
        self.logger = get_logger(__name__)
        self._exporter = None

        if OPENTELEMETRY_AVAILABLE:
            self._exporter = JaegerExporter(
                agent_host_name=endpoint.split(":")[0] if ":" in endpoint else endpoint,
                agent_port=int(endpoint.split(":")[1]) if ":" in endpoint else 14268,
            )

    async def export(self, spans: List[SpanData]) -> None:
        """Export spans to Jaeger."""
        if not self._exporter:
            self.logger.warning("Jaeger exporter not available")
            return

        try:
            # Convert spans to OpenTelemetry format
            otel_spans = []
            for span in spans:
                # Would convert SpanData to OpenTelemetry Span
                pass

            # Export spans
            # self._exporter.export(otel_spans)

        except Exception as e:
            self.logger.error(f"Failed to export spans to Jaeger: {str(e)}")

    async def shutdown(self) -> None:
        """Shutdown the Jaeger exporter."""
        if self._exporter:
            self._exporter.shutdown()


class ConsoleSpanExporter(SpanExporter):
    """Console span exporter for development."""

    def __init__(self):
        self.logger = get_logger(__name__)

    async def export(self, spans: List[SpanData]) -> None:
        """Export spans to console."""
        for span in spans:
            self.logger.info(
                f"TRACE: {span.trace_id} | SPAN: {span.span_id} | "
                f"Operation: {span.operation_name} | "
                f"Duration: {span.duration:.3f}s | "
                f"Status: {span.status_code}"
            )

    async def shutdown(self) -> None:
        """Shutdown the console exporter."""
        pass


class BatchSpanProcessor(SpanProcessor):
    """Batch processor for efficient span export."""

    def __init__(
        self, exporter: SpanExporter, max_batch_size: int = 512, export_interval: float = 5.0
    ):
        self.exporter = exporter
        self.max_batch_size = max_batch_size
        self.export_interval = export_interval
        self.logger = get_logger(__name__)

        self._spans_buffer: List[SpanData] = []
        self._buffer_lock = asyncio.Lock()
        self._export_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()

        # Start export task
        self._export_task = asyncio.create_task(self._export_loop())

    async def on_start(self, span: SpanData, parent_context: Optional[TraceContext]) -> None:
        """Handle span start."""
        # No action needed for batch processor
        pass

    async def on_end(self, span: SpanData) -> None:
        """Handle span end by adding to buffer."""
        async with self._buffer_lock:
            self._spans_buffer.append(span)

            # Export if buffer is full
            if len(self._spans_buffer) >= self.max_batch_size:
                await self._export_batch()

    async def _export_loop(self) -> None:
        """Background task for periodic export."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(self.export_interval)

                async with self._buffer_lock:
                    if self._spans_buffer:
                        await self._export_batch()

            except Exception as e:
                self.logger.error(f"Error in export loop: {str(e)}")

    async def _export_batch(self) -> None:
        """Export current batch of spans."""
        if not self._spans_buffer:
            return

        spans_to_export = self._spans_buffer.copy()
        self._spans_buffer.clear()

        try:
            await self.exporter.export(spans_to_export)
        except Exception as e:
            self.logger.error(f"Failed to export span batch: {str(e)}")

    async def shutdown(self) -> None:
        """Shutdown the processor."""
        self._shutdown_event.set()

        if self._export_task:
            self._export_task.cancel()
            try:
                await self._export_task
            except asyncio.CancelledError:
                pass

        # Export remaining spans
        async with self._buffer_lock:
            if self._spans_buffer:
                await self._export_batch()

        await self.exporter.shutdown()


class SamplingDecider:
    """Makes sampling decisions for traces."""

    def __init__(self, config: TracingConfiguration):
        self.config = config
        self.logger = get_logger(__name__)

        # Rate limiting for rate-limited sampling
        self._trace_counts: Dict[str, int] = defaultdict(int)
        self._last_reset = time.time()

        # Adaptive sampling state
        self._error_rates: Dict[str, float] = defaultdict(float)
        self._latency_p99: Dict[str, float] = defaultdict(float)

    def should_sample(
        self, operation_name: str, parent_context: Optional[TraceContext] = None
    ) -> bool:
        """Decide whether to sample a trace."""
        # Check if operation is excluded
        if operation_name in self.config.excluded_operations:
            return False

        # Parent sampling decision takes precedence
        if parent_context and not parent_context.is_sampled:
            return False

        # Apply sampling strategy
        if self.config.sampling_strategy == SamplingStrategy.ALWAYS:
            return True
        elif self.config.sampling_strategy == SamplingStrategy.NEVER:
            return False
        elif self.config.sampling_strategy == SamplingStrategy.PROBABILISTIC:
            return self._probabilistic_sampling(operation_name)
        elif self.config.sampling_strategy == SamplingStrategy.RATE_LIMITED:
            return self._rate_limited_sampling(operation_name)
        elif self.config.sampling_strategy == SamplingStrategy.ADAPTIVE:
            return self._adaptive_sampling(operation_name)
        elif self.config.sampling_strategy == SamplingStrategy.ERROR_BASED:
            return self._error_based_sampling(operation_name)

        return True

    def _probabilistic_sampling(self, operation_name: str) -> bool:
        """Probabilistic sampling based on configured rate."""
        # Check component-specific sampling rate
        sampling_rate = self.config.component_sampling.get(
            operation_name, self.config.sampling_rate
        )
        return hash(f"{operation_name}{time.time()}") % 100 < sampling_rate * 100

    def _rate_limited_sampling(self, operation_name: str) -> bool:
        """Rate-limited sampling to control trace volume."""
        current_time = time.time()

        # Reset counters every second
        if current_time - self._last_reset > 1.0:
            self._trace_counts.clear()
            self._last_reset = current_time

        # Check if we've exceeded the rate limit
        if self._trace_counts[operation_name] >= self.config.max_traces_per_second:
            return False

        self._trace_counts[operation_name] += 1
        return True

    def _adaptive_sampling(self, operation_name: str) -> bool:
        """Adaptive sampling based on error rates and latency."""
        error_rate = self._error_rates.get(operation_name, 0.0)
        latency = self._latency_p99.get(operation_name, 0.0)

        # Increase sampling for high error rates or high latency
        if error_rate > 0.05 or latency > 1.0:  # 5% error rate or 1s latency
            return True

        # Use probabilistic sampling otherwise
        return self._probabilistic_sampling(operation_name)

    def _error_based_sampling(self, operation_name: str) -> bool:
        """Error-based sampling - sample all errors and some successes."""
        # This would require knowing if the operation will error, which we don't at start
        # So we use historical error rates
        error_rate = self._error_rates.get(operation_name, 0.0)

        # Always sample if high error rate
        if error_rate > 0.01:  # 1% error rate
            return True

        # Otherwise use reduced probabilistic sampling
        return hash(f"{operation_name}{time.time()}") % 100 < 5  # 5% sampling

    def update_metrics(self, operation_name: str, duration: float, is_error: bool) -> None:
        """Update metrics for adaptive sampling."""
        # Update error rate (exponential moving average)
        current_error_rate = self._error_rates.get(operation_name, 0.0)
        self._error_rates[operation_name] = 0.9 * current_error_rate + 0.1 * (
            1.0 if is_error else 0.0
        )

        # Update latency (simple moving average for P99 approximation)
        current_latency = self._latency_p99.get(operation_name, 0.0)
        self._latency_p99[operation_name] = 0.9 * current_latency + 0.1 * duration


class CorrelationManager:
    """Manages trace correlation across service boundaries."""

    def __init__(self):
        self.logger = get_logger(__name__)
        self._correlations: Dict[CorrelationId, Set[TraceId]] = defaultdict(set)
        self._trace_to_correlation: Dict[TraceId, CorrelationId] = {}

    def create_correlation(self, correlation_id: Optional[CorrelationId] = None) -> CorrelationId:
        """Create a new correlation ID."""
        if not correlation_id:
            correlation_id = str(uuid.uuid4())

        if correlation_id not in self._correlations:
            self._correlations[correlation_id] = set()

        return correlation_id

    def add_trace_to_correlation(self, trace_id: TraceId, correlation_id: CorrelationId) -> None:
        """Add a trace to a correlation."""
        self._correlations[correlation_id].add(trace_id)
        self._trace_to_correlation[trace_id] = correlation_id

    def get_correlated_traces(self, correlation_id: CorrelationId) -> Set[TraceId]:
        """Get all traces for a correlation."""
        return self._correlations.get(correlation_id, set())

    def get_correlation_for_trace(self, trace_id: TraceId) -> Optional[CorrelationId]:
        """Get correlation ID for a trace."""
        return self._trace_to_correlation.get(trace_id)


class EnhancedSpan:
    """Enhanced span implementation with AI assistant specific features."""

    def __init__(
        self,
        span_data: SpanData,
        processor: SpanProcessor,
        parent_context: Optional[TraceContext] = None,
    ):
        self.span_data = span_data
        self.processor = processor
        self.parent_context = parent_context
        self.logger = get_logger(__name__)

        # State management
        self._is_recording = True
        self._is_ended = False

        # Performance tracking
        self._checkpoints: List[Dict[str, Any]] = []

    def set_attribute(self, key: str, value: Any) -> "EnhancedSpan":
        """Set span attribute."""
        if not self._is_recording:
            return self

        # Sanitize sensitive attributes
        if key in {"password", "token", "api_key", "secret"}:
            value = "[REDACTED]"

        # Truncate long values
        if isinstance(value, str) and len(value) > 1024:
            value = value[:1024] + "..."

        self.span_data.attributes[key] = value
        return self

    def set_attributes(self, attributes: Dict[str, Any]) -> "EnhancedSpan":
        """Set multiple span attributes."""
        for key, value in attributes.items():
            self.set_attribute(key, value)
        return self

    def add_event(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None,
    ) -> "EnhancedSpan":
        """Add an event to the span."""
        if not self._is_recording:
            return self

        event = {
            "name": name,
            "timestamp": timestamp or datetime.now(timezone.utc),
            "attributes": attributes or {},
        }

        self.span_data.events.append(event)
        return self

    def add_link(
        self, trace_id: TraceId, span_id: SpanId, attributes: Optional[Dict[str, Any]] = None
    ) -> "EnhancedSpan":
        """Add a link to another span."""
        if not self._is_recording:
            return self

        link = {"trace_id": trace_id, "span_id": span_id, "attributes": attributes or {}}

        self.span_data.links.append(link)
        return self

    def set_status(self, status_code: str, status_message: Optional[str] = None) -> "EnhancedSpan":
        """Set span status."""
        if not self._is_recording:
            return self

        self.span_data.status_code = status_code
        self.span_data.status_message = status_message
        self.span_data.is_error = status_code != "OK"
        return self

    def record_exception(
        self, exception: Exception, attributes: Optional[Dict[str, Any]] = None
    ) -> "EnhancedSpan":
        """Record an exception in the span."""
        if not self._is_recording:
            return self

        # Set error status
        self.set_status("ERROR", str(exception))

        # Add exception event
        exception_attributes = {
            "exception.type": type(exception).__name__,
            "exception.message": str(exception),
            "exception.stacktrace": traceback.format_exc(),
        }

        if attributes:
            exception_attributes.update(attributes)

        self.add_event("exception", exception_attributes)
        return self

    def add_checkpoint(
        self, name: str, metadata: Optional[Dict[str, Any]] = None
    ) -> "EnhancedSpan":
        """Add a performance checkpoint."""
        checkpoint = {
            "name": name,
            "timestamp": datetime.now(timezone.utc),
            "relative_time": time.time() - self.span_data.start_time.timestamp(),
            "metadata": metadata or {},
        }

        self._checkpoints.append(checkpoint)
        self.add_event(f"checkpoint.{name}", checkpoint)
        return self

    def end(self, end_time: Optional[datetime] = None) -> None:
        """End the span."""
        if self._is_ended:
            return

        self.span_data.end_time = end_time or datetime.now(timezone.utc)
        self.span_data.duration = (
            self.span_data.end_time - self.span_data.start_time
        ).total_seconds()

        # Add performance summary
        if self._checkpoints:
            self.set_attribute("checkpoints.count", len(self._checkpoints))
            self.set_attribute("checkpoints.names", [cp["name"] for cp in self._checkpoints])

        self._is_ended = True
        self._is_recording = False

        # Notify processor
        asyncio.create_task(self.processor.on_end(self.span_data))

    def __enter__(self) -> "EnhancedSpan":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        if exc_type is not None:
            self.record_exception(exc_val)
        self.end()


class TraceManager:
    """
    Advanced Distributed Tracing Manager for the AI Assistant.

    This manager provides comprehensive tracing capabilities including:
    - Distributed trace correlation across microservices
    - Performance monitoring and bottleneck detection
    - Error tracking and debugging support
    - Multimodal processing trace visualization
    - Integration with external tracing systems (Jaeger, Zipkin, etc.)
    - Adaptive sampling for optimal performance
    - Real-time trace analysis and alerting
    - Component-specific instrumentation
    - Baggage propagation for cross-cutting concerns
    - Custom span processors and exporters
    """

    def __init__(self, container: Container, config: Optional[TracingConfiguration] = None):
        """
        Initialize the trace manager.

        Args:
            container: Dependency injection container
            config: Tracing configuration
        """
        self.container = container
        self.config = config or TracingConfiguration()
        self.logger = get_logger(__name__)

        # Core services
        self.config_loader = container.get(ConfigLoader)
        self.event_bus = container.get(EventBus)
        self.health_check = container.get(HealthCheck)

        # Observability
        try:
            self.metrics = container.get(MetricsCollector)
        except Exception:
            self.metrics = None

        # Tracing infrastructure
        self.sampling_decider = SamplingDecider(self.config)
        self.correlation_manager = CorrelationManager()

        # State management
        self._active_spans: Dict[SpanId, EnhancedSpan] = {}
        self._trace_contexts: Dict[TraceId, TraceContext] = {}
        self._current_context: Optional[TraceContext] = None
        self._context_stack: List[TraceContext] = []

        # Processors and exporters
        self._processors: List[SpanProcessor] = []
        self._exporters: List[SpanExporter] = []

        # Performance tracking
        self._trace_statistics: Dict[str, Dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        self._error_counts: Dict[str, int] = defaultdict(int)

        # Threading
        self._context_lock = threading.RLock()

        # Initialize components
        self._setup_exporters()
        self._setup_processors()
        self._setup_monitoring()
        self._setup_opentelemetry()

        # Register health check
        self.health_check.register_component("trace_manager", self._health_check_callback)

        self.logger.info("TraceManager initialized successfully")

    def _setup_exporters(self) -> None:
        """Setup span exporters based on configuration."""
        try:
            if self.config.backend == TracingBackend.JAEGER and self.config.endpoint:
                exporter = JaegerSpanExporter(self.config.endpoint, self.config.service_name)
                self._exporters.append(exporter)
            elif self.config.backend == TracingBackend.CONSOLE:
                exporter = ConsoleSpanExporter()
                self._exporters.append(exporter)

            self.logger.info(f"Initialized {len(self._exporters)} span exporters")

        except Exception as e:
            self.logger.error(f"Failed to setup exporters: {str(e)}")

    def _setup_processors(self) -> None:
        """Setup span processors."""
        try:
            for exporter in self._exporters:
                processor = BatchSpanProcessor(
                    exporter=exporter,
                    max_batch_size=self.config.max_export_batch_size,
                    export_interval=self.config.export_interval,
                )
                self._processors.append(processor)

            self.logger.info(f"Initialized {len(self._processors)} span processors")

        except Exception as e:
            self.logger.error(f"Failed to setup processors: {str(e)}")

    def _setup_monitoring(self) -> None:
        """Setup monitoring and metrics."""
        try:
            if self.metrics:
                self.metrics.register_counter("traces_started_total")
                self.metrics.register_counter("traces_completed_total")
                self.metrics.register_counter("traces_failed_total")
                self.metrics.register_counter("spans_created_total")
                self.metrics.register_histogram("span_duration_seconds")
                self.metrics.register_gauge("active_spans")
                self.metrics.register_counter("sampling_decisions_total")

        except Exception as e:
            self.logger.warning(f"Failed to setup monitoring: {str(e)}")

    def _setup_opentelemetry(self) -> None:
        """Setup OpenTelemetry integration if available."""
        if not OPENTELEMETRY_AVAILABLE:
            self.logger.info("OpenTelemetry not available, using custom tracing")
            return

        try:
            # Set up tracer provider
            trace.set_tracer_provider(TracerProvider())
            tracer_provider = trace.get_tracer_provider()

            # Add span processors
            if self._exporters:
                for exporter in self._exporters:
                    if hasattr(exporter, "_exporter"):
                        processor = BatchSpanProcessor(exporter._exporter)
                        tracer_provider.add_span_processor(processor)

            # Set up propagators
            propagate.set_global_textmap(
                CompositePropagator(
                    [
                        B3MultiFormat(),
                        JaegerPropagator(),
                    ]
                )
            )

            self.logger.info("OpenTelemetry integration initialized")

        except Exception as e:
            self.logger.warning(f"Failed to setup OpenTelemetry: {str(e)}")

    @handle_exceptions
    def create_trace(
        self,
        operation_name: str,
        correlation_id: Optional[CorrelationId] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        baggage: Optional[Dict[str, str]] = None,
    ) -> TraceContext:
        """
        Create a new trace context.

        Args:
            operation_name: Name of the operation being traced
            correlation_id: Optional correlation ID for trace correlation
            user_id: Optional user identifier
            session_id: Optional session identifier
            baggage: Optional baggage data

        Returns:
            TraceContext for the new trace
        """
        trace_id = str(uuid.uuid4())

        # Create trace context
        context = TraceContext(
            trace_id=trace_id,
            span_id=str(uuid.uuid4()),
            correlation_id=correlation_id,
            service_name=self.config.service_name,
            operation_name=operation_name,
            user_id=user_id,
            session_id=session_id,
            baggage=baggage or {},
            is_sampled=self.sampling_decider.should_sample(operation_name),
        )

        # Store trace context
        with self._context_lock:
            self._trace_contexts[trace_id] = context

        # Add to correlation if provided
        if correlation_id:
            self.correlation_manager.add_trace_to_correlation(trace_id, correlation_id)

        # Emit trace started event
        asyncio.create_task(
            self.event_bus.emit(
                TraceStarted(
                    trace_id=trace_id,
                    operation_name=operation_name,
                    user_id=user_id,
                    session_id=session_id,
                    correlation_id=correlation_id,
                    is_sampled=context.is_sampled,
                )
            )
        )

        # Update metrics
        if self.metrics:
            self.metrics.increment("traces_started_total")
            self.metrics.increment(
                "sampling_decisions_total", tags={"sampled": str(context.is_sampled)}
            )

        self.logger.debug(f"Created trace: {trace_id} for operation: {operation_name}")
        return context

    @handle_exceptions
    def start_span(
        self,
        operation_name: str,
        span_type: SpanType = SpanType.COMPONENT_INITIALIZATION,
        parent_context: Optional[TraceContext] = None,
        attributes: Optional[Dict[str, Any]] = None,
        links: Optional[List[Dict[str, Any]]] = None,
    ) -> EnhancedSpan:
        """
        Start a new span.

        Args:
            operation_name: Name of the operation
            span_type: Type of the span
            parent_context: Optional parent trace context
            attributes: Optional span attributes
            links: Optional span links

        Returns:
            EnhancedSpan instance
        """
        # Use current context if no parent provided
        if not parent_context:
            parent_context = self.get_current_context()

        # Create span data
        span_id = str(uuid.uuid4())
        trace_id = parent_context.trace_id if parent_context else str(uuid.uuid4())

        span_data = SpanData(
            span_id=span_id,
            trace_id=trace_id,
            parent_span_id=parent_context.span_id if parent_context else None,
            operation_name=operation_name,
            span_type=span_type,
            service_name=self.config.service_name,
            attributes=attributes or {},
            links=links or {},
        )

        # Create enhanced span
        span = EnhancedSpan(
            span_data, self._processors[0] if self._processors else None, parent_context
        )

        # Store active span
        with self._context_lock:
            self._active_spans[span_id] = span

        # Notify processors
        for processor in self._processors:
            asyncio.create_task(processor.on_start(span_data, parent_context))

        # Emit span started event
        asyncio.create_task(
            self.event_bus.emit(
                SpanStarted(
                    trace_id=trace_id,
                    span_id=span_id,
                    operation_name=operation_name,
                    span_type=span_type.value,
                    parent_span_id=parent_context.span_id if parent_context else None,
                )
            )
        )

        # Update metrics
        if self.metrics:
            self.metrics.increment("spans_created_total")
            self.metrics.set("active_spans", len(self._active_spans))

        self.logger.debug(f"Started span: {span_id} for operation: {operation_name}")
        return span

    @contextmanager
    def trace(
        self,
        operation_name: str,
        span_type: SpanType = SpanType.COMPONENT_INITIALIZATION,
        attributes: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[CorrelationId] = None,
    ) -> EnhancedSpan:
        """
        Context manager for tracing operations.

        Args:
            operation_name: Name of the operation
            span_type: Type of the span
            attributes: Optional span attributes
            correlation_id: Optional correlation ID

        Yields:
            EnhancedSpan instance
        """
        # Create trace context if this is a root span
        parent_context = self.get_current_context()
        if not parent_context:
            parent_context = self.create_trace(operation_name, correlation_id)

        # Start span
        span = self.start_span(operation_name, span_type, parent_context, attributes)

        # Set as current context
        old_context = self._current_context
        self._current_context = TraceContext(
            trace_id=span.span_data.trace_id,
            span_id=span.span_data.span_id,
            parent_span_id=parent_context.span_id if parent_context else None,
            correlation_id=correlation_id
            or (parent_context.correlation_id if parent_context else None),
            operation_name=operation_name,
        )

        try:
            yield span
        except Exception as e:
            span.record_exception(e)
            raise
        finally:
            # Restore previous context
            self._current_context = old_context

            # End span
            span.end()

            # Remove from active spans
            with self._context_lock:
                self._active_spans.pop(span.span_data.span_id, None)

    @asynccontextmanager
    async def async_trace(
        self,
        operation_name: str,
        span_type: SpanType = SpanType.COMPONENT_INITIALIZATION,
        attributes: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[CorrelationId] = None,
    ) -> EnhancedSpan:
        """
        Async context manager for tracing operations.

        Args:
            operation_name: Name of the operation
            span_type: Type of the span
            attributes: Optional span attributes
            correlation_id: Optional correlation ID

        Yields:
            EnhancedSpan instance
        """
        # Create trace context if this is a root span
        parent_context = self.get_current_context()
        if not parent_context:
            parent_context = self.create_trace(operation_name, correlation_id)

        # Start span
        span = self.start_span(operation_name, span_type, parent_context, attributes)

        # Set as current context
        old_context = self._current_context
        self._current_context = TraceContext(
            trace_id=span.span_data.trace_id,
            span_id=span.span_data.span_id,
            parent_span_id=parent_context.span_id if parent_context else None,
            correlation_id=correlation_id
            or (parent_context.correlation_id if parent_context else None),
            operation_name=operation_name,
        )

        try:
            yield span
        except Exception as e:
            span.record_exception(e)
            raise
        finally:
            # Restore previous context
            self._current_context = old_context

            # End span
            span.end()

            # Remove from active spans
            with self._context_lock:
                self._active_spans.pop(span.span_data.span_id, None)

    def get_current_context(self) -> Optional[TraceContext]:
        """Get the current trace context."""
        with self._context_lock:
            return self._current_context

    def set_current_context(self, context: Optional[TraceContext]) -> None:
        """Set the current trace context."""
        with self._context_lock:
            self._current_context = context

    def push_context(self, context: TraceContext) -> None:
        """Push a context onto the context stack."""
        with self._context_lock:
            self._context_stack.append(self._current_context)
            self._current_context = context

    def pop_context(self) -> Optional[TraceContext]:
        """Pop a context from the context stack."""
        with self._context_lock:
            if self._context_stack:
                self._current_context = self._context_stack.pop()
            else:
                self._current_context = None
            return self._current_context

    def add_baggage(self, key: str, value: str) -> None:
        """Add baggage to the current context."""
        context = self.get_current_context()
        if context:
            context.baggage[key] = value

    def get_baggage(self, key: str) -> Optional[str]:
        """Get baggage from the current context."""
        context = self.get_current_context()
        if context:
            return context.baggage.get(key)
        return None

    def create_correlation(self, correlation_id: Optional[CorrelationId] = None) -> CorrelationId:
        """Create a new correlation ID."""
        return self.correlation_manager.create_correlation(correlation_id)

    def get_correlated_traces(self, correlation_id: CorrelationId) -> Set[TraceId]:
        """Get all traces for a correlation."""
        return self.correlation_manager.get_correlated_traces(correlation_id)

    def get_trace_statistics(self, operation_name: Optional[str] = None) -> Dict[str, Any]:
        """Get trace statistics."""
        if operation_name:
            return dict(self._trace_statistics.get(operation_name, {}))
        return {op: dict(stats) for op, stats in self._trace_statistics.items()}

    def get_active_spans(self) -> List[Dict[str, Any]]:
        """Get information about active spans."""
        with self._context_lock:
            return [
                {
                    "span_id": span.span_data.span_id,
                    "trace_id": span.span_data.trace_id,
                    "operation_name": span.span_data.operation_name,
                    "span_type": span.span_data.span_type.value,
                    "start_time": span.span_data.start_time.isoformat(),
                    "duration": (
                        datetime.now(timezone.utc) - span.span_data.start_time
                    ).total_seconds(),
                    "attributes": span.span_data.attributes,
                }
                for span in self._active_spans.values()
            ]

    def update_configuration(self, config: TracingConfiguration) -> None:
        """Update tracing configuration."""
        old_config = self.config
        self.config = config

        # Update sampling decider
        self.sampling_decider = SamplingDecider(config)

        # Emit configuration change event
        asyncio.create_task(
            self.event_bus.emit(
                TracingConfigurationChanged(
                    old_config=asdict(old_config), new_config=asdict(config)
                )
            )
        )

        self.logger.info("Tracing configuration updated")

    async def _update_statistics(
        self, operation_name: str, duration: float, is_error: bool
    ) -> None:
        """Update operation statistics."""
        stats = self._trace_statistics[operation_name]

        # Update counters
        stats["count"] = stats.get("count", 0) + 1
        stats["total_duration"] = stats.get("total_duration", 0.0) + duration
        stats["avg_duration"] = stats["total_duration"] / stats["count"]

        # Update error count
        if is_error:
            self._error_counts[operation_name] += 1
            stats["error_rate"] = self._error_counts[operation_name] / stats["count"]

        # Update min/max duration
        stats["min_duration"] = min(stats.get("min_duration", float("inf")), duration)
        stats["max_duration"] = max(stats.get("max_duration", 0.0), duration)

        # Update sampling decider metrics
        self.sampling_decider.update_metrics(operation_name, duration, is_error)

        # Check for performance thresholds
        if duration > 5.0:  # 5 second threshold
            await self.event_bus.emit(
                PerformanceThresholdExceeded(
                    operation_name=operation_name,
                    duration=duration,
                    threshold=5.0,
                    trace_id=self._current_context.trace_id if self._current_context else None,
                )
            )

    async def _health_check_callback(self) -> Dict[str, Any]:
        """Health check callback for the trace manager."""
        try:
            active_spans_count = len(self._active_spans)
            active_traces_count = len(self._trace_contexts)

            return {
                "status": "healthy",
                "active_spans": active_spans_count,
                "active_traces": active_traces_count,
                "processors": len(self._processors),
                "exporters": len(self._exporters),
                "sampling_strategy": self.config.sampling_strategy.value,
                "backend": self.config.backend.value,
            }
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    async def initialize(self) -> None:
        """Initialize the trace manager."""
        try:
            # Initialize processors
            for processor in self._processors:
                if hasattr(processor, "initialize"):
                    await processor.initialize()

            # Start background tasks
            asyncio.create_task(self._cleanup_loop())
            asyncio.create_task(self._statistics_update_loop())

            self.logger.info("TraceManager initialization completed")

        except Exception as e:
            self.logger.error(f"Failed to initialize TraceManager: {str(e)}")
            raise TracingError(f"Initialization failed: {str(e)}")

    async def _cleanup_loop(self) -> None:
        """Background task for cleaning up old traces and spans."""
        while True:
            try:
                current_time = datetime.now(timezone.utc)
                cutoff_time = current_time - timedelta(hours=1)  # Clean up traces older than 1 hour

                # Clean up old trace contexts
                with self._context_lock:
                    old_traces = [
                        trace_id
                        for trace_id, context in self._trace_contexts.items()
                        if context.start_time < cutoff_time
                    ]

                    for trace_id in old_traces:
                        del self._trace_contexts[trace_id]

                # Clean up old active spans (shouldn't happen but safety net)
                with self._context_lock:
                    old_spans = [
                        span_id
                        for span_id, span in self._active_spans.items()
                        if span.span_data.start_time < cutoff_time
                    ]

                    for span_id in old_spans:
                        span = self._active_spans[span_id]
                        span.end()  # Force end the span
                        del self._active_spans[span_id]

                if old_traces or old_spans:
                    self.logger.info(
                        f"Cleaned up {len(old_traces)} old traces and {len(old_spans)} old spans"
                    )

                await asyncio.sleep(300)  # Clean up every 5 minutes

            except Exception as e:
                self.logger.error(f"Error in cleanup loop: {str(e)}")
                await asyncio.sleep(300)

    async def _statistics_update_loop(self) -> None:
        """Background task for updating statistics."""
        while True:
            try:
                # Update metrics
                if self.metrics:
                    with self._context_lock:
                        self.metrics.set("active_spans", len(self._active_spans))

                    # Update operation statistics
                    for operation_name, stats in self._trace_statistics.items():
                        self.metrics.set(
                            "operation_avg_duration_seconds",
                            stats.get("avg_duration", 0.0),
                            tags={"operation": operation_name},
                        )
                        self.metrics.set(
                            "operation_error_rate",
                            stats.get("error_rate", 0.0),
                            tags={"operation": operation_name},
                        )

                await asyncio.sleep(30)  # Update every 30 seconds

            except Exception as e:
                self.logger.error(f"Error in statistics update loop: {str(e)}")
                await asyncio.sleep(30)

    async def shutdown(self) -> None:
        """Gracefully shutdown the trace manager."""
        self.logger.info("Starting trace manager shutdown...")

        try:
            # End all active spans
            with self._context_lock:
                active_span_ids = list(self._active_spans.keys())

            for span_id in active_span_ids:
                span = self._active_spans.get(span_id)
                if span:
                    span.end()

            # Shutdown processors
            for processor in self._processors:
                await processor.shutdown()

            # Shutdown exporters
            for exporter in self._exporters:
                await exporter.shutdown()

            self.logger.info("Trace manager shutdown completed")

        except Exception as e:
            self.logger.error(f"Error during shutdown: {str(e)}")
            raise TracingError(f"Shutdown failed: {str(e)}")

    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            if hasattr(self, "_active_spans") and self._active_spans:
                self.logger.warning("TraceManager destroyed with active spans")
        except Exception:
            pass  # Ignore cleanup errors in destructor
