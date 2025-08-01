"""
Advanced Logging Middleware for AI Assistant API
Author: Drmusab
Last Modified: 2025-01-20 11:30:00 UTC

This module provides comprehensive logging middleware for the AI assistant API,
including request/response logging, performance metrics, security auditing,
error tracking, and integration with the core observability system.
"""

import hashlib
import json
import logging
import time
import traceback
import uuid
from contextlib import asynccontextmanager, contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set, Union
from urllib.parse import parse_qs, urlparse

import asyncio

# External dependencies
import fastapi
import structlog
from fastapi import Request, Response
from fastapi.security import HTTPBearer
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

# Assistant components
from src.assistant.session_manager import EnhancedSessionManager

# Core imports
from src.core.config.loader import ConfigLoader
from src.core.dependency_injection import Container
from src.core.events.event_bus import EventBus
from src.core.events.event_types import (
    APIRequestCompleted,
    APIRequestFailed,
    APIRequestStarted,
    AuthenticationAttempted,
    AuthorizationChecked,
    ErrorOccurred,
    PerformanceThresholdExceeded,
    SecurityViolationDetected,
    UserInteractionCompleted,
    UserInteractionStarted,
)
from src.core.security.sanitization import SecuritySanitizer
from src.observability.logging.config import get_logger
from src.observability.logging.formatters import (
    PerformanceFormatter,
    SecurityFormatter,
    StructuredFormatter,
)

# Observability
from src.observability.monitoring.metrics import MetricsCollector
from src.observability.monitoring.tracing import TraceManager


class LogLevel(Enum):
    """Logging levels for different types of events."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    SECURITY = "security"
    PERFORMANCE = "performance"
    AUDIT = "audit"


class LogCategory(Enum):
    """Categories of log events."""

    API_REQUEST = "api_request"
    API_RESPONSE = "api_response"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    SECURITY = "security"
    PERFORMANCE = "performance"
    ERROR = "error"
    BUSINESS_LOGIC = "business_logic"
    DATA_ACCESS = "data_access"
    SYSTEM = "system"
    AUDIT = "audit"
    COMPLIANCE = "compliance"


class SensitivityLevel(Enum):
    """Data sensitivity levels for logging."""

    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    TOP_SECRET = "top_secret"


@dataclass
class LogContext:
    """Context information for log entries."""

    request_id: str
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    correlation_id: Optional[str] = None
    trace_id: Optional[str] = None
    span_id: Optional[str] = None

    # Request metadata
    method: Optional[str] = None
    path: Optional[str] = None
    query_params: Dict[str, Any] = field(default_factory=dict)
    headers: Dict[str, str] = field(default_factory=dict)

    # User context
    user_agent: Optional[str] = None
    client_ip: Optional[str] = None
    client_location: Optional[str] = None

    # Timing
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Security context
    auth_method: Optional[str] = None
    permissions: List[str] = field(default_factory=list)
    risk_score: float = 0.0

    # Performance context
    processing_time: Optional[float] = None
    memory_usage: Optional[float] = None
    cpu_usage: Optional[float] = None

    # Additional metadata
    tags: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LogEntry:
    """Structured log entry."""

    level: LogLevel
    category: LogCategory
    message: str
    context: LogContext

    # Event details
    event_type: Optional[str] = None
    component: Optional[str] = None
    operation: Optional[str] = None

    # Data and errors
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[Exception] = None
    error_trace: Optional[str] = None

    # Classification
    sensitivity: SensitivityLevel = SensitivityLevel.INTERNAL
    retention_period: int = 90  # days

    # Performance metrics
    response_time: Optional[float] = None
    status_code: Optional[int] = None
    response_size: Optional[int] = None

    # Compliance and audit
    compliance_tags: Set[str] = field(default_factory=set)
    audit_required: bool = False
    pii_detected: bool = False

    # Additional fields
    extra: Dict[str, Any] = field(default_factory=dict)


class DataSanitizer:
    """Sanitizes sensitive data before logging."""

    def __init__(self):
        self.logger = get_logger(__name__)

        # Sensitive field patterns
        self.sensitive_patterns = {
            "password",
            "token",
            "api_key",
            "secret",
            "private_key",
            "credit_card",
            "ssn",
            "social_security",
            "phone",
            "email",
            "authorization",
            "authentication",
            "session_id",
            "csrf_token",
        }

        # PII patterns
        self.pii_patterns = {
            "name",
            "address",
            "phone",
            "email",
            "ssn",
            "dob",
            "date_of_birth",
            "credit_card",
            "passport",
            "license",
            "ip_address",
            "location",
        }

    def sanitize_data(self, data: Any, sensitivity: SensitivityLevel) -> Any:
        """Sanitize data based on sensitivity level."""
        if sensitivity in [
            SensitivityLevel.CONFIDENTIAL,
            SensitivityLevel.RESTRICTED,
            SensitivityLevel.TOP_SECRET,
        ]:
            return self._deep_sanitize(data)
        elif sensitivity == SensitivityLevel.INTERNAL:
            return self._partial_sanitize(data)
        else:
            return data

    def _deep_sanitize(self, data: Any) -> Any:
        """Perform deep sanitization of sensitive data."""
        if isinstance(data, dict):
            sanitized = {}
            for key, value in data.items():
                if self._is_sensitive_field(key):
                    sanitized[key] = self._mask_value(value)
                else:
                    sanitized[key] = self._deep_sanitize(value)
            return sanitized
        elif isinstance(data, list):
            return [self._deep_sanitize(item) for item in data]
        elif isinstance(data, str) and self._contains_pii(data):
            return self._mask_value(data)
        else:
            return data

    def _partial_sanitize(self, data: Any) -> Any:
        """Perform partial sanitization for internal use."""
        if isinstance(data, dict):
            sanitized = {}
            for key, value in data.items():
                if any(pattern in key.lower() for pattern in self.sensitive_patterns):
                    sanitized[key] = self._mask_value(value)
                else:
                    sanitized[key] = self._partial_sanitize(value)
            return sanitized
        elif isinstance(data, list):
            return [self._partial_sanitize(item) for item in data]
        else:
            return data

    def _is_sensitive_field(self, field_name: str) -> bool:
        """Check if field name indicates sensitive data."""
        field_lower = field_name.lower()
        return any(pattern in field_lower for pattern in self.sensitive_patterns)

    def _contains_pii(self, text: str) -> bool:
        """Check if text contains personally identifiable information."""
        # Simplified PII detection - in production, use more sophisticated methods
        text_lower = text.lower()
        return any(pattern in text_lower for pattern in self.pii_patterns)

    def _mask_value(self, value: Any) -> str:
        """Mask sensitive values."""
        if value is None:
            return None

        value_str = str(value)
        if len(value_str) <= 4:
            return "***"
        else:
            return value_str[:2] + "*" * (len(value_str) - 4) + value_str[-2:]


class RequestIDGenerator:
    """Generates unique request IDs for tracking."""

    @staticmethod
    def generate() -> str:
        """Generate a unique request ID."""
        return str(uuid.uuid4())

    @staticmethod
    def extract_from_headers(headers: Dict[str, str]) -> Optional[str]:
        """Extract request ID from headers if present."""
        for header_name in ["X-Request-ID", "X-Correlation-ID", "Request-ID"]:
            if header_name in headers:
                return headers[header_name]
        return None


class PerformanceTracker:
    """Tracks performance metrics for requests."""

    def __init__(self):
        self.logger = get_logger(__name__)
        self.performance_thresholds = {
            "response_time": 5.0,  # seconds
            "memory_usage": 100.0,  # MB
            "cpu_usage": 80.0,  # percentage
        }

    @contextmanager
    def track_request(self, context: LogContext):
        """Track request performance metrics."""
        start_time = time.time()
        start_memory = self._get_memory_usage()

        try:
            yield
        finally:
            end_time = time.time()
            end_memory = self._get_memory_usage()

            # Calculate metrics
            response_time = end_time - start_time
            memory_delta = end_memory - start_memory

            # Update context
            context.processing_time = response_time
            context.memory_usage = memory_delta

            # Check thresholds
            self._check_performance_thresholds(context, response_time, memory_delta)

    def _get_memory_usage(self) -> float:
        """Get current memory usage."""
        try:
            import psutil

            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # MB
        except ImportError:
            return 0.0

    def _check_performance_thresholds(
        self, context: LogContext, response_time: float, memory_usage: float
    ):
        """Check if performance thresholds are exceeded."""
        if response_time > self.performance_thresholds["response_time"]:
            self.logger.warning(
                f"Response time threshold exceeded: {response_time:.2f}s "
                f"(threshold: {self.performance_thresholds['response_time']}s)",
                extra={"context": asdict(context)},
            )

        if memory_usage > self.performance_thresholds["memory_usage"]:
            self.logger.warning(
                f"Memory usage threshold exceeded: {memory_usage:.2f}MB "
                f"(threshold: {self.performance_thresholds['memory_usage']}MB)",
                extra={"context": asdict(context)},
            )


class SecurityLogger:
    """Specialized logging for security events."""

    def __init__(self, event_bus: EventBus):
        self.logger = get_logger("security")
        self.event_bus = event_bus

        # Security event patterns
        self.suspicious_patterns = {
            "sql_injection": ["union", "select", "drop", "insert", "delete"],
            "xss_attempt": ["<script", "javascript:", "onerror=", "onload="],
            "path_traversal": ["../", "..\\", "/etc/", "/proc/"],
            "command_injection": ["&&", "||", ";", "|", "`"],
        }

    def log_authentication_attempt(
        self, context: LogContext, success: bool, method: str, reason: Optional[str] = None
    ):
        """Log authentication attempts."""
        log_entry = LogEntry(
            level=LogLevel.SECURITY if not success else LogLevel.INFO,
            category=LogCategory.AUTHENTICATION,
            message=f"Authentication {'successful' if success else 'failed'} using {method}",
            context=context,
            event_type="authentication_attempt",
            data={
                "success": success,
                "method": method,
                "reason": reason,
                "client_ip": context.client_ip,
                "user_agent": context.user_agent,
            },
            audit_required=True,
        )

        self._emit_security_event(log_entry)

    def log_authorization_check(
        self,
        context: LogContext,
        resource: str,
        action: str,
        granted: bool,
        reason: Optional[str] = None,
    ):
        """Log authorization checks."""
        log_entry = LogEntry(
            level=LogLevel.SECURITY if not granted else LogLevel.DEBUG,
            category=LogCategory.AUTHORIZATION,
            message=f"Authorization {'granted' if granted else 'denied'} for {action} on {resource}",
            context=context,
            event_type="authorization_check",
            data={
                "resource": resource,
                "action": action,
                "granted": granted,
                "reason": reason,
                "permissions": context.permissions,
            },
            audit_required=not granted,
        )

        self._emit_security_event(log_entry)

    def detect_suspicious_activity(self, context: LogContext, request_data: Dict[str, Any]):
        """Detect and log suspicious activity."""
        threats_detected = []

        for threat_type, patterns in self.suspicious_patterns.items():
            if self._check_patterns(request_data, patterns):
                threats_detected.append(threat_type)

        if threats_detected:
            log_entry = LogEntry(
                level=LogLevel.CRITICAL,
                category=LogCategory.SECURITY,
                message=f"Suspicious activity detected: {', '.join(threats_detected)}",
                context=context,
                event_type="security_violation",
                data={
                    "threats": threats_detected,
                    "request_data": self._sanitize_for_security_log(request_data),
                    "risk_score": len(threats_detected) * 10,
                },
                audit_required=True,
                sensitivity=SensitivityLevel.RESTRICTED,
            )

            # Update risk score
            context.risk_score = len(threats_detected) * 10

            self._emit_security_event(log_entry)

    def _check_patterns(self, data: Dict[str, Any], patterns: List[str]) -> bool:
        """Check if data contains suspicious patterns."""
        data_str = json.dumps(data).lower()
        return any(pattern in data_str for pattern in patterns)

    def _sanitize_for_security_log(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize data for security logging while preserving threat indicators."""
        # Keep threat patterns but sanitize sensitive data
        sanitizer = DataSanitizer()
        return sanitizer.sanitize_data(data, SensitivityLevel.INTERNAL)

    def _emit_security_event(self, log_entry: LogEntry):
        """Emit security event to event bus."""
        asyncio.create_task(
            self.event_bus.emit(
                SecurityViolationDetected(
                    event_type=log_entry.event_type,
                    severity=log_entry.level.value,
                    description=log_entry.message,
                    context=asdict(log_entry.context),
                    data=log_entry.data,
                )
            )
        )


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Advanced Logging Middleware for the AI Assistant API.

    Features:
    - Comprehensive request/response logging
    - Performance monitoring and metrics
    - Security event detection and logging
    - Structured logging with correlation IDs
    - Data sanitization for sensitive information
    - Compliance and audit trail support
    - Integration with observability systems
    - Error tracking and debugging support
    """

    def __init__(self, app: ASGIApp, container: Container, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the logging middleware.

        Args:
            app: ASGI application
            container: Dependency injection container
            config: Optional configuration overrides
        """
        super().__init__(app)

        self.container = container
        self.logger = get_logger(__name__)

        # Get core services
        self.config_loader = container.get(ConfigLoader)
        self.event_bus = container.get(EventBus)
        self.metrics = container.get(MetricsCollector)
        self.tracer = container.get(TraceManager)

        try:
            self.session_manager = container.get(EnhancedSessionManager)
            self.security_sanitizer = container.get(SecuritySanitizer)
        except Exception:
            self.session_manager = None
            self.security_sanitizer = None

        # Configuration
        self.config = self._load_configuration(config)

        # Initialize components
        self.data_sanitizer = DataSanitizer()
        self.performance_tracker = PerformanceTracker()
        self.security_logger = SecurityLogger(self.event_bus)

        # Setup structured logging
        self._setup_structured_logging()

        # Setup metrics
        self._setup_metrics()

        # Excluded paths (don't log these)
        self.excluded_paths = set(
            self.config.get(
                "excluded_paths", ["/health", "/metrics", "/favicon.ico", "/robots.txt"]
            )
        )

        # Sensitive headers to exclude from logging
        self.sensitive_headers = set(
            self.config.get(
                "sensitive_headers",
                [
                    "authorization",
                    "x-api-key",
                    "cookie",
                    "set-cookie",
                    "x-auth-token",
                    "x-session-token",
                ],
            )
        )

        self.logger.info("LoggingMiddleware initialized successfully")

    def _load_configuration(self, config_override: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Load logging middleware configuration."""
        default_config = {
            "log_requests": True,
            "log_responses": True,
            "log_request_body": True,
            "log_response_body": False,
            "max_body_size": 1024,  # bytes
            "enable_performance_tracking": True,
            "enable_security_logging": True,
            "enable_audit_logging": True,
            "log_level": "INFO",
            "correlation_header": "X-Correlation-ID",
            "request_id_header": "X-Request-ID",
            "sensitive_data_masking": True,
            "pii_detection": True,
            "compliance_logging": True,
            "retention_days": 90,
        }

        # Merge with config from loader
        api_config = self.config_loader.get("api.logging", {})
        default_config.update(api_config)

        # Apply overrides
        if config_override:
            default_config.update(config_override)

        return default_config

    def _setup_structured_logging(self):
        """Setup structured logging formatters."""
        # Configure structlog for consistent structured logging
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer(),
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )

    def _setup_metrics(self):
        """Setup metrics collection."""
        try:
            # Register middleware metrics
            self.metrics.register_counter("api_requests_total", ["method", "endpoint", "status"])
            self.metrics.register_histogram("api_request_duration_seconds", ["method", "endpoint"])
            self.metrics.register_histogram("api_request_size_bytes", ["method", "endpoint"])
            self.metrics.register_histogram("api_response_size_bytes", ["method", "endpoint"])
            self.metrics.register_counter("api_errors_total", ["method", "endpoint", "error_type"])
            self.metrics.register_counter("security_events_total", ["event_type", "severity"])
            self.metrics.register_gauge("active_requests")

        except Exception as e:
            self.logger.warning(f"Failed to setup metrics: {str(e)}")

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process the request through the logging middleware.

        Args:
            request: Incoming HTTP request
            call_next: Next middleware/handler in the chain

        Returns:
            HTTP response
        """
        # Skip logging for excluded paths
        if request.url.path in self.excluded_paths:
            return await call_next(request)

        # Generate request context
        context = await self._create_log_context(request)

        # Start performance tracking
        start_time = time.time()
        request_size = await self._get_request_size(request)

        try:
            with self.performance_tracker.track_request(context):
                # Log incoming request
                if self.config["log_requests"]:
                    await self._log_request(request, context)

                # Security analysis
                if self.config["enable_security_logging"]:
                    await self._analyze_security(request, context)

                # Emit request started event
                await self.event_bus.emit(
                    APIRequestStarted(
                        request_id=context.request_id,
                        method=context.method,
                        path=context.path,
                        user_id=context.user_id,
                        session_id=context.session_id,
                        client_ip=context.client_ip,
                    )
                )

                # Process request
                response = await call_next(request)

                # Calculate metrics
                response_time = time.time() - start_time
                response_size = self._get_response_size(response)

                # Update context with response info
                context.processing_time = response_time

                # Log response
                if self.config["log_responses"]:
                    await self._log_response(response, context, response_time)

                # Update metrics
                self._update_metrics(context, response, response_time, request_size, response_size)

                # Emit request completed event
                await self.event_bus.emit(
                    APIRequestCompleted(
                        request_id=context.request_id,
                        method=context.method,
                        path=context.path,
                        status_code=response.status_code,
                        response_time=response_time,
                        user_id=context.user_id,
                        session_id=context.session_id,
                    )
                )

                return response

        except Exception as e:
            # Log error
            response_time = time.time() - start_time
            await self._log_error(e, context, response_time)

            # Update error metrics
            self._update_error_metrics(context, e)

            # Emit request failed event
            await self.event_bus.emit(
                APIRequestFailed(
                    request_id=context.request_id,
                    method=context.method,
                    path=context.path,
                    error_type=type(e).__name__,
                    error_message=str(e),
                    response_time=response_time,
                    user_id=context.user_id,
                    session_id=context.session_id,
                )
            )

            raise

    async def _create_log_context(self, request: Request) -> LogContext:
        """Create logging context from request."""
        # Generate or extract request ID
        request_id = RequestIDGenerator.extract_from_headers(dict(request.headers))
        if not request_id:
            request_id = RequestIDGenerator.generate()

        # Extract correlation ID
        correlation_id = request.headers.get(self.config["correlation_header"])

        # Extract trace context
        trace_id = None
        span_id = None
        if self.tracer:
            trace_context = self.tracer.get_current_trace_context()
            if trace_context:
                trace_id = trace_context.get("trace_id")
                span_id = trace_context.get("span_id")

        # Get client information
        client_ip = self._get_client_ip(request)
        user_agent = request.headers.get("user-agent")

        # Extract user and session info
        user_id = None
        session_id = None
        auth_method = None
        permissions = []

        # Try to extract user info from session or JWT
        if hasattr(request.state, "user_id"):
            user_id = request.state.user_id

        if hasattr(request.state, "session_id"):
            session_id = request.state.session_id

        if hasattr(request.state, "auth_method"):
            auth_method = request.state.auth_method

        if hasattr(request.state, "permissions"):
            permissions = request.state.permissions

        # Sanitize headers
        headers = self._sanitize_headers(dict(request.headers))

        # Parse query parameters
        query_params = dict(request.query_params)

        return LogContext(
            request_id=request_id,
            session_id=session_id,
            user_id=user_id,
            correlation_id=correlation_id,
            trace_id=trace_id,
            span_id=span_id,
            method=request.method,
            path=request.url.path,
            query_params=query_params,
            headers=headers,
            user_agent=user_agent,
            client_ip=client_ip,
            auth_method=auth_method,
            permissions=permissions,
        )

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address from request."""
        # Check for forwarded headers
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip

        # Fallback to direct client
        if hasattr(request, "client") and request.client:
            return request.client.host

        return "unknown"

    def _sanitize_headers(self, headers: Dict[str, str]) -> Dict[str, str]:
        """Sanitize sensitive headers for logging."""
        sanitized = {}

        for key, value in headers.items():
            key_lower = key.lower()
            if key_lower in self.sensitive_headers:
                sanitized[key] = self.data_sanitizer._mask_value(value)
            else:
                sanitized[key] = value

        return sanitized

    async def _get_request_size(self, request: Request) -> int:
        """Get request body size."""
        try:
            body = await request.body()
            return len(body) if body else 0
        except Exception:
            return 0

    def _get_response_size(self, response: Response) -> int:
        """Get response size."""
        try:
            if hasattr(response, "body"):
                return len(response.body) if response.body else 0
            return 0
        except Exception:
            return 0

    async def _log_request(self, request: Request, context: LogContext):
        """Log incoming request."""
        try:
            # Prepare request data
            request_data = {
                "method": request.method,
                "path": request.url.path,
                "query_params": context.query_params,
                "headers": context.headers,
                "client_ip": context.client_ip,
                "user_agent": context.user_agent,
            }

            # Add body if configured and not too large
            if self.config["log_request_body"] and request.method in ["POST", "PUT", "PATCH"]:
                try:
                    body = await request.body()
                    if body and len(body) <= self.config["max_body_size"]:
                        # Try to parse as JSON
                        try:
                            body_data = json.loads(body.decode("utf-8"))
                            request_data["body"] = self.data_sanitizer.sanitize_data(
                                body_data, SensitivityLevel.INTERNAL
                            )
                        except (json.JSONDecodeError, UnicodeDecodeError):
                            request_data["body"] = f"<binary data: {len(body)} bytes>"
                except Exception as e:
                    request_data["body_error"] = str(e)

            # Create log entry
            log_entry = LogEntry(
                level=LogLevel.INFO,
                category=LogCategory.API_REQUEST,
                message=f"API Request: {request.method} {request.url.path}",
                context=context,
                event_type="api_request",
                component="api_middleware",
                operation="request_processing",
                data=request_data,
                sensitivity=SensitivityLevel.INTERNAL,
            )

            # Log with structured format
            structured_logger = structlog.get_logger("api.request")
            structured_logger.info(
                log_entry.message,
                request_id=context.request_id,
                method=request.method,
                path=request.url.path,
                user_id=context.user_id,
                session_id=context.session_id,
                client_ip=context.client_ip,
                correlation_id=context.correlation_id,
                trace_id=context.trace_id,
                data=request_data,
            )

        except Exception as e:
            self.logger.error(f"Failed to log request: {str(e)}")

    async def _log_response(self, response: Response, context: LogContext, response_time: float):
        """Log outgoing response."""
        try:
            # Prepare response data
            response_data = {
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "response_time": response_time,
            }

            # Add body if configured and not too large
            if self.config["log_response_body"] and hasattr(response, "body") and response.body:
                if len(response.body) <= self.config["max_body_size"]:
                    try:
                        body_data = json.loads(response.body.decode("utf-8"))
                        response_data["body"] = self.data_sanitizer.sanitize_data(
                            body_data, SensitivityLevel.INTERNAL
                        )
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        response_data["body"] = f"<binary data: {len(response.body)} bytes>"

            # Determine log level based on status code
            if response.status_code >= 500:
                level = LogLevel.ERROR
            elif response.status_code >= 400:
                level = LogLevel.WARNING
            else:
                level = LogLevel.INFO

            # Create log entry
            log_entry = LogEntry(
                level=level,
                category=LogCategory.API_RESPONSE,
                message=f"API Response: {response.status_code} for {context.method} {context.path}",
                context=context,
                event_type="api_response",
                component="api_middleware",
                operation="response_processing",
                data=response_data,
                status_code=response.status_code,
                response_time=response_time,
                sensitivity=SensitivityLevel.INTERNAL,
            )

            # Log with structured format
            structured_logger = structlog.get_logger("api.response")
            structured_logger.log(
                level.value.upper(),
                log_entry.message,
                request_id=context.request_id,
                method=context.method,
                path=context.path,
                status_code=response.status_code,
                response_time=response_time,
                user_id=context.user_id,
                session_id=context.session_id,
                correlation_id=context.correlation_id,
                trace_id=context.trace_id,
                data=response_data,
            )

        except Exception as e:
            self.logger.error(f"Failed to log response: {str(e)}")

    async def _analyze_security(self, request: Request, context: LogContext):
        """Analyze request for security threats."""
        try:
            # Collect request data for analysis
            request_data = {
                "path": request.url.path,
                "query_params": dict(request.query_params),
                "headers": dict(request.headers),
                "method": request.method,
            }

            # Add body for analysis if available
            if request.method in ["POST", "PUT", "PATCH"]:
                try:
                    body = await request.body()
                    if body:
                        try:
                            request_data["body"] = json.loads(body.decode("utf-8"))
                        except (json.JSONDecodeError, UnicodeDecodeError):
                            request_data["body"] = body.decode("utf-8", errors="ignore")
                except Exception:
                    pass

            # Perform security analysis
            self.security_logger.detect_suspicious_activity(context, request_data)

            # Additional security checks
            await self._check_rate_limiting(context)
            await self._check_authentication_security(context)

        except Exception as e:
            self.logger.error(f"Security analysis failed: {str(e)}")

    async def _check_rate_limiting(self, context: LogContext):
        """Check for rate limiting violations."""
        # This would integrate with a rate limiting service
        # For now, just a placeholder
        pass

    async def _check_authentication_security(self, context: LogContext):
        """Check for authentication security issues."""
        # This would perform additional authentication security checks
        # For now, just a placeholder
        pass

    async def _log_error(self, error: Exception, context: LogContext, response_time: float):
        """Log error with full context."""
        try:
            # Create error log entry
            log_entry = LogEntry(
                level=LogLevel.ERROR,
                category=LogCategory.ERROR,
                message=f"API Error: {type(error).__name__}: {str(error)}",
                context=context,
                event_type="api_error",
                component="api_middleware",
                operation="error_handling",
                error=error,
                error_trace=traceback.format_exc(),
                response_time=response_time,
                data={
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "method": context.method,
                    "path": context.path,
                },
                sensitivity=SensitivityLevel.INTERNAL,
                audit_required=True,
            )

            # Log with structured format
            structured_logger = structlog.get_logger("api.error")
            structured_logger.error(
                log_entry.message,
                request_id=context.request_id,
                method=context.method,
                path=context.path,
                error_type=type(error).__name__,
                error_message=str(error),
                response_time=response_time,
                user_id=context.user_id,
                session_id=context.session_id,
                correlation_id=context.correlation_id,
                trace_id=context.trace_id,
                stack_trace=traceback.format_exc(),
            )

            # Emit error event
            await self.event_bus.emit(
                ErrorOccurred(
                    component="api_middleware",
                    error_type=type(error).__name__,
                    error_message=str(error),
                    context=asdict(context),
                    severity="error",
                    stack_trace=traceback.format_exc(),
                )
            )

        except Exception as e:
            self.logger.critical(f"Failed to log error: {str(e)}")

    def _update_metrics(
        self,
        context: LogContext,
        response: Response,
        response_time: float,
        request_size: int,
        response_size: int,
    ):
        """Update metrics with request/response data."""
        try:
            # Basic request metrics
            self.metrics.increment(
                "api_requests_total",
                tags={
                    "method": context.method,
                    "endpoint": self._normalize_endpoint(context.path),
                    "status": str(response.status_code),
                },
            )

            # Response time metrics
            self.metrics.record(
                "api_request_duration_seconds",
                response_time,
                tags={"method": context.method, "endpoint": self._normalize_endpoint(context.path)},
            )

            # Size metrics
            self.metrics.record(
                "api_request_size_bytes",
                request_size,
                tags={"method": context.method, "endpoint": self._normalize_endpoint(context.path)},
            )

            self.metrics.record(
                "api_response_size_bytes",
                response_size,
                tags={"method": context.method, "endpoint": self._normalize_endpoint(context.path)},
            )

        except Exception as e:
            self.logger.warning(f"Failed to update metrics: {str(e)}")

    def _update_error_metrics(self, context: LogContext, error: Exception):
        """Update error metrics."""
        try:
            self.metrics.increment(
                "api_errors_total",
                tags={
                    "method": context.method,
                    "endpoint": self._normalize_endpoint(context.path),
                    "error_type": type(error).__name__,
                },
            )
        except Exception as e:
            self.logger.warning(f"Failed to update error metrics: {str(e)}")

    def _normalize_endpoint(self, path: str) -> str:
        """Normalize endpoint path for metrics (remove IDs)."""
        # Replace UUIDs and numeric IDs with placeholders
        import re

        # Replace UUIDs
        path = re.sub(
            r"/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
            "/{id}",
            path,
            flags=re.IGNORECASE,
        )

        # Replace numeric IDs
        path = re.sub(r"/\d+", "/{id}", path)

        return path


class AuditLogger:
    """Specialized logger for audit and compliance events."""

    def __init__(self, event_bus: EventBus, config: Dict[str, Any]):
        self.logger = get_logger("audit")
        self.event_bus = event_bus
        self.config = config

        # Setup audit-specific structured logger
        self.audit_logger = structlog.get_logger("audit")

    async def log_data_access(
        self, context: LogContext, resource: str, action: str, data_classification: str
    ):
        """Log data access for compliance."""
        audit_entry = {
            "event_type": "data_access",
            "resource": resource,
            "action": action,
            "data_classification": data_classification,
            "user_id": context.user_id,
            "session_id": context.session_id,
            "client_ip": context.client_ip,
            "timestamp": context.timestamp.isoformat(),
            "request_id": context.request_id,
        }

        self.audit_logger.info(f"Data access: {action} on {resource}", **audit_entry)

    async def log_user_action(self, context: LogContext, action: str, target: str, result: str):
        """Log user actions for audit trail."""
        audit_entry = {
            "event_type": "user_action",
            "action": action,
            "target": target,
            "result": result,
            "user_id": context.user_id,
            "session_id": context.session_id,
            "client_ip": context.client_ip,
            "timestamp": context.timestamp.isoformat(),
            "request_id": context.request_id,
        }

        self.audit_logger.info(f"User action: {action} on {target} - {result}", **audit_entry)


def create_logging_middleware(
    container: Container, config: Optional[Dict[str, Any]] = None
) -> LoggingMiddleware:
    """
    Factory function to create logging middleware instance.

    Args:
        container: Dependency injection container
        config: Optional configuration overrides

    Returns:
        Configured logging middleware instance
    """
    return LoggingMiddleware(None, container, config)


# FastAPI integration helper
def add_logging_middleware(
    app: fastapi.FastAPI, container: Container, config: Optional[Dict[str, Any]] = None
):
    """
    Add logging middleware to FastAPI application.

    Args:
        app: FastAPI application instance
        container: Dependency injection container
        config: Optional configuration overrides
    """
    middleware = LoggingMiddleware(app, container, config)
    app.add_middleware(LoggingMiddleware, container=container, config=config)


__all__ = [
    "LoggingMiddleware",
    "LogLevel",
    "LogCategory",
    "SensitivityLevel",
    "LogContext",
    "LogEntry",
    "DataSanitizer",
    "SecurityLogger",
    "AuditLogger",
    "create_logging_middleware",
    "add_logging_middleware",
]
