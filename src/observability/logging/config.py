"""
Advanced Logging Configuration System
Author: Drmusab
Last Modified: 2025-06-26 10:52:10 UTC

This module provides comprehensive logging functionality for the AI assistant,
including structured logging, contextual information, performance monitoring,
multi-channel output, and deep integration with all core system components.
"""

import functools
import inspect
import json
import logging
import logging.handlers
import os
import queue
import sys
import threading
import time
import traceback
import uuid
import weakref
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Union

import asyncio

# Third-party imports
try:
    import colorama
    from colorama import Back, Fore, Style

    colorama.init()
    COLORAMA_AVAILABLE = True
except ImportError:
    COLORAMA_AVAILABLE = False

try:
    import structlog

    STRUCTLOG_AVAILABLE = True
except ImportError:
    STRUCTLOG_AVAILABLE = False

try:
    import uvloop

    UVLOOP_AVAILABLE = True
except ImportError:
    UVLOOP_AVAILABLE = False


class LogLevel(Enum):
    """Enhanced log levels with numeric values."""

    TRACE = 5
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50
    SECURITY = 60  # Special level for security events
    AUDIT = 70  # Special level for audit events


class LogFormat(Enum):
    """Log output formats."""

    JSON = "json"
    STRUCTURED = "structured"
    CONSOLE = "console"
    COMPACT = "compact"
    DETAILED = "detailed"
    SECURITY = "security"


class LogChannel(Enum):
    """Log output channels."""

    CONSOLE = "console"
    FILE = "file"
    SYSLOG = "syslog"
    REMOTE = "remote"
    STREAM = "stream"
    ELASTIC = "elastic"
    WEBHOOK = "webhook"


@dataclass
class LogContext:
    """Context information for log entries."""

    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    component: Optional[str] = None
    workflow_id: Optional[str] = None
    interaction_id: Optional[str] = None
    request_id: Optional[str] = None

    # Technical context
    thread_id: Optional[str] = None
    process_id: Optional[int] = None
    hostname: Optional[str] = None
    service_name: str = "ai_assistant"

    # Performance context
    execution_time: Optional[float] = None
    memory_usage: Optional[float] = None
    cpu_usage: Optional[float] = None

    # Business context
    operation: Optional[str] = None
    category: Optional[str] = None
    tags: Set[str] = field(default_factory=set)

    # Security context
    security_level: str = "normal"
    sensitive_data: bool = False

    # Additional metadata
    custom_fields: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class LoggingConfig:
    """Comprehensive logging configuration."""

    # Basic settings
    level: LogLevel = LogLevel.INFO
    format_type: LogFormat = LogFormat.STRUCTURED

    # Output channels
    console_enabled: bool = True
    file_enabled: bool = True
    remote_enabled: bool = False

    # File settings
    log_dir: Path = field(default_factory=lambda: Path("data/logs"))
    max_file_size: int = 100 * 1024 * 1024  # 100MB
    backup_count: int = 10

    # Performance settings
    async_logging: bool = True
    buffer_size: int = 1000
    flush_interval: float = 5.0

    # Filtering and sampling
    component_levels: Dict[str, LogLevel] = field(default_factory=dict)
    sampling_rate: float = 1.0
    rate_limit_per_second: int = 1000

    # Security settings
    sanitize_sensitive: bool = True
    audit_enabled: bool = True
    security_logging: bool = True

    # Integration settings
    correlation_enabled: bool = True
    context_propagation: bool = True
    performance_tracking: bool = True

    # Advanced features
    structured_logging: bool = True
    json_serialization: bool = True
    compression_enabled: bool = False
    encryption_enabled: bool = False


class LogSanitizer:
    """Sanitizes sensitive data from log messages."""

    def __init__(self):
        self.sensitive_patterns = [
            r"\bpassword\b.*?[:\s=]\s*\S+",
            r"\btoken\b.*?[:\s=]\s*\S+",
            r"\bapi[_-]?key\b.*?[:\s=]\s*\S+",
            r"\bsecret\b.*?[:\s=]\s*\S+",
            r"\bcredit[_-]?card\b.*?[:\s=]\s*\S+",
            r"\bssn\b.*?[:\s=]\s*\S+",
            r"\bemail\b.*?[:\s=]\s*\S+@\S+",
            r"\bphone\b.*?[:\s=]\s*[\d\-\(\)\+\s]+",
        ]

        self.replacement_text = "[REDACTED]"

    def sanitize(self, message: str, extra: Dict[str, Any] = None) -> tuple[str, Dict[str, Any]]:
        """Sanitize sensitive information from log messages and extra data."""
        import re

        # Sanitize message
        sanitized_message = message
        for pattern in self.sensitive_patterns:
            sanitized_message = re.sub(
                pattern, self.replacement_text, sanitized_message, flags=re.IGNORECASE
            )

        # Sanitize extra data
        sanitized_extra = {}
        if extra:
            for key, value in extra.items():
                if self._is_sensitive_field(key):
                    sanitized_extra[key] = self.replacement_text
                elif isinstance(value, str):
                    for pattern in self.sensitive_patterns:
                        value = re.sub(pattern, self.replacement_text, value, flags=re.IGNORECASE)
                    sanitized_extra[key] = value
                else:
                    sanitized_extra[key] = value

        return sanitized_message, sanitized_extra

    def _is_sensitive_field(self, field_name: str) -> bool:
        """Check if a field name indicates sensitive data."""
        sensitive_fields = {
            "password",
            "passwd",
            "pwd",
            "secret",
            "token",
            "key",
            "api_key",
            "access_token",
            "refresh_token",
            "auth_token",
            "session_token",
            "credit_card",
            "card_number",
            "cvv",
            "ssn",
            "social_security",
            "bank_account",
            "routing_number",
            "pin",
            "passcode",
        }

        field_lower = field_name.lower()
        return any(sensitive in field_lower for sensitive in sensitive_fields)


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured logging."""

    def __init__(self, config: LoggingConfig, include_context: bool = True):
        super().__init__()
        self.config = config
        self.include_context = include_context
        self.sanitizer = LogSanitizer() if config.sanitize_sensitive else None
        self.hostname = os.uname().nodename if hasattr(os, "uname") else "unknown"

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured data."""
        # Get context from record
        context = getattr(record, "context", None) or LogContext()

        # Base log entry
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "thread": record.thread,
            "thread_name": record.threadName,
            "process": record.process,
            "hostname": self.hostname,
        }

        # Add context information
        if self.include_context and context:
            log_entry.update(
                {
                    "correlation_id": context.correlation_id,
                    "session_id": context.session_id,
                    "user_id": context.user_id,
                    "component": context.component,
                    "workflow_id": context.workflow_id,
                    "interaction_id": context.interaction_id,
                    "request_id": context.request_id,
                    "operation": context.operation,
                    "category": context.category,
                    "service_name": context.service_name,
                    "security_level": context.security_level,
                    "tags": list(context.tags) if context.tags else [],
                }
            )

            # Add performance metrics if available
            if context.execution_time is not None:
                log_entry["execution_time"] = context.execution_time
            if context.memory_usage is not None:
                log_entry["memory_usage"] = context.memory_usage
            if context.cpu_usage is not None:
                log_entry["cpu_usage"] = context.cpu_usage

            # Add custom fields
            if context.custom_fields:
                log_entry["custom"] = context.custom_fields

        # Add exception information
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info),
            }

        # Add extra fields from record
        for key, value in record.__dict__.items():
            if key not in {
                "name",
                "msg",
                "args",
                "levelname",
                "levelno",
                "pathname",
                "filename",
                "module",
                "lineno",
                "funcName",
                "created",
                "msecs",
                "relativeCreated",
                "thread",
                "threadName",
                "processName",
                "process",
                "exc_info",
                "exc_text",
                "stack_info",
                "context",
            }:
                log_entry["extra"] = log_entry.get("extra", {})
                log_entry["extra"][key] = value

        # Sanitize sensitive data
        if self.sanitizer:
            log_entry["message"], extra = self.sanitizer.sanitize(
                log_entry["message"], log_entry.get("extra", {})
            )
            if extra:
                log_entry["extra"] = extra

        # Format based on configuration
        if self.config.format_type == LogFormat.JSON:
            return json.dumps(log_entry, default=str, ensure_ascii=False)
        elif self.config.format_type == LogFormat.CONSOLE:
            return self._format_console(log_entry)
        elif self.config.format_type == LogFormat.COMPACT:
            return self._format_compact(log_entry)
        else:
            return self._format_structured(log_entry)

    def _format_console(self, log_entry: Dict[str, Any]) -> str:
        """Format for console output with colors."""
        if not COLORAMA_AVAILABLE:
            return self._format_plain(log_entry)

        level = log_entry["level"]
        timestamp = log_entry["timestamp"][:19]  # Remove microseconds

        # Color mapping
        level_colors = {
            "TRACE": Fore.BLUE,
            "DEBUG": Fore.CYAN,
            "INFO": Fore.GREEN,
            "WARNING": Fore.YELLOW,
            "ERROR": Fore.RED,
            "CRITICAL": Fore.MAGENTA + Style.BRIGHT,
            "SECURITY": Fore.RED + Back.YELLOW,
            "AUDIT": Fore.WHITE + Back.BLUE,
        }

        color = level_colors.get(level, Fore.WHITE)

        # Build formatted message
        parts = [
            f"{Fore.WHITE}[{timestamp}]",
            f"{color}[{level:8}]",
            f"{Fore.BLUE}[{log_entry.get('component', log_entry['logger'])}]",
        ]

        if log_entry.get("correlation_id"):
            parts.append(f"{Fore.MAGENTA}[{log_entry['correlation_id'][:8]}]")

        parts.append(f"{Style.RESET_ALL}{log_entry['message']}")

        # Add exception info
        if "exception" in log_entry:
            parts.append(
                f"\n{Fore.RED}Exception: {log_entry['exception']['message']}{Style.RESET_ALL}"
            )

        return " ".join(parts)

    def _format_plain(self, log_entry: Dict[str, Any]) -> str:
        """Format for plain text output."""
        timestamp = log_entry["timestamp"][:19]
        level = log_entry["level"]
        component = log_entry.get("component", log_entry["logger"])
        message = log_entry["message"]

        base = f"[{timestamp}] [{level:8}] [{component}] {message}"

        if "exception" in log_entry:
            base += f"\nException: {log_entry['exception']['message']}"

        return base

    def _format_compact(self, log_entry: Dict[str, Any]) -> str:
        """Format for compact output."""
        timestamp = log_entry["timestamp"][11:19]  # Time only
        level = log_entry["level"][0]  # First letter only
        message = log_entry["message"][:100]  # Truncate message

        return f"{timestamp} {level} {message}"

    def _format_structured(self, log_entry: Dict[str, Any]) -> str:
        """Format for structured text output."""
        lines = [
            f"Time: {log_entry['timestamp']}",
            f"Level: {log_entry['level']}",
            f"Component: {log_entry.get('component', log_entry['logger'])}",
            f"Message: {log_entry['message']}",
        ]

        if log_entry.get("correlation_id"):
            lines.append(f"Correlation: {log_entry['correlation_id']}")

        if log_entry.get("operation"):
            lines.append(f"Operation: {log_entry['operation']}")

        if "execution_time" in log_entry:
            lines.append(f"Execution Time: {log_entry['execution_time']:.3f}s")

        if "exception" in log_entry:
            lines.append(f"Exception: {log_entry['exception']['message']}")

        return " | ".join(lines)


class AsyncLogHandler(logging.Handler):
    """Asynchronous log handler for high-performance logging."""

    def __init__(self, target_handler: logging.Handler, buffer_size: int = 1000):
        super().__init__()
        self.target_handler = target_handler
        self.buffer_size = buffer_size
        self.queue = queue.Queue(maxsize=buffer_size)
        self.worker_thread = None
        self.shutdown_event = threading.Event()
        self.start_worker()

    def start_worker(self):
        """Start the background worker thread."""
        self.worker_thread = threading.Thread(
            target=self._worker_loop, name="AsyncLogHandler", daemon=True
        )
        self.worker_thread.start()

    def _worker_loop(self):
        """Background worker loop for processing log records."""
        while not self.shutdown_event.is_set():
            try:
                # Wait for records with timeout
                try:
                    record = self.queue.get(timeout=1.0)
                    if record is None:  # Sentinel value for shutdown
                        break

                    # Process the record
                    self.target_handler.emit(record)
                    self.queue.task_done()

                except queue.Empty:
                    continue

            except Exception as e:
                # Log worker errors to stderr
                print(f"AsyncLogHandler worker error: {e}", file=sys.stderr)

    def emit(self, record: logging.LogRecord):
        """Emit a log record asynchronously."""
        try:
            if not self.shutdown_event.is_set():
                self.queue.put_nowait(record)
        except queue.Full:
            # Queue is full, drop the record or handle overflow
            if hasattr(self, "_overflow_count"):
                self._overflow_count += 1
            else:
                self._overflow_count = 1
                print(f"AsyncLogHandler queue overflow, dropping records", file=sys.stderr)

    def close(self):
        """Close the handler and cleanup resources."""
        self.shutdown_event.set()

        # Send sentinel value
        try:
            self.queue.put_nowait(None)
        except queue.Full:
            pass

        # Wait for worker to finish
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5.0)

        # Close target handler
        self.target_handler.close()
        super().close()


class ContextualLogger:
    """Logger with automatic context propagation."""

    def __init__(self, logger: logging.Logger, context: Optional[LogContext] = None):
        self.logger = logger
        self.context = context or LogContext()
        self._context_stack = []

    def with_context(self, **kwargs) -> "ContextualLogger":
        """Create a new logger with additional context."""
        new_context = LogContext(**{**asdict(self.context), **kwargs})
        return ContextualLogger(self.logger, new_context)

    def push_context(self, **kwargs) -> None:
        """Push additional context onto the stack."""
        self._context_stack.append(asdict(self.context))
        for key, value in kwargs.items():
            if hasattr(self.context, key):
                setattr(self.context, key, value)
            else:
                self.context.custom_fields[key] = value

    def pop_context(self) -> None:
        """Pop context from the stack."""
        if self._context_stack:
            old_context = self._context_stack.pop()
            self.context = LogContext(**old_context)

    def _log(self, level: int, message: str, *args, **kwargs):
        """Internal logging method with context."""
        extra = kwargs.get("extra", {})
        extra["context"] = self.context
        kwargs["extra"] = extra

        self.logger.log(level, message, *args, **kwargs)

    def trace(self, message: str, *args, **kwargs):
        """Log at TRACE level."""
        self._log(LogLevel.TRACE.value, message, *args, **kwargs)

    def debug(self, message: str, *args, **kwargs):
        """Log at DEBUG level."""
        self._log(LogLevel.DEBUG.value, message, *args, **kwargs)

    def info(self, message: str, *args, **kwargs):
        """Log at INFO level."""
        self._log(LogLevel.INFO.value, message, *args, **kwargs)

    def warning(self, message: str, *args, **kwargs):
        """Log at WARNING level."""
        self._log(LogLevel.WARNING.value, message, *args, **kwargs)

    def error(self, message: str, *args, **kwargs):
        """Log at ERROR level."""
        self._log(LogLevel.ERROR.value, message, *args, **kwargs)

    def critical(self, message: str, *args, **kwargs):
        """Log at CRITICAL level."""
        self._log(LogLevel.CRITICAL.value, message, *args, **kwargs)

    def security(self, message: str, *args, **kwargs):
        """Log security events."""
        self.context.security_level = "high"
        self.context.category = "security"
        self._log(LogLevel.SECURITY.value, message, *args, **kwargs)

    def audit(self, message: str, *args, **kwargs):
        """Log audit events."""
        self.context.category = "audit"
        self._log(LogLevel.AUDIT.value, message, *args, **kwargs)

    def exception(self, message: str, *args, **kwargs):
        """Log an exception."""
        kwargs["exc_info"] = True
        self.error(message, *args, **kwargs)


class PerformanceLogDecorator:
    """Decorator for automatic performance logging."""

    def __init__(
        self,
        logger: ContextualLogger,
        log_level: LogLevel = LogLevel.DEBUG,
        include_args: bool = False,
        include_result: bool = False,
    ):
        self.logger = logger
        self.log_level = log_level
        self.include_args = include_args
        self.include_result = include_result

    def __call__(self, func):
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            function_name = f"{func.__module__}.{func.__qualname__}"

            # Log function entry
            log_data = {"operation": function_name}
            if self.include_args:
                log_data["args"] = args
                log_data["kwargs"] = kwargs

            self.logger.with_context(**log_data).debug(f"Entering {function_name}")

            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time

                # Log successful completion
                completion_data = {
                    "operation": function_name,
                    "execution_time": execution_time,
                    "success": True,
                }
                if self.include_result:
                    completion_data["result"] = result

                self.logger.with_context(**completion_data).debug(
                    f"Completed {function_name} in {execution_time:.3f}s"
                )

                return result

            except Exception as e:
                execution_time = time.time() - start_time

                # Log exception
                error_data = {
                    "operation": function_name,
                    "execution_time": execution_time,
                    "success": False,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                }

                self.logger.with_context(**error_data).exception(
                    f"Failed {function_name} after {execution_time:.3f}s"
                )

                raise

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            function_name = f"{func.__module__}.{func.__qualname__}"

            # Log function entry
            log_data = {"operation": function_name}
            if self.include_args:
                log_data["args"] = args
                log_data["kwargs"] = kwargs

            self.logger.with_context(**log_data).debug(f"Entering {function_name}")

            try:
                result = await func(*args, **kwargs)
                execution_time = time.time() - start_time

                # Log successful completion
                completion_data = {
                    "operation": function_name,
                    "execution_time": execution_time,
                    "success": True,
                }
                if self.include_result:
                    completion_data["result"] = result

                self.logger.with_context(**completion_data).debug(
                    f"Completed {function_name} in {execution_time:.3f}s"
                )

                return result

            except Exception as e:
                execution_time = time.time() - start_time

                # Log exception
                error_data = {
                    "operation": function_name,
                    "execution_time": execution_time,
                    "success": False,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                }

                self.logger.with_context(**error_data).exception(
                    f"Failed {function_name} after {execution_time:.3f}s"
                )

                raise

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper


class LoggingManager:
    """Central logging management system."""

    def __init__(self, config: Optional[LoggingConfig] = None):
        self.config = config or LoggingConfig()
        self.loggers: Dict[str, ContextualLogger] = {}
        self.handlers: List[logging.Handler] = []
        self.is_configured = False
        self.sanitizer = LogSanitizer()
        self._setup_custom_levels()
        self._rate_limiter = {}
        self._last_rate_limit_reset = time.time()

    def _setup_custom_levels(self):
        """Setup custom log levels."""
        logging.addLevelName(LogLevel.TRACE.value, "TRACE")
        logging.addLevelName(LogLevel.SECURITY.value, "SECURITY")
        logging.addLevelName(LogLevel.AUDIT.value, "AUDIT")

    def configure(self, config: Optional[LoggingConfig] = None) -> None:
        """Configure the logging system."""
        if config:
            self.config = config

        # Clear existing handlers
        self._clear_handlers()

        # Setup root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(self.config.level.value)

        # Create directories
        if self.config.file_enabled:
            self.config.log_dir.mkdir(parents=True, exist_ok=True)

        # Setup handlers
        if self.config.console_enabled:
            self._setup_console_handler()

        if self.config.file_enabled:
            self._setup_file_handlers()

        if self.config.remote_enabled:
            self._setup_remote_handlers()

        self.is_configured = True

    def _clear_handlers(self):
        """Clear existing handlers."""
        for handler in self.handlers:
            if hasattr(handler, "close"):
                handler.close()
        self.handlers.clear()

        # Clear root logger handlers
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

    def _setup_console_handler(self):
        """Setup console output handler."""
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.config.level.value)

        # Use appropriate formatter
        if self.config.format_type == LogFormat.JSON:
            formatter = StructuredFormatter(self.config)
        else:
            formatter = StructuredFormatter(self.config)

        console_handler.setFormatter(formatter)

        # Add rate limiting filter
        console_handler.addFilter(self._create_rate_limit_filter())

        # Wrap in async handler if enabled
        if self.config.async_logging:
            console_handler = AsyncLogHandler(console_handler, self.config.buffer_size)

        self.handlers.append(console_handler)
        logging.getLogger().addHandler(console_handler)

    def _setup_file_handlers(self):
        """Setup file output handlers."""
        log_dir = self.config.log_dir

        # Application log
        app_handler = logging.handlers.RotatingFileHandler(
            filename=log_dir / "application.log",
            maxBytes=self.config.max_file_size,
            backupCount=self.config.backup_count,
            encoding="utf-8",
        )
        app_handler.setLevel(self.config.level.value)
        app_handler.setFormatter(StructuredFormatter(self.config))

        # Error log
        error_handler = logging.handlers.RotatingFileHandler(
            filename=log_dir / "error.log",
            maxBytes=self.config.max_file_size,
            backupCount=self.config.backup_count,
            encoding="utf-8",
        )
        error_handler.setLevel(LogLevel.ERROR.value)
        error_handler.setFormatter(StructuredFormatter(self.config))

        # Security log
        if self.config.security_logging:
            security_handler = logging.handlers.RotatingFileHandler(
                filename=log_dir / "security.log",
                maxBytes=self.config.max_file_size,
                backupCount=self.config.backup_count,
                encoding="utf-8",
            )
            security_handler.setLevel(LogLevel.SECURITY.value)
            security_handler.setFormatter(StructuredFormatter(self.config))
            security_handler.addFilter(self._create_security_filter())

            self.handlers.append(security_handler)
            logging.getLogger().addHandler(security_handler)

        # Audit log
        if self.config.audit_enabled:
            audit_handler = logging.handlers.RotatingFileHandler(
                filename=log_dir / "audit.log",
                maxBytes=self.config.max_file_size,
                backupCount=self.config.backup_count,
                encoding="utf-8",
            )
            audit_handler.setLevel(LogLevel.AUDIT.value)
            audit_handler.setFormatter(StructuredFormatter(self.config))
            audit_handler.addFilter(self._create_audit_filter())

            self.handlers.append(audit_handler)
            logging.getLogger().addHandler(audit_handler)

        # Wrap in async handlers if enabled
        if self.config.async_logging:
            app_handler = AsyncLogHandler(app_handler, self.config.buffer_size)
            error_handler = AsyncLogHandler(error_handler, self.config.buffer_size)

        self.handlers.extend([app_handler, error_handler])
        logging.getLogger().addHandler(app_handler)
        logging.getLogger().addHandler(error_handler)

    def _setup_remote_handlers(self):
        """Setup remote logging handlers."""
        # Syslog handler
        try:
            syslog_handler = logging.handlers.SysLogHandler(address="/dev/log")
            syslog_handler.setLevel(self.config.level.value)
            syslog_handler.setFormatter(StructuredFormatter(self.config))

            self.handlers.append(syslog_handler)
            logging.getLogger().addHandler(syslog_handler)
        except Exception as e:
            print(f"Failed to setup syslog handler: {e}", file=sys.stderr)

    def _create_rate_limit_filter(self):
        """Create a rate limiting filter."""

        def rate_limit_filter(record):
            current_time = time.time()

            # Reset rate limiter every second
            if current_time - self._last_rate_limit_reset >= 1.0:
                self._rate_limiter.clear()
                self._last_rate_limit_reset = current_time

            # Check rate limit
            logger_name = record.name
            current_count = self._rate_limiter.get(logger_name, 0)

            if current_count >= self.config.rate_limit_per_second:
                return False

            self._rate_limiter[logger_name] = current_count + 1
            return True

        return rate_limit_filter

    def _create_security_filter(self):
        """Create a filter for security events."""

        def security_filter(record):
            return record.levelno >= LogLevel.SECURITY.value

        return security_filter

    def _create_audit_filter(self):
        """Create a filter for audit events."""

        def audit_filter(record):
            return record.levelno >= LogLevel.AUDIT.value

        return audit_filter

    def get_logger(self, name: str, component: Optional[str] = None) -> ContextualLogger:
        """Get a contextual logger for a component."""
        if not self.is_configured:
            self.configure()

        if name in self.loggers:
            return self.loggers[name]

        # Create base logger
        base_logger = logging.getLogger(name)

        # Set component-specific level if configured
        if component and component in self.config.component_levels:
            base_logger.setLevel(self.config.component_levels[component].value)

        # Create context
        context = LogContext(component=component or name)

        # Create contextual logger
        contextual_logger = ContextualLogger(base_logger, context)
        self.loggers[name] = contextual_logger

        return contextual_logger

    def set_level(self, level: Union[LogLevel, str], component: Optional[str] = None):
        """Set log level globally or for a specific component."""
        if isinstance(level, str):
            level = LogLevel[level.upper()]

        if component:
            self.config.component_levels[component] = level
            # Update existing loggers
            for logger_name, logger in self.loggers.items():
                if logger.context.component == component:
                    logger.logger.setLevel(level.value)
        else:
            self.config.level = level
            logging.getLogger().setLevel(level.value)

    def flush(self):
        """Flush all handlers."""
        for handler in self.handlers:
            if hasattr(handler, "flush"):
                handler.flush()

    def shutdown(self):
        """Shutdown the logging system."""
        self.flush()

        for handler in self.handlers:
            if hasattr(handler, "close"):
                handler.close()

        self.handlers.clear()
        self.loggers.clear()


# Global logging manager instance
_logging_manager = None
_context_var: ContextVar[LogContext] = ContextVar("log_context", default=LogContext())


def configure_logging(config: Optional[LoggingConfig] = None) -> None:
    """Configure the global logging system."""
    global _logging_manager

    if _logging_manager is None:
        _logging_manager = LoggingManager()

    _logging_manager.configure(config)


def get_logger(name: Optional[str] = None, component: Optional[str] = None) -> ContextualLogger:
    """Get a logger instance."""
    global _logging_manager

    if _logging_manager is None:
        configure_logging()

    # Use caller's module name if not specified
    if name is None:
        frame = inspect.currentframe().f_back
        name = frame.f_globals.get("__name__", "unknown")

    return _logging_manager.get_logger(name, component)


def set_log_level(level: Union[LogLevel, str], component: Optional[str] = None) -> None:
    """Set log level."""
    global _logging_manager

    if _logging_manager is None:
        configure_logging()

    _logging_manager.set_level(level, component)


def with_log_context(**kwargs) -> ContextualLogger:
    """Create a logger with specific context."""
    logger = get_logger()
    return logger.with_context(**kwargs)


def log_performance(
    logger: Optional[ContextualLogger] = None,
    level: LogLevel = LogLevel.DEBUG,
    include_args: bool = False,
    include_result: bool = False,
):
    """Decorator for performance logging."""
    if logger is None:
        logger = get_logger()

    return PerformanceLogDecorator(logger, level, include_args, include_result)


def log_context(**kwargs):
    """Context manager for temporary log context."""
    logger = get_logger()

    class LogContextManager:
        def __enter__(self):
            logger.push_context(**kwargs)
            return logger

        def __exit__(self, exc_type, exc_val, exc_tb):
            logger.pop_context()

    return LogContextManager()


def create_default_config() -> LoggingConfig:
    """Create default logging configuration."""
    return LoggingConfig(
        level=LogLevel.INFO,
        format_type=LogFormat.STRUCTURED,
        console_enabled=True,
        file_enabled=True,
        async_logging=True,
        correlation_enabled=True,
        sanitize_sensitive=True,
        audit_enabled=True,
        security_logging=True,
    )


def create_development_config() -> LoggingConfig:
    """Create development logging configuration."""
    return LoggingConfig(
        level=LogLevel.DEBUG,
        format_type=LogFormat.CONSOLE,
        console_enabled=True,
        file_enabled=False,
        async_logging=False,
        correlation_enabled=True,
        sanitize_sensitive=False,
        audit_enabled=False,
        security_logging=False,
    )


def create_production_config() -> LoggingConfig:
    """Create production logging configuration."""
    return LoggingConfig(
        level=LogLevel.INFO,
        format_type=LogFormat.JSON,
        console_enabled=False,
        file_enabled=True,
        remote_enabled=True,
        async_logging=True,
        correlation_enabled=True,
        sanitize_sensitive=True,
        audit_enabled=True,
        security_logging=True,
        compression_enabled=True,
        rate_limit_per_second=500,
    )


# Initialize with default configuration
configure_logging()


# Export public interface
__all__ = [
    "LogLevel",
    "LogFormat",
    "LogChannel",
    "LogContext",
    "LoggingConfig",
    "ContextualLogger",
    "configure_logging",
    "get_logger",
    "set_log_level",
    "with_log_context",
    "log_performance",
    "log_context",
    "create_default_config",
    "create_development_config",
    "create_production_config",
]
