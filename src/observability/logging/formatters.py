"""
Advanced Logging Formatters for AI Assistant
Author: Drmusab
Last Modified: 2025-06-26 11:07:40 UTC

This module provides comprehensive logging formatters for the AI assistant system,
supporting structured logging, contextual information, performance metrics,
security considerations, and integration with all core system components.
"""

import json
import traceback
import sys
import threading
import time
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Union, Callable
from pathlib import Path
import logging
import uuid
import hashlib
import inspect
from dataclasses import dataclass, field, asdict
from enum import Enum
from contextvars import ContextVar
import re

# Core imports
from src.core.config.loader import ConfigLoader
from src.core.security.sanitization import SecuritySanitizer

# Type definitions
LogLevel = Union[int, str]
LogRecord = logging.LogRecord


class LogFormat(Enum):
    """Supported log output formats."""
    JSON = "json"
    STRUCTURED = "structured"
    PLAIN = "plain"
    CONSOLE = "console"
    AUDIT = "audit"
    PERFORMANCE = "performance"
    SECURITY = "security"
    DEBUG = "debug"
    COMPACT = "compact"
    ELK = "elk"
    PROMETHEUS = "prometheus"


class SensitivityLevel(Enum):
    """Data sensitivity levels for redaction."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    SECRET = "secret"


class ComponentType(Enum):
    """Component types for structured logging."""
    CORE_ENGINE = "core_engine"
    COMPONENT_MANAGER = "component_manager"
    WORKFLOW_ORCHESTRATOR = "workflow_orchestrator"
    INTERACTION_HANDLER = "interaction_handler"
    SESSION_MANAGER = "session_manager"
    PLUGIN_MANAGER = "plugin_manager"
    MEMORY_SYSTEM = "memory_system"
    LEARNING_SYSTEM = "learning_system"
    PROCESSING_COMPONENT = "processing_component"
    INTEGRATION = "integration"
    API = "api"
    SKILL = "skill"
    SECURITY = "security"
    UNKNOWN = "unknown"


# Context variables for correlation
correlation_id: ContextVar[Optional[str]] = ContextVar('correlation_id', default=None)
session_id: ContextVar[Optional[str]] = ContextVar('session_id', default=None)
user_id: ContextVar[Optional[str]] = ContextVar('user_id', default=None)
component_type: ContextVar[Optional[ComponentType]] = ContextVar('component_type', default=None)
trace_id: ContextVar[Optional[str]] = ContextVar('trace_id', default=None)
span_id: ContextVar[Optional[str]] = ContextVar('span_id', default=None)


@dataclass
class LogContext:
    """Comprehensive logging context information."""
    # Request/Session context
    correlation_id: Optional[str] = None
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    interaction_id: Optional[str] = None
    workflow_id: Optional[str] = None
    execution_id: Optional[str] = None
    
    # Component context
    component_type: Optional[ComponentType] = None
    component_name: Optional[str] = None
    component_version: Optional[str] = None
    
    # Tracing context
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    parent_span_id: Optional[str] = None
    
    # Performance context
    processing_time: Optional[float] = None
    memory_usage: Optional[float] = None
    cpu_usage: Optional[float] = None
    
    # Security context
    security_context: Dict[str, Any] = field(default_factory=dict)
    sensitivity_level: SensitivityLevel = SensitivityLevel.INTERNAL
    
    # Custom context
    custom_fields: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    
    # Metadata
    timestamp: Optional[datetime] = None
    thread_id: Optional[int] = None
    process_id: Optional[int] = None
    hostname: Optional[str] = None
    environment: Optional[str] = None


class SensitiveDataRedactor:
    """Redacts sensitive information from log messages and data."""
    
    def __init__(self, sanitizer: Optional[SecuritySanitizer] = None):
        self.sanitizer = sanitizer
        self._setup_patterns()
    
    def _setup_patterns(self) -> None:
        """Setup regex patterns for sensitive data detection."""
        self.patterns = {
            'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            'phone': re.compile(r'\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b'),
            'ssn': re.compile(r'\b\d{3}-?\d{2}-?\d{4}\b'),
            'credit_card': re.compile(r'\b(?:\d{4}[-.\s]?){3}\d{4}\b'),
            'api_key': re.compile(r'\b[A-Za-z0-9]{32,}\b'),
            'password': re.compile(r'password["\']?\s*[:=]\s*["\']?([^"\'\s]+)', re.IGNORECASE),
            'token': re.compile(r'token["\']?\s*[:=]\s*["\']?([^"\'\s]+)', re.IGNORECASE),
            'secret': re.compile(r'secret["\']?\s*[:=]\s*["\']?([^"\'\s]+)', re.IGNORECASE),
            'private_key': re.compile(r'-----BEGIN [A-Z ]+PRIVATE KEY-----.*?-----END [A-Z ]+PRIVATE KEY-----', re.DOTALL),
            'ip_address': re.compile(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'),
            'uuid': re.compile(r'\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b', re.IGNORECASE)
        }
        
        # Sensitive field names
        self.sensitive_fields = {
            'password', 'passwd', 'pwd', 'secret', 'token', 'api_key', 'apikey',
            'private_key', 'privatekey', 'auth_token', 'authorization', 'credential',
            'credentials', 'auth', 'session_key', 'sessionkey', 'access_token',
            'refresh_token', 'jwt', 'bearer', 'signature', 'hash', 'salt'
        }
    
    def redact_message(self, message: str, sensitivity: SensitivityLevel = SensitivityLevel.INTERNAL) -> str:
        """Redact sensitive information from log message."""
        if not message or sensitivity == SensitivityLevel.PUBLIC:
            return message
        
        redacted = message
        
        # Apply pattern-based redaction
        for pattern_name, pattern in self.patterns.items():
            if sensitivity in [SensitivityLevel.CONFIDENTIAL, SensitivityLevel.RESTRICTED, SensitivityLevel.SECRET]:
                # More aggressive redaction for higher sensitivity
                redacted = pattern.sub(f'[REDACTED_{pattern_name.upper()}]', redacted)
            elif sensitivity == SensitivityLevel.INTERNAL:
                # Partial redaction for internal logs
                if pattern_name in ['email', 'phone']:
                    redacted = pattern.sub(lambda m: self._partial_redact(m.group(0)), redacted)
                else:
                    redacted = pattern.sub(f'[REDACTED_{pattern_name.upper()}]', redacted)
        
        return redacted
    
    def redact_data(self, data: Dict[str, Any], sensitivity: SensitivityLevel = SensitivityLevel.INTERNAL) -> Dict[str, Any]:
        """Redact sensitive information from structured data."""
        if not data or sensitivity == SensitivityLevel.PUBLIC:
            return data
        
        def _redact_recursive(obj: Any, depth: int = 0) -> Any:
            if depth > 10:  # Prevent infinite recursion
                return "[MAX_DEPTH_REACHED]"
            
            if isinstance(obj, dict):
                redacted = {}
                for key, value in obj.items():
                    key_lower = key.lower()
                    
                    # Check if field name indicates sensitive data
                    if any(sensitive in key_lower for sensitive in self.sensitive_fields):
                        redacted[key] = "[REDACTED_SENSITIVE_FIELD]"
                    else:
                        redacted[key] = _redact_recursive(value, depth + 1)
                return redacted
            
            elif isinstance(obj, (list, tuple)):
                return type(obj)(_redact_recursive(item, depth + 1) for item in obj)
            
            elif isinstance(obj, str):
                return self.redact_message(obj, sensitivity)
            
            else:
                return obj
        
        return _redact_recursive(data)
    
    def _partial_redact(self, value: str) -> str:
        """Partially redact a value showing only first and last characters."""
        if len(value) <= 4:
            return "[REDACTED]"
        return f"{value[:2]}***{value[-2:]}"


class PerformanceMetrics:
    """Tracks and formats performance metrics for logging."""
    
    def __init__(self):
        self.start_times: Dict[str, float] = {}
        self.metrics: Dict[str, List[float]] = {}
    
    def start_timer(self, operation: str) -> None:
        """Start timing an operation."""
        self.start_times[operation] = time.perf_counter()
    
    def end_timer(self, operation: str) -> Optional[float]:
        """End timing an operation and return duration."""
        if operation in self.start_times:
            duration = time.perf_counter() - self.start_times[operation]
            if operation not in self.metrics:
                self.metrics[operation] = []
            self.metrics[operation].append(duration)
            del self.start_times[operation]
            return duration
        return None
    
    def get_stats(self, operation: str) -> Dict[str, float]:
        """Get statistics for an operation."""
        if operation not in self.metrics or not self.metrics[operation]:
            return {}
        
        values = self.metrics[operation]
        return {
            'count': len(values),
            'avg': sum(values) / len(values),
            'min': min(values),
            'max': max(values),
            'total': sum(values)
        }
    
    def reset_metrics(self) -> None:
        """Reset all metrics."""
        self.start_times.clear()
        self.metrics.clear()


class BaseFormatter(logging.Formatter):
    """Base formatter with common functionality."""
    
    def __init__(self, 
                 fmt: Optional[str] = None,
                 datefmt: Optional[str] = None,
                 style: str = '%',
                 validate: bool = True,
                 config: Optional[ConfigLoader] = None,
                 sanitizer: Optional[SecuritySanitizer] = None):
        super().__init__(fmt, datefmt, style, validate)
        self.config = config
        self.redactor = SensitiveDataRedactor(sanitizer)
        self.performance_metrics = PerformanceMetrics()
        self._setup_configuration()
    
    def _setup_configuration(self) -> None:
        """Setup formatter configuration."""
        if self.config:
            self.include_stack_trace = self.config.get("logging.include_stack_trace", True)
            self.max_message_length = self.config.get("logging.max_message_length", 10000)
            self.default_sensitivity = SensitivityLevel(
                self.config.get("logging.default_sensitivity", "internal")
            )
            self.enable_performance_logging = self.config.get("logging.enable_performance", True)
            self.redaction_enabled = self.config.get("logging.enable_redaction", True)
        else:
            self.include_stack_trace = True
            self.max_message_length = 10000
            self.default_sensitivity = SensitivityLevel.INTERNAL
            self.enable_performance_logging = True
            self.redaction_enabled = True
    
    def get_log_context(self, record: LogRecord) -> LogContext:
        """Extract comprehensive logging context from record and context vars."""
        context = LogContext()
        
        # Extract from context variables
        context.correlation_id = correlation_id.get()
        context.session_id = session_id.get()
        context.user_id = user_id.get()
        context.component_type = component_type.get()
        context.trace_id = trace_id.get()
        context.span_id = span_id.get()
        
        # Extract from log record
        context.timestamp = datetime.fromtimestamp(record.created, timezone.utc)
        context.thread_id = record.thread
        context.process_id = record.process
        
        # Extract custom attributes from record
        for attr_name in dir(record):
            if attr_name.startswith('ctx_'):
                key = attr_name[4:]  # Remove 'ctx_' prefix
                setattr(context, key, getattr(record, attr_name))
            elif attr_name.startswith('custom_'):
                key = attr_name[7:]  # Remove 'custom_' prefix
                context.custom_fields[key] = getattr(record, attr_name)
        
        # Component detection from logger name
        if not context.component_type:
            context.component_type = self._detect_component_type(record.name)
        
        # Extract component name from module
        if record.module:
            context.component_name = record.module
        
        return context
    
    def _detect_component_type(self, logger_name: str) -> ComponentType:
        """Detect component type from logger name."""
        name_lower = logger_name.lower()
        
        if 'core_engine' in name_lower:
            return ComponentType.CORE_ENGINE
        elif 'component_manager' in name_lower:
            return ComponentType.COMPONENT_MANAGER
        elif 'workflow' in name_lower:
            return ComponentType.WORKFLOW_ORCHESTRATOR
        elif 'interaction' in name_lower:
            return ComponentType.INTERACTION_HANDLER
        elif 'session' in name_lower:
            return ComponentType.SESSION_MANAGER
        elif 'plugin' in name_lower:
            return ComponentType.PLUGIN_MANAGER
        elif 'memory' in name_lower:
            return ComponentType.MEMORY_SYSTEM
        elif 'learning' in name_lower:
            return ComponentType.LEARNING_SYSTEM
        elif 'processing' in name_lower:
            return ComponentType.PROCESSING_COMPONENT
        elif 'integration' in name_lower:
            return ComponentType.INTEGRATION
        elif 'api' in name_lower:
            return ComponentType.API
        elif 'skill' in name_lower:
            return ComponentType.SKILL
        elif 'security' in name_lower or 'auth' in name_lower:
            return ComponentType.SECURITY
        else:
            return ComponentType.UNKNOWN
    
    def format_exception(self, ei) -> str:
        """Format exception information with security considerations."""
        if not self.include_stack_trace:
            return "Exception occurred (stack trace disabled)"
        
        # Format exception with limited stack trace for security
        exc_type, exc_value, exc_traceback = ei
        
        # Limit stack trace depth
        max_depth = 20
        current_tb = exc_traceback
        depth = 0
        
        while current_tb and depth < max_depth:
            current_tb = current_tb.tb_next
            depth += 1
        
        # Format with redaction
        exc_text = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
        
        # Redact sensitive information from stack trace
        if self.redaction_enabled:
            exc_text = self.redactor.redact_message(exc_text, self.default_sensitivity)
        
        return exc_text
    
    def truncate_message(self, message: str) -> str:
        """Truncate message if it exceeds maximum length."""
        if len(message) <= self.max_message_length:
            return message
        
        truncated = message[:self.max_message_length - 50]
        return f"{truncated}... [TRUNCATED - {len(message)} chars total]"


class JSONFormatter(BaseFormatter):
    """JSON formatter for structured logging."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.compact_output = kwargs.get('compact_output', False)
    
    def format(self, record: LogRecord) -> str:
        """Format log record as JSON."""
        context = self.get_log_context(record)
        
        # Build structured log entry
        log_entry = {
            # Core log information
            'timestamp': context.timestamp.isoformat() if context.timestamp else datetime.now(timezone.utc).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': self.truncate_message(record.getMessage()),
            
            # Location information
            'location': {
                'module': record.module,
                'filename': record.filename,
                'function': record.funcName,
                'line': record.lineno
            },
            
            # Process/Thread information
            'process': {
                'pid': context.process_id or record.process,
                'thread_id': context.thread_id or record.thread,
                'thread_name': record.threadName
            },
            
            # Context information
            'context': self._build_context_dict(context),
            
            # Component information
            'component': {
                'type': context.component_type.value if context.component_type else 'unknown',
                'name': context.component_name,
                'version': context.component_version
            }
        }
        
        # Add exception information if present
        if record.exc_info:
            log_entry['exception'] = {
                'type': record.exc_info[0].__name__ if record.exc_info[0] else None,
                'message': str(record.exc_info[1]) if record.exc_info[1] else None,
                'stack_trace': self.format_exception(record.exc_info)
            }
        
        # Add performance metrics if enabled
        if self.enable_performance_logging and hasattr(record, 'performance_data'):
            log_entry['performance'] = record.performance_data
        
        # Add custom fields
        if context.custom_fields:
            log_entry['custom'] = context.custom_fields
        
        # Add tags
        if context.tags:
            log_entry['tags'] = context.tags
        
        # Redact sensitive information
        if self.redaction_enabled:
            log_entry = self.redactor.redact_data(log_entry, context.sensitivity_level)
        
        # Serialize to JSON
        try:
            if self.compact_output:
                return json.dumps(log_entry, separators=(',', ':'), default=str)
            else:
                return json.dumps(log_entry, indent=2, default=str)
        except (TypeError, ValueError) as e:
            # Fallback if JSON serialization fails
            fallback_entry = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'level': 'ERROR',
                'logger': 'JSONFormatter',
                'message': f'Failed to serialize log entry: {str(e)}',
                'original_message': str(record.getMessage())
            }
            return json.dumps(fallback_entry, default=str)
    
    def _build_context_dict(self, context: LogContext) -> Dict[str, Any]:
        """Build context dictionary from LogContext."""
        context_dict = {}
        
        # Request/Session context
        if context.correlation_id:
            context_dict['correlation_id'] = context.correlation_id
        if context.session_id:
            context_dict['session_id'] = context.session_id
        if context.user_id:
            context_dict['user_id'] = context.user_id
        if context.interaction_id:
            context_dict['interaction_id'] = context.interaction_id
        if context.workflow_id:
            context_dict['workflow_id'] = context.workflow_id
        if context.execution_id:
            context_dict['execution_id'] = context.execution_id
        
        # Tracing context
        if context.trace_id:
            context_dict['trace_id'] = context.trace_id
        if context.span_id:
            context_dict['span_id'] = context.span_id
        if context.parent_span_id:
            context_dict['parent_span_id'] = context.parent_span_id
        
        # Performance context
        if context.processing_time:
            context_dict['processing_time'] = context.processing_time
        if context.memory_usage:
            context_dict['memory_usage'] = context.memory_usage
        if context.cpu_usage:
            context_dict['cpu_usage'] = context.cpu_usage
        
        # Security context
        if context.security_context:
            context_dict['security'] = context.security_context
        context_dict['sensitivity_level'] = context.sensitivity_level.value
        
        return context_dict


class StructuredFormatter(BaseFormatter):
    """Structured text formatter for human-readable logs."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.include_context = kwargs.get('include_context', True)
        self.color_coding = kwargs.get('color_coding', False)
        self._setup_colors()
    
    def _setup_colors(self) -> None:
        """Setup color codes for different log levels."""
        if self.color_coding:
            self.colors = {
                'DEBUG': '\033[36m',     # Cyan
                'INFO': '\033[32m',      # Green
                'WARNING': '\033[33m',   # Yellow
                'ERROR': '\033[31m',     # Red
                'CRITICAL': '\033[35m',  # Magenta
                'RESET': '\033[0m'       # Reset
            }
        else:
            self.colors = {}
    
    def format(self, record: LogRecord) -> str:
        """Format log record as structured text."""
        context = self.get_log_context(record)
        
        # Build log line components
        components = []
        
        # Timestamp
        timestamp = context.timestamp.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3] if context.timestamp else datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        components.append(f"[{timestamp}]")
        
        # Log level with color
        level = record.levelname
        if self.color_coding and level in self.colors:
            level = f"{self.colors[level]}{level}{self.colors['RESET']}"
        components.append(f"[{level:8}]")
        
        # Component information
        if context.component_type:
            components.append(f"[{context.component_type.value}]")
        
        # Logger name
        components.append(f"[{record.name}]")
        
        # Context information (compact)
        if self.include_context:
            context_parts = []
            if context.correlation_id:
                context_parts.append(f"corr:{context.correlation_id[:8]}")
            if context.session_id:
                context_parts.append(f"sess:{context.session_id[:8]}")
            if context.user_id:
                context_parts.append(f"user:{context.user_id[:8]}")
            if context.trace_id:
                context_parts.append(f"trace:{context.trace_id[:8]}")
            
            if context_parts:
                components.append(f"[{','.join(context_parts)}]")
        
        # Location information
        location = f"{record.module}:{record.funcName}:{record.lineno}"
        components.append(f"[{location}]")
        
        # Message
        message = self.truncate_message(record.getMessage())
        if self.redaction_enabled:
            message = self.redactor.redact_message(message, context.sensitivity_level)
        
        # Join components and add message
        log_line = ' '.join(components) + f" - {message}"
        
        # Add exception information if present
        if record.exc_info:
            exc_text = self.format_exception(record.exc_info)
            log_line += f"\n{exc_text}"
        
        # Add performance metrics if enabled and present
        if self.enable_performance_logging and hasattr(record, 'performance_data'):
            perf_data = record.performance_data
            perf_info = []
            for key, value in perf_data.items():
                if isinstance(value, float):
                    perf_info.append(f"{key}:{value:.3f}")
                else:
                    perf_info.append(f"{key}:{value}")
            if perf_info:
                log_line += f" | Performance: {', '.join(perf_info)}"
        
        # Add custom fields
        if context.custom_fields:
            custom_info = []
            for key, value in context.custom_fields.items():
                custom_info.append(f"{key}:{value}")
            if custom_info:
                log_line += f" | Custom: {', '.join(custom_info)}"
        
        return log_line


class ConsoleFormatter(StructuredFormatter):
    """Console formatter optimized for terminal output."""
    
    def __init__(self, **kwargs):
        kwargs.setdefault('color_coding', True)
        kwargs.setdefault('include_context', True)
        super().__init__(**kwargs)
        self.compact_mode = kwargs.get('compact_mode', False)
    
    def format(self, record: LogRecord) -> str:
        """Format log record for console output."""
        if self.compact_mode:
            return self._format_compact(record)
        else:
            return super().format(record)
    
    def _format_compact(self, record: LogRecord) -> str:
        """Format log record in compact mode for console."""
        context = self.get_log_context(record)
        
        # Compact timestamp
        timestamp = datetime.now().strftime('%H:%M:%S')
        
        # Level with color
        level = record.levelname[0]  # Just first letter
        if self.color_coding and record.levelname in self.colors:
            level = f"{self.colors[record.levelname]}{level}{self.colors['RESET']}"
        
        # Component type abbreviation
        comp_abbrev = ""
        if context.component_type:
            abbrevs = {
                ComponentType.CORE_ENGINE: "CE",
                ComponentType.COMPONENT_MANAGER: "CM",
                ComponentType.WORKFLOW_ORCHESTRATOR: "WO",
                ComponentType.INTERACTION_HANDLER: "IH",
                ComponentType.SESSION_MANAGER: "SM",
                ComponentType.PLUGIN_MANAGER: "PM",
                ComponentType.MEMORY_SYSTEM: "MS",
                ComponentType.LEARNING_SYSTEM: "LS",
                ComponentType.PROCESSING_COMPONENT: "PC",
                ComponentType.INTEGRATION: "IN",
                ComponentType.API: "AP",
                ComponentType.SKILL: "SK",
                ComponentType.SECURITY: "SC"
            }
            comp_abbrev = abbrevs.get(context.component_type, "UK")
        
        # Message
        message = self.truncate_message(record.getMessage())
        if self.redaction_enabled:
            message = self.redactor.redact_message(message, context.sensitivity_level)
        
        # Compact format
        return f"{timestamp} {level} {comp_abbrev:2} {message}"


class AuditFormatter(JSONFormatter):
    """Specialized formatter for audit logs with security focus."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.include_sensitive_fields = kwargs.get('include_sensitive_fields', False)
    
    def format(self, record: LogRecord) -> str:
        """Format audit log record with security considerations."""
        context = self.get_log_context(record)
        
        # Build audit log entry
        audit_entry = {
            # Audit metadata
            'audit_id': str(uuid.uuid4()),
            'timestamp': context.timestamp.isoformat() if context.timestamp else datetime.now(timezone.utc).isoformat(),
            'event_type': 'audit_log',
            
            # Security context
            'actor': {
                'user_id': context.user_id,
                'session_id': context.session_id,
                'correlation_id': context.correlation_id
            },
            
            # Action information
            'action': {
                'type': record.levelname,
                'description': record.getMessage(),
                'component': context.component_type.value if context.component_type else 'unknown',
                'location': f"{record.module}:{record.funcName}:{record.lineno}"
            },
            
            # Resources
            'resources': getattr(record, 'audit_resources', []),
            
            # Result
            'result': {
                'success': record.levelname not in ['ERROR', 'CRITICAL'],
                'message': record.getMessage()
            },
            
            # Security classification
            'classification': {
                'sensitivity': context.sensitivity_level.value,
                'categories': getattr(record, 'audit_categories', [])
            },
            
            # Technical details
            'technical': {
                'logger': record.name,
                'process_id': context.process_id,
                'thread_id': context.thread_id,
                'hostname': context.hostname
            }
        }
        
        # Add exception details for failed operations
        if record.exc_info:
            audit_entry['error'] = {
                'type': record.exc_info[0].__name__ if record.exc_info[0] else None,
                'message': str(record.exc_info[1]) if record.exc_info[1] else None
            }
            
            # Include stack trace only if configured
            if self.include_sensitive_fields and self.include_stack_trace:
                audit_entry['error']['stack_trace'] = self.format_exception(record.exc_info)
        
        # Add custom audit fields
        if hasattr(record, 'audit_data') and isinstance(record.audit_data, dict):
            audit_entry['custom'] = record.audit_data
        
        # Conditional redaction for audit logs
        if not self.include_sensitive_fields:
            audit_entry = self.redactor.redact_data(audit_entry, SensitivityLevel.CONFIDENTIAL)
        
        return json.dumps(audit_entry, separators=(',', ':'), default=str)


class PerformanceFormatter(JSONFormatter):
    """Specialized formatter for performance logs."""
    
    def format(self, record: LogRecord) -> str:
        """Format performance log record."""
        context = self.get_log_context(record)
        
        # Build performance log entry
        perf_entry = {
            'timestamp': context.timestamp.isoformat() if context.timestamp else datetime.now(timezone.utc).isoformat(),
            'event_type': 'performance_metric',
            'metric_name': getattr(record, 'metric_name', 'unknown'),
            'metric_value': getattr(record, 'metric_value', 0),
            'metric_unit': getattr(record, 'metric_unit', 'count'),
            'component': {
                'type': context.component_type.value if context.component_type else 'unknown',
                'name': context.component_name
            },
            'context': {
                'session_id': context.session_id,
                'user_id': context.user_id,
                'correlation_id': context.correlation_id,
                'trace_id': context.trace_id
            },
            'system': {
                'hostname': context.hostname,
                'process_id': context.process_id,
                'thread_id': context.thread_id
            }
        }
        
        # Add performance details if available
        if hasattr(record, 'performance_data'):
            perf_entry['metrics'] = record.performance_data
        
        # Add operation details
        if hasattr(record, 'operation_name'):
            perf_entry['operation'] = {
                'name': record.operation_name,
                'duration': getattr(record, 'operation_duration', None),
                'status': getattr(record, 'operation_status', 'unknown')
            }
        
        # Add resource usage
        if context.memory_usage or context.cpu_usage:
            perf_entry['resources'] = {
                'memory_mb': context.memory_usage,
                'cpu_percent': context.cpu_usage
            }
        
        return json.dumps(perf_entry, separators=(',', ':'), default=str)


class ELKFormatter(JSONFormatter):
    """Formatter optimized for ELK Stack (Elasticsearch, Logstash, Kibana)."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.index_prefix = kwargs.get('index_prefix', 'ai-assistant')
        self.include_raw_message = kwargs.get('include_raw_message', True)
    
    def format(self, record: LogRecord) -> str:
        """Format log record for ELK Stack."""
        context = self.get_log_context(record)
        
        # Build ELK-optimized log entry
        elk_entry = {
            # ELK metadata
            '@timestamp': context.timestamp.isoformat() if context.timestamp else datetime.now(timezone.utc).isoformat(),
            '@version': '1',
            'index_pattern': f"{self.index_prefix}-{datetime.now().strftime('%Y.%m.%d')}",
            
            # Log level and source
            'level': record.levelname,
            'level_value': record.levelno,
            'logger_name': record.name,
            'logger_module': record.module,
            
            # Message content
            'message': self.truncate_message(record.getMessage()),
            'message_template': getattr(record, 'msg', ''),
            
            # Location
            'source': {
                'file': record.filename,
                'function': record.funcName,
                'line': record.lineno,
                'path': record.pathname
            },
            
            # Process information
            'process': {
                'pid': context.process_id or record.process,
                'thread': {
                    'id': context.thread_id or record.thread,
                    'name': record.threadName
                }
            },
            
            # Context for filtering and correlation
            'context': {
                'correlation_id': context.correlation_id,
                'session_id': context.session_id,
                'user_id': context.user_id,
                'interaction_id': context.interaction_id,
                'workflow_id': context.workflow_id,
                'execution_id': context.execution_id
            },
            
            # Component information for filtering
            'component': {
                'type': context.component_type.value if context.component_type else 'unknown',
                'name': context.component_name,
                'version': context.component_version
            },
            
            # Tracing
            'trace': {
                'trace_id': context.trace_id,
                'span_id': context.span_id,
                'parent_span_id': context.parent_span_id
            },
            
            # Performance metrics
            'performance': {
                'processing_time': context.processing_time,
                'memory_usage': context.memory_usage,
                'cpu_usage': context.cpu_usage
            },
            
            # Security and classification
            'security': {
                'sensitivity_level': context.sensitivity_level.value,
                'classification': getattr(record, 'security_classification', 'internal')
            },
            
            # Environment
            'environment': {
                'hostname': context.hostname,
                'env': context.environment
            }
        }
        
        # Add exception information
        if record.exc_info:
            elk_entry['exception'] = {
                'class': record.exc_info[0].__name__ if record.exc_info[0] else None,
                'message': str(record.exc_info[1]) if record.exc_info[1] else None,
                'stack_trace': self.format_exception(record.exc_info) if self.include_stack_trace else None
            }
            elk_entry['has_exception'] = True
        else:
            elk_entry['has_exception'] = False
        
        # Add tags for easier filtering
        tags = list(context.tags) if context.tags else []
        tags.append(f"level_{record.levelname.lower()}")
        tags.append(f"component_{context.component_type.value if context.component_type else 'unknown'}")
        if record.exc_info:
            tags.append('exception')
        elk_entry['tags'] = tags
        
        # Add custom fields
        if context.custom_fields:
            elk_entry['custom'] = context.custom_fields
        
        # Include raw message if configured
        if self.include_raw_message:
            elk_entry['raw_message'] = str(record.getMessage())
        
        # Apply redaction
        if self.redaction_enabled:
            elk_entry = self.redactor.redact_data(elk_entry, context.sensitivity_level)
        
        return json.dumps(elk_entry, separators=(',', ':'), default=str)


class PrometheusFormatter(BaseFormatter):
    """Formatter for Prometheus metrics format."""
    
    def format(self, record: LogRecord) -> str:
        """Format log record as Prometheus metrics."""
        context = self.get_log_context(record)
        
        # Only format records that contain metric data
        if not hasattr(record, 'metric_name') or not hasattr(record, 'metric_value'):
            return ""
        
        metric_name = record.metric_name
        metric_value = record.metric_value
        metric_type = getattr(record, 'metric_type', 'gauge')
        metric_help = getattr(record, 'metric_help', f"AI Assistant metric: {metric_name}")
        
        # Build labels
        labels = {}
        if context.component_type:
            labels['component'] = context.component_type.value
        if context.component_name:
            labels['component_name'] = context.component_name
        if context.user_id:
            labels['user_id'] = context.user_id
        if context.session_id:
            labels['session_id'] = context.session_id
        
        # Add custom labels
        if hasattr(record, 'metric_labels') and isinstance(record.metric_labels, dict):
            labels.update(record.metric_labels)
        
        # Format labels
        label_str = ""
        if labels:
            label_pairs = [f'{k}="{v}"' for k, v in labels.items()]
            label_str = "{" + ",".join(label_pairs) + "}"
        
        # Build Prometheus metric format
        lines = [
            f"# HELP {metric_name} {metric_help}",
            f"# TYPE {metric_name} {metric_type}",
            f"{metric_name}{label_str} {metric_value}"
        ]
        
        return "\n".join(lines)


def get_formatter(format_type: LogFormat, **kwargs) -> BaseFormatter:
    """
    Factory function to get the appropriate formatter.
    
    Args:
        format_type: Type of formatter to create
        **kwargs: Additional arguments for formatter
        
    Returns:
        Configured formatter instance
    """
    formatters = {
        LogFormat.JSON: JSONFormatter,
        LogFormat.STRUCTURED: StructuredFormatter,
        LogFormat.CONSOLE: ConsoleFormatter,
        LogFormat.AUDIT: AuditFormatter,
        LogFormat.PERFORMANCE: PerformanceFormatter,
        LogFormat.ELK: ELKFormatter,
        LogFormat.PROMETHEUS: PrometheusFormatter,
        LogFormat.PLAIN: StructuredFormatter,  # Fallback to structured
        LogFormat.DEBUG: StructuredFormatter,  # Debug uses structured
        LogFormat.COMPACT: ConsoleFormatter    # Compact uses console
    }
    
    formatter_class = formatters.get(format_type, JSONFormatter)
    return formatter_class(**kwargs)


def set_context(**kwargs) -> None:
    """
    Set logging context variables.
    
    Args:
        **kwargs: Context variables to set
    """
    if 'correlation_id' in kwargs:
        correlation_id.set(kwargs['correlation_id'])
    if 'session_id' in kwargs:
        session_id.set(kwargs['session_id'])
    if 'user_id' in kwargs:
        user_id.set(kwargs['user_id'])
    if 'component_type' in kwargs:
        if isinstance(kwargs['component_type'], str):
            component_type.set(ComponentType(kwargs['component_type']))
        else:
            component_type.set(kwargs['component_type'])
    if 'trace_id' in kwargs:
        trace_id.set(kwargs['trace_id'])
    if 'span_id' in kwargs:
        span_id.set(kwargs['span_id'])


def get_context() -> Dict[str, Any]:
    """
    Get current logging context.
    
    Returns:
        Dictionary of current context variables
    """
    return {
        'correlation_id': correlation_id.get(),
        'session_id': session_id.get(),
        'user_id': user_id.get(),
        'component_type': component_type.get(),
        'trace_id': trace_id.get(),
        'span_id': span_id.get()
    }


def clear_context() -> None:
    """Clear all logging context variables."""
    correlation_id.set(None)
    session_id.set(None)
    user_id.set(None)
    component_type.set(None)
    trace_id.set(None)
    span_id.set(None)


# Context managers for automatic context management
class logging_context:
    """Context manager for automatic logging context management."""
    
    def __init__(self, **context_vars):
        self.context_vars = context_vars
        self.previous_context = {}
    
    def __enter__(self):
        # Save current context
        self.previous_context = get_context()
        
        # Set new context
        set_context(**self.context_vars)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore previous context
        set_context(**self.previous_context)


class component_context:
    """Context manager specifically for component logging."""
    
    def __init__(self, component_type: Union[ComponentType, str], 
                 component_name: Optional[str] = None,
                 **additional_context):
        self.component_type = component_type
        self.component_name = component_name
        self.additional_context = additional_context
    
    def __enter__(self):
        context_vars = {
            'component_type': self.component_type,
            **self.additional_context
        }
        if self.component_name:
            context_vars['component_name'] = self.component_name
        
        set_context(**context_vars)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        clear_context()


# Utility functions for structured logging
def log_performance(logger: logging.Logger, operation: str, duration: float, **metrics):
    """Log performance metrics."""
    logger.info(
        f"Performance: {operation} completed in {duration:.3f}s",
        extra={
            'metric_name': f'{operation}_duration',
            'metric_value': duration,
            'metric_unit': 'seconds',
            'operation_name': operation,
            'operation_duration': duration,
            'operation_status': 'completed',
            'performance_data': metrics
        }
    )


def log_audit(logger: logging.Logger, action: str, resource: str, result: str, **context):
    """Log audit events."""
    logger.info(
        f"Audit: {action} on {resource} - {result}",
        extra={
            'audit_action': action,
            'audit_resource': resource,
            'audit_result': result,
            'audit_data': context,
            'audit_resources': [resource],
            'audit_categories': ['access', 'data_operation']
        }
    )


def log_security(logger: logging.Logger, event: str, severity: str = 'info', **details):
    """Log security events."""
    level_map = {
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'critical': logging.CRITICAL
    }
    
    logger.log(
        level_map.get(severity, logging.INFO),
        f"Security: {event}",
        extra={
            'security_event': event,
            'security_severity': severity,
            'security_details': details,
            'security_classification': 'restricted'
        }
    )


def create_structured_logger(name: str, format_type: LogFormat = LogFormat.JSON, 
                           level: LogLevel = logging.INFO, **formatter_kwargs) -> logging.Logger:
    """
    Create a logger with structured formatting.
    
    Args:
        name: Logger name
        format_type: Type of formatter to use
        level: Logging level
        **formatter_kwargs: Additional formatter arguments
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create handler with structured formatter
    handler = logging.StreamHandler()
    formatter = get_formatter(format_type, **formatter_kwargs)
    handler.setFormatter(formatter)
    
    logger.addHandler(handler)
    return logger
