"""
Advanced Logging Handlers for AI Assistant
Author: Drmusab
Last Modified: 2025-06-26 11:17:17 UTC

This module provides comprehensive custom logging handlers for the AI assistant,
including event-driven handlers, security-aware handlers, performance handlers,
audit handlers, and integration with external systems.
"""

import os
import sys
import json
import logging
import logging.handlers
import threading
import time
import uuid
import queue
import asyncio
import socket
import ssl
from pathlib import Path
from typing import Optional, Dict, Any, List, Union, Callable, Set, AsyncGenerator
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
from contextvars import ContextVar
from concurrent.futures import ThreadPoolExecutor
import traceback
import hashlib
import weakref
from abc import ABC, abstractmethod
import subprocess
import tempfile
import gzip
import pickle
import base64

# Third-party imports
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    import elasticsearch
    ELASTICSEARCH_AVAILABLE = True
except ImportError:
    ELASTICSEARCH_AVAILABLE = False

try:
    import kafka
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False

try:
    import pika
    RABBITMQ_AVAILABLE = True
except ImportError:
    RABBITMQ_AVAILABLE = False

# Core imports
from src.core.config.loader import ConfigLoader
from src.core.events.event_bus import EventBus
from src.core.events.event_types import BaseEvent, LoggingEvent, AuditLogEntry, SecurityEvent
from src.core.error_handling import ErrorHandler, handle_exceptions
from src.core.dependency_injection import Container
from src.core.health_check import HealthCheck
from src.core.security.encryption import EncryptionManager
from src.core.security.sanitization import SecuritySanitizer

# Observability imports
from src.observability.monitoring.metrics import MetricsCollector
from src.observability.monitoring.tracing import TraceManager
from src.observability.logging.config import LogLevel, LogFormat, LogContext, LoggingConfig
from src.observability.logging.formatters import (
    BaseFormatter, JSONFormatter, StructuredFormatter, AuditFormatter,
    PerformanceFormatter, ELKFormatter, get_formatter
)
from src.observability.logging.filters import (
    SensitiveDataFilter, RateLimitFilter, ComponentFilter, SecurityFilter
)


class HandlerState(Enum):
    """Handler operational states."""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    ERROR = "error"
    SHUTDOWN = "shutdown"


class DeliveryMode(Enum):
    """Log delivery modes."""
    SYNCHRONOUS = "synchronous"
    ASYNCHRONOUS = "asynchronous"
    BATCH = "batch"
    STREAMING = "streaming"
    BUFFERED = "buffered"


class CompressionType(Enum):
    """Compression types for log data."""
    NONE = "none"
    GZIP = "gzip"
    BZIP2 = "bzip2"
    LZMA = "lzma"


class FailurePolicy(Enum):
    """Failure handling policies."""
    DROP = "drop"                    # Drop failed logs
    RETRY = "retry"                  # Retry failed logs
    FALLBACK = "fallback"            # Use fallback handler
    DEAD_LETTER = "dead_letter"      # Send to dead letter queue
    CIRCUIT_BREAKER = "circuit_breaker"  # Use circuit breaker pattern


@dataclass
class HandlerConfig:
    """Configuration for logging handlers."""
    # Basic settings
    level: LogLevel = LogLevel.INFO
    formatter: str = "json"
    enabled: bool = True
    
    # Delivery settings
    delivery_mode: DeliveryMode = DeliveryMode.ASYNCHRONOUS
    batch_size: int = 100
    batch_timeout: float = 5.0
    buffer_size: int = 1000
    
    # Reliability settings
    failure_policy: FailurePolicy = FailurePolicy.RETRY
    max_retries: int = 3
    retry_delay: float = 1.0
    retry_backoff: float = 2.0
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: float = 60.0
    
    # Performance settings
    compression: CompressionType = CompressionType.NONE
    compression_level: int = 6
    async_workers: int = 2
    queue_timeout: float = 1.0
    
    # Security settings
    encryption_enabled: bool = False
    tls_enabled: bool = False
    authentication_required: bool = False
    sanitization_enabled: bool = True
    
    # Filters
    filters: List[str] = field(default_factory=list)
    
    # Custom properties
    custom_properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LogEntry:
    """Enhanced log entry for handler processing."""
    record: logging.LogRecord
    formatted_message: str
    context: LogContext
    timestamp: datetime
    level: str
    logger_name: str
    
    # Metadata
    entry_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    retry_count: int = 0
    priority: int = 0
    
    # Processing state
    processed: bool = False
    delivered: bool = False
    failed: bool = False
    error_message: Optional[str] = None
    
    # Additional data
    extra_data: Dict[str, Any] = field(default_factory=dict)


class LogHandlerError(Exception):
    """Custom exception for log handler operations."""
    
    def __init__(self, message: str, handler_name: Optional[str] = None, 
                 error_code: Optional[str] = None):
        super().__init__(message)
        self.handler_name = handler_name
        self.error_code = error_code
        self.timestamp = datetime.now(timezone.utc)


class CircuitBreaker:
    """Circuit breaker for handler reliability."""
    
    def __init__(self, failure_threshold: int = 5, timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = "closed"  # closed, open, half_open
        self._lock = threading.Lock()
    
    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        with self._lock:
            if self.state == "open":
                if (datetime.now(timezone.utc) - self.last_failure_time).total_seconds() > self.timeout:
                    self.state = "half_open"
                else:
                    raise LogHandlerError("Circuit breaker is open")
            
            try:
                result = func(*args, **kwargs)
                if self.state == "half_open":
                    self.state = "closed"
                    self.failure_count = 0
                return result
            except Exception as e:
                self.failure_count += 1
                self.last_failure_time = datetime.now(timezone.utc)
                
                if self.failure_count >= self.failure_threshold:
                    self.state = "open"
                
                raise e


class BaseEnhancedHandler(logging.Handler, ABC):
    """Base class for enhanced logging handlers."""
    
    def __init__(self, config: HandlerConfig, name: Optional[str] = None):
        super().__init__()
        self.handler_config = config
        self.handler_name = name or self.__class__.__name__
        self.state = HandlerState.INITIALIZING
        
        # Setup components
        self._setup_formatter()
        self._setup_filters()
        self._setup_reliability()
        self._setup_processing()
        
        # State management
        self.start_time = datetime.now(timezone.utc)
        self.stats = {
            'records_processed': 0,
            'records_delivered': 0,
            'records_failed': 0,
            'errors': 0,
            'retries': 0
        }
        
        # Locks and queues
        self._lock = threading.Lock()
        self._shutdown_event = threading.Event()
        
        self.state = HandlerState.ACTIVE
    
    def _setup_formatter(self) -> None:
        """Setup log formatter."""
        try:
            formatter = get_formatter(
                LogFormat(self.handler_config.formatter),
                config=getattr(self, 'config', None),
                sanitizer=getattr(self, 'sanitizer', None)
            )
            self.setFormatter(formatter)
        except Exception as e:
            # Fallback to basic formatter
            self.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
    
    def _setup_filters(self) -> None:
        """Setup log filters."""
        for filter_name in self.handler_config.filters:
            try:
                if filter_name == "sensitive_data":
                    self.addFilter(SensitiveDataFilter())
                elif filter_name == "rate_limit":
                    self.addFilter(RateLimitFilter())
                elif filter_name == "component":
                    self.addFilter(ComponentFilter())
                elif filter_name == "security":
                    self.addFilter(SecurityFilter())
            except Exception as e:
                # Log filter setup error but continue
                pass
    
    def _setup_reliability(self) -> None:
        """Setup reliability mechanisms."""
        if self.handler_config.failure_policy == FailurePolicy.CIRCUIT_BREAKER:
            self.circuit_breaker = CircuitBreaker(
                self.handler_config.circuit_breaker_threshold,
                self.handler_config.circuit_breaker_timeout
            )
        else:
            self.circuit_breaker = None
    
    def _setup_processing(self) -> None:
        """Setup processing infrastructure."""
        if self.handler_config.delivery_mode == DeliveryMode.ASYNCHRONOUS:
            self.processing_queue = queue.Queue(maxsize=self.handler_config.buffer_size)
            self.workers = []
            for i in range(self.handler_config.async_workers):
                worker = threading.Thread(
                    target=self._worker_loop,
                    name=f"{self.handler_name}_worker_{i}",
                    daemon=True
                )
                worker.start()
                self.workers.append(worker)
        
        if self.handler_config.delivery_mode == DeliveryMode.BATCH:
            self.batch_queue = []
            self.batch_timer = threading.Timer(
                self.handler_config.batch_timeout,
                self._flush_batch
            )
            self.batch_timer.start()
    
    def emit(self, record: logging.LogRecord) -> None:
        """Emit log record with enhanced processing."""
        try:
            if self.state != HandlerState.ACTIVE:
                return
            
            # Create enhanced log entry
            entry = self._create_log_entry(record)
            
            # Process based on delivery mode
            if self.handler_config.delivery_mode == DeliveryMode.SYNCHRONOUS:
                self._handle_synchronous(entry)
            elif self.handler_config.delivery_mode == DeliveryMode.ASYNCHRONOUS:
                self._handle_asynchronous(entry)
            elif self.handler_config.delivery_mode == DeliveryMode.BATCH:
                self._handle_batch(entry)
            elif self.handler_config.delivery_mode == DeliveryMode.STREAMING:
                self._handle_streaming(entry)
            else:
                self._handle_buffered(entry)
            
            self.stats['records_processed'] += 1
            
        except Exception as e:
            self.stats['errors'] += 1
            self.handleError(record)
    
    def _create_log_entry(self, record: logging.LogRecord) -> LogEntry:
        """Create enhanced log entry from record."""
        # Format message
        formatted_message = self.format(record)
        
        # Extract context
        context = getattr(record, 'context', LogContext())
        
        # Create entry
        entry = LogEntry(
            record=record,
            formatted_message=formatted_message,
            context=context,
            timestamp=datetime.fromtimestamp(record.created, timezone.utc),
            level=record.levelname,
            logger_name=record.name
        )
        
        # Add priority based on log level
        level_priorities = {
            'CRITICAL': 4,
            'ERROR': 3,
            'WARNING': 2,
            'INFO': 1,
            'DEBUG': 0
        }
        entry.priority = level_priorities.get(record.levelname, 0)
        
        return entry
    
    def _handle_synchronous(self, entry: LogEntry) -> None:
        """Handle synchronous log delivery."""
        try:
            if self.circuit_breaker:
                self.circuit_breaker.call(self._deliver_log, entry)
            else:
                self._deliver_log(entry)
            
            entry.delivered = True
            self.stats['records_delivered'] += 1
            
        except Exception as e:
            self._handle_delivery_failure(entry, e)
    
    def _handle_asynchronous(self, entry: LogEntry) -> None:
        """Handle asynchronous log delivery."""
        try:
            self.processing_queue.put(entry, timeout=self.handler_config.queue_timeout)
        except queue.Full:
            # Handle queue full based on failure policy
            if self.handler_config.failure_policy == FailurePolicy.DROP:
                self.stats['records_failed'] += 1
            else:
                # Try synchronous delivery as fallback
                self._handle_synchronous(entry)
    
    def _handle_batch(self, entry: LogEntry) -> None:
        """Handle batch log delivery."""
        with self._lock:
            self.batch_queue.append(entry)
            
            if len(self.batch_queue) >= self.handler_config.batch_size:
                self._flush_batch()
    
    def _handle_streaming(self, entry: LogEntry) -> None:
        """Handle streaming log delivery."""
        # For streaming, we deliver immediately but non-blocking
        if hasattr(self, '_stream_connection') and self._stream_connection:
            try:
                self._stream_deliver(entry)
                entry.delivered = True
                self.stats['records_delivered'] += 1
            except Exception as e:
                self._handle_delivery_failure(entry, e)
        else:
            # Fallback to async if no stream connection
            self._handle_asynchronous(entry)
    
    def _handle_buffered(self, entry: LogEntry) -> None:
        """Handle buffered log delivery."""
        # Similar to async but with different buffering strategy
        self._handle_asynchronous(entry)
    
    def _worker_loop(self) -> None:
        """Worker loop for asynchronous processing."""
        while not self._shutdown_event.is_set():
            try:
                entry = self.processing_queue.get(timeout=1.0)
                
                try:
                    if self.circuit_breaker:
                        self.circuit_breaker.call(self._deliver_log, entry)
                    else:
                        self._deliver_log(entry)
                    
                    entry.delivered = True
                    self.stats['records_delivered'] += 1
                    
                except Exception as e:
                    self._handle_delivery_failure(entry, e)
                
                self.processing_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                self.stats['errors'] += 1
    
    def _flush_batch(self) -> None:
        """Flush batched log entries."""
        with self._lock:
            if not self.batch_queue:
                return
            
            batch = self.batch_queue.copy()
            self.batch_queue.clear()
        
        try:
            self._deliver_batch(batch)
            
            for entry in batch:
                entry.delivered = True
                self.stats['records_delivered'] += 1
                
        except Exception as e:
            for entry in batch:
                self._handle_delivery_failure(entry, e)
        
        # Restart batch timer
        if not self._shutdown_event.is_set():
            self.batch_timer = threading.Timer(
                self.handler_config.batch_timeout,
                self._flush_batch
            )
            self.batch_timer.start()
    
    def _handle_delivery_failure(self, entry: LogEntry, error: Exception) -> None:
        """Handle log delivery failure."""
        entry.failed = True
        entry.error_message = str(error)
        self.stats['records_failed'] += 1
        
        # Apply failure policy
        if self.handler_config.failure_policy == FailurePolicy.RETRY:
            if entry.retry_count < self.handler_config.max_retries:
                entry.retry_count += 1
                self.stats['retries'] += 1
                
                # Schedule retry with backoff
                delay = (self.handler_config.retry_delay * 
                        (self.handler_config.retry_backoff ** entry.retry_count))
                
                timer = threading.Timer(delay, self._retry_delivery, args=[entry])
                timer.start()
            
        elif self.handler_config.failure_policy == FailurePolicy.FALLBACK:
            self._handle_fallback(entry)
        
        elif self.handler_config.failure_policy == FailurePolicy.DEAD_LETTER:
            self._send_to_dead_letter(entry)
    
    def _retry_delivery(self, entry: LogEntry) -> None:
        """Retry log delivery."""
        try:
            self._deliver_log(entry)
            entry.delivered = True
            entry.failed = False
            self.stats['records_delivered'] += 1
            
        except Exception as e:
            self._handle_delivery_failure(entry, e)
    
    def _handle_fallback(self, entry: LogEntry) -> None:
        """Handle fallback delivery."""
        # Default fallback is to print to stderr
        try:
            print(entry.formatted_message, file=sys.stderr)
            entry.delivered = True
        except Exception:
            pass
    
    def _send_to_dead_letter(self, entry: LogEntry) -> None:
        """Send entry to dead letter queue."""
        # Default implementation stores in a file
        try:
            dead_letter_path = Path("data/logs/dead_letter.log")
            dead_letter_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(dead_letter_path, 'a') as f:
                dead_letter_entry = {
                    'timestamp': entry.timestamp.isoformat(),
                    'handler': self.handler_name,
                    'error': entry.error_message,
                    'retry_count': entry.retry_count,
                    'original_message': entry.formatted_message
                }
                f.write(json.dumps(dead_letter_entry) + '\n')
                
        except Exception:
            pass
    
    @abstractmethod
    def _deliver_log(self, entry: LogEntry) -> None:
        """Deliver log entry to destination. Must be implemented by subclasses."""
        pass
    
    def _deliver_batch(self, entries: List[LogEntry]) -> None:
        """Deliver batch of log entries. Can be overridden by subclasses."""
        for entry in entries:
            self._deliver_log(entry)
    
    def _stream_deliver(self, entry: LogEntry) -> None:
        """Deliver log entry via streaming. Can be overridden by subclasses."""
        self._deliver_log(entry)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get handler statistics."""
        uptime = (datetime.now(timezone.utc) - self.start_time).total_seconds()
        
        return {
            'handler_name': self.handler_name,
            'state': self.state.value,
            'uptime_seconds': uptime,
            'stats': self.stats.copy(),
            'config': asdict(self.handler_config)
        }
    
    def close(self) -> None:
        """Close handler and cleanup resources."""
        self.state = HandlerState.SHUTDOWN
        self._shutdown_event.set()
        
        # Stop batch timer
        if hasattr(self, 'batch_timer'):
            self.batch_timer.cancel()
        
        # Wait for workers to finish
        if hasattr(self, 'workers'):
            for worker in self.workers:
                worker.join(timeout=5.0)
        
        # Flush any remaining logs
        if hasattr(self, 'batch_queue') and self.batch_queue:
            self._flush_batch()
        
        super().close()


class EventBusHandler(BaseEnhancedHandler):
    """Handler that forwards logs to the event bus."""
    
    def __init__(self, event_bus: EventBus, config: HandlerConfig):
        self.event_bus = event_bus
        super().__init__(config, "EventBusHandler")
    
    def _deliver_log(self, entry: LogEntry) -> None:
        """Deliver log entry to event bus."""
        # Create logging event
        log_event = LoggingEvent(
            level=entry.level,
            logger_name=entry.logger_name,
            message=entry.formatted_message,
            timestamp=entry.timestamp,
            session_id=entry.context.session_id,
            user_id=entry.context.user_id,
            component=entry.context.component_name,
            correlation_id=entry.context.correlation_id,
            extra_data=entry.extra_data
        )
        
        # Emit event asynchronously
        asyncio.create_task(self.event_bus.emit(log_event))


class DatabaseHandler(BaseEnhancedHandler):
    """Handler that stores logs in a database."""
    
    def __init__(self, database_manager, config: HandlerConfig):
        self.database = database_manager
        super().__init__(config, "DatabaseHandler")
        self._setup_database()
    
    def _setup_database(self) -> None:
        """Setup database tables for logging."""
        try:
            # Create logs table if it doesn't exist
            create_table_sql = """
            CREATE TABLE IF NOT EXISTS logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                level TEXT NOT NULL,
                logger_name TEXT NOT NULL,
                message TEXT NOT NULL,
                module TEXT,
                function TEXT,
                line_number INTEGER,
                thread_id INTEGER,
                process_id INTEGER,
                session_id TEXT,
                user_id TEXT,
                correlation_id TEXT,
                component TEXT,
                trace_id TEXT,
                span_id TEXT,
                exception_info TEXT,
                extra_data TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_timestamp (timestamp),
                INDEX idx_level (level),
                INDEX idx_session_id (session_id),
                INDEX idx_correlation_id (correlation_id)
            )
            """
            asyncio.create_task(self.database.execute(create_table_sql))
            
        except Exception as e:
            # Log setup error but continue
            pass
    
    def _deliver_log(self, entry: LogEntry) -> None:
        """Deliver log entry to database."""
        # Prepare log data for database
        log_data = {
            'timestamp': entry.timestamp.isoformat(),
            'level': entry.level,
            'logger_name': entry.logger_name,
            'message': entry.record.getMessage(),
            'module': entry.record.module,
            'function': entry.record.funcName,
            'line_number': entry.record.lineno,
            'thread_id': entry.record.thread,
            'process_id': entry.record.process,
            'session_id': entry.context.session_id,
            'user_id': entry.context.user_id,
            'correlation_id': entry.context.correlation_id,
            'component': entry.context.component_name,
            'trace_id': entry.context.trace_id,
            'span_id': entry.context.span_id,
            'exception_info': (
                json.dumps(entry.record.exc_info) 
                if entry.record.exc_info else None
            ),
            'extra_data': json.dumps(entry.extra_data) if entry.extra_data else None
        }
        
        # Insert into database
        insert_sql = """
        INSERT INTO logs (
            timestamp, level, logger_name, message, module, function, 
            line_number, thread_id, process_id, session_id, user_id,
            correlation_id, component, trace_id, span_id, exception_info, extra_data
        ) VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        )
        """
        
        asyncio.create_task(self.database.execute(insert_sql, tuple(log_data.values())))
    
    def _deliver_batch(self, entries: List[LogEntry]) -> None:
        """Deliver batch of log entries to database."""
        if not entries:
            return
        
        # Prepare batch data
        batch_data = []
        for entry in entries:
            log_data = (
                entry.timestamp.isoformat(),
                entry.level,
                entry.logger_name,
                entry.record.getMessage(),
                entry.record.module,
                entry.record.funcName,
                entry.record.lineno,
                entry.record.thread,
                entry.record.process,
                entry.context.session_id,
                entry.context.user_id,
                entry.context.correlation_id,
                entry.context.component_name,
                entry.context.trace_id,
                entry.context.span_id,
                json.dumps(entry.record.exc_info) if entry.record.exc_info else None,
                json.dumps(entry.extra_data) if entry.extra_data else None
            )
            batch_data.append(log_data)
        
        # Batch insert
        insert_sql = """
        INSERT INTO logs (
            timestamp, level, logger_name, message, module, function, 
            line_number, thread_id, process_id, session_id, user_id,
            correlation_id, component, trace_id, span_id, exception_info, extra_data
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        asyncio.create_task(self.database.execute_many(insert_sql, batch_data))


class ElasticsearchHandler(BaseEnhancedHandler):
    """Handler that sends logs to Elasticsearch."""
    
    def __init__(self, elasticsearch_client, config: HandlerConfig, index_prefix: str = "ai-assistant"):
        self.elasticsearch = elasticsearch_client
        self.index_prefix = index_prefix
        super().__init__(config, "ElasticsearchHandler")
        self._setup_index_template()
    
    def _setup_index_template(self) -> None:
        """Setup Elasticsearch index template."""
        if not ELASTICSEARCH_AVAILABLE:
            raise LogHandlerError("Elasticsearch client not available")
        
        try:
            # Define index template
            template = {
                "index_patterns": [f"{self.index_prefix}-*"],
                "template": {
                    "settings": {
                        "number_of_shards": 1,
                        "number_of_replicas": 0,
                        "refresh_interval": "5s"
                    },
                    "mappings": {
                        "properties": {
                            "@timestamp": {"type": "date"},
                            "level": {"type": "keyword"},
                            "logger_name": {"type": "keyword"},
                            "message": {"type": "text"},
                            "component": {"type": "keyword"},
                            "session_id": {"type": "keyword"},
                            "user_id": {"type": "keyword"},
                            "correlation_id": {"type": "keyword"},
                            "trace_id": {"type": "keyword"},
                            "span_id": {"type": "keyword"},
                            "tags": {"type": "keyword"},
                            "exception": {
                                "properties": {
                                    "class": {"type": "keyword"},
                                    "message": {"type": "text"},
                                    "stack_trace": {"type": "text"}
                                }
                            }
                        }
                    }
                }
            }
            
            # Create or update template
            self.elasticsearch.indices.put_index_template(
                name=f"{self.index_prefix}-template",
                body=template
            )
            
        except Exception as e:
            # Log template setup error but continue
            pass
    
    def _get_index_name(self, timestamp: datetime) -> str:
        """Generate index name based on timestamp."""
        return f"{self.index_prefix}-{timestamp.strftime('%Y.%m.%d')}"
    
    def _deliver_log(self, entry: LogEntry) -> None:
        """Deliver log entry to Elasticsearch."""
        # Prepare document
        doc = {
            '@timestamp': entry.timestamp.isoformat(),
            'level': entry.level,
            'logger_name': entry.logger_name,
            'message': entry.record.getMessage(),
            'component': entry.context.component_name,
            'session_id': entry.context.session_id,
            'user_id': entry.context.user_id,
            'correlation_id': entry.context.correlation_id,
            'trace_id': entry.context.trace_id,
            'span_id': entry.context.span_id,
            'source': {
                'file': entry.record.filename,
                'function': entry.record.funcName,
                'line': entry.record.lineno
            },
            'process': {
                'pid': entry.record.process,
                'thread': {
                    'id': entry.record.thread,
                    'name': entry.record.threadName
                }
            }
        }
        
        # Add exception information
        if entry.record.exc_info:
            doc['exception'] = {
                'class': entry.record.exc_info[0].__name__,
                'message': str(entry.record.exc_info[1]),
                'stack_trace': traceback.format_exception(*entry.record.exc_info)
            }
        
        # Add tags
        tags = [f"level_{entry.level.lower()}"]
        if entry.context.component_name:
            tags.append(f"component_{entry.context.component_name}")
        if entry.record.exc_info:
            tags.append('exception')
        doc['tags'] = tags
        
        # Add extra data
        if entry.extra_data:
            doc['extra'] = entry.extra_data
        
        # Index document
        index_name = self._get_index_name(entry.timestamp)
        self.elasticsearch.index(
            index=index_name,
            body=doc,
            id=entry.entry_id
        )
    
    def _deliver_batch(self, entries: List[LogEntry]) -> None:
        """Deliver batch of log entries to Elasticsearch."""
        if not entries:
            return
        
        # Prepare bulk operations
        bulk_body = []
        
        for entry in entries:
            index_name = self._get_index_name(entry.timestamp)
            
            # Index operation
            bulk_body.append({
                'index': {
                    '_index': index_name,
                    '_id': entry.entry_id
                }
            })
            
            # Document
            doc = {
                '@timestamp': entry.timestamp.isoformat(),
                'level': entry.level,
                'logger_name': entry.logger_name,
                'message': entry.record.getMessage(),
                'component': entry.context.component_name,
                'session_id': entry.context.session_id,
                'user_id': entry.context.user_id,
                'correlation_id': entry.context.correlation_id,
                'trace_id': entry.context.trace_id,
                'span_id': entry.context.span_id
            }
            
            if entry.extra_data:
                doc['extra'] = entry.extra_data
            
            bulk_body.append(doc)
        
        # Execute bulk operation
        self.elasticsearch.bulk(body=bulk_body)


class RedisHandler(BaseEnhancedHandler):
    """Handler that sends logs to Redis."""
    
    def __init__(self, redis_client, config: HandlerConfig, key_prefix: str = "ai-assistant:logs"):
        self.redis = redis_client
        self.key_prefix = key_prefix
        super().__init__(config, "RedisHandler")
    
    def _deliver_log(self, entry: LogEntry) -> None:
        """Deliver log entry to Redis."""
        if not REDIS_AVAILABLE:
            raise LogHandlerError("Redis client not available")
        
        # Prepare log data
        log_data = {
            'timestamp': entry.timestamp.isoformat(),
            'level': entry.level,
            'logger_name': entry.logger_name,
            'message': entry.formatted_message,
            'session_id': entry.context.session_id,
            'user_id': entry.context.user_id,
            'correlation_id': entry.context.correlation_id,
            'component': entry.context.component_name
        }
        
        if entry.extra_data:
            log_data['extra'] = entry.extra_data
        
        # Store in Redis list
        key = f"{self.key_prefix}:{entry.timestamp.strftime('%Y-%m-%d')}"
        self.redis.lpush(key, json.dumps(log_data))
        
        # Set expiration (30 days)
        self.redis.expire(key, 30 * 24 * 3600)
    
    def _deliver_batch(self, entries: List[LogEntry]) -> None:
        """Deliver batch of log entries to Redis."""
        if not entries:
            return
        
        # Group by date for efficient batch operations
        date_groups = {}
        for entry in entries:
            date_key = entry.timestamp.strftime('%Y-%m-%d')
            if date_key not in date_groups:
                date_groups[date_key] = []
            
            log_data = {
                'timestamp': entry.timestamp.isoformat(),
                'level': entry.level,
                'logger_name': entry.logger_name,
                'message': entry.formatted_message,
                'session_id': entry.context.session_id,
                'user_id': entry.context.user_id,
                'correlation_id': entry.context.correlation_id,
                'component': entry.context.component_name
            }
            
            if entry.extra_data:
                log_data['extra'] = entry.extra_data
            
            date_groups[date_key].append(json.dumps(log_data))
        
        # Batch insert for each date
        pipeline = self.redis.pipeline()
        for date_key, log_entries in date_groups.items():
            key = f"{self.key_prefix}:{date_key}"
            pipeline.lpush(key, *log_entries)
            pipeline.expire(key, 30 * 24 * 3600)
        
        pipeline.execute()


class KafkaHandler(BaseEnhancedHandler):
    """Handler that sends logs to Apache Kafka."""
    
    def __init__(self, kafka_producer, config: HandlerConfig, topic: str = "ai-assistant-logs"):
        self.producer = kafka_producer
        self.topic = topic
        super().__init__(config, "KafkaHandler")
    
    def _deliver_log(self, entry: LogEntry) -> None:
        """Deliver log entry to Kafka."""
        if not KAFKA_AVAILABLE:
            raise LogHandlerError("Kafka client not available")
        
        # Prepare message
        message = {
            'timestamp': entry.timestamp.isoformat(),
            'level': entry.level,
            'logger_name': entry.logger_name,
            'message': entry.formatted_message,
            'session_id': entry.context.session_id,
            'user_id': entry.context.user_id,
            'correlation_id': entry.context.correlation_id,
            'component': entry.context.component_name
        }
        
        if entry.extra_data:
            message['extra'] = entry.extra_data
        
        # Send to Kafka
        self.producer.send(
            self.topic,
            value=json.dumps(message).encode('utf-8'),
            key=entry.context.correlation_id.encode('utf-8') if entry.context.correlation_id else None
        )


class WebhookHandler(BaseEnhancedHandler):
    """Handler that sends logs to HTTP webhooks."""
    
    def __init__(self, webhook_url: str, config: HandlerConfig, headers: Optional[Dict[str, str]] = None):
        self.webhook_url = webhook_url
        self.headers = headers or {}
        super().__init__(config, "WebhookHandler")
        
        # Setup HTTP session
        import requests
        self.session = requests.Session()
        self.session.headers.update(self.headers)
    
    def _deliver_log(self, entry: LogEntry) -> None:
        """Deliver log entry to webhook."""
        # Prepare payload
        payload = {
            'timestamp': entry.timestamp.isoformat(),
            'level': entry.level,
            'logger_name': entry.logger_name,
            'message': entry.formatted_message,
            'session_id': entry.context.session_id,
            'user_id': entry.context.user_id,
            'correlation_id': entry.context.correlation_id,
            'component': entry.context.component_name
        }
        
        if entry.extra_data:
            payload['extra'] = entry.extra_data
        
        # Send HTTP POST request
        response = self.session.post(
            self.webhook_url,
            json=payload,
            timeout=10.0
        )
        response.raise_for_status()
    
    def _deliver_batch(self, entries: List[LogEntry]) -> None:
        """Deliver batch of log entries to webhook."""
        if not entries:
            return
        
        # Prepare batch payload
        batch_payload = {
            'logs': [],
            'batch_size': len(entries),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        for entry in entries:
            log_data = {
                'timestamp': entry.timestamp.isoformat(),
                'level': entry.level,
                'logger_name': entry.logger_name,
                'message': entry.formatted_message,
                'session_id': entry.context.session_id,
                'user_id': entry.context.user_id,
                'correlation_id': entry.context.correlation_id,
                'component': entry.context.component_name
            }
            
            if entry.extra_data:
                log_data['extra'] = entry.extra_data
            
            batch_payload['logs'].append(log_data)
        
        # Send batch request
        response = self.session.post(
            self.webhook_url,
            json=batch_payload,
            timeout=30.0
        )
        response.raise_for_status()


class AuditHandler(BaseEnhancedHandler):
    """Specialized handler for audit logs with security focus."""
    
    def __init__(self, audit_file: Path, config: HandlerConfig, 
                 encryption_manager: Optional[EncryptionManager] = None):
        self.audit_file = audit_file
        self.encryption_manager = encryption_manager
        super().__init__(config, "AuditHandler")
        
        # Setup audit file
        self.audit_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Use audit formatter
        self.setFormatter(AuditFormatter())
        
        # Setup rotation
        self.rotating_handler = logging.handlers.RotatingFileHandler(
            filename=self.audit_file,
            maxBytes=100 * 1024 * 1024,  # 100MB
            backupCount=10,
            encoding='utf-8'
        )
    
    def _deliver_log(self, entry: LogEntry) -> None:
        """Deliver audit log entry to secure file."""
        # Prepare audit data
        audit_data = {
            'audit_id': entry.entry_id,
            'timestamp': entry.timestamp.isoformat(),
            'event_type': getattr(entry.record, 'audit_event_type', 'unknown'),
            'actor': entry.context.user_id or 'system',
            'resource': getattr(entry.record, 'audit_resource', 'unknown'),
            'action': getattr(entry.record, 'audit_action', 'unknown'),
            'result': getattr(entry.record, 'audit_result', 'unknown'),
            'session_id': entry.context.session_id,
            'correlation_id': entry.context.correlation_id,
            'source_ip': getattr(entry.record, 'source_ip', None),
            'user_agent': getattr(entry.record, 'user_agent', None),
            'details': entry.record.getMessage()
        }
        
        # Add risk assessment
        risk_level = getattr(entry.record, 'risk_level', 'low')
        audit_data['risk_level'] = risk_level
        
        # Serialize audit entry
        audit_line = json.dumps(audit_data, separators=(',', ':'))
        
        # Encrypt if enabled
        if self.encryption_manager and self.handler_config.encryption_enabled:
            audit_line = self.encryption_manager.encrypt(audit_line)
        
        # Write to file with integrity check
        with open(self.audit_file, 'a') as f:
            # Add checksum for integrity verification
            checksum = hashlib.sha256(audit_line.encode()).hexdigest()
            f.write(f"{audit_line}|{checksum}\n")


class SecurityHandler(BaseEnhancedHandler):
    """Specialized handler for security events."""
    
    def __init__(self, config: HandlerConfig, alert_webhook: Optional[str] = None):
        self.alert_webhook = alert_webhook
        super().__init__(config, "SecurityHandler")
        
        # Setup HTTP session for alerts
        if self.alert_webhook:
            import requests
            self.alert_session = requests.Session()
        
        # Security event thresholds
        self.alert_thresholds = {
            'failed_login': 5,
            'permission_denied': 10,
            'suspicious_activity': 1
        }
        
        # Event counters
        self.event_counters = defaultdict(int)
        self.counter_reset_time = datetime.now(timezone.utc)
    
    def _deliver_log(self, entry: LogEntry) -> None:
        """Deliver security log entry with alerting."""
        # Extract security event type
        security_event = getattr(entry.record, 'security_event', 'unknown')
        severity = getattr(entry.record, 'security_severity', 'info')
        
        # Update counters
        self.event_counters[security_event] += 1
        
        # Check if alert threshold is exceeded
        threshold = self.alert_thresholds.get(security_event, float('inf'))
        if self.event_counters[security_event] >= threshold:
            self._send_security_alert(entry, security_event, severity)
        
        # Reset counters every hour
        if (datetime.now(timezone.utc) - self.counter_reset_time).total_seconds() > 3600:
            self.event_counters.clear()
            self.counter_reset_time = datetime.now(timezone.utc)
        
        # Store security event
        security_data = {
            'security_id': entry.entry_id,
            'timestamp': entry.timestamp.isoformat(),
            'event_type': security_event,
            'severity': severity,
            'source_ip': getattr(entry.record, 'source_ip', None),
            'user_id': entry.context.user_id,
            'session_id': entry.context.session_id,
            'user_agent': getattr(entry.record, 'user_agent', None),
            'details': entry.record.getMessage(),
            'risk_score': getattr(entry.record, 'risk_score', 0)
        }
        
        # Write to security log file
        security_file = Path("data/logs/security.log")
        security_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(security_file, 'a') as f:
            f.write(json.dumps(security_data) + '\n')
    
    def _send_security_alert(self, entry: LogEntry, event_type: str, severity: str) -> None:
        """Send security alert."""
        if not self.alert_webhook:
            return
        
        alert_data = {
            'alert_type': 'security_threshold_exceeded',
            'event_type': event_type,
            'severity': severity,
            'count': self.event_counters[event_type],
            'threshold': self.alert_thresholds.get(event_type),
            'user_id': entry.context.user_id,
            'session_id': entry.context.session_id,
            'timestamp': entry.timestamp.isoformat(),
            'message': entry.record.getMessage()
        }
        
        try:
            self.alert_session.post(
                self.alert_webhook,
                json=alert_data,
                timeout=10.0
            )
        except Exception as e:
            # Log alert delivery failure but don't raise
            pass


class PerformanceHandler(BaseEnhancedHandler):
    """Specialized handler for performance metrics logging."""
    
    def __init__(self, config: HandlerConfig, metrics_collector: Optional[MetricsCollector] = None):
        self.metrics_collector = metrics_collector
        super().__init__(config, "PerformanceHandler")
        
        # Use performance formatter
        self.setFormatter(PerformanceFormatter())
        
        # Performance tracking
        self.performance_data = {}
        self.aggregation_window = 60.0  # 1 minute
        self.last_aggregation = datetime.now(timezone.utc)
    
    def _deliver_log(self, entry: LogEntry) -> None:
        """Deliver performance log entry with metrics aggregation."""
        # Extract performance metrics
        metric_name = getattr(entry.record, 'metric_name', 'unknown')
        metric_value = getattr(entry.record, 'metric_value', 0)
        metric_unit = getattr(entry.record, 'metric_unit', 'count')
        component = entry.context.component_name or 'unknown'
        
        # Store performance data
        key = f"{component}_{metric_name}"
        if key not in self.performance_data:
            self.performance_data[key] = []
        
        self.performance_data[key].append({
            'value': metric_value,
            'timestamp': entry.timestamp,
            'unit': metric_unit,
            'session_id': entry.context.session_id
        })
        
        # Update metrics collector if available
        if self.metrics_collector:
            self.metrics_collector.record(metric_name, metric_value, tags={
                'component': component,
                'unit': metric_unit
            })
        
        # Aggregate performance data periodically
        current_time = datetime.now(timezone.utc)
        if (current_time - self.last_aggregation).total_seconds() > self.aggregation_window:
            self._aggregate_performance_data()
            self.last_aggregation = current_time
        
        # Write performance data
        perf_data = {
            'performance_id': entry.entry_id,
            'timestamp': entry.timestamp.isoformat(),
            'metric_name': metric_name,
            'metric_value': metric_value,
            'metric_unit': metric_unit,
            'component': component,
            'session_id': entry.context.session_id,
            'correlation_id': entry.context.correlation_id
        }
        
        # Write to performance log file
        perf_file = Path("data/logs/performance.log")
        perf_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(perf_file, 'a') as f:
            f.write(json.dumps(perf_data) + '\n')
    
    def _aggregate_performance_data(self) -> None:
        """Aggregate performance data for reporting."""
        aggregated_data = {}
        
        for key, values in self.performance_data.items():
            if not values:
                continue
            
            metric_values = [v['value'] for v in values]
            aggregated_data[key] = {
                'count': len(metric_values),
                'sum': sum(metric_values),
                'avg': sum(metric_values) / len(metric_values),
                'min': min(metric_values),
                'max': max(metric_values),
                'window_start': min(v['timestamp'] for v in values).isoformat(),
                'window_end': max(v['timestamp'] for v in values).isoformat()
            }
        
        # Write aggregated data
        if aggregated_data:
            agg_file = Path("data/logs/performance_aggregated.log")
            agg_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(agg_file, 'a') as f:
                f.write(json.dumps({
                    'aggregation_timestamp': datetime.now(timezone.utc).isoformat(),
                    'window_seconds': self.aggregation_window,
                    'metrics': aggregated_data
                }) + '\n')
        
        # Clear performance data
        self.performance_data.clear()


class HandlerManager:
    """Manager for all logging handlers."""
    
    def __init__(self, container: Container):
        self.container = container
        self.config = container.get(ConfigLoader)
        self.logger = logging.getLogger(__name__)
        
        # Registered handlers
        self.handlers: Dict[str, BaseEnhancedHandler] = {}
        self.handler_configs: Dict[str, HandlerConfig] = {}
        
        # Load handler configurations
        self._load_handler_configs()
    
    def _load_handler_configs(self) -> None:
        """Load handler configurations from config."""
        handlers_config = self.config.get("logging.handlers", {})
        
        for handler_name, handler_config in handlers_config.items():
            try:
                config = HandlerConfig(**handler_config)
                self.handler_configs[handler_name] = config
            except Exception as e:
                self.logger.error(f"Failed to load config for handler {handler_name}: {str(e)}")
    
    def register_handler(self, name: str, handler: BaseEnhancedHandler) -> None:
        """Register a logging handler."""
        self.handlers[name] = handler
        self.logger.info(f"Registered logging handler: {name}")
    
    def get_handler(self, name: str) -> Optional[BaseEnhancedHandler]:
        """Get a registered handler by name."""
        return self.handlers.get(name)
    
    def create_event_bus_handler(self) -> EventBusHandler:
        """Create an event bus handler."""
        config = self.handler_configs.get("event_bus", HandlerConfig())
        event_bus = self.container.get(EventBus)
        return EventBusHandler(event_bus, config)
    
    def create_database_handler(self) -> DatabaseHandler:
        """Create a database handler."""
        config = self.handler_configs.get("database", HandlerConfig())
        database = self.container.get("DatabaseManager")
        return DatabaseHandler(database, config)
    
    def create_elasticsearch_handler(self) -> ElasticsearchHandler:
        """Create an Elasticsearch handler."""
        if not ELASTICSEARCH_AVAILABLE:
            raise LogHandlerError("Elasticsearch not available")
        
        config = self.handler_configs.get("elasticsearch", HandlerConfig())
        es_config = self.config.get("logging.elasticsearch", {})
        
        # Create Elasticsearch client
        es_client = elasticsearch.Elasticsearch(
            hosts=es_config.get("hosts", ["localhost:9200"]),
            timeout=es_config.get("timeout", 30),
            max_retries=es_config.get("max_retries", 3)
        )
        
        return ElasticsearchHandler(
            es_client, 
            config, 
            es_config.get("index_prefix", "ai-assistant")
        )
    
    def create_redis_handler(self) -> RedisHandler:
        """Create a Redis handler."""
        if not REDIS_AVAILABLE:
            raise LogHandlerError("Redis not available")
        
        config = self.handler_configs.get("redis", HandlerConfig())
        redis_config = self.config.get("logging.redis", {})
        
        # Create Redis client
        redis_client = redis.Redis(
            host=redis_config.get("host", "localhost"),
            port=redis_config.get("port", 6379),
            db=redis_config.get("db", 0),
            password=redis_config.get("password")
        )
        
        return RedisHandler(
            redis_client,
            config,
            redis_config.get("key_prefix", "ai-assistant:logs")
        )
    
    def create_audit_handler(self) -> AuditHandler:
        """Create an audit handler."""
        config = self.handler_configs.get("audit", HandlerConfig())
        audit_config = self.config.get("logging.audit", {})
        
        audit_file = Path(audit_config.get("file", "data/logs/audit.log"))
        encryption_manager = None
        
        if audit_config.get("encryption_enabled", False):
            try:
                encryption_manager = self.container.get(EncryptionManager)
            except Exception:
                pass
        
        return AuditHandler(audit_file, config, encryption_manager)
    
    def create_security_handler(self) -> SecurityHandler:
        """Create a security handler."""
        config = self.handler_configs.get("security", HandlerConfig())
        security_config = self.config.get("logging.security", {})
        
        return SecurityHandler(
            config,
            security_config.get("alert_webhook")
        )
    
    def create_performance_handler(self) -> PerformanceHandler:
        """Create a performance handler."""
        config = self.handler_configs.get("performance", HandlerConfig())
        
        metrics_collector = None
        try:
            metrics_collector = self.container.get(MetricsCollector)
        except Exception:
            pass
        
        return PerformanceHandler(config, metrics_collector)
    
    def get_all_handlers(self) -> Dict[str, BaseEnhancedHandler]:
        """Get all registered handlers."""
        return self.handlers.copy()
    
    def get_handler_stats(self) -> Dict[str, Any]:
        """Get statistics for all handlers."""
        stats = {}
        for name, handler in self.handlers.items():
            try:
                stats[name] = handler.get_stats()
            except Exception as e:
                stats[name] = {'error': str(e)}
        return stats
    
    def shutdown_all_handlers(self) -> None:
        """Shutdown all registered handlers."""
        for name, handler in self.handlers.items():
            try:
                handler.close()
                self.logger.info(f"Closed handler: {name}")
            except Exception as e:
                self.logger.error(f"Error closing handler {name}: {str(e)}")
        
        self.handlers.clear()


# Convenience functions for creating handlers
def get_logger_with_handlers(name: str, container: Container, 
                           handler_names: Optional[List[str]] = None) -> logging.Logger:
    """Get a logger configured with specified handlers."""
    logger = logging.getLogger(name)
    
    if handler_names:
        handler_manager = HandlerManager(container)
        
        for handler_name in handler_names:
            try:
                if handler_name == "event_bus":
                    handler = handler_manager.create_event_bus_handler()
                elif handler_name == "database":
                    handler = handler_manager.create_database_handler()
                elif handler_name == "elasticsearch":
                    handler = handler_manager.create_elasticsearch_handler()
                elif handler_name == "redis":
                    handler = handler_manager.create_redis_handler()
                elif handler_name == "audit":
                    handler = handler_manager.create_audit_handler()
                elif handler_name == "security":
                    handler = handler_manager.create_security_handler()
                elif handler_name == "performance":
                    handler = handler_manager.create_performance_handler()
                else:
                    continue
                
                logger.addHandler(handler)
                
            except Exception as e:
                # Log handler creation error but continue
                logging.getLogger(__name__).error(
                    f"Failed to create handler {handler_name}: {str(e)}"
                )
    
    return logger


# Export commonly used classes
__all__ = [
    'HandlerConfig',
    'LogEntry',
    'BaseEnhancedHandler',
    'EventBusHandler',
    'DatabaseHandler',
    'ElasticsearchHandler',
    'RedisHandler',
    'KafkaHandler',
    'WebhookHandler',
    'AuditHandler',
    'SecurityHandler',
    'PerformanceHandler',
    'HandlerManager',
    'get_logger_with_handlers'
]
