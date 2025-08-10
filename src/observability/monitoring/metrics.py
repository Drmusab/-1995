"""
Advanced Metrics Collection System
Author: Drmusab
Last Modified: 2025-01-20 12:00:00 UTC

This module provides comprehensive metrics collection for the AI assistant system,
supporting various metric types, export backends, and real-time monitoring integration
with all core components including the engine, orchestrator, session manager, and more.
"""

import gc
import hashlib
import inspect
import json
import logging
import statistics
import threading
import time
import uuid
import weakref
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Set, Type, Union

import asyncio
import psutil

# Core imports
from src.core.config.loader import ConfigLoader
from src.core.dependency_injection import Container
from src.core.error_handling import ErrorHandler, handle_exceptions
from src.core.events.event_bus import EventBus
from src.core.events.event_types import (
    ComponentHealthChanged,
    MetricExported,
    MetricRecorded,
    MetricThresholdExceeded,
    PerformanceWarning,
    PluginLoaded,
    PluginUnloaded,
    ProcessingCompleted,
    ProcessingStarted,
    SessionEnded,
    SessionStarted,
    SystemStateChanged,
    UserInteractionCompleted,
    UserInteractionStarted,
    WorkflowCompleted,
    WorkflowStarted,
)
# Core imports - conditional to avoid circular dependencies
try:
    from src.core.health_check import HealthCheck
except ImportError:
    HealthCheck = None

# Observability
from src.observability.logging.config import get_logger


class MetricType(Enum):
    """Types of metrics supported by the system."""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"
    TIMER = "timer"
    SET = "set"


class MetricAggregation(Enum):
    """Aggregation methods for metrics."""

    SUM = "sum"
    AVG = "avg"
    MIN = "min"
    MAX = "max"
    COUNT = "count"
    P50 = "p50"
    P90 = "p90"
    P95 = "p95"
    P99 = "p99"
    RATE = "rate"


class ExportFormat(Enum):
    """Supported metric export formats."""

    PROMETHEUS = "prometheus"
    STATSD = "statsd"
    JSON = "json"
    CSV = "csv"
    INFLUXDB = "influxdb"
    CUSTOM = "custom"


@dataclass
class MetricConfig:
    """Configuration for individual metrics."""

    name: str
    metric_type: MetricType
    description: str = ""
    unit: str = ""
    tags: Dict[str, str] = field(default_factory=dict)
    buckets: Optional[List[float]] = None  # For histograms
    quantiles: Optional[List[float]] = None  # For summaries
    max_age_seconds: float = 3600.0  # Data retention
    sample_rate: float = 1.0  # Sampling rate
    enabled: bool = True


@dataclass
class MetricValue:
    """Represents a metric value with metadata."""

    value: Union[int, float]
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class MetricSample:
    """A sample for histogram/summary metrics."""

    value: float
    timestamp: datetime
    weight: float = 1.0


class MetricError(Exception):
    """Custom exception for metric operations."""

    def __init__(self, message: str, metric_name: Optional[str] = None):
        super().__init__(message)
        self.metric_name = metric_name
        self.timestamp = datetime.now(timezone.utc)


class MetricBackend(ABC):
    """Abstract base class for metric export backends."""

    @abstractmethod
    async def export_metrics(self, metrics: Dict[str, Any]) -> None:
        """Export metrics to the backend."""
        pass

    @abstractmethod
    def format_metric(self, name: str, value: Any, tags: Dict[str, str]) -> str:
        """Format a metric for export."""
        pass

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the backend."""
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        """Cleanup backend resources."""
        pass


class PrometheusBackend(MetricBackend):
    """Prometheus metrics export backend."""

    def __init__(self, endpoint: str = "http://localhost:9090"):
        self.endpoint = endpoint
        self.logger = get_logger(__name__)

    async def initialize(self) -> None:
        """Initialize Prometheus backend."""
        self.logger.info(f"Initialized Prometheus backend: {self.endpoint}")

    def format_metric(self, name: str, value: Any, tags: Dict[str, str]) -> str:
        """Format metric in Prometheus format."""
        # Convert metric name to Prometheus format
        prom_name = name.replace("-", "_").replace(".", "_")

        # Format tags
        if tags:
            tag_str = ",".join([f'{k}="{v}"' for k, v in tags.items()])
            return f"{prom_name}{{{tag_str}}} {value}"
        else:
            return f"{prom_name} {value}"

    async def export_metrics(self, metrics: Dict[str, Any]) -> None:
        """Export metrics to Prometheus."""
        try:
            # In a real implementation, this would use the Prometheus client library
            # or send metrics to a Prometheus pushgateway
            formatted_metrics = []

            for name, metric_data in metrics.items():
                if isinstance(metric_data, dict) and "value" in metric_data:
                    formatted = self.format_metric(
                        name, metric_data["value"], metric_data.get("tags", {})
                    )
                    formatted_metrics.append(formatted)

            # Log for now (would be replaced with actual export)
            self.logger.debug(f"Exported {len(formatted_metrics)} metrics to Prometheus")

        except Exception as e:
            self.logger.error(f"Failed to export metrics to Prometheus: {str(e)}")

    async def cleanup(self) -> None:
        """Cleanup Prometheus backend."""
        pass


class StatsDBackend(MetricBackend):
    """StatsD metrics export backend."""

    def __init__(self, host: str = "localhost", port: int = 8125):
        self.host = host
        self.port = port
        self.logger = get_logger(__name__)

    async def initialize(self) -> None:
        """Initialize StatsD backend."""
        self.logger.info(f"Initialized StatsD backend: {self.host}:{self.port}")

    def format_metric(self, name: str, value: Any, tags: Dict[str, str]) -> str:
        """Format metric in StatsD format."""
        # StatsD format: metric.name:value|type|@sample_rate|#tag1:value1,tag2:value2
        tag_str = ""
        if tags:
            tag_str = "|#" + ",".join([f"{k}:{v}" for k, v in tags.items()])

        return f"{name}:{value}|g{tag_str}"  # g for gauge, would vary by type

    async def export_metrics(self, metrics: Dict[str, Any]) -> None:
        """Export metrics to StatsD."""
        try:
            # In a real implementation, this would use a StatsD client
            formatted_metrics = []

            for name, metric_data in metrics.items():
                if isinstance(metric_data, dict) and "value" in metric_data:
                    formatted = self.format_metric(
                        name, metric_data["value"], metric_data.get("tags", {})
                    )
                    formatted_metrics.append(formatted)

            self.logger.debug(f"Exported {len(formatted_metrics)} metrics to StatsD")

        except Exception as e:
            self.logger.error(f"Failed to export metrics to StatsD: {str(e)}")

    async def cleanup(self) -> None:
        """Cleanup StatsD backend."""
        pass


class JSONBackend(MetricBackend):
    """JSON file export backend."""

    def __init__(self, file_path: str = "metrics.json"):
        self.file_path = Path(file_path)
        self.logger = get_logger(__name__)

    async def initialize(self) -> None:
        """Initialize JSON backend."""
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Initialized JSON backend: {self.file_path}")

    def format_metric(self, name: str, value: Any, tags: Dict[str, str]) -> str:
        """Format metric as JSON string."""
        metric_dict = {
            "name": name,
            "value": value,
            "tags": tags,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        return json.dumps(metric_dict)

    async def export_metrics(self, metrics: Dict[str, Any]) -> None:
        """Export metrics to JSON file."""
        try:
            export_data = {"timestamp": datetime.now(timezone.utc).isoformat(), "metrics": metrics}

            with open(self.file_path, "w") as f:
                json.dump(export_data, f, indent=2, default=str)

            self.logger.debug(f"Exported {len(metrics)} metrics to JSON file")

        except Exception as e:
            self.logger.error(f"Failed to export metrics to JSON: {str(e)}")

    async def cleanup(self) -> None:
        """Cleanup JSON backend."""
        pass


class MetricStore:
    """In-memory metric storage with efficient operations."""

    def __init__(self, max_samples: int = 10000):
        self.max_samples = max_samples
        self._metrics: Dict[str, Dict[str, Any]] = {}
        self._samples: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_samples))
        self._lock = threading.RLock()
        self.logger = get_logger(__name__)

    def record_counter(self, name: str, value: float = 1.0, tags: Dict[str, str] = None) -> None:
        """Record a counter metric."""
        with self._lock:
            tags = tags or {}
            if name not in self._metrics:
                self._metrics[name] = {
                    "type": "counter",
                    "value": 0.0,
                    "tags": tags,
                    "created_at": datetime.now(timezone.utc),
                }

            self._metrics[name]["value"] += value
            self._metrics[name]["last_updated"] = datetime.now(timezone.utc)
            self._metrics[name]["tags"].update(tags)

    def set_gauge(self, name: str, value: float, tags: Dict[str, str] = None) -> None:
        """Set a gauge metric value."""
        with self._lock:
            tags = tags or {}
            self._metrics[name] = {
                "type": "gauge",
                "value": value,
                "tags": tags,
                "last_updated": datetime.now(timezone.utc),
            }

    def record_histogram(self, name: str, value: float, tags: Dict[str, str] = None) -> None:
        """Record a histogram sample."""
        with self._lock:
            tags = tags or {}
            sample = MetricSample(value, datetime.now(timezone.utc))
            self._samples[name].append(sample)

            # Update histogram metrics
            samples = [s.value for s in self._samples[name]]
            if samples:
                self._metrics[name] = {
                    "type": "histogram",
                    "count": len(samples),
                    "sum": sum(samples),
                    "min": min(samples),
                    "max": max(samples),
                    "avg": statistics.mean(samples),
                    "p50": statistics.median(samples),
                    "p90": self._percentile(samples, 0.90),
                    "p95": self._percentile(samples, 0.95),
                    "p99": self._percentile(samples, 0.99),
                    "tags": tags,
                    "last_updated": datetime.now(timezone.utc),
                }

    def record_timer(self, name: str, duration: float, tags: Dict[str, str] = None) -> None:
        """Record a timer metric."""
        self.record_histogram(name, duration, tags)

    def add_to_set(self, name: str, value: str, tags: Dict[str, str] = None) -> None:
        """Add value to a set metric."""
        with self._lock:
            tags = tags or {}
            if name not in self._metrics:
                self._metrics[name] = {
                    "type": "set",
                    "values": set(),
                    "tags": tags,
                    "created_at": datetime.now(timezone.utc),
                }

            self._metrics[name]["values"].add(value)
            self._metrics[name]["count"] = len(self._metrics[name]["values"])
            self._metrics[name]["last_updated"] = datetime.now(timezone.utc)

    def get_metric(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a metric by name."""
        with self._lock:
            return self._metrics.get(name)

    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all metrics."""
        with self._lock:
            return dict(self._metrics)

    def clear_metric(self, name: str) -> None:
        """Clear a specific metric."""
        with self._lock:
            self._metrics.pop(name, None)
            self._samples.pop(name, None)

    def clear_all(self) -> None:
        """Clear all metrics."""
        with self._lock:
            self._metrics.clear()
            self._samples.clear()

    def cleanup_old_samples(self, max_age_seconds: float) -> None:
        """Clean up old samples from histograms."""
        with self._lock:
            cutoff_time = datetime.now(timezone.utc) - timedelta(seconds=max_age_seconds)

            for name, samples in self._samples.items():
                # Remove old samples
                while samples and samples[0].timestamp < cutoff_time:
                    samples.popleft()

    def _percentile(self, values: List[float], percentile: float) -> float:
        """Calculate percentile of values."""
        if not values:
            return 0.0

        sorted_values = sorted(values)
        index = int(percentile * len(sorted_values))
        if index >= len(sorted_values):
            index = len(sorted_values) - 1
        return sorted_values[index]


class SystemMetricsCollector:
    """Collects system-level metrics like CPU, memory, etc."""

    def __init__(self):
        self.logger = get_logger(__name__)
        self._last_cpu_times = None
        self._last_network_stats = None

    def collect_system_metrics(self) -> Dict[str, float]:
        """Collect current system metrics."""
        metrics = {}

        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=None)
            metrics["system_cpu_usage_percent"] = cpu_percent

            # Memory metrics
            memory = psutil.virtual_memory()
            metrics["system_memory_usage_percent"] = memory.percent
            metrics["system_memory_used_bytes"] = memory.used
            metrics["system_memory_available_bytes"] = memory.available

            # Disk metrics
            disk = psutil.disk_usage("/")
            metrics["system_disk_usage_percent"] = (disk.used / disk.total) * 100
            metrics["system_disk_used_bytes"] = disk.used
            metrics["system_disk_free_bytes"] = disk.free

            # Network metrics
            network = psutil.net_io_counters()
            if self._last_network_stats:
                bytes_sent_delta = network.bytes_sent - self._last_network_stats.bytes_sent
                bytes_recv_delta = network.bytes_recv - self._last_network_stats.bytes_recv
                metrics["system_network_bytes_sent_rate"] = max(0, bytes_sent_delta)
                metrics["system_network_bytes_recv_rate"] = max(0, bytes_recv_delta)

            self._last_network_stats = network

            # Process metrics
            process = psutil.Process()
            metrics["process_memory_rss_bytes"] = process.memory_info().rss
            metrics["process_memory_vms_bytes"] = process.memory_info().vms
            metrics["process_cpu_percent"] = process.cpu_percent()
            metrics["process_num_threads"] = process.num_threads()
            metrics["process_num_fds"] = process.num_fds() if hasattr(process, "num_fds") else 0

            # Python GC metrics
            gc_stats = gc.get_stats()
            for i, stat in enumerate(gc_stats):
                metrics[f"python_gc_generation_{i}_collections"] = stat.get("collections", 0)
                metrics[f"python_gc_generation_{i}_collected"] = stat.get("collected", 0)
                metrics[f"python_gc_generation_{i}_uncollectable"] = stat.get("uncollectable", 0)

            return metrics

        except Exception as e:
            self.logger.error(f"Failed to collect system metrics: {str(e)}")
            return {}


class MetricsCollector:
    """
    Advanced Metrics Collection System for the AI Assistant.

    This collector provides comprehensive metrics gathering including:
    - System performance metrics (CPU, memory, disk, network)
    - Application metrics (request counts, response times, error rates)
    - Business metrics (user interactions, workflow completions)
    - Component health metrics (service availability, processing times)
    - Custom metrics from plugins and extensions

    Features:
    - Multiple metric types (counters, gauges, histograms, timers)
    - Flexible tagging and labeling
    - Multiple export backends (Prometheus, StatsD, JSON)
    - Real-time metric streaming
    - Automatic cleanup and retention
    - Performance-optimized collection
    - Thread-safe operations
    """

    def __init__(self, container: Container):
        """
        Initialize the metrics collector.

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

        # Metric storage and configuration
        self.metric_store = MetricStore(max_samples=self.config.get("metrics.max_samples", 10000))
        self.system_collector = SystemMetricsCollector()

        # Metric configurations
        self.metric_configs: Dict[str, MetricConfig] = {}
        self.export_backends: List[MetricBackend] = []

        # Performance tracking
        self.collection_stats = {
            "total_metrics": 0,
            "collection_errors": 0,
            "export_errors": 0,
            "last_collection_time": None,
            "collection_duration": 0.0,
        }

        # Configuration
        self.collection_interval = self.config.get("metrics.collection_interval", 30.0)
        self.export_interval = self.config.get("metrics.export_interval", 60.0)
        self.enable_system_metrics = self.config.get("metrics.enable_system_metrics", True)
        self.enable_realtime_export = self.config.get("metrics.enable_realtime_export", False)
        self.retention_seconds = self.config.get("metrics.retention_seconds", 3600.0)

        # Background tasks
        self.collection_task: Optional[asyncio.Task] = None
        self.export_task: Optional[asyncio.Task] = None
        self.cleanup_task: Optional[asyncio.Task] = None

        # Thread safety
        self._shutdown_event = asyncio.Event()

        # Setup backends and register metrics
        self._setup_export_backends()
        self._register_builtin_metrics()

        # Register health check
        self.health_check.register_component("metrics_collector", self._health_check_callback)

        self.logger.info("MetricsCollector initialized successfully")

    def _setup_export_backends(self) -> None:
        """Setup metric export backends based on configuration."""
        try:
            backends_config = self.config.get("metrics.backends", ["json"])

            for backend_type in backends_config:
                if backend_type == "prometheus":
                    endpoint = self.config.get(
                        "metrics.prometheus.endpoint", "http://localhost:9090"
                    )
                    self.export_backends.append(PrometheusBackend(endpoint))
                elif backend_type == "statsd":
                    host = self.config.get("metrics.statsd.host", "localhost")
                    port = self.config.get("metrics.statsd.port", 8125)
                    self.export_backends.append(StatsDBackend(host, port))
                elif backend_type == "json":
                    file_path = self.config.get(
                        "metrics.json.file_path", "data/metrics/metrics.json"
                    )
                    self.export_backends.append(JSONBackend(file_path))

            self.logger.info(f"Configured {len(self.export_backends)} export backends")

        except Exception as e:
            self.logger.error(f"Failed to setup export backends: {str(e)}")

    def _register_builtin_metrics(self) -> None:
        """Register built-in metrics for core system components."""
        try:
            # System metrics
            if self.enable_system_metrics:
                system_metrics = [
                    ("system_cpu_usage_percent", MetricType.GAUGE, "System CPU usage percentage"),
                    (
                        "system_memory_usage_percent",
                        MetricType.GAUGE,
                        "System memory usage percentage",
                    ),
                    ("system_disk_usage_percent", MetricType.GAUGE, "System disk usage percentage"),
                    ("process_memory_rss_bytes", MetricType.GAUGE, "Process RSS memory in bytes"),
                    ("process_cpu_percent", MetricType.GAUGE, "Process CPU usage percentage"),
                ]

                for name, metric_type, description in system_metrics:
                    self.register_metric(
                        name,
                        metric_type,
                        description,
                        unit="percent" if "percent" in name else "bytes",
                    )

            # Core engine metrics
            core_metrics = [
                ("engine_requests_total", MetricType.COUNTER, "Total engine requests"),
                (
                    "engine_processing_duration_seconds",
                    MetricType.HISTOGRAM,
                    "Engine processing duration",
                ),
                ("engine_active_sessions", MetricType.GAUGE, "Number of active sessions"),
                ("engine_errors_total", MetricType.COUNTER, "Total engine errors"),
                ("engine_component_health", MetricType.GAUGE, "Component health status"),
            ]

            for name, metric_type, description in core_metrics:
                unit = "seconds" if "duration" in name else ""
                self.register_metric(name, metric_type, description, unit=unit)

            # Workflow metrics
            workflow_metrics = [
                ("workflow_executions_total", MetricType.COUNTER, "Total workflow executions"),
                (
                    "workflow_executions_successful",
                    MetricType.COUNTER,
                    "Successful workflow executions",
                ),
                ("workflow_executions_failed", MetricType.COUNTER, "Failed workflow executions"),
                (
                    "workflow_execution_duration_seconds",
                    MetricType.HISTOGRAM,
                    "Workflow execution duration",
                ),
                ("active_workflows", MetricType.GAUGE, "Number of active workflows"),
                ("workflow_steps_executed", MetricType.COUNTER, "Total workflow steps executed"),
                ("workflow_step_duration_seconds", MetricType.HISTOGRAM, "Workflow step duration"),
            ]

            for name, metric_type, description in workflow_metrics:
                unit = "seconds" if "duration" in name else ""
                self.register_metric(name, metric_type, description, unit=unit)

            # Session metrics
            session_metrics = [
                ("sessions_created_total", MetricType.COUNTER, "Total sessions created"),
                ("sessions_ended_total", MetricType.COUNTER, "Total sessions ended"),
                ("sessions_expired_total", MetricType.COUNTER, "Total sessions expired"),
                ("active_sessions", MetricType.GAUGE, "Number of active sessions"),
                ("session_duration_seconds", MetricType.HISTOGRAM, "Session duration"),
                ("session_memory_usage_mb", MetricType.HISTOGRAM, "Session memory usage"),
                ("session_errors_total", MetricType.COUNTER, "Total session errors"),
            ]

            for name, metric_type, description in session_metrics:
                unit = "seconds" if "duration" in name else "megabytes" if "mb" in name else ""
                self.register_metric(name, metric_type, description, unit=unit)

            # Interaction metrics
            interaction_metrics = [
                ("interactions_total", MetricType.COUNTER, "Total user interactions"),
                ("interactions_successful", MetricType.COUNTER, "Successful interactions"),
                ("interactions_failed", MetricType.COUNTER, "Failed interactions"),
                ("interaction_duration_seconds", MetricType.HISTOGRAM, "Interaction duration"),
                (
                    "response_generation_time_seconds",
                    MetricType.HISTOGRAM,
                    "Response generation time",
                ),
                ("active_interactions", MetricType.GAUGE, "Number of active interactions"),
                ("user_messages_total", MetricType.COUNTER, "Total user messages"),
                ("assistant_responses_total", MetricType.COUNTER, "Total assistant responses"),
            ]

            for name, metric_type, description in interaction_metrics:
                unit = "seconds" if "duration" in name or "time" in name else ""
                self.register_metric(name, metric_type, description, unit=unit)

            # Plugin metrics
            plugin_metrics = [
                ("plugins_loaded_total", MetricType.COUNTER, "Total plugins loaded"),
                ("plugins_failed_total", MetricType.COUNTER, "Total plugin failures"),
                ("plugins_active", MetricType.GAUGE, "Number of active plugins"),
                ("plugin_load_duration_seconds", MetricType.HISTOGRAM, "Plugin load duration"),
                (
                    "plugin_execution_duration_seconds",
                    MetricType.HISTOGRAM,
                    "Plugin execution duration",
                ),
            ]

            for name, metric_type, description in plugin_metrics:
                unit = "seconds" if "duration" in name else ""
                self.register_metric(name, metric_type, description, unit=unit)

            # Component metrics
            component_metrics = [
                (
                    "component_registrations_total",
                    MetricType.COUNTER,
                    "Total component registrations",
                ),
                (
                    "component_initializations_total",
                    MetricType.COUNTER,
                    "Total component initializations",
                ),
                ("component_failures_total", MetricType.COUNTER, "Total component failures"),
                (
                    "component_initialization_duration_seconds",
                    MetricType.HISTOGRAM,
                    "Component initialization duration",
                ),
                ("components_running", MetricType.GAUGE, "Number of running components"),
                ("components_failed", MetricType.GAUGE, "Number of failed components"),
            ]

            for name, metric_type, description in component_metrics:
                unit = "seconds" if "duration" in name else ""
                self.register_metric(name, metric_type, description, unit=unit)

            self.logger.info(f"Registered {len(self.metric_configs)} built-in metrics")

        except Exception as e:
            self.logger.error(f"Failed to register built-in metrics: {str(e)}")

    async def initialize(self) -> None:
        """Initialize the metrics collector."""
        try:
            # Initialize export backends
            for backend in self.export_backends:
                await backend.initialize()

            # Register event handlers
            await self._register_event_handlers()

            # Start background tasks
            if self.collection_interval > 0:
                self.collection_task = asyncio.create_task(self._collection_loop())

            if self.export_interval > 0:
                self.export_task = asyncio.create_task(self._export_loop())

            self.cleanup_task = asyncio.create_task(self._cleanup_loop())

            self.logger.info("MetricsCollector initialization completed")

        except Exception as e:
            self.logger.error(f"Failed to initialize MetricsCollector: {str(e)}")
            raise MetricError(f"Initialization failed: {str(e)}")

    async def _register_event_handlers(self) -> None:
        """Register event handlers for automatic metric collection."""
        # Engine events
        self.event_bus.subscribe("processing_started", self._handle_processing_started)
        self.event_bus.subscribe("processing_completed", self._handle_processing_completed)
        self.event_bus.subscribe("engine_started", self._handle_engine_started)

        # Workflow events
        self.event_bus.subscribe("workflow_started", self._handle_workflow_started)
        self.event_bus.subscribe("workflow_completed", self._handle_workflow_completed)
        self.event_bus.subscribe("workflow_failed", self._handle_workflow_failed)
        self.event_bus.subscribe("workflow_step_completed", self._handle_workflow_step_completed)

        # Session events
        self.event_bus.subscribe("session_started", self._handle_session_started)
        self.event_bus.subscribe("session_ended", self._handle_session_ended)
        self.event_bus.subscribe("session_expired", self._handle_session_expired)

        # Interaction events
        self.event_bus.subscribe("user_interaction_started", self._handle_interaction_started)
        self.event_bus.subscribe("user_interaction_completed", self._handle_interaction_completed)
        self.event_bus.subscribe("user_interaction_failed", self._handle_interaction_failed)

        # Plugin events
        self.event_bus.subscribe("plugin_loaded", self._handle_plugin_loaded)
        self.event_bus.subscribe("plugin_unloaded", self._handle_plugin_unloaded)
        self.event_bus.subscribe("plugin_error", self._handle_plugin_error)

        # Component events
        self.event_bus.subscribe("component_registered", self._handle_component_registered)
        self.event_bus.subscribe("component_initialized", self._handle_component_initialized)
        self.event_bus.subscribe("component_failed", self._handle_component_failed)

        # System events
        self.event_bus.subscribe("component_health_changed", self._handle_component_health_changed)

    def register_metric(
        self,
        name: str,
        metric_type: MetricType,
        description: str = "",
        unit: str = "",
        tags: Optional[Dict[str, str]] = None,
        buckets: Optional[List[float]] = None,
        quantiles: Optional[List[float]] = None,
        **kwargs,
    ) -> None:
        """
        Register a new metric configuration.

        Args:
            name: Metric name
            metric_type: Type of metric
            description: Metric description
            unit: Unit of measurement
            tags: Default tags
            buckets: Histogram buckets
            quantiles: Summary quantiles
            **kwargs: Additional configuration
        """
        if name in self.metric_configs:
            self.logger.warning(f"Metric {name} already registered, updating configuration")

        config = MetricConfig(
            name=name,
            metric_type=metric_type,
            description=description,
            unit=unit,
            tags=tags or {},
            buckets=buckets,
            quantiles=quantiles,
            **kwargs,
        )

        self.metric_configs[name] = config
        self.logger.debug(f"Registered metric: {name} ({metric_type.value})")

    def increment(
        self, name: str, value: float = 1.0, tags: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Increment a counter metric.

        Args:
            name: Metric name
            value: Increment value
            tags: Additional tags
        """
        try:
            config = self.metric_configs.get(name)
            if config and not config.enabled:
                return

            # Apply sampling if configured
            if config and config.sample_rate < 1.0:
                import random

                if random.random() > config.sample_rate:
                    return

            # Merge tags
            final_tags = {}
            if config:
                final_tags.update(config.tags)
            if tags:
                final_tags.update(tags)

            self.metric_store.record_counter(name, value, final_tags)
            self.collection_stats["total_metrics"] += 1

            # Emit event for real-time export
            if self.enable_realtime_export:
                asyncio.create_task(
                    self.event_bus.emit(
                        MetricRecorded(
                            metric_name=name, metric_type="counter", value=value, tags=final_tags
                        )
                    )
                )

        except Exception as e:
            self.collection_stats["collection_errors"] += 1
            self.logger.error(f"Failed to increment metric {name}: {str(e)}")

    def set(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """
        Set a gauge metric value.

        Args:
            name: Metric name
            value: Metric value
            tags: Additional tags
        """
        try:
            config = self.metric_configs.get(name)
            if config and not config.enabled:
                return

            # Merge tags
            final_tags = {}
            if config:
                final_tags.update(config.tags)
            if tags:
                final_tags.update(tags)

            self.metric_store.set_gauge(name, value, final_tags)
            self.collection_stats["total_metrics"] += 1

            # Emit event for real-time export
            if self.enable_realtime_export:
                asyncio.create_task(
                    self.event_bus.emit(
                        MetricRecorded(
                            metric_name=name, metric_type="gauge", value=value, tags=final_tags
                        )
                    )
                )

        except Exception as e:
            self.collection_stats["collection_errors"] += 1
            self.logger.error(f"Failed to set metric {name}: {str(e)}")

    def record(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """
        Record a histogram value.

        Args:
            name: Metric name
            value: Value to record
            tags: Additional tags
        """
        try:
            config = self.metric_configs.get(name)
            if config and not config.enabled:
                return

            # Apply sampling if configured
            if config and config.sample_rate < 1.0:
                import random

                if random.random() > config.sample_rate:
                    return

            # Merge tags
            final_tags = {}
            if config:
                final_tags.update(config.tags)
            if tags:
                final_tags.update(tags)

            self.metric_store.record_histogram(name, value, final_tags)
            self.collection_stats["total_metrics"] += 1

            # Emit event for real-time export
            if self.enable_realtime_export:
                asyncio.create_task(
                    self.event_bus.emit(
                        MetricRecorded(
                            metric_name=name, metric_type="histogram", value=value, tags=final_tags
                        )
                    )
                )

        except Exception as e:
            self.collection_stats["collection_errors"] += 1
            self.logger.error(f"Failed to record metric {name}: {str(e)}")

    def time(self, name: str, duration: float, tags: Optional[Dict[str, str]] = None) -> None:
        """
        Record a timing metric.

        Args:
            name: Metric name
            duration: Duration in seconds
            tags: Additional tags
        """
        self.record(name, duration, tags)

    @asynccontextmanager
    async def timer(self, name: str, tags: Optional[Dict[str, str]] = None):
        """
        Context manager for timing operations.

        Args:
            name: Metric name
            tags: Additional tags
        """
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.time(name, duration, tags)

    def add_to_set(self, name: str, value: str, tags: Optional[Dict[str, str]] = None) -> None:
        """
        Add value to a set metric.

        Args:
            name: Metric name
            value: Value to add
            tags: Additional tags
        """
        try:
            config = self.metric_configs.get(name)
            if config and not config.enabled:
                return

            # Merge tags
            final_tags = {}
            if config:
                final_tags.update(config.tags)
            if tags:
                final_tags.update(tags)

            self.metric_store.add_to_set(name, value, final_tags)
            self.collection_stats["total_metrics"] += 1

        except Exception as e:
            self.collection_stats["collection_errors"] += 1
            self.logger.error(f"Failed to add to set metric {name}: {str(e)}")

    def get_metric(self, name: str) -> Optional[Dict[str, Any]]:
        """Get current value of a metric."""
        return self.metric_store.get_metric(name)

    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all current metrics."""
        return self.metric_store.get_all_metrics()

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        return dict(self.collection_stats)

    async def export_metrics(self) -> None:
        """Export metrics to all configured backends."""
        try:
            metrics = self.get_all_metrics()

            if not metrics:
                return

            # Export to all backends
            export_tasks = []
            for backend in self.export_backends:
                task = asyncio.create_task(backend.export_metrics(metrics))
                export_tasks.append(task)

            # Wait for all exports to complete
            results = await asyncio.gather(*export_tasks, return_exceptions=True)

            # Check for export errors
            export_errors = sum(1 for result in results if isinstance(result, Exception))
            self.collection_stats["export_errors"] += export_errors

            if export_errors == 0:
                await self.event_bus.emit(
                    MetricExported(
                        backend_count=len(self.export_backends),
                        metric_count=len(metrics),
                        timestamp=datetime.now(timezone.utc),
                    )
                )

            self.logger.debug(
                f"Exported {len(metrics)} metrics to {len(self.export_backends)} backends"
            )

        except Exception as e:
            self.collection_stats["export_errors"] += 1
            self.logger.error(f"Failed to export metrics: {str(e)}")

    async def _collection_loop(self) -> None:
        """Background task for periodic metric collection."""
        while not self._shutdown_event.is_set():
            try:
                start_time = time.time()

                # Collect system metrics
                if self.enable_system_metrics:
                    system_metrics = self.system_collector.collect_system_metrics()
                    for name, value in system_metrics.items():
                        self.set(name, value)

                # Update collection stats
                collection_duration = time.time() - start_time
                self.collection_stats["last_collection_time"] = datetime.now(timezone.utc)
                self.collection_stats["collection_duration"] = collection_duration

                await asyncio.sleep(self.collection_interval)

            except Exception as e:
                self.collection_stats["collection_errors"] += 1
                self.logger.error(f"Error in collection loop: {str(e)}")
                await asyncio.sleep(self.collection_interval)

    async def _export_loop(self) -> None:
        """Background task for periodic metric export."""
        while not self._shutdown_event.is_set():
            try:
                await self.export_metrics()
                await asyncio.sleep(self.export_interval)

            except Exception as e:
                self.collection_stats["export_errors"] += 1
                self.logger.error(f"Error in export loop: {str(e)}")
                await asyncio.sleep(self.export_interval)

    async def _cleanup_loop(self) -> None:
        """Background task for cleaning up old metrics."""
        while not self._shutdown_event.is_set():
            try:
                # Clean up old samples
                self.metric_store.cleanup_old_samples(self.retention_seconds)

                # Clean up collection stats periodically
                await asyncio.sleep(300)  # Clean up every 5 minutes

            except Exception as e:
                self.logger.error(f"Error in cleanup loop: {str(e)}")
                await asyncio.sleep(300)

    # Event handlers for automatic metric collection
    async def _handle_processing_started(self, event) -> None:
        """Handle processing started events."""
        self.increment("engine_requests_total", tags={"session_id": event.session_id})

    async def _handle_processing_completed(self, event) -> None:
        """Handle processing completed events."""
        self.record("engine_processing_duration_seconds", event.processing_time)
        if hasattr(event, "confidence"):
            self.record("engine_confidence_score", event.confidence)

    async def _handle_engine_started(self, event) -> None:
        """Handle engine started events."""
        self.increment("engine_starts_total")
        self.set("engine_components_loaded", event.components_loaded)

    async def _handle_workflow_started(self, event) -> None:
        """Handle workflow started events."""
        self.increment("workflow_executions_total", tags={"workflow_id": event.workflow_id})

    async def _handle_workflow_completed(self, event) -> None:
        """Handle workflow completed events."""
        self.increment("workflow_executions_successful", tags={"workflow_id": event.workflow_id})
        self.record("workflow_execution_duration_seconds", event.execution_time)
        self.set("workflow_steps_completed", event.steps_completed)

    async def _handle_workflow_failed(self, event) -> None:
        """Handle workflow failed events."""
        self.increment("workflow_executions_failed", tags={"workflow_id": event.workflow_id})

    async def _handle_workflow_step_completed(self, event) -> None:
        """Handle workflow step completed events."""
        self.increment(
            "workflow_steps_executed",
            tags={"workflow_id": event.workflow_id, "step_type": event.step_type},
        )
        self.record("workflow_step_duration_seconds", event.execution_time)

    async def _handle_session_started(self, event) -> None:
        """Handle session started events."""
        self.increment("sessions_created_total", tags={"user_id": event.user_id or "anonymous"})

    async def _handle_session_ended(self, event) -> None:
        """Handle session ended events."""
        self.increment("sessions_ended_total", tags={"reason": event.reason})
        self.record("session_duration_seconds", event.duration)
        self.set("session_interaction_count", event.interaction_count)

    async def _handle_session_expired(self, event) -> None:
        """Handle session expired events."""
        self.increment("sessions_expired_total")

    async def _handle_interaction_started(self, event) -> None:
        """Handle interaction started events."""
        self.increment(
            "interactions_total",
            tags={
                "interaction_mode": event.interaction_mode,
                "user_id": event.user_id or "anonymous",
            },
        )

    async def _handle_interaction_completed(self, event) -> None:
        """Handle interaction completed events."""
        self.increment("interactions_successful")
        self.record("interaction_duration_seconds", event.duration)
        self.set("interaction_message_count", event.message_count)

    async def _handle_interaction_failed(self, event) -> None:
        """Handle interaction failed events."""
        self.increment("interactions_failed", tags={"error_type": event.error_type})

    async def _handle_plugin_loaded(self, event) -> None:
        """Handle plugin loaded events."""
        self.increment("plugins_loaded_total", tags={"plugin_type": event.plugin_type})
        self.record("plugin_load_duration_seconds", event.load_time)

    async def _handle_plugin_unloaded(self, event) -> None:
        """Handle plugin unloaded events."""
        self.increment("plugins_unloaded_total", tags={"plugin_type": event.plugin_type})

    async def _handle_plugin_error(self, event) -> None:
        """Handle plugin error events."""
        self.increment("plugins_failed_total", tags={"error_type": event.error_type})

    async def _handle_component_registered(self, event) -> None:
        """Handle component registered events."""
        self.increment(
            "component_registrations_total", tags={"component_type": event.component_type}
        )

    async def _handle_component_initialized(self, event) -> None:
        """Handle component initialized events."""
        self.increment("component_initializations_total")
        self.record("component_initialization_duration_seconds", event.initialization_time)

    async def _handle_component_failed(self, event) -> None:
        """Handle component failed events."""
        self.increment("component_failures_total", tags={"error_type": event.error_type})

    async def _handle_component_health_changed(self, event) -> None:
        """Handle component health changed events."""
        health_value = 1.0 if event.healthy else 0.0
        self.set("component_health", health_value, tags={"component": event.component})

    async def _health_check_callback(self) -> Dict[str, Any]:
        """Health check callback for the metrics collector."""
        try:
            metrics_count = len(self.get_all_metrics())
            collection_errors = self.collection_stats["collection_errors"]
            export_errors = self.collection_stats["export_errors"]

            # Calculate health score
            total_operations = self.collection_stats["total_metrics"]
            total_errors = collection_errors + export_errors

            if total_operations > 0:
                error_rate = total_errors / total_operations
                health_score = max(0.0, 1.0 - error_rate)
            else:
                health_score = 1.0

            status = (
                "healthy"
                if health_score >= 0.9
                else "degraded" if health_score >= 0.5 else "unhealthy"
            )

            return {
                "status": status,
                "health_score": health_score,
                "metrics_count": metrics_count,
                "total_metrics_collected": self.collection_stats["total_metrics"],
                "collection_errors": collection_errors,
                "export_errors": export_errors,
                "backends_configured": len(self.export_backends),
                "last_collection": (
                    self.collection_stats["last_collection_time"].isoformat()
                    if self.collection_stats["last_collection_time"]
                    else None
                ),
            }

        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    async def cleanup(self) -> None:
        """Cleanup resources and stop background tasks."""
        try:
            self._shutdown_event.set()

            # Cancel background tasks
            for task in [self.collection_task, self.export_task, self.cleanup_task]:
                if task and not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

            # Export final metrics
            await self.export_metrics()

            # Cleanup backends
            for backend in self.export_backends:
                await backend.cleanup()

            self.logger.info("MetricsCollector cleanup completed")

        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")

    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            if hasattr(self, "_shutdown_event") and not self._shutdown_event.is_set():
                self.logger.warning("MetricsCollector destroyed without proper cleanup")
        except Exception:
            pass  # Ignore cleanup errors in destructor
