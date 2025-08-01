"""
Advanced Dashboard Management System for AI Assistant Observability
Author: Drmusab
Last Modified: 2025-01-26 10:07:02 UTC

This module provides comprehensive dashboard management for monitoring and observability
of the AI assistant system, integrating with all core components including the core engine,
component manager, workflow orchestrator, interaction handler, session manager, and plugin manager.
"""

import base64
import hashlib
import inspect
import io
import json
import logging
import threading
import time
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
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Set, Type, TypeVar, Union

import asyncio
import numpy as np
import pandas as pd

# Web framework and visualization
try:
    import plotly.express as px
    import plotly.graph_objects as go
    import plotly.utils
    from plotly.io import to_html, to_image
    from plotly.subplots import make_subplots
except ImportError:
    go = px = make_subplots = to_html = to_image = None

try:
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.backends.backend_agg import FigureCanvasAgg
except ImportError:
    plt = mdates = FigureCanvasAgg = sns = None

try:
    import markdown
    from jinja2 import Environment, FileSystemLoader, Template
except ImportError:
    Template = Environment = FileSystemLoader = markdown = None

from src.assistant.component_manager import EnhancedComponentManager

# Assistant components
from src.assistant.core_engine import EnhancedCoreEngine
from src.assistant.interaction_handler import InteractionHandler
from src.assistant.plugin_manager import EnhancedPluginManager
from src.assistant.session_manager import EnhancedSessionManager
from src.assistant.workflow_orchestrator import WorkflowOrchestrator

# Core imports
from src.core.config.loader import ConfigLoader
from src.core.dependency_injection import Container
from src.core.error_handling import ErrorHandler, handle_exceptions
from src.core.events.event_bus import EventBus
from src.core.events.event_types import (
    AlertTriggered,
    ComponentHealthChanged,
    DashboardCreated,
    DashboardDeleted,
    DashboardExported,
    DashboardShared,
    DashboardUpdated,
    DashboardViewed,
    ErrorOccurred,
    MetricThresholdExceeded,
    PluginLoaded,
    SessionEnded,
    SystemStateChanged,
    UserInteractionCompleted,
    WorkflowCompleted,
)
from src.core.health_check import HealthCheck
from src.core.security.authentication import AuthenticationManager
from src.core.security.authorization import AuthorizationManager
from src.observability.logging.config import get_logger
from src.observability.monitoring.alerting import AlertManager

# Observability
from src.observability.monitoring.metrics import MetricsCollector
from src.observability.monitoring.tracing import TraceManager

# Type definitions
T = TypeVar("T")


class DashboardType(Enum):
    """Types of dashboards available in the system."""

    OVERVIEW = "overview"  # System overview
    COMPONENT_HEALTH = "component_health"  # Component monitoring
    WORKFLOW_ANALYTICS = "workflow_analytics"  # Workflow execution
    USER_INTERACTIONS = "user_interactions"  # User interaction patterns
    SESSION_MONITORING = "session_monitoring"  # Session management
    PLUGIN_OBSERVABILITY = "plugin_observability"  # Plugin system
    PERFORMANCE_METRICS = "performance_metrics"  # Performance monitoring
    SECURITY_AUDIT = "security_audit"  # Security monitoring
    ERROR_TRACKING = "error_tracking"  # Error analysis
    RESOURCE_USAGE = "resource_usage"  # Resource utilization
    REAL_TIME_MONITORING = "real_time_monitoring"  # Live monitoring
    CUSTOM = "custom"  # Custom dashboards


class ChartType(Enum):
    """Types of charts supported by the dashboard system."""

    LINE_CHART = "line_chart"
    BAR_CHART = "bar_chart"
    PIE_CHART = "pie_chart"
    SCATTER_PLOT = "scatter_plot"
    HEATMAP = "heatmap"
    HISTOGRAM = "histogram"
    BOX_PLOT = "box_plot"
    GAUGE = "gauge"
    TABLE = "table"
    TIMELINE = "timeline"
    TREEMAP = "treemap"
    SANKEY = "sankey"
    NETWORK_GRAPH = "network_graph"
    MAP = "map"
    CANDLESTICK = "candlestick"


class ExportFormat(Enum):
    """Export formats for dashboards."""

    PDF = "pdf"
    PNG = "png"
    SVG = "svg"
    HTML = "html"
    JSON = "json"
    CSV = "csv"
    EXCEL = "excel"


class RefreshMode(Enum):
    """Dashboard refresh modes."""

    MANUAL = "manual"
    AUTO = "auto"
    REAL_TIME = "real_time"
    SCHEDULED = "scheduled"


class AccessLevel(Enum):
    """Dashboard access levels."""

    PUBLIC = "public"
    AUTHENTICATED = "authenticated"
    PRIVATE = "private"
    RESTRICTED = "restricted"


@dataclass
class ChartConfiguration:
    """Configuration for individual charts."""

    chart_id: str
    chart_type: ChartType
    title: str
    description: Optional[str] = None

    # Data configuration
    data_source: str = "metrics"
    query: Optional[str] = None
    filters: Dict[str, Any] = field(default_factory=dict)
    aggregation: Optional[str] = None

    # Visual configuration
    width: int = 12  # Bootstrap grid system (1-12)
    height: int = 400
    color_scheme: str = "plotly"
    theme: str = "light"

    # Chart-specific options
    x_axis: Optional[str] = None
    y_axis: Optional[str] = None
    group_by: Optional[str] = None
    sort_by: Optional[str] = None
    limit: Optional[int] = None

    # Interactive features
    interactive: bool = True
    downloadable: bool = True
    drilldown_enabled: bool = False

    # Refresh settings
    refresh_interval: int = 30  # seconds
    cache_duration: int = 300  # seconds

    # Styling
    custom_css: Optional[str] = None
    custom_options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DashboardLayout:
    """Dashboard layout configuration."""

    layout_id: str
    name: str
    description: Optional[str] = None

    # Grid configuration
    rows: int = 12
    columns: int = 12

    # Charts and positioning
    charts: List[ChartConfiguration] = field(default_factory=list)
    chart_positions: Dict[str, Dict[str, int]] = field(
        default_factory=dict
    )  # chart_id -> {row, col, width, height}

    # Layout styling
    background_color: str = "#ffffff"
    border_style: str = "none"
    padding: int = 10
    gap: int = 5

    # Responsive design
    responsive: bool = True
    mobile_columns: int = 1
    tablet_columns: int = 2


@dataclass
class DashboardMetadata:
    """Metadata for dashboard instances."""

    dashboard_id: str
    name: str
    dashboard_type: DashboardType
    description: Optional[str] = None

    # Ownership and access
    created_by: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    access_level: AccessLevel = AccessLevel.PRIVATE
    allowed_users: Set[str] = field(default_factory=set)
    allowed_roles: Set[str] = field(default_factory=set)

    # Layout and configuration
    layout: DashboardLayout = field(
        default_factory=lambda: DashboardLayout("default", "Default Layout")
    )

    # Refresh and caching
    refresh_mode: RefreshMode = RefreshMode.AUTO
    refresh_interval: int = 30
    cache_enabled: bool = True
    cache_duration: int = 300

    # Features
    real_time_enabled: bool = False
    export_enabled: bool = True
    sharing_enabled: bool = True
    alerts_enabled: bool = True

    # Styling
    theme: str = "light"
    custom_css: Optional[str] = None

    # Version and tags
    version: str = "1.0.0"
    tags: Set[str] = field(default_factory=set)
    category: Optional[str] = None


@dataclass
class DashboardInstance:
    """Runtime dashboard instance."""

    metadata: DashboardMetadata

    # Runtime state
    is_active: bool = False
    last_accessed: Optional[datetime] = None
    view_count: int = 0
    active_viewers: Set[str] = field(default_factory=set)

    # Data and rendering
    cached_data: Dict[str, Any] = field(default_factory=dict)
    rendered_charts: Dict[str, str] = field(default_factory=dict)  # chart_id -> HTML
    last_updated: Optional[datetime] = None

    # Performance metrics
    render_time: float = 0.0
    data_fetch_time: float = 0.0
    cache_hit_rate: float = 0.0

    # Errors and warnings
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class DashboardError(Exception):
    """Custom exception for dashboard operations."""

    def __init__(
        self,
        message: str,
        dashboard_id: Optional[str] = None,
        chart_id: Optional[str] = None,
        error_code: Optional[str] = None,
    ):
        super().__init__(message)
        self.dashboard_id = dashboard_id
        self.chart_id = chart_id
        self.error_code = error_code
        self.timestamp = datetime.now(timezone.utc)


class DataProvider(ABC):
    """Abstract base class for dashboard data providers."""

    @abstractmethod
    async def fetch_data(self, query: str, filters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Fetch data for dashboard charts."""
        pass

    @abstractmethod
    def supports_real_time(self) -> bool:
        """Check if provider supports real-time data."""
        pass

    @abstractmethod
    async def subscribe_to_updates(self, callback: Callable) -> str:
        """Subscribe to real-time data updates."""
        pass


class MetricsDataProvider(DataProvider):
    """Data provider for metrics collection."""

    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
        self.logger = get_logger(__name__)
        self._subscribers: Dict[str, Callable] = {}

    async def fetch_data(self, query: str, filters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Fetch metrics data."""
        try:
            # Parse query to determine metric and aggregation
            parts = query.split("|")
            metric_name = parts[0].strip()

            # Get metric data
            data = await self.metrics.get_metric_data(
                metric_name=metric_name,
                filters=filters or {},
                start_time=filters.get("start_time") if filters else None,
                end_time=filters.get("end_time") if filters else None,
            )

            return {
                "data": data,
                "metric_name": metric_name,
                "timestamp": datetime.now(timezone.utc),
                "filters": filters or {},
            }

        except Exception as e:
            self.logger.error(f"Failed to fetch metrics data: {str(e)}")
            return {"data": [], "error": str(e)}

    def supports_real_time(self) -> bool:
        """Metrics support real-time updates."""
        return True

    async def subscribe_to_updates(self, callback: Callable) -> str:
        """Subscribe to metric updates."""
        subscription_id = str(uuid.uuid4())
        self._subscribers[subscription_id] = callback
        return subscription_id


class ComponentHealthDataProvider(DataProvider):
    """Data provider for component health monitoring."""

    def __init__(self, component_manager: EnhancedComponentManager):
        self.component_manager = component_manager
        self.logger = get_logger(__name__)

    async def fetch_data(self, query: str, filters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Fetch component health data."""
        try:
            # Get component status
            component_status = self.component_manager.get_component_status()

            # Process data based on query
            if query == "health_overview":
                data = {
                    "total_components": component_status.get("total_components", 0),
                    "running_components": component_status.get("running_components", 0),
                    "failed_components": component_status.get("failed_components", 0),
                    "components": component_status.get("components", {}),
                }
            else:
                data = component_status

            return {"data": data, "timestamp": datetime.now(timezone.utc), "query": query}

        except Exception as e:
            self.logger.error(f"Failed to fetch component health data: {str(e)}")
            return {"data": {}, "error": str(e)}

    def supports_real_time(self) -> bool:
        """Component health supports real-time updates."""
        return True

    async def subscribe_to_updates(self, callback: Callable) -> str:
        """Subscribe to component health updates."""
        # Would integrate with component manager events
        return str(uuid.uuid4())


class ChartRenderer(ABC):
    """Abstract base class for chart renderers."""

    @abstractmethod
    async def render_chart(self, config: ChartConfiguration, data: Dict[str, Any]) -> str:
        """Render chart as HTML."""
        pass

    @abstractmethod
    def get_supported_types(self) -> List[ChartType]:
        """Get supported chart types."""
        pass


class PlotlyChartRenderer(ChartRenderer):
    """Chart renderer using Plotly."""

    def __init__(self):
        self.logger = get_logger(__name__)

        if not go:
            raise DashboardError("Plotly is required for chart rendering")

    def get_supported_types(self) -> List[ChartType]:
        """Get Plotly supported chart types."""
        return [
            ChartType.LINE_CHART,
            ChartType.BAR_CHART,
            ChartType.PIE_CHART,
            ChartType.SCATTER_PLOT,
            ChartType.HEATMAP,
            ChartType.HISTOGRAM,
            ChartType.BOX_PLOT,
            ChartType.GAUGE,
            ChartType.TIMELINE,
            ChartType.TREEMAP,
            ChartType.SANKEY,
        ]

    async def render_chart(self, config: ChartConfiguration, data: Dict[str, Any]) -> str:
        """Render chart using Plotly."""
        try:
            chart_data = data.get("data", [])

            if config.chart_type == ChartType.LINE_CHART:
                fig = self._create_line_chart(config, chart_data)
            elif config.chart_type == ChartType.BAR_CHART:
                fig = self._create_bar_chart(config, chart_data)
            elif config.chart_type == ChartType.PIE_CHART:
                fig = self._create_pie_chart(config, chart_data)
            elif config.chart_type == ChartType.SCATTER_PLOT:
                fig = self._create_scatter_plot(config, chart_data)
            elif config.chart_type == ChartType.HEATMAP:
                fig = self._create_heatmap(config, chart_data)
            elif config.chart_type == ChartType.GAUGE:
                fig = self._create_gauge(config, chart_data)
            else:
                raise DashboardError(f"Unsupported chart type: {config.chart_type}")

            # Apply common styling
            self._apply_styling(fig, config)

            # Convert to HTML
            html = to_html(
                fig,
                include_plotlyjs="cdn",
                div_id=f"chart_{config.chart_id}",
                config={"displayModeBar": config.interactive, "responsive": True},
            )

            return html

        except Exception as e:
            self.logger.error(f"Failed to render chart {config.chart_id}: {str(e)}")
            return f"<div>Error rendering chart: {str(e)}</div>"

    def _create_line_chart(self, config: ChartConfiguration, data: List[Dict]) -> go.Figure:
        """Create line chart."""
        fig = go.Figure()

        if not data:
            return fig

        # Extract x and y data
        x_values = [item.get(config.x_axis, item.get("timestamp", "")) for item in data]
        y_values = [item.get(config.y_axis, item.get("value", 0)) for item in data]

        fig.add_trace(go.Scatter(x=x_values, y=y_values, mode="lines+markers", name=config.title))

        return fig

    def _create_bar_chart(self, config: ChartConfiguration, data: List[Dict]) -> go.Figure:
        """Create bar chart."""
        fig = go.Figure()

        if not data:
            return fig

        x_values = [item.get(config.x_axis, item.get("label", "")) for item in data]
        y_values = [item.get(config.y_axis, item.get("value", 0)) for item in data]

        fig.add_trace(go.Bar(x=x_values, y=y_values, name=config.title))

        return fig

    def _create_pie_chart(self, config: ChartConfiguration, data: List[Dict]) -> go.Figure:
        """Create pie chart."""
        fig = go.Figure()

        if not data:
            return fig

        labels = [item.get("label", item.get(config.x_axis, "")) for item in data]
        values = [item.get("value", item.get(config.y_axis, 0)) for item in data]

        fig.add_trace(go.Pie(labels=labels, values=values, name=config.title))

        return fig

    def _create_scatter_plot(self, config: ChartConfiguration, data: List[Dict]) -> go.Figure:
        """Create scatter plot."""
        fig = go.Figure()

        if not data:
            return fig

        x_values = [item.get(config.x_axis, 0) for item in data]
        y_values = [item.get(config.y_axis, 0) for item in data]

        fig.add_trace(go.Scatter(x=x_values, y=y_values, mode="markers", name=config.title))

        return fig

    def _create_heatmap(self, config: ChartConfiguration, data: List[Dict]) -> go.Figure:
        """Create heatmap."""
        fig = go.Figure()

        if not data:
            return fig

        # Convert data to matrix format
        # This is a simplified version - would need more sophisticated processing
        z_values = [[item.get("value", 0) for item in data]]

        fig.add_trace(go.Heatmap(z=z_values, name=config.title))

        return fig

    def _create_gauge(self, config: ChartConfiguration, data: List[Dict]) -> go.Figure:
        """Create gauge chart."""
        fig = go.Figure()

        value = data[0].get("value", 0) if data else 0

        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=value,
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": config.title},
                gauge={
                    "axis": {"range": [None, 100]},
                    "bar": {"color": "darkblue"},
                    "steps": [
                        {"range": [0, 50], "color": "lightgray"},
                        {"range": [50, 100], "color": "gray"},
                    ],
                    "threshold": {
                        "line": {"color": "red", "width": 4},
                        "thickness": 0.75,
                        "value": 90,
                    },
                },
            )
        )

        return fig

    def _apply_styling(self, fig: go.Figure, config: ChartConfiguration) -> None:
        """Apply styling to figure."""
        fig.update_layout(
            title=config.title, height=config.height, template=config.color_scheme, showlegend=True
        )

        if config.x_axis:
            fig.update_xaxes(title_text=config.x_axis)
        if config.y_axis:
            fig.update_yaxes(title_text=config.y_axis)


class DashboardTemplate:
    """Template for dashboard creation."""

    def __init__(self, template_id: str, name: str, dashboard_type: DashboardType):
        self.template_id = template_id
        self.name = name
        self.dashboard_type = dashboard_type
        self.charts: List[ChartConfiguration] = []
        self.layout = DashboardLayout(f"{template_id}_layout", f"{name} Layout")

    def add_chart(
        self,
        chart_config: ChartConfiguration,
        row: int = 0,
        col: int = 0,
        width: int = 12,
        height: int = 1,
    ) -> "DashboardTemplate":
        """Add chart to template."""
        self.charts.append(chart_config)
        self.layout.charts.append(chart_config)
        self.layout.chart_positions[chart_config.chart_id] = {
            "row": row,
            "col": col,
            "width": width,
            "height": height,
        }
        return self

    def create_dashboard(self, dashboard_id: str, name: str, created_by: str) -> DashboardMetadata:
        """Create dashboard from template."""
        return DashboardMetadata(
            dashboard_id=dashboard_id,
            name=name,
            dashboard_type=self.dashboard_type,
            created_by=created_by,
            layout=self.layout,
        )


class EnhancedDashboardManager:
    """
    Advanced Dashboard Management System for AI Assistant Observability.

    This manager provides comprehensive dashboard capabilities including:
    - Real-time dashboard rendering and updates
    - Integration with all core system components
    - Component health monitoring dashboards
    - Workflow execution visualization
    - User interaction analytics
    - Session management monitoring
    - Plugin system observability
    - Performance metrics visualization
    - Security and audit dashboards
    - Custom dashboard creation
    - WebSocket-based real-time updates
    - Export capabilities (PDF, PNG, JSON)
    - Dashboard templates and themes
    - Role-based access control
    - Alert integration
    - Mobile-responsive design
    """

    def __init__(self, container: Container):
        """
        Initialize the enhanced dashboard manager.

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

        # Assistant components
        self.core_engine = container.get(EnhancedCoreEngine)
        self.component_manager = container.get(EnhancedComponentManager)
        self.workflow_orchestrator = container.get(WorkflowOrchestrator)
        self.interaction_handler = container.get(InteractionHandler)
        self.session_manager = container.get(EnhancedSessionManager)
        self.plugin_manager = container.get(EnhancedPluginManager)

        # Observability
        self.metrics = container.get(MetricsCollector)
        self.tracer = container.get(TraceManager)
        self.alert_manager = container.get(AlertManager)

        # Security
        try:
            self.auth_manager = container.get(AuthenticationManager)
            self.authz_manager = container.get(AuthorizationManager)
        except Exception:
            self.auth_manager = None
            self.authz_manager = None

        # Dashboard management
        self.dashboards: Dict[str, DashboardInstance] = {}
        self.templates: Dict[str, DashboardTemplate] = {}
        self.active_viewers: Dict[str, Set[str]] = defaultdict(set)  # dashboard_id -> user_ids

        # Data providers
        self.data_providers: Dict[str, DataProvider] = {}
        self.chart_renderers: Dict[ChartType, ChartRenderer] = {}

        # Real-time updates
        self.websocket_connections: Dict[str, Set] = defaultdict(set)  # dashboard_id -> connections
        self.update_queue = asyncio.Queue()
        self.cache: Dict[str, Dict[str, Any]] = {}

        # Performance tracking
        self.render_stats: Dict[str, List[float]] = defaultdict(list)
        self.access_stats: Dict[str, int] = defaultdict(int)

        # Configuration
        self.enable_real_time = self.config.get("dashboards.enable_real_time", True)
        self.cache_enabled = self.config.get("dashboards.cache_enabled", True)
        self.cache_ttl = self.config.get("dashboards.cache_ttl", 300)
        self.max_dashboards = self.config.get("dashboards.max_dashboards", 100)
        self.export_enabled = self.config.get("dashboards.export_enabled", True)

        # Initialize components
        self._setup_data_providers()
        self._setup_chart_renderers()
        self._setup_templates()
        self._setup_monitoring()

        # Register health check
        self.health_check.register_component("dashboard_manager", self._health_check_callback)

        self.logger.info("EnhancedDashboardManager initialized successfully")

    def _setup_data_providers(self) -> None:
        """Setup data providers for different data sources."""
        try:
            # Metrics data provider
            self.data_providers["metrics"] = MetricsDataProvider(self.metrics)

            # Component health data provider
            self.data_providers["component_health"] = ComponentHealthDataProvider(
                self.component_manager
            )

            # Add more data providers as needed
            self.logger.info(f"Initialized {len(self.data_providers)} data providers")

        except Exception as e:
            self.logger.error(f"Failed to setup data providers: {str(e)}")

    def _setup_chart_renderers(self) -> None:
        """Setup chart renderers for different visualization libraries."""
        try:
            # Plotly renderer
            if go:
                plotly_renderer = PlotlyChartRenderer()
                for chart_type in plotly_renderer.get_supported_types():
                    self.chart_renderers[chart_type] = plotly_renderer

            self.logger.info(
                f"Initialized {len(set(self.chart_renderers.values()))} chart renderers"
            )

        except Exception as e:
            self.logger.error(f"Failed to setup chart renderers: {str(e)}")

    def _setup_templates(self) -> None:
        """Setup built-in dashboard templates."""
        try:
            # System Overview Template
            overview_template = DashboardTemplate(
                "system_overview", "System Overview", DashboardType.OVERVIEW
            )

            overview_template.add_chart(
                ChartConfiguration(
                    chart_id="system_health",
                    chart_type=ChartType.GAUGE,
                    title="System Health",
                    data_source="component_health",
                    query="health_overview",
                    width=6,
                    height=300,
                ),
                row=0,
                col=0,
                width=6,
                height=1,
            )

            overview_template.add_chart(
                ChartConfiguration(
                    chart_id="active_sessions",
                    chart_type=ChartType.LINE_CHART,
                    title="Active Sessions",
                    data_source="metrics",
                    query="active_sessions",
                    x_axis="timestamp",
                    y_axis="value",
                    width=6,
                    height=300,
                ),
                row=0,
                col=6,
                width=6,
                height=1,
            )

            self.templates["system_overview"] = overview_template

            # Component Health Template
            component_template = DashboardTemplate(
                "component_health", "Component Health Monitoring", DashboardType.COMPONENT_HEALTH
            )

            component_template.add_chart(
                ChartConfiguration(
                    chart_id="component_status",
                    chart_type=ChartType.PIE_CHART,
                    title="Component Status Distribution",
                    data_source="component_health",
                    query="status_distribution",
                    width=12,
                    height=400,
                ),
                row=0,
                col=0,
                width=12,
                height=1,
            )

            self.templates["component_health"] = component_template

            # Workflow Analytics Template
            workflow_template = DashboardTemplate(
                "workflow_analytics",
                "Workflow Execution Analytics",
                DashboardType.WORKFLOW_ANALYTICS,
            )

            workflow_template.add_chart(
                ChartConfiguration(
                    chart_id="workflow_completion_rate",
                    chart_type=ChartType.BAR_CHART,
                    title="Workflow Completion Rate",
                    data_source="metrics",
                    query="workflow_completion_rate",
                    x_axis="workflow_type",
                    y_axis="completion_rate",
                    width=6,
                    height=400,
                ),
                row=0,
                col=0,
                width=6,
                height=1,
            )

            workflow_template.add_chart(
                ChartConfiguration(
                    chart_id="workflow_execution_time",
                    chart_type=ChartType.LINE_CHART,
                    title="Average Execution Time",
                    data_source="metrics",
                    query="workflow_execution_time",
                    x_axis="timestamp",
                    y_axis="avg_time",
                    width=6,
                    height=400,
                ),
                row=0,
                col=6,
                width=6,
                height=1,
            )

            self.templates["workflow_analytics"] = workflow_template

            self.logger.info(f"Initialized {len(self.templates)} dashboard templates")

        except Exception as e:
            self.logger.error(f"Failed to setup templates: {str(e)}")

    def _setup_monitoring(self) -> None:
        """Setup monitoring and metrics for dashboard system."""
        try:
            # Register dashboard metrics
            self.metrics.register_counter("dashboards_created_total")
            self.metrics.register_counter("dashboards_viewed_total")
            self.metrics.register_counter("dashboards_exported_total")
            self.metrics.register_gauge("active_dashboards")
            self.metrics.register_gauge("dashboard_viewers")
            self.metrics.register_histogram("dashboard_render_time_seconds")
            self.metrics.register_histogram("chart_render_time_seconds")

        except Exception as e:
            self.logger.warning(f"Failed to setup monitoring: {str(e)}")

    async def initialize(self) -> None:
        """Initialize the dashboard manager."""
        try:
            # Initialize data providers
            for provider in self.data_providers.values():
                if hasattr(provider, "initialize"):
                    await provider.initialize()

            # Start background tasks
            if self.enable_real_time:
                asyncio.create_task(self._real_time_update_loop())

            asyncio.create_task(self._cache_cleanup_loop())
            asyncio.create_task(self._performance_monitoring_loop())

            # Register event handlers
            await self._register_event_handlers()

            self.logger.info("DashboardManager initialization completed")

        except Exception as e:
            self.logger.error(f"Failed to initialize DashboardManager: {str(e)}")
            raise DashboardError(f"Initialization failed: {str(e)}")

    async def _register_event_handlers(self) -> None:
        """Register event handlers for system events."""
        # Component health events
        self.event_bus.subscribe("component_health_changed", self._handle_component_health_change)

        # Workflow events
        self.event_bus.subscribe("workflow_completed", self._handle_workflow_completed)

        # User interaction events
        self.event_bus.subscribe("user_interaction_completed", self._handle_user_interaction)

        # Session events
        self.event_bus.subscribe("session_ended", self._handle_session_ended)

        # Plugin events
        self.event_bus.subscribe("plugin_loaded", self._handle_plugin_loaded)

        # Error events
        self.event_bus.subscribe("error_occurred", self._handle_error_occurred)

    @handle_exceptions
    async def create_dashboard(
        self,
        name: str,
        dashboard_type: DashboardType,
        created_by: str,
        template_id: Optional[str] = None,
        layout: Optional[DashboardLayout] = None,
        access_level: AccessLevel = AccessLevel.PRIVATE,
        **kwargs,
    ) -> str:
        """
        Create a new dashboard.

        Args:
            name: Dashboard name
            dashboard_type: Type of dashboard
            created_by: User creating the dashboard
            template_id: Optional template to use
            layout: Optional custom layout
            access_level: Dashboard access level
            **kwargs: Additional metadata

        Returns:
            Dashboard ID
        """
        dashboard_id = str(uuid.uuid4())

        try:
            # Check dashboard limit
            if len(self.dashboards) >= self.max_dashboards:
                raise DashboardError("Maximum number of dashboards reached")

            # Create from template if specified
            if template_id and template_id in self.templates:
                template = self.templates[template_id]
                metadata = template.create_dashboard(dashboard_id, name, created_by)
                metadata.dashboard_type = dashboard_type
                metadata.access_level = access_level
            else:
                # Create new dashboard
                metadata = DashboardMetadata(
                    dashboard_id=dashboard_id,
                    name=name,
                    dashboard_type=dashboard_type,
                    created_by=created_by,
                    access_level=access_level,
                    layout=layout or DashboardLayout(f"{dashboard_id}_layout", f"{name} Layout"),
                )

            # Apply additional metadata
            for key, value in kwargs.items():
                if hasattr(metadata, key):
                    setattr(metadata, key, value)

            # Create dashboard instance
            dashboard = DashboardInstance(metadata=metadata)

            # Store dashboard
            self.dashboards[dashboard_id] = dashboard

            # Emit creation event
            await self.event_bus.emit(
                DashboardCreated(
                    dashboard_id=dashboard_id,
                    name=name,
                    dashboard_type=dashboard_type.value,
                    created_by=created_by,
                )
            )

            # Update metrics
            self.metrics.increment("dashboards_created_total")
            self.metrics.set("active_dashboards", len(self.dashboards))

            self.logger.info(f"Created dashboard: {dashboard_id} ({name})")
            return dashboard_id

        except Exception as e:
            self.logger.error(f"Failed to create dashboard: {str(e)}")
            raise DashboardError(f"Failed to create dashboard: {str(e)}")

    @handle_exceptions
    async def update_dashboard(
        self, dashboard_id: str, user_id: str, updates: Dict[str, Any]
    ) -> None:
        """
        Update an existing dashboard.

        Args:
            dashboard_id: Dashboard identifier
            user_id: User making the update
            updates: Updates to apply
        """
        if dashboard_id not in self.dashboards:
            raise DashboardError(f"Dashboard {dashboard_id} not found")

        dashboard = self.dashboards[dashboard_id]

        # Check permissions
        if not await self._check_dashboard_permission(dashboard, user_id, "update"):
            raise DashboardError(
                f"User {user_id} not authorized to update dashboard {dashboard_id}"
            )

        try:
            # Apply updates
            metadata = dashboard.metadata

            for key, value in updates.items():
                if hasattr(metadata, key):
                    setattr(metadata, key, value)

            metadata.updated_at = datetime.now(timezone.utc)

            # Clear cached data if layout changed
            if "layout" in updates:
                dashboard.cached_data.clear()
                dashboard.rendered_charts.clear()

            # Emit update event
            await self.event_bus.emit(
                DashboardUpdated(
                    dashboard_id=dashboard_id, updated_by=user_id, updates=list(updates.keys())
                )
            )

            self.logger.info(f"Updated dashboard: {dashboard_id}")

        except Exception as e:
            self.logger.error(f"Failed to update dashboard {dashboard_id}: {str(e)}")
            raise DashboardError(f"Failed to update dashboard: {str(e)}")

    @handle_exceptions
    async def delete_dashboard(self, dashboard_id: str, user_id: str) -> None:
        """
        Delete a dashboard.

        Args:
            dashboard_id: Dashboard identifier
            user_id: User deleting the dashboard
        """
        if dashboard_id not in self.dashboards:
            raise DashboardError(f"Dashboard {dashboard_id} not found")

        dashboard = self.dashboards[dashboard_id]

        # Check permissions
        if not await self._check_dashboard_permission(dashboard, user_id, "delete"):
            raise DashboardError(
                f"User {user_id} not authorized to delete dashboard {dashboard_id}"
            )

        try:
            # Remove dashboard
            del self.dashboards[dashboard_id]

            # Clean up cache
            self.cache.pop(dashboard_id, None)

            # Clean up active viewers
            self.active_viewers.pop(dashboard_id, None)

            # Clean up WebSocket connections
            self.websocket_connections.pop(dashboard_id, None)

            # Emit deletion event
            await self.event_bus.emit(
                DashboardDeleted(dashboard_id=dashboard_id, deleted_by=user_id)
            )

            # Update metrics
            self.metrics.set("active_dashboards", len(self.dashboards))

            self.logger.info(f"Deleted dashboard: {dashboard_id}")

        except Exception as e:
            self.logger.error(f"Failed to delete dashboard {dashboard_id}: {str(e)}")
            raise DashboardError(f"Failed to delete dashboard: {str(e)}")

    @handle_exceptions
    async def render_dashboard(
        self, dashboard_id: str, user_id: str, force_refresh: bool = False
    ) -> str:
        """
        Render a dashboard as HTML.

        Args:
            dashboard_id: Dashboard identifier
            user_id: User requesting the dashboard
            force_refresh: Force refresh of cached data

        Returns:
            Rendered HTML
        """
        if dashboard_id not in self.dashboards:
            raise DashboardError(f"Dashboard {dashboard_id} not found")

        dashboard = self.dashboards[dashboard_id]

        # Check permissions
        if not await self._check_dashboard_permission(dashboard, user_id, "view"):
            raise DashboardError(f"User {user_id} not authorized to view dashboard {dashboard_id}")

        start_time = time.time()

        try:
            # Update access tracking
            dashboard.last_accessed = datetime.now(timezone.utc)
            dashboard.view_count += 1
            dashboard.active_viewers.add(user_id)
            self.active_viewers[dashboard_id].add(user_id)

            # Check cache
            cache_key = f"{dashboard_id}_html"
            if (
                not force_refresh
                and self.cache_enabled
                and cache_key in self.cache
                and self._is_cache_valid(cache_key)
            ):

                dashboard.cache_hit_rate += 0.1
                return self.cache[cache_key]["data"]

            # Render charts
            rendered_charts = {}
            for chart_config in dashboard.metadata.layout.charts:
                chart_html = await self._render_chart(chart_config, dashboard_id, force_refresh)
                rendered_charts[chart_config.chart_id] = chart_html

            dashboard.rendered_charts = rendered_charts

            # Generate dashboard HTML
            html = await self._generate_dashboard_html(dashboard, rendered_charts)

            # Cache result
            if self.cache_enabled:
                self.cache[cache_key] = {
                    "data": html,
                    "timestamp": datetime.now(timezone.utc),
                    "ttl": self.cache_ttl,
                }

            # Update performance metrics
            render_time = time.time() - start_time
            dashboard.render_time = render_time
            dashboard.last_updated = datetime.now(timezone.utc)

            self.render_stats[dashboard_id].append(render_time)
            self.access_stats[dashboard_id] += 1

            # Emit view event
            await self.event_bus.emit(
                DashboardViewed(
                    dashboard_id=dashboard_id, viewed_by=user_id, render_time=render_time
                )
            )

            # Update metrics
            self.metrics.increment("dashboards_viewed_total")
            self.metrics.record("dashboard_render_time_seconds", render_time)
            self.metrics.set("dashboard_viewers", len(self.active_viewers[dashboard_id]))

            self.logger.debug(f"Rendered dashboard {dashboard_id} in {render_time:.2f}s")
            return html

        except Exception as e:
            self.logger.error(f"Failed to render dashboard {dashboard_id}: {str(e)}")
            raise DashboardError(f"Failed to render dashboard: {str(e)}")

    async def _render_chart(
        self, chart_config: ChartConfiguration, dashboard_id: str, force_refresh: bool = False
    ) -> str:
        """Render an individual chart."""
        chart_start_time = time.time()

        try:
            # Check chart cache
            cache_key = f"{dashboard_id}_{chart_config.chart_id}_data"

            if (
                not force_refresh
                and self.cache_enabled
                and cache_key in self.cache
                and self._is_cache_valid(cache_key)
            ):

                data = self.cache[cache_key]["data"]
            else:
                # Fetch data
                provider = self.data_providers.get(chart_config.data_source)
                if not provider:
                    raise DashboardError(f"Data provider {chart_config.data_source} not found")

                data = await provider.fetch_data(chart_config.query, chart_config.filters)

                # Cache data
                if self.cache_enabled:
                    self.cache[cache_key] = {
                        "data": data,
                        "timestamp": datetime.now(timezone.utc),
                        "ttl": chart_config.cache_duration,
                    }

            # Render chart
            renderer = self.chart_renderers.get(chart_config.chart_type)
            if not renderer:
                raise DashboardError(f"No renderer found for chart type {chart_config.chart_type}")

            html = await renderer.render_chart(chart_config, data)

            # Update metrics
            chart_render_time = time.time() - chart_start_time
            self.metrics.record("chart_render_time_seconds", chart_render_time)

            return html

        except Exception as e:
            self.logger.error(f"Failed to render chart {chart_config.chart_id}: {str(e)}")
            return f'<div class="alert alert-danger">Error rendering chart: {str(e)}</div>'

    async def _generate_dashboard_html(
        self, dashboard: DashboardInstance, rendered_charts: Dict[str, str]
    ) -> str:
        """Generate complete dashboard HTML."""
        try:
            # Create HTML template
            html = f"""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>{dashboard.metadata.name}</title>
                <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
                <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
                <style>
                    .dashboard-container {{
                        padding: 20px;
                        background-color: {dashboard.metadata.layout.background_color};
                    }}
                    .chart-container {{
                        margin-bottom: {dashboard.metadata.layout.gap}px;
                        border: {dashboard.metadata.layout.border_style};
                        padding: {dashboard.metadata.layout.padding}px;
                    }}
                    {dashboard.metadata.custom_css or ''}
                </style>
            </head>
            <body>
                <div class="dashboard-container">
                    <div class="dashboard-header mb-4">
                        <h1>{dashboard.metadata.name}</h1>
                        <p class="text-muted">{dashboard.metadata.description or ''}</p>
                        <div class="dashboard-info">
                            <small class="text-muted">
                                Last updated: {dashboard.last_updated.strftime('%Y-%m-%d %H:%M:%S') if dashboard.last_updated else 'Never'}
                                | Views: {dashboard.view_count}
                                | Active viewers: {len(dashboard.active_viewers)}
                            </small>
                        </div>
                    </div>
                    <div class="dashboard-content">
                        <div class="row">
            """

            # Add charts based on layout
            layout = dashboard.metadata.layout
            for chart_config in layout.charts:
                position = layout.chart_positions.get(chart_config.chart_id, {})
                width = position.get("width", chart_config.width)

                chart_html = rendered_charts.get(chart_config.chart_id, "")

                html += f"""
                            <div class="col-md-{width}">
                                <div class="chart-container">
                                    <div class="chart-header">
                                        <h5>{chart_config.title}</h5>
                                        {f'<p class="text-muted">{chart_config.description}</p>' if chart_config.description else ''}
                                    </div>
                                    <div class="chart-content">
                                        {chart_html}
                                    </div>
                                </div>
                            </div>
                """

            html += """
                        </div>
                    </div>
                </div>
                <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
            """

            # Add real-time update script if enabled
            if dashboard.metadata.real_time_enabled and self.enable_real_time:
                html += f"""
                <script>
                    // WebSocket connection for real-time updates
                    const ws = new WebSocket('ws://localhost:8000/ws/dashboard/{dashboard.metadata.dashboard_id}');
                    ws.onmessage = function(event) {{
                        const data = JSON.parse(event.data);
                        if (data.type === 'chart_update') {{
                            // Update specific chart
                            const chartElement = document.getElementById('chart_' + data.chart_id);
                            if (chartElement) {{
                                chartElement.innerHTML = data.html;
                            }}
                        }} else if (data.type === 'dashboard_refresh') {{
                            // Refresh entire dashboard
                            location.reload();
                        }}
                    }};
                </script>
                """

            html += """
            </body>
            </html>
            """

            return html

        except Exception as e:
            self.logger.error(f"Failed to generate dashboard HTML: {str(e)}")
            return f'<div class="alert alert-danger">Error generating dashboard: {str(e)}</div>'

    @handle_exceptions
    async def export_dashboard(
        self,
        dashboard_id: str,
        user_id: str,
        export_format: ExportFormat,
        options: Optional[Dict[str, Any]] = None,
    ) -> bytes:
        """
        Export dashboard in specified format.

        Args:
            dashboard_id: Dashboard identifier
            user_id: User requesting export
            export_format: Export format
            options: Export options

        Returns:
            Exported data as bytes
        """
        if not self.export_enabled:
            raise DashboardError("Dashboard export is disabled")

        if dashboard_id not in self.dashboards:
            raise DashboardError(f"Dashboard {dashboard_id} not found")

        dashboard = self.dashboards[dashboard_id]

        # Check permissions
        if not await self._check_dashboard_permission(dashboard, user_id, "export"):
            raise DashboardError(
                f"User {user_id} not authorized to export dashboard {dashboard_id}"
            )

        try:
            if export_format == ExportFormat.HTML:
                html = await self.render_dashboard(dashboard_id, user_id)
                return html.encode("utf-8")

            elif export_format == ExportFormat.JSON:
                data = {
                    "metadata": asdict(dashboard.metadata),
                    "data": dashboard.cached_data,
                    "export_timestamp": datetime.now(timezone.utc).isoformat(),
                }
                return json.dumps(data, indent=2, default=str).encode("utf-8")

            elif export_format == ExportFormat.PDF:
                # Would implement PDF generation using libraries like weasyprint
                raise DashboardError("PDF export not yet implemented")

            elif export_format == ExportFormat.PNG:
                # Would implement PNG export using headless browser
                raise DashboardError("PNG export not yet implemented")

            else:
                raise DashboardError(f"Unsupported export format: {export_format}")

        except Exception as e:
            self.logger.error(f"Failed to export dashboard {dashboard_id}: {str(e)}")
            raise DashboardError(f"Failed to export dashboard: {str(e)}")

    async def get_dashboard_list(
        self,
        user_id: str,
        dashboard_type: Optional[DashboardType] = None,
        access_level: Optional[AccessLevel] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get list of dashboards accessible to user.

        Args:
            user_id: User identifier
            dashboard_type: Optional type filter
            access_level: Optional access level filter

        Returns:
            List of dashboard metadata
        """
        dashboards = []

        for dashboard in self.dashboards.values():
            # Check access permissions
            if await self._check_dashboard_permission(dashboard, user_id, "view"):
                # Apply filters
                if dashboard_type and dashboard.metadata.dashboard_type != dashboard_type:
                    continue
                if access_level and dashboard.metadata.access_level != access_level:
                    continue

                # Add to list
                dashboards.append(
                    {
                        "dashboard_id": dashboard.metadata.dashboard_id,
                        "name": dashboard.metadata.name,
                        "dashboard_type": dashboard.metadata.dashboard_type.value,
                        "description": dashboard.metadata.description,
                        "created_by": dashboard.metadata.created_by,
                        "created_at": dashboard.metadata.created_at.isoformat(),
                        "updated_at": dashboard.metadata.updated_at.isoformat(),
                        "access_level": dashboard.metadata.access_level.value,
                        "view_count": dashboard.view_count,
                        "last_accessed": (
                            dashboard.last_accessed.isoformat() if dashboard.last_accessed else None
                        ),
                        "active_viewers": len(dashboard.active_viewers),
                        "chart_count": len(dashboard.metadata.layout.charts),
                    }
                )

        return sorted(dashboards, key=lambda x: x["updated_at"], reverse=True)

    async def get_dashboard_templates(self) -> List[Dict[str, Any]]:
        """Get available dashboard templates."""
        templates = []

        for template in self.templates.values():
            templates.append(
                {
                    "template_id": template.template_id,
                    "name": template.name,
                    "dashboard_type": template.dashboard_type.value,
                    "chart_count": len(template.charts),
                    "description": f"Template for {template.name}",
                }
            )

        return templates

    async def _check_dashboard_permission(
        self, dashboard: DashboardInstance, user_id: str, operation: str
    ) -> bool:
        """Check if user has permission for dashboard operation."""
        try:
            metadata = dashboard.metadata

            # Owner always has access
            if metadata.created_by == user_id:
                return True

            # Check access level
            if metadata.access_level == AccessLevel.PUBLIC:
                return True
            elif metadata.access_level == AccessLevel.AUTHENTICATED:
                # Would check if user is authenticated
                return self.auth_manager is not None
            elif metadata.access_level == AccessLevel.PRIVATE:
                return user_id in metadata.allowed_users
            elif metadata.access_level == AccessLevel.RESTRICTED:
                # Check authorization if available
                if self.authz_manager:
                    return await self.authz_manager.check_permission(
                        user_id, f"dashboard:{operation}", dashboard.metadata.dashboard_id
                    )
                return user_id in metadata.allowed_users

            return False

        except Exception as e:
            self.logger.error(f"Permission check failed: {str(e)}")
            return False

    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid."""
        if cache_key not in self.cache:
            return False

        cache_entry = self.cache[cache_key]
        now = datetime.now(timezone.utc)
        cache_time = cache_entry["timestamp"]
        ttl = cache_entry.get("ttl", self.cache_ttl)

        return (now - cache_time).total_seconds() < ttl

    async def _real_time_update_loop(self) -> None:
        """Background task for real-time dashboard updates."""
        while True:
            try:
                # Check for dashboards with real-time enabled
                for dashboard_id, dashboard in self.dashboards.items():
                    if (
                        dashboard.metadata.real_time_enabled
                        and dashboard.active_viewers
                        and dashboard_id in self.websocket_connections
                    ):

                        # Check if any charts need updates
                        for chart_config in dashboard.metadata.layout.charts:
                            if chart_config.refresh_interval <= 30:  # Real-time threshold
                                # Fetch fresh data
                                try:
                                    provider = self.data_providers.get(chart_config.data_source)
                                    if provider and provider.supports_real_time():
                                        # Would implement real-time data fetch and update
                                        pass
                                except Exception as e:
                                    self.logger.warning(
                                        f"Real-time update failed for chart {chart_config.chart_id}: {str(e)}"
                                    )

                await asyncio.sleep(5)  # Check every 5 seconds

            except Exception as e:
                self.logger.error(f"Real-time update loop error: {str(e)}")
                await asyncio.sleep(5)

    async def _cache_cleanup_loop(self) -> None:
        """Background task for cache cleanup."""
        while True:
            try:
                now = datetime.now(timezone.utc)
                expired_keys = []

                for cache_key, cache_entry in self.cache.items():
                    cache_time = cache_entry["timestamp"]
                    ttl = cache_entry.get("ttl", self.cache_ttl)

                    if (now - cache_time).total_seconds() > ttl:
                        expired_keys.append(cache_key)

                # Remove expired entries
                for key in expired_keys:
                    del self.cache[key]

                if expired_keys:
                    self.logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")

                await asyncio.sleep(60)  # Cleanup every minute

            except Exception as e:
                self.logger.error(f"Cache cleanup error: {str(e)}")
                await asyncio.sleep(60)

    async def _performance_monitoring_loop(self) -> None:
        """Background task for performance monitoring."""
        while True:
            try:
                # Update performance metrics
                total_render_times = []
                for render_times in self.render_stats.values():
                    total_render_times.extend(render_times)

                if total_render_times:
                    avg_render_time = sum(total_render_times) / len(total_render_times)
                    self.metrics.set("dashboard_avg_render_time_seconds", avg_render_time)

                # Monitor cache performance
                if self.cache:
                    cache_size = len(self.cache)
                    self.metrics.set("dashboard_cache_size", cache_size)

                # Clean up old performance data
                for dashboard_id in list(self.render_stats.keys()):
                    if len(self.render_stats[dashboard_id]) > 100:
                        self.render_stats[dashboard_id] = self.render_stats[dashboard_id][-50:]

                await asyncio.sleep(30)  # Monitor every 30 seconds

            except Exception as e:
                self.logger.error(f"Performance monitoring error: {str(e)}")
                await asyncio.sleep(30)

    async def _handle_component_health_change(self, event) -> None:
        """Handle component health change events."""
        try:
            # Update component health dashboards
            for dashboard_id, dashboard in self.dashboards.items():
                if dashboard.metadata.dashboard_type == DashboardType.COMPONENT_HEALTH:
                    # Invalidate cache for component health charts
                    cache_keys_to_remove = [
                        key
                        for key in self.cache.keys()
                        if key.startswith(f"{dashboard_id}_") and "component" in key
                    ]
                    for key in cache_keys_to_remove:
                        del self.cache[key]

                    # Notify real-time viewers
                    if dashboard.metadata.real_time_enabled:
                        await self._notify_dashboard_update(
                            dashboard_id, "component_health_changed"
                        )

        except Exception as e:
            self.logger.error(f"Error handling component health change: {str(e)}")

    async def _handle_workflow_completed(self, event) -> None:
        """Handle workflow completion events."""
        try:
            # Update workflow metrics
            workflow_id = getattr(event, "workflow_id", None)
            if workflow_id:
                # Refresh dashboards that track workflow metrics
                await self._refresh_workflow_dashboards(workflow_id)
        except Exception as e:
            self.logger.error(f"Error handling workflow completion: {str(e)}")
