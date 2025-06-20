"""
Advanced Alerting System for AI Assistant
Author: Drmusab
Last Modified: 2025-06-20 12:12:46 UTC

This module provides comprehensive alerting capabilities for the AI assistant,
including real-time monitoring, intelligent alert routing, escalation policies,
and integration with all core system components.
"""

from pathlib import Path
from typing import Optional, Dict, Any, List, Set, Callable, Type, Union, AsyncGenerator, TypeVar
import asyncio
import threading
import time
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
from contextlib import asynccontextmanager
import uuid
import json
import hashlib
from collections import defaultdict, deque
import weakref
from abc import ABC, abstractmethod
import logging
import inspect
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import ssl
import requests
import yaml
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# Core imports
from src.core.config.loader import ConfigLoader
from src.core.events.event_bus import EventBus
from src.core.events.event_types import (
    ComponentHealthChanged, SystemStateChanged, EngineStarted, EngineShutdown,
    ProcessingError, SessionStarted, SessionEnded, SessionExpired,
    WorkflowFailed, PluginError, MemoryOperationCompleted, ErrorOccurred,
    UserInteractionFailed, PerformanceThresholdExceeded, SecurityViolation
)
from src.core.error_handling import ErrorHandler, handle_exceptions
from src.core.dependency_injection import Container
from src.core.health_check import HealthCheck

# Monitoring components
from src.observability.monitoring.metrics import MetricsCollector
from src.observability.monitoring.tracing import TraceManager
from src.observability.logging.config import get_logger

# Assistant components
from src.assistant.core_engine import EnhancedCoreEngine
from src.assistant.component_manager import EnhancedComponentManager
from src.assistant.session_manager import EnhancedSessionManager
from src.assistant.workflow_orchestrator import WorkflowOrchestrator
from src.assistant.plugin_manager import EnhancedPluginManager

# Storage and caching
try:
    from src.integrations.storage.database import DatabaseManager
    from src.integrations.cache.redis_cache import RedisCache
except ImportError:
    DatabaseManager = None
    RedisCache = None

# Type definitions
T = TypeVar('T')


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AlertStatus(Enum):
    """Alert lifecycle status."""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"
    ESCALATED = "escalated"
    CLOSED = "closed"


class AlertCategory(Enum):
    """Alert categories for classification."""
    SYSTEM_HEALTH = "system_health"
    PERFORMANCE = "performance"
    SECURITY = "security"
    COMPONENT_FAILURE = "component_failure"
    WORKFLOW_ERROR = "workflow_error"
    USER_EXPERIENCE = "user_experience"
    RESOURCE_USAGE = "resource_usage"
    DATA_INTEGRITY = "data_integrity"
    INTEGRATION_FAILURE = "integration_failure"
    CUSTOM = "custom"


class NotificationChannel(Enum):
    """Notification delivery channels."""
    EMAIL = "email"
    SLACK = "slack"
    DISCORD = "discord"
    WEBHOOK = "webhook"
    SMS = "sms"
    PAGERDUTY = "pagerduty"
    TEAMS = "teams"
    TELEGRAM = "telegram"
    CONSOLE = "console"
    LOG = "log"


@dataclass
class AlertRule:
    """Definition of an alert rule."""
    rule_id: str
    name: str
    description: str
    
    # Condition definition
    metric_name: Optional[str] = None
    threshold_value: Optional[float] = None
    operator: str = ">"  # >, <, >=, <=, ==, !=
    evaluation_window: timedelta = field(default=timedelta(minutes=5))
    evaluation_interval: timedelta = field(default=timedelta(minutes=1))
    
    # Custom condition function
    custom_condition: Optional[Callable[[Dict[str, Any]], bool]] = None
    
    # Alert configuration
    severity: AlertSeverity = AlertSeverity.WARNING
    category: AlertCategory = AlertCategory.SYSTEM_HEALTH
    
    # Notification settings
    notification_channels: List[NotificationChannel] = field(default_factory=list)
    notification_template: Optional[str] = None
    notification_delay: timedelta = field(default=timedelta(seconds=0))
    
    # Behavior settings
    enabled: bool = True
    auto_resolve: bool = True
    auto_resolve_timeout: timedelta = field(default=timedelta(hours=1))
    max_alerts_per_hour: int = 10
    suppress_duplicates: bool = True
    
    # Escalation settings
    escalation_enabled: bool = False
    escalation_delay: timedelta = field(default=timedelta(minutes=30))
    escalation_channels: List[NotificationChannel] = field(default_factory=list)
    
    # Metadata
    tags: Set[str] = field(default_factory=set)
    owner: Optional[str] = None
    documentation_url: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # State tracking
    last_triggered: Optional[datetime] = None
    trigger_count: int = 0
    last_evaluation: Optional[datetime] = None


@dataclass
class Alert:
    """Individual alert instance."""
    alert_id: str
    rule_id: str
    rule_name: str
    
    # Alert content
    title: str
    description: str
    severity: AlertSeverity
    category: AlertCategory
    
    # Context information
    affected_component: Optional[str] = None
    affected_user: Optional[str] = None
    affected_session: Optional[str] = None
    metric_value: Optional[float] = None
    threshold_value: Optional[float] = None
    
    # Lifecycle tracking
    status: AlertStatus = AlertStatus.ACTIVE
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    resolved_by: Optional[str] = None
    
    # Additional data
    context: Dict[str, Any] = field(default_factory=dict)
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    
    # Notification tracking
    notifications_sent: List[Dict[str, Any]] = field(default_factory=list)
    escalated: bool = False
    escalated_at: Optional[datetime] = None
    
    # Resolution information
    resolution_note: Optional[str] = None
    auto_resolved: bool = False


@dataclass
class NotificationTarget:
    """Configuration for a notification target."""
    target_id: str
    channel: NotificationChannel
    config: Dict[str, Any] = field(default_factory=dict)
    
    # Filtering
    severity_filter: Set[AlertSeverity] = field(default_factory=set)
    category_filter: Set[AlertCategory] = field(default_factory=set)
    component_filter: Set[str] = field(default_factory=set)
    
    # Rate limiting
    rate_limit_enabled: bool = True
    max_notifications_per_hour: int = 20
    quiet_hours_start: Optional[int] = None  # Hour (0-23)
    quiet_hours_end: Optional[int] = None    # Hour (0-23)
    
    # Retry settings
    retry_enabled: bool = True
    max_retries: int = 3
    retry_delay: timedelta = field(default=timedelta(minutes=5))
    
    # Status
    enabled: bool = True
    last_notification: Optional[datetime] = None
    notification_count: int = 0
    failure_count: int = 0


class AlertingError(Exception):
    """Custom exception for alerting operations."""
    
    def __init__(self, message: str, alert_id: Optional[str] = None, 
                 rule_id: Optional[str] = None, error_code: Optional[str] = None):
        super().__init__(message)
        self.alert_id = alert_id
        self.rule_id = rule_id
        self.error_code = error_code
        self.timestamp = datetime.now(timezone.utc)


class NotificationSender(ABC):
    """Abstract base class for notification senders."""
    
    @abstractmethod
    async def send_notification(self, alert: Alert, target: NotificationTarget) -> bool:
        """Send notification for an alert."""
        pass
    
    @abstractmethod
    def can_handle(self, channel: NotificationChannel) -> bool:
        """Check if this sender can handle the given channel."""
        pass
    
    @abstractmethod
    async def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate notification target configuration."""
        pass


class EmailNotificationSender(NotificationSender):
    """Email notification sender."""
    
    def __init__(self, logger):
        self.logger = logger
    
    def can_handle(self, channel: NotificationChannel) -> bool:
        return channel == NotificationChannel.EMAIL
    
    async def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate email configuration."""
        required_fields = ['smtp_host', 'smtp_port', 'from_email', 'to_emails']
        return all(field in config for field in required_fields)
    
    async def send_notification(self, alert: Alert, target: NotificationTarget) -> bool:
        """Send email notification."""
        try:
            config = target.config
            
            # Create message
            msg = MIMEMultipart()
            msg['From'] = config['from_email']
            msg['To'] = ', '.join(config['to_emails'])
            msg['Subject'] = f"[{alert.severity.value.upper()}] {alert.title}"
            
            # Create email body
            body = self._create_email_body(alert)
            msg.attach(MIMEText(body, 'html'))
            
            # Send email
            context = ssl.create_default_context()
            
            with smtplib.SMTP(config['smtp_host'], config['smtp_port']) as server:
                if config.get('use_tls', True):
                    server.starttls(context=context)
                
                if 'username' in config and 'password' in config:
                    server.login(config['username'], config['password'])
                
                text = msg.as_string()
                server.sendmail(config['from_email'], config['to_emails'], text)
            
            self.logger.info(f"Email notification sent for alert {alert.alert_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send email notification: {str(e)}")
            return False
    
    def _create_email_body(self, alert: Alert) -> str:
        """Create HTML email body."""
        severity_colors = {
            AlertSeverity.INFO: "#17a2b8",
            AlertSeverity.WARNING: "#ffc107",
            AlertSeverity.ERROR: "#dc3545",
            AlertSeverity.CRITICAL: "#e83e8c",
            AlertSeverity.EMERGENCY: "#6f42c1"
        }
        
        color = severity_colors.get(alert.severity, "#6c757d")
        
        return f"""
        <html>
        <body style="font-family: Arial, sans-serif; margin: 0; padding: 20px;">
            <div style="border-left: 4px solid {color}; padding-left: 20px;">
                <h2 style="color: {color}; margin-top: 0;">
                    {alert.severity.value.upper()} Alert: {alert.title}
                </h2>
                
                <p><strong>Description:</strong> {alert.description}</p>
                
                <table style="border-collapse: collapse; width: 100%; margin-top: 20px;">
                    <tr>
                        <td style="border: 1px solid #ddd; padding: 8px; background-color: #f9f9f9;"><strong>Alert ID:</strong></td>
                        <td style="border: 1px solid #ddd; padding: 8px;">{alert.alert_id}</td>
                    </tr>
                    <tr>
                        <td style="border: 1px solid #ddd; padding: 8px; background-color: #f9f9f9;"><strong>Category:</strong></td>
                        <td style="border: 1px solid #ddd; padding: 8px;">{alert.category.value}</td>
                    </tr>
                    <tr>
                        <td style="border: 1px solid #ddd; padding: 8px; background-color: #f9f9f9;"><strong>Created At:</strong></td>
                        <td style="border: 1px solid #ddd; padding: 8px;">{alert.created_at.strftime('%Y-%m-%d %H:%M:%S UTC')}</td>
                    </tr>
                    {"<tr><td style='border: 1px solid #ddd; padding: 8px; background-color: #f9f9f9;'><strong>Component:</strong></td><td style='border: 1px solid #ddd; padding: 8px;'>" + alert.affected_component + "</td></tr>" if alert.affected_component else ""}
                    {"<tr><td style='border: 1px solid #ddd; padding: 8px; background-color: #f9f9f9;'><strong>Metric Value:</strong></td><td style='border: 1px solid #ddd; padding: 8px;'>" + str(alert.metric_value) + "</td></tr>" if alert.metric_value is not None else ""}
                    {"<tr><td style='border: 1px solid #ddd; padding: 8px; background-color: #f9f9f9;'><strong>Threshold:</strong></td><td style='border: 1px solid #ddd; padding: 8px;'>" + str(alert.threshold_value) + "</td></tr>" if alert.threshold_value is not None else ""}
                </table>
                
                <p style="margin-top: 20px; font-size: 12px; color: #666;">
                    This is an automated alert from the AI Assistant monitoring system.
                </p>
            </div>
        </body>
        </html>
        """


class SlackNotificationSender(NotificationSender):
    """Slack notification sender."""
    
    def __init__(self, logger):
        self.logger = logger
    
    def can_handle(self, channel: NotificationChannel) -> bool:
        return channel == NotificationChannel.SLACK
    
    async def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate Slack configuration."""
        return 'webhook_url' in config or ('bot_token' in config and 'channel' in config)
    
    async def send_notification(self, alert: Alert, target: NotificationTarget) -> bool:
        """Send Slack notification."""
        try:
            config = target.config
            
            # Create Slack message
            message = self._create_slack_message(alert)
            
            if 'webhook_url' in config:
                # Use webhook
                response = requests.post(
                    config['webhook_url'],
                    json=message,
                    timeout=10
                )
                response.raise_for_status()
            else:
                # Use bot token
                headers = {
                    'Authorization': f"Bearer {config['bot_token']}",
                    'Content-Type': 'application/json'
                }
                
                payload = {
                    'channel': config['channel'],
                    **message
                }
                
                response = requests.post(
                    'https://slack.com/api/chat.postMessage',
                    headers=headers,
                    json=payload,
                    timeout=10
                )
                response.raise_for_status()
            
            self.logger.info(f"Slack notification sent for alert {alert.alert_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send Slack notification: {str(e)}")
            return False
    
    def _create_slack_message(self, alert: Alert) -> Dict[str, Any]:
        """Create Slack message payload."""
        severity_colors = {
            AlertSeverity.INFO: "#36a64f",
            AlertSeverity.WARNING: "#ff9500",
            AlertSeverity.ERROR: "#ff0000",
            AlertSeverity.CRITICAL: "#8b0000",
            AlertSeverity.EMERGENCY: "#4b0082"
        }
        
        color = severity_colors.get(alert.severity, "#cccccc")
        
        fields = [
            {"title": "Alert ID", "value": alert.alert_id, "short": True},
            {"title": "Category", "value": alert.category.value, "short": True},
            {"title": "Created At", "value": alert.created_at.strftime('%Y-%m-%d %H:%M:%S UTC'), "short": True}
        ]
        
        if alert.affected_component:
            fields.append({"title": "Component", "value": alert.affected_component, "short": True})
        
        if alert.metric_value is not None:
            fields.append({"title": "Value", "value": str(alert.metric_value), "short": True})
        
        if alert.threshold_value is not None:
            fields.append({"title": "Threshold", "value": str(alert.threshold_value), "short": True})
        
        return {
            "attachments": [{
                "color": color,
                "title": f"{alert.severity.value.upper()} Alert",
                "text": alert.title,
                "fields": fields,
                "footer": "AI Assistant Monitoring",
                "ts": int(alert.created_at.timestamp())
            }]
        }


class WebhookNotificationSender(NotificationSender):
    """Generic webhook notification sender."""
    
    def __init__(self, logger):
        self.logger = logger
    
    def can_handle(self, channel: NotificationChannel) -> bool:
        return channel == NotificationChannel.WEBHOOK
    
    async def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate webhook configuration."""
        return 'url' in config
    
    async def send_notification(self, alert: Alert, target: NotificationTarget) -> bool:
        """Send webhook notification."""
        try:
            config = target.config
            
            # Create webhook payload
            payload = {
                'alert_id': alert.alert_id,
                'rule_id': alert.rule_id,
                'title': alert.title,
                'description': alert.description,
                'severity': alert.severity.value,
                'category': alert.category.value,
                'status': alert.status.value,
                'created_at': alert.created_at.isoformat(),
                'affected_component': alert.affected_component,
                'metric_value': alert.metric_value,
                'threshold_value': alert.threshold_value,
                'context': alert.context,
                'labels': alert.labels,
                'annotations': alert.annotations
            }
            
            headers = config.get('headers', {})
            headers.setdefault('Content-Type', 'application/json')
            
            response = requests.post(
                config['url'],
                json=payload,
                headers=headers,
                timeout=config.get('timeout', 10)
            )
            response.raise_for_status()
            
            self.logger.info(f"Webhook notification sent for alert {alert.alert_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send webhook notification: {str(e)}")
            return False


class ConsoleNotificationSender(NotificationSender):
    """Console/log notification sender."""
    
    def __init__(self, logger):
        self.logger = logger
    
    def can_handle(self, channel: NotificationChannel) -> bool:
        return channel in [NotificationChannel.CONSOLE, NotificationChannel.LOG]
    
    async def validate_config(self, config: Dict[str, Any]) -> bool:
        """Console notifications don't need configuration."""
        return True
    
    async def send_notification(self, alert: Alert, target: NotificationTarget) -> bool:
        """Send console/log notification."""
        try:
            message = (
                f"[{alert.severity.value.upper()}] ALERT: {alert.title}\n"
                f"  Description: {alert.description}\n"
                f"  Alert ID: {alert.alert_id}\n"
                f"  Category: {alert.category.value}\n"
                f"  Created: {alert.created_at.strftime('%Y-%m-%d %H:%M:%S UTC')}"
            )
            
            if alert.affected_component:
                message += f"\n  Component: {alert.affected_component}"
            
            if alert.metric_value is not None:
                message += f"\n  Value: {alert.metric_value}"
            
            if alert.threshold_value is not None:
                message += f"\n  Threshold: {alert.threshold_value}"
            
            if target.channel == NotificationChannel.CONSOLE:
                print(f"\n{'='*60}")
                print(message)
                print('='*60)
            else:
                # Log with appropriate level
                if alert.severity == AlertSeverity.CRITICAL:
                    self.logger.critical(message)
                elif alert.severity == AlertSeverity.ERROR:
                    self.logger.error(message)
                elif alert.severity == AlertSeverity.WARNING:
                    self.logger.warning(message)
                else:
                    self.logger.info(message)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send console notification: {str(e)}")
            return False


class AlertManager:
    """Manages alert lifecycle and state."""
    
    def __init__(self, logger, database: Optional[DatabaseManager] = None):
        self.logger = logger
        self.database = database
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: deque = deque(maxlen=10000)
        self.alert_lock = asyncio.Lock()
    
    async def create_alert(self, alert: Alert) -> None:
        """Create a new alert."""
        async with self.alert_lock:
            self.active_alerts[alert.alert_id] = alert
            
            # Store in database if available
            if self.database:
                try:
                    await self.database.execute(
                        """
                        INSERT INTO alerts (
                            alert_id, rule_id, title, description, severity, category,
                            status, created_at, affected_component, metric_value, threshold_value,
                            context, labels, annotations
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            alert.alert_id, alert.rule_id, alert.title, alert.description,
                            alert.severity.value, alert.category.value, alert.status.value,
                            alert.created_at, alert.affected_component, alert.metric_value,
                            alert.threshold_value, json.dumps(alert.context),
                            json.dumps(alert.labels), json.dumps(alert.annotations)
                        )
                    )
                except Exception as e:
                    self.logger.warning(f"Failed to store alert in database: {str(e)}")
    
    async def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """Acknowledge an alert."""
        async with self.alert_lock:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.status = AlertStatus.ACKNOWLEDGED
                alert.acknowledged_at = datetime.now(timezone.utc)
                alert.acknowledged_by = acknowledged_by
                
                # Update database
                if self.database:
                    try:
                        await self.database.execute(
                            """
                            UPDATE alerts SET status = ?, acknowledged_at = ?, acknowledged_by = ?
                            WHERE alert_id = ?
                            """,
                            (alert.status.value, alert.acknowledged_at, acknowledged_by, alert_id)
                        )
                    except Exception as e:
                        self.logger.warning(f"Failed to update alert in database: {str(e)}")
                
                return True
            return False
    
    async def resolve_alert(self, alert_id: str, resolved_by: str = None, 
                          resolution_note: str = None, auto_resolved: bool = False) -> bool:
        """Resolve an alert."""
        async with self.alert_lock:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.status = AlertStatus.RESOLVED
                alert.resolved_at = datetime.now(timezone.utc)
                alert.resolved_by = resolved_by
                alert.resolution_note = resolution_note
                alert.auto_resolved = auto_resolved
                
                # Move to history
                self.alert_history.append(alert)
                del self.active_alerts[alert_id]
                
                # Update database
                if self.database:
                    try:
                        await self.database.execute(
                            """
                            UPDATE alerts SET status = ?, resolved_at = ?, resolved_by = ?,
                            resolution_note = ?, auto_resolved = ?
                            WHERE alert_id = ?
                            """,
                            (
                                alert.status.value, alert.resolved_at, resolved_by,
                                resolution_note, auto_resolved, alert_id
                            )
                        )
                    except Exception as e:
                        self.logger.warning(f"Failed to update alert in database: {str(e)}")
                
                return True
            return False
    
    async def get_active_alerts(self, 
                              severity_filter: Optional[Set[AlertSeverity]] = None,
                              category_filter: Optional[Set[AlertCategory]] = None) -> List[Alert]:
        """Get active alerts with optional filtering."""
        async with self.alert_lock:
            alerts = list(self.active_alerts.values())
            
            if severity_filter:
                alerts = [a for a in alerts if a.severity in severity_filter]
            
            if category_filter:
                alerts = [a for a in alerts if a.category in category_filter]
            
            return sorted(alerts, key=lambda a: a.created_at, reverse=True)
    
    async def cleanup_old_alerts(self, max_age: timedelta = timedelta(hours=24)) -> int:
        """Clean up old resolved alerts."""
        cutoff_time = datetime.now(timezone.utc) - max_age
        cleaned_count = 0
        
        # Clean from memory
        self.alert_history = deque(
            [alert for alert in self.alert_history if alert.resolved_at and alert.resolved_at > cutoff_time],
            maxlen=10000
        )
        
        # Clean from database
        if self.database:
            try:
                result = await self.database.execute(
                    "DELETE FROM alerts WHERE status = 'resolved' AND resolved_at < ?",
                    (cutoff_time,)
                )
                cleaned_count = result.rowcount if result else 0
            except Exception as e:
                self.logger.warning(f"Failed to clean old alerts from database: {str(e)}")
        
        return cleaned_count


class AlertingEngine:
    """
    Advanced Alerting System for the AI Assistant.
    
    This engine provides comprehensive alerting capabilities including:
    - Rule-based and threshold-based alerting
    - Multiple notification channels (email, Slack, webhooks, etc.)
    - Alert lifecycle management (creation, acknowledgment, resolution)
    - Escalation policies and suppression rules
    - Integration with core system components
    - Real-time alert evaluation and delivery
    - Performance monitoring and optimization
    - Customizable alert templates and formatting
    - Alert analytics and reporting
    """
    
    def __init__(self, container: Container):
        """
        Initialize the alerting engine.
        
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
        
        # Monitoring components
        self.metrics = container.get(MetricsCollector)
        self.tracer = container.get(TraceManager)
        
        # Assistant components
        try:
            self.core_engine = container.get(EnhancedCoreEngine)
            self.component_manager = container.get(EnhancedComponentManager)
            self.session_manager = container.get(EnhancedSessionManager)
            self.workflow_orchestrator = container.get(WorkflowOrchestrator)
            self.plugin_manager = container.get(EnhancedPluginManager)
        except Exception as e:
            self.logger.warning(f"Some assistant components not available: {str(e)}")
            self.core_engine = None
            self.component_manager = None
            self.session_manager = None
            self.workflow_orchestrator = None
            self.plugin_manager = None
        
        # Storage components
        try:
            self.database = container.get(DatabaseManager)
            self.redis_cache = container.get(RedisCache)
        except Exception:
            self.database = None
            self.redis_cache = None
        
        # Alert management
        self.alert_rules: Dict[str, AlertRule] = {}
        self.notification_targets: Dict[str, NotificationTarget] = {}
        self.alert_manager = AlertManager(self.logger, self.database)
        
        # Notification infrastructure
        self.notification_senders: List[NotificationSender] = []
        self.notification_queue: asyncio.Queue = asyncio.Queue()
        self.notification_semaphore = asyncio.Semaphore(5)  # Max concurrent notifications
        
        # State management
        self.rule_evaluations: Dict[str, datetime] = {}
        self.alert_counts: Dict[str, int] = defaultdict(int)
        self.suppressed_alerts: Set[str] = set()
        
        # Performance tracking
        self.evaluation_times: deque = deque(maxlen=1000)
        self.notification_times: deque = deque(maxlen=1000)
        
        # Configuration
        self.evaluation_interval = self.config.get("alerting.evaluation_interval", 60.0)
        self.notification_timeout = self.config.get("alerting.notification_timeout", 30.0)
        self.max_concurrent_notifications = self.config.get("alerting.max_concurrent_notifications", 5)
        self.enable_auto_resolution = self.config.get("alerting.enable_auto_resolution", True)
        
        # Background tasks
        self.evaluation_task: Optional[asyncio.Task] = None
        self.notification_task: Optional[asyncio.Task] = None
        self.cleanup_task: Optional[asyncio.Task] = None
        
        # Initialize components
        self._setup_notification_senders()
        self._setup_builtin_rules()
        self._setup_monitoring()
        
        # Register health check
        self.health_check.register_component("alerting_engine", self._health_check_callback)
        
        self.logger.info("AlertingEngine initialized successfully")

    def _setup_notification_senders(self) -> None:
        """Setup notification senders for different channels."""
        try:
            self.notification_senders = [
                EmailNotificationSender(self.logger),
                SlackNotificationSender(self.logger),
                WebhookNotificationSender(self.logger),
                ConsoleNotificationSender(self.logger)
            ]
            
            self.logger.info(f"Initialized {len(self.notification_senders)} notification senders")
            
        except Exception as e:
            self.logger.error(f"Failed to setup notification senders: {str(e)}")

    def _setup_builtin_rules(self) -> None:
        """Setup built-in alert rules."""
        try:
            # System health rules
            self.register_alert_rule(AlertRule(
                rule_id="system_cpu_high",
                name="High System CPU Usage",
                description="System CPU usage is above 80%",
                metric_name="system_cpu_usage_percent",
                threshold_value=80.0,
                operator=">",
                severity=AlertSeverity.WARNING,
                category=AlertCategory.PERFORMANCE,
                notification_channels=[NotificationChannel.EMAIL, NotificationChannel.LOG]
            ))
            
            self.register_alert_rule(AlertRule(
                rule_id="system_memory_high",
                name="High System Memory Usage",
                description="System memory usage is above 85%",
                metric_name="system_memory_usage_percent",
                threshold_value=85.0,
                operator=">",
                severity=AlertSeverity.WARNING,
                category=AlertCategory.PERFORMANCE,
                notification_channels=[NotificationChannel.EMAIL, NotificationChannel.LOG]
            ))
            
            # Component health rules
            self.register_alert_rule(AlertRule(
                rule_id="component_unhealthy",
                name="Component Health Check Failed",
                description="A critical component has failed its health check",
                custom_condition=self._check_component_health,
                severity=AlertSeverity.ERROR,
                category=AlertCategory.COMPONENT_FAILURE,
                notification_channels=[NotificationChannel.EMAIL, NotificationChannel.SLACK, NotificationChannel.LOG]
            ))
            
            # Session management rules
            self.register_alert_rule(AlertRule(
                rule_id="session_failure_rate_high",
                name="High Session Failure Rate",
                description="Session failure rate is above 10%",
                metric_name="session_failure_rate_percent",
                threshold_value=10.0,
                operator=">",
                severity=AlertSeverity.ERROR,
                category=AlertCategory.USER_EXPERIENCE,
                notification_channels=[NotificationChannel.EMAIL, NotificationChannel.LOG]
            ))
            
            # Workflow failure rules
            self.register_alert_rule(AlertRule(
                rule_id="workflow_failure_rate_high",
                name="High Workflow Failure Rate",
                description="Workflow failure rate is above 5%",
                metric_name="workflow_failure_rate_percent",
                threshold_value=5.0,
                operator=">",
                severity=AlertSeverity.WARNING,
                category=AlertCategory.WORKFLOW_ERROR,
                notification_channels=[NotificationChannel.EMAIL, NotificationChannel.LOG]
            ))
            
            # Response time rules
            self.register_alert_rule(AlertRule(
                rule_id="response_time_high",
                name="High Response Time",
                description="Average response time is above 5 seconds",
                metric_name="average_response_time_seconds",
                threshold_value=5.0,
                operator=">",
                severity=AlertSeverity.WARNING,
                category=AlertCategory.PERFORMANCE,
                notification_channels=[NotificationChannel.LOG]
            ))
            
            self.logger.info("Built-in alert rules registered")
            
        except Exception as e:
            self.logger.error(f"Failed to setup built-in rules: {str(e)}")

    def _setup_monitoring(self) -> None:
        """Setup monitoring and metrics."""
        try:
            # Register alerting metrics
            self.metrics.register_counter("alerts_created_total")
            self.metrics.register_counter("alerts_resolved_total")
            self.metrics.register_counter("notifications_sent_total")
            self.metrics.register_counter("notifications_failed_total")
            self.metrics.register_gauge("active_alerts")
            self.metrics.register_histogram("alert_evaluation_duration_seconds")
            self.metrics.register_histogram("notification_delivery_duration_seconds")
            
        except Exception as e:
            self.logger.warning(f"Failed to setup monitoring: {str(e)}")

    async def initialize(self) -> None:
        """Initialize the alerting engine."""
        try:
            # Load configuration
            await self._load_alert_configuration()
            
            # Register event handlers
            await self._register_event_handlers()
            
            # Start background tasks
            self.evaluation_task = asyncio.create_task(self._rule_evaluation_loop())
            self.notification_task = asyncio.create_task(self._notification_delivery_loop())
            self.cleanup_task = asyncio.create_task(self._cleanup_loop())
            
            # Initialize database schema if needed
            if self.database:
                await self._initialize_database_schema()
            
            self.logger.info("AlertingEngine initialization completed")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize AlertingEngine: {str(e)}")
            raise AlertingError(f"Initialization failed: {str(e)}")

    async def _load_alert_configuration(self) -> None:
        """Load alert configuration from files."""
        try:
            # Load notification targets
            targets_config = self.config.get("alerting.notification_targets", {})
            for target_id, target_config in targets_config.items():
                channel = NotificationChannel(target_config.get("channel", "email"))
                target = NotificationTarget(
                    target_id=target_id,
                    channel=channel,
                    config=target_config.get("config", {}),
                    severity_filter=set(AlertSeverity(s) for s in target_config.get("severity_filter", [])),
                    category_filter=set(AlertCategory(c) for c in target_config.get("category_filter", [])),
                    enabled=target_config.get("enabled", True)
                )
                self.notification_targets[target_id] = target
            
            # Load custom alert rules from configuration
            rules_config = self.config.get("alerting.custom_rules", {})
            for rule_id, rule_config in rules_config.items():
                rule = AlertRule(
                    rule_id=rule_id,
                    name=rule_config.get("name", rule_id),
                    description=rule_config.get("description", ""),
                    metric_name=rule_config.get("metric_name"),
                    threshold_value=rule_config.get("threshold_value"),
                    operator=rule_config.get("operator", ">"),
                    severity=AlertSeverity(rule_config.get("severity", "warning")),
                    category=AlertCategory(rule_config.get("category", "system_health")),
                    notification_channels=[
                        NotificationChannel(ch) for ch in rule_config.get("notification_channels", ["log"])
                    ],
                    enabled=rule_config.get("enabled", True)
                )
                self.register_alert_rule(rule)
            
            self.logger.info(f"Loaded {len(self.notification_targets)} notification targets and {len(rules_config)} custom rules")
            
        except Exception as e:
            self.logger.warning(f"Failed to load alert configuration: {str(e)}")

    async def _register_event_handlers(self) -> None:
        """Register event handlers for system events."""
        # Component health events
        self.event_bus.subscribe("component_health_changed", self._handle_component_health_change)
        
        # System state events
        self.event_bus.subscribe("system_state_changed", self._handle_system_state_change)
        
        # Error events
        self.event_bus.subscribe("error_occurred", self._handle_error_event)
        self.event_bus.subscribe("processing_error", self._handle_processing_error)
        self.event_bus.subscribe("workflow_failed", self._handle_workflow_failure)
        self.event_bus.subscribe("plugin_error", self._handle_plugin_error)
        
        # Performance events
        self.event_bus.subscribe("performance_threshold_exceeded", self._handle_performance_threshold)
        
        # Security events
        self.event_bus.subscribe("security_violation", self._handle_security_violation)
        
        # User interaction events
        self.event_bus.subscribe("user_interaction_failed", self._handle_interaction_failure)

    async def _initialize_database_schema(self) -> None:
        """Initialize database schema for alerts."""
        try:
            await self.database.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    alert_id TEXT PRIMARY KEY,
                    rule_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    category TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    acknowledged_at TIMESTAMP,
                    resolved_at TIMESTAMP,
                    acknowledged_by TEXT,
                    resolved_by TEXT,
                    affected_component TEXT,
                    metric_value REAL,
                    threshold_value REAL,
                    context TEXT,
                    labels TEXT,
                    annotations TEXT,
                    resolution_note TEXT,
                    auto_resolved BOOLEAN DEFAULT FALSE
                )
            """)
            
            await self.database.execute("""
                CREATE INDEX IF NOT EXISTS idx_alerts_status ON alerts(status)
            """)
            
            await self.database.execute("""
                CREATE INDEX IF NOT EXISTS idx_alerts_severity ON alerts(severity)
            """)
            
            await self.database.execute("""
                CREATE INDEX IF NOT EXISTS idx_alerts_created_at ON alerts(created_at)
            """)
            
        except Exception as e:
            self.logger.warning(f"Failed to initialize database schema: {str(e)}")

    @handle_exceptions
    def register_alert_rule(self, rule: AlertRule) -> None:
        """
        Register an alert rule.
        
        Args:
            rule: Alert rule to register
        """
        self.alert_rules[rule.rule_id] = rule
        self.logger.info(f"Registered alert rule: {rule.rule_id} ({rule.name})")

    @handle_exceptions
    def register_notification_target(self, target: NotificationTarget) -> None:
        """
        Register a notification target.
        
        Args:
            target: Notification target to register
        """
        self.notification_targets[target.target_id] = target
        self.logger.info(f"Registered notification target: {target.target_id} ({target.channel.value})")

    @handle_exceptions
    async def create_alert(
        self,
        rule_id: str,
        title: str,
        description: str,
        severity: AlertSeverity = AlertSeverity.WARNING,
        category: AlertCategory = AlertCategory.SYSTEM_HEALTH,
        affected_component: Optional[str] = None,
        metric_value: Optional[float] = None,
        threshold_value: Optional[float] = None,
        context: Optional[Dict[str, Any]] = None,
        labels: Optional[Dict[str, str]] = None,
        annotations: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Create a new alert.
        
        Args:
            rule_id: ID of the rule that triggered this alert
            title: Alert title
            description: Alert description
            severity: Alert severity level
            category: Alert category
            affected_component: Name of affected component
            metric_value: Current metric value
            threshold_value: Threshold that was exceeded
            context: Additional context data
            labels: Alert labels
            annotations: Alert annotations
            
        Returns:
            Alert ID
        """
        alert_id = str(uuid.uuid4())
        
        # Check for suppression
        if self._is_alert_suppressed(rule_id, context or {}):
            self.logger.debug(f"Alert suppressed for rule {rule_id}")
            return alert_id
        
        # Check rate limiting
        current_time = datetime.now(timezone.utc)
        rule = self.alert_rules.get(rule_id)
        if rule and self._is_rate_limited(rule, current_time):
            self.logger.debug(f"Alert rate limited for rule {rule_id}")
            return alert_id
        
        # Create alert
        alert = Alert(
            alert_id=alert_id,
            rule_id=rule_id,
            rule_name=rule.name if rule else rule_id,
            title=title,
            description=description,
            severity=severity,
            category=category,
            affected_component=affected_component,
            metric_value=metric_value,
            threshold_value=threshold_value,
            context=context or {},
            labels=labels or {},
            annotations=annotations or {}
        )
        
        # Store alert
        await self.alert_manager.create_alert(alert)
        
        # Update rule statistics
        if rule:
            rule.last_triggered = current_time
            rule.trigger_count += 1
        
        # Queue for notification
        await self.notification_queue.put(alert)
        
        # Update metrics
        self.metrics.increment("alerts_created_total")
        self.metrics.set("active_alerts", len(self.alert_manager.active_alerts))
        
        self.logger.info(f"Created alert: {alert_id} ({title}) - {severity.value}")
        return alert_id

    def _is_alert_suppressed(self, rule_id: str, context: Dict[str, Any]) -> bool:
        """Check if alert should be suppressed."""
        # Check global suppression
        if rule_id in self.suppressed_alerts:
            return True
        
        # Check component-specific suppression
        component = context.get("component")
        if component and f"{rule_id}:{component}" in self.suppressed_alerts:
            return True
        
        return False

    def _is_rate_limited(self, rule: AlertRule, current_time: datetime) -> bool:
        """Check if alert is rate limited."""
        if not rule.last_triggered:
            return False
        
        # Check if we're within the rate limit window
        time_window = timedelta(hours=1)
        if current_time - rule.last_triggered < time_window:
            # Count alerts in the last hour
            hour_ago = current_time - time_window
            rule_key = f"alerts_hour_{rule.rule_id}"
            
            if rule_key in self.alert_counts:
                if self.alert_counts[rule_key] >= rule.max_alerts_per_hour:
                    return True
                self.alert_counts[rule_key] += 1
            else:
                self.alert_counts[rule_key] = 1
        else:
            # Reset counter for new hour
            rule_key = f"alerts_hour_{rule.rule_id}"
            self.alert_counts[rule_key] = 1
        
        return False

    async def _rule_evaluation_loop(self) -> None:
        """Background task for evaluating alert rules."""
        while True:
            try:
                evaluation_start = time.time()
                
                # Evaluate all enabled rules
                for rule in self.alert_rules.values():
                    if not rule.enabled:
                        continue
                    
                    try:
                        await self._evaluate_rule(rule)
                    except Exception as e:
                        self.logger.error(f"Error evaluating rule {rule.rule_id}: {str(e)}")
                
                # Auto-resolve alerts if enabled
                if self.enable_auto_resolution:
                    await self._auto_resolve_alerts()
                
                # Record evaluation time
                evaluation_time = time.time() - evaluation_start
                self.evaluation_times.append(evaluation_time)
                self.metrics.record("alert_evaluation_duration_seconds", evaluation_time)
                
                # Sleep until next evaluation
                await asyncio.sleep(self.evaluation_interval)
                
            except Exception as e:
                self.logger.error(f"Error in rule evaluation loop: {str(e)}")
                await asyncio.sleep(self.evaluation_interval)

    async def _evaluate_rule(self, rule: AlertRule) -> None:
        """Evaluate a single alert rule."""
        current_time = datetime.now(timezone.utc)
        
        # Check if it's time to evaluate this rule
        last_eval = self.rule_evaluations.get(rule.rule_id)
        if last_eval and (current_time - last_eval) < rule.evaluation_interval:
            return
        
        self.rule_evaluations[rule.rule_id] = current_time
        
        try:
            # Evaluate custom condition if present
            if rule.custom_condition:
                context = await self._gather_evaluation_context()
                if rule.custom_condition(context):
                    await self._trigger_alert_from_rule(rule, context)
                return
            
            # Evaluate metric-based condition
            if rule.metric_name:
                metric_value = await self._get_metric_value(rule.metric_name)
                if metric_value is not None and self._evaluate_threshold(
                    metric_value, rule.threshold_value, rule.operator
                ):
                    context = {
                        'metric_name': rule.metric_name,
                        'metric_value': metric_value,
                        'threshold_value': rule.threshold_value,
                        'operator': rule.operator
                    }
                    await self._trigger_alert_from_rule(rule, context, metric_value)
            
        except Exception as e:
            self.logger.error(f"Error evaluating rule {rule.rule_id}: {str(e)}")

    async def _gather_evaluation_context(self) -> Dict[str, Any]:
        """Gather context data for rule evaluation."""
        context = {}
        
        try:
            # System metrics
            if hasattr(self.metrics, 'get_metric'):
                context['system_cpu_usage_percent'] = await self.metrics.get_metric('system_cpu_usage_percent', 0.0)
                context['system_memory_usage_percent'] = await self.metrics.get_metric('system_memory_usage_percent', 0.0)
                context['active_sessions'] = await self.metrics.get_metric('active_sessions', 0)
                context['active_workflows'] = await self.metrics.get_metric('active_workflows', 0)
                context['average_response_time_seconds'] = await self.metrics.get_metric('average_response_time_seconds', 0.0)
            
            # Component health
            if self.component_manager:
                component_status = self.component_manager.get_component_status()
                context['component_health'] = component_status
                context['failed_components'] = component_status.get('failed_components', 0)
                context['total_components'] = component_status.get('total_components', 0)
            
            # Session information
            if self.session_manager:
                session_stats = self.session_manager.get_session_statistics()
                context['session_stats'] = session_stats
                if session_stats.get('total_active_sessions', 0) > 0:
                    # Calculate failure rate
                    failed_sessions = session_stats.get('failed_sessions', 0)
                    total_sessions = session_stats.get('total_active_sessions', 1)
                    context['session_failure_rate_percent'] = (failed_sessions / total_sessions) * 100
                else:
                    context['session_failure_rate_percent'] = 0.0
            
            # Workflow information
            if self.workflow_orchestrator:
                try:
                    active_executions = self.workflow_orchestrator.get_active_executions()
                    context['active_workflow_executions'] = len(active_executions)
                    
                    # Calculate workflow failure rate
                    failed_workflows = len([e for e in active_executions if e.get('state') == 'failed'])
                    total_workflows = len(active_executions)
                    if total_workflows > 0:
                        context['workflow_failure_rate_percent'] = (failed_workflows / total_workflows) * 100
                    else:
                        context['workflow_failure_rate_percent'] = 0.0
                except Exception:
                    context['workflow_failure_rate_percent'] = 0.0
            
        except Exception as e:
            self.logger.warning(f"Error gathering evaluation context: {str(e)}")
        
        return context

    async def _get_metric_value(self, metric_name: str) -> Optional[float]:
        """Get current value of a metric."""
        try:
            if hasattr(self.metrics, 'get_metric'):
                return await self.metrics.get_metric(metric_name)
            return None
        except Exception:
            return None

    def _evaluate_threshold(self, value: float, threshold: float, operator: str) -> bool:
        """Evaluate threshold condition."""
        if operator == ">":
            return value > threshold
        elif operator == ">=":
            return value >= threshold
        elif operator == "<":
            return value < threshold
        elif operator == "<=":
            return value <= threshold
        elif operator == "==":
            return value == threshold
        elif operator == "!=":
            return value != threshold
        else:
            return False

    async def _trigger_alert_from_rule(
        self,
        rule: AlertRule,
        context: Dict[str, Any],
        metric_value: Optional[float] = None
    ) -> None:
        """Trigger an alert from a rule evaluation."""
        # Check if similar alert already exists
        existing_alerts = await self.alert_manager.get_active_alerts()
        for alert in existing_alerts:
            if (alert.rule_id == rule.rule_id and 
                alert.affected_component == context.get('affected_component')):
                # Similar alert already exists
                return
        
        await self.create_alert(
            rule_id=rule.rule_id,
            title=rule.name,
            description=rule.description,
            severity=rule.severity,
            category=rule.category,
            affected_component=context.get('affected_component'),
            metric_value=metric_value,
            threshold_value=rule.threshold_value,
            context=context
        )

    def _check_component_health(self, context: Dict[str, Any]) -> bool:
        """Custom condition to check component health."""
        component_health = context.get('component_health', {})
        failed_components = component_health.get('failed_components', 0)
        return failed_components > 0

    async def _auto_resolve_alerts(self) -> None:
        """Auto-resolve alerts that are no longer triggered."""
        if not self.enable_auto_resolution:
            return
        
        current_time = datetime.now(timezone.utc)
        active_alerts = await self.alert_manager.get_active_alerts()
        
        for alert in active_alerts:
            rule = self.alert_rules.get(alert.rule_id)
            if not rule or not rule.auto_resolve:
                continue
            
            # Check if alert should auto-resolve
            if current_time - alert.created_at > rule.auto_resolve_timeout:
                # Re-evaluate the condition
                should_resolve = False
                
                if rule.custom_condition:
                    context = await self._gather_evaluation_context()
                    should_resolve = not rule.custom_condition(context)
                elif rule.metric_name:
                    metric_value = await self._get_metric_value(rule.metric_name)
                    if metric_value is not None:
                        should_resolve = not self._evaluate_threshold(
                            metric_value, rule.threshold_value, rule.operator
                        )
                
                if should_resolve:
                    await self.alert_manager.resolve_alert(
                        alert.alert_id,
                        resolution_note="Auto-resolved: condition no longer met",
                        auto_resolved=True
                    )
                    self.metrics.increment("alerts_resolved_total")

    async def _notification_delivery_loop(self) -> None:
        """Background task for delivering notifications."""
        while True:
            try:
                # Get alert from queue
                alert = await self.notification_queue.get()
                
                # Deliver notifications
                await self._deliver_notifications(alert)
                
                # Mark task as done
                self.notification_queue.task_done()
                
            except Exception as e:
                self.logger.error(f"Error in notification delivery loop: {str(e)}")

    async def _deliver_notifications(self, alert: Alert) -> None:
        """Deliver notifications for an alert."""
        rule = self.alert_rules.get(alert.rule_id)
        if not rule:
            return
        
        # Find matching notification targets
        targets = self._find_notification_targets(alert, rule)
        
        if not targets:
            self.logger.warning(f"No notification targets found for alert {alert.alert_id}")
            return
        
        # Send notifications concurrently
        async with self.notification_semaphore:
            notification_start = time.time()
            tasks = []
            
            for target in targets:
                if self._should_send_notification(target, alert):
                    task = asyncio.create_task(self._send_notification(alert, target))
                    tasks.append(task)
            
            if tasks:
                # Wait for all notifications to complete
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Track results
                successful_notifications = 0
                failed_notifications = 0
                
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        failed_notifications += 1
                        self.logger.error(f"Notification failed: {str(result)}")
                    elif result:
                        successful_notifications += 1
                    else:
                        failed_notifications += 1
                
                # Record metrics
                notification_time = time.time() - notification_start
                self.notification_times.append(notification_time)
                self.metrics.record("notification_delivery_duration_seconds", notification_time)
                self.metrics.increment("notifications_sent_total", value=successful_notifications)
                self.metrics.increment("notifications_failed_total", value=failed_notifications)
                
                self.logger.info(
                    f"Sent {successful_notifications} notifications for alert {alert.alert_id} "
                    f"({failed_notifications} failed) in {notification_time:.2f}s"
                )

    def _find_notification_targets(self, alert: Alert, rule: AlertRule) -> List[NotificationTarget]:
        """Find notification targets for an alert."""
        targets = []
        
        # Check rule-specific channels
        for channel in rule.notification_channels:
            for target in self.notification_targets.values():
                if (target.channel == channel and 
                    target.enabled and
                    self._target_matches_alert(target, alert)):
                    targets.append(target)
        
        return targets

    def _target_matches_alert(self, target: NotificationTarget, alert: Alert) -> bool:
        """Check if a target matches an alert's filters."""
        # Check severity filter
        if target.severity_filter and alert.severity not in target.severity_filter:
            return False
        
        # Check category filter
        if target.category_filter and alert.category not in target.category_filter:
            return False
        
        # Check component filter
        if (target.component_filter and alert.affected_component and 
            alert.affected_component not in target.component_filter):
            return False
        
        return True

    def _should_send_notification(self, target: NotificationTarget, alert: Alert) -> bool:
        """Check if notification should be sent to target."""
        current_time = datetime.now(timezone.utc)
        
        # Check quiet hours
        if (target.quiet_hours_start is not None and target.quiet_hours_end is not None):
            current_hour = current_time.hour
            if target.quiet_hours_start <= target.quiet_hours_end:
                # Same day quiet hours
                if target.quiet_hours_start <= current_hour < target.quiet_hours_end:
                    return False
            else:
                # Overnight quiet hours
                if current_hour >= target.quiet_hours_start or current_hour < target.quiet_hours_end:
                    return False
        
        # Check rate limiting
        if target.rate_limit_enabled and target.last_notification:
            hour_ago = current_time - timedelta(hours=1)
            if target.last_notification > hour_ago and target.notification_count >= target.max_notifications_per_hour:
                return False
        
        return True

    async def _send_notification(self, alert: Alert, target: NotificationTarget) -> bool:
        """Send notification to a specific target."""
        # Find appropriate sender
        sender = None
        for s in self.notification_senders:
            if s.can_handle(target.channel):
                sender = s
                break
        
        if not sender:
            self.logger.error(f"No sender found for channel {target.channel.value}")
            return False
        
        try:
            # Validate target configuration
            if not await sender.validate_config(target.config):
                self.logger.error(f"Invalid configuration for target {target.target_id}")
                return False
            
            # Send notification with timeout
            success = await asyncio.wait_for(
                sender.send_notification(alert, target),
                timeout=self.notification_timeout
            )
            
            if success:
                # Update target statistics
                target.last_notification = datetime.now(timezone.utc)
                target.notification_count += 1
                
                # Record notification in alert
                alert.notifications_sent.append({
                    'target_id': target.target_id,
                    'channel': target.channel.value,
                    'sent_at': target.last_notification.isoformat(),
                    'success': True
                })
            else:
                target
