"""
Advanced Health Check System for AI Assistant
Author: Drmusab
Last Modified: 2025-01-13 12:55:16 UTC

This module provides comprehensive health monitoring for the AI assistant system,
including component health tracking, dependency monitoring, automated recovery,
performance metrics, and predictive health analytics.
"""

from pathlib import Path
from typing import Optional, Dict, Any, List, Set, Callable, Type, Union, AsyncGenerator, TypeVar
import asyncio
import threading
import time
import psutil
import platform
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
import traceback
from concurrent.futures import ThreadPoolExecutor
import statistics
import subprocess
import socket
import requests
import aiohttp
import asyncpg
import redis

# Core imports
from src.core.config.loader import ConfigLoader
from src.core.events.event_bus import EventBus
from src.core.events.event_types import (
    HealthCheckStarted, HealthCheckCompleted, HealthCheckFailed,
    ComponentHealthChanged, DependencyHealthChanged, SystemHealthChanged,
    HealthThresholdExceeded, AutoRecoveryStarted, AutoRecoveryCompleted,
    HealthPredictionAlert, CircuitBreakerStateChanged, ErrorOccurred
)
from src.core.error_handling import ErrorHandler, handle_exceptions
from src.core.dependency_injection import Container

# Observability
from src.observability.monitoring.metrics import MetricsCollector
from src.observability.monitoring.tracing import TraceManager
from src.observability.logging.config import get_logger

# Type definitions
T = TypeVar('T')


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"
    UNKNOWN = "unknown"
    DOWN = "down"


class DependencyType(Enum):
    """Types of system dependencies."""
    DATABASE = "database"
    CACHE = "cache"
    MESSAGE_QUEUE = "message_queue"
    EXTERNAL_API = "external_api"
    FILE_SYSTEM = "file_system"
    NETWORK = "network"
    MODEL_SERVICE = "model_service"
    AUTHENTICATION = "authentication"
    STORAGE = "storage"


class CheckFrequency(Enum):
    """Health check frequencies."""
    CRITICAL = 5      # Every 5 seconds
    HIGH = 15         # Every 15 seconds
    NORMAL = 30       # Every 30 seconds
    LOW = 60          # Every 60 seconds
    BACKGROUND = 300  # Every 5 minutes


class RecoveryAction(Enum):
    """Automated recovery actions."""
    RESTART_COMPONENT = "restart_component"
    CIRCUIT_BREAKER_OPEN = "circuit_breaker_open"
    CIRCUIT_BREAKER_CLOSE = "circuit_breaker_close"
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    FAILOVER = "failover"
    ALERT_ONLY = "alert_only"
    GRACEFUL_DEGRADATION = "graceful_degradation"


@dataclass
class HealthMetrics:
    """Health metrics for a component or dependency."""
    response_time_ms: float = 0.0
    cpu_usage_percent: float = 0.0
    memory_usage_mb: float = 0.0
    memory_usage_percent: float = 0.0
    disk_usage_percent: float = 0.0
    network_latency_ms: float = 0.0
    error_rate: float = 0.0
    throughput: float = 0.0
    uptime_seconds: float = 0.0
    custom_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class HealthThreshold:
    """Health threshold configuration."""
    metric_name: str
    warning_threshold: float
    critical_threshold: float
    comparison: str = "greater_than"  # greater_than, less_than, equals
    recovery_threshold: Optional[float] = None
    consecutive_failures: int = 3
    time_window_seconds: int = 300


@dataclass
class HealthCheckResult:
    """Result of a health check operation."""
    component_id: str
    status: HealthStatus
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Metrics and details
    metrics: HealthMetrics = field(default_factory=HealthMetrics)
    details: Dict[str, Any] = field(default_factory=dict)
    
    # Error information
    error_message: Optional[str] = None
    error_type: Optional[str] = None
    error_stack: Optional[str] = None
    
    # Performance data
    check_duration_ms: float = 0.0
    
    # Health score (0.0 to 1.0)
    health_score: float = 1.0
    
    # Recommendations
    recommendations: List[str] = field(default_factory=list)


@dataclass
class DependencyHealth:
    """Health information for external dependencies."""
    dependency_id: str
    dependency_type: DependencyType
    endpoint: str
    status: HealthStatus
    
    # Connection details
    is_reachable: bool = False
    response_time_ms: float = 0.0
    last_successful_check: Optional[datetime] = None
    consecutive_failures: int = 0
    
    # Configuration
    timeout_seconds: float = 10.0
    retry_count: int = 3
    
    # Custom health data
    custom_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemHealth:
    """Overall system health status."""
    overall_status: HealthStatus
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Component health summary
    healthy_components: int = 0
    degraded_components: int = 0
    unhealthy_components: int = 0
    total_components: int = 0
    
    # Dependency health summary
    healthy_dependencies: int = 0
    unhealthy_dependencies: int = 0
    total_dependencies: int = 0
    
    # System metrics
    system_metrics: HealthMetrics = field(default_factory=HealthMetrics)
    
    # Alerts and recommendations
    active_alerts: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    # Health score (0.0 to 1.0)
    overall_health_score: float = 1.0


class HealthCheckError(Exception):
    """Custom exception for health check operations."""
    
    def __init__(self, message: str, component_id: Optional[str] = None, 
                 error_code: Optional[str] = None):
        super().__init__(message)
        self.component_id = component_id
        self.error_code = error_code
        self.timestamp = datetime.now(timezone.utc)


class HealthChecker(ABC):
    """Abstract base class for component health checkers."""
    
    @abstractmethod
    async def check_health(self) -> HealthCheckResult:
        """Perform health check and return result."""
        pass
    
    @abstractmethod
    def get_component_id(self) -> str:
        """Get the component identifier."""
        pass
    
    def get_check_frequency(self) -> CheckFrequency:
        """Get the preferred check frequency."""
        return CheckFrequency.NORMAL
    
    def get_thresholds(self) -> List[HealthThreshold]:
        """Get health thresholds for this component."""
        return []
    
    def supports_auto_recovery(self) -> bool:
        """Check if component supports automated recovery."""
        return False
    
    async def auto_recover(self, issue: str) -> bool:
        """Attempt automated recovery for the component."""
        return False


class DependencyChecker(ABC):
    """Abstract base class for dependency health checkers."""
    
    @abstractmethod
    async def check_dependency(self) -> DependencyHealth:
        """Check dependency health."""
        pass
    
    @abstractmethod
    def get_dependency_id(self) -> str:
        """Get the dependency identifier."""
        pass
    
    @abstractmethod
    def get_dependency_type(self) -> DependencyType:
        """Get the dependency type."""
        pass


class DatabaseHealthChecker(DependencyChecker):
    """Health checker for database connections."""
    
    def __init__(self, connection_string: str, dependency_id: str = "database"):
        self.connection_string = connection_string
        self.dependency_id = dependency_id
        self.logger = get_logger(__name__)
    
    def get_dependency_id(self) -> str:
        return self.dependency_id
    
    def get_dependency_type(self) -> DependencyType:
        return DependencyType.DATABASE
    
    async def check_dependency(self) -> DependencyHealth:
        """Check database connectivity and performance."""
        start_time = time.time()
        
        health = DependencyHealth(
            dependency_id=self.dependency_id,
            dependency_type=DependencyType.DATABASE,
            endpoint=self.connection_string
        )
        
        try:
            # Try to connect and execute a simple query
            conn = await asyncpg.connect(self.connection_string)
            
            # Test query
            await conn.execute("SELECT 1")
            
            # Measure response time
            health.response_time_ms = (time.time() - start_time) * 1000
            health.is_reachable = True
            health.status = HealthStatus.HEALTHY
            health.last_successful_check = datetime.now(timezone.utc)
            health.consecutive_failures = 0
            
            await conn.close()
            
        except Exception as e:
            health.response_time_ms = (time.time() - start_time) * 1000
            health.is_reachable = False
            health.status = HealthStatus.UNHEALTHY
            health.consecutive_failures += 1
            health.custom_data['error'] = str(e)
            
            self.logger.error(f"Database health check failed: {str(e)}")
        
        return health


class RedisHealthChecker(DependencyChecker):
    """Health checker for Redis cache."""
    
    def __init__(self, redis_url: str, dependency_id: str = "redis"):
        self.redis_url = redis_url
        self.dependency_id = dependency_id
        self.logger = get_logger(__name__)
    
    def get_dependency_id(self) -> str:
        return self.dependency_id
    
    def get_dependency_type(self) -> DependencyType:
        return DependencyType.CACHE
    
    async def check_dependency(self) -> DependencyHealth:
        """Check Redis connectivity and performance."""
        start_time = time.time()
        
        health = DependencyHealth(
            dependency_id=self.dependency_id,
            dependency_type=DependencyType.CACHE,
            endpoint=self.redis_url
        )
        
        try:
            # Connect to Redis
            redis_client = redis.from_url(self.redis_url)
            
            # Test ping
            redis_client.ping()
            
            # Test set/get
            test_key = f"health_check_{int(time.time())}"
            redis_client.set(test_key, "test_value", ex=60)
            result = redis_client.get(test_key)
            
            if result == b"test_value":
                health.response_time_ms = (time.time() - start_time) * 1000
                health.is_reachable = True
                health.status = HealthStatus.HEALTHY
                health.last_successful_check = datetime.now(timezone.utc)
                health.consecutive_failures = 0
                
                # Get Redis info
                info = redis_client.info()
                health.custom_data.update({
                    'used_memory': info.get('used_memory', 0),
                    'connected_clients': info.get('connected_clients', 0),
                    'keyspace_hits': info.get('keyspace_hits', 0),
                    'keyspace_misses': info.get('keyspace_misses', 0)
                })
            else:
                raise Exception("Redis set/get test failed")
                
            redis_client.close()
            
        except Exception as e:
            health.response_time_ms = (time.time() - start_time) * 1000
            health.is_reachable = False
            health.status = HealthStatus.UNHEALTHY
            health.consecutive_failures += 1
            health.custom_data['error'] = str(e)
            
            self.logger.error(f"Redis health check failed: {str(e)}")
        
        return health


class APIHealthChecker(DependencyChecker):
    """Health checker for external APIs."""
    
    def __init__(self, api_url: str, dependency_id: str, 
                 expected_status: int = 200, timeout: float = 10.0):
        self.api_url = api_url
        self.dependency_id = dependency_id
        self.expected_status = expected_status
        self.timeout = timeout
        self.logger = get_logger(__name__)
    
    def get_dependency_id(self) -> str:
        return self.dependency_id
    
    def get_dependency_type(self) -> DependencyType:
        return DependencyType.EXTERNAL_API
    
    async def check_dependency(self) -> DependencyHealth:
        """Check API availability and response time."""
        start_time = time.time()
        
        health = DependencyHealth(
            dependency_id=self.dependency_id,
            dependency_type=DependencyType.EXTERNAL_API,
            endpoint=self.api_url,
            timeout_seconds=self.timeout
        )
        
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                async with session.get(self.api_url) as response:
                    health.response_time_ms = (time.time() - start_time) * 1000
                    
                    if response.status == self.expected_status:
                        health.is_reachable = True
                        health.status = HealthStatus.HEALTHY
                        health.last_successful_check = datetime.now(timezone.utc)
                        health.consecutive_failures = 0
                    else:
                        health.is_reachable = False
                        health.status = HealthStatus.DEGRADED
                        health.custom_data['status_code'] = response.status
                        health.custom_data['expected_status'] = self.expected_status
                        
        except Exception as e:
            health.response_time_ms = (time.time() - start_time) * 1000
            health.is_reachable = False
            health.status = HealthStatus.UNHEALTHY
            health.consecutive_failures += 1
            health.custom_data['error'] = str(e)
            
            self.logger.error(f"API health check failed for {self.api_url}: {str(e)}")
        
        return health


class SystemResourceChecker:
    """Checker for system resource health."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
    
    def get_system_metrics(self) -> HealthMetrics:
        """Get current system resource metrics."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_mb = memory.used / (1024 * 1024)
            memory_percent = memory.percent
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            
            # Network stats
            network = psutil.net_io_counters()
            
            return HealthMetrics(
                cpu_usage_percent=cpu_percent,
                memory_usage_mb=memory_mb,
                memory_usage_percent=memory_percent,
                disk_usage_percent=disk_percent,
                custom_metrics={
                    'bytes_sent': network.bytes_sent,
                    'bytes_recv': network.bytes_recv,
                    'packets_sent': network.packets_sent,
                    'packets_recv': network.packets_recv,
                    'load_average': psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 0.0
                }
            )
            
        except Exception as e:
            self.logger.error(f"Failed to get system metrics: {str(e)}")
            return HealthMetrics()


class CircuitBreaker:
    """Circuit breaker for health checks and recovery."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.logger = get_logger(__name__)
    
    async def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        if self.state == "OPEN":
            if self._should_attempt_reset():
                self.state = "HALF_OPEN"
            else:
                raise HealthCheckError("Circuit breaker is OPEN")
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset."""
        if self.last_failure_time is None:
            return True
        
        return (time.time() - self.last_failure_time) >= self.recovery_timeout
    
    def _on_success(self):
        """Handle successful execution."""
        self.failure_count = 0
        if self.state == "HALF_OPEN":
            self.state = "CLOSED"
            self.logger.info("Circuit breaker reset to CLOSED")
    
    def _on_failure(self):
        """Handle failed execution."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            self.logger.warning(f"Circuit breaker opened after {self.failure_count} failures")


class HealthPredictor:
    """Predictive health analytics using historical data."""
    
    def __init__(self, history_size: int = 1000):
        self.history_size = history_size
        self.component_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=history_size))
        self.logger = get_logger(__name__)
    
    def add_health_data(self, component_id: str, result: HealthCheckResult):
        """Add health check result to history."""
        self.component_history[component_id].append({
            'timestamp': result.timestamp,
            'status': result.status,
            'health_score': result.health_score,
            'metrics': asdict(result.metrics)
        })
    
    def predict_health_issues(self, component_id: str) -> List[Dict[str, Any]]:
        """Predict potential health issues based on trends."""
        history = self.component_history.get(component_id, deque())
        if len(history) < 10:  # Need minimum data points
            return []
        
        predictions = []
        
        try:
            # Analyze health score trend
            recent_scores = [h['health_score'] for h in list(history)[-20:]]
            if len(recent_scores) >= 10:
                # Check for declining trend
                if self._is_declining_trend(recent_scores):
                    predictions.append({
                        'type': 'declining_health',
                        'severity': 'warning',
                        'message': f'Health score declining for {component_id}',
                        'confidence': 0.8
                    })
            
            # Analyze response time trend
            recent_response_times = [
                h['metrics'].get('response_time_ms', 0) 
                for h in list(history)[-20:]
            ]
            if len(recent_response_times) >= 10:
                avg_response_time = statistics.mean(recent_response_times)
                if avg_response_time > 1000:  # > 1 second
                    predictions.append({
                        'type': 'high_response_time',
                        'severity': 'warning',
                        'message': f'Average response time increasing for {component_id}',
                        'confidence': 0.7
                    })
            
            # Analyze error patterns
            recent_statuses = [h['status'] for h in list(history)[-10:]]
            unhealthy_count = sum(1 for status in recent_statuses if status in [HealthStatus.UNHEALTHY, HealthStatus.CRITICAL])
            if unhealthy_count >= 3:
                predictions.append({
                    'type': 'recurring_failures',
                    'severity': 'critical',
                    'message': f'Recurring failures detected for {component_id}',
                    'confidence': 0.9
                })
            
        except Exception as e:
            self.logger.error(f"Error predicting health issues for {component_id}: {str(e)}")
        
        return predictions
    
    def _is_declining_trend(self, values: List[float]) -> bool:
        """Check if values show a declining trend."""
        if len(values) < 5:
            return False
        
        # Simple linear trend analysis
        n = len(values)
        x = list(range(n))
        
        # Calculate correlation coefficient
        mean_x = statistics.mean(x)
        mean_y = statistics.mean(values)
        
        numerator = sum((x[i] - mean_x) * (values[i] - mean_y) for i in range(n))
        denominator_x = sum((x[i] - mean_x) ** 2 for i in range(n))
        denominator_y = sum((values[i] - mean_y) ** 2 for i in range(n))
        
        if denominator_x == 0 or denominator_y == 0:
            return False
        
        correlation = numerator / (denominator_x * denominator_y) ** 0.5
        
        # Negative correlation indicates declining trend
        return correlation < -0.5


class AutoRecoveryManager:
    """Manages automated recovery actions."""
    
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.recovery_strategies: Dict[str, List[Callable]] = defaultdict(list)
        self.recovery_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.logger = get_logger(__name__)
    
    def register_recovery_strategy(self, component_id: str, strategy: Callable):
        """Register a recovery strategy for a component."""
        self.recovery_strategies[component_id].append(strategy)
    
    async def attempt_recovery(self, component_id: str, issue: str) -> bool:
        """Attempt automated recovery for a component."""
        strategies = self.recovery_strategies.get(component_id, [])
        if not strategies:
            self.logger.info(f"No recovery strategies available for {component_id}")
            return False
        
        await self.event_bus.emit(AutoRecoveryStarted(
            component_id=component_id,
            issue=issue,
            strategies_available=len(strategies)
        ))
        
        recovery_success = False
        
        for strategy in strategies:
            try:
                self.logger.info(f"Attempting recovery for {component_id} using {strategy.__name__}")
                
                success = await strategy(component_id, issue)
                if success:
                    recovery_success = True
                    self.logger.info(f"Recovery successful for {component_id}")
                    break
                    
            except Exception as e:
                self.logger.error(f"Recovery strategy {strategy.__name__} failed for {component_id}: {str(e)}")
        
        # Record recovery attempt
        self.recovery_history[component_id].append({
            'timestamp': datetime.now(timezone.utc),
            'issue': issue,
            'success': recovery_success,
            'strategies_tried': len(strategies)
        })
        
        await self.event_bus.emit(AutoRecoveryCompleted(
            component_id=component_id,
            issue=issue,
            success=recovery_success
        ))
        
        return recovery_success


class HealthCheck:
    """
    Advanced Health Check System for the AI Assistant.
    
    This system provides comprehensive health monitoring including:
    - Component health tracking with custom checks
    - External dependency monitoring
    - System resource monitoring
    - Automated recovery mechanisms
    - Predictive health analytics
    - Circuit breaker integration
    - Real-time health reporting
    - Performance threshold monitoring
    - Health history and trending
    """
    
    def __init__(self, container: Container):
        """
        Initialize the health check system.
        
        Args:
            container: Dependency injection container
        """
        self.container = container
        self.logger = get_logger(__name__)
        
        # Core services
        self.config = container.get(ConfigLoader)
        self.event_bus = container.get(EventBus)
        self.error_handler = container.get(ErrorHandler)
        
        # Observability
        try:
            self.metrics = container.get(MetricsCollector)
            self.tracer = container.get(TraceManager)
        except Exception:
            self.metrics = None
            self.tracer = None
        
        # Component health tracking
        self.component_checkers: Dict[str, HealthChecker] = {}
        self.component_callbacks: Dict[str, Callable] = {}
        self.component_health: Dict[str, HealthCheckResult] = {}
        self.component_thresholds: Dict[str, List[HealthThreshold]] = {}
        
        # Dependency health tracking
        self.dependency_checkers: Dict[str, DependencyChecker] = {}
        self.dependency_health: Dict[str, DependencyHealth] = {}
        
        # System health
        self.system_checker = SystemResourceChecker()
        self.system_health = SystemHealth(overall_status=HealthStatus.UNKNOWN)
        
        # Advanced features
        self.health_predictor = HealthPredictor()
        self.auto_recovery = AutoRecoveryManager(self.event_bus)
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        
        # Configuration
        self.global_check_interval = self.config.get("health_check.global_interval", 30.0)
        self.enable_auto_recovery = self.config.get("health_check.enable_auto_recovery", True)
        self.enable_predictions = self.config.get("health_check.enable_predictions", True)
        self.max_concurrent_checks = self.config.get("health_check.max_concurrent_checks", 10)
        
        # State management
        self.is_running = False
        self.check_tasks: Dict[str, asyncio.Task] = {}
        self.check_semaphore = asyncio.Semaphore(self.max_concurrent_checks)
        
        # Performance tracking
        self.check_performance: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Setup monitoring
        self._setup_monitoring()
        self._setup_default_dependencies()
        
        self.logger.info("HealthCheck system initialized")

    def _setup_monitoring(self) -> None:
        """Setup monitoring and metrics."""
        if self.metrics:
            # Register health check metrics
            self.metrics.register_counter("health_checks_total")
            self.metrics.register_counter("health_checks_failed")
            self.metrics.register_histogram("health_check_duration_seconds")
            self.metrics.register_gauge("component_health_score")
            self.metrics.register_gauge("system_health_score")
            self.metrics.register_counter("auto_recovery_attempts")
            self.metrics.register_counter("circuit_breaker_trips")

    def _setup_default_dependencies(self) -> None:
        """Setup default dependency checkers."""
        try:
            # Database health checker
            db_url = self.config.get("database.url")
            if db_url:
                self.register_dependency_checker(DatabaseHealthChecker(db_url))
            
            # Redis health checker
            redis_url = self.config.get("cache.redis_url")
            if redis_url:
                self.register_dependency_checker(RedisHealthChecker(redis_url))
            
            # External API health checkers
            api_configs = self.config.get("health_check.external_apis", {})
            for api_name, api_config in api_configs.items():
                checker = APIHealthChecker(
                    api_url=api_config.get("url"),
                    dependency_id=api_name,
                    expected_status=api_config.get("expected_status", 200),
                    timeout=api_config.get("timeout", 10.0)
                )
                self.register_dependency_checker(checker)
                
        except Exception as e:
            self.logger.warning(f"Failed to setup default dependency checkers: {str(e)}")

    @handle_exceptions
    def register_component(self, component_id: str, health_callback: Callable[[], Dict[str, Any]],
                          frequency: CheckFrequency = CheckFrequency.NORMAL,
                          thresholds: Optional[List[HealthThreshold]] = None) -> None:
        """
        Register a component for health monitoring.
        
        Args:
            component_id: Unique component identifier
            health_callback: Async function that returns health status
            frequency: Check frequency
            thresholds: Health thresholds for the component
        """
        self.component_callbacks[component_id] = health_callback
        
        if thresholds:
            self.component_thresholds[component_id] = thresholds
        
        # Create circuit breaker for component
        self.circuit_breakers[component_id] = CircuitBreaker()
        
        # Start monitoring task if system is running
        if self.is_running:
            self._start_component_monitoring(component_id, frequency)
        
        self.logger.info(f"Registered component for health monitoring: {component_id}")

    def register_component_checker(self, checker: HealthChecker) -> None:
        """
        Register a component health checker.
        
        Args:
            checker: Health checker instance
        """
        component_id = checker.get_component_id()
        self.component_checkers[component_id] = checker
        
        # Register thresholds
        thresholds = checker.get_thresholds()
        if thresholds:
            self.component_thresholds[component_id] = thresholds
        
        # Register auto-recovery if supported
        if checker.supports_auto_recovery():
            self.auto_recovery.register_recovery_strategy(
                component_id, 
                checker.auto_recover
            )
        
        # Create circuit breaker
        self.circuit_breakers[component_id] = CircuitBreaker()
        
        # Start monitoring if system is running
        if self.is_running:
            frequency = checker.get_check_frequency()
            self._start_component_monitoring(component_id, frequency)
        
        self.logger.info(f"Registered component checker: {component_id}")

    def register_dependency_checker(self, checker: DependencyChecker) -> None:
        """
        Register a dependency health checker.
        
        Args:
            checker: Dependency checker instance
        """
        dependency_id = checker.get_dependency_id()
        self.dependency_checkers[dependency_id] = checker
        
        # Create circuit breaker
        self.circuit_breakers[dependency_id] = CircuitBreaker()
        
        self.logger.info(f"Registered dependency checker: {dependency_id}")

    async def start(self) -> None:
        """Start the health check system."""
        if self.is_running:
            self.logger.warning("Health check system is already running")
            return
        
        self.is_running = True
        
        try:
            # Start component monitoring tasks
            for component_id in self.component_callbacks.keys():
                self._start_component_monitoring(component_id, CheckFrequency.NORMAL)
            
            for component_id, checker in self.component_checkers.items():
                frequency = checker.get_check_frequency()
                self._start_component_monitoring(component_id, frequency)
            
            # Start dependency monitoring
            asyncio.create_task(self._dependency_monitoring_loop())
            
            # Start system monitoring
            asyncio.create_task(self._system_monitoring_loop())
            
            # Start health aggregation
            asyncio.create_task(self._health_aggregation_loop())
            
            # Start predictive analytics
            if self.enable_predictions:
                asyncio.create_task(self._predictive_analytics_loop())
            
            await self.event_bus.emit(HealthCheckStarted())
            
            self.logger.info("Health check system started")
            
        except Exception as e:
            self.is_running = False
            self.logger.error(f"Failed to start health check system: {str(e)}")
            raise HealthCheckError(f"Failed to start health check system: {str(e)}")

    def _start_component_monitoring(self, component_id: str, frequency: CheckFrequency) -> None:
        """Start monitoring task for a component."""
        if component_id in self.check_tasks:
            self.check_tasks[component_id].cancel()
        
        self.check_tasks[component_id] = asyncio.create_task(
            self._component_monitoring_loop(component_id, frequency.value)
        )

    async def _component_monitoring_loop(self, component_id: str, interval: float) -> None:
        """Monitoring loop for a specific component."""
        while self.is_running:
            try:
                await self._check_component_health(component_id)
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in component monitoring loop for {component_id}: {str(e)}")
                await asyncio.sleep(interval)

    async def _check_component_health(self, component_id: str) -> None:
        """Check health for a specific component."""
        async with self.check_semaphore:
            start_time = time.time()
            
            try:
                # Get circuit breaker
                circuit_breaker = self.circuit_breakers.get(component_id)
                
                # Perform health check through circuit breaker
                if circuit_breaker:
                    result = await circuit_breaker.call(self._perform_component_check, component_id)
                else:
                    result = await self._perform_component_check(component_id)
                
                # Store result
                self.component_health[component_id] = result
                
                # Add to predictive analytics
                if self.enable_predictions:
                    self.health_predictor.add_health_data(component_id, result)
                
                # Check thresholds
                await self._check_component_thresholds(component_id, result)
                
                # Update metrics
                if self.metrics:
                    self.metrics.increment("health_checks_total", tags={'component': component_id})
                    self.metrics.record("health_check_duration_seconds", result.check_duration_ms / 1000)
                    self.metrics.set("component_health_score", result.health_score, tags={'component': component_id})
                
                # Track performance
                self.check_performance[component_id].append(result.check_duration_ms)
                
            except Exception as e:
                # Create error result
                error_result = HealthCheckResult(
                    component_id=component_id,
                    status=HealthStatus.CRITICAL,
                    error_message=str(e),
                    error_type=type(e).__name__,
                    error_stack=traceback.format_exc(),
                    check_duration_ms=(time.time() - start_time) * 1000,
                    health_score=0.0
                )
                
                self.component_health[component_id] = error_result
                
                if self.metrics:
                    self.metrics.increment("health_checks_failed", tags={'component': component_id})
                
                # Attempt auto-recovery
                if self.enable_auto_recovery:
                    await self.auto_recovery.attempt_recovery(component_id, str(e))
                
                self.logger.error(f"Health check failed for component {component_id}: {str(e)}")

    async def _perform_component_check(self, component_id: str) -> HealthCheckResult:
        """Perform the actual component health check."""
        start_time = time.time()
        
        # Use registered checker if available
        if component_id in self.component_checkers:
            checker = self.component_checkers[component_id]
            return await checker.check_health()
        
        # Use callback if available
        elif component_id in self.component_callbacks:
            callback = self.component_callbacks[component_id]
            
            try:
                if asyncio.iscoroutinefunction(callback):
                    health_data = await callback()
                else:
                    health_data = callback()
                
                # Convert health data to HealthCheckResult
                status = HealthStatus(health_data.get('status', 'unknown'))
                
                result = HealthCheckResult(
                    component_id=component_id,
                    status=status,
                    check_duration_ms=(time.time() - start_time) * 1000
                )
                
                # Extract metrics if available
                if 'metrics' in health_data:
                    metrics_data = health_data['metrics']
                    result.metrics = HealthMetrics(**metrics_data)
                
                # Extract details
                if 'details' in health_data:
                    result.details = health_data['details']
                
                # Calculate health score
                result.health_score = self._calculate_health_score(status, result.metrics)
                
                return result
                
            except Exception as e:
                return HealthCheckResult(
                    component_id=component_id,
                    status=HealthStatus.CRITICAL,
                    error_message=str(e),
                    error_type=type(e).__name__,
                    check_duration_ms=(time.time() - start_time) * 1000,
                    health_score=0.0
                )
        
        else:
            raise HealthCheckError(f"No health checker or callback registered for {component_id}")

    def _calculate_health_score(self, status: HealthStatus, metrics: HealthMetrics) -> float:
        """Calculate health score based on status and metrics."""
        # Base score from status
        status_scores = {
            HealthStatus.HEALTHY: 1.0,
            HealthStatus.DEGRADED: 0.7,
            HealthStatus.UNHEALTHY: 0.3,
            HealthStatus.CRITICAL: 0.1,
            HealthStatus.DOWN: 0.0,
            HealthStatus.UNKNOWN: 0.5
        }
        
        base_score = status_scores.get(status, 0.5)
        
        # Adjust based on metrics
        score_adjustments = 0.0
        adjustment_count = 0
        
        # Response time adjustment
        if metrics.response_time_ms > 0:
            if metrics.response_time_ms < 100:
                score_adjustments += 0.1
            elif metrics.response_time_ms > 5000:
                score_adjustments -= 0.2
            adjustment_count += 1
        
        # CPU usage adjustment
        if metrics.cpu_usage_percent > 0:
            if metrics.cpu_usage_percent > 90:
                score_adjustments -= 0.2
            elif metrics.cpu_usage_percent < 50:
                score_adjustments += 0.05
            adjustment_count += 1
        
        # Memory usage adjustment
        if metrics.memory_usage_percent > 0:
            if metrics.memory_usage_percent > 85:
                score_adjustments -= 0.15
            elif metrics.memory_usage_percent < 60:
                score_adjustments += 0.05
            adjustment_count += 1
        
        # Error rate adjustment
        if metrics.error_rate > 0:
            score_adjustments -= min(metrics.error_rate * 0.5, 0.3)
            adjustment_count += 1
        
        # Apply adjustments
        if adjustment_count > 0:
            avg_adjustment = score_adjustments / adjustment_count
            final_score = base_score + avg_adjustment
        else:
            final_score = base_score
        
        return max(0.0, min(1.0, final_score))

    async def _check_component_thresholds(self, component_id: str, result: HealthCheckResult) -> None:
        """Check if component violates any thresholds."""
        thresholds = self.component_thresholds.get(component_id, [])
        
        for threshold in thresholds:
            metric_value = getattr(result.metrics, threshold.metric_name, None)
            if metric_value is None:
                metric_value = result.metrics.custom_metrics.get(threshold.metric_name)
            
            if metric_value is not None:
                violated = self._check_threshold_violation(metric_value, threshold)
                
                if violated:
                    await self.event_bus.emit(HealthThresholdExceeded(
                        component_id=component_id,
                        metric_name=threshold.metric_name,
                        threshold_value=threshold.critical_threshold,
                        actual_value=metric_value,
                        severity="critical" if metric_value > threshold.critical_threshold else "warning"
                    ))
                    
                    # Trigger auto-recovery if enabled
                    if self.enable_auto_recovery:
                        issue = f"Threshold violation: {threshold.metric_name} = {metric_value}"
                        await self.auto_recovery.attempt_recovery(component_id, issue)

    def _check_threshold_violation(self, value: float, threshold: HealthThreshold) -> bool:
        """Check if a value violates a threshold."""
        if threshold.comparison == "greater_than":
            return value > threshold.critical_threshold
        elif threshold.comparison == "less_than":
            return value < threshold.critical_threshold
        elif threshold.comparison == "equals":
            return abs(value - threshold.critical_threshold) < 0.001
        
        return False

    async def _dependency_monitoring_loop(self) -> None:
        """Monitor all dependencies."""
        while self.is_running:
            try:
                # Check all dependencies
                for dependency_id, checker in self.dependency_checkers.items():
                    try:
                        circuit_breaker = self.circuit_breakers.get(dependency_id)
                        
                        if circuit_breaker:
                            health = await circuit_breaker.call(checker.check_dependency)
                        else:
                            health = await checker.check_dependency()
                        
                        self.dependency_health[dependency_id] = health
                        
                        # Emit health change event if status changed
                        previous_health = self.dependency_health.get(dependency_id)
                        if previous_health and previous_health.status != health.status:
                            await self.event_bus.emit(DependencyHealthChanged(
                                dependency_id=dependency_id,
                                old_status=previous_health.status.value,
                                new_status=health.status.value,
                                dependency_type=health.dependency_type.value
                            ))
                        
                    except Exception as e:
                        self.logger.error(f"Dependency check failed for {dependency_id}: {str(e)}")
                
                await asyncio.sleep(60)  # Check dependencies every minute
                
            except Exception as e:
                self.logger.error(f"Error in dependency monitoring loop: {str(e)}")
                await asyncio.sleep(60)

    async def _system_monitoring_loop(self) -> None:
        """Monitor overall system health."""
        while self.is_running:
            try:
                # Get system metrics
                system_metrics = self.system_checker.get_system_metrics()
                
                # Update system health
                self.system_health.system_metrics = system_metrics
                self.system_health.timestamp = datetime.now(timezone.utc)
                
                await asyncio.sleep(30)  # Check system every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Error in system monitoring loop: {str(e)}")
                await asyncio.sleep(30)

    async def _health_aggregation_loop(self) -> None:
        """Aggregate component and dependency health into overall system health."""
        while self.is_running:
            try:
                # Count component health statuses
                healthy_components = 0
                degraded_components = 0
                unhealthy_components = 0
                
                for result in self.component_health.values():
                    if result.status == HealthStatus.HEALTHY:
                        healthy_components += 1
                    elif result.status in [HealthStatus.DEGRADED, HealthStatus.UNKNOWN]:
                        degraded_components += 1
                    else:
                        unhealthy_components += 1
                
                total_components = len(self.component_health)
                
                # Count dependency health statuses
                healthy_dependencies = 0
                unhealthy_dependencies = 0
                
                for health in self.dependency_health.values():
                    if health.status == HealthStatus.HEALTHY:
                        healthy_dependencies += 1
                    else:
                        unhealthy_dependencies += 1
                
                total_dependencies = len(self.dependency_health)
                
                # Calculate overall status
                overall_status = self._calculate_overall_status(
                    healthy_components, degraded_components, unhealthy_components,
                    healthy_dependencies, unhealthy_dependencies
                )
                
                # Calculate overall health score
                if total_components > 0:
                    component_scores = [result.health_score for result in self.component_health.values()]
                    avg_component_score = statistics.mean(component_scores)
                else:
                    avg_component_score = 1.0
                
                # Dependency health contribution
                if total_dependencies > 0:
                    dependency_score = healthy_dependencies / total_dependencies
                else:
                    dependency_score = 1.0
                
                # Weight components more heavily than dependencies
                overall_health_score = (avg_component_score * 0.7) + (dependency_score * 0.3)
                
                # Update system health
                previous_status = self.system_health.overall_status
                
                self.system_health.overall_status = overall_status
                self.system_health.healthy_components = healthy_components
                self.system_health.degraded_components = degraded_components
                self.system_health.unhealthy_components = unhealthy_components
                self.system_health.total_components = total_components
                self.system_health.healthy_dependencies = healthy_dependencies
                self.system_health.unhealthy_dependencies = unhealthy_dependencies
                self.system_health.total_dependencies = total_dependencies
                self.system_health.overall_health_score = overall_health_score
                self.system_health.timestamp = datetime.now(timezone.utc)
                
                # Emit system health change event
                if previous_status != overall_status:
                    await self.event_bus.emit(SystemHealthChanged(
                        old_status=previous_status.value,
                        new_status=overall_status.value,
                        health_score=overall_health_score
                    ))
                
                # Update metrics
                if self.metrics:
                    self.metrics.set("system_health_score", overall_health_score)
                
                await asyncio.sleep(15)  # Aggregate every 15 seconds
                
            except Exception as e:
                self.logger.error(f"Error in health aggregation loop: {str(e)}")
                await asyncio.sleep(15)

    def _calculate_overall_status(self, healthy_components: int, degraded_components: int,
                                unhealthy_components: int, healthy_dependencies: int,
                                unhealthy_dependencies: int) -> HealthStatus:
        """Calculate overall system health status."""
        total_components = healthy_components + degraded_components + unhealthy_components
        total_dependencies = healthy_dependencies + unhealthy_dependencies
        
        if total_components == 0:
            return HealthStatus.UNKNOWN
        
        # If any critical components are down
        if unhealthy_components > 0:
            return HealthStatus.CRITICAL
        
        # If dependencies are failing
        if total_dependencies > 0 and unhealthy_dependencies / total_dependencies > 0.5:
            return HealthStatus.UNHEALTHY
        
        # If significant degradation
        if degraded_components / total_components > 0.3:
            return HealthStatus.DEGRADED
        
        # If some degradation or dependency issues
        if degraded_components > 0 or unhealthy_dependencies > 0:
            return HealthStatus.DEGRADED
        
        return HealthStatus.HEALTHY

    async def _predictive_analytics_loop(self) -> None:
        """Run predictive health analytics."""
        while self.is_running:
            try:
                # Run predictions for all components
                for component_id in self.component_health.keys():
                    predictions = self.health_predictor.predict_health_issues(component_id)
                    
                    for prediction in predictions:
                        await self.event_bus.emit(HealthPredictionAlert(
                            component_id=component_id,
                            prediction_type=prediction['type'],
                            severity=prediction['severity'],
                            message=prediction['message'],
                            confidence=prediction['confidence']
                        ))
                
                await asyncio.sleep(300)  # Run predictions every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Error in predictive analytics loop: {str(e)}")
                await asyncio.sleep(300)

    @handle_exceptions
    async def get_system_health(self) -> SystemHealth:
        """
        Get current system health status.
        
        Returns:
            Current system health information
        """
        return self.system_health

    @handle_exceptions
    async def get_component_health(self, component_id: str) -> Optional[HealthCheckResult]:
        """
        Get health status for a specific component.
        
        Args:
            component_id: Component identifier
            
        Returns:
            Component health result or None if not found
        """
        return self.component_health.get(component_id)

    @handle_exceptions
    async def get_dependency_health(self, dependency_id: str) -> Optional[DependencyHealth]:
        """
        Get health status for a specific dependency.
        
        Args:
            dependency_id: Dependency identifier
            
        Returns:
            Dependency health information or None if not found
        """
        return self.dependency_health.get(dependency_id)

    @handle_exceptions
    async def force_health_check(self, component_id: str) -> HealthCheckResult:
        """
        Force an immediate health check for a component.
        
        Args:
            component_id: Component identifier
            
        Returns:
            Health check result
        """
        await self._check_component_health(component_id)
        result = self.component_health.get(component_id)
        
        if result is None:
            raise HealthCheckError(f"Component {component_id} not found or check failed")
        
        return result

    def get_health_summary(self) -> Dict[str, Any]:
        """Get a summary of overall health status."""
        return {
            'system_health': {
                'status': self.system_health.overall_status.value,
                'score': self.system_health.overall_health_score,
                'timestamp': self.system_health.timestamp.isoformat()
            },
            'components': {
                'total': self.system_health.total_components,
                'healthy': self.system_health.healthy_components,
                'degraded': self.system_health.degraded_components,
                'unhealthy': self.system_health.unhealthy_components
            },
            'dependencies': {
                'total': self.system_health.total_dependencies,
                'healthy': self.system_health.healthy_dependencies,
                'unhealthy': self.system_health.unhealthy_dependencies
            },
            'system_metrics': asdict(self.system_health.system_metrics),
            'active_alerts': self.system_health.active_alerts,
            'recommendations': self.system_health.recommendations
        }

    def get_component_list(self) -> List[Dict[str, Any]]:
        """Get list of all monitored components."""
        components = []
        
        for component_id, result in self.component_health.items():
            components.append({
                'component_id': component_id,
                'status': result.status.value,
                'health_score': result.health_score,
                'last_check': result.timestamp.isoformat(),
                'response_time_ms': result.metrics.response_time_ms,
                'error_count': len([e for e in [result.error_message] if e])
            })
        
        return components

    def get_dependency_list(self) -> List[Dict[str, Any]]:
        """Get list of all monitored dependencies."""
        dependencies = []
        
        for dependency_id, health in self.dependency_health.items():
            dependencies.append({
                'dependency_id': dependency_id,
                'type': health.dependency_type.value,
                'status': health.status.value,
                'endpoint': health.endpoint,
                'is_reachable': health.is_reachable,
                'response_time_ms': health.response_time_ms,
                'consecutive_failures': health.consecutive_failures
            })
        
        return dependencies

    async def stop(self) -> None:
        """Stop the health check system."""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Cancel all monitoring tasks
        for task in self.check_tasks.values():
            task.cancel()
        
        # Wait for tasks to complete
        if self.check_tasks:
            await asyncio.gather(*self.check_tasks.values(), return_exceptions=True)
        
        self.check_tasks.clear()
        
        await self.event_bus.emit(HealthCheckCompleted())
        
        self.logger.info("Health check system stopped")

    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            if hasattr(self, 'is_running') and self.is_running:
                self.logger.warning("HealthCheck destroyed while still running")
        except Exception:
            pass  # Ignore cleanup errors in destructor
