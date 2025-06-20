"""
Comprehensive Event and Request Handler System
Author: Drmusab
Last Modified: 2025-01-20 04:07:33 UTC

This module provides a centralized handler system for the AI assistant,
managing all types of events, requests, and system interactions with
proper error handling, monitoring, and integration with core components.
"""

from pathlib import Path
from typing import Optional, Dict, Any, List, Set, Callable, Type, Union, AsyncGenerator, TypeVar, Protocol, runtime_checkable
import asyncio
import threading
import time
import inspect
import traceback
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
from concurrent.futures import ThreadPoolExecutor
import functools
import signal

# Core imports
from src.core.config.loader import ConfigLoader
from src.core.events.event_bus import EventBus
from src.core.events.event_types import (
    BaseEvent, ErrorOccurred, SystemStateChanged, ComponentHealthChanged,
    RequestReceived, RequestProcessed, RequestFailed, ResponseGenerated,
    UserInteractionStarted, UserInteractionCompleted, UserInteractionFailed,
    SessionStarted, SessionEnded, WorkflowStarted, WorkflowCompleted,
    PluginLoaded, PluginEnabled, ComponentRegistered, ComponentInitialized,
    MemoryOperationStarted, MemoryOperationCompleted, LearningEventOccurred,
    SecurityViolation, PerformanceThresholdExceeded, HealthCheckCompleted,
    ResourceUtilizationChanged, ConfigurationChanged, BackupCompleted
)
from src.core.error_handling import ErrorHandler, handle_exceptions
from src.core.dependency_injection import Container
from src.core.health_check import HealthCheck
from src.core.security.authentication import AuthenticationManager
from src.core.security.authorization import AuthorizationManager
from src.core.security.sanitization import SecuritySanitizer

# Assistant components
from src.assistant.core_engine import EnhancedCoreEngine, MultimodalInput, ProcessingResult
from src.assistant.component_manager import EnhancedComponentManager
from src.assistant.workflow_orchestrator import WorkflowOrchestrator
from src.assistant.interaction_handler import InteractionHandler
from src.assistant.session_manager import EnhancedSessionManager
from src.assistant.plugin_manager import EnhancedPluginManager

# Processing components
from src.processing.natural_language.intent_manager import IntentManager
from src.processing.natural_language.language_chain import LanguageChain
from src.processing.speech.audio_pipeline import EnhancedAudioPipeline
from src.processing.vision.vision_processor import VisionProcessor
from src.processing.multimodal.fusion_strategies import MultimodalFusionStrategy

# Memory and learning
from src.memory.core_memory.memory_manager import MemoryManager
from src.memory.operations.context_manager import ContextManager
from src.learning.continual_learning import ContinualLearner
from src.learning.preference_learning import PreferenceLearner
from src.learning.feedback_processor import FeedbackProcessor

# Observability
from src.observability.monitoring.metrics import MetricsCollector
from src.observability.monitoring.tracing import TraceManager
from src.observability.logging.config import get_logger

# Skills management
from src.skills.skill_factory import SkillFactory
from src.skills.skill_registry import SkillRegistry

# Type definitions
T = TypeVar('T')
HandlerResult = TypeVar('HandlerResult')


class HandlerType(Enum):
    """Types of handlers in the system."""
    EVENT = "event"
    REQUEST = "request"
    RESPONSE = "response"
    ERROR = "error"
    LIFECYCLE = "lifecycle"
    SECURITY = "security"
    RESOURCE = "resource"
    MONITORING = "monitoring"
    MIDDLEWARE = "middleware"
    PLUGIN = "plugin"
    WORKFLOW = "workflow"
    MEMORY = "memory"
    LEARNING = "learning"


class HandlerPriority(Enum):
    """Handler execution priorities."""
    CRITICAL = 0    # System-critical handlers
    HIGH = 1        # High-priority handlers
    NORMAL = 2      # Normal priority handlers
    LOW = 3         # Low priority handlers
    BACKGROUND = 4  # Background processing


class HandlerState(Enum):
    """Handler execution states."""
    IDLE = "idle"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class ExecutionMode(Enum):
    """Handler execution modes."""
    SYNCHRONOUS = "synchronous"
    ASYNCHRONOUS = "asynchronous"
    PARALLEL = "parallel"
    SEQUENTIAL = "sequential"
    STREAMING = "streaming"
    BATCH = "batch"


@runtime_checkable
class HandlerInterface(Protocol):
    """Base protocol that all handlers should implement."""
    
    async def handle(self, event: Any, context: Dict[str, Any]) -> Any:
        """Handle the event/request."""
        ...
    
    def can_handle(self, event: Any) -> bool:
        """Check if this handler can process the event."""
        ...
    
    def get_priority(self) -> HandlerPriority:
        """Get handler priority."""
        ...


@dataclass
class HandlerMetadata:
    """Metadata for handler registration."""
    handler_id: str
    handler_type: HandlerType
    priority: HandlerPriority = HandlerPriority.NORMAL
    execution_mode: ExecutionMode = ExecutionMode.ASYNCHRONOUS
    
    # Event matching
    event_types: Set[Type] = field(default_factory=set)
    event_patterns: List[str] = field(default_factory=list)
    
    # Execution constraints
    timeout_seconds: float = 30.0
    max_retries: int = 3
    retry_delay: float = 1.0
    
    # Dependencies
    required_components: Set[str] = field(default_factory=set)
    optional_components: Set[str] = field(default_factory=set)
    
    # Security
    requires_authentication: bool = False
    required_permissions: Set[str] = field(default_factory=set)
    
    # Performance
    enable_metrics: bool = True
    enable_tracing: bool = True
    
    # Metadata
    description: Optional[str] = None
    version: str = "1.0.0"
    author: Optional[str] = None
    tags: Set[str] = field(default_factory=set)


@dataclass
class HandlerContext:
    """Context information for handler execution."""
    handler_id: str
    execution_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Request information
    request_id: Optional[str] = None
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    
    # Timing
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    timeout_at: Optional[datetime] = None
    
    # State
    state: HandlerState = HandlerState.IDLE
    retry_count: int = 0
    
    # Context data
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Performance
    metrics: Dict[str, float] = field(default_factory=dict)
    trace_id: Optional[str] = None


@dataclass
class HandlerResult:
    """Result of handler execution."""
    success: bool
    handler_id: str
    execution_id: str
    
    # Result data
    result: Any = None
    error: Optional[Exception] = None
    
    # Performance metrics
    execution_time: float = 0.0
    memory_usage: Optional[float] = None
    
    # Metadata
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metrics: Dict[str, Any] = field(default_factory=dict)


class HandlerError(Exception):
    """Custom exception for handler operations."""
    
    def __init__(self, message: str, handler_id: Optional[str] = None, 
                 execution_id: Optional[str] = None, error_code: Optional[str] = None):
        super().__init__(message)
        self.handler_id = handler_id
        self.execution_id = execution_id
        self.error_code = error_code
        self.timestamp = datetime.now(timezone.utc)


class BaseHandler(ABC):
    """Abstract base class for all handlers."""
    
    def __init__(self, metadata: HandlerMetadata):
        self.metadata = metadata
        self.logger = get_logger(f"handler_{metadata.handler_id}")
        self._execution_count = 0
        self._error_count = 0
        self._last_execution: Optional[datetime] = None
        self._performance_metrics: Dict[str, List[float]] = defaultdict(list)
    
    @abstractmethod
    async def handle(self, event: Any, context: HandlerContext) -> HandlerResult:
        """Handle the event/request."""
        pass
    
    def can_handle(self, event: Any) -> bool:
        """Check if this handler can process the event."""
        if self.metadata.event_types:
            return type(event) in self.metadata.event_types
        return True
    
    def get_priority(self) -> HandlerPriority:
        """Get handler priority."""
        return self.metadata.priority
    
    async def initialize(self, container: Container) -> None:
        """Initialize handler with dependencies."""
        pass
    
    async def cleanup(self) -> None:
        """Cleanup handler resources."""
        pass
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get handler performance metrics."""
        return {
            "execution_count": self._execution_count,
            "error_count": self._error_count,
            "last_execution": self._last_execution.isoformat() if self._last_execution else None,
            "avg_execution_time": self._get_avg_metric("execution_time"),
            "avg_memory_usage": self._get_avg_metric("memory_usage")
        }
    
    def _get_avg_metric(self, metric_name: str) -> float:
        """Get average value for a metric."""
        values = self._performance_metrics.get(metric_name, [])
        return sum(values) / len(values) if values else 0.0
    
    def _record_execution(self, execution_time: float, memory_usage: Optional[float] = None):
        """Record execution metrics."""
        self._execution_count += 1
        self._last_execution = datetime.now(timezone.utc)
        self._performance_metrics["execution_time"].append(execution_time)
        
        if memory_usage is not None:
            self._performance_metrics["memory_usage"].append(memory_usage)


class EventHandler(BaseHandler):
    """Handler for system events."""
    
    def __init__(self, handler_id: str, event_types: Set[Type], 
                 handler_func: Callable, priority: HandlerPriority = HandlerPriority.NORMAL):
        metadata = HandlerMetadata(
            handler_id=handler_id,
            handler_type=HandlerType.EVENT,
            priority=priority,
            event_types=event_types
        )
        super().__init__(metadata)
        self.handler_func = handler_func
    
    async def handle(self, event: Any, context: HandlerContext) -> HandlerResult:
        """Handle system events."""
        start_time = time.time()
        
        try:
            # Call the handler function
            if asyncio.iscoroutinefunction(self.handler_func):
                result = await self.handler_func(event, context)
            else:
                result = self.handler_func(event, context)
            
            execution_time = time.time() - start_time
            self._record_execution(execution_time)
            
            return HandlerResult(
                success=True,
                handler_id=self.metadata.handler_id,
                execution_id=context.execution_id,
                result=result,
                execution_time=execution_time
            )
            
        except Exception as e:
            self._error_count += 1
            execution_time = time.time() - start_time
            
            return HandlerResult(
                success=False,
                handler_id=self.metadata.handler_id,
                execution_id=context.execution_id,
                error=e,
                execution_time=execution_time
            )


class RequestHandler(BaseHandler):
    """Handler for processing requests."""
    
    def __init__(self, handler_id: str, request_processor: Callable,
                 priority: HandlerPriority = HandlerPriority.NORMAL):
        metadata = HandlerMetadata(
            handler_id=handler_id,
            handler_type=HandlerType.REQUEST,
            priority=priority
        )
        super().__init__(metadata)
        self.request_processor = request_processor
    
    async def handle(self, request: Any, context: HandlerContext) -> HandlerResult:
        """Handle requests."""
        start_time = time.time()
        
        try:
            # Process the request
            if asyncio.iscoroutinefunction(self.request_processor):
                result = await self.request_processor(request, context)
            else:
                result = self.request_processor(request, context)
            
            execution_time = time.time() - start_time
            self._record_execution(execution_time)
            
            return HandlerResult(
                success=True,
                handler_id=self.metadata.handler_id,
                execution_id=context.execution_id,
                result=result,
                execution_time=execution_time
            )
            
        except Exception as e:
            self._error_count += 1
            execution_time = time.time() - start_time
            
            return HandlerResult(
                success=False,
                handler_id=self.metadata.handler_id,
                execution_id=context.execution_id,
                error=e,
                execution_time=execution_time
            )


class MiddlewareHandler(BaseHandler):
    """Handler for middleware processing."""
    
    def __init__(self, handler_id: str, middleware_func: Callable,
                 priority: HandlerPriority = HandlerPriority.NORMAL):
        metadata = HandlerMetadata(
            handler_id=handler_id,
            handler_type=HandlerType.MIDDLEWARE,
            priority=priority
        )
        super().__init__(metadata)
        self.middleware_func = middleware_func
    
    async def handle(self, data: Any, context: HandlerContext) -> HandlerResult:
        """Handle middleware processing."""
        start_time = time.time()
        
        try:
            # Process through middleware
            if asyncio.iscoroutinefunction(self.middleware_func):
                result = await self.middleware_func(data, context)
            else:
                result = self.middleware_func(data, context)
            
            execution_time = time.time() - start_time
            self._record_execution(execution_time)
            
            return HandlerResult(
                success=True,
                handler_id=self.metadata.handler_id,
                execution_id=context.execution_id,
                result=result,
                execution_time=execution_time
            )
            
        except Exception as e:
            self._error_count += 1
            execution_time = time.time() - start_time
            
            return HandlerResult(
                success=False,
                handler_id=self.metadata.handler_id,
                execution_id=context.execution_id,
                error=e,
                execution_time=execution_time
            )


class SecurityHandler(BaseHandler):
    """Handler for security operations."""
    
    def __init__(self, container: Container):
        metadata = HandlerMetadata(
            handler_id="security_handler",
            handler_type=HandlerType.SECURITY,
            priority=HandlerPriority.CRITICAL,
            requires_authentication=False  # Security handler handles auth itself
        )
        super().__init__(metadata)
        
        # Security components
        try:
            self.auth_manager = container.get(AuthenticationManager)
            self.authz_manager = container.get(AuthorizationManager)
            self.security_sanitizer = container.get(SecuritySanitizer)
        except Exception:
            self.logger.warning("Some security components not available")
            self.auth_manager = None
            self.authz_manager = None
            self.security_sanitizer = None
    
    async def handle(self, security_event: Any, context: HandlerContext) -> HandlerResult:
        """Handle security events and validations."""
        start_time = time.time()
        
        try:
            result = {}
            
            # Handle authentication events
            if hasattr(security_event, 'user_id') and self.auth_manager:
                is_authenticated = await self.auth_manager.is_authenticated(security_event.user_id)
                result['authenticated'] = is_authenticated
                
                if not is_authenticated:
                    raise HandlerError("Authentication failed", self.metadata.handler_id)
            
            # Handle authorization checks
            if (hasattr(security_event, 'user_id') and 
                hasattr(security_event, 'resource') and 
                hasattr(security_event, 'action') and 
                self.authz_manager):
                
                is_authorized = await self.authz_manager.check_permission(
                    security_event.user_id,
                    security_event.resource,
                    security_event.action
                )
                result['authorized'] = is_authorized
                
                if not is_authorized:
                    raise HandlerError("Authorization failed", self.metadata.handler_id)
            
            # Handle input sanitization
            if hasattr(security_event, 'data') and self.security_sanitizer:
                sanitized_data = await self.security_sanitizer.sanitize(security_event.data)
                result['sanitized_data'] = sanitized_data
            
            execution_time = time.time() - start_time
            self._record_execution(execution_time)
            
            return HandlerResult(
                success=True,
                handler_id=self.metadata.handler_id,
                execution_id=context.execution_id,
                result=result,
                execution_time=execution_time
            )
            
        except Exception as e:
            self._error_count += 1
            execution_time = time.time() - start_time
            
            return HandlerResult(
                success=False,
                handler_id=self.metadata.handler_id,
                execution_id=context.execution_id,
                error=e,
                execution_time=execution_time
            )


class LifecycleHandler(BaseHandler):
    """Handler for component lifecycle events."""
    
    def __init__(self, container: Container):
        metadata = HandlerMetadata(
            handler_id="lifecycle_handler",
            handler_type=HandlerType.LIFECYCLE,
            priority=HandlerPriority.HIGH,
            event_types={ComponentRegistered, ComponentInitialized, 
                        PluginLoaded, PluginEnabled, SessionStarted, SessionEnded}
        )
        super().__init__(metadata)
        self.container = container
        self.component_states: Dict[str, str] = {}
        self.lifecycle_hooks: Dict[str, List[Callable]] = defaultdict(list)
    
    async def handle(self, lifecycle_event: Any, context: HandlerContext) -> HandlerResult:
        """Handle component lifecycle events."""
        start_time = time.time()
        
        try:
            result = {}
            event_type = type(lifecycle_event).__name__
            
            # Component registration
            if isinstance(lifecycle_event, ComponentRegistered):
                self.component_states[lifecycle_event.component_id] = "registered"
                await self._execute_hooks("component_registered", lifecycle_event)
                result['action'] = 'component_registered'
            
            # Component initialization
            elif isinstance(lifecycle_event, ComponentInitialized):
                self.component_states[lifecycle_event.component_id] = "initialized"
                await self._execute_hooks("component_initialized", lifecycle_event)
                result['action'] = 'component_initialized'
            
            # Plugin events
            elif isinstance(lifecycle_event, PluginLoaded):
                await self._execute_hooks("plugin_loaded", lifecycle_event)
                result['action'] = 'plugin_loaded'
            
            elif isinstance(lifecycle_event, PluginEnabled):
                await self._execute_hooks("plugin_enabled", lifecycle_event)
                result['action'] = 'plugin_enabled'
            
            # Session events
            elif isinstance(lifecycle_event, SessionStarted):
                await self._execute_hooks("session_started", lifecycle_event)
                result['action'] = 'session_started'
            
            elif isinstance(lifecycle_event, SessionEnded):
                await self._execute_hooks("session_ended", lifecycle_event)
                result['action'] = 'session_ended'
            
            execution_time = time.time() - start_time
            self._record_execution(execution_time)
            
            return HandlerResult(
                success=True,
                handler_id=self.metadata.handler_id,
                execution_id=context.execution_id,
                result=result,
                execution_time=execution_time
            )
            
        except Exception as e:
            self._error_count += 1
            execution_time = time.time() - start_time
            
            return HandlerResult(
                success=False,
                handler_id=self.metadata.handler_id,
                execution_id=context.execution_id,
                error=e,
                execution_time=execution_time
            )
    
    async def _execute_hooks(self, hook_type: str, event: Any) -> None:
        """Execute registered lifecycle hooks."""
        for hook in self.lifecycle_hooks[hook_type]:
            try:
                if asyncio.iscoroutinefunction(hook):
                    await hook(event)
                else:
                    hook(event)
            except Exception as e:
                self.logger.error(f"Lifecycle hook failed for {hook_type}: {str(e)}")
    
    def register_hook(self, hook_type: str, hook_func: Callable) -> None:
        """Register a lifecycle hook."""
        self.lifecycle_hooks[hook_type].append(hook_func)


class PerformanceHandler(BaseHandler):
    """Handler for performance monitoring and optimization."""
    
    def __init__(self, container: Container):
        metadata = HandlerMetadata(
            handler_id="performance_handler",
            handler_type=HandlerType.MONITORING,
            priority=HandlerPriority.LOW,
            event_types={PerformanceThresholdExceeded, ResourceUtilizationChanged}
        )
        super().__init__(metadata)
        
        try:
            self.metrics = container.get(MetricsCollector)
            self.tracer = container.get(TraceManager)
        except Exception:
            self.logger.warning("Monitoring components not available")
            self.metrics = None
            self.tracer = None
        
        self.performance_data: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.thresholds: Dict[str, float] = {
            "response_time": 5.0,
            "memory_usage": 80.0,
            "cpu_usage": 85.0,
            "error_rate": 5.0
        }
    
    async def handle(self, performance_event: Any, context: HandlerContext) -> HandlerResult:
        """Handle performance monitoring events."""
        start_time = time.time()
        
        try:
            result = {}
            
            if isinstance(performance_event, PerformanceThresholdExceeded):
                await self._handle_threshold_exceeded(performance_event)
                result['action'] = 'threshold_exceeded_handled'
            
            elif isinstance(performance_event, ResourceUtilizationChanged):
                await self._handle_resource_utilization(performance_event)
                result['action'] = 'resource_utilization_tracked'
            
            # Record performance data
            if hasattr(performance_event, 'metric_name') and hasattr(performance_event, 'value'):
                self.performance_data[performance_event.metric_name].append({
                    'value': performance_event.value,
                    'timestamp': datetime.now(timezone.utc)
                })
            
            execution_time = time.time() - start_time
            self._record_execution(execution_time)
            
            return HandlerResult(
                success=True,
                handler_id=self.metadata.handler_id,
                execution_id=context.execution_id,
                result=result,
                execution_time=execution_time
            )
            
        except Exception as e:
            self._error_count += 1
            execution_time = time.time() - start_time
            
            return HandlerResult(
                success=False,
                handler_id=self.metadata.handler_id,
                execution_id=context.execution_id,
                error=e,
                execution_time=execution_time
            )
    
    async def _handle_threshold_exceeded(self, event: Any) -> None:
        """Handle performance threshold exceeded events."""
        metric_name = getattr(event, 'metric_name', 'unknown')
        value = getattr(event, 'value', 0)
        threshold = getattr(event, 'threshold', 0)
        
        self.logger.warning(
            f"Performance threshold exceeded for {metric_name}: "
            f"{value} > {threshold}"
        )
        
        # Implement performance optimization strategies
        if metric_name == "memory_usage" and value > 90:
            await self._trigger_memory_cleanup()
        elif metric_name == "response_time" and value > 10:
            await self._optimize_response_time()
    
    async def _handle_resource_utilization(self, event: Any) -> None:
        """Handle resource utilization changes."""
        if self.metrics:
            # Update metrics
            self.metrics.set("system_resource_utilization", getattr(event, 'value', 0),
                           tags={'resource': getattr(event, 'resource_type', 'unknown')})
    
    async def _trigger_memory_cleanup(self) -> None:
        """Trigger memory cleanup operations."""
        self.logger.info("Triggering memory cleanup due to high memory usage")
        # Implementation would trigger garbage collection, cache cleanup, etc.
    
    async def _optimize_response_time(self) -> None:
        """Optimize response time."""
        self.logger.info("Optimizing response time due to slow responses")
        # Implementation would adjust quality settings, enable caching, etc.


class MultimodalHandler(BaseHandler):
    """Handler for multimodal processing requests."""
    
    def __init__(self, container: Container):
        metadata = HandlerMetadata(
            handler_id="multimodal_handler",
            handler_type=HandlerType.REQUEST,
            priority=HandlerPriority.HIGH,
            execution_mode=ExecutionMode.ASYNCHRONOUS,
            timeout_seconds=60.0
        )
        super().__init__(metadata)
        
        # Core processing components
        self.core_engine = container.get(EnhancedCoreEngine)
        self.intent_manager = container.get(IntentManager)
        self.audio_pipeline = container.get(EnhancedAudioPipeline)
        self.vision_processor = container.get(VisionProcessor)
        self.fusion_strategy = container.get(MultimodalFusionStrategy)
        self.memory_manager = container.get(MemoryManager)
    
    async def handle(self, multimodal_request: MultimodalInput, 
                    context: HandlerContext) -> HandlerResult:
        """Handle multimodal processing requests."""
        start_time = time.time()
        
        try:
            # Process through core engine
            processing_result = await self.core_engine.process_multimodal_input(
                multimodal_request,
                multimodal_request.context
            )
            
            # Extract key metrics
            execution_time = time.time() - start_time
            self._record_execution(execution_time)
            
            return HandlerResult(
                success=processing_result.success,
                handler_id=self.metadata.handler_id,
                execution_id=context.execution_id,
                result=processing_result,
                execution_time=execution_time,
                metrics={
                    'confidence': processing_result.overall_confidence,
                    'component_timings': processing_result.component_timings,
                    'memory_usage': processing_result.memory_usage
                }
            )
            
        except Exception as e:
            self._error_count += 1
            execution_time = time.time() - start_time
            
            return HandlerResult(
                success=False,
                handler_id=self.metadata.handler_id,
                execution_id=context.execution_id,
                error=e,
                execution_time=execution_time
            )


class WorkflowHandler(BaseHandler):
    """Handler for workflow execution requests."""
    
    def __init__(self, container: Container):
        metadata = HandlerMetadata(
            handler_id="workflow_handler",
            handler_type=HandlerType.WORKFLOW,
            priority=HandlerPriority.NORMAL,
            event_types={WorkflowStarted, WorkflowCompleted}
        )
        super().__init__(metadata)
        
        self.workflow_orchestrator = container.get(WorkflowOrchestrator)
        self.active_workflows: Dict[str, Dict[str, Any]] = {}
    
    async def handle(self, workflow_request: Any, context: HandlerContext) -> HandlerResult:
        """Handle workflow execution requests."""
        start_time = time.time()
        
        try:
            result = {}
            
            if isinstance(workflow_request, WorkflowStarted):
                # Track workflow execution
                self.active_workflows[workflow_request.execution_id] = {
                    'workflow_id': workflow_request.workflow_id,
                    'session_id': workflow_request.session_id,
                    'start_time': datetime.now(timezone.utc)
                }
                result['action'] = 'workflow_tracking_started'
            
            elif isinstance(workflow_request, WorkflowCompleted):
                # Update workflow completion
                if workflow_request.execution_id in self.active_workflows:
                    workflow_info = self.active_workflows.pop(workflow_request.execution_id)
                    workflow_info['completion_time'] = datetime.now(timezone.utc)
                    workflow_info['execution_time'] = workflow_request.execution_time
                    result['workflow_info'] = workflow_info
                result['action'] = 'workflow_completed'
            
            else:
                # Handle direct workflow execution requests
                if hasattr(workflow_request, 'workflow_id'):
                    execution_id = await self.workflow_orchestrator.execute_workflow(
                        workflow_request.workflow_id,
                        getattr(workflow_request, 'session_id', context.session_id or 'default'),
                        getattr(workflow_request, 'input_data', {}),
                        getattr(workflow_request, 'user_id', context.user_id)
                    )
                    result['execution_id'] = execution_id
                    result['action'] = 'workflow_execution_started'
            
            execution_time = time.time() - start_time
            self._record_execution(execution_time)
            
            return HandlerResult(
                success=True,
                handler_id=self.metadata.handler_id,
                execution_id=context.execution_id,
                result=result,
                execution_time=execution_time
            )
            
        except Exception as e:
            self._error_count += 1
            execution_time = time.time() - start_time
            
            return HandlerResult(
                success=False,
                handler_id=self.metadata.handler_id,
                execution_id=context.execution_id,
                error=e,
                execution_time=execution_time
            )


class MemoryHandler(BaseHandler):
    """Handler for memory operations."""
    
    def __init__(self, container: Container):
        metadata = HandlerMetadata(
            handler_id="memory_handler",
            handler_type=HandlerType.MEMORY,
            priority=HandlerPriority.NORMAL,
            event_types={MemoryOperationStarted, MemoryOperationCompleted}
        )
        super().__init__(metadata)
        
        self.memory_manager = container.get(MemoryManager)
        self.context_manager = container.get(ContextManager)
        self.memory_operations: Dict[str, Dict[str, Any]] = {}
    
    async def handle(self, memory_request: Any, context: HandlerContext) -> HandlerResult:
        """Handle memory operations."""
        start_time = time.time()
        
        try:
            result = {}
            
            if isinstance(memory_request, MemoryOperationStarted):
                # Track memory operation
                self.memory_operations[memory_request.operation_id] = {
                    'operation_type': memory_request.operation_type,
                    'session_id': memory_request.session_id,
                    'start_time': datetime.now(timezone.utc)
                }
                result['action'] = 'memory_operation_tracked'
            
            elif isinstance(memory_request, MemoryOperationCompleted):
                # Update operation completion
                if memory_request.operation_id in self.memory_operations:
                    op_info = self.memory_operations.pop(memory_request.operation_id)
                    op_info['completion_time'] = datetime.now(timezone.utc)
                    result['operation_info'] = op_info
                result['action'] = 'memory_operation_completed'
            
            else:
                # Handle direct memory requests
                if hasattr(memory_request, 'operation_type'):
                    operation_type = memory_request.operation_type
                    
                    if operation_type == 'store':
                        await self.memory_manager.store_memory(
                            memory_request.data,
                            getattr(memory_request, 'memory_type', 'episodic'),
                            getattr(memory_request, 'session_id', context.session_id)
                        )
                        result['action'] = 'memory_stored'
                    
                    elif operation_type == 'retrieve':
                        memories = await self.memory_manager.retrieve_memories(
                            getattr(memory_request, 'query', ''),
                            getattr(memory_request, 'memory_type', 'episodic'),
                            getattr(memory_request, 'limit', 10)
                        )
                        result['memories'] = memories
                        result['action'] = 'memory_retrieved'
                    
                    elif operation_type == 'consolidate':
                        await self.memory_manager.consolidate_memories()
                        result['action'] = 'memory_consolidated'
            
            execution_time = time.time() - start_time
            self._record_execution(execution_time)
            
            return HandlerResult(
                success=True,
                handler_id=self.metadata.handler_id,
                execution_id=context.execution_id,
                result=result,
                execution_time=execution_time
            )
            
        except Exception as e:
            self._error_count += 1
            execution_time = time.time() - start_time
            
            return HandlerResult(
                success=False,
                handler_id=self.metadata.handler_id,
                execution_id=context.execution_id,
                error=e,
                execution_time=execution_time
            )


class LearningHandler(BaseHandler):
    """Handler for learning and adaptation events."""
    
    def __init__(self, container: Container):
        metadata = HandlerMetadata(
            handler_id="learning_handler",
            handler_type=HandlerType.LEARNING,
            priority=HandlerPriority.LOW,
            event_types={LearningEventOccurred}
        )
        super().__init__(metadata)
        
        try:
            self.continual_learner = container.get(ContinualLearner)
            self.preference_learner = container.get(PreferenceLearner)
            self.feedback_processor = container.get(FeedbackProcessor)
        except Exception:
            self.logger.warning("Learning components not available")
            self.continual_learner = None
            self.preference_learner = None
            self.feedback_processor = None
        
        self.learning_events: deque = deque(maxlen=1000)
    
    async def handle(self, learning_event: Any, context: HandlerContext) -> HandlerResult:
        """Handle learning events."""
        start_time = time.time()
        
        try:
            result = {}
            
            if isinstance(learning_event, LearningEventOccurred):
                # Process learning event
                self.learning_events.append({
                    'event_type': learning_event.event_type,
                    'data': learning_event.data,
                    'timestamp': datetime.now(timezone.utc)
                })
                
                # Update learning systems
                if self.continual_learner:
                    await self.continual_learner.process_learning_event(learning_event)
                
                if (self.preference_learner and 
                    hasattr(learning_event, 'user_id') and 
                    learning_event.user_id):
                    await self.preference_learner.update_from_event(
                        learning_event.user_id, learning_event
                    )
                
                result['action'] = 'learning_event_processed'
            
            else:
                # Handle direct learning requests
                if hasattr(learning_event, 'learning_type'):
                    learning_type = learning_event.learning_type
                    
                    if learning_type == 'feedback' and self.feedback_processor:
                        await self.feedback_processor.process_feedback(
                            getattr(learning_event, 'feedback_data', {})
                        )
                        result['action'] = 'feedback_processed'
                    
                    elif learning_type == 'preference' and self.preference_learner:
                        await self.preference_learner.update_preferences(
                            getattr(learning_event, 'user_id', ''),
                            getattr(learning_event, 'preferences', {})
                        )
                        result['action'] = 'preferences_updated'
            
            execution_time = time.time() - start_time
            self._record_execution(execution_time)
            
            return HandlerResult(
                success=True,
                handler_id=self.metadata.handler_id,
                execution_id=context.execution_id,
                result=result,
                execution_time=execution_time
            )
            
        except Exception as e:
            self._error_count += 1
            execution_time = time.time() - start_time
            
            return HandlerResult(
                success=False,
                handler_id=self.metadata.handler_id,
                execution_id=context.execution_id,
                error=e,
                execution_time=execution_time
            )


class HandlerRegistry:
    """Registry for managing all handlers in the system."""
    
    def __init__(self, logger):
        self.logger = logger
        self._handlers: Dict[str, BaseHandler] = {}
        self._handlers_by_type: Dict[HandlerType, List[BaseHandler]] = defaultdict(list)
        self._handlers_by_event: Dict[Type, List[BaseHandler]] = defaultdict(list)
        self._middleware_handlers: List[BaseHandler] = []
        self._registry_lock = asyncio.Lock()
    
    async def register_handler(self, handler: BaseHandler) -> None:
        """Register a handler."""
        async with self._registry_lock:
            handler_id = handler.metadata.handler_id
            
            if handler_id in self._handlers:
                raise HandlerError(f"Handler {handler_id} is already registered")
            
            self._handlers[handler_id] = handler
            self._handlers_by_type[handler.metadata.handler_type].append(handler)
            
            # Index by event types
            for event_type in handler.metadata.event_types:
                self._handlers_by_event[event_type].append(handler)
            
            # Track middleware handlers separately
            if handler.metadata.handler_type == HandlerType.MIDDLEWARE:
                self._middleware_handlers.append(handler)
                # Sort by priority
                self._middleware_handlers.sort(key=lambda h: h.metadata.priority.value)
            
            self.logger.info(f"Registered handler: {handler_id}")
    
    async def unregister_handler(self, handler_id: str) -> None:
        """Unregister a handler."""
        async with self._registry_lock:
            if handler_id not in self._handlers:
                return
            
            handler = self._handlers[handler_id]
            
            # Remove from all indexes
            self._handlers_by_type[handler.metadata.handler_type] = [
                h for h in self._handlers_by_type[handler.metadata.handler_type]
                if h.metadata.handler_id != handler_id
            ]
            
            for event_type in handler.metadata.event_types:
                self._handlers_by_event[event_type] = [
                    h for h in self._handlers_by_event[event_type]
                    if h.metadata.handler_id != handler_id
                ]
            
            if handler in self._middleware_handlers:
                self._middleware_handlers.remove(handler)
            
            del self._handlers[handler_id]
            self.logger.info(f"Unregistered handler: {handler_id}")
    
    def get_handler(self, handler_id: str) -> Optional[BaseHandler]:
        """Get a handler by ID."""
        return self._handlers.get(handler_id)
    
    def get_handlers_for_event(self, event: Any) -> List[BaseHandler]:
        """Get handlers that can process an event."""
        event_type = type(event)
        handlers = self._handlers_by_event.get(event_type, [])
        
        # Add handlers that can handle this event type
        additional_handlers = [
            h for h in self._handlers.values()
            if h not in handlers and h.can_handle(event)
        ]
        
        handlers.extend(additional_handlers)
        
        # Sort by priority
        handlers.sort(key=lambda h: h.metadata.priority.value)
        
        return handlers
    
    def get_handlers_by_type(self, handler_type: HandlerType) -> List[BaseHandler]:
        """Get handlers by type."""
        return list(self._handlers_by_type[handler_type])
    
    def get_middleware_handlers(self) -> List[BaseHandler]:
        """Get middleware handlers in priority order."""
        return list(self._middleware_handlers)
    
    def list_handlers(self) -> List[Dict[str, Any]]:
        """List all registered handlers."""
        return [
            {
                'handler_id': h.metadata.handler_id,
                'handler_type': h.metadata.handler_type.value,
                'priority': h.metadata.priority.value,
                'execution_mode': h.metadata.execution_mode.value,
                'metrics': h.get_metrics()
            }
            for h in self._handlers.values()
        ]


class EnhancedHandlerManager:
    """
    Comprehensive Handler Management System for the AI Assistant.
    
    This manager coordinates all types of handlers in the system including:
    - Event handlers for system-wide events
    - Request handlers for processing different types of requests
    - Middleware handlers for cross-cutting concerns
    - Security handlers for authentication and authorization
    - Performance handlers for monitoring and optimization
    - Lifecycle handlers for component management
    - Specialized handlers for multimodal, workflow, memory, and learning operations
    
    Features:
    - Unified handler interface and registration
    - Intelligent handler selection and routing
    - Parallel and sequential execution modes
    - Comprehensive error handling and recovery
    - Performance monitoring and optimization
    - Security integration and validation
    - Event-driven architecture support
    - Plugin and component integration
    - Real-time metrics and health monitoring
    """
    
    def __init__(self, container: Container):
        """
        Initialize the enhanced handler manager.
        
        Args:
            container: Dependency injection container
        """
        self.container = container
        self.logger = get_logger(__name__)
        self.config = container.get(ConfigLoader)
        self.event_bus = container.get(EventBus)
        self.error_handler = container.get(ErrorHandler)
        self.health_check = container.get(HealthCheck)
        
        # Handler management
        self.handler_registry = HandlerRegistry(self.logger)
        
        # Execution infrastructure
        self.execution_semaphore = asyncio.Semaphore(100)  # Max concurrent handlers
        self.thread_pool = ThreadPoolExecutor(
            max_workers=10,
            thread_name_prefix="handler_manager"
        )
        
        # State management
        self._initialization_lock = asyncio.Lock()
        self._shutdown_event = asyncio.Event()
        self._active_executions: Dict[str, HandlerContext] = {}
        
        # Performance tracking
        self._execution_stats: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self._performance_metrics: deque = deque(maxlen=10000)
        
        # Configuration
        self._enable_middleware = self.config.get("handlers.enable_middleware", True)
        self._enable_parallel_execution = self.config.get("handlers.enable_parallel", True)
        self._default_timeout = self.config.get("handlers.default_timeout", 30.0)
        self._max_retries = self.config.get("handlers.max_retries", 3)
        
        # Setup monitoring
        self._setup_monitoring()
        
        # Register health check
        self.health_check.register_component("handler_manager", self._health_check_callback)
        
        self.logger.info("EnhancedHandlerManager initialized")

    def _setup_monitoring(self) -> None:
        """Setup monitoring and metrics."""
        try:
            self.metrics = self.container.get(MetricsCollector)
            self.tracer = self.container.get(TraceManager)
            
            # Register handler metrics
            self.metrics.register_counter("handler_executions_total")
            self.metrics.register_counter("handler_errors_total")
            self.metrics.register_histogram("handler_execution_duration_seconds")
            self.metrics.register_gauge("active_handler_executions")
            self.metrics.register_counter("middleware_executions_total")
            
        except Exception as e:
            self.logger.warning(f"Failed to setup monitoring: {str(e)}")
            self.metrics = None
            self.tracer = None

    async def initialize(self) -> None:
        """Initialize the handler manager and register built-in handlers."""
        async with self._initialization_lock:
            try:
                self.logger.info("Initializing handler manager...")
                
                # Register built-in handlers
                await self._register_builtin_handlers()
                
                # Subscribe to system events
                await self._subscribe_to_events()
                
                # Start background tasks
                asyncio.create_task(self._handler_cleanup_loop())
                asyncio.create_task(self._performance_monitoring_loop())
                
                self.logger.info("Handler manager initialized successfully")
                
            except Exception as e:
                self.logger.error(f"Failed to initialize handler manager: {str(e)}")
                raise HandlerError(f"Handler manager initialization failed: {str(e)}")

    async def _register_builtin_handlers(self) -> None:
        """Register built-in system handlers."""
        try:
            # Security handler
            security_handler = SecurityHandler(self.container)
            await self.handler_registry.register_handler(security_handler)
            
            # Lifecycle handler
            lifecycle_handler = LifecycleHandler(self.container)
            await self.handler_registry.register_handler(lifecycle_handler)
            
            # Performance handler
            performance_handler = PerformanceHandler(self.container)
            await self.handler_registry.register_handler(performance_handler)
            
            # Multimodal handler
            multimodal_handler = MultimodalHandler(self.container)
            await self.handler_registry.register_handler(multimodal_handler)
            
            # Workflow handler
            workflow_handler = WorkflowHandler(self.container)
            await self.handler_registry.register_handler(workflow_handler)
            
            # Memory handler
            memory_handler = MemoryHandler(self.container)
            await self.handler_registry.register_handler(memory_handler)
            
            # Learning handler
            learning_handler = LearningHandler(self.container)
            await self.handler_registry.register_handler(learning_handler)
            
            # Initialize all handlers
            for handler in self.handler_registry._handlers.values():
                if hasattr(handler, 'initialize'):
                    await handler.initialize(self.container)
            
            self.logger.info("Built-in handlers registered successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to register built-in handlers: {str(e)}")
            raise

    async def _subscribe_to_events(self) -> None:
        """Subscribe to system events for automatic handling."""
        # Subscribe to all events for automatic handler routing
        self.event_bus.subscribe("*", self.handle_event)

    @handle_exceptions
    async def register_handler(self, handler: BaseHandler) -> None:
        """
        Register a custom handler.
        
        Args:
            handler: Handler instance to register
        """
        await self.handler_registry.register_handler(handler)
        
        # Initialize if not already done
        if hasattr(handler, 'initialize'):
            await handler.initialize(self.container)

    @handle_exceptions
    async def register_event_handler(
        self,
        handler_id: str,
        event_types: Set[Type],
        handler_func: Callable,
        priority: HandlerPriority = HandlerPriority.NORMAL
    ) -> None:
        """
        Register an event handler function.
        
        Args:
            handler_id: Unique handler identifier
            event_types: Set of event types this handler processes
            handler_func: Handler function
            priority: Handler priority
        """
        handler = EventHandler(handler_id, event_types, handler_func, priority)
        await self.register_handler(handler)

    @handle_exceptions
    async def register_request_handler(
        self,
        handler_id: str,
        request_processor: Callable,
        priority: HandlerPriority = HandlerPriority.NORMAL
    ) -> None:
        """
        Register a request handler function.
        
        Args:
            handler_id: Unique handler identifier
            request_processor: Request processing function
            priority: Handler priority
        """
        handler = RequestHandler(handler_id, request_processor, priority)
        await self.register_handler(handler)

    @handle_exceptions
    async def register_middleware(
        self,
        handler_id: str,
        middleware_func: Callable,
        priority: HandlerPriority = HandlerPriority.NORMAL
    ) -> None:
        """
        Register a middleware handler.
        
        Args:
            handler_id: Unique handler identifier
            middleware_func: Middleware function
            priority: Handler priority
        """
        handler = MiddlewareHandler(handler_id, middleware_func, priority)
        await self.register_handler(handler)

    @handle_exceptions
    async def handle_event(self, event: Any) -> List[HandlerResult]:
        """
        Handle a system event through appropriate handlers.
        
        Args:
            event: Event to process
            
        Returns:
            List of handler results
        """
        # Get handlers for this event
        handlers = self.handler_registry.get_handlers_for_event(event)
        
        if not handlers:
            return []
        
        # Create execution context
        context = HandlerContext(
            handler_id="event_dispatcher",
            request_id=getattr(event, 'request_id', None),
            session_id=getattr(event, 'session_id', None),
            user_id=getattr(event, 'user_id', None)
        )
        
        # Execute handlers
        return await self._execute_handlers(handlers, event, context)

    @handle_exceptions
    async def handle_request(
        self,
        request: Any,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        timeout: Optional[float] = None
    ) -> HandlerResult:
        """
        Handle a request through appropriate handlers.
        
        Args:
            request: Request to process
            session_id: Optional session identifier
            user_id: Optional user identifier
            timeout: Optional timeout override
            
        Returns:
            Handler result
        """
        # Get request handlers
        handlers = self.handler_registry.get_handlers_by_type(HandlerType.REQUEST)
        
        # Filter handlers that can handle this request
        compatible_handlers = [h for h in handlers if h.can_handle(request)]
        
        if not compatible_handlers:
            raise HandlerError(f"No handler found for request type: {type(request).__name__}")
        
        # Use the highest priority handler
        handler = compatible_handlers[0]
        
        # Create execution context
        context = HandlerContext(
            handler_id=handler.metadata.handler_id,
            session_id=session_id,
            user_id=user_id,
            timeout_at=datetime.now(timezone.utc) + timedelta(
                seconds=timeout or self._default_timeout
            )
        )
        
        # Execute handler
        results = await self._execute_handlers([handler], request, context)
        return results[0] if results else HandlerResult(
            success=False,
            handler_id=handler.metadata.handler_id,
            execution_id=context.execution_id,
            error=HandlerError("No result returned from handler")
        )

    async def _execute_handlers(
        self,
        handlers: List[BaseHandler],
        data: Any,
        context: HandlerContext
    ) -> List[HandlerResult]:
        """Execute a list of handlers."""
        if not handlers:
            return []
        
        async with self.execution_semaphore:
            # Apply middleware if enabled
            if self._enable_middleware:
                data = await self._apply_middleware(data, context)
            
            # Track active execution
            self._active_executions[context.execution_id] = context
            
            try:
                results = []
                
                if self._enable_parallel_execution and len(handlers) > 1:
                    # Execute handlers in parallel
                    tasks = [
                        self._execute_single_handler(handler, data, context)
                        for handler in handlers
                    ]
                    
                    handler_results = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    for handler, result in zip(handlers, handler_results):
                        if isinstance(result, Exception):
                            result = HandlerResult(
                                success=False,
                                handler_id=handler.metadata.handler_id,
                                execution_id=context.execution_id,
                                error=result
                            )
                        results.append(result)
                else:
                    # Execute handlers sequentially
                    for handler in handlers:
                        result = await self._execute_single_handler(handler, data, context)
                        results.append(result)
                        
                        # Stop on first critical failure if configured
                        if (not result.success and 
                            handler.metadata.priority == HandlerPriority.CRITICAL):
                            break
                
                return results
                
            finally:
                # Remove from active executions
                self._active_executions.pop(context.execution_id, None)

    async def _execute_single_handler(
        self,
        handler: BaseHandler,
        data: Any,
        context: HandlerContext
    ) -> HandlerResult:
        """Execute a single handler with error handling and retries."""
        start_time = time.time()
        retry_count = 0
        last_error = None
        
        while retry_count <= handler.metadata.max_retries:
            try:
                # Check timeout
                if (context.timeout_at and 
                    datetime.now(timezone.utc) > context.timeout_at):
                    raise HandlerError("Handler execution timed out", handler.metadata.handler_id)
                
                # Update context
                context.handler_id = handler.metadata.handler_id
                context.state = HandlerState.EXECUTING
                context.retry_count = retry_count
                
                # Execute handler
                with self.tracer.trace(f"handler_{handler.metadata.handler_id}") if self.tracer else None:
                    result = await handler.handle(data, context)
                
                # Update metrics
                execution_time = time.time() - start_time
                
                if self.metrics:
                    self.metrics.increment("handler_executions_total",
                                         tags={'handler_id': handler.metadata.handler_id})
                    self.metrics.record("handler_execution_duration_seconds", execution_time,
                                      tags={'handler_id': handler.metadata.handler_id})
                
                # Track performance
                self._performance_metrics.append({
                    'handler_id': handler.metadata.handler_id,
                    'execution_time': execution_time,
                    'success': result.success,
                    'timestamp': datetime.now(timezone.utc)
                })
                
                context.state = HandlerState.COMPLETED if result.success else HandlerState.FAILED
                return result
                
            except Exception as e:
                last_error = e
                retry_count += 1
                
                if self.metrics:
                    self.metrics.increment("handler_errors_total",
                                         tags={'handler_id': handler.metadata.handler_id})
                
                if retry_count <= handler.metadata.max_retries:
                    self.logger.warning(
                        f"Handler {handler.metadata.handler_id} failed, "
                        f"retrying ({retry_count}/{handler.metadata.max_retries}): {str(e)}"
                    )
                    await asyncio.sleep(handler.metadata.retry_delay * retry_count)
                else:
                    self.logger.error(
                        f"Handler {handler.metadata.handler_id} failed after "
                        f"{handler.metadata.max_retries} retries: {str(e)}"
                    )
        
        # All retries exhausted
        execution_time = time.time() - start_time
        context.state = HandlerState.FAILED
        
        return HandlerResult(
            success=False,
            handler_id=handler.metadata.handler_id,
            execution_id=context.execution_id,
            error=last_error,
            execution_time=execution_time
        )

    async def _apply_middleware(self, data: Any, context: HandlerContext) -> Any:
        """Apply middleware handlers to process data."""
        middleware_handlers = self.handler_registry.get_middleware_handlers()
        
        current_data = data
        
        for middleware in middleware_handlers:
            try:
                middleware_context = HandlerContext(
                    handler_id=middleware.metadata.handler_id,
                    execution_id=context.execution_id,
                    session_id=context.session_id,
                    user_id=context.user_id
                )
                
                result = await middleware.handle(current_data, middleware_context)
                
                if result.success and result.result is not None:
                    current_data = result.result
                
                if self.metrics:
                    self.metrics.increment("middleware_executions_total",
                                         tags={'middleware_id': middleware.metadata.handler_id})
                
            except Exception as e:
                self.logger.error(f"Middleware {middleware.metadata.handler_id} failed: {str(e)}")
                # Continue with original data if middleware fails
        
        return current_data

    def get_handler_status(self, handler_id: Optional[str] = None) -> Dict[str, Any]:
        """Get status information for handlers."""
        if handler_id:
            handler = self.handler_registry.get_handler(handler_id)
            if not handler:
                raise HandlerError(f"Handler {handler_id} not found")
            
            return {
                'handler_id': handler_id,
                'handler_type': handler.metadata.handler_type.value,
                'priority': handler.metadata.priority.value,
                'execution_mode': handler.metadata.execution_mode.value,
                'metrics': handler.get_metrics(),
                'metadata': asdict(handler.metadata)
            }
        else:
            # Return status for all handlers
            handlers = self.handler_registry.list_handlers()
            
            return {
                'total_handlers': len(handlers),
                'active_executions': len(self._active_executions),
                'handlers_by_type': {
                    handler_type.value: len(self.handler_registry.get_handlers_by_type(handler_type))
                    for handler_type in HandlerType
                },
                'handlers': handlers
            }

    def get_active_executions(self) -> List[Dict[str, Any]]:
        """Get information about active handler executions."""
        return [
            {
                'execution_id': context.execution_i
