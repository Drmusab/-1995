"""
Advanced Event Handlers System for AI Assistant
Author: Drmusab
Last Modified: 2025-05-26 16:45:23 UTC

This module provides comprehensive event handling for the AI assistant system,
managing event processing, routing, correlation, and automated responses across
all core components with high performance and reliability.
"""

from pathlib import Path
from typing import Optional, Dict, Any, List, Set, Callable, Type, Union, AsyncGenerator, TypeVar, Protocol
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
from concurrent.futures import ThreadPoolExecutor
import traceback

# Core imports
from src.core.config.loader import ConfigLoader
from src.core.events.event_bus import EventBus
from src.core.events.event_types import (
    # System events
    EngineStarted, EngineShutdown, SystemStateChanged, SystemShutdownStarted,
    SystemShutdownCompleted, SystemHealthCheck, SystemPerformanceAlert,
    
    # Component events
    ComponentRegistered, ComponentInitialized, ComponentStarted, ComponentStopped,
    ComponentFailed, ComponentHealthChanged, ComponentRestarted, ComponentUpgraded,
    
    # Workflow events
    WorkflowStarted, WorkflowCompleted, WorkflowFailed, WorkflowPaused, WorkflowResumed,
    WorkflowStepStarted, WorkflowStepCompleted, WorkflowStepFailed, WorkflowStepSkipped,
    WorkflowBranchingOccurred, WorkflowMerged, WorkflowAdapted, WorkflowCancelled,
    
    # Interaction events
    UserInteractionStarted, UserInteractionCompleted, UserInteractionFailed,
    SessionStarted, SessionEnded, SessionExpired, SessionRestored, SessionContextUpdated,
    MessageReceived, MessageSent, MessageProcessed, ModalityDetected, InteractionModeChanged,
    
    # Plugin events
    PluginLoaded, PluginUnloaded, PluginEnabled, PluginDisabled, PluginError,
    PluginHotReloaded, PluginSecurityViolation, PluginPerformanceWarning,
    
    # Memory events
    MemoryOperationStarted, MemoryOperationCompleted, MemoryConsolidationStarted,
    MemoryConsolidationCompleted, MemoryLimitReached, MemoryCorruption,
    
    # Learning events
    LearningEventOccurred, FeedbackReceived, ModelAdapted, UserPreferenceUpdated,
    
    # Processing events
    ProcessingStarted, ProcessingCompleted, ProcessingError, ModalityProcessingStarted,
    ModalityProcessingCompleted, FusionStarted, FusionCompleted,
    
    # Error and performance events
    ErrorOccurred, PerformanceThresholdExceeded, ResourceLimitReached,
    SecurityIncident, AuditEvent, ConfigurationChanged
)
from src.core.error_handling import ErrorHandler, handle_exceptions
from src.core.dependency_injection import Container
from src.core.health_check import HealthCheck

# Observability
from src.observability.monitoring.metrics import MetricsCollector
from src.observability.monitoring.tracing import TraceManager
from src.observability.logging.config import get_logger

# Type definitions
T = TypeVar('T')
EventType = TypeVar('EventType')


class EventPriority(Enum):
    """Event processing priorities."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3
    EMERGENCY = 4


class EventCategory(Enum):
    """Event categories for classification."""
    SYSTEM = "system"
    COMPONENT = "component"
    WORKFLOW = "workflow"
    INTERACTION = "interaction"
    SESSION = "session"
    PLUGIN = "plugin"
    MEMORY = "memory"
    LEARNING = "learning"
    PROCESSING = "processing"
    ERROR = "error"
    PERFORMANCE = "performance"
    SECURITY = "security"
    AUDIT = "audit"


class EventHandlerState(Enum):
    """Event handler states."""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    PAUSED = "paused"
    ERROR = "error"
    SHUTDOWN = "shutdown"


class EventPattern(Enum):
    """Event pattern types for correlation."""
    SEQUENCE = "sequence"
    PARALLEL = "parallel"
    CONDITIONAL = "conditional"
    TEMPORAL = "temporal"
    CAUSAL = "causal"
    AGGREGATE = "aggregate"


@dataclass
class EventMetadata:
    """Metadata for event processing."""
    event_id: str
    event_type: str
    category: EventCategory
    priority: EventPriority = EventPriority.NORMAL
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    source_component: Optional[str] = None
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    correlation_id: Optional[str] = None
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    tags: Set[str] = field(default_factory=set)
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EventHandlerMetrics:
    """Metrics for event handler performance."""
    events_processed: int = 0
    events_failed: int = 0
    average_processing_time: float = 0.0
    peak_processing_time: float = 0.0
    queue_size: int = 0
    error_rate: float = 0.0
    throughput: float = 0.0
    last_reset: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class EventCorrelation:
    """Event correlation information."""
    correlation_id: str
    pattern: EventPattern
    events: List[str] = field(default_factory=list)
    conditions: Dict[str, Any] = field(default_factory=dict)
    timeout: float = 300.0  # 5 minutes
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed: bool = False
    result: Optional[Any] = None


class EventHandlerError(Exception):
    """Custom exception for event handler operations."""
    
    def __init__(self, message: str, handler_name: Optional[str] = None, 
                 event_type: Optional[str] = None):
        super().__init__(message)
        self.handler_name = handler_name
        self.event_type = event_type
        self.timestamp = datetime.now(timezone.utc)


class BaseEventHandler(ABC):
    """Abstract base class for event handlers."""
    
    def __init__(self, name: str, container: Container):
        """Initialize the event handler."""
        self.name = name
        self.container = container
        self.logger = get_logger(f"event_handler_{name}")
        self.state = EventHandlerState.INITIALIZING
        self.metrics = EventHandlerMetrics()
        self.processing_times = deque(maxlen=1000)
        self.error_history = deque(maxlen=100)
        
        # Configuration
        self.config = container.get(ConfigLoader)
        self.event_bus = container.get(EventBus)
        self.error_handler = container.get(ErrorHandler)
        
        # Observability
        try:
            self.metrics_collector = container.get(MetricsCollector)
            self.tracer = container.get(TraceManager)
        except Exception:
            self.metrics_collector = None
            self.tracer = None
        
        # Event processing configuration
        self.batch_size = self.config.get(f"event_handlers.{name}.batch_size", 10)
        self.batch_timeout = self.config.get(f"event_handlers.{name}.batch_timeout", 1.0)
        self.max_queue_size = self.config.get(f"event_handlers.{name}.max_queue_size", 1000)
        self.enable_batching = self.config.get(f"event_handlers.{name}.enable_batching", False)
        self.enable_correlation = self.config.get(f"event_handlers.{name}.enable_correlation", True)
        
        # Event queue and processing
        self.event_queue = asyncio.Queue(maxsize=self.max_queue_size)
        self.correlation_store: Dict[str, EventCorrelation] = {}
        self.processing_tasks: List[asyncio.Task] = []
        
        # Setup metrics
        self._setup_metrics()
    
    def _setup_metrics(self) -> None:
        """Setup metrics for this handler."""
        if self.metrics_collector:
            self.metrics_collector.register_counter(f"event_handler_{self.name}_events_total")
            self.metrics_collector.register_counter(f"event_handler_{self.name}_events_failed")
            self.metrics_collector.register_histogram(f"event_handler_{self.name}_processing_time_seconds")
            self.metrics_collector.register_gauge(f"event_handler_{self.name}_queue_size")
    
    @abstractmethod
    async def handle_event(self, event: Any, metadata: EventMetadata) -> Optional[Any]:
        """Handle a single event."""
        pass
    
    def can_handle(self, event: Any) -> bool:
        """Check if this handler can process the event."""
        return True
    
    def get_priority(self, event: Any) -> EventPriority:
        """Get priority for event processing."""
        return EventPriority.NORMAL
    
    def get_category(self, event: Any) -> EventCategory:
        """Get category for event classification."""
        return EventCategory.SYSTEM
    
    async def initialize(self) -> None:
        """Initialize the event handler."""
        try:
            # Start processing tasks
            if self.enable_batching:
                self.processing_tasks.append(
                    asyncio.create_task(self._batch_processing_loop())
                )
            else:
                for _ in range(self.config.get(f"event_handlers.{self.name}.worker_count", 2)):
                    self.processing_tasks.append(
                        asyncio.create_task(self._single_processing_loop())
                    )
            
            # Start correlation cleanup task
            if self.enable_correlation:
                self.processing_tasks.append(
                    asyncio.create_task(self._correlation_cleanup_loop())
                )
            
            # Start metrics update task
            self.processing_tasks.append(
                asyncio.create_task(self._metrics_update_loop())
            )
            
            self.state = EventHandlerState.ACTIVE
            self.logger.info(f"Event handler {self.name} initialized successfully")
            
        except Exception as e:
            self.state = EventHandlerState.ERROR
            self.logger.error(f"Failed to initialize event handler {self.name}: {str(e)}")
            raise EventHandlerError(f"Initialization failed: {str(e)}", self.name)
    
    async def process_event(self, event: Any) -> None:
        """Process an event asynchronously."""
        if self.state != EventHandlerState.ACTIVE:
            self.logger.warning(f"Handler {self.name} not active, dropping event")
            return
        
        if not self.can_handle(event):
            return
        
        # Create event metadata
        metadata = EventMetadata(
            event_id=str(uuid.uuid4()),
            event_type=type(event).__name__,
            category=self.get_category(event),
            priority=self.get_priority(event),
            source_component=getattr(event, 'component', None),
            session_id=getattr(event, 'session_id', None),
            user_id=getattr(event, 'user_id', None),
            correlation_id=getattr(event, 'correlation_id', None)
        )
        
        try:
            # Add to processing queue
            await self.event_queue.put((event, metadata))
            self.metrics.queue_size = self.event_queue.qsize()
            
            if self.metrics_collector:
                self.metrics_collector.set(f"event_handler_{self.name}_queue_size", self.metrics.queue_size)
            
        except asyncio.QueueFull:
            self.logger.error(f"Event queue full for handler {self.name}, dropping event")
            self.metrics.events_failed += 1
            
            if self.metrics_collector:
                self.metrics_collector.increment(f"event_handler_{self.name}_events_failed")
    
    async def _single_processing_loop(self) -> None:
        """Process events one by one."""
        while self.state == EventHandlerState.ACTIVE:
            try:
                # Get event from queue
                event, metadata = await asyncio.wait_for(
                    self.event_queue.get(), 
                    timeout=1.0
                )
                
                # Process event
                await self._process_single_event(event, metadata)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"Error in processing loop for {self.name}: {str(e)}")
                await asyncio.sleep(1.0)
    
    async def _batch_processing_loop(self) -> None:
        """Process events in batches."""
        batch = []
        last_batch_time = time.time()
        
        while self.state == EventHandlerState.ACTIVE:
            try:
                # Try to get events for batch
                try:
                    event, metadata = await asyncio.wait_for(
                        self.event_queue.get(), 
                        timeout=0.1
                    )
                    batch.append((event, metadata))
                except asyncio.TimeoutError:
                    pass
                
                current_time = time.time()
                
                # Process batch if conditions are met
                if (len(batch) >= self.batch_size or 
                    (batch and current_time - last_batch_time >= self.batch_timeout)):
                    
                    if batch:
                        await self._process_event_batch(batch)
                        batch.clear()
                        last_batch_time = current_time
                
                if not batch:
                    await asyncio.sleep(0.01)
                    
            except Exception as e:
                self.logger.error(f"Error in batch processing loop for {self.name}: {str(e)}")
                batch.clear()
                await asyncio.sleep(1.0)
    
    async def _process_single_event(self, event: Any, metadata: EventMetadata) -> None:
        """Process a single event with error handling and metrics."""
        start_time = time.time()
        
        try:
            with self.tracer.trace(f"event_handler_{self.name}") if self.tracer else None:
                # Handle correlation if enabled
                if self.enable_correlation and metadata.correlation_id:
                    await self._handle_event_correlation(event, metadata)
                
                # Process the event
                result = await self.handle_event(event, metadata)
                
                # Update success metrics
                processing_time = time.time() - start_time
                self.processing_times.append(processing_time)
                self.metrics.events_processed += 1
                
                if self.metrics_collector:
                    self.metrics_collector.increment(f"event_handler_{self.name}_events_total")
                    self.metrics_collector.record(f"event_handler_{self.name}_processing_time_seconds", processing_time)
                
                self.logger.debug(f"Processed event {metadata.event_type} in {processing_time:.3f}s")
                
        except Exception as e:
            # Handle processing error
            processing_time = time.time() - start_time
            self.metrics.events_failed += 1
            self.error_history.append({
                'timestamp': datetime.now(timezone.utc),
                'event_type': metadata.event_type,
                'error': str(e),
                'processing_time': processing_time
            })
            
            if self.metrics_collector:
                self.metrics_collector.increment(f"event_handler_{self.name}_events_failed")
            
            # Retry if configured
            if metadata.retry_count < metadata.max_retries:
                metadata.retry_count += 1
                await asyncio.sleep(min(2 ** metadata.retry_count, 30))  # Exponential backoff
                await self.event_queue.put((event, metadata))
                self.logger.warning(f"Retrying event {metadata.event_type} (attempt {metadata.retry_count})")
            else:
                self.logger.error(f"Failed to process event {metadata.event_type} after {metadata.retry_count} retries: {str(e)}")
                
                # Send to error handler
                if self.error_handler:
                    await self.error_handler.handle_error(e, {
                        'handler': self.name,
                        'event_type': metadata.event_type,
                        'event_id': metadata.event_id
                    })
    
    async def _process_event_batch(self, batch: List[tuple]) -> None:
        """Process a batch of events."""
        start_time = time.time()
        
        try:
            # Group events by type for efficient processing
            grouped_events = defaultdict(list)
            for event, metadata in batch:
                grouped_events[metadata.event_type].append((event, metadata))
            
            # Process each group
            for event_type, events in grouped_events.items():
                try:
                    await self._handle_event_group(event_type, events)
                except Exception as e:
                    self.logger.error(f"Failed to process event group {event_type}: {str(e)}")
                    # Fall back to individual processing
                    for event, metadata in events:
                        await self._process_single_event(event, metadata)
            
            processing_time = time.time() - start_time
            self.logger.debug(f"Processed batch of {len(batch)} events in {processing_time:.3f}s")
            
        except Exception as e:
            self.logger.error(f"Failed to process event batch: {str(e)}")
            # Fall back to individual processing
            for event, metadata in batch:
                await self._process_single_event(event, metadata)
    
    async def _handle_event_group(self, event_type: str, events: List[tuple]) -> None:
        """Handle a group of events of the same type."""
        # Default implementation processes individually
        for event, metadata in events:
            await self._process_single_event(event, metadata)
    
    async def _handle_event_correlation(self, event: Any, metadata: EventMetadata) -> None:
        """Handle event correlation logic."""
        correlation_id = metadata.correlation_id
        if not correlation_id:
            return
        
        # Get or create correlation
        if correlation_id not in self.correlation_store:
            self.correlation_store[correlation_id] = EventCorrelation(
                correlation_id=correlation_id,
                pattern=EventPattern.SEQUENCE  # Default pattern
            )
        
        correlation = self.correlation_store[correlation_id]
        correlation.events.append(metadata.event_id)
        
        # Check if correlation is complete
        if await self._is_correlation_complete(correlation, event, metadata):
            correlation.completed = True
            correlation.result = await self._process_correlated_events(correlation)
            
            # Cleanup
            del self.correlation_store[correlation_id]
    
    async def _is_correlation_complete(self, correlation: EventCorrelation, 
                                     event: Any, metadata: EventMetadata) -> bool:
        """Check if event correlation is complete."""
        # Simple implementation - can be extended for complex patterns
        return len(correlation.events) >= correlation.conditions.get('expected_count', 1)
    
    async def _process_correlated_events(self, correlation: EventCorrelation) -> Any:
        """Process correlated events together."""
        self.logger.debug(f"Processing correlated events: {correlation.correlation_id}")
        # Default implementation - can be overridden by specific handlers
        return {'correlation_id': correlation.correlation_id, 'event_count': len(correlation.events)}
    
    async def _correlation_cleanup_loop(self) -> None:
        """Clean up expired event correlations."""
        while self.state == EventHandlerState.ACTIVE:
            try:
                current_time = datetime.now(timezone.utc)
                expired_correlations = []
                
                for correlation_id, correlation in self.correlation_store.items():
                    if (current_time - correlation.created_at).total_seconds() > correlation.timeout:
                        expired_correlations.append(correlation_id)
                
                for correlation_id in expired_correlations:
                    del self.correlation_store[correlation_id]
                    self.logger.debug(f"Cleaned up expired correlation: {correlation_id}")
                
                await asyncio.sleep(60)  # Cleanup every minute
                
            except Exception as e:
                self.logger.error(f"Error in correlation cleanup: {str(e)}")
                await asyncio.sleep(60)
    
    async def _metrics_update_loop(self) -> None:
        """Update handler metrics periodically."""
        while self.state == EventHandlerState.ACTIVE:
            try:
                # Calculate metrics
                if self.processing_times:
                    self.metrics.average_processing_time = sum(self.processing_times) / len(self.processing_times)
                    self.metrics.peak_processing_time = max(self.processing_times)
                
                if self.metrics.events_processed > 0:
                    self.metrics.error_rate = self.metrics.events_failed / self.metrics.events_processed
                
                # Calculate throughput (events per second)
                current_time = datetime.now(timezone.utc)
                time_diff = (current_time - self.metrics.last_reset).total_seconds()
                if time_diff > 0:
                    self.metrics.throughput = self.metrics.events_processed / time_diff
                
                self.metrics.queue_size = self.event_queue.qsize()
                
                await asyncio.sleep(30)  # Update every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Error updating metrics: {str(e)}")
                await asyncio.sleep(30)
    
    async def pause(self) -> None:
        """Pause event processing."""
        self.state = EventHandlerState.PAUSED
        self.logger.info(f"Event handler {self.name} paused")
    
    async def resume(self) -> None:
        """Resume event processing."""
        if self.state == EventHandlerState.PAUSED:
            self.state = EventHandlerState.ACTIVE
            self.logger.info(f"Event handler {self.name} resumed")
    
    async def shutdown(self) -> None:
        """Shutdown the event handler gracefully."""
        self.state = EventHandlerState.SHUTDOWN
        
        # Cancel processing tasks
        for task in self.processing_tasks:
            task.cancel()
        
        if self.processing_tasks:
            await asyncio.gather(*self.processing_tasks, return_exceptions=True)
        
        # Process remaining events
        while not self.event_queue.empty():
            try:
                event, metadata = self.event_queue.get_nowait()
                await self._process_single_event(event, metadata)
            except asyncio.QueueEmpty:
                break
            except Exception as e:
                self.logger.error(f"Error processing final events: {str(e)}")
        
        self.logger.info(f"Event handler {self.name} shutdown completed")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get handler metrics."""
        return asdict(self.metrics)
    
    def get_status(self) -> Dict[str, Any]:
        """Get handler status."""
        return {
            'name': self.name,
            'state': self.state.value,
            'queue_size': self.event_queue.qsize(),
            'correlation_count': len(self.correlation_store),
            'processing_tasks': len(self.processing_tasks),
            'metrics': self.get_metrics()
        }


class SystemEventHandler(BaseEventHandler):
    """Handler for system-level events."""
    
    def __init__(self, container: Container):
        super().__init__("system", container)
        self.system_state = {}
        self.component_states = {}
        
    def can_handle(self, event: Any) -> bool:
        """Check if this handler can process system events."""
        return isinstance(event, (
            EngineStarted, EngineShutdown, SystemStateChanged, SystemShutdownStarted,
            SystemShutdownCompleted, SystemHealthCheck, SystemPerformanceAlert
        ))
    
    def get_category(self, event: Any) -> EventCategory:
        """Get category for system events."""
        return EventCategory.SYSTEM
    
    def get_priority(self, event: Any) -> EventPriority:
        """Get priority for system events."""
        if isinstance(event, (EngineShutdown, SystemShutdownStarted)):
            return EventPriority.CRITICAL
        elif isinstance(event, SystemPerformanceAlert):
            return EventPriority.HIGH
        return EventPriority.NORMAL
    
    async def handle_event(self, event: Any, metadata: EventMetadata) -> Optional[Any]:
        """Handle system events."""
        if isinstance(event, EngineStarted):
            return await self._handle_engine_started(event, metadata)
        elif isinstance(event, EngineShutdown):
            return await self._handle_engine_shutdown(event, metadata)
        elif isinstance(event, SystemStateChanged):
            return await self._handle_system_state_changed(event, metadata)
        elif isinstance(event, SystemShutdownStarted):
            return await self._handle_shutdown_started(event, metadata)
        elif isinstance(event, SystemShutdownCompleted):
            return await self._handle_shutdown_completed(event, metadata)
        elif isinstance(event, SystemHealthCheck):
            return await self._handle_health_check(event, metadata)
        elif isinstance(event, SystemPerformanceAlert):
            return await self._handle_performance_alert(event, metadata)
        
        return None
    
    async def _handle_engine_started(self, event: EngineStarted, metadata: EventMetadata) -> Dict[str, Any]:
        """Handle engine started event."""
        self.system_state['engine_started'] = True
        self.system_state['engine_start_time'] = event.startup_time
        self.system_state['engine_version'] = event.version
        
        self.logger.info(f"Engine started: version {event.version}, components: {event.components_loaded}")
        
        return {'status': 'acknowledged', 'engine_id': event.engine_id}
    
    async def _handle_engine_shutdown(self, event: EngineShutdown, metadata: EventMetadata) -> Dict[str, Any]:
        """Handle engine shutdown event."""
        self.system_state['engine_started'] = False
        self.system_state['engine_shutdown_time'] = event.shutdown_time
        
        uptime = event.uptime_seconds if hasattr(event, 'uptime_seconds') else 0
        self.logger.info(f"Engine shutdown after {uptime:.2f}s uptime")
        
        return {'status': 'acknowledged', 'uptime': uptime}
    
    async def _handle_system_state_changed(self, event: SystemStateChanged, metadata: EventMetadata) -> Dict[str, Any]:
        """Handle system state change event."""
        self.system_state[event.component] = event.new_state
        
        self.logger.info(f"System state changed: {event.component} -> {event.new_state}")
        
        return {'status': 'state_updated', 'component': event.component}
    
    async def _handle_shutdown_started(self, event: SystemShutdownStarted, metadata: EventMetadata) -> Dict[str, Any]:
        """Handle shutdown started event."""
        self.system_state['shutdown_in_progress'] = True
        
        self.logger.info("System shutdown initiated")
        
        # Trigger graceful shutdown of other components
        await self._trigger_graceful_shutdown()
        
        return {'status': 'shutdown_initiated'}
    
    async def _handle_shutdown_completed(self, event: SystemShutdownCompleted, metadata: EventMetadata) -> Dict[str, Any]:
        """Handle shutdown completed event."""
        self.system_state['shutdown_completed'] = True
        
        shutdown_time = getattr(event, 'shutdown_time', 0)
        self.logger.info(f"System shutdown completed in {shutdown_time:.2f}s")
        
        return {'status': 'shutdown_completed', 'shutdown_time': shutdown_time}
    
    async def _handle_health_check(self, event: SystemHealthCheck, metadata: EventMetadata) -> Dict[str, Any]:
        """Handle system health check event."""
        # Aggregate health status from all components
        overall_health = self._calculate_system_health()
        
        self.logger.debug(f"System health check: {overall_health}")
        
        return {'health_status': overall_health}
    
    async def _handle_performance_alert(self, event: SystemPerformanceAlert, metadata: EventMetadata) -> Dict[str, Any]:
        """Handle system performance alert."""
        alert_type = getattr(event, 'alert_type', 'unknown')
        metric_value = getattr(event, 'metric_value', 0)
        threshold = getattr(event, 'threshold', 0)
        
        self.logger.warning(f"Performance alert: {alert_type} = {metric_value} (threshold: {threshold})")
        
        # Trigger performance optimization if available
        await self._trigger_performance_optimization(alert_type, metric_value, threshold)
        
        return {'status': 'alert_processed', 'alert_type': alert_type}
    
    async def _trigger_graceful_shutdown(self) -> None:
        """Trigger graceful shutdown of system components."""
        try:
            # This would coordinate with component manager for graceful shutdown
            self.logger.info("Triggering graceful component shutdown")
        except Exception as e:
            self.logger.error(f"Error during graceful shutdown: {str(e)}")
    
    def _calculate_system_health(self) -> Dict[str, Any]:
        """Calculate overall system health."""
        # Simple implementation - can be enhanced with actual health metrics
        healthy_components = sum(1 for state in self.component_states.values() if state == 'healthy')
        total_components = len(self.component_states) or 1
        
        return {
            'overall_status': 'healthy' if healthy_components / total_components > 0.8 else 'degraded',
            'healthy_components': healthy_components,
            'total_components': total_components,
            'health_ratio': healthy_components / total_components
        }
    
    async def _trigger_performance_optimization(self, alert_type: str, value: float, threshold: float) -> None:
        """Trigger performance optimization based on alert."""
        try:
            # This would implement performance optimization strategies
            self.logger.info(f"Triggering performance optimization for {alert_type}")
        except Exception as e:
            self.logger.error(f"Error during performance optimization: {str(e)}")


class ComponentEventHandler(BaseEventHandler):
    """Handler for component-related events."""
    
    def __init__(self, container: Container):
        super().__init__("component", container)
        self.component_registry = {}
        self.dependency_graph = defaultdict(set)
        
    def can_handle(self, event: Any) -> bool:
        """Check if this handler can process component events."""
        return isinstance(event, (
            ComponentRegistered, ComponentInitialized, ComponentStarted, ComponentStopped,
            ComponentFailed, ComponentHealthChanged, ComponentRestarted, ComponentUpgraded
        ))
    
    def get_category(self, event: Any) -> EventCategory:
        """Get category for component events."""
        return EventCategory.COMPONENT
    
    def get_priority(self, event: Any) -> EventPriority:
        """Get priority for component events."""
        if isinstance(event, ComponentFailed):
            return EventPriority.HIGH
        elif isinstance(event, ComponentHealthChanged) and not event.healthy:
            return EventPriority.HIGH
        return EventPriority.NORMAL
    
    async def handle_event(self, event: Any, metadata: EventMetadata) -> Optional[Any]:
        """Handle component events."""
        if isinstance(event, ComponentRegistered):
            return await self._handle_component_registered(event, metadata)
        elif isinstance(event, ComponentInitialized):
            return await self._handle_component_initialized(event, metadata)
        elif isinstance(event, ComponentStarted):
            return await self._handle_component_started(event, metadata)
        elif isinstance(event, ComponentStopped):
            return await self._handle_component_stopped(event, metadata)
        elif isinstance(event, ComponentFailed):
            return await self._handle_component_failed(event, metadata)
        elif isinstance(event, ComponentHealthChanged):
            return await self._handle_component_health_changed(event, metadata)
        elif isinstance(event, ComponentRestarted):
            return await self._handle_component_restarted(event, metadata)
        elif isinstance(event, ComponentUpgraded):
            return await self._handle_component_upgraded(event, metadata)
        
        return None
    
    async def _handle_component_registered(self, event: ComponentRegistered, metadata: EventMetadata) -> Dict[str, Any]:
        """Handle component registration event."""
        component_id = event.component_id
        self.component_registry[component_id] = {
            'component_type': event.component_type,
            'priority': event.priority,
            'registered_at': datetime.now(timezone.utc),
            'state': 'registered'
        }
        
        self.logger.info(f"Component registered: {component_id} ({event.component_type})")
        
        return {'status': 'registered', 'component_id': component_id}
    
    async def _handle_component_initialized(self, event: ComponentInitialized, metadata: EventMetadata) -> Dict[str, Any]:
        """Handle component initialization event."""
        component_id = event.component_id
        if component_id in self.component_registry:
            self.component_registry[component_id].update({
                'state': 'initialized',
                'initialization_time': event.initialization_time,
                'initialized_at': datetime.now(timezone.utc)
            })
        
        self.logger.info(f"Component initialized: {component_id} in {event.initialization_time:.3f}s")
        
        return {'status': 'initialized', 'component_id': component_id}
    
    async def _handle_component_started(self, event: ComponentStarted, metadata: EventMetadata) -> Dict[str, Any]:
        """Handle component started event."""
        component_id = event.component_id
        if component_id in self.component_registry:
            self.component_registry[component_id].update({
                'state': 'running',
                'started_at': datetime.now(timezone.utc)
            })
        
        self.logger.info(f"Component started: {component_id}")
        
        return {'status': 'started', 'component_id': component_id}
    
    async def _handle_component_stopped(self, event: ComponentStopped, metadata: EventMetadata) -> Dict[str, Any]:
        """Handle component stopped event."""
        component_id = event.component_id
        if component_id in self.component_registry:
            self.component_registry[component_id].update({
                'state': 'stopped',
                'stopped_at': datetime.now(timezone.utc)
            })
        
        self.logger.info(f"Component stopped: {component_id}")
        
        return {'status': 'stopped', 'component_id': component_id}
    
    async def _handle_component_failed(self, event: ComponentFailed, metadata: EventMetadata) -> Dict[str, Any]:
        """Handle component failure event."""
        component_id = event.component_id
        if component_id in self.component_registry:
            self.component_registry[component_id].update({
                'state': 'failed',
                'error_message': event.error_message,
                'error_type': event.error_type,
                'failed_at': datetime.now(timezone.utc)
            })
        
        self.logger.error(f"Component failed: {component_id} - {event.error_message}")
        
        # Trigger recovery if available
        await self._trigger_component_recovery(component_id, event.error_type)
        
        return {'status': 'failure_handled', 'component_id': component_id}
    
    async def _handle_component_health_changed(self, event: ComponentHealthChanged, metadata: EventMetadata) -> Dict[str, Any]:
        """Handle component health change event."""
        component_id = event.component
        healthy = event.healthy
        
        if component_id in self.component_registry:
            self.component_registry[component_id].update({
                'healthy': healthy,
                'health_details': event.details,
                'last_health_check': datetime.now(timezone.utc)
            })
        
        status = "healthy" if healthy else "unhealthy"
        self.logger.info(f"Component health changed: {component_id} -> {status}")
        
        if not healthy:
            # Trigger health recovery
            await self._trigger_health_recovery(component_id, event.details)
        
        return {'status': 'health_updated', 'component_id': component_id, 'healthy': healthy}
    
    async def _handle_component_restarted(self, event: ComponentRestarted, metadata: EventMetadata) -> Dict[str, Any]:
        """Handle component restart event."""
        component_id = event.component_id
        if component_id in self.component_registry:
            self.component_registry[component_id].update({
                'state': 'running',
                'restart_count': self.component_registry[component_id].get('restart_count', 0) + 1,
                'last_restart': datetime.now(timezone.utc)
            })
        
        self.logger.info(f"Component restarted: {component_id}")
        
        return {'status': 'restarted', 'component_id': component_id}
    
    async def _handle_component_upgraded(self, event: ComponentUpgraded, metadata: EventMetadata) -> Dict[str, Any]:
        """Handle component upgrade event."""
        component_id = event.component_id
        if component_id in self.component_registry:
            self.component_registry[component_id].update({
                'version': event.new_version,
                'previous_version': event.old_version,
                'upgraded_at': datetime.now(timezone.utc)
            })
        
        self.logger.info(f"Component upgraded: {component_id} {event.old_version} -> {event.new_version}")
        
        return {'status': 'upgraded', 'component_id': component_id}
    
    async def _trigger_component_recovery(self, component_id: str, error_type: str) -> None:
        """Trigger component recovery procedures."""
        try:
            self.logger.info(f"Triggering recovery for component {component_id}")
            # This would implement component recovery strategies
        except Exception as e:
            self.logger.error(f"Error during component recovery: {str(e)}")
    
    async def _trigger_health_recovery(self, component_id: str, health_details: Dict[str, Any]) -> None:
        """Trigger health recovery procedures."""
        try:
            self.logger.info(f"Triggering health recovery for component {component_id}")
            # This would implement health recovery strategies
        except Exception as e:
            self.logger.error(f"Error during health recovery: {str(e)}")


class WorkflowEventHandler(BaseEventHandler):
    """Handler for workflow-related events."""
    
    def __init__(self, container: Container):
        super().__init__("workflow", container)
        self.active_workflows = {}
        self.workflow_metrics = defaultdict(lambda: {
            'total_executions': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'average_duration': 0.0
        })
    
    def can_handle(self, event: Any) -> bool:
        """Check if this handler can process workflow events."""
        return isinstance(event, (
            WorkflowStarted, WorkflowCompleted, WorkflowFailed, WorkflowPaused, WorkflowResumed,
            WorkflowStepStarted, WorkflowStepCompleted, WorkflowStepFailed, WorkflowStepSkipped,
            WorkflowBranchingOccurred, WorkflowMerged, WorkflowAdapted, WorkflowCancelled
        ))
    
    def get_category(self, event: Any) -> EventCategory:
        """Get category for workflow events."""
        return EventCategory.WORKFLOW
    
    def get_priority(self, event: Any) -> EventPriority:
        """Get priority for workflow events."""
        if isinstance(event, WorkflowFailed):
            return EventPriority.HIGH
        elif isinstance(event, WorkflowStepFailed):
            return EventPriority.NORMAL
        return EventPriority.NORMAL
    
    async def handle_event(self, event: Any, metadata: EventMetadata) -> Optional[Any]:
        """Handle workflow events."""
        if isinstance(event, WorkflowStarted):
            return await self._handle_workflow_started(event, metadata)
        elif isinstance(event, WorkflowCompleted):
            return await self._handle_workflow_completed(event, metadata)
        elif isinstance(event, WorkflowFailed):
            return await self._handle_workflow_failed(event, metadata)
        elif isinstance(event, WorkflowPaused):
            return await self._handle_workflow_paused(event, metadata)
        elif isinstance(event, WorkflowResumed):
            return await self._handle_workflow_resumed(event, metadata)
        elif isinstance(event, WorkflowStepStarted):
            return await self._handle_step_started(event, metadata)
        elif isinstance(event, WorkflowStepCompleted):
            return await self._handle_step_completed(event, metadata)
        elif isinstance(event, WorkflowStepFailed):
            return await self._handle_step_failed(event, metadata)
        elif isinstance(event, WorkflowCancelled):
            return await self._handle_workflow_cancelled(event, metadata)
        
        return None
    
    async def _handle_workflow_started(self, event: WorkflowStarted, metadata: EventMetadata) -> Dict[str, Any]:
        """Handle workflow started event."""
        execution_id = event.execution_id
        workflow_id = event.workflow_id
        
        self.active_workflows[execution_id] = {
            'workflow_id': workflow_id,
            'workflow_name': event.workflow_name,
            'session_id': event.session_id,
            'started_at': datetime.now(timezone.utc),
            'status': 'running',
            'steps': {}
        }
        
        self.workflow_metrics[workflow_id]['total_executions'] += 1
        
        self.logger.info(f"Workflow started: {event.workflow_name} (execution: {execution_id})")
        
        return {'status': 'tracking_started', 'execution_id': execution_id}
    
    async def _handle_workflow_completed(self, event: WorkflowCompleted, metadata: EventMetadata) -> Dict[str, Any]:
        """Handle workflow completed event."""
        execution_id = event.execution_id
        workflow_id = event.workflow_id
        
        if execution_id in self.active_workflows:
            workflow = self.active_workflows[execution_id]
            workflow.update({
                'status': 'completed',
                'completed_at': datetime.now(timezone.utc),
                'execution_time': event.execution_time,
                'steps_completed': event.steps_completed
            })
            
            # Update metrics
            self.workflow_metrics[workflow_id]['successful_executions'] += 1
            
            # Calculate average duration
            metrics = self.workflow_metrics[workflow_id]
            total_successful = metrics['successful_executions']
            if total_successful > 0:
                old_avg = metrics['average_duration']
                metrics['average_duration'] = (
                    (old_avg * (total_successful - 1) + event.execution_time) / total_successful
                )
            
            # Clean up from active workflows
            del self.active_workflows[execution_id]
        
        self.logger.info(f"Workflow completed: {execution_id} in {event.execution_time:.2f}s")
        
        return {'status': 'completion_tracked', 'execution_id': execution_id}
    
    async def _handle_workflow_failed(self, event: WorkflowFailed, metadata: EventMetadata) -> Dict[str, Any]:
        """Handle workflow failed event."""
        execution_id = event.execution_id
        workflow_id = event.workflow_id
        
        if execution_id in self.active_workflows:
            workflow = self.active_workflows[execution_id]
            workflow.update({
                'status': 'failed',
                'failed_at': datetime.now(timezone.utc),
                'error_message': event.error_message,
                'execution_time': event.execution_time
            })
            
            # Update metrics
            self.workflow_metrics[workflow_id]['failed_executions'] += 1
            
            # Clean up from active workflows
            del self.active_workflows[execution_id]
        
        self.logger.error(f"Workflow failed: {execution_id} - {event.error_message}")
        
        # Trigger failure analysis
        await self._analyze_workflow_failure(workflow_id, event.error_message)
        
        return {'status': 'failure_tracked', 'execution_id': execution_id}
    
    async def _handle_workflow_paused(self, event: WorkflowPaused, metadata: EventMetadata) -> Dict[str, Any]:
        """Handle workflow paused event."""
        execution_id = event.execution_id
        
        if execution_id in self.active_workflows:
            self.active_workflows[execution_id].update({
                'status': 'paused',
                'paused_at': datetime.now(timezone.utc)
            })
        
        self.logger.info(f"Workflow paused: {execution_id}")
        
        return {'status': 'pause_tracked', 'execution_id': execution_id}
    
    async def _handle_workflow_resumed(self, event: WorkflowResumed, metadata: EventMetadata) -> Dict[str, Any]:
        """Handle workflow resumed event."""
        execution_id = event.execution_id
        
        if execution_id in self.active_workflows:
            self.active_workflows[execution_id].update({
                'status': 'running',
                'resumed_at': datetime.now(timezone.utc)
            })
        
        self.logger.info(f"Workflow resumed: {execution_id}")
        
        return {'status': 'resume_tracked', 'execution_id': execution_id}
    
    async def _handle_step_started(self, event: WorkflowStepStarted, metadata: EventMetadata) -> Dict[str, Any]:
        """Handle workflow step started event."""
        execution_id = event.execution_id
        step_id = event.step_id
        
        if execution_id in self.active_workflows:
            workflow = self.active_workflows[execution_id]
            workflow['steps'][step_id] = {
                'step_name': event.step_name,
                'step_type': event.step_type,
                'started_at': datetime.now(timezone.utc),
                'status': 'running'
            }
        
        self.logger.debug(f"Step started: {event.step_name} in workflow {execution_id}")
        
        return {'status': 'step_tracking_started', 'step_id': step_id}
    
    async def _handle_step_completed(self, event: WorkflowStepCompleted, metadata: EventMetadata) -> Dict[str, Any]:
        """Handle workflow step completed event."""
        execution_id = event.execution_id
        step_id = event.step_id
        
        if execution_id in self.active_workflows:
            workflow = self.active_workflows[execution_id]
            if step_id in workflow['steps']:
                workflow['steps'][step_id].update({
                    'status': 'completed',
                    'completed_at': datetime.now(timezone.utc),
                    'execution_time': event.execution_time,
                    'success': event.success
                })
        
        self.logger.debug(f"Step completed: {event.step_name} in {event.execution_time:.3f}s")
        
        return {'status': 'step_completion_tracked', 'step_id': step_id}
    
    async def _handle_step_failed(self, event: WorkflowStepFailed, metadata: EventMetadata) -> Dict[str, Any]:
        """Handle workflow step failed event."""
        execution_id = event.execution_id
        step_id = event.step_id
        
        if execution_id in self.active_workflows:
            workflow = self.active_workflows[execution_id]
            if step_id in workflow['steps']:
                workflow['steps'][step_id].update({
                    'status': 'failed',
                    'failed_at': datetime.now(timezone.utc),
                    'error_message': event.error_message,
                    'error_type': event.error_type
                })
        
        self.logger.warning(f"Step failed: {step_id} - {event.error_message}")
        
        return {'status': 'step_failure_tracked', 'step_id': step_id}
    
    async def _handle_workflow_cancelled(self, event: WorkflowCancelled, metadata: EventMetadata) -> Dict[str, Any]:
        """Handle workflow cancelled event."""
        execution_id = event.execution_id
        
        if execution_id in self.active_workflows:
            workflow = self.active_workflows[execution_id]
            workflow.update({
                'status': 'cancelled',
                'cancelled_at': datetime.now(timezone.utc)
            })
            
            # Clean up from active workflows
            del self.active_workflows[execution_id]
        
        self.logger.info(f"Workflow cancelled: {execution_id}")
        
        return {'status': 'cancellation_tracked', 'execution_id': execution_id}
    
    async def _analyze_workflow_failure(self, workflow_id: str, error_message: str) -> None:
        """Analyze workflow failure patterns."""
        try:
            metrics = self.workflow_metrics[workflow_id]
            failure_rate = metrics['failed_executions'] / max(metrics['total_executions'], 1)
            
            if failure_rate > 0.5:  # High failure rate
                self.logger.warning(f"High failure rate detected for workflow {workflow_id}: {failure_rate:.2f}")
                # Trigger workflow optimization or alerting
        except Exception as e:
            self.logger.error(f"Error analyzing workflow failure: {str(e)}")


class InteractionEventHandler(BaseEventHandler):
    """Handler for user interaction events."""
    
    def __init__(self, container: Container):
        super().__init__("interaction", container)
        self.active_interactions = {}
        self.user_statistics = defaultdict(lambda: {
            'total_interactions': 0,
            'successful_interactions': 0,
            'average_response_time': 0.0,
            'preferred_modalities': defaultdict(int)
        })
    
    def can_handle(self, event: Any) -> bool:
        """Check if this handler can process interaction events."""
        return isinstance(event, (
            UserInteractionStarted, UserInteractionCompleted, UserInteractionFailed,
            MessageReceived, MessageSent, MessageProcessed, ModalityDetected, InteractionModeChanged
        ))
    
    def get_category(self, event: Any) -> EventCategory:
        """Get category for interaction events."""
        return EventCategory.INTERACTION
    
    def get_priority(self, event: Any) -> EventPriority:
        """Get priority for interaction events."""
        if isinstance(event, UserInteractionFailed):
            return EventPriority.HIGH
        return EventPriority.NORMAL
    
    async def handle_event(self, event: Any, metadata: EventMetadata) -> Optional[Any]:
        """Handle interaction events."""
        if isinstance(event, UserInteractionStarted):
            return await self._handle_interaction_started(event, metadata)
        elif isinstance(event, UserInteractionCompleted):
            return await self._handle_interaction_completed(event, metadata)
        elif isinstance(event, UserInteractionFailed):
            return await self._handle_interaction_failed(event, metadata)
        elif isinstance(event, MessageReceived):
            return await self._handle_message_received(event, metadata)
        elif isinstance(event, MessageSent):
            return await self._handle_message_sent(event, metadata)
        elif isinstance(event, MessageProcessed):
            return await self._handle_message_processed(event, metadata)
        elif isinstance(event, ModalityDetected):
            return await self._handle_modality_detected(event, metadata)
        elif isinstance(event, InteractionModeChanged):
            return await self._handle_interaction_mode_changed(event, metadata)
        
        return None
    
    async def _handle_interaction_started(self, event: UserInteractionStarted, metadata: EventMetadata) -> Dict[str, Any]:
        """Handle interaction started event."""
        interaction_id = event.interaction_id
        user_id = event.user_id
        
        self.active_interactions[interaction_id] = {
            'session_id': event.session_id,
            'user_id': user_id,
            'interaction_mode': event.interaction_mode,
            'input_modalities': event.input_modalities,
            'output_modalities': event.output_modalities,
            'started_at': datetime.now(timezone.utc),
            'status': 'active',
            'messages': []
        }
        
        if user_id:
            self.user_statistics[user_id]['total_interactions'] += 1
            
            # Track modality preferences
            for modality in event.input_modalities:
                self.user_statistics[user_id]['preferred_modalities'][modality] += 1
        
        self.logger.info(f"Interaction started: {interaction_id} for user {user_id}")
        
        return {'status': 'tracking_started', 'interaction_id': interaction_id}
    
    async def _handle_interaction_completed(self, event: UserInteractionCompleted, metadata: EventMetadata) -> Dict[str, Any]:
        """Handle interaction completed event."""
        interaction_id = event.interaction_id
        user_id = event.user_id
        
        if interaction_id in self.active_interactions:
            interaction = self.active_interactions[interaction_id]
            interaction.update({
                'status': 'completed',
                'completed_at': datetime.now(timezone.utc),
                'duration': event.duration,
                'message_count': event.message_count,
                'reason': event.reason
            })
            
            if user_id:
                # Update user statistics
                stats = self.user_statistics[user_id]
                stats['successful_interactions'] += 1
                
                # Update average response time
                total_successful = stats['successful_interactions']
                if total_successful > 0:
                    old_avg = stats['average_response_time']
                    stats['average_response_time'] = (
                        (old_avg * (total_successful - 1) + event.duration) / total_successful
                    )
            
            # Clean up from active interactions
            del self.active_interactions[interaction_id]
        
        self.logger.info(f"Interaction completed: {interaction_id} in {event.duration:.2f}s")
        
        return {'status': 'completion_tracked', 'interaction_id': interaction_id}
    
    async def _handle_interaction_failed(self, event: UserInteractionFailed, metadata: EventMetadata) -> Dict[str, Any]:
        """Handle interaction failed event."""
        interaction_id = event.interaction_id
        
        if interaction_id in self.active_interactions:
            interaction = self.active_interactions[interaction_id]
            interaction.update({
                'status': 'failed',
                'failed_at': datetime.now(timezone.utc),
                'error_message': event.error_message,
                'error_type': event.error_type
            })
            
            # Clean up from active interactions
            del self.active_interactions[interaction_id]
        
        self.logger.error(f"Interaction failed: {interaction_id} - {event.error_message}")
        
        # Trigger failure analysis
        await self._analyze_interaction_failure(interaction_id, event.error_type)
        
        return {'status': 'failure_tracked', 'interaction_id': interaction_id}
    
    async def _handle_message_received(self, event: MessageReceived, metadata: EventMetadata) -> Dict[str, Any]:
        """Handle message received event."""
        interaction_id = event.interaction_id
        message_id = event.message_id
        
        if interaction_id in self.active_interactions:
            interaction = self.active_interactions[interaction_id]
            interaction['messages'].append({
                'message_id': message_id,
                'type': 'received',
                'modality': event.modality,
                'content_preview': event.content_preview,
                'timestamp': datetime.now(timezone.utc)
            })
        
        self.logger.debug(f"Message received: {message_id} in interaction {interaction_id}")
        
        return {'status': 'message_tracked', 'message_id': message_id}
    
    async def _handle_message_sent(self, event: MessageSent, metadata: EventMetadata) -> Dict[str, Any]:
        """Handle message sent event."""
        interaction_id = event.interaction_id
        message_id = event.message_id
        
        if interaction_id in self.active_interactions:
            interaction = self.active_interactions[interaction_id]
            interaction['messages'].append({
                'message_id': message_id,
                'type': 'sent',
                'modalities': event.modalities,
                'content_preview': event.content_preview,
                'timestamp': datetime.now(timezone.utc)
            })
        
        self.logger.debug(f"Message sent: {message_id} in interaction {interaction_id}")
        
        return {'status': 'message_tracked', 'message_id': message_id}
    
    async def _handle_message_processed(self, event: MessageProcessed, metadata: EventMetadata) -> Dict[str, Any]:
        """Handle message processed event."""
        message_id = event.message_id
        interaction_id = event.interaction_id
        
        # Find and update message in interaction
        if interaction_id in self.active_interactions:
            interaction = self.active_interactions[interaction_id]
            for message in interaction['messages']:
                if message['message_id'] == message_id:
                    message.update({
                        'processing_time': event.processing_time,
                        'confidence': event.confidence,
                        'success': event.success,
                        'processed_at': datetime.now(timezone.utc)
                    })
                    break
        
        self.logger.debug(f"Message processed: {message_id} in {event.processing_time:.3f}s")
        
        return {'status': 'processed'}
