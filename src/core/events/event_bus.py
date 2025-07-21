"""
Advanced Event Bus System for AI Assistant
Author: Drmusab
Last Modified: 2025-06-13 10:20:08 UTC

This module provides a comprehensive event-driven communication system for the AI assistant,
enabling asynchronous, type-safe, and high-performance event processing across all components.
"""

from pathlib import Path
from typing import Optional, Dict, Any, List, Set, Callable, Type, Union, AsyncGenerator, TypeVar, Generic
import asyncio
import threading
import time
import weakref
import uuid
import json
import hashlib
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
from contextlib import asynccontextmanager
from collections import defaultdict, deque
from abc import ABC, abstractmethod
import logging
import inspect
from concurrent.futures import ThreadPoolExecutor
import pickle
import base64

# Core imports
# Remove circular dependency imports - these will be injected later if needed
# from src.core.config.loader import ConfigLoader
# from src.core.error_handling import ErrorHandler, handle_exceptions  
# from src.core.dependency_injection import Container
# from src.core.health_check import HealthCheck

# Event types
from src.core.events.event_types import (
    BaseEvent, EventMetadata, EventPriority, EventCategory
)

# Observability - remove circular dependencies
# from src.observability.monitoring.metrics import MetricsCollector
# from src.observability.monitoring.tracing import TraceManager
from src.observability.logging.config import get_logger

# Type definitions
T = TypeVar('T', bound=BaseEvent)
EventHandler = Callable[[T], Any]
EventFilter = Callable[[BaseEvent], bool]


class EventState(Enum):
    """Event processing states."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    DEAD_LETTER = "dead_letter"
    EXPIRED = "expired"


class SubscriptionType(Enum):
    """Types of event subscriptions."""
    EXACT = "exact"           # Exact event type match
    WILDCARD = "wildcard"     # Pattern-based matching
    CATEGORY = "category"     # Category-based matching
    CONDITIONAL = "conditional"  # Condition-based matching
    BROADCAST = "broadcast"   # Receive all events


class EventDeliveryMode(Enum):
    """Event delivery modes."""
    ASYNC = "async"          # Asynchronous delivery
    SYNC = "sync"            # Synchronous delivery
    FIRE_AND_FORGET = "fire_and_forget"  # No response expected
    REQUEST_REPLY = "request_reply"      # Expect a response
    BATCH = "batch"          # Batch delivery


class CircuitBreakerState(Enum):
    """Circuit breaker states for event handling."""
    CLOSED = "closed"        # Normal operation
    OPEN = "open"            # Failing, events rejected
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class EventSubscription:
    """Represents an event subscription."""
    subscription_id: str
    event_types: Set[str]
    handler: EventHandler
    subscription_type: SubscriptionType = SubscriptionType.EXACT
    delivery_mode: EventDeliveryMode = EventDeliveryMode.ASYNC
    
    # Filtering and routing
    filter_func: Optional[EventFilter] = None
    priority_threshold: EventPriority = EventPriority.LOW
    tags_filter: Set[str] = field(default_factory=set)
    source_filter: Set[str] = field(default_factory=set)
    
    # Reliability
    max_retries: int = 3
    retry_delay: float = 1.0
    timeout_seconds: float = 30.0
    enable_dead_letter: bool = True
    
    # Performance
    max_queue_size: int = 1000
    batch_size: int = 1
    batch_timeout: float = 1.0
    
    # State tracking
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_event_at: Optional[datetime] = None
    total_events_received: int = 0
    successful_events: int = 0
    failed_events: int = 0
    
    # Metadata
    subscriber_name: Optional[str] = None
    description: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EventEnvelope:
    """Wrapper for events with delivery metadata."""
    event_id: str
    event: BaseEvent
    subscription_id: str
    
    # Processing state
    state: EventState = EventState.PENDING
    attempt_count: int = 0
    max_retries: int = 3
    
    # Timing
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    scheduled_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Delivery tracking
    delivery_attempts: List[Dict[str, Any]] = field(default_factory=list)
    last_error: Optional[str] = None
    
    # Performance metrics
    processing_time: float = 0.0
    queue_time: float = 0.0
    
    # Metadata
    correlation_id: Optional[str] = None
    reply_to: Optional[str] = None
    headers: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CircuitBreaker:
    """Circuit breaker for event handler reliability."""
    handler_id: str
    state: CircuitBreakerState = CircuitBreakerState.CLOSED
    failure_count: int = 0
    failure_threshold: int = 5
    success_count: int = 0
    success_threshold: int = 3
    timeout_duration: float = 60.0
    last_failure_time: Optional[datetime] = None
    
    def should_allow_request(self) -> bool:
        """Check if request should be allowed."""
        if self.state == CircuitBreakerState.CLOSED:
            return True
        elif self.state == CircuitBreakerState.OPEN:
            if self.last_failure_time:
                time_since_failure = (datetime.now(timezone.utc) - self.last_failure_time).total_seconds()
                if time_since_failure > self.timeout_duration:
                    self.state = CircuitBreakerState.HALF_OPEN
                    return True
            return False
        elif self.state == CircuitBreakerState.HALF_OPEN:
            return True
        return False
    
    def record_success(self) -> None:
        """Record a successful operation."""
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                self.state = CircuitBreakerState.CLOSED
                self.failure_count = 0
                self.success_count = 0
        elif self.state == CircuitBreakerState.CLOSED:
            self.failure_count = 0
    
    def record_failure(self) -> None:
        """Record a failed operation."""
        self.failure_count += 1
        self.last_failure_time = datetime.now(timezone.utc)
        
        if self.state == CircuitBreakerState.CLOSED:
            if self.failure_count >= self.failure_threshold:
                self.state = CircuitBreakerState.OPEN
        elif self.state == CircuitBreakerState.HALF_OPEN:
            self.state = CircuitBreakerState.OPEN
            self.success_count = 0


class EventQueue:
    """Priority queue for event processing."""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self._queues: Dict[EventPriority, asyncio.Queue] = {
            priority: asyncio.Queue(maxsize=max_size // 5)
            for priority in EventPriority
        }
        self._total_size = 0
        self._lock = asyncio.Lock()
        self._priority_order = [
            EventPriority.EMERGENCY,
            EventPriority.CRITICAL,
            EventPriority.HIGH,
            EventPriority.NORMAL,
            EventPriority.LOW
        ]
    
    async def put(self, envelope: EventEnvelope) -> bool:
        """Add event envelope to queue."""
        async with self._lock:
            if self._total_size >= self.max_size:
                return False
            
            priority = envelope.event.metadata.priority
            queue = self._queues[priority]
            
            try:
                queue.put_nowait(envelope)
                self._total_size += 1
                return True
            except asyncio.QueueFull:
                return False
    
    async def get(self) -> Optional[EventEnvelope]:
        """Get next event envelope from queue (priority order)."""
        for priority in self._priority_order:
            queue = self._queues[priority]
            if not queue.empty():
                try:
                    envelope = queue.get_nowait()
                    async with self._lock:
                        self._total_size -= 1
                    return envelope
                except asyncio.QueueEmpty:
                    continue
        return None
    
    async def size(self) -> int:
        """Get total queue size."""
        return self._total_size
    
    async def sizes_by_priority(self) -> Dict[EventPriority, int]:
        """Get queue sizes by priority."""
        return {
            priority: queue.qsize()
            for priority, queue in self._queues.items()
        }


class EventStore:
    """Persistent storage for events."""
    
    def __init__(self, storage_path: Path, max_events: int = 100000):
        self.storage_path = storage_path
        self.max_events = max_events
        self._events: deque = deque(maxlen=max_events)
        self._index: Dict[str, EventEnvelope] = {}
        self._lock = threading.Lock()
        
        # Ensure storage directory exists
        self.storage_path.mkdir(parents=True, exist_ok=True)
    
    def store_event(self, envelope: EventEnvelope) -> None:
        """Store event envelope."""
        with self._lock:
            # Remove old event if index is full
            if len(self._index) >= self.max_events and envelope.event_id not in self._index:
                if self._events:
                    old_envelope = self._events[0]
                    self._index.pop(old_envelope.event_id, None)
            
            self._events.append(envelope)
            self._index[envelope.event_id] = envelope
    
    def get_event(self, event_id: str) -> Optional[EventEnvelope]:
        """Get event by ID."""
        with self._lock:
            return self._index.get(event_id)
    
    def get_events_by_criteria(
        self,
        event_type: Optional[str] = None,
        state: Optional[EventState] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[EventEnvelope]:
        """Get events by criteria."""
        with self._lock:
            results = []
            
            for envelope in reversed(self._events):
                if len(results) >= limit:
                    break
                
                # Apply filters
                if event_type and envelope.event.event_type != event_type:
                    continue
                
                if state and envelope.state != state:
                    continue
                
                if start_time and envelope.created_at < start_time:
                    continue
                
                if end_time and envelope.created_at > end_time:
                    continue
                
                results.append(envelope)
            
            return results
    
    async def persist_to_disk(self) -> None:
        """Persist events to disk."""
        try:
            events_file = self.storage_path / "events.jsonl"
            temp_file = self.storage_path / "events.tmp"
            
            with open(temp_file, 'w') as f:
                with self._lock:
                    for envelope in self._events:
                        # Serialize envelope
                        data = {
                            'event_id': envelope.event_id,
                            'event_type': envelope.event.event_type,
                            'event_data': asdict(envelope.event),
                            'subscription_id': envelope.subscription_id,
                            'state': envelope.state.value,
                            'created_at': envelope.created_at.isoformat(),
                            'metadata': envelope.headers
                        }
                        f.write(json.dumps(data) + '\n')
            
            # Atomic rename
            temp_file.rename(events_file)
            
        except Exception as e:
            logging.error(f"Failed to persist events to disk: {str(e)}")
    
    async def load_from_disk(self) -> None:
        """Load events from disk."""
        try:
            events_file = self.storage_path / "events.jsonl"
            if not events_file.exists():
                return
            
            with open(events_file, 'r') as f:
                with self._lock:
                    self._events.clear()
                    self._index.clear()
                    
                    for line in f:
                        data = json.loads(line.strip())
                        
                        # Reconstruct envelope (simplified)
                        envelope = EventEnvelope(
                            event_id=data['event_id'],
                            event=self._reconstruct_event(data),
                            subscription_id=data['subscription_id'],
                            state=EventState(data['state']),
                            created_at=datetime.fromisoformat(data['created_at']),
                            headers=data.get('metadata', {})
                        )
                        
                        self._events.append(envelope)
                        self._index[envelope.event_id] = envelope
            
        except Exception as e:
            logging.error(f"Failed to load events from disk: {str(e)}")
    
    def _reconstruct_event(self, data: Dict[str, Any]) -> BaseEvent:
        """Reconstruct event from serialized data."""
        # This would need more sophisticated event reconstruction
        # For now, create a basic event
        event_data = data['event_data']
        event = BaseEvent(
            event_type=data['event_type'],
            timestamp=datetime.fromisoformat(event_data['timestamp']),
            source=event_data['source'],
            data=event_data['data']
        )
        return event


class RateLimiter:
    """Rate limiter for event processing."""
    
    def __init__(self, max_events: int, time_window: float):
        self.max_events = max_events
        self.time_window = time_window
        self._events: deque = deque()
        self._lock = threading.Lock()
    
    def is_allowed(self) -> bool:
        """Check if event is allowed under rate limit."""
        current_time = time.time()
        
        with self._lock:
            # Remove old events outside time window
            while self._events and current_time - self._events[0] > self.time_window:
                self._events.popleft()
            
            # Check if under limit
            if len(self._events) < self.max_events:
                self._events.append(current_time)
                return True
            
            return False
    
    def get_remaining_capacity(self) -> int:
        """Get remaining capacity in current window."""
        current_time = time.time()
        
        with self._lock:
            # Remove old events
            while self._events and current_time - self._events[0] > self.time_window:
                self._events.popleft()
            
            return max(0, self.max_events - len(self._events))


class EventBusError(Exception):
    """Custom exception for event bus operations."""
    
    def __init__(self, message: str, event_id: Optional[str] = None, 
                 subscription_id: Optional[str] = None):
        super().__init__(message)
        self.event_id = event_id
        self.subscription_id = subscription_id
        self.timestamp = datetime.now(timezone.utc)


class EnhancedEventBus:
    """
    Advanced Event Bus System for the AI Assistant.
    
    This event bus provides comprehensive event-driven communication including:
    - Type-safe event handling with priority support
    - Flexible subscription patterns (exact, wildcard, conditional)
    - High-performance asynchronous processing
    - Event persistence and replay capabilities
    - Circuit breaker pattern for reliability
    - Rate limiting and backpressure handling
    - Dead letter queue for failed events
    - Comprehensive monitoring and observability
    - Event batching and bulk operations
    - Request-reply patterns
    - Event filtering and routing
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the enhanced event bus.
        
        Args:
            config: Optional configuration dictionary
        """
        self.logger = get_logger(__name__)
        
        # Core services - will be set later to avoid circular dependencies
        self.config = config or {}
        self.error_handler = None
        self.health_check = None
        self.metrics = None
        self.tracer = None
        
        # Configuration
        self._load_configuration()
        
        # Core components
        self._setup_core_components()
        
        # Setup monitoring (will be minimal without metrics)
        self._setup_monitoring()
        
        # State management
        self.subscriptions: Dict[str, EventSubscription] = {}
        self.event_handlers: Dict[str, Set[str]] = defaultdict(set)  # event_type -> subscription_ids
        self.wildcard_handlers: Dict[str, Set[str]] = defaultdict(set)  # pattern -> subscription_ids
        self.category_handlers: Dict[EventCategory, Set[str]] = defaultdict(set)
        self.conditional_handlers: List[str] = []  # subscription_ids with conditions
        self.broadcast_handlers: Set[str] = set()
        
        # Processing infrastructure
        self.event_queue = EventQueue(self.max_queue_size)
        self.dead_letter_queue = EventQueue(self.max_dead_letter_size)
        self.event_store = EventStore(
            Path(self.storage_path),
            self.max_stored_events
        ) if self.enable_persistence else None
        
        # Reliability components
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.rate_limiters: Dict[str, RateLimiter] = {}
        
        # Processing workers
        self.worker_tasks: List[asyncio.Task] = []
        self.is_running = False
        self.shutdown_event = asyncio.Event()
        
        # Performance tracking
        self.event_stats = {
            'total_emitted': 0,
            'total_delivered': 0,
            'total_failed': 0,
            'total_retries': 0,
            'average_processing_time': 0.0
        }
        self.processing_times: deque = deque(maxlen=1000)
        
        # Thread pool for synchronous handlers
        self.thread_pool = ThreadPoolExecutor(
            max_workers=self.max_sync_workers,
            thread_name_prefix="event_bus"
        )
        
        # Request-reply support
        self.pending_replies: Dict[str, asyncio.Future] = {}
        self.reply_timeout = 30.0
        
        self.logger.info("EnhancedEventBus initialized successfully")

    def set_dependencies(self, error_handler=None, health_check=None, metrics=None, tracer=None):
        """Set dependencies after initialization to avoid circular imports."""
        if error_handler:
            self.error_handler = error_handler
        if health_check:
            self.health_check = health_check
            # Register health check
            self.health_check.register_component("event_bus", self._health_check_callback)
        if metrics:
            self.metrics = metrics
            # Re-setup monitoring now that metrics are available
            self._setup_monitoring()
        if tracer:
            self.tracer = tracer

    def _load_configuration(self) -> None:
        """Load configuration settings."""
        # Since config is now a dict, access values directly
        self.max_queue_size = self.config.get("max_queue_size", 10000)
        self.max_dead_letter_size = self.config.get("max_dead_letter_size", 1000)
        self.max_workers = self.config.get("max_workers", 10)
        self.max_sync_workers = self.config.get("max_sync_workers", 5)
        self.enable_persistence = self.config.get("enable_persistence", True)
        self.storage_path = self.config.get("storage_path", "data/events")
        self.max_stored_events = self.config.get("max_stored_events", 100000)
        self.enable_rate_limiting = self.config.get("enable_rate_limiting", True)
        self.default_rate_limit = self.config.get("default_rate_limit", 1000)
        self.rate_limit_window = self.config.get("rate_limit_window", 60)
        self.enable_circuit_breaker = self.config.get("enable_circuit_breaker", True)
        self.circuit_breaker_threshold = self.config.get("circuit_breaker_threshold", 5)

    def _setup_core_components(self) -> None:
        """Setup core event bus components."""
        pass  # Components already initialized in __init__

    def _setup_monitoring(self) -> None:
        """Setup monitoring and metrics."""
        if not self.metrics:
            return
        
        try:
            # Register event bus metrics
            self.metrics.register_counter("event_bus_events_emitted_total")
            self.metrics.register_counter("event_bus_events_delivered_total")
            self.metrics.register_counter("event_bus_events_failed_total")
            self.metrics.register_counter("event_bus_events_retried_total")
            self.metrics.register_histogram("event_bus_processing_duration_seconds")
            self.metrics.register_histogram("event_bus_queue_time_seconds")
            self.metrics.register_gauge("event_bus_queue_size")
            self.metrics.register_gauge("event_bus_active_subscriptions")
            self.metrics.register_gauge("event_bus_circuit_breakers_open")
            self.metrics.register_counter("event_bus_rate_limited_events_total")
            
        except Exception as e:
            self.logger.warning(f"Failed to setup event bus monitoring: {str(e)}")

    async def initialize(self) -> None:
        """Initialize the event bus."""
        try:
            # Load persisted events
            if self.event_store:
                await self.event_store.load_from_disk()
            
            # Start processing workers
            await self.start_workers()
            
            # Start background tasks
            asyncio.create_task(self._metrics_update_loop())
            asyncio.create_task(self._cleanup_loop())
            
            if self.enable_persistence:
                asyncio.create_task(self._persistence_loop())
            
            self.logger.info("Event bus initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize event bus: {str(e)}")
            raise EventBusError(f"Event bus initialization failed: {str(e)}")

    async def start_workers(self) -> None:
        """Start event processing workers."""
        if self.is_running:
            return
        
        self.is_running = True
        self.shutdown_event.clear()
        
        # Start worker tasks
        for i in range(self.max_workers):
            task = asyncio.create_task(self._worker_loop(f"worker_{i}"))
            self.worker_tasks.append(task)
        
        # Start dead letter processor
        dead_letter_task = asyncio.create_task(self._dead_letter_processor())
        self.worker_tasks.append(dead_letter_task)
        
        self.logger.info(f"Started {len(self.worker_tasks)} event processing workers")

    async def stop_workers(self) -> None:
        """Stop event processing workers."""
        if not self.is_running:
            return
        
        self.is_running = False
        self.shutdown_event.set()
        
        # Cancel all worker tasks
        for task in self.worker_tasks:
            task.cancel()
        
        # Wait for workers to finish
        if self.worker_tasks:
            await asyncio.gather(*self.worker_tasks, return_exceptions=True)
        
        self.worker_tasks.clear()
        self.logger.info("Stopped all event processing workers")

    async def emit(
        self,
        event: BaseEvent,
        wait_for_delivery: bool = False,
        timeout: Optional[float] = None
    ) -> Optional[str]:
        """
        Emit an event to the event bus.
        
        Args:
            event: Event to emit
            wait_for_delivery: Whether to wait for event delivery
            timeout: Timeout for waiting (if wait_for_delivery is True)
            
        Returns:
            Event ID or correlation ID for tracking
        """
        if not self.is_running:
            raise EventBusError("Event bus is not running")
        
        # Generate event ID if not present
        if not hasattr(event, 'event_id') or not event.event_id:
            event.event_id = str(uuid.uuid4())
        
        # Update event metadata
        if not event.metadata:
            event.metadata = EventMetadata()
        
        event.metadata.emitted_at = datetime.now(timezone.utc)
        
        # Get matching subscriptions
        matching_subscriptions = self._get_matching_subscriptions(event)
        
        if not matching_subscriptions:
            self.logger.debug(f"No subscribers for event {event.event_type}")
            return event.event_id
        
        # Create event envelopes for each subscription
        envelopes = []
        for subscription_id in matching_subscriptions:
            subscription = self.subscriptions[subscription_id]
            
            # Check rate limiting
            if self._is_rate_limited(subscription_id):
                if self.metrics:
                    self.metrics.increment("event_bus_rate_limited_events_total")
                continue
            
            # Create envelope
            envelope = EventEnvelope(
                event_id=f"{event.event_id}_{subscription_id}",
                event=event,
                subscription_id=subscription_id,
                correlation_id=event.event_id
            )
            
            # Set delivery schedule
            if subscription.delivery_mode == EventDeliveryMode.BATCH:
                envelope.scheduled_at = datetime.now(timezone.utc) + timedelta(seconds=subscription.batch_timeout)
            
            envelopes.append(envelope)
        
        # Queue events for processing
        queued_count = 0
        for envelope in envelopes:
            if await self.event_queue.put(envelope):
                queued_count += 1
                
                # Store in event store
                if self.event_store:
                    self.event_store.store_event(envelope)
            else:
                self.logger.warning(f"Event queue full, dropping envelope {envelope.event_id}")
        
        # Update metrics
        if self.metrics:
            self.metrics.increment("event_bus_events_emitted_total")
        
        self.event_stats['total_emitted'] += 1
        
        self.logger.debug(f"Emitted event {event.event_id} to {queued_count} subscribers")
        
        # Wait for delivery if requested
        if wait_for_delivery and envelopes:
            return await self._wait_for_delivery(envelopes, timeout)
        
        return event.event_id

    def _get_matching_subscriptions(self, event: BaseEvent) -> Set[str]:
        """Get subscriptions that match the event."""
        matching = set()
        
        # Exact type matches
        matching.update(self.event_handlers.get(event.event_type, set()))
        
        # Category matches
        if hasattr(event, 'category'):
            matching.update(self.category_handlers.get(event.category, set()))
        
        # Wildcard matches
        for pattern, subscription_ids in self.wildcard_handlers.items():
            if self._matches_pattern(event.event_type, pattern):
                matching.update(subscription_ids)
        
        # Conditional matches
        for subscription_id in self.conditional_handlers:
            subscription = self.subscriptions[subscription_id]
            if subscription.filter_func and subscription.filter_func(event):
                matching.add(subscription_id)
        
        # Broadcast subscriptions
        matching.update(self.broadcast_handlers)
        
        # Filter by subscription criteria
        filtered_matching = set()
        for subscription_id in matching:
            subscription = self.subscriptions[subscription_id]
            
            # Check priority threshold
            if event.metadata.priority.value < subscription.priority_threshold.value:
                continue
            
            # Check tags filter
            if subscription.tags_filter and not subscription.tags_filter.intersection(event.metadata.tags):
                continue
            
            # Check source filter
            if subscription.source_filter and event.source not in subscription.source_filter:
                continue
            
            filtered_matching.add(subscription_id)
        
        return filtered_matching

    def _matches_pattern(self, event_type: str, pattern: str) -> bool:
        """Check if event type matches wildcard pattern."""
        import fnmatch
        return fnmatch.fnmatch(event_type, pattern)

    def _is_rate_limited(self, subscription_id: str) -> bool:
        """Check if subscription is rate limited."""
        if not self.enable_rate_limiting:
            return False
        
        if subscription_id not in self.rate_limiters:
            subscription = self.subscriptions[subscription_id]
            max_events = getattr(subscription, 'rate_limit', self.default_rate_limit)
            self.rate_limiters[subscription_id] = RateLimiter(max_events, self.rate_limit_window)
        
        return not self.rate_limiters[subscription_id].is_allowed()

    async def _wait_for_delivery(
        self,
        envelopes: List[EventEnvelope],
        timeout: Optional[float]
    ) -> str:
        """Wait for event delivery completion."""
        correlation_id = str(uuid.uuid4())
        
        # Create future for tracking delivery
        future = asyncio.Future()
        self.pending_replies[correlation_id] = future
        
        # Set correlation ID on envelopes
        for envelope in envelopes:
            envelope.correlation_id = correlation_id
        
        try:
            # Wait for completion
            await asyncio.wait_for(future, timeout=timeout or self.reply_timeout)
            return correlation_id
        except asyncio.TimeoutError:
            raise EventBusError(f"Event delivery timed out after {timeout or self.reply_timeout}s")
        finally:
            self.pending_replies.pop(correlation_id, None)

    def subscribe(
        self,
        event_types: Union[str, List[str], Set[str]],
        handler: EventHandler,
        subscription_type: SubscriptionType = SubscriptionType.EXACT,
        **kwargs
    ) -> str:
        """
        Subscribe to events.
        
        Args:
            event_types: Event type(s) to subscribe to
            handler: Event handler function
            subscription_type: Type of subscription
            **kwargs: Additional subscription options
            
        Returns:
            Subscription ID
        """
        # Normalize event types
        if isinstance(event_types, str):
            event_types = {event_types}
        elif isinstance(event_types, list):
            event_types = set(event_types)
        
        # Create subscription
        subscription_id = str(uuid.uuid4())
        subscription = EventSubscription(
            subscription_id=subscription_id,
            event_types=event_types,
            handler=handler,
            subscription_type=subscription_type,
            **kwargs
        )
        
        # Store subscription
        self.subscriptions[subscription_id] = subscription
        
        # Register in appropriate indexes
        if subscription_type == SubscriptionType.EXACT:
            for event_type in event_types:
                self.event_handlers[event_type].add(subscription_id)
        elif subscription_type == SubscriptionType.WILDCARD:
            for pattern in event_types:
                self.wildcard_handlers[pattern].add(subscription_id)
        elif subscription_type == SubscriptionType.CATEGORY:
            for category_name in event_types:
                try:
                    category = EventCategory(category_name)
                    self.category_handlers[category].add(subscription_id)
                except ValueError:
                    self.logger.warning(f"Invalid event category: {category_name}")
        elif subscription_type == SubscriptionType.CONDITIONAL:
            self.conditional_handlers.append(subscription_id)
        elif subscription_type == SubscriptionType.BROADCAST:
            self.broadcast_handlers.add(subscription_id)
        
        # Setup circuit breaker if enabled
        if self.enable_circuit_breaker:
            self.circuit_breakers[subscription_id] = CircuitBreaker(
                handler_id=subscription_id,
                failure_threshold=self.circuit_breaker_threshold
            )
        
        # Update metrics
        if self.metrics:
            self.metrics.set("event_bus_active_subscriptions", len(self.subscriptions))
        
        self.logger.info(
            f"Created subscription {subscription_id} for {event_types} "
            f"(type: {subscription_type.value})"
        )
        
        return subscription_id

    def unsubscribe(self, subscription_id: str) -> bool:
        """
        Unsubscribe from events.
        
        Args:
            subscription_id: Subscription ID to remove
            
        Returns:
            True if subscription was removed, False if not found
        """
        if subscription_id not in self.subscriptions:
            return False
        
        subscription = self.subscriptions[subscription_id]
        
        # Remove from indexes
        if subscription.subscription_type == SubscriptionType.EXACT:
            for event_type in subscription.event_types:
                self.event_handlers[event_type].discard(subscription_id)
        elif subscription.subscription_type == SubscriptionType.WILDCARD:
            for pattern in subscription.event_types:
                self.wildcard_handlers[pattern].discard(subscription_id)
        elif subscription.subscription_type == SubscriptionType.CATEGORY:
            for category_name in subscription.event_types:
                try:
                    category = EventCategory(category_name)
                    self.category_handlers[category].discard(subscription_id)
                except ValueError:
                    pass
        elif subscription.subscription_type == SubscriptionType.CONDITIONAL:
            if subscription_id in self.conditional_handlers:
                self.conditional_handlers.remove(subscription_id)
        elif subscription.subscription_type == SubscriptionType.BROADCAST:
            self.broadcast_handlers.discard(subscription_id)
        
        # Remove subscription
        del self.subscriptions[subscription_id]
        
        # Remove circuit breaker
        self.circuit_breakers.pop(subscription_id, None)
        
        # Remove rate limiter
        self.rate_limiters.pop(subscription_id, None)
        
        # Update metrics
        if self.metrics:
            self.metrics.set("event_bus_active_subscriptions", len(self.subscriptions))
        
        self.logger.info(f"Removed subscription {subscription_id}")
        return True

    async def _worker_loop(self, worker_name: str) -> None:
        """Main worker loop for processing events."""
        self.logger.info(f"Started event processing worker: {worker_name}")
        
        while self.is_running:
            try:
                # Get next event envelope
                envelope = await self.event_queue.get()
                if not envelope:
                    await asyncio.sleep(0.1)
                    continue
                
                # Process the event
                await self._process_event_envelope(envelope)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in worker {worker_name}: {str(e)}")
                await asyncio.sleep(1)
        
        self.logger.info(f"Stopped event processing worker: {worker_name}")

    async def _process_event_envelope(self, envelope: EventEnvelope) -> None:
        """Process a single event envelope."""
        start_time = time.time()
        envelope.started_at = datetime.now(timezone.utc)
        envelope.state = EventState.PROCESSING
        envelope.queue_time = (envelope.started_at - envelope.created_at).total_seconds()
        
        subscription_id = envelope.subscription_id
        subscription = self.subscriptions.get(subscription_id)
        
        if not subscription:
            self.logger.warning(f"Subscription {subscription_id} not found for envelope {envelope.event_id}")
            return
        
        # Check circuit breaker
        circuit_breaker = self.circuit_breakers.get(subscription_id)
        if circuit_breaker and not circuit_breaker.should_allow_request():
            self.logger.warning(f"Circuit breaker open for subscription {subscription_id}")
            envelope.state = EventState.FAILED
            envelope.last_error = "Circuit breaker is open"
            await self._handle_failed_event(envelope)
            return
        
        try:
            # Update subscription stats
            subscription.last_event_at = datetime.now(timezone.utc)
            subscription.total_events_received += 1
            
            # Trace event processing
            trace_context = {}
            if self.tracer:
                trace_context = self.tracer.trace("event_processing")
                trace_context.__enter__()
                trace_context.set_attributes({
                    "event_type": envelope.event.event_type,
                    "event_id": envelope.event_id,
                    "subscription_id": subscription_id,
                    "attempt": envelope.attempt_count + 1
                })
            
            # Execute handler
            await self._execute_handler(subscription, envelope)
            
            # Record success
            envelope.state = EventState.COMPLETED
            envelope.completed_at = datetime.now(timezone.utc)
            envelope.processing_time = time.time() - start_time
            
            subscription.successful_events += 1
            
            # Record circuit breaker success
            if circuit_breaker:
                circuit_breaker.record_success()
            
            # Update metrics
            if self.metrics:
                self.metrics.increment("event_bus_events_delivered_total")
                self.metrics.record("event_bus_processing_duration_seconds", envelope.processing_time)
                self.metrics.record("event_bus_queue_time_seconds", envelope.queue_time)
            
            self.event_stats['total_delivered'] += 1
            self.processing_times.append(envelope.processing_time)
            
            # Check for reply waiting
            if envelope.correlation_id and envelope.correlation_id in self.pending_replies:
                future = self.pending_replies[envelope.correlation_id]
                if not future.done():
                    future.set_result(envelope.event_id)
            
            self.logger.debug(f"Successfully processed envelope {envelope.event_id}")
            
        except Exception as e:
            # Record failure
            envelope.state = EventState.FAILED
            envelope.last_error = str(e)
            envelope.attempt_count += 1
            envelope.processing_time = time.time() - start_time
            
            subscription.failed_events += 1
            
            # Record circuit breaker failure
            if circuit_breaker:
                circuit_breaker.record_failure()
            
            # Handle retry logic
            await self._handle_failed_event(envelope)
            
            self.logger.error(f"Failed to process envelope {envelope.event_id}: {str(e)}")
            
        finally:
            if trace_context:
                trace_context.__exit__(None, None, None)

    async def _execute_handler(self, subscription: EventSubscription, envelope: EventEnvelope) -> Any:
        """Execute event handler with appropriate execution mode."""
        handler = subscription.handler
        event = envelope.event
        
        # Determine execution mode
        if subscription.delivery_mode == EventDeliveryMode.SYNC:
            # Synchronous execution in thread pool
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(self.thread_pool, handler, event)
        
        elif subscription.delivery_mode == EventDeliveryMode.ASYNC:
            # Asynchronous execution
            if asyncio.iscoroutinefunction(handler):
                return await asyncio.wait_for(handler(event), timeout=subscription.timeout_seconds)
            else:
                # Run sync handler in thread pool
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(self.thread_pool, handler, event)
        
        elif subscription.delivery_mode == EventDeliveryMode.FIRE_AND_FORGET:
            # Fire and forget - don't wait for completion
            if asyncio.iscoroutinefunction(handler):
                asyncio.create_task(handler(event))
            else:
                self.thread_pool.submit(handler, event)
            return None
        
        elif subscription.delivery_mode == EventDeliveryMode.REQUEST_REPLY:
            # Request-reply pattern
            if asyncio.iscoroutinefunction(handler):
                result = await asyncio.wait_for(handler(event), timeout=subscription.timeout_seconds)
            else:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(self.thread_pool, handler, event)
            
            # Send reply if requested
            if envelope.reply_to:
                reply_event = BaseEvent(
                    event_type=f"{event.event_type}_reply",
                    source="event_bus",
                    data={"original_event_id": envelope.event_id, "result": result}
                )
                await self.emit(reply_event)
            
            return result
        
        else:
            # Default to async
            if asyncio.iscoroutinefunction(handler):
                return await handler(event)
            else:
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(self.thread_pool, handler, event)

    async def _handle_failed_event(self, envelope: EventEnvelope) -> None:
        """Handle failed event processing."""
        subscription = self.subscriptions.get(envelope.subscription_id)
        if not subscription:
            return
        
        # Record delivery attempt
        envelope.delivery_attempts.append({
            "attempt": envelope.attempt_count,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error": envelope.last_error
        })
        
        # Check retry logic
        if envelope.attempt_count < subscription.max_retries:
            # Schedule retry
            envelope.state = EventState.RETRYING
            
            # Calculate retry delay with exponential backoff
            delay = subscription.retry_delay * (2 ** (envelope.attempt_count - 1))
            envelope.scheduled_at = datetime.now(timezone.utc) + timedelta(seconds=delay)
            
            # Re-queue event
            if await self.event_queue.put(envelope):
                if self.metrics:
                    self.metrics.increment("event_bus_events_retried_total")
                self.event_stats['total_retries'] += 1
                self.logger.debug(f"Scheduled retry for envelope {envelope.event_id} in {delay}s")
            else:
                # Queue full, send to dead letter
                await self._send_to_dead_letter(envelope)
        else:
            # Max retries exceeded
            if subscription.enable_dead_letter:
                await self._send_to_dead_letter(envelope)
            else:
                envelope.state = EventState.FAILED
                if self.metrics:
                    self.metrics.increment("event_bus_events_failed_total")
                self.event_stats['total_failed'] += 1

    async def _send_to_dead_letter(self, envelope: EventEnvelope) -> None:
        """Send failed event to dead letter queue."""
        envelope.state = EventState.DEAD_LETTER
        
        if await self.dead_letter_queue.put(envelope):
            self.logger.warning(f"Sent envelope {envelope.event_id} to dead letter queue")
        else:
            self.logger.error(f"Dead letter queue full, dropping envelope {envelope.event_id}")
        
        if self.metrics:
            self.metrics.increment("event_bus_events_failed_total")
        
        self.event_stats['total_failed'] += 1

    async def _dead_letter_processor(self) -> None:
        """Process events in dead letter queue."""
        self.logger.info("Started dead letter processor")
        
        while self.is_running:
            try:
                envelope = await self.dead_letter_queue.get()
                if not envelope:
                    await asyncio.sleep(1)
                    continue
                
                # Log dead letter event
                self.logger.error(
                    f"Dead letter event: {envelope.event_id} "
                    f"(type: {envelope.event.event_type}, "
                    f"subscription: {envelope.subscription_id}, "
                    f"attempts: {envelope.attempt_count})"
                )
                
                # Store in event store for analysis
                if self.event_store:
                    self.event_store.store_event(envelope)
                
                # Could implement additional dead letter handling here
                # (e.g., notifications, analysis, manual reprocessing)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in dead letter processor: {str(e)}")
                await asyncio.sleep(1)
        
        self.logger.info("Stopped dead letter processor")

    async def _metrics_update_loop(self) -> None:
        """Background task to update metrics."""
        while self.is_running:
            try:
                if self.metrics:
                    # Update queue size metrics
                    queue_size = await self.event_queue.size()
                    self.metrics.set("event_bus_queue_size", queue_size)
                    
                    # Update circuit breaker metrics
                    open_breakers = sum(
                        1 for cb in self.circuit_breakers.values()
                        if cb.state == CircuitBreakerState.OPEN
                    )
                    self.metrics.set("event_bus_circuit_breakers_open", open_breakers)
                    
                    # Update average processing time
                    if self.processing_times:
                        avg_time = sum(self.processing_times) / len(self.processing_times)
                        self.event_stats['average_processing_time'] = avg_time
                
                await asyncio.sleep(10)  # Update every 10 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in metrics update loop: {str(e)}")
                await asyncio.sleep(10)

    async def _cleanup_loop(self) -> None:
        """Background task for cleanup operations."""
        while self.is_running:
            try:
                current_time = datetime.now(timezone.utc)
                
                # Clean up expired events
                expired_replies = []
                for correlation_id, future in self.pending_replies.items():
                    if future.done():
                        expired_replies.append(correlation_id)
                
                for correlation_id in expired_replies:
                    self.pending_replies.pop(correlation_id, None)
                
                # Reset circuit breakers after timeout
                for circuit_breaker in self.circuit_breakers.values():
                    if (circuit_breaker.state == CircuitBreakerState.OPEN and
                        circuit_breaker.last_failure_time and
                        (current_time - circuit_breaker.last_failure_time).total_seconds() > circuit_breaker.timeout_duration):
                        circuit_breaker.state = CircuitBreakerState.HALF_OPEN
                
                await asyncio.sleep(60)  # Cleanup every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in cleanup loop: {str(e)}")
                await asyncio.sleep(60)

    async def _persistence_loop(self) -> None:
        """Background task for event persistence."""
        if not self.event_store:
            return
        
        while self.is_running:
            try:
                await self.event_store.persist_to_disk()
                await asyncio.sleep(300)  # Persist every 5 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in persistence loop: {str(e)}")
                await asyncio.sleep(300)

    def get_subscription_info(self, subscription_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a subscription."""
        subscription = self.subscriptions.get(subscription_id)
        if not subscription:
            return None
        
        return {
            "subscription_id": subscription_id,
            "event_types": list(subscription.event_types),
            "subscription_type": subscription.subscription_type.value,
            "delivery_mode": subscription.delivery_mode.value,
            "created_at": subscription.created_at.isoformat(),
            "last_event_at": subscription.last_event_at.isoformat() if subscription.last_event_at else None,
            "total_events_received": subscription.total_events_received,
            "successful_events": subscription.successful_events,
            "failed_events": subscription.failed_events,
            "subscriber_name": subscription.subscriber_name,
            "description": subscription.description
        }

    def list_subscriptions(self) -> List[Dict[str, Any]]:
        """List all active subscriptions."""
        return [
            self.get_subscription_info(sub_id)
            for sub_id in self.subscriptions.keys()
        ]

    async def get_event_history(
        self,
        event_type: Optional[str] = None,
        state: Optional[EventState] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get event history from the event store."""
        if not self.event_store:
            return []
        
        envelopes = self.event_store.get_events_by_criteria(
            event_type=event_type,
            state=state,
            start_time=start_time,
            end_time=end_time,
            limit=limit
        )
        
        return [
            {
                "event_id": envelope.event_id,
                "event_type": envelope.event.event_type,
                "subscription_id": envelope.subscription_id,
                "state": envelope.state.value,
                "created_at": envelope.created_at.isoformat(),
                "started_at": envelope.started_at.isoformat() if envelope.started_at else None,
                "completed_at": envelope.completed_at.isoformat() if envelope.completed_at else None,
                "attempt_count": envelope.attempt_count,
                "processing_time": envelope.processing_time,
                "queue_time": envelope.queue_time,
                "last_error": envelope.last_error
            }
            for envelope in envelopes
        ]

    def get_statistics(self) -> Dict[str, Any]:
        """Get event bus statistics."""
        circuit_breaker_stats = {
            "total": len(self.circuit_breakers),
            "closed": sum(1 for cb in self.circuit_breakers.values() if cb.state == CircuitBreakerState.CLOSED),
            "open": sum(1 for cb in self.circuit_breakers.values() if cb.state == CircuitBreakerState.OPEN),
            "half_open": sum(1 for cb in self.circuit_breakers.values() if cb.state == CircuitBreakerState.HALF_OPEN)
        }
        
        return {
            "subscriptions": {
                "total": len(self.subscriptions),
                "by_type": {
                    sub_type.value: sum(1 for s in self.subscriptions.values() if s.subscription_type == sub_type)
                    for sub_type in SubscriptionType
                }
            },
            "events": self.event_stats,
            "queue": {
                "size": asyncio.create_task(self.event_queue.size()) if self.is_running else 0,
                "dead_letter_size": asyncio.create_task(self.dead_letter_queue.size()) if self.is_running else 0
            },
            "circuit_breakers": circuit_breaker_stats,
            "rate_limiters": len(self.rate_limiters),
            "workers": {
                "active": len(self.worker_tasks),
                "running": self.is_running
            },
            "persistence": {
                "enabled": self.enable_persistence,
                "stored_events": len(self.event_store._events) if self.event_store else 0
            }
        }

    async def _health_check_callback(self) -> Dict[str, Any]:
        """Health check callback for the event bus."""
        try:
            queue_size = await self.event_queue.size()
            dead_letter_size = await self.dead_letter_queue.size()
            
            # Calculate health score
            health_factors = {
                "workers_running": 1.0 if self.is_running else 0.0,
                "queue_not_full": max(0.0, 1.0 - (queue_size / self.max_queue_size)),
                "low_error_rate": max(0.0, 1.0 - (self.event_stats['total_failed'] / max(1, self.event_stats['total_emitted']))),
                "circuit_breakers_healthy": 1.0 - (sum(1 for cb in self.circuit_breakers.values() if cb.state == CircuitBreakerState.OPEN) / max(1, len(self.circuit_breakers)))
            }
            
            health_score = sum(health_factors.values()) / len(health_factors)
            
            return {
                "status": "healthy" if health_score >= 0.8 else "degraded" if health_score >= 0.5 else "unhealthy",
                "health_score": health_score,
                "workers_running": self.is_running,
                "active_workers": len(self.worker_tasks),
                "queue_size": queue_size,
                "dead_letter_size": dead_letter_size,
                "active_subscriptions": len(self.subscriptions),
                "total_events_emitted": self.event_stats['total_emitted'],
                "total_events_delivered": self.event_stats['total_delivered'],
                "total_events_failed": self.event_stats['total_failed'],
                "average_processing_time": self.event_stats['average_processing_time']
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }

    async def shutdown(self) -> None:
        """Gracefully shutdown the event bus."""
        self.logger.info("Starting event bus shutdown...")
        
        try:
            # Stop accepting new events
            await self.stop_workers()
            
            # Process remaining events with timeout
            remaining_events = await self.event_queue.size()
            if remaining_events > 0:
                self.logger.info(f"Processing {remaining_events} remaining events...")
                
                # Give some time to process remaining events
                timeout = min(30.0, remaining_events * 0.1)  # Max 30 seconds
                try:
                    await asyncio.wait_for(
                        self._drain_queue(),
                        timeout=timeout
                    )
                except asyncio.TimeoutError:
                    self.logger.warning("Timeout waiting for remaining events to process")
            
            # Persist final state
            if self.event_store:
                await self.event_store.persist_to_disk()
            
            # Shutdown thread pool
            self.thread_pool.shutdown(wait=True)
            
            self.logger.info("Event bus shutdown completed")
            
        except Exception as e:
            self.logger.error(f"Error during event bus shutdown: {str(e)}")
            raise EventBusError(f"Shutdown failed: {str(e)}")

    async def _drain_queue(self) -> None:
        """Drain remaining events from queue."""
        while True:
            queue_size = await self.event_queue.size()
            if queue_size == 0:
                break
            await asyncio.sleep(0.1)

    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            if hasattr(self, 'thread_pool'):
                self.thread_pool.shutdown(wait=False)
        except Exception:
            pass  # Ignore cleanup errors in destructor


# Backward compatibility alias
EventBus = EnhancedEventBus
