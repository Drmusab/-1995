"""
Advanced Rate Limiting Middleware for AI Assistant
Author: Drmusab
Last Modified: 2025-01-21 11:05:17 UTC

This module provides comprehensive rate limiting capabilities for the AI assistant,
including adaptive rate limiting, distributed rate limiting, user-specific limits,
integration with authentication and session management, and intelligent throttling
based on system load and user behavior.
"""

from pathlib import Path
from typing import Optional, Dict, Any, List, Set, Callable, Type, Union, AsyncGenerator, TypeVar
import asyncio
import threading
import time
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
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
import math
import statistics

# Core imports
from src.core.config.loader import ConfigLoader
from src.core.events.event_bus import EventBus
from src.core.events.event_types import (
    RateLimitTriggered, RateLimitReset, RateLimitConfigured, RateLimitBypass,
    UserBlocked, UserUnblocked, SystemOverload, LoadBalancingTriggered,
    ErrorOccurred, SystemStateChanged, ComponentHealthChanged,
    SessionStarted, SessionEnded, UserAuthenticated
)
from src.core.error_handling import ErrorHandler, handle_exceptions
from src.core.dependency_injection import Container
from src.core.health_check import HealthCheck
from src.core.security.authentication import AuthenticationManager
from src.core.security.authorization import AuthorizationManager

# Assistant components
from src.assistant.session_manager import SessionManager
from src.assistant.component_manager import ComponentManager

# Memory and cache
from src.memory.core_memory.memory_manager import MemoryManager
from src.integrations.cache.redis_cache import RedisCache
from src.integrations.storage.database import DatabaseManager

# Learning and adaptation
from src.learning.continual_learning import ContinualLearner
from src.learning.preference_learning import PreferenceLearner
from src.learning.feedback_processor import FeedbackProcessor

# Observability
from src.observability.monitoring.metrics import MetricsCollector
from src.observability.monitoring.tracing import TraceManager
from src.observability.logging.config import get_logger

# Type definitions
T = TypeVar('T')


class RateLimitType(Enum):
    """Types of rate limiting strategies."""
    TOKEN_BUCKET = "token_bucket"          # Token bucket algorithm
    SLIDING_WINDOW = "sliding_window"      # Sliding window counter
    FIXED_WINDOW = "fixed_window"          # Fixed window counter
    LEAKY_BUCKET = "leaky_bucket"          # Leaky bucket algorithm
    ADAPTIVE = "adaptive"                  # Adaptive rate limiting
    CIRCUIT_BREAKER = "circuit_breaker"    # Circuit breaker pattern


class LimitScope(Enum):
    """Scope of rate limiting."""
    GLOBAL = "global"                      # System-wide limits
    USER = "user"                          # Per-user limits
    SESSION = "session"                    # Per-session limits
    IP_ADDRESS = "ip_address"              # Per-IP limits
    API_KEY = "api_key"                    # Per-API key limits
    ENDPOINT = "endpoint"                  # Per-endpoint limits
    RESOURCE = "resource"                  # Per-resource limits


class ActionType(Enum):
    """Actions to take when rate limit is exceeded."""
    BLOCK = "block"                        # Block the request
    DELAY = "delay"                        # Delay the request
    QUEUE = "queue"                        # Queue the request
    REDUCE_QUALITY = "reduce_quality"      # Reduce response quality
    WARN = "warn"                          # Warn but allow
    CUSTOM = "custom"                      # Custom action


class PriorityLevel(Enum):
    """Priority levels for rate limiting."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3
    SYSTEM = 4


class UserTier(Enum):
    """User tiers with different limits."""
    FREE = "free"
    PREMIUM = "premium"
    ENTERPRISE = "enterprise"
    ADMIN = "admin"
    SYSTEM = "system"


@dataclass
class RateLimitRule:
    """Configuration for a rate limiting rule."""
    rule_id: str
    name: str
    description: Optional[str] = None
    
    # Limit configuration
    limit_type: RateLimitType = RateLimitType.TOKEN_BUCKET
    scope: LimitScope = LimitScope.USER
    requests_per_second: float = 10.0
    requests_per_minute: float = 600.0
    requests_per_hour: float = 36000.0
    requests_per_day: float = 864000.0
    burst_capacity: int = 20
    
    # Conditions
    endpoints: Set[str] = field(default_factory=set)
    methods: Set[str] = field(default_factory=set)
    user_tiers: Set[UserTier] = field(default_factory=set)
    ip_ranges: List[str] = field(default_factory=list)
    
    # Actions
    action: ActionType = ActionType.BLOCK
    delay_seconds: float = 1.0
    queue_size: int = 100
    retry_after_seconds: int = 60
    
    # Adaptive behavior
    adaptive_enabled: bool = False
    load_threshold: float = 0.8
    adaptation_factor: float = 0.5
    
    # Circuit breaker settings
    failure_threshold: int = 5
    recovery_timeout_seconds: int = 30
    
    # Metadata
    priority: PriorityLevel = PriorityLevel.NORMAL
    enabled: bool = True
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = None
    tags: Set[str] = field(default_factory=set)


@dataclass
class RateLimitState:
    """Current state of rate limiting for a specific scope."""
    identifier: str
    rule_id: str
    scope: LimitScope
    
    # Token bucket state
    tokens: float = 0.0
    last_refill: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Sliding window state
    request_times: deque = field(default_factory=lambda: deque(maxlen=1000))
    
    # Counters
    requests_count: int = 0
    rejected_count: int = 0
    delayed_count: int = 0
    
    # Circuit breaker state
    failure_count: int = 0
    last_failure: Optional[datetime] = None
    circuit_open: bool = False
    
    # Adaptive state
    current_limit: float = 0.0
    adaptation_history: deque = field(default_factory=lambda: deque(maxlen=100))
    
    # Performance metrics
    average_response_time: float = 0.0
    success_rate: float = 1.0
    
    # Timestamps
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_request: Optional[datetime] = None
    last_reset: Optional[datetime] = None


@dataclass
class RateLimitResult:
    """Result of rate limit check."""
    allowed: bool
    rule_id: str
    identifier: str
    action: ActionType
    
    # Limit information
    remaining_requests: int = 0
    reset_time: Optional[datetime] = None
    retry_after: Optional[int] = None
    
    # Performance impact
    delay_seconds: float = 0.0
    quality_reduction: float = 0.0
    
    # Context
    current_load: float = 0.0
    burst_used: bool = False
    circuit_state: str = "closed"
    
    # Metadata
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    reason: Optional[str] = None


class RateLimitError(Exception):
    """Custom exception for rate limiting operations."""
    
    def __init__(self, message: str, rate_limit_result: Optional[RateLimitResult] = None,
                 error_code: Optional[str] = None):
        super().__init__(message)
        self.rate_limit_result = rate_limit_result
        self.error_code = error_code
        self.timestamp = datetime.now(timezone.utc)


class RateLimitStrategy(ABC):
    """Abstract base class for rate limiting strategies."""
    
    @abstractmethod
    async def check_limit(self, rule: RateLimitRule, state: RateLimitState,
                         context: Dict[str, Any]) -> RateLimitResult:
        """Check if request should be allowed."""
        pass
    
    @abstractmethod
    async def update_state(self, state: RateLimitState, allowed: bool,
                          context: Dict[str, Any]) -> None:
        """Update rate limit state after request."""
        pass


class TokenBucketStrategy(RateLimitStrategy):
    """Token bucket rate limiting strategy."""
    
    def __init__(self, logger):
        self.logger = logger
    
    async def check_limit(self, rule: RateLimitRule, state: RateLimitState,
                         context: Dict[str, Any]) -> RateLimitResult:
        """Check token bucket limit."""
        current_time = datetime.now(timezone.utc)
        
        # Refill tokens
        time_passed = (current_time - state.last_refill).total_seconds()
        tokens_to_add = time_passed * rule.requests_per_second
        state.tokens = min(rule.burst_capacity, state.tokens + tokens_to_add)
        state.last_refill = current_time
        
        # Check if tokens available
        if state.tokens >= 1.0:
            state.tokens -= 1.0
            allowed = True
            remaining = int(state.tokens)
        else:
            allowed = False
            remaining = 0
        
        # Calculate reset time
        if not allowed:
            time_to_next_token = (1.0 - state.tokens) / rule.requests_per_second
            reset_time = current_time + timedelta(seconds=time_to_next_token)
        else:
            reset_time = None
        
        return RateLimitResult(
            allowed=allowed,
            rule_id=rule.rule_id,
            identifier=state.identifier,
            action=rule.action if not allowed else ActionType.WARN,
            remaining_requests=remaining,
            reset_time=reset_time,
            retry_after=int(time_to_next_token) if not allowed else None,
            burst_used=state.tokens < rule.requests_per_second
        )
    
    async def update_state(self, state: RateLimitState, allowed: bool,
                          context: Dict[str, Any]) -> None:
        """Update token bucket state."""
        current_time = datetime.now(timezone.utc)
        state.last_request = current_time
        state.requests_count += 1
        
        if not allowed:
            state.rejected_count += 1


class SlidingWindowStrategy(RateLimitStrategy):
    """Sliding window rate limiting strategy."""
    
    def __init__(self, logger):
        self.logger = logger
    
    async def check_limit(self, rule: RateLimitRule, state: RateLimitState,
                         context: Dict[str, Any]) -> RateLimitResult:
        """Check sliding window limit."""
        current_time = datetime.now(timezone.utc)
        window_start = current_time - timedelta(seconds=60)  # 1 minute window
        
        # Remove old requests
        while state.request_times and state.request_times[0] < window_start:
            state.request_times.popleft()
        
        # Check if under limit
        current_requests = len(state.request_times)
        allowed = current_requests < rule.requests_per_minute
        
        if allowed:
            state.request_times.append(current_time)
        
        remaining = max(0, int(rule.requests_per_minute) - current_requests)
        reset_time = window_start + timedelta(seconds=60)
        
        return RateLimitResult(
            allowed=allowed,
            rule_id=rule.rule_id,
            identifier=state.identifier,
            action=rule.action if not allowed else ActionType.WARN,
            remaining_requests=remaining,
            reset_time=reset_time,
            retry_after=rule.retry_after_seconds if not allowed else None
        )
    
    async def update_state(self, state: RateLimitState, allowed: bool,
                          context: Dict[str, Any]) -> None:
        """Update sliding window state."""
        current_time = datetime.now(timezone.utc)
        state.last_request = current_time
        state.requests_count += 1
        
        if not allowed:
            state.rejected_count += 1


class AdaptiveStrategy(RateLimitStrategy):
    """Adaptive rate limiting strategy that adjusts based on system load."""
    
    def __init__(self, logger, metrics_collector: Optional[MetricsCollector] = None):
        self.logger = logger
        self.metrics = metrics_collector
    
    async def check_limit(self, rule: RateLimitRule, state: RateLimitState,
                         context: Dict[str, Any]) -> RateLimitResult:
        """Check adaptive limit based on system load."""
        current_time = datetime.now(timezone.utc)
        
        # Get system load
        system_load = await self._get_system_load(context)
        
        # Adapt limit based on load
        if system_load > rule.load_threshold:
            adaptation_factor = 1.0 - ((system_load - rule.load_threshold) * rule.adaptation_factor)
            adapted_limit = rule.requests_per_second * max(0.1, adaptation_factor)
        else:
            adapted_limit = rule.requests_per_second
        
        state.current_limit = adapted_limit
        state.adaptation_history.append({
            'timestamp': current_time,
            'system_load': system_load,
            'adapted_limit': adapted_limit
        })
        
        # Use token bucket with adapted limit
        time_passed = (current_time - state.last_refill).total_seconds()
        tokens_to_add = time_passed * adapted_limit
        state.tokens = min(rule.burst_capacity, state.tokens + tokens_to_add)
        state.last_refill = current_time
        
        # Check availability
        if state.tokens >= 1.0:
            state.tokens -= 1.0
            allowed = True
            remaining = int(state.tokens)
        else:
            allowed = False
            remaining = 0
        
        return RateLimitResult(
            allowed=allowed,
            rule_id=rule.rule_id,
            identifier=state.identifier,
            action=rule.action if not allowed else ActionType.WARN,
            remaining_requests=remaining,
            current_load=system_load,
            retry_after=rule.retry_after_seconds if not allowed else None,
            reason=f"Adaptive limit: {adapted_limit:.2f} req/s (load: {system_load:.2f})"
        )
    
    async def update_state(self, state: RateLimitState, allowed: bool,
                          context: Dict[str, Any]) -> None:
        """Update adaptive state."""
        current_time = datetime.now(timezone.utc)
        state.last_request = current_time
        state.requests_count += 1
        
        if not allowed:
            state.rejected_count += 1
        
        # Update success rate
        total_requests = state.requests_count
        successful_requests = total_requests - state.rejected_count
        state.success_rate = successful_requests / max(1, total_requests)
    
    async def _get_system_load(self, context: Dict[str, Any]) -> float:
        """Get current system load."""
        try:
            # Get load from context or metrics
            if 'system_load' in context:
                return context['system_load']
            
            if self.metrics:
                # Get CPU and memory metrics
                cpu_usage = self.metrics.get_current_value("system_cpu_usage") or 0.0
                memory_usage = self.metrics.get_current_value("system_memory_usage") or 0.0
                active_connections = self.metrics.get_current_value("active_connections") or 0.0
                
                # Combine metrics into load score
                load = (cpu_usage * 0.4 + memory_usage * 0.4 + min(active_connections / 1000, 1.0) * 0.2)
                return min(1.0, load)
            
            return 0.0
            
        except Exception as e:
            self.logger.warning(f"Failed to get system load: {str(e)}")
            return 0.0


class CircuitBreakerStrategy(RateLimitStrategy):
    """Circuit breaker pattern for rate limiting."""
    
    def __init__(self, logger):
        self.logger = logger
    
    async def check_limit(self, rule: RateLimitRule, state: RateLimitState,
                         context: Dict[str, Any]) -> RateLimitResult:
        """Check circuit breaker state."""
        current_time = datetime.now(timezone.utc)
        
        # Check if circuit should be reset
        if (state.circuit_open and state.last_failure and
            (current_time - state.last_failure).total_seconds() > rule.recovery_timeout_seconds):
            state.circuit_open = False
            state.failure_count = 0
            state.last_failure = None
            
            self.logger.info(f"Circuit breaker reset for {state.identifier}")
        
        # If circuit is open, block request
        if state.circuit_open:
            return RateLimitResult(
                allowed=False,
                rule_id=rule.rule_id,
                identifier=state.identifier,
                action=ActionType.BLOCK,
                remaining_requests=0,
                retry_after=rule.recovery_timeout_seconds,
                circuit_state="open",
                reason="Circuit breaker is open"
            )
        
        # Allow request if circuit is closed
        return RateLimitResult(
            allowed=True,
            rule_id=rule.rule_id,
            identifier=state.identifier,
            action=ActionType.WARN,
            circuit_state="closed"
        )
    
    async def update_state(self, state: RateLimitState, allowed: bool,
                          context: Dict[str, Any]) -> None:
        """Update circuit breaker state."""
        current_time = datetime.now(timezone.utc)
        state.last_request = current_time
        state.requests_count += 1
        
        # Check for failures in context
        request_failed = context.get('request_failed', False)
        
        if request_failed:
            state.failure_count += 1
            state.last_failure = current_time
            
            # Open circuit if threshold exceeded
            if state.failure_count >= 5:  # Rule-based threshold
                state.circuit_open = True
                self.logger.warning(f"Circuit breaker opened for {state.identifier}")
        else:
            # Reset failure count on success
            state.failure_count = 0


class RateLimitStorage(ABC):
    """Abstract storage backend for rate limit state."""
    
    @abstractmethod
    async def get_state(self, identifier: str, rule_id: str) -> Optional[RateLimitState]:
        """Get rate limit state."""
        pass
    
    @abstractmethod
    async def set_state(self, state: RateLimitState) -> None:
        """Store rate limit state."""
        pass
    
    @abstractmethod
    async def delete_state(self, identifier: str, rule_id: str) -> None:
        """Delete rate limit state."""
        pass
    
    @abstractmethod
    async def cleanup_expired_states(self) -> int:
        """Clean up expired states."""
        pass


class MemoryStorage(RateLimitStorage):
    """In-memory storage for rate limit state."""
    
    def __init__(self):
        self.states: Dict[str, RateLimitState] = {}
        self.lock = threading.Lock()
    
    def _make_key(self, identifier: str, rule_id: str) -> str:
        """Create storage key."""
        return f"{rule_id}:{identifier}"
    
    async def get_state(self, identifier: str, rule_id: str) -> Optional[RateLimitState]:
        """Get state from memory."""
        with self.lock:
            return self.states.get(self._make_key(identifier, rule_id))
    
    async def set_state(self, state: RateLimitState) -> None:
        """Store state in memory."""
        with self.lock:
            key = self._make_key(state.identifier, state.rule_id)
            self.states[key] = state
    
    async def delete_state(self, identifier: str, rule_id: str) -> None:
        """Delete state from memory."""
        with self.lock:
            key = self._make_key(identifier, rule_id)
            self.states.pop(key, None)
    
    async def cleanup_expired_states(self) -> int:
        """Clean up old states."""
        current_time = datetime.now(timezone.utc)
        expired_keys = []
        
        with self.lock:
            for key, state in self.states.items():
                # Remove states older than 24 hours with no recent activity
                if (state.last_request and 
                    (current_time - state.last_request).total_seconds() > 86400):
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self.states[key]
        
        return len(expired_keys)


class RedisStorage(RateLimitStorage):
    """Redis-based storage for distributed rate limiting."""
    
    def __init__(self, redis_cache: RedisCache, ttl_seconds: int = 3600):
        self.redis = redis_cache
        self.ttl = ttl_seconds
        self.logger = get_logger(__name__)
    
    def _make_key(self, identifier: str, rule_id: str) -> str:
        """Create Redis key."""
        return f"rate_limit:{rule_id}:{identifier}"
    
    async def get_state(self, identifier: str, rule_id: str) -> Optional[RateLimitState]:
        """Get state from Redis."""
        try:
            key = self._make_key(identifier, rule_id)
            data = await self.redis.get(key)
            
            if data:
                state_dict = json.loads(data)
                return self._deserialize_state(state_dict)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get rate limit state from Redis: {str(e)}")
            return None
    
    async def set_state(self, state: RateLimitState) -> None:
        """Store state in Redis."""
        try:
            key = self._make_key(state.identifier, state.rule_id)
            data = json.dumps(self._serialize_state(state), default=str)
            await self.redis.set(key, data, ttl=self.ttl)
            
        except Exception as e:
            self.logger.error(f"Failed to set rate limit state in Redis: {str(e)}")
    
    async def delete_state(self, identifier: str, rule_id: str) -> None:
        """Delete state from Redis."""
        try:
            key = self._make_key(identifier, rule_id)
            await self.redis.delete(key)
            
        except Exception as e:
            self.logger.error(f"Failed to delete rate limit state from Redis: {str(e)}")
    
    async def cleanup_expired_states(self) -> int:
        """Redis handles expiration automatically."""
        return 0
    
    def _serialize_state(self, state: RateLimitState) -> Dict[str, Any]:
        """Serialize state for Redis storage."""
        return {
            'identifier': state.identifier,
            'rule_id': state.rule_id,
            'scope': state.scope.value,
            'tokens': state.tokens,
            'last_refill': state.last_refill.isoformat(),
            'request_times': [t.isoformat() for t in list(state.request_times)],
            'requests_count': state.requests_count,
            'rejected_count': state.rejected_count,
            'delayed_count': state.delayed_count,
            'failure_count': state.failure_count,
            'last_failure': state.last_failure.isoformat() if state.last_failure else None,
            'circuit_open': state.circuit_open,
            'current_limit': state.current_limit,
            'average_response_time': state.average_response_time,
            'success_rate': state.success_rate,
            'created_at': state.created_at.isoformat(),
            'last_request': state.last_request.isoformat() if state.last_request else None,
            'last_reset': state.last_reset.isoformat() if state.last_reset else None
        }
    
    def _deserialize_state(self, data: Dict[str, Any]) -> RateLimitState:
        """Deserialize state from Redis storage."""
        state = RateLimitState(
            identifier=data['identifier'],
            rule_id=data['rule_id'],
            scope=LimitScope(data['scope']),
            tokens=data['tokens'],
            last_refill=datetime.fromisoformat(data['last_refill']),
            requests_count=data['requests_count'],
            rejected_count=data['rejected_count'],
            delayed_count=data['delayed_count'],
            failure_count=data['failure_count'],
            circuit_open=data['circuit_open'],
            current_limit=data['current_limit'],
            average_response_time=data['average_response_time'],
            success_rate=data['success_rate'],
            created_at=datetime.fromisoformat(data['created_at'])
        )
        
        # Restore request times
        state.request_times = deque([
            datetime.fromisoformat(t) for t in data['request_times']
        ], maxlen=1000)
        
        # Restore optional timestamps
        if data['last_failure']:
            state.last_failure = datetime.fromisoformat(data['last_failure'])
        if data['last_request']:
            state.last_request = datetime.fromisoformat(data['last_request'])
        if data['last_reset']:
            state.last_reset = datetime.fromisoformat(data['last_reset'])
        
        return state


class EnhancedRateLimiter:
    """
    Advanced Rate Limiting System for the AI Assistant.
    
    This system provides comprehensive rate limiting capabilities including:
    - Multiple rate limiting algorithms (token bucket, sliding window, adaptive)
    - User-specific and session-specific limits
    - Integration with authentication and authorization
    - Distributed rate limiting with Redis
    - Adaptive limits based on system load
    - Circuit breaker pattern for fault tolerance
    - Intelligent throttling and quality reduction
    - Learning from user behavior patterns
    - Real-time monitoring and alerting
    - Integration with all core system components
    """
    
    def __init__(self, container: Container):
        """
        Initialize the enhanced rate limiter.
        
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
        self.session_manager = container.get(SessionManager)
        self.component_manager = container.get(ComponentManager)
        
        # Security
        try:
            self.auth_manager = container.get(AuthenticationManager)
            self.authz_manager = container.get(AuthorizationManager)
        except Exception:
            self.auth_manager = None
            self.authz_manager = None
        
        # Storage and caching
        try:
            self.redis_cache = container.get(RedisCache)
            self.database = container.get(DatabaseManager)
        except Exception:
            self.redis_cache = None
            self.database = None
        
        # Learning systems
        try:
            self.continual_learner = container.get(ContinualLearner)
            self.preference_learner = container.get(PreferenceLearner)
            self.feedback_processor = container.get(FeedbackProcessor)
        except Exception:
            self.continual_learner = None
            self.preference_learner = None
            self.feedback_processor = None
        
        # Observability
        self.metrics = container.get(MetricsCollector)
        self.tracer = container.get(TraceManager)
        
        # Rate limiting components
        self.rules: Dict[str, RateLimitRule] = {}
        self.strategies: Dict[RateLimitType, RateLimitStrategy] = {}
        self.storage: RateLimitStorage
        
        # State management
        self.active_blocks: Set[str] = set()
        self.request_queue: asyncio.Queue = asyncio.Queue()
        self.processing_semaphore = asyncio.Semaphore(100)
        
        # Configuration
        self.enable_distributed = self.config.get("rate_limiting.enable_distributed", True)
        self.enable_adaptive = self.config.get("rate_limiting.enable_adaptive", True)
        self.enable_learning = self.config.get("rate_limiting.enable_learning", True)
        self.default_user_tier = UserTier(self.config.get("rate_limiting.default_user_tier", "free"))
        
        # Performance tracking
        self.request_stats: deque = deque(maxlen=10000)
        self.block_stats: Dict[str, int] = defaultdict(int)
        
        # Initialize components
        self._setup_strategies()
        self._setup_storage()
        self._setup_default_rules()
        self._setup_monitoring()
        
        # Register health check
        self.health_check.register_component("rate_limiter", self._health_check_callback)
        
        self.logger.info("EnhancedRateLimiter initialized successfully")

    def _setup_strategies(self) -> None:
        """Setup rate limiting strategies."""
        self.strategies = {
            RateLimitType.TOKEN_BUCKET: TokenBucketStrategy(self.logger),
            RateLimitType.SLIDING_WINDOW: SlidingWindowStrategy(self.logger),
            RateLimitType.ADAPTIVE: AdaptiveStrategy(self.logger, self.metrics),
            RateLimitType.CIRCUIT_BREAKER: CircuitBreakerStrategy(self.logger)
        }
        
        self.logger.info(f"Initialized {len(self.strategies)} rate limiting strategies")

    def _setup_storage(self) -> None:
        """Setup storage backend for rate limit state."""
        if self.enable_distributed and self.redis_cache:
            self.storage = RedisStorage(self.redis_cache)
            self.logger.info("Using Redis storage for distributed rate limiting")
        else:
            self.storage = MemoryStorage()
            self.logger.info("Using memory storage for rate limiting")

    def _setup_default_rules(self) -> None:
        """Setup default rate limiting rules."""
        # Global API rate limit
        global_rule = RateLimitRule(
            rule_id="global_api",
            name="Global API Rate Limit",
            description="Overall API rate limit",
            limit_type=RateLimitType.TOKEN_BUCKET,
            scope=LimitScope.GLOBAL,
            requests_per_second=100.0,
            requests_per_minute=6000.0,
            burst_capacity=200,
            action=ActionType.QUEUE
        )
        self.rules[global_rule.rule_id] = global_rule
        
        # Free user limits
        free_user_rule = RateLimitRule(
            rule_id="free_user",
            name="Free User Rate Limit",
            description="Rate limits for free tier users",
            limit_type=RateLimitType.TOKEN_BUCKET,
            scope=LimitScope.USER,
            requests_per_second=1.0,
            requests_per_minute=60.0,
            requests_per_hour=3600.0,
            burst_capacity=5,
            user_tiers={UserTier.FREE},
            action=ActionType.BLOCK
        )
        self.rules[free_user_rule.rule_id] = free_user_rule
        
        # Premium user limits
        premium_user_rule = RateLimitRule(
            rule_id="premium_user",
            name="Premium User Rate Limit",
            description="Rate limits for premium tier users",
            limit_type=RateLimitType.ADAPTIVE,
            scope=LimitScope.USER,
            requests_per_second=10.0,
            requests_per_minute=600.0,
            requests_per_hour=36000.0,
            burst_capacity=20,
            user_tiers={UserTier.PREMIUM},
            action=ActionType.REDUCE_QUALITY,
            adaptive_enabled=True
        )
        self.rules[premium_user_rule.rule_id] = premium_user_rule
        
        # High-cost endpoint limits
        expensive_endpoint_rule = RateLimitRule(
            rule_id="expensive_endpoints",
            name="Expensive Endpoint Rate Limit",
            description="Limits for resource-intensive endpoints",
            limit_type=RateLimitType.SLIDING_WINDOW,
            scope=LimitScope.USER,
            requests_per_minute=10.0,
            requests_per_hour=100.0,
            endpoints={"/api/v1/generate", "/api/v1/analyze", "/api/v1/process"},
            action=ActionType.DELAY,
            delay_seconds=2.0
        )
        self.rules[expensive_endpoint_rule.rule_id] = expensive_endpoint_rule
        
        # Circuit breaker for failing services
        circuit_breaker_rule = RateLimitRule(
            rule_id="circuit_breaker",
            name="Circuit Breaker",
            description="Circuit breaker for failing services",
            limit_type=RateLimitType.CIRCUIT_BREAKER,
            scope=LimitScope.ENDPOINT,
            failure_threshold=5,
            recovery_timeout_seconds=30,
            action=ActionType.BLOCK
        )
        self.rules[circuit_breaker_rule.rule_id] = circuit_breaker_rule
        
        self.logger.info(f"Setup {len(self.rules)} default rate limiting rules")

    def _setup_monitoring(self) -> None:
        """Setup monitoring and metrics."""
        try:
            # Register rate limiting metrics
            self.metrics.register_counter("rate_limit_checks_total")
            self.metrics.register_counter("rate_limit_blocks_total")
            self.metrics.register_counter("rate_limit_delays_total")
            self.metrics.register_histogram("rate_limit_check_duration_seconds")
            self.metrics.register_gauge("active_rate_limit_blocks")
            self.metrics.register_gauge("rate_limit_queue_size")
            
        except Exception as e:
            self.logger.warning(f"Failed to setup monitoring: {str(e)}")

    async def initialize(self) -> None:
        """Initialize the rate limiter."""
        try:
            # Load additional rules from configuration
            await self._load_rules_from_config()
            
            # Start background tasks
            asyncio.create_task(self._cleanup_loop())
            asyncio.create_task(self._monitoring_loop())
            asyncio.create_task(self._queue_processor_loop())
            
            if self.enable_learning:
                asyncio.create_task(self._learning_loop())
            
            # Register event handlers
            await self._register_event_handlers()
            
            self.logger.info("RateLimiter initialization completed")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize RateLimiter: {str(e)}")
            raise RateLimitError(f"Initialization failed: {str(e)}")

    async def _register_event_handlers(self) -> None:
        """Register event handlers for system events."""
        # User authentication events
        self.event_bus.subscribe("user_authenticated", self._handle_user_authenticated)
        
        # Session events
        self.event_bus.subscribe("session_started", self._handle_session_started)
        self.event_bus.subscribe("session_ended", self._handle_session_ended)
        
        # System load events
        self.event_bus.subscribe("system_overload", self._handle_system_overload)
        self.event_bus.subscribe("component_health_changed", self._handle_component_health_change)

    @handle_exceptions
    async def check_rate_limit(
        self,
        request_context: Dict[str, Any],
        endpoint: Optional[str] = None,
        method: Optional[str] = None
    ) -> RateLimitResult:
        """
        Check if request should be rate limited.
        
        Args:
            request_context: Request context information
            endpoint: API endpoint being accessed
            method: HTTP method
            
        Returns:
            Rate limit check result
        """
        async with self.processing_semaphore:
            start_time = time.time()
            
            try:
                with self.tracer.trace("rate_limit_check") as span:
                    span.set_attributes({
                        "endpoint": endpoint or "unknown",
                        "method": method or "unknown",
                        "user_id": request_context.get("user_id", "anonymous"),
                        "session_id": request_context.get("session_id", "unknown")
                    })
                    
                    # Extract identifier information
                    identifiers = await self._extract_identifiers(request_context)
                    
                    # Find applicable rules
                    applicable_rules = await self._find_applicable_rules(
                        request_context, endpoint, method, identifiers
                    )
                    
                    if not applicable_rules:
                        # No rules apply, allow request
                        return RateLimitResult(
                            allowed=True,
                            rule_id="none",
                            identifier="none",
                            action=ActionType.WARN,
                            reason="No applicable rules"
                        )
                    
                    # Check each applicable rule
                    most_restrictive_result = None
                    
                    for rule in applicable_rules:
                        for identifier in identifiers:
                            if identifier['scope'] == rule.scope:
                                result = await self._check_single_rule(
                                    rule, identifier['value'], request_context
                                )
                                
                                # Keep the most restrictive result
                                if not result.allowed:
                                    most_restrictive_result = result
                                    break
                                elif (most_restrictive_result is None or 
                                      result.remaining_requests < most_restrictive_result.remaining_requests):
                                    most_restrictive_result = result
                    
                    final_result = most_restrictive_result or RateLimitResult(
                        allowed=True,
                        rule_id="default",
                        identifier="none",
                        action=ActionType.WARN
                    )
                    
                    # Update metrics
                    check_duration = time.time() - start_time
                    self.metrics.increment("rate_limit_checks_total")
                    self.metrics.record("rate_limit_check_duration_seconds", check_duration)
                    
                    if not final_result.allowed:
                        self.metrics.increment("rate_limit_blocks_total")
                        
                        # Emit rate limit triggered event
                        await self.event_bus.emit(RateLimitTriggered(
                            rule_id=final_result.rule_id,
                            identifier=final_result.identifier,
                            action=final_result.action.value,
                            retry_after=final_result.retry_after,
                            endpoint=endpoint
                        ))
                    
                    # Track statistics
                    self.request_stats.append({
                        'timestamp': datetime.now(timezone.utc),
                        'allowed': final_result.allowed,
                        'rule_id': final_result.rule_id,
                        'identifier': final_result.identifier,
                        'endpoint': endpoint,
                        'response_time': check_duration
                    })
                    
                    return final_result
                    
            except Exception as e:
                self.logger.error(f"Rate limit check failed: {str(e)}")
                
                # Allow request on error to avoid blocking legitimate traffic
                return RateLimitResult(
                    allowed=True,
                    rule_id="error",
                    identifier="unknown",
                    action=ActionType.WARN,
                    reason=f"Rate limit check error: {str(e)}"
                )

    async def _extract_identifiers(self, request_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract rate limiting identifiers from request context."""
        identifiers = []
        
        # User identifier
        user_id = request_context.get("user_id")
        if user_id:
            identifiers.append({
                'scope': LimitScope.USER,
                'value': user_id
            })
        
        # Session identifier
        session_id = request_context.get("session_id")
        if session_id:
            identifiers.append({
                'scope': LimitScope.SESSION,
                'value': session_id
            })
        
        # IP address identifier
        ip_address = request_context.get("client_ip")
        if ip_address:
            identifiers.append({
                'scope': LimitScope.IP_ADDRESS,
                'value': ip_address
            })
        
        # API key identifier
        api_key = request_context.get("api_key")
        if api_key:
            identifiers.append({
                'scope': LimitScope.API_KEY,
                'value': api_key
            })
        
        # Global identifier
        identifiers.append({
            'scope': LimitScope.GLOBAL,
            'value': "global"
        })
        
        return identifiers

    async def _find_applicable_rules(
        self,
        request_context: Dict[str, Any],
        endpoint: Optional[str],
        method: Optional[str],
        identifiers: List[Dict[str, Any]]
    ) -> List[RateLimitRule]:
        """Find rules that apply to the current request."""
        applicable_rules = []
        
        # Get user tier
        user_tier = await self._get_user_tier(request_context.get("user_id"))
        
        for rule in self.rules.values():
            if not rule.enabled:
                continue
            
            # Check if rule has expired
            if rule.expires_at and datetime.now(timezone.utc) > rule.expires_at:
                continue
            
            # Check scope match
            scope_matches = any(
                identifier['scope'] == rule.scope for identifier in identifiers
            )
            if not scope_matches:
                continue
            
            # Check endpoint match
            if rule.endpoints and endpoint and endpoint not in rule.endpoints:
                continue
            
            # Check method match
            if rule.methods and method and method not in rule.methods:
                continue
            
            # Check user tier match
            if rule.user_tiers and user_tier not in rule.user_tiers:
                continue
            
            # Check IP range match
            if rule.ip_ranges:
                client_ip = request_context.get("client_ip")
                if not client_ip or not self._ip_in_ranges(client_ip, rule.ip_ranges):
                    continue
            
            applicable_rules.append(rule)
        
        # Sort by priority
        applicable_rules.sort(key=lambda r: r.priority.value, reverse=True)
        
        return applicable_rules

    async def _check_single_rule(
        self,
        rule: RateLimitRule,
        identifier: str,
        request_context: Dict[str, Any]
    ) -> RateLimitResult:
        """Check a single rate limiting rule."""
        # Get current state
        state = await self.storage.get_state(identifier, rule.rule_id)
        
        if state is None:
            # Create new state
            state = RateLimitState(
                identifier=identifier,
                rule_id=rule.rule_id,
                scope=rule.scope,
                tokens=rule.burst_capacity,
                current_limit=rule.requests_per_second
            )
        
        # Get appropriate strategy
        strategy = self.strategies.get(rule.limit_type)
        if not strategy:
            self.logger.error(f"No strategy found for limit type: {rule.limit_type}")
            return RateLimitResult(
                allowed=True,
                rule_id=rule.rule_id,
                identifier=identifier,
                action=ActionType.WARN,
                reason="No strategy available"
            )
        
        # Check rate limit
        result = await strategy.check_limit(rule, state, request_context)
        
        # Update state
        await strategy.update_state(state, result.allowed, request_context)
        
        # Store updated state
        await self.storage.set_state(state)
        
        return result

    async def _get_user_tier(self, user_id: Optional[str]) -> UserTier:
        """Get user tier for rate limiting."""
        if not user_id:
            return self.default_user_tier
        
        try:
            # Get user tier from preference learner or auth system
            if self.preference_learner:
                preferences = await self.preference_learner.get_user_preferences(user_id)
                if preferences and 'tier' in preferences:
                    return UserTier(preferences['tier'])
            
            # Check with auth manager
            if self.auth_manager:
                user_info = await self.auth_manager.get_user_info(user_id)
                if user_info and 'tier' in user_info:
                    return UserTier(user_info['tier'])
            
            return self.default_user_tier
            
        except Exception as e:
            self.logger.warning(f"Failed to get user tier for {user_id}: {str(e)}")
            return self.default_user_tier

    def _ip_in_ranges(self, ip: str, ranges: List[str]) -> bool:
        """Check if IP address is in any of the specified ranges."""
        try:
            import ipaddress
            ip_addr = ipaddress.ip_address(ip)
            
            for ip_range in ranges:
                try:
                    network = ipaddress.ip_network(ip_range, strict=False)
                    if ip_addr in network:
                        return True
                except ValueError:
                    continue
            
            return False
            
        except Exception as e:
            self.logger.warning(f"Failed to check IP ranges: {str(e)}")
            return False

    @handle_exceptions
    async def update_rule(self, rule: RateLimitRule) -> None:
        """
        Update or add a rate limiting rule.
        
        Args:
            rule: Rate limiting rule to update/add
        """
        self.rules[rule.rule_id] = rule
        
        # Emit configuration event
        await self.event_bus.emit(RateLimitConfigured(
            rule_id=rule.rule_id,
            rule_name=rule.name,
            scope=rule.scope.value,
            limit_type=rule.limit_type.value,
            requests_per_second=rule.requests_per_second
        ))
        
        self.logger.info(f"Updated rate limiting rule: {rule.rule_id}")

    @handle_exceptions
    async def remove_rule(self, rule_id: str) -> None:
        """
        Remove a rate limiting rule.
        
        Args:
            rule_id: ID of rule to remove
        """
        if rule_id in self.rules:
            del self.rules[rule_id]
            self.logger.info(f"Removed rate limiting rule: {rule_id}")

    @handle_exceptions
    async def reset_user_limits(self, user_id: str) -> None:
        """
        Reset rate limits for a specific user.
        
        Args:
            user_id: User ID to reset limits for
        """
        # Find and reset all user-related states
        for rule_id in self.rules.keys():
            await self.storage.delete_state(user_id, rule_id)
        
        # Emit reset event
        await self.event_bus.emit(RateLimitReset(
            identifier=user_id,
            scope=LimitScope.USER.value,
            reason="Manual reset"
        ))
        
        self.logger.info(f"Reset rate limits for user: {user_id}")

    @handle_exceptions
    async def bypass_limits(
        self,
        identifier: str,
        scope: LimitScope,
        duration_seconds: int = 3600
    ) -> None:
        """
        Temporarily bypass rate limits for an identifier.
        
        Args:
            identifier: Identifier to bypass
            scope: Scope of bypass
            duration_seconds: Duration of bypass in seconds
        """
        # Implementation would add temporary bypass
        # For now, just emit event
        await self.event_bus.emit(RateLimitBypass(
            identifier=identifier,
            scope=scope.value,
            duration_seconds=duration_seconds,
            reason="Manual bypass"
        ))
        
        self.logger.info(f"Bypassed rate limits for {scope.value}: {identifier}")

    def get_rate_limit_status(self, identifier: str, scope: LimitScope) -> Dict[str, Any]:
        """Get current rate limit status for an identifier."""
        status = {
            'identifier': identifier,
            'scope': scope.value,
            'rules': [],
            'is_blocked': identifier in self.active_blocks
        }
        
        # Find applicable rules
        for rule in self.rules.values():
            if rule.scope == scope:
                status['rules'].append({
                    'rule_id': rule.rule_id,
                    'name': rule.name,
                    'limit_type': rule.limit_type.value,
                    'requests_per_second': rule.requests_per_second,
                    'enabled': rule.enabled
                })
        
        return status

    def get_system_status(self) -> Dict[str, Any]:
        """Get overall rate limiting system status."""
        total_rules = len(self.rules)
        enabled_rules = len([r for r in self.rules.values() if r.enabled])
        
        # Calculate recent request statistics
        recent_requests = [
            r for r in self.request_stats 
            if (datetime.now(timezone.utc) - r['timestamp']).total_seconds() < 300
        ]
        
        recent_blocks = len([r for r in recent_requests if not r['allowed']])
        recent_total = len(recent_requests)
        block_rate = recent_blocks / max(1, recent_total)
        
        return {
            'total_rules': total_rules,
            'enabled_rules': enabled_rules,
            'active_blocks': len(self.active_blocks),
            'queue_size': self.request_queue.qsize(),
            'recent_requests': recent_total,
            'recent_blocks': recent_blocks,
            'block_rate': block_rate,
            'storage_type': type(self.storage).__name__,
            'strategies_available': list(self.strategies.keys()),
            'enable_distributed': self.enable_distributed,
            'enable_adaptive': self.enable_adaptive,
            'enable_learning': self.enable_learning
        }

    async def _load_rules_from_config(self) -> None:
        """Load additional rules from configuration."""
        try:
            rules_config = self.config.get("rate_limiting.rules", [])
            
            for rule_config in rules_config:
                rule = self._config_to_rule(rule_config)
                self.rules[rule.rule_id] = rule
            
            self.logger.info(f"Loaded {len(rules_config)} rules from configuration")
            
        except Exception as e:
            self.logger.warning(f"Failed to load rules from config: {str(e)}")

    def _config_to_rule(self, config: Dict[str, Any]) -> RateLimitRule:
        """Convert configuration dictionary to RateLimitRule."""
        # Convert enum values
        config['limit_type'] = RateLimitType(config.get('limit_type', 'token_bucket'))
        config['scope'] = LimitScope(config.get('scope', 'user'))
        config['action'] = ActionType(config.get('action', 'block'))
        config['priority'] = PriorityLevel(config.get('priority', 'normal'))
        
        # Convert sets
        if 'endpoints' in config:
            config['endpoints'] = set(config['endpoints'])
        if 'methods' in config:
            config['methods'] = set(config['methods'])
        if 'user_tiers' in config:
            config['user_tiers'] = {UserTier(tier) for tier in config['user_tiers']}
        if 'tags' in config:
            config['tags'] = set(config['tags'])
        
        # Convert timestamps
        if 'expires_at' in config and config['expires_at']:
            config['expires_at'] = datetime.fromisoformat(config['expires_at'])
        
        return RateLimitRule(**config)

    async def _cleanup_loop(self) -> None:
        """Background task for cleaning up expired state."""
        while True:
            try:
                # Clean up expired states
                cleaned_count = await self.storage.cleanup_expired_states()
                
                if cleaned_count > 0:
                    self.logger.debug(f"Cleaned up {cleaned_count} expired rate limit states")
                
                await asyncio.sleep(300)  # Clean every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Error in cleanup loop: {str(e)}")
                await asyncio.sleep(60)

    async def _monitoring_loop(self) -> None:
        """Background task for monitoring and metrics."""
        while True:
            try:
                # Update metrics
                self.metrics.set("active_rate_limit_blocks", len(self.active_blocks))
                self.metrics.set("rate_limit_queue_size", self.request_queue.qsize())
                
                await asyncio.sleep(30)  # Monitor every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {str(e)}")
                await asyncio.sleep(30)

    async def _queue_processor_loop(self) -> None:
        """Background task for processing queued requests."""
        while True:
            try:
                # Process queued requests
                # This would implement request queuing logic
                await asyncio.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Error in queue processor: {str(e)}")
                await asyncio.sleep(1)

    async def _learning_loop(self) -> None:
        """Background task for learning from rate limiting patterns."""
        if not self.enable_learning:
            return
        
        while True:
            try:
                # Analyze patterns and adapt rules
                await self._analyze_usage_patterns()
                
                await asyncio.sleep(3600)  # Learn every hour
                
            except Exception as e:
                self.logger.error(f"Error in learning loop: {str(e)}")
                await asyncio.sleep(3600)

    async def _analyze_usage_patterns(self) -> None:
        """Analyze usage patterns and adapt rate limiting."""
        try:
            # Analyze recent request patterns
            if len(self.request_stats) < 100:
                return
            
            recent_stats = list(self.request_stats)[-1000:]  # Last 1000 requests
            
            # Calculate patterns
            block_rate = len([s for s in recent_stats if not s['allowed']]) / len(recent_stats)
            
            # Analyze by endpoint
            endpoint_stats = defaultdict(list)
            for stat in recent_stats:
                endpoint = stat.get('endpoint', 'unknown')
                endpoint_stats[endpoint].append(stat)
            
            # Suggest rule adaptations
            for endpoint, stats in endpoint_stats.items():
                endpoint_block_rate = len([s for s in stats if not s['allowed']]) / len(stats)
                
                if endpoint_block_rate > 0.1:  # High block rate
                    self.logger.info(f"High block rate for {endpoint}: {endpoint_block_rate:.2%}")
                    # Could suggest rule adjustments here
            
            # Learn from patterns with continual learner
            if self.continual_learner:
                learning_data = {
                    'type': 'rate_limiting_patterns',
                    'block_rate': block_rate,
                    'endpoint_patterns': dict(endpoint_stats),
                    'timestamp': datetime.now(timezone.utc)
                }
                
                await self.continual_learner.learn_from_data(learning_data)
            
        except Exception as e:
            self.logger.warning(f"Failed to analyze usage patterns: {str(e)}")

    async def _handle_user_authenticated(self, event) -> None:
        """Handle user authentication events."""
        user_id = event.user_id
        
        # Update user tier if needed
        user_tier = await self._get_user_tier(user_id)
        
        self.logger.debug(f"User {user_id} authenticated with tier {user_tier.value}")

    async def _handle_session_started(self, event) -> None:
        """Handle session start events."""
        session_id = event.session_id
        
        # Initialize session-specific rate limiting if needed
        self.logger.debug(f"Session started: {session_id}")

    async def _handle_session_ended(self, event) -> None:
        """Handle session end events."""
        session_id = event.session_id
        
        # Clean up session-specific rate limit states
        for rule_id in self.rules.keys():
            await self.storage.delete_state(session_id, rule_id)
        
        self.logger.debug(f"Cleaned up rate limits for ended session: {session_id}")

    async def _handle_system_overload(self, event) -> None:
        """Handle system overload events."""
        # Activate more aggressive rate limiting during overload
        self.logger.warning("System overload detected, activating aggressive rate limiting")
        
        # Could temporarily modify rules here

    async def _handle_component_health_change(self, event) -> None:
        """Handle component health change events."""
        if not event.healthy:
            # Adjust rate limits when components are unhealthy
            self.logger.warning(f"Component {event.component} unhealthy, may adjust rate limits")

    async def _health_check_callback(self) -> Dict[str, Any]:
        """Health check callback for the rate limiter."""
        try:
            # Check storage health
            storage_healthy = True
            try:
                # Test storage with a dummy operation
                test_state = RateLimitState(
                    identifier="health_check",
                    rule_id="test",
                    scope=LimitScope.GLOBAL
                )
                await self.storage.set_state(test_state)
                await self.storage.delete_state("health_check", "test")
            except Exception:
                storage_healthy = False
            
            # Calculate health metrics
            recent_blocks = len([
                r for r in self.request_stats 
                if (datetime.now(timezone.utc) - r['timestamp']).total_seconds() < 300
                and not r['allowed']
            ])
            
            total_recent = len([
                r for r in self.request_stats 
                if (datetime.now(timezone.utc) - r['timestamp']).total_seconds() < 300
            ])
            
            block_rate = recent_blocks / max(1, total_recent)
            
            return {
                "status": "healthy" if storage_healthy and block_rate < 0.5 else "degraded",
                "storage_healthy": storage_healthy,
                "total_rules": len(self.rules),
                "enabled_rules": len([r for r in self.rules.values() if r.enabled]),
                "active_blocks": len(self.active_blocks),
                "recent_block_rate": block_rate,
                "queue_size": self.request_queue.qsize()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }

    async def cleanup(self) -> None:
        """Cleanup resources."""
        try:
            # Clean up storage
            if hasattr(self.storage, 'cleanup'):
                await self.storage.cleanup()
            
            self.logger.info("RateLimiter cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")

    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            if hasattr(self, 'active_blocks'):
                self.logger.debug("RateLimiter destroyed")
        except Exception:
            pass  # Ignore cleanup errors in destructor
