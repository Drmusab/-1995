"""
Comprehensive Skill Factory for AI Assistant
Author: Drmusab
Last Modified: 2025-01-20 11:00:00 UTC

This module provides a comprehensive skill factory that integrates with the existing
core system architecture, offering dynamic skill creation, lifecycle management,
composition, monitoring, and hot-reload capabilities.
"""

import importlib
import inspect
import json
import time
import uuid
import weakref
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Set, Type, Union

import asyncio

# Assistant components
from src.assistant.core import ComponentInterface, ComponentManager
from src.assistant.core import SessionManager
from src.assistant.core import WorkflowOrchestrator

# Core imports
from src.core.config.loader import ConfigLoader
from src.core.dependency_injection import Container
from src.core.error_handling import ErrorHandler, handle_exceptions
from src.core.events.event_bus import EventBus
from src.core.events.event_types import BaseEvent, EventCategory, EventPriority, EventSeverity
from src.core.health_check import HealthCheck

# Learning systems
from src.learning.continual_learning import ContinualLearner
from src.learning.feedback_processor import FeedbackProcessor
from src.learning.preference_learning import PreferenceLearner

# Memory systems
from src.memory.core_memory.memory_manager import MemoryManager
from src.memory.core_memory.memory_types import WorkingMemory
from src.memory.operations.context_manager import ContextManager
from src.observability.logging.config import get_logger

# Observability
from src.observability.monitoring.metrics import MetricsCollector
from src.observability.monitoring.tracing import TraceManager

# Skills
from src.skills.skill_registry import (
    SkillCapability,
    SkillInterface,
    SkillMetadata,
    SkillRegistration,
    SkillRegistry,
    SkillState,
    SkillType,
)
from src.skills.skill_validator import SkillValidator, ValidationReport


class SkillExecutionMode(Enum):
    """Skill execution modes."""

    SYNCHRONOUS = "synchronous"
    ASYNCHRONOUS = "asynchronous"
    STREAMING = "streaming"
    BATCH = "batch"


class SkillCacheStrategy(Enum):
    """Skill caching strategies."""

    NO_CACHE = "no_cache"
    MEMORY_CACHE = "memory_cache"
    PERSISTENT_CACHE = "persistent_cache"
    DISTRIBUTED_CACHE = "distributed_cache"


class CircuitBreakerState(Enum):
    """Circuit breaker states for skill resilience."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class SkillExecutionContext:
    """Context for skill execution."""

    session_id: Optional[str] = None
    user_id: Optional[str] = None
    correlation_id: Optional[str] = None
    execution_mode: SkillExecutionMode = SkillExecutionMode.SYNCHRONOUS
    timeout_seconds: float = 30.0
    cache_strategy: SkillCacheStrategy = SkillCacheStrategy.MEMORY_CACHE
    retry_count: int = 3
    circuit_breaker_enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SkillExecutionResult:
    """Result of skill execution."""

    skill_id: str
    execution_id: str
    success: bool
    result: Any = None
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    cache_hit: bool = False
    circuit_breaker_triggered: bool = False
    retry_count: int = 0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SkillTemplate:
    """Template for skill scaffolding."""

    template_id: str
    name: str
    description: str
    skill_type: SkillType
    template_path: Path
    variables: Dict[str, Any] = field(default_factory=dict)
    required_capabilities: List[str] = field(default_factory=list)


@dataclass
class CircuitBreaker:
    """Circuit breaker for skill resilience."""

    skill_id: str
    state: CircuitBreakerState = CircuitBreakerState.CLOSED
    failure_count: int = 0
    failure_threshold: int = 5
    recovery_timeout: timedelta = field(default_factory=lambda: timedelta(seconds=60))
    last_failure_time: Optional[datetime] = None
    success_threshold: int = 3
    success_count: int = 0


@dataclass
class SkillComposition:
    """Represents a composition of multiple skills."""

    composition_id: str
    name: str
    skills: List[str]
    execution_order: List[str]
    data_flow: Dict[str, Any]
    error_handling: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


# Skill Factory Events
@dataclass
class SkillCreated(BaseEvent):
    """Event fired when a skill is created."""

    skill_id: str
    skill_type: SkillType
    execution_id: str
    category: EventCategory = EventCategory.SKILL


@dataclass
class SkillExecutionStarted(BaseEvent):
    """Event fired when skill execution starts."""

    skill_id: str
    execution_id: str
    context: Dict[str, Any]
    category: EventCategory = EventCategory.SKILL


@dataclass
class SkillExecutionCompleted(BaseEvent):
    """Event fired when skill execution completes."""

    skill_id: str
    execution_id: str
    success: bool
    execution_time_ms: float
    category: EventCategory = EventCategory.SKILL


@dataclass
class SkillExecutionFailed(BaseEvent):
    """Event fired when skill execution fails."""

    skill_id: str
    execution_id: str
    error: str
    retry_count: int
    category: EventCategory = EventCategory.SKILL
    severity: EventSeverity = EventSeverity.ERROR


@dataclass
class CircuitBreakerTriggered(BaseEvent):
    """Event fired when circuit breaker is triggered."""

    skill_id: str
    state: CircuitBreakerState
    failure_count: int
    category: EventCategory = EventCategory.SKILL
    priority: EventPriority = EventPriority.HIGH


@dataclass
class SkillHotReloaded(BaseEvent):
    """Event fired when a skill is hot-reloaded."""

    skill_id: str
    old_version: str
    new_version: str
    category: EventCategory = EventCategory.SKILL


class SkillFactory(ComponentInterface):
    """
    Comprehensive skill factory providing dynamic skill creation and management.

    Features:
    - Dynamic skill creation and dependency injection
    - Skill lifecycle management (creation, initialization, cleanup)
    - Support for different skill types (builtin, custom, meta-skills)
    - Skill versioning and compatibility checking
    - Integration with core system components
    - Skill composition and chaining
    - Performance monitoring and optimization
    - Security and validation
    - Skill caching and reuse
    - Hot-reload capabilities
    - Circuit breaker pattern for resilience
    - Skill templates and scaffolding
    """

    def __init__(self, container: Container):
        """Initialize the comprehensive skill factory."""
        self.container = container
        self.logger = get_logger(__name__)
        self.config = container.get(ConfigLoader)
        self.event_bus = container.get(EventBus)
        self.error_handler = container.get(ErrorHandler)
        self.health_check = container.get(HealthCheck)

        # Core system integration
        self.component_manager = container.get(ComponentManager)
        self.workflow_orchestrator = container.get(WorkflowOrchestrator)
        self.session_manager = container.get(SessionManager)

        # Skills management
        self.skill_registry = container.get(SkillRegistry)
        self.skill_validator = container.get(SkillValidator)

        # Memory systems
        self.memory_manager = container.get(MemoryManager)
        self.context_manager = container.get(ContextManager)
        self.working_memory = container.get(WorkingMemory)

        # Learning systems
        self.continual_learner = container.get(ContinualLearner)
        self.preference_learner = container.get(PreferenceLearner)
        self.feedback_processor = container.get(FeedbackProcessor)

        # Observability
        self.metrics = container.get(MetricsCollector)
        self.tracer = container.get(TraceManager)

        # Factory state
        self.active_skills: Dict[str, Any] = {}  # skill_id -> instance
        self.skill_cache: Dict[str, Any] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.execution_history: deque = deque(maxlen=10000)
        self.skill_templates: Dict[str, SkillTemplate] = {}
        self.skill_compositions: Dict[str, SkillComposition] = {}

        # Performance tracking
        self.skill_performance: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.skill_usage_stats: Dict[str, int] = defaultdict(int)

        # Configuration
        self.max_concurrent_executions = self.config.get("skills.max_concurrent", 100)
        self.default_timeout = self.config.get("skills.default_timeout", 30.0)
        self.cache_enabled = self.config.get("skills.cache_enabled", True)
        self.hot_reload_enabled = self.config.get("skills.hot_reload_enabled", False)
        self.circuit_breaker_enabled = self.config.get("skills.circuit_breaker_enabled", True)

        # Concurrency control
        self.execution_semaphore = asyncio.Semaphore(self.max_concurrent_executions)
        self.initialization_lock = asyncio.Lock()

        # Health state
        self.is_healthy = True
        self.last_health_check = datetime.now(timezone.utc)

        # Initialize templates
        self._initialize_skill_templates()

        self.logger.info("SkillFactory initialized successfully")

    async def initialize(self) -> None:
        """Initialize the skill factory component."""
        try:
            # Discover and register built-in skills
            await self._discover_builtin_skills()

            # Setup hot-reload if enabled
            if self.hot_reload_enabled:
                await self._setup_hot_reload()

            # Initialize circuit breakers for registered skills
            for skill_id in self.skill_registry.skills:
                self._initialize_circuit_breaker(skill_id)

            self.logger.info("SkillFactory initialization completed")

        except Exception as e:
            self.logger.error(f"Failed to initialize SkillFactory: {str(e)}")
            raise

    async def start(self) -> None:
        """Start the skill factory component."""
        self.logger.info("SkillFactory started")

    async def stop(self) -> None:
        """Stop the skill factory component."""
        # Cleanup all active skills
        await self._cleanup_all_skills()
        self.logger.info("SkillFactory stopped")

    async def cleanup(self) -> None:
        """Cleanup skill factory resources."""
        await self.stop()

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the skill factory."""
        health_status = {
            "status": "healthy" if self.is_healthy else "unhealthy",
            "active_skills": len(self.active_skills),
            "total_registered_skills": len(self.skill_registry.skills),
            "cache_size": len(self.skill_cache),
            "execution_history_size": len(self.execution_history),
            "circuit_breakers": {
                skill_id: cb.state.value for skill_id, cb in self.circuit_breakers.items()
            },
            "last_health_check": self.last_health_check.isoformat(),
        }

        # Check circuit breaker health
        open_breakers = sum(
            1 for cb in self.circuit_breakers.values() if cb.state == CircuitBreakerState.OPEN
        )

        if open_breakers > len(self.circuit_breakers) * 0.5:  # More than 50% open
            health_status["status"] = "degraded"
            health_status["warning"] = f"{open_breakers} circuit breakers are open"

        self.last_health_check = datetime.now(timezone.utc)
        return health_status

    @handle_exceptions
    async def create_skill(
        self,
        skill_id: str,
        config: Optional[Dict[str, Any]] = None,
        context: Optional[SkillExecutionContext] = None,
    ) -> Any:
        """
        Create and initialize a skill instance.

        Args:
            skill_id: Identifier of the skill to create
            config: Configuration for the skill
            context: Execution context

        Returns:
            Initialized skill instance
        """
        execution_id = str(uuid.uuid4())
        start_time = time.time()

        try:
            # Get skill registration
            registration = self.skill_registry.get_skill(skill_id)
            if not registration:
                raise ValueError(f"Skill {skill_id} not found in registry")

            # Check if skill is available
            if not self.skill_registry.is_skill_available(skill_id):
                raise ValueError(f"Skill {skill_id} is not available")

            # Check circuit breaker
            if self.circuit_breaker_enabled and skill_id in self.circuit_breakers:
                circuit_breaker = self.circuit_breakers[skill_id]
                if not self._can_execute_skill(circuit_breaker):
                    raise ValueError(f"Circuit breaker open for skill {skill_id}")

            # Check cache first
            if self.cache_enabled and skill_id in self.skill_cache:
                instance = self.skill_cache[skill_id]
                self.logger.debug(f"Retrieved skill {skill_id} from cache")
                self.metrics.increment("skill_factory.cache_hits")
                return instance

            async with self.initialization_lock:
                # Double-check cache after acquiring lock
                if self.cache_enabled and skill_id in self.skill_cache:
                    return self.skill_cache[skill_id]

                # Create skill instance
                skill_class = registration.skill_class
                instance = skill_class()

                # Inject dependencies
                await self._inject_dependencies(instance, skill_id)

                # Initialize skill
                skill_config = config or {}
                await instance.initialize(skill_config)

                # Store in active skills
                self.active_skills[skill_id] = instance

                # Cache if enabled
                if self.cache_enabled:
                    self.skill_cache[skill_id] = instance

                # Update registration
                registration.instance = instance
                registration.last_accessed = datetime.now(timezone.utc)
                registration.access_count += 1

                # Update skill state
                await self.skill_registry.update_skill_state(skill_id, SkillState.ACTIVE)

                # Update metrics
                execution_time = (time.time() - start_time) * 1000
                self.metrics.increment("skill_factory.creations.total")
                self.metrics.histogram("skill_factory.creation_time_ms", execution_time)

                # Fire event
                await self.event_bus.emit(
                    SkillCreated(
                        skill_id=skill_id,
                        skill_type=registration.metadata.skill_type,
                        execution_id=execution_id,
                    )
                )

                self.logger.info(f"Created skill instance: {skill_id}")
                return instance

        except Exception as e:
            # Update circuit breaker on failure
            if self.circuit_breaker_enabled and skill_id in self.circuit_breakers:
                await self._record_failure(skill_id, str(e))

            self.metrics.increment("skill_factory.creation_failures")
            self.logger.error(f"Failed to create skill {skill_id}: {str(e)}")
            raise

    @handle_exceptions
    async def execute_skill(
        self, skill_id: str, input_data: Any, context: Optional[SkillExecutionContext] = None
    ) -> SkillExecutionResult:
        """
        Execute a skill with comprehensive monitoring and error handling.

        Args:
            skill_id: Identifier of the skill to execute
            input_data: Input data for the skill
            context: Execution context

        Returns:
            Skill execution result
        """
        execution_id = str(uuid.uuid4())
        start_time = time.time()

        # Use default context if none provided
        if context is None:
            context = SkillExecutionContext()

        # Create execution result
        result = SkillExecutionResult(skill_id=skill_id, execution_id=execution_id, success=False)

        try:
            # Acquire execution semaphore
            async with self.execution_semaphore:
                # Fire execution started event
                await self.event_bus.emit(
                    SkillExecutionStarted(
                        skill_id=skill_id, execution_id=execution_id, context=context.__dict__
                    )
                )

                # Get or create skill instance
                if skill_id in self.active_skills:
                    instance = self.active_skills[skill_id]
                else:
                    instance = await self.create_skill(skill_id)

                # Validate input if skill supports it
                if hasattr(instance, "validate"):
                    if not await instance.validate(input_data):
                        raise ValueError("Input validation failed")

                # Build execution context
                exec_context = {
                    "session_id": context.session_id,
                    "user_id": context.user_id,
                    "correlation_id": context.correlation_id,
                    "execution_id": execution_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    **context.metadata,
                }

                # Execute skill with timeout
                try:
                    skill_result = await asyncio.wait_for(
                        instance.execute(input_data, exec_context), timeout=context.timeout_seconds
                    )

                    result.success = True
                    result.result = skill_result

                    # Record success for circuit breaker
                    if self.circuit_breaker_enabled and skill_id in self.circuit_breakers:
                        await self._record_success(skill_id)

                except asyncio.TimeoutError:
                    raise TimeoutError(
                        f"Skill execution timed out after {context.timeout_seconds}s"
                    )

                # Update performance stats
                execution_time = (time.time() - start_time) * 1000
                result.execution_time_ms = execution_time

                self._update_performance_stats(skill_id, execution_time, True)

                # Update metrics
                self.metrics.increment("skill_factory.executions.total")
                self.metrics.increment(f"skill_factory.executions.{skill_id}")
                self.metrics.histogram("skill_factory.execution_time_ms", execution_time)

                # Fire execution completed event
                await self.event_bus.emit(
                    SkillExecutionCompleted(
                        skill_id=skill_id,
                        execution_id=execution_id,
                        success=True,
                        execution_time_ms=execution_time,
                    )
                )

                self.logger.debug(
                    f"Executed skill {skill_id} successfully in {execution_time:.2f}ms"
                )

        except Exception as e:
            result.success = False
            result.error = str(e)
            result.execution_time_ms = (time.time() - start_time) * 1000

            # Record failure for circuit breaker
            if self.circuit_breaker_enabled and skill_id in self.circuit_breakers:
                await self._record_failure(skill_id, str(e))

            self._update_performance_stats(skill_id, result.execution_time_ms, False)

            # Update metrics
            self.metrics.increment("skill_factory.execution_failures")
            self.metrics.increment(f"skill_factory.execution_failures.{skill_id}")

            # Fire execution failed event
            await self.event_bus.emit(
                SkillExecutionFailed(
                    skill_id=skill_id,
                    execution_id=execution_id,
                    error=str(e),
                    retry_count=result.retry_count,
                )
            )

            self.logger.error(f"Skill execution failed for {skill_id}: {str(e)}")

            # Implement retry logic if configured
            if context.retry_count > 0 and result.retry_count < context.retry_count:
                self.logger.info(f"Retrying skill {skill_id}, attempt {result.retry_count + 1}")
                await asyncio.sleep(2**result.retry_count)  # Exponential backoff

                retry_context = SkillExecutionContext(
                    session_id=context.session_id,
                    user_id=context.user_id,
                    correlation_id=context.correlation_id,
                    execution_mode=context.execution_mode,
                    timeout_seconds=context.timeout_seconds,
                    cache_strategy=context.cache_strategy,
                    retry_count=context.retry_count,
                    circuit_breaker_enabled=context.circuit_breaker_enabled,
                    metadata=context.metadata,
                )

                result.retry_count += 1
                return await self.execute_skill(skill_id, input_data, retry_context)

        # Store execution history
        self.execution_history.append(result)

        # Update usage statistics
        self.skill_usage_stats[skill_id] += 1

        return result

    @handle_exceptions
    async def compose_skills(
        self,
        composition: SkillComposition,
        input_data: Any,
        context: Optional[SkillExecutionContext] = None,
    ) -> Any:
        """
        Execute a composition of skills in a specified order.

        Args:
            composition: Skill composition definition
            input_data: Initial input data
            context: Execution context

        Returns:
            Final result from the composition
        """
        if context is None:
            context = SkillExecutionContext()

        current_data = input_data
        results = {}

        try:
            for skill_id in composition.execution_order:
                if skill_id not in composition.skills:
                    raise ValueError(f"Skill {skill_id} not found in composition")

                # Execute skill
                execution_result = await self.execute_skill(skill_id, current_data, context)

                if not execution_result.success:
                    error_handling = composition.error_handling.get(skill_id, {})

                    if error_handling.get("continue_on_error", False):
                        self.logger.warning(
                            f"Skill {skill_id} failed but continuing: {execution_result.error}"
                        )
                        results[skill_id] = None
                        continue
                    else:
                        raise Exception(f"Skill {skill_id} failed: {execution_result.error}")

                results[skill_id] = execution_result.result

                # Apply data flow transformations
                if skill_id in composition.data_flow:
                    transformation = composition.data_flow[skill_id]
                    current_data = self._apply_data_transformation(
                        execution_result.result, transformation
                    )
                else:
                    current_data = execution_result.result

            return current_data

        except Exception as e:
            self.logger.error(f"Skill composition {composition.composition_id} failed: {str(e)}")
            raise

    @handle_exceptions
    async def hot_reload_skill(self, skill_id: str) -> bool:
        """
        Hot-reload a skill without stopping the system.

        Args:
            skill_id: Identifier of the skill to reload

        Returns:
            True if reload successful, False otherwise
        """
        if not self.hot_reload_enabled:
            self.logger.warning("Hot-reload is disabled")
            return False

        try:
            # Get current registration
            current_registration = self.skill_registry.get_skill(skill_id)
            if not current_registration:
                self.logger.error(f"Skill {skill_id} not found for hot-reload")
                return False

            old_version = current_registration.metadata.version

            # Remove from cache
            if skill_id in self.skill_cache:
                del self.skill_cache[skill_id]

            # Cleanup current instance
            if skill_id in self.active_skills:
                instance = self.active_skills[skill_id]
                if hasattr(instance, "cleanup"):
                    await instance.cleanup()
                del self.active_skills[skill_id]

            # Reload module
            module_name = current_registration.skill_class.__module__
            module = importlib.import_module(module_name)
            importlib.reload(module)

            # Re-register skill
            skill_class = getattr(module, current_registration.skill_class.__name__)

            # Validate reloaded skill
            temp_instance = skill_class()
            new_metadata = temp_instance.get_metadata()

            validation_report = await self.skill_validator.validate_skill(
                skill_id, skill_class, new_metadata
            )

            if not validation_report.is_valid:
                self.logger.error(f"Hot-reload validation failed for {skill_id}")
                return False

            # Update registration
            current_registration.skill_class = skill_class
            current_registration.metadata = new_metadata
            current_registration.instance = None

            # Fire hot-reload event
            await self.event_bus.emit(
                SkillHotReloaded(
                    skill_id=skill_id, old_version=old_version, new_version=new_metadata.version
                )
            )

            self.logger.info(f"Successfully hot-reloaded skill {skill_id}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to hot-reload skill {skill_id}: {str(e)}")
            return False

    @handle_exceptions
    async def create_skill_from_template(
        self, template_id: str, skill_name: str, variables: Dict[str, Any]
    ) -> str:
        """
        Create a new skill from a template.

        Args:
            template_id: Template identifier
            skill_name: Name for the new skill
            variables: Template variables

        Returns:
            ID of the created skill
        """
        if template_id not in self.skill_templates:
            raise ValueError(f"Template {template_id} not found")

        template = self.skill_templates[template_id]
        skill_id = f"custom.{skill_name.lower()}"

        try:
            # Create skill directory
            skill_dir = Path(f"src/skills/custom/{skill_name.lower()}")
            skill_dir.mkdir(parents=True, exist_ok=True)

            # Process template files
            for template_file in template.template_path.rglob("*.py"):
                relative_path = template_file.relative_to(template.template_path)
                target_file = skill_dir / relative_path

                # Read template content
                template_content = template_file.read_text()

                # Replace variables
                for var_name, var_value in variables.items():
                    template_content = template_content.replace(
                        f"{{{{ {var_name} }}}}", str(var_value)
                    )

                # Write processed file
                target_file.parent.mkdir(parents=True, exist_ok=True)
                target_file.write_text(template_content)

            self.logger.info(f"Created skill {skill_id} from template {template_id}")
            return skill_id

        except Exception as e:
            self.logger.error(f"Failed to create skill from template: {str(e)}")
            raise

    def _initialize_skill_templates(self):
        """Initialize built-in skill templates."""
        templates_dir = Path("src/skills/templates")
        if not templates_dir.exists():
            return

        for template_dir in templates_dir.iterdir():
            if template_dir.is_dir():
                config_file = template_dir / "template.json"
                if config_file.exists():
                    try:
                        config = json.loads(config_file.read_text())
                        template = SkillTemplate(
                            template_id=config["template_id"],
                            name=config["name"],
                            description=config["description"],
                            skill_type=SkillType(config["skill_type"]),
                            template_path=template_dir,
                            variables=config.get("variables", {}),
                            required_capabilities=config.get("required_capabilities", []),
                        )
                        self.skill_templates[template.template_id] = template

                    except Exception as e:
                        self.logger.warning(f"Failed to load template {template_dir}: {str(e)}")

    async def _discover_builtin_skills(self):
        """Discover and register built-in skills."""
        builtin_paths = self.config.get("skills.builtin_paths", ["src/skills/builtin"])
        discovered = await self.skill_registry.discover_skills(builtin_paths)
        self.logger.info(f"Discovered {discovered} built-in skills")

    async def _setup_hot_reload(self):
        """Setup hot-reload monitoring."""
        # This would typically use file system monitoring
        # For now, we'll just log that it's enabled
        self.logger.info("Hot-reload monitoring enabled")

    def _initialize_circuit_breaker(self, skill_id: str):
        """Initialize circuit breaker for a skill."""
        if self.circuit_breaker_enabled:
            self.circuit_breakers[skill_id] = CircuitBreaker(skill_id=skill_id)

    def _can_execute_skill(self, circuit_breaker: CircuitBreaker) -> bool:
        """Check if skill can be executed based on circuit breaker state."""
        if circuit_breaker.state == CircuitBreakerState.CLOSED:
            return True

        if circuit_breaker.state == CircuitBreakerState.OPEN:
            # Check if recovery timeout has passed
            if (
                circuit_breaker.last_failure_time
                and datetime.now(timezone.utc) - circuit_breaker.last_failure_time
                > circuit_breaker.recovery_timeout
            ):
                circuit_breaker.state = CircuitBreakerState.HALF_OPEN
                circuit_breaker.success_count = 0
                return True
            return False

        if circuit_breaker.state == CircuitBreakerState.HALF_OPEN:
            return True

        return False

    async def _record_success(self, skill_id: str):
        """Record successful execution for circuit breaker."""
        if skill_id not in self.circuit_breakers:
            return

        circuit_breaker = self.circuit_breakers[skill_id]

        if circuit_breaker.state == CircuitBreakerState.HALF_OPEN:
            circuit_breaker.success_count += 1
            if circuit_breaker.success_count >= circuit_breaker.success_threshold:
                circuit_breaker.state = CircuitBreakerState.CLOSED
                circuit_breaker.failure_count = 0

                await self.event_bus.emit(
                    CircuitBreakerTriggered(
                        skill_id=skill_id, state=CircuitBreakerState.CLOSED, failure_count=0
                    )
                )
        else:
            circuit_breaker.failure_count = 0

    async def _record_failure(self, skill_id: str, error: str):
        """Record failed execution for circuit breaker."""
        if skill_id not in self.circuit_breakers:
            return

        circuit_breaker = self.circuit_breakers[skill_id]
        circuit_breaker.failure_count += 1
        circuit_breaker.last_failure_time = datetime.now(timezone.utc)

        if (
            circuit_breaker.state == CircuitBreakerState.CLOSED
            and circuit_breaker.failure_count >= circuit_breaker.failure_threshold
        ):

            circuit_breaker.state = CircuitBreakerState.OPEN

            await self.event_bus.emit(
                CircuitBreakerTriggered(
                    skill_id=skill_id,
                    state=CircuitBreakerState.OPEN,
                    failure_count=circuit_breaker.failure_count,
                )
            )

        elif circuit_breaker.state == CircuitBreakerState.HALF_OPEN:
            circuit_breaker.state = CircuitBreakerState.OPEN
            circuit_breaker.success_count = 0

    async def _inject_dependencies(self, instance: Any, skill_id: str):
        """Inject dependencies into skill instance."""
        # Check if skill implements dependency injection
        if hasattr(instance, "set_dependencies"):
            dependencies = {
                "container": self.container,
                "config": self.config,
                "event_bus": self.event_bus,
                "logger": get_logger(f"skill.{skill_id}"),
                "metrics": self.metrics,
                "memory_manager": self.memory_manager,
                "context_manager": self.context_manager,
                "session_manager": self.session_manager,
            }

            await instance.set_dependencies(dependencies)

    def _update_performance_stats(self, skill_id: str, execution_time: float, success: bool):
        """Update performance statistics for a skill."""
        if skill_id not in self.skill_performance:
            self.skill_performance[skill_id] = {
                "total_executions": 0,
                "successful_executions": 0,
                "total_time_ms": 0,
                "average_time_ms": 0,
                "min_time_ms": float("inf"),
                "max_time_ms": 0,
                "success_rate": 0,
            }

        stats = self.skill_performance[skill_id]
        stats["total_executions"] += 1

        if success:
            stats["successful_executions"] += 1

        stats["total_time_ms"] += execution_time
        stats["average_time_ms"] = stats["total_time_ms"] / stats["total_executions"]
        stats["min_time_ms"] = min(stats["min_time_ms"], execution_time)
        stats["max_time_ms"] = max(stats["max_time_ms"], execution_time)
        stats["success_rate"] = (stats["successful_executions"] / stats["total_executions"]) * 100

    def _apply_data_transformation(self, data: Any, transformation: Dict[str, Any]) -> Any:
        """Apply data transformation in skill composition."""
        # Simple transformation implementation
        transform_type = transformation.get("type", "passthrough")

        if transform_type == "passthrough":
            return data
        elif transform_type == "extract_field":
            field_name = transformation.get("field")
            if isinstance(data, dict) and field_name in data:
                return data[field_name]
        elif transform_type == "wrap":
            wrapper_key = transformation.get("key", "data")
            return {wrapper_key: data}

        return data

    async def _cleanup_all_skills(self):
        """Cleanup all active skill instances."""
        for skill_id, instance in self.active_skills.items():
            try:
                if hasattr(instance, "cleanup"):
                    await instance.cleanup()
            except Exception as e:
                self.logger.error(f"Error cleaning up skill {skill_id}: {str(e)}")

        self.active_skills.clear()
        self.skill_cache.clear()

    # Public API methods
    def get_skill_performance_stats(self, skill_id: Optional[str] = None) -> Dict[str, Any]:
        """Get performance statistics for skills."""
        if skill_id:
            return self.skill_performance.get(skill_id, {})
        return dict(self.skill_performance)

    def get_execution_history(
        self, skill_id: Optional[str] = None, limit: int = 100
    ) -> List[SkillExecutionResult]:
        """Get execution history."""
        history = list(self.execution_history)

        if skill_id:
            history = [r for r in history if r.skill_id == skill_id]

        return history[-limit:]

    def get_circuit_breaker_status(self, skill_id: Optional[str] = None) -> Dict[str, Any]:
        """Get circuit breaker status."""
        if skill_id:
            cb = self.circuit_breakers.get(skill_id)
            return cb.__dict__ if cb else {}

        return {skill_id: cb.__dict__ for skill_id, cb in self.circuit_breakers.items()}

    def get_factory_statistics(self) -> Dict[str, Any]:
        """Get comprehensive factory statistics."""
        return {
            "active_skills": len(self.active_skills),
            "cached_skills": len(self.skill_cache),
            "total_executions": len(self.execution_history),
            "skill_usage_stats": dict(self.skill_usage_stats),
            "circuit_breaker_summary": {
                "total": len(self.circuit_breakers),
                "closed": sum(
                    1
                    for cb in self.circuit_breakers.values()
                    if cb.state == CircuitBreakerState.CLOSED
                ),
                "open": sum(
                    1
                    for cb in self.circuit_breakers.values()
                    if cb.state == CircuitBreakerState.OPEN
                ),
                "half_open": sum(
                    1
                    for cb in self.circuit_breakers.values()
                    if cb.state == CircuitBreakerState.HALF_OPEN
                ),
            },
            "templates_available": len(self.skill_templates),
            "health_status": self.is_healthy,
        }
