"""
Advanced Skill Registry for AI Assistant
Author: Drmusab
Last Modified: 2025-01-20 10:30:00 UTC

This module provides comprehensive skill registration and management capabilities,
including skill discovery, versioning, dependency resolution, and lifecycle management.
"""

import importlib
import inspect
import json
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Type, Union

import asyncio

# Core imports
from src.core.config.loader import ConfigLoader
from src.core.dependency_injection import Container
from src.core.error_handling import ErrorHandler, handle_exceptions
from src.core.events.event_bus import EventBus
from src.core.events.event_types import BaseEvent, EventCategory, EventPriority, EventSeverity
from src.core.health_check import HealthCheck
from src.observability.logging.config import get_logger

# Observability
from src.observability.monitoring.metrics import MetricsCollector
from src.observability.monitoring.tracing import TraceManager


class SkillType(Enum):
    """Types of skills supported by the system."""

    BUILTIN = "builtin"  # Built-in system skills
    CUSTOM = "custom"  # User-defined custom skills
    META = "meta"  # Meta-skills that compose other skills
    EXTERNAL = "external"  # External service skills
    TEMPLATE = "template"  # Template skills for scaffolding


class SkillState(Enum):
    """Skill lifecycle states."""

    UNREGISTERED = "unregistered"
    REGISTERED = "registered"
    VALIDATED = "validated"
    INITIALIZED = "initialized"
    ACTIVE = "active"
    INACTIVE = "inactive"
    FAILED = "failed"
    DEPRECATED = "deprecated"


@dataclass
class SkillCapability:
    """Represents a specific capability of a skill."""

    name: str
    description: str
    input_types: List[str]
    output_types: List[str]
    required_resources: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SkillMetadata:
    """Comprehensive metadata for a skill."""

    skill_id: str
    name: str
    version: str
    description: str
    author: str
    skill_type: SkillType
    capabilities: List[SkillCapability]
    dependencies: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    min_system_version: Optional[str] = None
    max_system_version: Optional[str] = None
    configuration_schema: Optional[Dict[str, Any]] = None
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    security_level: str = "standard"
    resource_requirements: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SkillRegistration:
    """Represents a registered skill in the system."""

    skill_id: str
    skill_class: Type
    metadata: SkillMetadata
    state: SkillState = SkillState.REGISTERED
    instance: Optional[Any] = None
    registration_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_accessed: Optional[datetime] = None
    access_count: int = 0
    performance_stats: Dict[str, Any] = field(default_factory=dict)
    error_count: int = 0
    last_error: Optional[str] = None


# Event types for skill registry
@dataclass
class SkillRegistered(BaseEvent):
    """Event fired when a skill is registered."""

    skill_id: str
    skill_name: str
    skill_type: SkillType
    version: str
    category: EventCategory = EventCategory.SKILL


@dataclass
class SkillUnregistered(BaseEvent):
    """Event fired when a skill is unregistered."""

    skill_id: str
    skill_name: str
    reason: str
    category: EventCategory = EventCategory.SKILL


@dataclass
class SkillStateChanged(BaseEvent):
    """Event fired when a skill state changes."""

    skill_id: str
    old_state: SkillState
    new_state: SkillState
    reason: Optional[str] = None
    category: EventCategory = EventCategory.SKILL


class SkillInterface(ABC):
    """Base interface that all skills must implement."""

    @abstractmethod
    def get_metadata(self) -> SkillMetadata:
        """Get skill metadata."""
        pass

    @abstractmethod
    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the skill."""
        pass

    @abstractmethod
    async def execute(self, input_data: Any, context: Dict[str, Any]) -> Any:
        """Execute the skill."""
        pass

    async def cleanup(self) -> None:
        """Cleanup skill resources."""
        pass

    async def validate(self, input_data: Any) -> bool:
        """Validate input data."""
        return True

    async def health_check(self) -> Dict[str, Any]:
        """Check skill health."""
        return {"status": "healthy"}


class SkillRegistry:
    """
    Comprehensive skill registry for managing skill lifecycle and discovery.

    Features:
    - Skill registration and discovery
    - Versioning and compatibility checking
    - Dependency resolution
    - Performance monitoring
    - Hot-reload capabilities
    - Skill templates and scaffolding
    """

    def __init__(self, container: Container):
        """Initialize the skill registry."""
        self.container = container
        self.logger = get_logger(__name__)
        self.config = container.get(ConfigLoader)
        self.event_bus = container.get(EventBus)
        self.error_handler = container.get(ErrorHandler)
        self.health_check = container.get(HealthCheck)

        # Observability
        self.metrics = container.get(MetricsCollector)
        self.tracer = container.get(TraceManager)

        # State management
        self.skills: Dict[str, SkillRegistration] = {}
        self.skills_by_type: Dict[SkillType, List[str]] = defaultdict(list)
        self.skills_by_capability: Dict[str, List[str]] = defaultdict(list)
        self.skill_dependencies: Dict[str, Set[str]] = {}
        self.skill_dependents: Dict[str, Set[str]] = {}

        # Performance tracking
        self.skill_performance: Dict[str, Dict[str, Any]] = {}
        self.skill_usage_history: deque = deque(maxlen=10000)

        # Configuration
        self.auto_discovery_enabled = self.config.get("skills.auto_discovery", True)
        self.skill_paths = self.config.get("skills.paths", [])
        self.max_concurrent_initializations = self.config.get("skills.max_concurrent_init", 10)

        # Health tracking
        self.is_healthy = True
        self.last_health_check = datetime.now(timezone.utc)

        self.logger.info("SkillRegistry initialized successfully")

    @handle_exceptions
    async def register_skill(
        self, skill_id: str, skill_class: Type, metadata: Optional[SkillMetadata] = None
    ) -> bool:
        """
        Register a skill with the registry.

        Args:
            skill_id: Unique identifier for the skill
            skill_class: The skill class to register
            metadata: Optional metadata, will be extracted from skill if not provided

        Returns:
            True if registration successful, False otherwise
        """
        try:
            # Extract metadata if not provided
            if metadata is None:
                if hasattr(skill_class, "get_metadata"):
                    temp_instance = skill_class()
                    metadata = temp_instance.get_metadata()
                else:
                    raise ValueError(f"Skill {skill_id} must provide metadata")

            # Validate skill interface
            if not self._validate_skill_interface(skill_class):
                raise ValueError(f"Skill {skill_id} does not implement required interface")

            # Check for existing registration
            if skill_id in self.skills:
                existing = self.skills[skill_id]
                if existing.metadata.version == metadata.version:
                    self.logger.warning(
                        f"Skill {skill_id} version {metadata.version} already registered"
                    )
                    return False
                else:
                    self.logger.info(
                        f"Updating skill {skill_id} from {existing.metadata.version} to {metadata.version}"
                    )

            # Create registration
            registration = SkillRegistration(
                skill_id=skill_id, skill_class=skill_class, metadata=metadata
            )

            # Store registration
            self.skills[skill_id] = registration
            self.skills_by_type[metadata.skill_type].append(skill_id)

            # Index capabilities
            for capability in metadata.capabilities:
                self.skills_by_capability[capability.name].append(skill_id)

            # Process dependencies
            self._process_skill_dependencies(skill_id, metadata.dependencies)

            # Update metrics
            self.metrics.increment("skill_registry.registrations.total")
            self.metrics.increment(f"skill_registry.registrations.{metadata.skill_type.value}")

            # Fire event
            await self.event_bus.emit(
                SkillRegistered(
                    skill_id=skill_id,
                    skill_name=metadata.name,
                    skill_type=metadata.skill_type,
                    version=metadata.version,
                )
            )

            self.logger.info(f"Successfully registered skill: {skill_id} v{metadata.version}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to register skill {skill_id}: {str(e)}")
            self.metrics.increment("skill_registry.registration_failures.total")
            return False

    @handle_exceptions
    async def unregister_skill(self, skill_id: str, reason: str = "manual") -> bool:
        """Unregister a skill from the registry."""
        try:
            if skill_id not in self.skills:
                self.logger.warning(f"Skill {skill_id} not found for unregistration")
                return False

            registration = self.skills[skill_id]

            # Cleanup instance if exists
            if registration.instance:
                try:
                    if hasattr(registration.instance, "cleanup"):
                        await registration.instance.cleanup()
                except Exception as e:
                    self.logger.error(f"Error cleaning up skill {skill_id}: {str(e)}")

            # Remove from indices
            self.skills_by_type[registration.metadata.skill_type].remove(skill_id)

            for capability in registration.metadata.capabilities:
                if skill_id in self.skills_by_capability[capability.name]:
                    self.skills_by_capability[capability.name].remove(skill_id)

            # Remove dependencies
            if skill_id in self.skill_dependencies:
                del self.skill_dependencies[skill_id]

            # Remove from dependents
            for dependent_set in self.skill_dependents.values():
                dependent_set.discard(skill_id)

            # Remove registration
            del self.skills[skill_id]

            # Update metrics
            self.metrics.increment("skill_registry.unregistrations.total")

            # Fire event
            await self.event_bus.emit(
                SkillUnregistered(
                    skill_id=skill_id, skill_name=registration.metadata.name, reason=reason
                )
            )

            self.logger.info(f"Successfully unregistered skill: {skill_id}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to unregister skill {skill_id}: {str(e)}")
            return False

    def get_skill(self, skill_id: str) -> Optional[SkillRegistration]:
        """Get a skill registration by ID."""
        return self.skills.get(skill_id)

    def get_skills_by_type(self, skill_type: SkillType) -> List[SkillRegistration]:
        """Get all skills of a specific type."""
        skill_ids = self.skills_by_type.get(skill_type, [])
        return [self.skills[skill_id] for skill_id in skill_ids]

    def get_skills_by_capability(self, capability_name: str) -> List[SkillRegistration]:
        """Get all skills that provide a specific capability."""
        skill_ids = self.skills_by_capability.get(capability_name, [])
        return [self.skills[skill_id] for skill_id in skill_ids]

    def search_skills(
        self, query: str, filters: Optional[Dict[str, Any]] = None
    ) -> List[SkillRegistration]:
        """Search skills based on query and filters."""
        results = []

        for skill_id, registration in self.skills.items():
            metadata = registration.metadata

            # Text search
            if (
                query.lower() in metadata.name.lower()
                or query.lower() in metadata.description.lower()
                or any(query.lower() in tag.lower() for tag in metadata.tags)
            ):

                # Apply filters
                if filters:
                    if "skill_type" in filters and metadata.skill_type != filters["skill_type"]:
                        continue
                    if "min_version" in filters and metadata.version < filters["min_version"]:
                        continue
                    if "capabilities" in filters:
                        required_caps = set(filters["capabilities"])
                        available_caps = set(cap.name for cap in metadata.capabilities)
                        if not required_caps.issubset(available_caps):
                            continue

                results.append(registration)

        return results

    async def discover_skills(self, paths: Optional[List[str]] = None) -> int:
        """
        Discover and register skills from specified paths.

        Args:
            paths: List of paths to search for skills

        Returns:
            Number of skills discovered and registered
        """
        if not paths:
            paths = self.skill_paths

        discovered_count = 0

        for path_str in paths:
            path = Path(path_str)
            if not path.exists():
                self.logger.warning(f"Skill discovery path does not exist: {path}")
                continue

            try:
                count = await self._discover_skills_in_path(path)
                discovered_count += count
                self.logger.info(f"Discovered {count} skills in {path}")
            except Exception as e:
                self.logger.error(f"Error discovering skills in {path}: {str(e)}")

        self.metrics.set("skill_registry.total_skills", len(self.skills))
        return discovered_count

    async def _discover_skills_in_path(self, path: Path) -> int:
        """Discover skills in a specific path."""
        count = 0

        for python_file in path.rglob("*.py"):
            if python_file.name.startswith("__"):
                continue

            try:
                # Convert path to module name
                relative_path = python_file.relative_to(Path.cwd())
                module_name = str(relative_path).replace("/", ".").replace("\\", ".")[:-3]

                # Import module
                module = importlib.import_module(module_name)

                # Find skill classes
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if (
                        issubclass(obj, SkillInterface)
                        and obj != SkillInterface
                        and hasattr(obj, "get_metadata")
                    ):

                        # Try to register the skill
                        skill_id = f"{module_name}.{name}"
                        if await self.register_skill(skill_id, obj):
                            count += 1

            except Exception as e:
                self.logger.debug(f"Could not process {python_file}: {str(e)}")

        return count

    def _validate_skill_interface(self, skill_class: Type) -> bool:
        """Validate that a skill class implements the required interface."""
        required_methods = ["get_metadata", "initialize", "execute"]

        for method_name in required_methods:
            if not hasattr(skill_class, method_name):
                return False

            method = getattr(skill_class, method_name)
            if not callable(method):
                return False

        return True

    def _process_skill_dependencies(self, skill_id: str, dependencies: List[str]) -> None:
        """Process and store skill dependencies."""
        if dependencies:
            self.skill_dependencies[skill_id] = set(dependencies)

            # Update dependents
            for dep_id in dependencies:
                if dep_id not in self.skill_dependents:
                    self.skill_dependents[dep_id] = set()
                self.skill_dependents[dep_id].add(skill_id)

    async def get_skill_statistics(self) -> Dict[str, Any]:
        """Get comprehensive skill registry statistics."""
        stats = {
            "total_skills": len(self.skills),
            "skills_by_type": {
                skill_type.value: len(skill_ids)
                for skill_type, skill_ids in self.skills_by_type.items()
            },
            "skills_by_state": defaultdict(int),
            "total_capabilities": len(self.skills_by_capability),
            "dependency_graph_size": len(self.skill_dependencies),
            "performance_metrics": self.skill_performance,
            "health_status": self.is_healthy,
            "last_health_check": self.last_health_check.isoformat(),
        }

        # Count skills by state
        for registration in self.skills.values():
            stats["skills_by_state"][registration.state.value] += 1

        return stats

    async def health_check_all_skills(self) -> Dict[str, Any]:
        """Perform health check on all registered skills."""
        health_results = {}

        for skill_id, registration in self.skills.items():
            try:
                if registration.instance and hasattr(registration.instance, "health_check"):
                    health_results[skill_id] = await registration.instance.health_check()
                else:
                    health_results[skill_id] = {"status": "not_initialized"}

            except Exception as e:
                health_results[skill_id] = {"status": "error", "error": str(e)}
                registration.error_count += 1
                registration.last_error = str(e)

        self.last_health_check = datetime.now(timezone.utc)
        return health_results

    def get_skill_dependencies(self, skill_id: str) -> Set[str]:
        """Get dependencies for a specific skill."""
        return self.skill_dependencies.get(skill_id, set())

    def get_skill_dependents(self, skill_id: str) -> Set[str]:
        """Get skills that depend on a specific skill."""
        return self.skill_dependents.get(skill_id, set())

    def is_skill_available(self, skill_id: str) -> bool:
        """Check if a skill is available for use."""
        registration = self.skills.get(skill_id)
        return registration is not None and registration.state == SkillState.ACTIVE

    async def update_skill_state(
        self, skill_id: str, new_state: SkillState, reason: Optional[str] = None
    ) -> bool:
        """Update the state of a skill."""
        if skill_id not in self.skills:
            return False

        registration = self.skills[skill_id]
        old_state = registration.state

        if old_state == new_state:
            return True

        registration.state = new_state

        # Fire event
        await self.event_bus.emit(
            SkillStateChanged(
                skill_id=skill_id, old_state=old_state, new_state=new_state, reason=reason
            )
        )

        self.logger.info(f"Updated skill {skill_id} state: {old_state.value} -> {new_state.value}")
        return True
