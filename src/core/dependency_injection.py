"""
Advanced Dependency Injection Container
Author: Drmusab
Last Modified: 2025-06-13 12:38:46 UTC

This module provides a comprehensive dependency injection container for the AI assistant,
featuring type-safe resolution, lifecycle management, circular dependency detection,
auto-wiring, factory patterns, and seamless integration with all core components.
"""

from pathlib import Path
from typing import (
    Optional, Dict, Any, List, Set, Callable, Type, Union, TypeVar, Generic,
    get_type_hints, get_origin, get_args, ForwardRef, cast, overload
)
import asyncio
import threading
import time
import inspect
import weakref
from datetime import datetime, timezone
from dataclasses import dataclass, field
from enum import Enum
from contextlib import asynccontextmanager, contextmanager
import logging
import uuid
import json
import yaml
from collections import defaultdict, deque
from abc import ABC, abstractmethod
import importlib
import sys




# Type definitions
T = TypeVar('T')
Factory = Callable[..., T]
AsyncFactory = Callable[..., Any]  # Returns awaitable
async def register_plugin_manager_update(container: Container):
    """Register the enhanced plugin manager."""
    from src.assistant.plugin_manager_enhanced import EnhancedPluginManager
    
    # Register as singleton
    container.register_singleton(EnhancedPluginManager)
    
    # Also register with the original PluginManager interface for compatibility
    container.register_alias('PluginManager', EnhancedPluginManager)

class LifecycleScope(Enum):
    """Dependency lifecycle scopes."""
    SINGLETON = "singleton"        # Single instance for entire application
    TRANSIENT = "transient"        # New instance every time
    SCOPED = "scoped"             # Single instance per scope (e.g., request/session)
    THREAD_LOCAL = "thread_local"  # Single instance per thread
    LAZY_SINGLETON = "lazy_singleton"  # Singleton created on first access


class RegistrationStrategy(Enum):
    """Dependency registration strategies."""
    TYPE_BASED = "type_based"      # Register by type
    NAME_BASED = "name_based"      # Register by name
    INTERFACE_BASED = "interface_based"  # Register by interface
    FACTORY_BASED = "factory_based"  # Register using factory function


class ResolutionMode(Enum):
    """Dependency resolution modes."""
    STRICT = "strict"              # Fail if dependency not found
    OPTIONAL = "optional"          # Return None if dependency not found
    LAZY = "lazy"                  # Create lazy proxy
    AUTO_WIRE = "auto_wire"        # Automatically resolve constructor dependencies


@dataclass
class DependencyMetadata:
    """Metadata for registered dependencies."""
    registration_id: str
    registered_type: Type
    implementation_type: Optional[Type] = None
    instance: Optional[Any] = None
    factory: Optional[Callable] = None
    scope: LifecycleScope = LifecycleScope.SINGLETON
    strategy: RegistrationStrategy = RegistrationStrategy.TYPE_BASED
    
    # Registration metadata
    name: Optional[str] = None
    tags: Set[str] = field(default_factory=set)
    priority: int = 0
    condition: Optional[Callable[[], bool]] = None
    
    # Lifecycle tracking
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_accessed: Optional[datetime] = None
    access_count: int = 0
    
    # Dependencies
    dependencies: List[str] = field(default_factory=list)
    dependents: Set[str] = field(default_factory=set)
    
    # Configuration
    config_section: Optional[str] = None
    initialization_params: Dict[str, Any] = field(default_factory=dict)
    
    # Validation
    is_valid: bool = True
    validation_errors: List[str] = field(default_factory=list)


@dataclass
class ResolutionContext:
    """Context for dependency resolution."""
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    scope_id: Optional[str] = None
    resolution_chain: List[Type] = field(default_factory=list)
    mode: ResolutionMode = ResolutionMode.STRICT
    tags: Set[str] = field(default_factory=set)
    parameters: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class DependencyError(Exception):
    """Base exception for dependency injection errors."""
    
    def __init__(self, message: str, dependency_type: Optional[Type] = None,
                 context: Optional[ResolutionContext] = None):
        super().__init__(message)
        self.dependency_type = dependency_type
        self.context = context
        self.timestamp = datetime.now(timezone.utc)


class CircularDependencyError(DependencyError):
    """Exception raised when circular dependencies are detected."""
    
    def __init__(self, cycle: List[Type], context: Optional[ResolutionContext] = None):
        self.cycle = cycle
        cycle_names = " -> ".join(t.__name__ for t in cycle)
        super().__init__(f"Circular dependency detected: {cycle_names}", context=context)


class MissingDependencyError(DependencyError):
    """Exception raised when a required dependency is not registered."""
    pass


class InvalidRegistrationError(DependencyError):
    """Exception raised when registration is invalid."""
    pass


class LazyProxy(Generic[T]):
    """Lazy proxy for deferred dependency resolution."""
    
    def __init__(self, container: 'Container', dependency_type: Type[T],
                 context: Optional[ResolutionContext] = None):
        self._container = container
        self._dependency_type = dependency_type
        self._context = context or ResolutionContext()
        self._instance: Optional[T] = None
        self._resolved = False
        self._lock = threading.Lock()
    
    def _resolve(self) -> T:
        """Resolve the actual dependency."""
        if not self._resolved:
            with self._lock:
                if not self._resolved:
                    self._instance = self._container.get(self._dependency_type, self._context)
                    self._resolved = True
        return self._instance
    
    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the resolved instance."""
        return getattr(self._resolve(), name)
    
    def __call__(self, *args, **kwargs) -> Any:
        """Delegate calls to the resolved instance."""
        return self._resolve()(*args, **kwargs)
    
    def __repr__(self) -> str:
        return f"LazyProxy[{self._dependency_type.__name__}]"


class ScopedContainer:
    """Container for scoped dependencies."""
    
    def __init__(self, scope_id: str, parent_container: 'Container'):
        self.scope_id = scope_id
        self.parent_container = parent_container
        self.scoped_instances: Dict[str, Any] = {}
        self.created_at = datetime.now(timezone.utc)
        self.access_count = 0
        self._lock = threading.Lock()
    
    def get_scoped_instance(self, registration_id: str) -> Optional[Any]:
        """Get a scoped instance."""
        with self._lock:
            return self.scoped_instances.get(registration_id)
    
    def set_scoped_instance(self, registration_id: str, instance: Any) -> None:
        """Set a scoped instance."""
        with self._lock:
            self.scoped_instances[registration_id] = instance
            self.access_count += 1
    
    def clear(self) -> None:
        """Clear all scoped instances."""
        with self._lock:
            # Call cleanup methods if available
            for instance in self.scoped_instances.values():
                if hasattr(instance, 'cleanup') and callable(instance.cleanup):
                    try:
                        if asyncio.iscoroutinefunction(instance.cleanup):
                            # Schedule cleanup for later
                            asyncio.create_task(instance.cleanup())
                        else:
                            instance.cleanup()
                    except Exception as e:
                        # Log but don't fail
                        pass
            
            self.scoped_instances.clear()


class DependencyValidator:
    """Validates dependency registrations and resolutions."""
    
    def __init__(self, logger):
        self.logger = logger
    
    def validate_registration(self, metadata: DependencyMetadata) -> List[str]:
        """Validate a dependency registration."""
        errors = []
        
        # Check required fields
        if not metadata.registered_type:
            errors.append("Registered type is required")
        
        # Validate scope and strategy compatibility
        if metadata.scope == LifecycleScope.SCOPED and metadata.strategy == RegistrationStrategy.FACTORY_BASED:
            if not metadata.factory:
                errors.append("Factory function required for factory-based scoped registration")
        
        # Validate factory function
        if metadata.factory:
            try:
                sig = inspect.signature(metadata.factory)
                # Factory should be callable
                if not callable(metadata.factory):
                    errors.append("Factory must be callable")
            except Exception as e:
                errors.append(f"Invalid factory function: {str(e)}")
        
        # Validate implementation type
        if metadata.implementation_type:
            if not inspect.isclass(metadata.implementation_type):
                errors.append("Implementation type must be a class")
            
            # Check if implementation implements registered type
            if (hasattr(metadata.registered_type, '__origin__') or 
                not issubclass(metadata.implementation_type, metadata.registered_type)):
                # Allow for generic types and interfaces
                pass
        
        return errors
    
    def validate_resolution_chain(self, chain: List[Type]) -> Optional[List[Type]]:
        """Check for circular dependencies in resolution chain."""
        seen = set()
        for dep_type in chain:
            if dep_type in seen:
                # Find the cycle
                cycle_start = chain.index(dep_type)
                return chain[cycle_start:] + [dep_type]
            seen.add(dep_type)
        return None


class ContainerConfiguration:
    """Configuration for the dependency injection container."""
    
    def __init__(self):
        self.auto_wire_enabled = True
        self.strict_mode = True
        self.lazy_loading_enabled = True
        self.circular_dependency_detection = True
        self.performance_monitoring = True
        self.validation_enabled = True
        self.thread_safe = True
        self.default_scope = LifecycleScope.SINGLETON
        self.registration_validation = True
        self.hot_swapping_enabled = False
        self.plugin_integration_enabled = True
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ContainerConfiguration':
        """Create configuration from dictionary."""
        config = cls()
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config


class Container:
    """
    Advanced Dependency Injection Container.
    
    Features:
    - Type-safe dependency resolution with generics
    - Multiple lifecycle scopes (singleton, transient, scoped)
    - Circular dependency detection and resolution
    - Auto-wiring based on type annotations
    - Factory pattern support for complex object creation
    - Conditional registration based on configuration
    - Lazy initialization for performance optimization
    - Scoped containers for session-specific dependencies
    - Hot-swapping for development and testing
    - Configuration-driven registration from YAML/JSON
    - Plugin system integration for dynamic registration
    - Health monitoring of registered components
    - Performance tracking and optimization
    """
    
    def __init__(self, config: Optional[ContainerConfiguration] = None, logger=None):
        """
        Initialize the dependency injection container.
        
        Args:
            config: Container configuration
            logger: Logger instance
        """
        self.config = config or ContainerConfiguration()
        self.logger = logger or self._setup_default_logger()
        
        # Core registrations storage
        self._registrations: Dict[str, DependencyMetadata] = {}
        self._type_mappings: Dict[Type, str] = {}
        self._name_mappings: Dict[str, str] = {}
        self._interface_mappings: Dict[Type, List[str]] = defaultdict(list)
        
        # Instance management
        self._singleton_instances: Dict[str, Any] = {}
        self._scoped_containers: Dict[str, ScopedContainer] = {}
        self._thread_local_instances = threading.local()
        
        # Resolution tracking
        self._resolution_stats: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self._performance_metrics: Dict[str, List[float]] = defaultdict(list)
        
        # Validation and health
        self._validator = DependencyValidator(self.logger)
        self._health_status: Dict[str, bool] = {}
        
        # Thread safety
        self._lock = threading.RLock() if self.config.thread_safe else None
        
        # Hot-swapping support
        self._original_registrations: Dict[str, DependencyMetadata] = {}
        
        # Plugin integration
        self._plugin_registrations: Set[str] = set()
        
        self.logger.info("Dependency injection container initialized")

    def _setup_default_logger(self):
        """Setup default logger if none provided."""
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    def _with_lock(self, func: Callable) -> Callable:
        """Decorator to ensure thread safety."""
        if not self.config.thread_safe:
            return func
        
        def wrapper(*args, **kwargs):
            with self._lock:
                return func(*args, **kwargs)
        return wrapper

    def register(
        self,
        registered_type: Type[T],
        implementation: Optional[Union[Type[T], T, Factory[T]]] = None,
        scope: LifecycleScope = None,
        name: Optional[str] = None,
        tags: Optional[Set[str]] = None,
        condition: Optional[Callable[[], bool]] = None,
        **kwargs
    ) -> 'Container':
        """
        Register a dependency in the container.
        
        Args:
            registered_type: The type to register
            implementation: Implementation type, instance, or factory function
            scope: Lifecycle scope for the dependency
            name: Optional name for named registration
            tags: Optional tags for categorization
            condition: Optional condition for conditional registration
            **kwargs: Additional configuration parameters
            
        Returns:
            Self for method chaining
        """
        registration_id = self._generate_registration_id(registered_type, name)
        scope = scope or self.config.default_scope
        tags = tags or set()
        
        # Determine registration strategy and metadata
        metadata = DependencyMetadata(
            registration_id=registration_id,
            registered_type=registered_type,
            scope=scope,
            name=name,
            tags=tags,
            condition=condition,
            **kwargs
        )
        
        # Handle different implementation types
        if implementation is None:
            # Register type as itself
            metadata.implementation_type = registered_type
            metadata.strategy = RegistrationStrategy.TYPE_BASED
        elif inspect.isclass(implementation):
            # Register implementation type
            metadata.implementation_type = implementation
            metadata.strategy = RegistrationStrategy.TYPE_BASED
        elif callable(implementation):
            # Register factory function
            metadata.factory = implementation
            metadata.strategy = RegistrationStrategy.FACTORY_BASED
        else:
            # Register instance
            metadata.instance = implementation
            metadata.strategy = RegistrationStrategy.TYPE_BASED
            metadata.scope = LifecycleScope.SINGLETON  # Instances are always singleton
        
        return self._register_metadata(metadata)

    def _generate_registration_id(self, registered_type: Type, name: Optional[str]) -> str:
        """Generate a unique registration ID."""
        type_name = getattr(registered_type, '__name__', str(registered_type))
        if name:
            return f"{type_name}#{name}"
        return type_name

    @_with_lock
    def _register_metadata(self, metadata: DependencyMetadata) -> 'Container':
        """Register dependency metadata."""
        # Validate registration
        if self.config.registration_validation:
            errors = self._validator.validate_registration(metadata)
            if errors:
                metadata.is_valid = False
                metadata.validation_errors = errors
                raise InvalidRegistrationError(
                    f"Invalid registration for {metadata.registered_type}: {', '.join(errors)}"
                )
        
        # Store registration
        self._registrations[metadata.registration_id] = metadata
        
        # Update mappings
        self._type_mappings[metadata.registered_type] = metadata.registration_id
        
        if metadata.name:
            self._name_mappings[metadata.name] = metadata.registration_id
        
        # Handle interface mappings
        if metadata.implementation_type:
            for base in inspect.getmro(metadata.implementation_type)[1:]:  # Skip self
                self._interface_mappings[base].append(metadata.registration_id)
        
        # Log registration
        self.logger.debug(
            f"Registered {metadata.registered_type.__name__} "
            f"with scope {metadata.scope.value} and strategy {metadata.strategy.value}"
        )
        
        return self

    def register_singleton(self, registered_type: Type[T], 
                          implementation: Optional[Union[Type[T], T]] = None,
                          **kwargs) -> 'Container':
        """Register a singleton dependency."""
        return self.register(registered_type, implementation, LifecycleScope.SINGLETON, **kwargs)

    def register_transient(self, registered_type: Type[T],
                          implementation: Optional[Union[Type[T], Factory[T]]] = None,
                          **kwargs) -> 'Container':
        """Register a transient dependency."""
        return self.register(registered_type, implementation, LifecycleScope.TRANSIENT, **kwargs)

    def register_scoped(self, registered_type: Type[T],
                       implementation: Optional[Union[Type[T], Factory[T]]] = None,
                       **kwargs) -> 'Container':
        """Register a scoped dependency."""
        return self.register(registered_type, implementation, LifecycleScope.SCOPED, **kwargs)

    def register_factory(self, registered_type: Type[T], factory: Factory[T],
                        scope: LifecycleScope = None, **kwargs) -> 'Container':
        """Register a factory function."""
        return self.register(registered_type, factory, scope or LifecycleScope.TRANSIENT, **kwargs)

    def register_instance(self, registered_type: Type[T], instance: T,
                         **kwargs) -> 'Container':
        """Register a specific instance."""
        return self.register(registered_type, instance, LifecycleScope.SINGLETON, **kwargs)

    def register_conditional(self, registered_type: Type[T],
                           implementation: Union[Type[T], Factory[T]],
                           condition: Callable[[], bool],
                           **kwargs) -> 'Container':
        """Register a conditional dependency."""
        return self.register(registered_type, implementation, condition=condition, **kwargs)

    @overload
    def get(self, dependency_type: Type[T]) -> T:
        ...

    @overload
    def get(self, dependency_type: Type[T], context: ResolutionContext) -> T:
        ...

    @overload
    def get(self, dependency_type: Type[T], default: T) -> T:
        ...

    def get(self, dependency_type: Type[T], 
           context: Optional[ResolutionContext] = None,
           default: Optional[T] = None) -> T:
        """
        Resolve and return a dependency.
        
        Args:
            dependency_type: Type of dependency to resolve
            context: Optional resolution context
            default: Default value if dependency not found (optional mode)
            
        Returns:
            Resolved dependency instance
            
        Raises:
            MissingDependencyError: If dependency not found in strict mode
            CircularDependencyError: If circular dependency detected
        """
        context = context or ResolutionContext()
        
        try:
            return self._resolve_dependency(dependency_type, context)
        except MissingDependencyError:
            if default is not None or context.mode == ResolutionMode.OPTIONAL:
                return default
            raise

    def get_named(self, name: str, dependency_type: Type[T] = None,
                 context: Optional[ResolutionContext] = None) -> T:
        """
        Resolve a named dependency.
        
        Args:
            name: Name of the dependency
            dependency_type: Optional type hint for type safety
            context: Optional resolution context
            
        Returns:
            Resolved dependency instance
        """
        context = context or ResolutionContext()
        
        if name not in self._name_mappings:
            raise MissingDependencyError(f"Named dependency '{name}' not found")
        
        registration_id = self._name_mappings[name]
        metadata = self._registrations[registration_id]
        
        # Type check if provided
        if dependency_type and not issubclass(metadata.registered_type, dependency_type):
            raise DependencyError(
                f"Named dependency '{name}' type mismatch: "
                f"expected {dependency_type}, got {metadata.registered_type}"
            )
        
        return self._create_instance(metadata, context)

    def get_all(self, interface_type: Type[T],
               context: Optional[ResolutionContext] = None) -> List[T]:
        """
        Get all implementations of an interface.
        
        Args:
            interface_type: Interface type to find implementations for
            context: Optional resolution context
            
        Returns:
            List of all implementations
        """
        context = context or ResolutionContext()
        implementations = []
        
        # Get all registrations that implement the interface
        registration_ids = self._interface_mappings.get(interface_type, [])
        
        for registration_id in registration_ids:
            metadata = self._registrations[registration_id]
            
            # Check condition
            if metadata.condition and not metadata.condition():
                continue
            
            try:
                instance = self._create_instance(metadata, context)
                implementations.append(instance)
            except Exception as e:
                self.logger.warning(f"Failed to create instance of {registration_id}: {str(e)}")
        
        return implementations

    def get_lazy(self, dependency_type: Type[T],
                context: Optional[ResolutionContext] = None) -> LazyProxy[T]:
        """
        Get a lazy proxy for a dependency.
        
        Args:
            dependency_type: Type of dependency
            context: Optional resolution context
            
        Returns:
            Lazy proxy that resolves dependency on first access
        """
        context = context or ResolutionContext()
        context.mode = ResolutionMode.LAZY
        return LazyProxy(self, dependency_type, context)

    def _resolve_dependency(self, dependency_type: Type[T],
                           context: ResolutionContext) -> T:
        """Internal dependency resolution logic."""
        start_time = time.time()
        
        try:
            # Check for circular dependencies
            if self.config.circular_dependency_detection:
                cycle = self._validator.validate_resolution_chain(
                    context.resolution_chain + [dependency_type]
                )
                if cycle:
                    raise CircularDependencyError(cycle, context)
            
            # Add to resolution chain
            context.resolution_chain.append(dependency_type)
            
            # Find registration
            registration_id = self._find_registration(dependency_type, context)
            if not registration_id:
                if context.mode == ResolutionMode.AUTO_WIRE:
                    return self._auto_wire_dependency(dependency_type, context)
                raise MissingDependencyError(
                    f"No registration found for type {dependency_type}", dependency_type, context
                )
            
            metadata = self._registrations[registration_id]
            
            # Check condition
            if metadata.condition and not metadata.condition():
                raise MissingDependencyError(
                    f"Conditional dependency {dependency_type} condition not met", dependency_type, context
                )
            
            # Create instance
            instance = self._create_instance(metadata, context)
            
            # Update metrics
            resolution_time = time.time() - start_time
            self._update_resolution_metrics(registration_id, resolution_time)
            
            return instance
            
        finally:
            # Remove from resolution chain
            if dependency_type in context.resolution_chain:
                context.resolution_chain.remove(dependency_type)

    def _find_registration(self, dependency_type: Type, context: ResolutionContext) -> Optional[str]:
        """Find registration ID for dependency type."""
        # Direct type mapping
        if dependency_type in self._type_mappings:
            return self._type_mappings[dependency_type]
        
        # Interface mapping - find best match
        candidates = self._interface_mappings.get(dependency_type, [])
        if candidates:
            # Apply filters based on context tags
            if context.tags:
                filtered_candidates = []
                for candidate_id in candidates:
                    metadata = self._registrations[candidate_id]
                    if context.tags.intersection(metadata.tags):
                        filtered_candidates.append(candidate_id)
                if filtered_candidates:
                    candidates = filtered_candidates
            
            # Return highest priority candidate
            if candidates:
                best_candidate = max(candidates, 
                                   key=lambda cid: self._registrations[cid].priority)
                return best_candidate
        
        return None

    def _create_instance(self, metadata: DependencyMetadata, context: ResolutionContext) -> Any:
        """Create instance based on registration metadata."""
        # Update access tracking
        metadata.last_accessed = datetime.now(timezone.utc)
        metadata.access_count += 1
        
        # Handle different scopes
        if metadata.scope == LifecycleScope.SINGLETON:
            return self._get_singleton_instance(metadata, context)
        elif metadata.scope == LifecycleScope.LAZY_SINGLETON:
            return self._get_lazy_singleton_instance(metadata, context)
        elif metadata.scope == LifecycleScope.TRANSIENT:
            return self._create_new_instance(metadata, context)
        elif metadata.scope == LifecycleScope.SCOPED:
            return self._get_scoped_instance(metadata, context)
        elif metadata.scope == LifecycleScope.THREAD_LOCAL:
            return self._get_thread_local_instance(metadata, context)
        else:
            raise DependencyError(f"Unsupported scope: {metadata.scope}")

    @_with_lock
    def _get_singleton_instance(self, metadata: DependencyMetadata, context: ResolutionContext) -> Any:
        """Get or create singleton instance."""
        if metadata.registration_id in self._singleton_instances:
            return self._singleton_instances[metadata.registration_id]
        
        # Create new singleton instance
        instance = self._create_new_instance(metadata, context)
        self._singleton_instances[metadata.registration_id] = instance
        return instance

    def _get_lazy_singleton_instance(self, metadata: DependencyMetadata, context: ResolutionContext) -> Any:
        """Get lazy singleton instance."""
        if metadata.registration_id not in self._singleton_instances:
            with self._lock if self._lock else nullcontext():
                if metadata.registration_id not in self._singleton_instances:
                    instance = self._create_new_instance(metadata, context)
                    self._singleton_instances[metadata.registration_id] = instance
        
        return self._singleton_instances[metadata.registration_id]

    def _get_scoped_instance(self, metadata: DependencyMetadata, context: ResolutionContext) -> Any:
        """Get or create scoped instance."""
        scope_id = context.scope_id or "default"
        
        # Get or create scoped container
        if scope_id not in self._scoped_containers:
            self._scoped_containers[scope_id] = ScopedContainer(scope_id, self)
        
        scoped_container = self._scoped_containers[scope_id]
        
        # Get or create scoped instance
        instance = scoped_container.get_scoped_instance(metadata.registration_id)
        if instance is None:
            instance = self._create_new_instance(metadata, context)
            scoped_container.set_scoped_instance(metadata.registration_id, instance)
        
        return instance

    def _get_thread_local_instance(self, metadata: DependencyMetadata, context: ResolutionContext) -> Any:
        """Get or create thread-local instance."""
        if not hasattr(self._thread_local_instances, 'instances'):
            self._thread_local_instances.instances = {}
        
        instances = self._thread_local_instances.instances
        
        if metadata.registration_id not in instances:
            instance = self._create_new_instance(metadata, context)
            instances[metadata.registration_id] = instance
        
        return instances[metadata.registration_id]

    def _create_new_instance(self, metadata: DependencyMetadata, context: ResolutionContext) -> Any:
        """Create a new instance."""
        try:
            if metadata.instance is not None:
                # Return pre-created instance
                return metadata.instance
            elif metadata.factory:
                # Use factory function
                return self._invoke_factory(metadata.factory, context)
            elif metadata.implementation_type:
                # Create instance from type
                return self._create_from_type(metadata.implementation_type, context)
            else:
                raise DependencyError(f"No way to create instance for {metadata.registration_id}")
                
        except Exception as e:
            self.logger.error(f"Failed to create instance of {metadata.registration_id}: {str(e)}")
            raise DependencyError(
                f"Failed to create instance of {metadata.registered_type}: {str(e)}",
                metadata.registered_type, context
            ) from e

    def _invoke_factory(self, factory: Callable, context: ResolutionContext) -> Any:
        """Invoke factory function with dependency injection."""
        sig = inspect.signature(factory)
        kwargs = {}
        
        # Resolve factory parameters
        for param_name, param in sig.parameters.items():
            if param.annotation != inspect.Parameter.empty:
                try:
                    # Special handling for container injection
                    if param.annotation == Container or param.annotation == 'Container':
                        kwargs[param_name] = self
                    else:
                        # Resolve dependency
                        dependency = self._resolve_dependency(param.annotation, context)
                        kwargs[param_name] = dependency
                except MissingDependencyError:
                    if param.default != inspect.Parameter.empty:
                        kwargs[param_name] = param.default
                    else:
                        raise
        
        # Add context parameters
        kwargs.update(context.parameters)
        
        # Call factory
        result = factory(**kwargs)
        
        # Handle async factories
        if asyncio.iscoroutine(result):
            # If we're not in an async context, we need to run the coroutine
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Create a new task
                    return asyncio.create_task(result)
                else:
                    return loop.run_until_complete(result)
            except RuntimeError:
                # No event loop, run in new loop
                return asyncio.run(result)
        
        return result

    def _create_from_type(self, implementation_type: Type, context: ResolutionContext) -> Any:
        """Create instance from type with auto-wiring."""
        try:
            # Get constructor
            init_method = implementation_type.__init__
            sig = inspect.signature(init_method)
            kwargs = {}
            
            # Resolve constructor parameters
            for param_name, param in sig.parameters.items():
                if param_name == 'self':
                    continue
                
                if param.annotation != inspect.Parameter.empty:
                    try:
                        # Special handling for container injection
                        if param.annotation == Container or param.annotation == 'Container':
                            kwargs[param_name] = self
                        else:
                            # Resolve dependency
                            dependency = self._resolve_dependency(param.annotation, context)
                            kwargs[param_name] = dependency
                    except MissingDependencyError:
                        if param.default != inspect.Parameter.empty:
                            kwargs[param_name] = param.default
                        elif context.mode == ResolutionMode.OPTIONAL:
                            kwargs[param_name] = None
                        else:
                            raise
            
            # Add context parameters
            kwargs.update(context.parameters)
            
            # Create instance
            return implementation_type(**kwargs)
            
        except Exception as e:
            raise DependencyError(
                f"Failed to create instance of {implementation_type}: {str(e)}"
            ) from e

    def _auto_wire_dependency(self, dependency_type: Type, context: ResolutionContext) -> Any:
        """Auto-wire dependency if auto-wiring is enabled."""
        if not self.config.auto_wire_enabled:
            raise MissingDependencyError(
                f"Auto-wiring disabled and no registration found for {dependency_type}"
            )
        
        if not inspect.isclass(dependency_type):
            raise MissingDependencyError(
                f"Cannot auto-wire non-class type {dependency_type}"
            )
        
        # Temporarily register and resolve
        temp_registration_id = f"auto_wire_{dependency_type.__name__}_{uuid.uuid4().hex[:8]}"
        
        metadata = DependencyMetadata(
            registration_id=temp_registration_id,
            registered_type=dependency_type,
            implementation_type=dependency_type,
            scope=LifecycleScope.TRANSIENT,
            strategy=RegistrationStrategy.TYPE_BASED
        )
        
        try:
            return self._create_from_type(dependency_type, context)
        finally:
            # Clean up temporary registration
            pass

    def _update_resolution_metrics(self, registration_id: str, resolution_time: float) -> None:
        """Update performance metrics for resolution."""
        if not self.config.performance_monitoring:
            return
        
        stats = self._resolution_stats[registration_id]
        stats['total_resolutions'] = stats.get('total_resolutions', 0) + 1
        stats['total_time'] = stats.get('total_time', 0.0) + resolution_time
        stats['average_time'] = stats['total_time'] / stats['total_resolutions']
        stats['last_resolution'] = datetime.now(timezone.utc)
        
        # Track recent performance
        self._performance_metrics[registration_id].append(resolution_time)
        if len(self._performance_metrics[registration_id]) > 100:
            self._performance_metrics[registration_id].pop(0)

    def create_scope(self, scope_id: Optional[str] = None) -> str:
        """
        Create a new dependency scope.
        
        Args:
            scope_id: Optional scope identifier
            
        Returns:
            Scope identifier
        """
        scope_id = scope_id or f"scope_{uuid.uuid4().hex[:8]}"
        
        if scope_id not in self._scoped_containers:
            self._scoped_containers[scope_id] = ScopedContainer(scope_id, self)
        
        return scope_id

    def destroy_scope(self, scope_id: str) -> None:
        """
        Destroy a dependency scope and cleanup all scoped instances.
        
        Args:
            scope_id: Scope identifier to destroy
        """
        if scope_id in self._scoped_containers:
            scoped_container = self._scoped_containers[scope_id]
            scoped_container.clear()
            del self._scoped_containers[scope_id]
            
            self.logger.debug(f"Destroyed scope: {scope_id}")

    @contextmanager
    def scope(self, scope_id: Optional[str] = None):
        """
        Context manager for scoped dependencies.
        
        Args:
            scope_id: Optional scope identifier
            
        Yields:
            Scope identifier
        """
        scope_id = self.create_scope(scope_id)
        try:
            yield scope_id
        finally:
            self.destroy_scope(scope_id)

    def is_registered(self, dependency_type: Type, name: Optional[str] = None) -> bool:
        """
        Check if a dependency is registered.
        
        Args:
            dependency_type: Type to check
            name: Optional name for named registration
            
        Returns:
            True if registered, False otherwise
        """
        if name:
            return name in self._name_mappings
        return dependency_type in self._type_mappings

    def unregister(self, dependency_type: Type, name: Optional[str] = None) -> bool:
        """
        Unregister a dependency.
        
        Args:
            dependency_type: Type to unregister
            name: Optional name for named registration
            
        Returns:
            True if successfully unregistered, False if not found
        """
        registration_id = None
        
        if name and name in self._name_mappings:
            registration_id = self._name_mappings[name]
        elif dependency_type in self._type_mappings:
            registration_id = self._type_mappings[dependency_type]
        
        if not registration_id or registration_id not in self._registrations:
            return False
        
        return self._unregister_by_id(registration_id)

    @_with_lock
    def _unregister_by_id(self, registration_id: str) -> bool:
        """Unregister dependency by registration ID."""
        if registration_id not in self._registrations:
            return False
        
        metadata = self._registrations[registration_id]
        
        # Remove from mappings
        if metadata.registered_type in self._type_mappings:
            del self._type_mappings[metadata.registered_type]
        
        if metadata.name and metadata.name in self._name_mappings:
            del self._name_mappings[metadata.name]
        
        # Remove from interface mappings
        if metadata.implementation_type:
            for base in inspect.getmro(metadata.implementation_type)[1:]:
                if base in self._interface_mappings:
                    self._interface_mappings[base] = [
                        rid for rid in self._interface_mappings[base] 
                        if rid != registration_id
                    ]
        
        # Clean up instances
        self._singleton_instances.pop(registration_id, None)
        
        # Remove from scoped containers
        for scoped_container in self._scoped_containers.values():
            scoped_container.scoped_instances.pop(registration_id, None)
        
        # Remove registration
        del self._registrations[registration_id]
        
        # Clean up metrics
        self._resolution_stats.pop(registration_id, None)
        self._performance_metrics.pop(registration_id, None)
        
        self.logger.debug(f"Unregistered dependency: {registration_id}")
        return True

    def clear(self) -> None:
        """Clear all registrations and instances."""
        with self._lock if self._lock else nullcontext():
            # Clear all scoped containers
            for scope_id in list(self._scoped_containers.keys()):
                self.destroy_scope(scope_id)
            
            # Clear all mappings and instances
            self._registrations.clear()
            self._type_mappings.clear()
            self._name_mappings.clear()
            self._interface_mappings.clear()
            self._singleton_instances.clear()
            self._scoped_containers.clear()
            
            # Clear metrics
            self._resolution_stats.clear()
            self._performance_metrics.clear()
            self._health_status.clear()
            
            self.logger.info("Container cleared")

    def get_registrations(self) -> Dict[str, DependencyMetadata]:
        """Get all current registrations."""
        return dict(self._registrations)

    def get_registration_info(self, dependency_type: Type, name: Optional[str] = None) -> Optional[DependencyMetadata]:
        """Get registration information for a dependency."""
        registration_id = None
        
        if name and name in self._name_mappings:
            registration_id = self._name_mappings[name]
        elif dependency_type in self._type_mappings:
            registration_id = self._type_mappings[dependency_type]
        
        if registration_id and registration_id in self._registrations:
            return self._registrations[registration_id]
        
        return None

    def get_performance_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get performance metrics for all registrations."""
        return dict(self._resolution_stats)

    def get_health_status(self) -> Dict[str, Any]:
        """Get container health status."""
        total_registrations = len(self._registrations)
        healthy_registrations = sum(1 for meta in self._registrations.values() if meta.is_valid)
        
        return {
            'status': 'healthy' if total_registrations == healthy_registrations else 'degraded',
            'total_registrations': total_registrations,
            'healthy_registrations': healthy_registrations,
            'invalid_registrations': total_registrations - healthy_registrations,
            'active_scopes': len(self._scoped_containers),
            'singleton_instances': len(self._singleton_instances),
            'thread_safe': self.config.thread_safe,
            'auto_wire_enabled': self.config.auto_wire_enabled,
            'performance_monitoring': self.config.performance_monitoring
        }

    def validate_all_registrations(self) -> Dict[str, List[str]]:
        """Validate all registrations and return errors."""
        validation_results = {}
        
        for registration_id, metadata in self._registrations.items():
            errors = self._validator.validate_registration(metadata)
            if errors:
                validation_results[registration_id] = errors
                metadata.is_valid = False
                metadata.validation_errors = errors
            else:
                metadata.is_valid = True
                metadata.validation_errors.clear()
        
        return validation_results

    def load_configuration(self, config_path: Union[str, Path]) -> None:
        """
        Load container configuration from file.
        
        Args:
            config_path: Path to configuration file (YAML or JSON)
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise ValueError(f"Configuration file not found: {config_path}")
        
        # Load configuration
        with open(config_path, 'r') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                config_data = yaml.safe_load(f)
            elif config_path.suffix.lower() == '.json':
                config_data = json.load(f)
            else:
                raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")
        
        # Apply container configuration
        if 'container' in config_data:
            container_config = ContainerConfiguration.from_dict(config_data['container'])
            self.config = container_config
        
        # Load registrations
        if 'registrations' in config_data:
            self._load_registrations_from_config(config_data['registrations'])
        
        self.logger.info(f"Loaded configuration from: {config_path}")

    def _load_registrations_from_config(self, registrations_config: List[Dict[str, Any]]) -> None:
        """Load registrations from configuration data."""
        for reg_config in registrations_config:
            try:
                # Parse registration configuration
                type_name = reg_config['type']
                implementation_name = reg_config.get('implementation', type_name)
                scope = LifecycleScope(reg_config.get('scope', 'singleton'))
                name = reg_config.get('name')
                tags = set(reg_config.get('tags', []))
                
                # Import types
                registered_type = self._import_type(type_name)
                implementation_type = self._import_type(implementation_name) if implementation_name != type_name else None
                
                # Register
                self.register(
                    registered_type=registered_type,
                    implementation=implementation_type,
                    scope=scope,
                    name=name,
                    tags=tags
                )
                
            except Exception as e:
                self.logger.error(f"Failed to load registration from config: {reg_config}: {str(e)}")

    def _import_type(self, type_name: str) -> Type:
        """Import type from string."""
        try:
            if '.' in type_name:
                module_name, class_name = type_name.rsplit('.', 1)
                module = importlib.import_module(module_name)
                return getattr(module, class_name)
            else:
                # Try to find in current module or builtins
                return eval(type_name)
        except Exception as e:
            raise ImportError(f"Cannot import type '{type_name}': {str(e)}")

    def register_from_plugin(self, plugin_id: str, registration_func: Callable[['Container'], None]) -> None:
        """
        Register dependencies from a plugin.
        
        Args:
            plugin_id: Plugin identifier
            registration_func: Function that registers dependencies
        """
        if not self.config.plugin_integration_enabled:
            raise DependencyError("Plugin integration is disabled")
        
        try:
            # Mark start of plugin registrations
            initial_count = len(self._registrations)
            
            # Call plugin registration function
            registration_func(self)
            
            # Track plugin registrations
            final_count = len(self._registrations)
            new_registrations = final_count - initial_count
            
            if new_registrations > 0:
                self._plugin_registrations.add(plugin_id)
                self.logger.info(f"Plugin '{plugin_id}' registered {new_registrations} dependencies")
                
        except Exception as e:
            self.logger.error(f"Failed to register dependencies from plugin '{plugin_id}': {str(e)}")
            raise DependencyError(f"Plugin registration failed: {str(e)}") from e

    def unregister_plugin(self, plugin_id: str) -> int:
        """
        Unregister all dependencies from a plugin.
        
        Args:
            plugin_id: Plugin identifier
            
        Returns:
            Number of dependencies unregistered
        """
        if plugin_id not in self._plugin_registrations:
            return 0
        
        # This is a simplified version - in practice, you'd need to track
        # which registrations belong to which plugin
        unregistered_count = 0
        
        # Remove plugin from tracking
        self._plugin_registrations.discard(plugin_id)
        
        self.logger.info(f"Unregistered {unregistered_count} dependencies from plugin '{plugin_id}'")
        return unregistered_count

    def enable_hot_swapping(self) -> None:
        """Enable hot-swapping for development."""
        if not self.config.hot_swapping_enabled:
            self.config.hot_swapping_enabled = True
            # Backup current registrations
            self._original_registrations = dict(self._registrations)
            self.logger.info("Hot-swapping enabled")

    def disable_hot_swapping(self) -> None:
        """Disable hot-swapping."""
        if self.config.hot_swapping_enabled:
            self.config.hot_swapping_enabled = False
            self._original_registrations.clear()
            self.logger.info("Hot-swapping disabled")

    def hot_swap(self, dependency_type: Type[T], new_implementation: Union[Type[T], T, Factory[T]]) -> bool:
        """
        Hot-swap a dependency implementation.
        
        Args:
            dependency_type: Type to swap
            new_implementation: New implementation
            
        Returns:
            True if successfully swapped, False otherwise
        """
        if not self.config.hot_swapping_enabled:
            raise DependencyError("Hot-swapping is not enabled")
        
        if dependency_type not in self._type_mappings:
            return False
        
        registration_id = self._type_mappings[dependency_type]
        metadata = self._registrations[registration_id]
        
        # Backup original if not already done
        if registration_id not in self._original_registrations:
            self._original_registrations[registration_id] = DependencyMetadata(**metadata.__dict__)
        
        # Update implementation
        if inspect.isclass(new_implementation):
            metadata.implementation_type = new_implementation
            metadata.strategy = RegistrationStrategy.TYPE_BASED
        elif callable(new_implementation):
            metadata.factory = new_implementation
            metadata.strategy = RegistrationStrategy.FACTORY_BASED
        else:
            metadata.instance = new_implementation
            metadata.strategy = RegistrationStrategy.TYPE_BASED
        
        # Clear existing instances to force recreation
        self._singleton_instances.pop(registration_id, None)
        for scoped_container in self._scoped_containers.values():
            scoped_container.scoped_instances.pop(registration_id, None)
        
        self.logger.info(f"Hot-swapped implementation for {dependency_type.__name__}")
        return True

    def restore_original(self, dependency_type: Type) -> bool:
        """
        Restore original implementation after hot-swap.
        
        Args:
            dependency_type: Type to restore
            
        Returns:
            True if successfully restored, False otherwise
        """
        if not self.config.hot_swapping_enabled:
            return False
        
        if dependency_type not in self._type_mappings:
            return False
        
        registration_id = self._type_mappings[dependency_type]
        
        if registration_id not in self._original_registrations:
            return False
        
        # Restore original metadata
        original_metadata = self._original_registrations[registration_id]
        self._registrations[registration_id] = original_metadata
        
        # Clear instances to force recreation
        self._singleton_instances.pop(registration_id, None)
        for scoped_container in self._scoped_containers.values():
            scoped_container.scoped_instances.pop(registration_id, None)
        
        self.logger.info(f"Restored original implementation for {dependency_type.__name__}")
        return True

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup resources."""
        self.clear()

    def __repr__(self) -> str:
        """String representation of container."""
        return (f"Container(registrations={len(self._registrations)}, "
                f"singletons={len(self._singleton_instances)}, "
                f"scopes={len(self._scoped_containers)})")


# Utility function for creating configured container
def create_container(config_path: Optional[Union[str, Path]] = None,
                    **config_overrides) -> Container:
    """
    Create a configured dependency injection container.
    
    Args:
        config_path: Optional path to configuration file
        **config_overrides: Configuration overrides
        
    Returns:
        Configured container instance
    """
    # Create configuration
    config = ContainerConfiguration()
    
    # Apply overrides
    for key, value in config_overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    # Create container
    container = Container(config)
    
    # Load configuration file if provided
    if config_path:
        container.load_configuration(config_path)
    
    return container


# Context manager for null operations
@contextmanager
def nullcontext():
    """Null context manager for conditional locking."""
    yield
