"""
Advanced Component Management System
Author: Drmusab
Last Modified: 2025-05-26 14:58:15 UTC

This module provides comprehensive component lifecycle management for the AI assistant,
including registration, dependency resolution, health monitoring, and graceful shutdown.
"""

from pathlib import Path
from typing import Optional, Dict, Any, List, Set, Callable, Type, Union, AsyncGenerator
import asyncio
import threading
import time
from datetime import datetime, timezone
from dataclasses import dataclass, field
from enum import Enum
from contextlib import asynccontextmanager
import logging
import inspect
from collections import defaultdict, deque
import weakref
from abc import ABC, abstractmethod

# Core imports
from src.core.config.loader import ConfigLoader
from src.core.events.event_bus import EventBus
from src.core.events.event_types import (
    ComponentRegistered,
    ComponentInitialized,
    ComponentStarted,
    ComponentStopped,
    ComponentFailed,
    ComponentHealthChanged,
    DependencyResolved,
    SystemShutdownStarted,
    SystemShutdownCompleted
)
from src.core.error_handling import ErrorHandler, handle_exceptions
from src.core.dependency_injection import Container
from src.core.health_check import HealthCheck

# Observability
from src.observability.monitoring.metrics import MetricsCollector
from src.observability.monitoring.tracing import TraceManager
from src.observability.logging.config import get_logger

# Learning and adaptation
from src.learning.feedback_processor import FeedbackProcessor


class ComponentState(Enum):
    """Component lifecycle states."""
    UNREGISTERED = "unregistered"
    REGISTERED = "registered"
    INITIALIZING = "initializing"
    INITIALIZED = "initialized"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"
    DISABLED = "disabled"


class ComponentPriority(Enum):
    """Component initialization and shutdown priorities."""
    CRITICAL = 0      # Core system components (config, logging, events)
    ESSENTIAL = 1     # Critical dependencies (database, cache)
    HIGH = 2          # Important components (auth, security)
    NORMAL = 3        # Standard components (processors, analyzers)
    LOW = 4           # Optional components (monitoring, analytics)
    BACKGROUND = 5    # Background services and optimizations


class DependencyType(Enum):
    """Types of component dependencies."""
    REQUIRED = "required"        # Must be available for component to function
    OPTIONAL = "optional"        # Can function without but may have reduced capability
    WEAK = "weak"               # Loose coupling, no initialization dependency
    CONDITIONAL = "conditional"  # Required only under certain conditions


@dataclass
class ComponentDependency:
    """Represents a dependency between components."""
    component_id: str
    dependency_type: DependencyType = DependencyType.REQUIRED
    condition: Optional[Callable[[], bool]] = None
    timeout_seconds: float = 30.0
    retry_count: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComponentMetadata:
    """Metadata for component registration."""
    component_id: str
    component_type: Type
    priority: ComponentPriority = ComponentPriority.NORMAL
    dependencies: List[ComponentDependency] = field(default_factory=list)
    provides: Set[str] = field(default_factory=set)
    config_section: Optional[str] = None
    health_check_interval: float = 60.0
    auto_restart: bool = True
    restart_max_attempts: int = 3
    restart_backoff_factor: float = 2.0
    shutdown_timeout: float = 30.0
    initialization_timeout: float = 60.0
    tags: Set[str] = field(default_factory=set)
    description: Optional[str] = None
    version: str = "1.0.0"


@dataclass
class ComponentInfo:
    """Runtime information about a component."""
    metadata: ComponentMetadata
    instance: Optional[Any] = None
    state: ComponentState = ComponentState.UNREGISTERED
    startup_time: Optional[datetime] = None
    last_health_check: Optional[datetime] = None
    health_status: str = "unknown"
    error_count: int = 0
    restart_count: int = 0
    last_error: Optional[Exception] = None
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    resource_usage: Dict[str, Any] = field(default_factory=dict)


class ComponentError(Exception):
    """Custom exception for component management operations."""
    
    def __init__(self, message: str, component_id: Optional[str] = None, error_code: Optional[str] = None):
        super().__init__(message)
        self.component_id = component_id
        self.error_code = error_code
        self.timestamp = datetime.now(timezone.utc)


class ComponentInterface(ABC):
    """Base interface that all managed components should implement."""
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the component."""
        pass
    
    async def start(self) -> None:
        """Start the component (optional)."""
        pass
    
    async def stop(self) -> None:
        """Stop the component (optional)."""
        pass
    
    async def cleanup(self) -> None:
        """Cleanup resources (optional)."""
        pass
    
    async def health_check(self) -> Dict[str, Any]:
        """Return component health status (optional)."""
        return {"status": "healthy"}


class ComponentFactory:
    """Factory for creating component instances with dependency injection."""
    
    def __init__(self, container: Container):
        self.container = container
        self.logger = get_logger(__name__)
    
    def create_component(self, component_type: Type, config: Optional[Dict[str, Any]] = None) -> Any:
        """Create a component instance with proper dependency injection."""
        try:
            # Get constructor parameters
            sig = inspect.signature(component_type.__init__)
            kwargs = {}
            
            for param_name, param in sig.parameters.items():
                if param_name == 'self':
                    continue
                
                # Try to resolve from container
                if param.annotation != inspect.Parameter.empty:
                    try:
                        if param.annotation == Container:
                            kwargs[param_name] = self.container
                        else:
                            kwargs[param_name] = self.container.get(param.annotation)
                    except Exception as e:
                        if param.default == inspect.Parameter.empty:
                            self.logger.warning(f"Could not resolve dependency {param_name}: {e}")
                
                # Override with config if provided
                if config and param_name in config:
                    kwargs[param_name] = config[param_name]
            
            return component_type(**kwargs)
            
        except Exception as e:
            raise ComponentError(f"Failed to create component {component_type.__name__}: {str(e)}")


class DependencyResolver:
    """Resolves component dependencies and determines initialization order."""
    
    def __init__(self, logger):
        self.logger = logger
        self._dependency_graph: Dict[str, Set[str]] = defaultdict(set)
        self._reverse_graph: Dict[str, Set[str]] = defaultdict(set)
    
    def add_dependency(self, component_id: str, dependency: ComponentDependency) -> None:
        """Add a dependency relationship."""
        if dependency.dependency_type in [DependencyType.REQUIRED, DependencyType.OPTIONAL]:
            self._dependency_graph[component_id].add(dependency.component_id)
            self._reverse_graph[dependency.component_id].add(component_id)
    
    def resolve_initialization_order(self, component_ids: Set[str]) -> List[List[str]]:
        """Resolve component initialization order using topological sort."""
        # Filter graph to only include registered components
        filtered_graph = {
            comp_id: deps.intersection(component_ids) 
            for comp_id, deps in self._dependency_graph.items() 
            if comp_id in component_ids
        }
        
        # Add components with no dependencies
        for comp_id in component_ids:
            if comp_id not in filtered_graph:
                filtered_graph[comp_id] = set()
        
        # Topological sort by priority and dependencies
        result = []
        remaining = set(component_ids)
        
        while remaining:
            # Find components with no unresolved dependencies
            ready = {
                comp_id for comp_id in remaining 
                if not filtered_graph[comp_id].intersection(remaining)
            }
            
            if not ready:
                # Circular dependency detected
                cycle = self._detect_cycle(filtered_graph, remaining)
                raise ComponentError(f"Circular dependency detected: {cycle}")
            
            result.append(sorted(ready))
            remaining -= ready
        
        return result
    
    def _detect_cycle(self, graph: Dict[str, Set[str]], nodes: Set[str]) -> List[str]:
        """Detect circular dependencies using DFS."""
        visited = set()
        rec_stack = set()
        
        def dfs(node: str, path: List[str]) -> Optional[List[str]]:
            if node in rec_stack:
                cycle_start = path.index(node)
                return path[cycle_start:] + [node]
            
            if node in visited:
                return None
            
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in graph.get(node, set()):
                if neighbor in nodes:
                    cycle = dfs(neighbor, path + [node])
                    if cycle:
                        return cycle
            
            rec_stack.remove(node)
            return None
        
        for node in nodes:
            if node not in visited:
                cycle = dfs(node, [])
                if cycle:
                    return cycle
        
        return []


class ComponentMonitor:
    """Monitors component health and performance."""
    
    def __init__(self, logger, metrics: Optional[MetricsCollector] = None):
        self.logger = logger
        self.metrics = metrics
        self._monitoring_tasks: Dict[str, asyncio.Task] = {}
        self._health_callbacks: Dict[str, Callable] = {}
    
    async def start_monitoring(self, component_id: str, interval: float, health_callback: Callable) -> None:
        """Start monitoring a component."""
        self._health_callbacks[component_id] = health_callback
        
        if component_id in self._monitoring_tasks:
            self._monitoring_tasks[component_id].cancel()
        
        self._monitoring_tasks[component_id] = asyncio.create_task(
            self._monitor_component(component_id, interval)
        )
    
    async def stop_monitoring(self, component_id: str) -> None:
        """Stop monitoring a component."""
        if component_id in self._monitoring_tasks:
            self._monitoring_tasks[component_id].cancel()
            del self._monitoring_tasks[component_id]
        
        self._health_callbacks.pop(component_id, None)
    
    async def _monitor_component(self, component_id: str, interval: float) -> None:
        """Monitor component health at regular intervals."""
        while True:
            try:
                await asyncio.sleep(interval)
                
                if component_id in self._health_callbacks:
                    health_status = await self._health_callbacks[component_id]()
                    
                    if self.metrics:
                        self.metrics.set(f"component_health_{component_id}", 
                                       1 if health_status.get("status") == "healthy" else 0)
                        
                        if "response_time" in health_status:
                            self.metrics.record(f"component_health_response_time_{component_id}",
                                             health_status["response_time"])
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Health check failed for component {component_id}: {str(e)}")
                if self.metrics:
                    self.metrics.increment(f"component_health_errors_{component_id}")


class EnhancedComponentManager:
    """
    Advanced component lifecycle manager for the AI assistant system.
    
    Features:
    - Comprehensive component registration and discovery
    - Intelligent dependency resolution and initialization ordering
    - Health monitoring with automatic recovery
    - Graceful shutdown with dependency-aware ordering
    - Performance monitoring and resource tracking
    - Event-driven architecture integration
    - Hot-reload capabilities for development
    - Configuration-driven component management
    """
    
    def __init__(self, container: Container):
        """
        Initialize the enhanced component manager.
        
        Args:
            container: Dependency injection container
        """
        self.container = container
        self.logger = get_logger(__name__)
        self.config = container.get(ConfigLoader)
        self.event_bus = container.get(EventBus)
        self.error_handler = container.get(ErrorHandler)
        self.health_check = container.get(HealthCheck)
        
        # Component management
        self._components: Dict[str, ComponentInfo] = {}
        self._component_factory = ComponentFactory(container)
        self._dependency_resolver = DependencyResolver(self.logger)
        
        # Monitoring and observability
        self._setup_monitoring()
        self._component_monitor = ComponentMonitor(self.logger, self.metrics)
        
        # State management
        self._initialization_lock = asyncio.Lock()
        self._shutdown_event = asyncio.Event()
        self._startup_completed = asyncio.Event()
        
        # Performance tracking
        self._startup_times: Dict[str, float] = {}
        self._error_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Configuration
        self._auto_discovery_enabled = self.config.get("components.auto_discovery", True)
        self._parallel_initialization = self.config.get("components.parallel_initialization", True)
        self._health_monitoring_enabled = self.config.get("components.health_monitoring", True)
        
        # Register core health check
        self.health_check.register_component("component_manager", self._health_check_callback)
        
        self.logger.info("EnhancedComponentManager initialized")

    def _setup_monitoring(self) -> None:
        """Setup monitoring and metrics collection."""
        try:
            self.metrics = self.container.get(MetricsCollector)
            self.tracer = self.container.get(TraceManager)
            
            # Register component management metrics
            self.metrics.register_counter("component_registrations_total")
            self.metrics.register_counter("component_initializations_total")
            self.metrics.register_counter("component_failures_total")
            self.metrics.register_histogram("component_initialization_duration_seconds")
            self.metrics.register_gauge("components_running")
            self.metrics.register_gauge("components_failed")
            
        except Exception as e:
            self.logger.warning(f"Failed to setup monitoring: {str(e)}")
            self.metrics = None
            self.tracer = None

    @handle_exceptions
    def register_component(
        self,
        component_id: str,
        component_type: Type,
        priority: ComponentPriority = ComponentPriority.NORMAL,
        dependencies: Optional[List[ComponentDependency]] = None,
        **kwargs
    ) -> None:
        """
        Register a component for management.
        
        Args:
            component_id: Unique identifier for the component
            component_type: Component class type
            priority: Initialization priority
            dependencies: List of component dependencies
            **kwargs: Additional metadata
        """
        if component_id in self._components:
            raise ComponentError(f"Component {component_id} is already registered")
        
        # Create metadata
        metadata = ComponentMetadata(
            component_id=component_id,
            component_type=component_type,
            priority=priority,
            dependencies=dependencies or [],
            **kwargs
        )
        
        # Register dependencies
        for dep in metadata.dependencies:
            self._dependency_resolver.add_dependency(component_id, dep)
        
        # Create component info
        component_info = ComponentInfo(
            metadata=metadata,
            state=ComponentState.REGISTERED
        )
        
        self._components[component_id] = component_info
        
        # Emit registration event
        asyncio.create_task(self.event_bus.emit(ComponentRegistered(
            component_id=component_id,
            component_type=component_type.__name__,
            priority=priority.value
        )))
        
        if self.metrics:
            self.metrics.increment("component_registrations_total")
        
        self.logger.info(f"Registered component: {component_id} (priority: {priority.value})")

    @handle_exceptions
    async def discover_components(self, search_paths: Optional[List[Path]] = None) -> None:
        """
        Automatically discover and register components.
        
        Args:
            search_paths: Paths to search for components
        """
        if not self._auto_discovery_enabled:
            return
        
        search_paths = search_paths or [
            Path("src/processing"),
            Path("src/integrations"),
            Path("src/memory"),
            Path("src/skills"),
            Path("src/reasoning")
        ]
        
        discovered_count = 0
        
        with self.tracer.trace("component_discovery") if self.tracer else None:
            for search_path in search_paths:
                if not search_path.exists():
                    continue
                
                for py_file in search_path.rglob("*.py"):
                    if py_file.name.startswith("__"):
                        continue
                    
                    try:
                        # Dynamic import and inspection would go here
                        # This is a simplified version
                        discovered_count += await self._discover_from_file(py_file)
                        
                    except Exception as e:
                        self.logger.warning(f"Failed to discover components in {py_file}: {str(e)}")
        
        self.logger.info(f"Discovered {discovered_count} components")

    async def _discover_from_file(self, file_path: Path) -> int:
        """Discover components from a Python file."""
        # This would implement actual component discovery logic
        # For now, it's a placeholder
        return 0

    @handle_exceptions
    async def initialize_all(self) -> None:
        """Initialize all registered components in dependency order."""
        async with self._initialization_lock:
            if self._startup_completed.is_set():
                self.logger.warning("Components already initialized")
                return
            
            start_time = time.time()
            
            with self.tracer.trace("component_initialization") if self.tracer else None:
                try:
                    # Resolve initialization order
                    component_ids = set(self._components.keys())
                    initialization_batches = self._dependency_resolver.resolve_initialization_order(component_ids)
                    
                    # Group by priority within each batch
                    prioritized_batches = []
                    for batch in initialization_batches:
                        priority_groups = defaultdict(list)
                        for comp_id in batch:
                            priority = self._components[comp_id].metadata.priority
                            priority_groups[priority].append(comp_id)
                        
                        # Sort by priority
                        for priority in sorted(priority_groups.keys(), key=lambda p: p.value):
                            prioritized_batches.append(priority_groups[priority])
                    
                    # Initialize components
                    initialized_count = 0
                    failed_count = 0
                    
                    for batch in prioritized_batches:
                        if self._parallel_initialization and len(batch) > 1:
                            # Parallel initialization within batch
                            tasks = [self._initialize_component(comp_id) for comp_id in batch]
                            results = await asyncio.gather(*tasks, return_exceptions=True)
                            
                            for comp_id, result in zip(batch, results):
                                if isinstance(result, Exception):
                                    failed_count += 1
                                    self.logger.error(f"Failed to initialize {comp_id}: {result}")
                                else:
                                    initialized_count += 1
                        else:
                            # Sequential initialization
                            for comp_id in batch:
                                try:
                                    await self._initialize_component(comp_id)
                                    initialized_count += 1
                                except Exception as e:
                                    failed_count += 1
                                    self.logger.error(f"Failed to initialize {comp_id}: {str(e)}")
                    
                    # Update metrics
                    if self.metrics:
                        self.metrics.set("components_running", initialized_count)
                        self.metrics.set("components_failed", failed_count)
                    
                    initialization_time = time.time() - start_time
                    self.logger.info(
                        f"Component initialization completed: {initialized_count} initialized, "
                        f"{failed_count} failed in {initialization_time:.2f}s"
                    )
                    
                    self._startup_completed.set()
                    
                except Exception as e:
                    self.logger.error(f"Component initialization failed: {str(e)}")
                    raise ComponentError(f"Failed to initialize components: {str(e)}")

    async def _initialize_component(self, component_id: str) -> None:
        """Initialize a specific component."""
        component_info = self._components.get(component_id)
        if not component_info:
            raise ComponentError(f"Component {component_id} not found")
        
        if component_info.state != ComponentState.REGISTERED:
            self.logger.debug(f"Component {component_id} already initialized (state: {component_info.state})")
            return
        
        start_time = time.time()
        component_info.state = ComponentState.INITIALIZING
        
        try:
            # Create component instance
            config = self._get_component_config(component_id)
            instance = self._component_factory.create_component(
                component_info.metadata.component_type, 
                config
            )
            
            # Initialize the component
            if hasattr(instance, 'initialize'):
                await instance.initialize()
            
            # Start the component if it supports it
            if hasattr(instance, 'start'):
                await instance.start()
            
            # Update component info
            component_info.instance = instance
            component_info.state = ComponentState.RUNNING
            component_info.startup_time = datetime.now(timezone.utc)
            
            initialization_time = time.time() - start_time
            self._startup_times[component_id] = initialization_time
            
            # Start health monitoring
            if self._health_monitoring_enabled and hasattr(instance, 'health_check'):
                await self._component_monitor.start_monitoring(
                    component_id,
                    component_info.metadata.health_check_interval,
                    lambda: instance.health_check()
                )
            
            # Emit initialization event
            await self.event_bus.emit(ComponentInitialized(
                component_id=component_id,
                initialization_time=initialization_time
            ))
            
            if self.metrics:
                self.metrics.increment("component_initializations_total")
                self.metrics.record("component_initialization_duration_seconds", initialization_time)
            
            self.logger.info(f"Initialized component: {component_id} in {initialization_time:.2f}s")
            
        except Exception as e:
            component_info.state = ComponentState.FAILED
            component_info.last_error = e
            component_info.error_count += 1
            
            # Record error
            self._error_history[component_id].append({
                "timestamp": datetime.now(timezone.utc),
                "error": str(e),
                "error_type": type(e).__name__
            })
            
            # Emit failure event
            await self.event_bus.emit(ComponentFailed(
                component_id=component_id,
                error_message=str(e),
                error_type=type(e).__name__
            ))
            
            if self.metrics:
                self.metrics.increment("component_failures_total")
            
            raise ComponentError(f"Failed to initialize component {component_id}: {str(e)}", component_id)

    def _get_component_config(self, component_id: str) -> Optional[Dict[str, Any]]:
        """Get configuration for a specific component."""
        component_info = self._components.get(component_id)
        if not component_info or not component_info.metadata.config_section:
            return None
        
        return self.config.get(component_info.metadata.config_section, {})

    @handle_exceptions
    async def get_component(self, component_id: str) -> Any:
        """
        Get a component instance by ID.
        
        Args:
            component_id: Component identifier
            
        Returns:
            Component instance
            
        Raises:
            ComponentError: If component not found or not running
        """
        component_info = self._components.get(component_id)
        if not component_info:
            raise ComponentError(f"Component {component_id} not found")
        
        if component_info.state != ComponentState.RUNNING:
            raise ComponentError(f"Component {component_id} is not running (state: {component_info.state})")
        
        return component_info.instance

    @handle_exceptions
    async def stop_component(self, component_id: str, timeout: Optional[float] = None) -> None:
        """
        Stop a specific component.
        
        Args:
            component_id: Component identifier
            timeout: Shutdown timeout in seconds
        """
        component_info = self._components.get(component_id)
        if not component_info or component_info.state != ComponentState.RUNNING:
            self.logger.warning(f"Component {component_id} is not running")
            return
        
        timeout = timeout or component_info.metadata.shutdown_timeout
        component_info.state = ComponentState.STOPPING
        
        try:
            # Stop health monitoring
            await self._component_monitor.stop_monitoring(component_id)
            
            # Stop the component
            if hasattr(component_info.instance, 'stop'):
                await asyncio.wait_for(
                    component_info.instance.stop(),
                    timeout=timeout
                )
            
            # Cleanup resources
            if hasattr(component_info.instance, 'cleanup'):
                await asyncio.wait_for(
                    component_info.instance.cleanup(),
                    timeout=timeout
                )
            
            component_info.state = ComponentState.STOPPED
            component_info.instance = None
            
            # Emit stopped event
            await self.event_bus.emit(ComponentStopped(component_id=component_id))
            
            self.logger.info(f"Stopped component: {component_id}")
            
        except asyncio.TimeoutError:
            self.logger.error(f"Component {component_id} shutdown timed out")
            component_info.state = ComponentState.FAILED
            raise ComponentError(f"Component {component_id} shutdown timed out")
        except Exception as e:
            self.logger.error(f"Error stopping component {component_id}: {str(e)}")
            component_info.state = ComponentState.FAILED
            raise ComponentError(f"Failed to stop component {component_id}: {str(e)}")

    @handle_exceptions
    async def restart_component(self, component_id: str) -> None:
        """
        Restart a specific component.
        
        Args:
            component_id: Component identifier
        """
        component_info = self._components.get(component_id)
        if not component_info:
            raise ComponentError(f"Component {component_id} not found")
        
        self.logger.info(f"Restarting component: {component_id}")
        
        # Stop the component
        await self.stop_component(component_id)
        
        # Wait a moment
        await asyncio.sleep(1.0)
        
        # Reinitialize
        component_info.state = ComponentState.REGISTERED
        component_info.restart_count += 1
        
        await self._initialize_component(component_id)

    async def shutdown_all(self, timeout: float = 30.0) -> None:
        """
        Gracefully shutdown all components in reverse dependency order.
        
        Args:
            timeout: Global shutdown timeout in seconds
        """
        if self._shutdown_event.is_set():
            self.logger.warning("Shutdown already in progress")
            return
        
        self._shutdown_event.set()
        
        # Emit shutdown started event
        await self.event_bus.emit(SystemShutdownStarted())
        
        start_time = time.time()
        
        try:
            # Get running components
            running_components = [
                comp_id for comp_id, info in self._components.items()
                if info.state == ComponentState.RUNNING
            ]
            
            # Resolve shutdown order (reverse of initialization)
            if running_components:
                initialization_batches = self._dependency_resolver.resolve_initialization_order(
                    set(running_components)
                )
                shutdown_batches = list(reversed(initialization_batches))
                
                # Shutdown components
                for batch in shutdown_batches:
                    # Reverse priority order for shutdown
                    batch_by_priority = defaultdict(list)
                    for comp_id in batch:
                        priority = self._components[comp_id].metadata.priority
                        batch_by_priority[priority].append(comp_id)
                    
                    for priority in sorted(batch_by_priority.keys(), key=lambda p: p.value, reverse=True):
                        tasks = [self.stop_component(comp_id) for comp_id in batch_by_priority[priority]]
                        await asyncio.gather(*tasks, return_exceptions=True)
            
            shutdown_time = time.time() - start_time
            
            # Stop component monitoring
            for comp_id in list(self._component_monitor._monitoring_tasks.keys()):
                await self._component_monitor.stop_monitoring(comp_id)
            
            # Emit shutdown completed event
            await self.event_bus.emit(SystemShutdownCompleted(shutdown_time=shutdown_time))
            
            self.logger.info(f"System shutdown completed in {shutdown_time:.2f}s")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {str(e)}")
            raise ComponentError(f"Shutdown failed: {str(e)}")

    def get_component_status(self, component_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get status information for components.
        
        Args:
            component_id: Specific component ID, or None for all components
            
        Returns:
            Component status information
        """
        if component_id:
            if component_id not in self._components:
                raise ComponentError(f"Component {component_id} not found")
            
            component_info = self._components[component_id]
            return {
                "component_id": component_id,
                "state": component_info.state.value,
                "startup_time": component_info.startup_time.isoformat() if component_info.startup_time else None,
                "health_status": component_info.health_status,
                "error_count": component_info.error_count,
                "restart_count": component_info.restart_count,
                "last_error": str(component_info.last_error) if component_info.last_error else None,
                "performance_metrics": component_info.performance_metrics,
                "resource_usage": component_info.resource_usage
            }
        else:
            # Return status for all components
            return {
                "total_components": len(self._components),
                "running_components": len([
                    info for info in self._components.values() 
                    if info.state == ComponentState.RUNNING
                ]),
                "failed_components": len([
                    info for info in self._components.values() 
                    if info.state == ComponentState.FAILED
                ]),
                "components": {
                    comp_id: {
                        "state": info.state.value,
                        "health_status": info.health_status,
                        "error_count": info.error_count
                    }
                    for comp_id, info in self._components.items()
                }
            }

    def list_components(self, state_filter: Optional[ComponentState] = None) -> List[str]:
        """
        List component IDs, optionally filtered by state.
        
        Args:
            state_filter: Optional state filter
            
        Returns:
            List of component IDs
        """
        if state_filter:
            return [
                comp_id for comp_id, info in self._components.items()
                if info.state == state_filter
            ]
        else:
            return list(self._components.keys())

    async def _health_check_callback(self) -> Dict[str, Any]:
        """Health check callback for the component manager."""
        try:
            total_components = len(self._components)
            running_components = len([
                info for info in self._components.values() 
                if info.state == ComponentState.RUNNING
            ])
            failed_components = len([
                info for info in self._components.values() 
                if info.state == ComponentState.FAILED
            ])
            
            # Calculate health score
            if total_components == 0:
                health_score = 1.0
            else:
                health_score = running_components / total_components
            
            status = "healthy" if health_score >= 0.8 else "degraded" if health_score >= 0.5 else "unhealthy"
            
            return {
                "status": status,
                "health_score": health_score,
                "total_components": total_components,
                "running_components": running_components,
                "failed_components": failed_components,
                "startup_completed": self._startup_completed.is_set(),
                "shutdown_in_progress": self._shutdown_event.is_set()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }

    @asynccontextmanager
    async def managed_lifecycle(self):
        """Context manager for complete component lifecycle management."""
        try:
            # Discover and initialize components
            await self.discover_components()
            await self.initialize_all()
            yield self
        finally:
            # Graceful shutdown
            await self.shutdown_all()

    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            if hasattr(self, '_components') and any(
                info.state == ComponentState.RUNNING 
                for info in self._components.values()
            ):
                self.logger.warning("ComponentManager destroyed with running components")
        except Exception:
            pass  # Ignore cleanup errors in destructor
