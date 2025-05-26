"""
Advanced Plugin Management System
Author: Drmusab
Last Modified: 2025-05-26 16:13:03 UTC

This module provides comprehensive plugin lifecycle management for the AI assistant,
including dynamic loading, sandboxing, dependency resolution, hot-reloading, and
seamless integration with the core system components.
"""

from pathlib import Path
from typing import Optional, Dict, Any, List, Set, Callable, Type, Union, AsyncGenerator, Protocol
import asyncio
import threading
import time
import importlib
import importlib.util
import sys
import inspect
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from dataclasses import dataclass, field
from enum import Enum
from contextlib import asynccontextmanager
import logging
import weakref
from abc import ABC, abstractmethod
import uuid
import shutil
import tempfile
import zipfile
import tarfile
from concurrent.futures import ThreadPoolExecutor

# Core imports
from src.core.config.loader import ConfigLoader
from src.core.events.event_bus import EventBus
from src.core.events.event_types import (
    PluginLoaded, PluginUnloaded, PluginEnabled, PluginDisabled,
    PluginError, PluginDependencyResolved, PluginHotReloaded,
    PluginSecurityViolation, PluginPerformanceWarning
)
from src.core.error_handling import ErrorHandler, handle_exceptions
from src.core.dependency_injection import Container
from src.core.health_check import HealthCheck
from src.core.security.authentication import AuthenticationManager
from src.core.security.authorization import AuthorizationManager
from src.core.security.sanitization import SecuritySanitizer

# Assistant components
from src.assistant.component_manager import EnhancedComponentManager, ComponentMetadata, ComponentPriority
from src.assistant.workflow_orchestrator import WorkflowOrchestrator, WorkflowDefinition
from src.assistant.session_manager import EnhancedSessionManager
from src.assistant.interaction_handler import InteractionHandler

# Skills and processing
from src.skills.skill_factory import SkillFactory
from src.skills.skill_registry import SkillRegistry
from src.skills.skill_validator import SkillValidator

# Observability
from src.observability.monitoring.metrics import MetricsCollector
from src.observability.monitoring.tracing import TraceManager
from src.observability.logging.config import get_logger

# Memory and learning
from src.memory.memory_manager import MemoryManager
from src.learning.feedback_processor import FeedbackProcessor


class PluginState(Enum):
    """Plugin lifecycle states."""
    DISCOVERED = "discovered"
    VALIDATED = "validated"
    LOADED = "loaded"
    INITIALIZED = "initialized"
    ENABLED = "enabled"
    DISABLED = "disabled"
    SUSPENDED = "suspended"
    ERROR = "error"
    UNLOADED = "unloaded"
    UPDATING = "updating"
    INSTALLING = "installing"
    UNINSTALLING = "uninstalling"


class PluginType(Enum):
    """Types of plugins supported by the system."""
    SKILL = "skill"                      # New AI skills
    PROCESSOR = "processor"              # Data processors
    INTEGRATION = "integration"          # External integrations
    UI_COMPONENT = "ui_component"        # User interface components
    WORKFLOW_EXTENSION = "workflow_extension"  # Workflow extensions
    MEMORY_PROVIDER = "memory_provider"  # Memory providers
    LEARNING_MODULE = "learning_module"  # Learning modules
    SECURITY_MODULE = "security_module"  # Security modules
    MIDDLEWARE = "middleware"            # API middleware
    UTILITY = "utility"                  # Utility functions
    THEME = "theme"                      # UI themes
    LANGUAGE_PACK = "language_pack"      # Localization


class PluginLoadMode(Enum):
    """Plugin loading modes."""
    EAGER = "eager"          # Load immediately
    LAZY = "lazy"            # Load on first use
    ON_DEMAND = "on_demand"  # Load when explicitly requested
    SCHEDULED = "scheduled"  # Load at specific times


class SecurityLevel(Enum):
    """Plugin security levels."""
    UNTRUSTED = "untrusted"      # No special permissions
    SANDBOX = "sandbox"          # Limited sandbox environment
    TRUSTED = "trusted"          # Full system access
    SYSTEM = "system"            # System-level access
    VERIFIED = "verified"        # Cryptographically verified


@dataclass
class PluginDependency:
    """Represents a plugin dependency."""
    plugin_id: str
    version_requirement: str = "*"  # Semantic version requirement
    optional: bool = False
    auto_install: bool = False
    load_order: int = 0  # Loading order priority


@dataclass
class PluginCapability:
    """Represents a capability provided by a plugin."""
    name: str
    version: str
    interface: Type
    description: Optional[str] = None
    category: Optional[str] = None


@dataclass
class PluginMetadata:
    """Comprehensive plugin metadata."""
    plugin_id: str
    name: str
    version: str
    description: str
    author: str
    
    # Plugin characteristics
    plugin_type: PluginType
    load_mode: PluginLoadMode = PluginLoadMode.EAGER
    security_level: SecurityLevel = SecurityLevel.UNTRUSTED
    
    # Dependencies and capabilities
    dependencies: List[PluginDependency] = field(default_factory=list)
    system_dependencies: List[str] = field(default_factory=list)
    provides: List[PluginCapability] = field(default_factory=list)
    
    # Entry points
    main_class: Optional[str] = None
    entry_points: Dict[str, str] = field(default_factory=dict)
    
    # Configuration
    config_schema: Dict[str, Any] = field(default_factory=dict)
    default_config: Dict[str, Any] = field(default_factory=dict)
    
    # Resource requirements
    memory_limit_mb: float = 256.0
    cpu_limit_percent: float = 10.0
    network_access: bool = False
    file_system_access: bool = False
    
    # Lifecycle hooks
    install_hooks: List[str] = field(default_factory=list)
    uninstall_hooks: List[str] = field(default_factory=list)
    
    # Metadata
    homepage: Optional[str] = None
    repository: Optional[str] = None
    license: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)
    
    # System integration
    api_version: str = "1.0.0"
    min_system_version: str = "1.0.0"
    max_system_version: Optional[str] = None
    
    # Security
    signature: Optional[str] = None
    checksum: Optional[str] = None
    permissions: Set[str] = field(default_factory=set)
    
    # Installation info
    installation_date: Optional[datetime] = None
    update_date: Optional[datetime] = None
    source_url: Optional[str] = None


@dataclass
class PluginInfo:
    """Runtime plugin information."""
    metadata: PluginMetadata
    state: PluginState = PluginState.DISCOVERED
    
    # Runtime data
    module: Optional[Any] = None
    instance: Optional[Any] = None
    installation_path: Optional[Path] = None
    
    # Performance metrics
    load_time: float = 0.0
    memory_usage: float = 0.0
    cpu_usage: float = 0.0
    error_count: int = 0
    warning_count: int = 0
    
    # Health information
    last_health_check: Optional[datetime] = None
    health_status: str = "unknown"
    
    # Usage statistics
    activation_count: int = 0
    last_used: Optional[datetime] = None
    usage_statistics: Dict[str, Any] = field(default_factory=dict)
    
    # Error tracking
    last_error: Optional[Exception] = None
    error_history: List[Dict[str, Any]] = field(default_factory=list)


class PluginError(Exception):
    """Custom exception for plugin management operations."""
    
    def __init__(self, message: str, plugin_id: Optional[str] = None, 
                 error_code: Optional[str] = None):
        super().__init__(message)
        self.plugin_id = plugin_id
        self.error_code = error_code
        self.timestamp = datetime.now(timezone.utc)


class PluginInterface(Protocol):
    """Base interface that all plugins should implement."""
    
    def get_metadata(self) -> PluginMetadata:
        """Get plugin metadata."""
        ...
    
    async def initialize(self, container: Container) -> None:
        """Initialize the plugin."""
        ...
    
    async def activate(self) -> None:
        """Activate the plugin."""
        ...
    
    async def deactivate(self) -> None:
        """Deactivate the plugin."""
        ...
    
    async def cleanup(self) -> None:
        """Cleanup plugin resources."""
        ...
    
    async def health_check(self) -> Dict[str, Any]:
        """Return plugin health status."""
        ...


class PluginSandbox:
    """Provides sandboxed execution environment for untrusted plugins."""
    
    def __init__(self, plugin_id: str, security_level: SecurityLevel):
        self.plugin_id = plugin_id
        self.security_level = security_level
        self.logger = get_logger(f"plugin_sandbox_{plugin_id}")
        self._resource_limits = {}
        self._allowed_modules = set()
        self._restricted_attributes = set()
    
    def configure_sandbox(self, config: Dict[str, Any]) -> None:
        """Configure sandbox restrictions."""
        self._resource_limits = config.get('resource_limits', {})
        self._allowed_modules = set(config.get('allowed_modules', []))
        self._restricted_attributes = set(config.get('restricted_attributes', []))
    
    def execute_in_sandbox(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function in sandboxed environment."""
        if self.security_level == SecurityLevel.TRUSTED:
            return func(*args, **kwargs)
        
        # Implement sandbox restrictions
        original_import = __builtins__['__import__']
        
        def restricted_import(name, *args, **kwargs):
            if name not in self._allowed_modules and not name.startswith('src.'):
                raise ImportError(f"Module {name} not allowed in sandbox")
            return original_import(name, *args, **kwargs)
        
        try:
            __builtins__['__import__'] = restricted_import
            return func(*args, **kwargs)
        finally:
            __builtins__['__import__'] = original_import


class PluginLoader:
    """Handles dynamic plugin loading and unloading."""
    
    def __init__(self, logger):
        self.logger = logger
        self._loaded_modules: Dict[str, Any] = {}
        self._module_paths: Dict[str, Path] = {}
    
    async def load_plugin_from_path(self, plugin_path: Path) -> Optional[Any]:
        """Load a plugin from a file path."""
        try:
            # Read plugin metadata
            metadata_file = plugin_path / "plugin.json"
            if not metadata_file.exists():
                raise PluginError(f"Plugin metadata not found: {metadata_file}")
            
            with open(metadata_file, 'r') as f:
                metadata_dict = json.load(f)
            
            metadata = self._parse_metadata(metadata_dict)
            
            # Load the plugin module
            module_path = plugin_path / "__init__.py"
            if not module_path.exists():
                module_path = plugin_path / f"{metadata.plugin_id}.py"
            
            if not module_path.exists():
                raise PluginError(f"Plugin entry point not found: {module_path}")
            
            spec = importlib.util.spec_from_file_location(metadata.plugin_id, module_path)
            if spec is None or spec.loader is None:
                raise PluginError(f"Failed to create module spec for {metadata.plugin_id}")
            
            module = importlib.util.module_from_spec(spec)
            self._loaded_modules[metadata.plugin_id] = module
            self._module_paths[metadata.plugin_id] = plugin_path
            
            # Execute the module
            spec.loader.exec_module(module)
            
            # Get the main plugin class
            if metadata.main_class:
                plugin_class = getattr(module, metadata.main_class)
                return plugin_class()
            else:
                # Look for a class implementing PluginInterface
                for name in dir(module):
                    obj = getattr(module, name)
                    if (inspect.isclass(obj) and 
                        hasattr(obj, 'get_metadata') and 
                        hasattr(obj, 'initialize')):
                        return obj()
            
            raise PluginError(f"No valid plugin class found in {metadata.plugin_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to load plugin from {plugin_path}: {str(e)}")
            raise PluginError(f"Plugin loading failed: {str(e)}")
    
    def _parse_metadata(self, metadata_dict: Dict[str, Any]) -> PluginMetadata:
        """Parse plugin metadata from dictionary."""
        # Convert dictionary to PluginMetadata with proper type conversion
        return PluginMetadata(
            plugin_id=metadata_dict['plugin_id'],
            name=metadata_dict['name'],
            version=metadata_dict['version'],
            description=metadata_dict['description'],
            author=metadata_dict['author'],
            plugin_type=PluginType(metadata_dict.get('plugin_type', 'utility')),
            load_mode=PluginLoadMode(metadata_dict.get('load_mode', 'eager')),
            security_level=SecurityLevel(metadata_dict.get('security_level', 'untrusted')),
            **{k: v for k, v in metadata_dict.items() 
               if k not in ['plugin_id', 'name', 'version', 'description', 'author']}
        )
    
    async def unload_plugin(self, plugin_id: str) -> None:
        """Unload a plugin module."""
        if plugin_id in self._loaded_modules:
            # Remove from sys.modules if present
            module_name = f"plugin_{plugin_id}"
            if module_name in sys.modules:
                del sys.modules[module_name]
            
            del self._loaded_modules[plugin_id]
            self._module_paths.pop(plugin_id, None)
            
            self.logger.info(f"Unloaded plugin module: {plugin_id}")


class PluginRegistry:
    """Manages plugin registration and discovery."""
    
    def __init__(self, logger):
        self.logger = logger
        self._plugins: Dict[str, PluginInfo] = {}
        self._capabilities: Dict[str, List[str]] = {}  # capability -> plugin_ids
        self._categories: Dict[str, List[str]] = {}    # category -> plugin_ids
        self._registry_lock = asyncio.Lock()
    
    async def register_plugin(self, plugin_info: PluginInfo) -> None:
        """Register a plugin in the registry."""
        async with self._registry_lock:
            plugin_id = plugin_info.metadata.plugin_id
            
            if plugin_id in self._plugins:
                raise PluginError(f"Plugin {plugin_id} is already registered")
            
            self._plugins[plugin_id] = plugin_info
            
            # Index capabilities
            for capability in plugin_info.metadata.provides:
                if capability.name not in self._capabilities:
                    self._capabilities[capability.name] = []
                self._capabilities[capability.name].append(plugin_id)
            
            # Index categories
            for category in plugin_info.metadata.categories:
                if category not in self._categories:
                    self._categories[category] = []
                self._categories[category].append(plugin_id)
            
            self.logger.info(f"Registered plugin: {plugin_id}")
    
    async def unregister_plugin(self, plugin_id: str) -> None:
        """Unregister a plugin from the registry."""
        async with self._registry_lock:
            if plugin_id not in self._plugins:
                return
            
            plugin_info = self._plugins[plugin_id]
            
            # Remove from capability index
            for capability in plugin_info.metadata.provides:
                if capability.name in self._capabilities:
                    self._capabilities[capability.name] = [
                        pid for pid in self._capabilities[capability.name] 
                        if pid != plugin_id
                    ]
                    if not self._capabilities[capability.name]:
                        del self._capabilities[capability.name]
            
            # Remove from category index
            for category in plugin_info.metadata.categories:
                if category in self._categories:
                    self._categories[category] = [
                        pid for pid in self._categories[category] 
                        if pid != plugin_id
                    ]
                    if not self._categories[category]:
                        del self._categories[category]
            
            del self._plugins[plugin_id]
            self.logger.info(f"Unregistered plugin: {plugin_id}")
    
    def get_plugin(self, plugin_id: str) -> Optional[PluginInfo]:
        """Get plugin information by ID."""
        return self._plugins.get(plugin_id)
    
    def list_plugins(self, 
                    state: Optional[PluginState] = None,
                    plugin_type: Optional[PluginType] = None,
                    category: Optional[str] = None) -> List[PluginInfo]:
        """List plugins with optional filtering."""
        plugins = list(self._plugins.values())
        
        if state:
            plugins = [p for p in plugins if p.state == state]
        
        if plugin_type:
            plugins = [p for p in plugins if p.metadata.plugin_type == plugin_type]
        
        if category:
            plugins = [p for p in plugins if category in p.metadata.categories]
        
        return plugins
    
    def find_plugins_by_capability(self, capability: str) -> List[str]:
        """Find plugins that provide a specific capability."""
        return self._capabilities.get(capability, [])


class EnhancedPluginManager:
    """
    Advanced Plugin Management System for the AI Assistant.
    
    Features:
    - Dynamic plugin loading and unloading
    - Comprehensive dependency resolution
    - Security sandboxing and validation
    - Hot-reloading during development
    - Performance monitoring and resource management
    - Integration with all core system components
    - Plugin marketplace and repository management
    - Automatic updates and version management
    - Event-driven plugin communication
    - Plugin composition and orchestration
    """
    
    def __init__(self, container: Container):
        """
        Initialize the enhanced plugin manager.
        
        Args:
            container: Dependency injection container
        """
        self.container = container
        self.logger = get_logger(__name__)
        self.config = container.get(ConfigLoader)
        self.event_bus = container.get(EventBus)
        self.error_handler = container.get(ErrorHandler)
        self.health_check = container.get(HealthCheck)
        
        # Core component integration
        self.component_manager = container.get(EnhancedComponentManager)
        self.workflow_orchestrator = container.get(WorkflowOrchestrator)
        self.session_manager = container.get(EnhancedSessionManager)
        self.interaction_handler = container.get(InteractionHandler)
        
        # Skills and processing
        self.skill_factory = container.get(SkillFactory)
        self.skill_registry = container.get(SkillRegistry)
        self.skill_validator = container.get(SkillValidator)
        
        # Memory and learning
        self.memory_manager = container.get(MemoryManager)
        self.feedback_processor = container.get(FeedbackProcessor)
        
        # Security
        try:
            self.auth_manager = container.get(AuthenticationManager)
            self.authz_manager = container.get(AuthorizationManager)
            self.security_sanitizer = container.get(SecuritySanitizer)
        except Exception:
            self.logger.warning("Security components not available")
            self.auth_manager = None
            self.authz_manager = None
            self.security_sanitizer = None
        
        # Setup plugin management components
        self._setup_plugin_infrastructure()
        self._setup_monitoring()
        
        # Configuration
        self._plugin_directories = self._get_plugin_directories()
        self._auto_discovery_enabled = self.config.get("plugins.auto_discovery", True)
        self._hot_reload_enabled = self.config.get("plugins.hot_reload", False)
        self._security_validation_enabled = self.config.get("plugins.security_validation", True)
        
        # State management
        self._initialization_lock = asyncio.Lock()
        self._shutdown_event = asyncio.Event()
        self._background_tasks: List[asyncio.Task] = []
        
        # Performance tracking
        self._plugin_performance: Dict[str, Dict[str, float]] = {}
        
        self.logger.info("EnhancedPluginManager initialized")

    def _setup_plugin_infrastructure(self) -> None:
        """Setup plugin management infrastructure."""
        self.plugin_loader = PluginLoader(self.logger)
        self.plugin_registry = PluginRegistry(self.logger)
        
        # Thread pool for plugin operations
        self.thread_pool = ThreadPoolExecutor(
            max_workers=4,
            thread_name_prefix="plugin_manager"
        )

    def _setup_monitoring(self) -> None:
        """Setup monitoring and metrics collection."""
        try:
            self.metrics = self.container.get(MetricsCollector)
            self.tracer = self.container.get(TraceManager)
            
            # Register plugin metrics
            self.metrics.register_counter("plugins_loaded_total")
            self.metrics.register_counter("plugins_failed_total")
            self.metrics.register_gauge("plugins_active")
            self.metrics.register_histogram("plugin_load_duration_seconds")
            self.metrics.register_histogram("plugin_execution_duration_seconds")
            
        except Exception as e:
            self.logger.warning(f"Failed to setup monitoring: {str(e)}")
            self.metrics = None
            self.tracer = None

    def _get_plugin_directories(self) -> List[Path]:
        """Get plugin directories from configuration."""
        plugin_dirs = self.config.get("plugins.directories", [
            "plugins/",
            "src/plugins/",
            "data/plugins/"
        ])
        
        return [Path(d) for d in plugin_dirs]

    async def initialize(self) -> None:
        """Initialize the plugin manager and discover plugins."""
        async with self._initialization_lock:
            try:
                self.logger.info("Initializing plugin manager...")
                
                # Register event handlers
                await self._register_event_handlers()
                
                # Setup health monitoring
                self.health_check.register_component("plugin_manager", self._health_check_callback)
                
                # Discover and load plugins
                if self._auto_discovery_enabled:
                    await self.discover_plugins()
                
                # Start background tasks
                await self._start_background_tasks()
                
                self.logger.info("Plugin manager initialized successfully")
                
            except Exception as e:
                self.logger.error(f"Failed to initialize plugin manager: {str(e)}")
                raise PluginError(f"Plugin manager initialization failed: {str(e)}")

    async def _register_event_handlers(self) -> None:
        """Register event handlers for system events."""
        # Component events
        self.event_bus.subscribe("component_registered", self._handle_component_registered)
        self.event_bus.subscribe("component_failed", self._handle_component_failed)
        
        # Workflow events
        self.event_bus.subscribe("workflow_completed", self._handle_workflow_completed)
        
        # Session events
        self.event_bus.subscribe("session_started", self._handle_session_started)
        self.event_bus.subscribe("session_ended", self._handle_session_ended)
        
        # System events
        self.event_bus.subscribe("system_shutdown_started", self._handle_system_shutdown)

    async def _start_background_tasks(self) -> None:
        """Start background maintenance tasks."""
        # Plugin health monitoring
        self._background_tasks.append(
            asyncio.create_task(self._plugin_health_monitor_loop())
        )
        
        # Hot reload monitoring
        if self._hot_reload_enabled:
            self._background_tasks.append(
                asyncio.create_task(self._hot_reload_monitor_loop())
            )
        
        # Performance monitoring
        self._background_tasks.append(
            asyncio.create_task(self._performance_monitor_loop())
        )
        
        # Plugin cleanup
        self._background_tasks.append(
            asyncio.create_task(self._cleanup_monitor_loop())
        )

    @handle_exceptions
    async def discover_plugins(self) -> List[str]:
        """
        Discover plugins in configured directories.
        
        Returns:
            List of discovered plugin IDs
        """
        discovered_plugins = []
        
        with self.tracer.trace("plugin_discovery") if self.tracer else None:
            for plugin_dir in self._plugin_directories:
                if not plugin_dir.exists():
                    continue
                
                self.logger.info(f"Discovering plugins in: {plugin_dir}")
                
                # Look for plugin directories
                for item in plugin_dir.iterdir():
                    if item.is_dir() and not item.name.startswith('.'):
                        try:
                            plugin_id = await self._discover_single_plugin(item)
                            if plugin_id:
                                discovered_plugins.append(plugin_id)
                        except Exception as e:
                            self.logger.warning(f"Failed to discover plugin in {item}: {str(e)}")
                
                # Look for plugin archives
                for item in plugin_dir.iterdir():
                    if item.is_file() and item.suffix in ['.zip', '.tar.gz']:
                        try:
                            plugin_id = await self._discover_plugin_archive(item)
                            if plugin_id:
                                discovered_plugins.append(plugin_id)
                        except Exception as e:
                            self.logger.warning(f"Failed to discover plugin archive {item}: {str(e)}")
        
        self.logger.info(f"Discovered {len(discovered_plugins)} plugins: {discovered_plugins}")
        return discovered_plugins

    async def _discover_single_plugin(self, plugin_path: Path) -> Optional[str]:
        """Discover a single plugin from a directory."""
        metadata_file = plugin_path / "plugin.json"
        if not metadata_file.exists():
            return None
        
        try:
            with open(metadata_file, 'r') as f:
                metadata_dict = json.load(f)
            
            metadata = self.plugin_loader._parse_metadata(metadata_dict)
            
            # Validate plugin
            if self._security_validation_enabled:
                await self._validate_plugin_security(plugin_path, metadata)
            
            # Create plugin info
            plugin_info = PluginInfo(
                metadata=metadata,
                state=PluginState.DISCOVERED,
                installation_path=plugin_path
            )
            
            # Register plugin
            await self.plugin_registry.register_plugin(plugin_info)
            
            # Auto-load if configured
            if metadata.load_mode == PluginLoadMode.EAGER:
                await self.load_plugin(metadata.plugin_id)
            
            return metadata.plugin_id
            
        except Exception as e:
            self.logger.error(f"Failed to discover plugin {plugin_path}: {str(e)}")
            return None

    async def _discover_plugin_archive(self, archive_path: Path) -> Optional[str]:
        """Discover a plugin from an archive file."""
        temp_dir = Path(tempfile.mkdtemp(prefix="plugin_extract_"))
        
        try:
            # Extract archive
            if archive_path.suffix == '.zip':
                with zipfile.ZipFile(archive_path, 'r') as zip_file:
                    zip_file.extractall(temp_dir)
            elif archive_path.suffix == '.tar.gz':
                with tarfile.open(archive_path, 'r:gz') as tar_file:
                    tar_file.extractall(temp_dir)
            
            # Find plugin directory in extracted files
            for item in temp_dir.iterdir():
                if item.is_dir():
                    plugin_id = await self._discover_single_plugin(item)
                    if plugin_id:
                        # Move to permanent location
                        plugin_dir = self._plugin_directories[0] / plugin_id
                        shutil.move(str(item), str(plugin_dir))
                        
                        # Update installation path
                        plugin_info = self.plugin_registry.get_plugin(plugin_id)
                        if plugin_info:
                            plugin_info.installation_path = plugin_dir
                        
                        return plugin_id
            
            return None
            
        finally:
            # Cleanup temp directory
            shutil.rmtree(temp_dir, ignore_errors=True)

    async def _validate_plugin_security(self, plugin_path: Path, metadata: PluginMetadata) -> None:
        """Validate plugin security."""
        if self.security_sanitizer is None:
            return
        
        # Check file permissions
        for file_path in plugin_path.rglob("*"):
            if file_path.is_file():
                # Basic security checks
                if file_path.suffix in ['.exe', '.bat', '.sh'] and metadata.security_level != SecurityLevel.TRUSTED:
                    raise PluginError(f"Executable files not allowed in untrusted plugins: {file_path}")
        
        # Validate metadata signature if present
        if metadata.signature and self.auth_manager:
            # Implement signature validation
            pass

    @handle_exceptions
    async def load_plugin(self, plugin_id: str) -> None:
        """
        Load a plugin into the system.
        
        Args:
            plugin_id: Plugin identifier
        """
        plugin_info = self.plugin_registry.get_plugin(plugin_id)
        if not plugin_info:
            raise PluginError(f"Plugin {plugin_id} not found in registry")
        
        if plugin_info.state in [PluginState.LOADED, PluginState.ENABLED]:
            self.logger.warning(f"Plugin {plugin_id} is already loaded")
            return
        
        start_time = time.time()
        
        try:
            with self.tracer.trace("plugin_loading") if self.tracer else None:
                self.logger.info(f"Loading plugin: {plugin_id}")
                
                # Check dependencies
                await self._resolve_plugin_dependencies(plugin_info)
                
                # Load plugin module
                if plugin_info.installation_path:
                    instance = await self.plugin_loader.load_plugin_from_path(
                        plugin_info.installation_path
                    )
                    plugin_info.instance = instance
                
                # Initialize plugin
                if hasattr(plugin_info.instance, 'initialize'):
                    await plugin_info.instance.initialize(self.container)
                
                # Update state
                plugin_info.state = PluginState.LOADED
                plugin_info.load_time = time.time() - start_time
                
                # Register with component manager if it's a component
                await self._register_plugin_as_component(plugin_info)
                
                # Register skills if it's a skill plugin
                await self._register_plugin_skills(plugin_info)
                
                # Register workflows if it provides them
                await self._register_plugin_workflows(plugin_info)
                
                # Emit event
                await self.event_bus.emit(PluginLoaded(
                    plugin_id=plugin_id,
                    plugin_type=plugin_info.metadata.plugin_type.value,
                    load_time=plugin_info.load_time
                ))
                
                if self.metrics:
                    self.metrics.increment("plugins_loaded_total")
                    self.metrics.record("plugin_load_duration_seconds", plugin_info.load_time)
                
                self.logger.info(f"Successfully loaded plugin {plugin_id} in {plugin_info.load_time:.2f}s")
                
        except Exception as e:
            plugin_info.state = PluginState.ERROR
            plugin_info.last_error = e
            plugin_info.error_count += 1
            
            await self.event_bus.emit(PluginError(
                plugin_id=plugin_id,
                error_message=str(e),
                error_type=type(e).__name__
            ))
            
            if self.metrics:
                self.metrics.increment("plugins_failed_total")
            
            self.logger.error(f"Failed to load plugin {plugin_id}: {str(e)}")
            raise PluginError(f"Failed to load plugin {plugin_id}: {str(e)}")

    async def _resolve_plugin_dependencies(self, plugin_info: PluginInfo) -> None:
        """Resolve plugin dependencies."""
        for dependency in plugin_info.metadata.dependencies:
            dep_plugin = self.plugin_registry.get_plugin(dependency.plugin_id)
            
            if not dep_plugin:
                if dependency.auto_install:
                    # Try to install dependency
                    await self._auto_install_plugin(dependency.plugin_id)
                elif not dependency.optional:
                    raise PluginError(f"Required dependency {dependency.plugin_id} not found")
            else:
                # Ensure dependency is loaded
                if dep_plugin.state not in [PluginState.LOADED, PluginState.ENABLED]:
                    await self.load_plugin(dependency.plugin_id)

    async def _auto_install_plugin(self, plugin_id: str) -> None:
        """Auto-install a plugin dependency."""
        # This would implement automatic plugin installation from repository
        self.logger.warning(f"Auto-installation of {plugin_id} not implemented")

    async def _register_plugin_as_component(self, plugin_info: PluginInfo) -> None:
        """Register plugin as a system component."""
        if plugin_info.metadata.plugin_type in [PluginType.PROCESSOR, PluginType.INTEGRATION]:
            try:
                metadata = ComponentMetadata(
                    component_id=f"plugin_{plugin_info.metadata.plugin_id}",
                    component_type=type(plugin_info.instance),
                    priority=ComponentPriority.NORMAL,
                    description=plugin_info.metadata.description
                )
                
                self.component_manager.register_component(
                    f"plugin_{plugin_info.metadata.plugin_id}",
                    type(plugin_info.instance),
                    ComponentPriority.NORMAL,
                    []
                )
                
            except Exception as e:
                self.logger.warning(f"Failed to register plugin {plugin_info.metadata.plugin_id} as component: {str(e)}")

    async def _register_plugin_skills(self, plugin_info: PluginInfo) -> None:
        """Register skills provided by plugin."""
        if plugin_info.metadata.plugin_type == PluginType.SKILL:
            try:
                if hasattr(plugin_info.instance, 'get_skills'):
                    skills = plugin_info.instance.get_skills()
                    for skill in skills:
                        self.skill_registry.register_skill(skill)
                        
            except Exception as e:
                self.logger.warning(f"Failed to register skills from plugin {plugin_info.metadata.plugin_id}: {str(e)}")

    async def _register_plugin_workflows(self, plugin_info: PluginInfo) -> None:
        """Register workflows provided by plugin."""
        if plugin_info.metadata.plugin_type == PluginType.WORKFLOW_EXTENSION:
            try:
                if hasattr(plugin_info.instance, 'get_workflows'):
                    workflows = plugin_info.instance.get_workflows()
                    for workflow in workflows:
                        self.workflow_orchestrator.register_workflow(workflow)
                        
            except Exception as e:
                self.logger.warning(f"Failed to register workflows from plugin {plugin_info.metadata.plugin_id}: {str(e)}")

    @handle_exceptions
    async def enable_plugin(self, plugin_id: str) -> None:
        """
        Enable a loaded plugin.
        
        Args:
            plugin_id: Plugin identifier
        """
        plugin_info = self.plugin_registry.get_plugin(plugin_id)
        if not plugin_info:
            raise PluginError(f"Plugin {plugin_id} not found")
        
        if plugin_info.state != PluginState.LOADED:
            raise PluginError(f"Plugin {plugin_id} must be loaded before enabling")
        
        try:
            # Activate plugin
            if hasattr(plugin_info.instance, 'activate'):
                await plugin_info.instance.activate()
            
            plugin_info.state = PluginState.ENABLED
            plugin_info.activation_count += 1
            plugin_info.last_used = datetime.now(timezone.utc)
            
            await self.event_bus.emit(PluginEnabled(
                plugin_id=plugin_id,
                plugin_type=plugin_info.metadata.plugin_type.value
            ))
            
            self.logger.info(f"Enabled plugin: {plugin_id}")
            
        except Exception as e:
            plugin_info.state = PluginState.ERROR
            plugin_info.last_error = e
            self.logger.error(f"Failed to enable plugin {plugin_id}: {str(e)}")
            raise PluginError(f"Failed to enable plugin {plugin_id}: {str(e)}")

    @handle_exceptions
    async def disable_plugin(self, plugin_id: str) -> None:
        """
        Disable an enabled plugin.
        
        Args:
            plugin_id: Plugin identifier
        """
        plugin_info = self.plugin_registry.get_plugin(plugin_id)
        if not plugin_info:
            raise PluginError(f"Plugin {plugin_id} not found")
        
        if plugin_info.state != PluginState.ENABLED:
            self.logger.warning(f"Plugin {plugin_id} is not enabled")
            return
        
        try:
            # Deactivate plugin
            if hasattr(plugin_info.instance, 'deactivate'):
                await plugin_info.instance.deactivate()
            
            plugin_info.state = PluginState.DISABLED
            
            await self.event_bus.emit(PluginDisabled(
                plugin_id=plugin_id,
                plugin_type=plugin_info.metadata.plugin_type.value
            ))
            
            self.logger.info(f"Disabled plugin: {plugin_id}")
            
        except Exception as e:
            plugin_info.state = PluginState.ERROR
            plugin_info.last_error = e
            self.logger.error(f"Failed to disable plugin {plugin_id}: {str(e)}")
            raise PluginError(f"Failed to disable plugin {plugin_id}: {str(e)}")

    @handle_exceptions
    async def unload_plugin(self, plugin_id: str, force: bool = False) -> None:
        """
        Unload a plugin from the system.
        
        Args:
            plugin_id: Plugin identifier
            force: Force unload even if plugin is in use
        """
        plugin_info = self.plugin_registry.get_plugin(plugin_id)
        if not plugin_info:
            self.logger.warning(f"Plugin {plugin_id} not found in registry")
            return
        
        try:
            # Disable plugin first if enabled
            if plugin_info.state == PluginState.ENABLED:
                await self.disable_plugin(plugin_id)
            
            # Cleanup plugin resources
            if hasattr(plugin_info.instance, 'cleanup'):
                await plugin_info.instance.cleanup()
            
            # Unload module
            await self.plugin_loader.unload_plugin(plugin_id)
            
            # Update state
            plugin_info.state = PluginState.UNLOADED
            plugin_info.instance = None
            plugin_info.module = None
            
            await self.event_bus.emit(PluginUnloaded(
                plugin_id=plugin_id,
                plugin_type=plugin_info.metadata.plugin_type.value
            ))
            
            self.logger.info(f"Unloaded plugin: {plugin_id}")
            
        except Exception as e:
            plugin_info.state = PluginState.ERROR
            plugin_info.last_error = e
            self.logger.error(f"Failed to unload plugin {plugin_id}: {str(e)}")
            if not force:
                raise PluginError(f"Failed to unload plugin {plugin_id}: {str(e)}")

    async def hot_reload_plugin(self, plugin_id: str) -> None:
        """
        Hot-reload a plugin (unload and reload).
        
        Args:
            plugin_id: Plugin identifier
        """
        plugin_info = self.plugin_registry.get_plugin(plugin_id)
        if not plugin_info:
            raise PluginError(f"Plugin {plugin_id} not found")
        
        self.logger.info(f"Hot-reloading plugin: {plugin_id}")
        
        # Save current state
        was_enabled = plugin_info.state == PluginState.ENABLED
        
        try:
            # Unload plugin
            await self.unload_plugin(plugin_id)
            
            # Wait a moment
            await asyncio.sleep(0.1)
            
            # Reload plugin
            await self.load_plugin(plugin_id)
            
            # Re-enable if it was enabled
            if was_enabled:
                await self.enable_plugin(plugin_id)
            
            await self.event_bus.emit(PluginHotReloaded(
                plugin_id=plugin_id,
                plugin_type=plugin_info.metadata.plugin_type.value
            ))
            
            self.logger.info(f"Successfully hot-reloaded plugin: {plugin_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to hot-reload plugin {plugin_id}: {str(e)}")
            raise PluginError(f"Hot-reload failed for {plugin_id}: {str(e)}")

    def get_plugin_info(self, plugin_id: str) -> Optional[PluginInfo]:
        """Get plugin information."""
        return self.plugin_registry.get_plugin(plugin_id)

    def list_plugins(self, 
                    state: Optional[PluginState] = None,
                    plugin_type: Optional[PluginType] = None) -> List[Dict[str, Any]]:
        """List plugins with optional filtering."""
        plugins = self.plugin_registry.list_plugins(state, plugin_type)
        
        return [
            {
                "plugin_id": p.metadata.plugin_id,
                "name": p.metadata.name,
                "version": p.metadata.version,
                "type": p.metadata.plugin_type.value,
                "state": p.state.value,
                "description": p.metadata.description,
                "author": p.metadata.author,
                "load_time": p.load_time,
                "memory_usage": p.memory_usage,
                "error_count": p.error_count,
                "last_used": p.last_used.isoformat() if p.last_used else None
            }
            for p in plugins
        ]

    def get_plugin_capabilities(self) -> Dict[str, List[str]]:
        """Get all available plugin capabilities."""
        return dict(self.plugin_registry._capabilities)

    async def get_plugin_status(self) -> Dict[str, Any]:
        """Get comprehensive plugin system status."""
        plugins = self.plugin_registry.list_plugins()
        
        return {
            "total_plugins": len(plugins),
            "loaded_plugins": len([p for p in plugins if p.state == PluginState.LOADED]),
            "enabled_plugins": len([p for p in plugins if p.state == PluginState.ENABLED]),
            "failed_plugins": len([p for p in plugins if p.state == PluginState.ERROR]),
            "auto_discovery_enabled": self._auto_discovery_enabled,
            "hot_reload_enabled": self._hot_reload_enabled,
            "security_validation_enabled": self._security_validation_enabled,
            "plugin_directories": [str(d) for d in self._plugin_directories]
        }

    async def _plugin_health_monitor_loop(self) -> None:
        """Background task for plugin health monitoring."""
        while not self._shutdown_event.is_set():
            try:
                plugins = self.plugin_registry.list_plugins(PluginState.ENABLED)
                
                for plugin_info in plugins:
                    try:
                        if hasattr(plugin_info.instance, 'health_check'):
                            health_result = await plugin_info.instance.health_check()
                            plugin_info.health_status = health_result.get('status', 'unknown')
                            plugin_info.last_health_check = datetime.now(timezone.utc)
                            
                            # Check for performance issues
                            if 'memory_usage' in health_result:
                                plugin_info.memory_usage = health_result['memory_usage']
                                
                                if plugin_info.memory_usage > plugin_info.metadata.memory_limit_mb:
                                    await self.event_bus.emit(PluginPerformanceWarning(
                                        plugin_id=plugin_info.metadata.plugin_id,
                                        metric="memory_usage",
                                        value=plugin_info.memory_usage,
                                        limit=plugin_info.metadata.memory_limit_mb
                                    ))
                    
                    except Exception as e:
                        plugin_info.error_count += 1
                        plugin_info.last_error = e
                        self.logger.warning(f"Health check failed for plugin {plugin_info.metadata.plugin_id}: {str(e)}")
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Plugin health monitoring error: {str(e)}")
                await asyncio.sleep(60)

    async def _hot_reload_monitor_loop(self) -> None:
        """Background task for hot-reload monitoring."""
        if not self._hot_reload_enabled:
            return
        
        file_mtimes = {}
        
        while not self._shutdown_event.is_set():
            try:
                plugins = self.plugin_registry.list_plugins()
                
                for plugin_info in plugins:
                    if plugin_info.installation_path and plugin_info.installation_path.exists():
                        # Check for file modifications
                        for py_file in plugin_info.installation_path.rglob("*.py"):
                            try:
                                mtime = py_file.stat().st_mtime
                                file_key = str(py_file)
                                
                                if file_key in file_mtimes and file_mtimes[file_key] != mtime:
                                    self.logger.info(f"File modified: {py_file}, hot-reloading plugin {plugin_info.metadata.plugin_id}")
                                    await self.hot_reload_plugin(plugin_info.metadata.plugin_id)
                                
                                file_mtimes[file_key] = mtime
                                
                            except Exception as e:
                                self.logger.warning(f"Error checking file {py_file}: {str(e)}")
                
                await asyncio.sleep(2)  # Check every 2 seconds
                
            except Exception as e:
                self.logger.error(f"Hot-reload monitoring error: {str(e)}")
                await asyncio.sleep(2)

    async def _performance_monitor_loop(self) -> None:
        """Background task for performance monitoring."""
        while not self._shutdown_event.is_set():
            try:
                plugins = self.plugin_registry.list_plugins(PluginState.ENABLED)
                
                for plugin_info in plugins:
                    # Update metrics
                    if self.metrics:
                        self.metrics.set(
                            "plugin_memory_usage_mb",
                            plugin_info.memory_usage,
                            tags={'plugin_id': plugin_info.metadata.plugin_id}
                        )
                        
                        self.metrics.set(
                            "plugin_error_count",
                            plugin_info.error_count,
                            tags={'plugin_id': plugin_info.metadata.plugin_id}
                        )
                
                # Update global metrics
                if self.metrics:
                    enabled_count = len([p for p in plugins if p.state == PluginState.ENABLED])
                    self.metrics.set("plugins_active", enabled_count)
                
                await asyncio.sleep(30)  # Monitor every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Performance monitoring error: {str(e)}")
                await asyncio.sleep(30)

    async def _cleanup_monitor_loop(self) -> None:
        """Background task for plugin cleanup."""
        while not self._shutdown_event.is_set():
            try:
                # Cleanup failed plugins
                failed_plugins = self.plugin_registry.list_plugins(PluginState.ERROR)
                
                for plugin_info in failed_plugins:
                    # Auto-recovery for certain types of errors
                    if plugin_info.error_count < 3:
                        try:
                            self.logger.info(f"Attempting recovery for failed plugin: {plugin_info.metadata.plugin_id}")
                            await self.hot_reload_plugin(plugin_info.metadata.plugin_id)
                        except Exception as e:
                            self.logger.warning(f"Recovery failed for {plugin_info.metadata.plugin_id}: {str(e)}")
                
                await asyncio.sleep(300)  # Cleanup every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Cleanup monitoring error: {str(e)}")
                await asyncio.sleep(300)

    async def _handle_component_registered(self, event) -> None:
        """Handle component registration events."""
        # Check if any plugins are waiting for this component
        pass

    async def _handle_component_failed(self, event) -> None:
        """Handle component failure events."""
        # Check if this affects any plugins
        pass

    async def _handle_workflow_completed(self, event) -> None:
        """Handle workflow completion events."""
        # Update plugin usage statistics
        pass

    async def _handle_session_started(self, event) -> None:
        """Handle session start events."""
        # Initialize session-specific plugins
        pass

    async def _handle_session_ended(self, event) -> None:
        """Handle session end events."""
        # Cleanup session-specific plugin resources
        pass

    async def _handle_system_shutdown(self, event) -> None:
        """Handle system shutdown events."""
        self._shutdown_event.set()
        await self.shutdown()

    async def _health_check_callback(self) -> Dict[str, Any]:
        """Health check callback for the plugin manager."""
        try:
            plugins = self.plugin_registry.list_plugins()
            enabled_plugins = [p for p in plugins if p.state == PluginState.ENABLED]
            failed_plugins = [p for p in plugins if p.state == PluginState.ERROR]
            
            return {
                "status": "healthy" if len(failed_plugins) == 0 else "degraded",
                "total_plugins": len(plugins),
                "enabled_plugins": len(enabled_plugins),
                "failed_plugins": len(failed_plugins),
                "auto_discovery_enabled": self._auto_discovery_enabled,
                "hot_reload_enabled": self._hot_reload_enabled
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }

    async def shutdown(self) -> None:
        """Gracefully shutdown the plugin manager."""
        self.logger.info("Starting plugin manager shutdown...")
        
        try:
            # Cancel background tasks
            for task in self._background_tasks:
                task.cancel()
            
            if self._background_tasks:
                await asyncio.gather(*self._background_tasks, return_exceptions=True)
            
            # Disable all enabled plugins
            enabled_plugins = self.plugin_registry.list_plugins(PluginState.ENABLED)
            for plugin_info in enabled_plugins:
                try:
                    await self.disable_plugin(plugin_info.metadata.plugin_id)
                except Exception as e:
                    self.logger.error(f"Error disabling plugin {plugin_info.metadata.plugin_id}: {str(e)}")
            
            # Unload all loaded plugins
            loaded_plugins = self.plugin_registry.list_plugins(PluginState.LOADED)
            for plugin_info in loaded_plugins:
                try:
                    await self.unload_plugin(plugin_info.metadata.plugin_id, force=True)
                except Exception as e:
                    self.logger.error(f"Error unloading plugin {plugin_info.metadata.plugin_id}: {str(e)}")
            
            # Shutdown thread pool
            self.thread_pool.shutdown(wait=True)
            
            self.logger.info("Plugin manager shutdown completed")
            
        except Exception as e:
            self.logger.error(f"Error during plugin manager shutdown: {str(e)}")
            raise PluginError(f"Plugin manager shutdown failed: {str(e)}")

    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            if hasattr(self, 'thread_pool'):
                self.thread_pool.shutdown(wait=False)
        except Exception:
            pass  # Ignore cleanup errors in destructor
