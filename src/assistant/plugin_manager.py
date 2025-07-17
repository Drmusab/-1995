"""
Enhanced Plugin Manager with Skill Registry Integration
Author: Drmusab
Last Modified: 2025-07-17 21:25:00 UTC

This module provides an enhanced plugin manager that integrates with the skill registry,
includes plugin validation, sandboxing, and secure execution.
"""

import asyncio
import importlib.util
import inspect
import json
import os
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Type, Union, Callable
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import uuid
import shutil
import subprocess
import threading
import resource
import signal
from contextlib import contextmanager

import numpy as np

# Core imports
from src.core.dependency_injection import Container
from src.core.events.event_bus import EventBus
from src.core.events.event_types import (
    BaseEvent, EventCategory, EventPriority, EventSeverity,
    ComponentRegistered, ComponentUnregistered, ComponentHealthChanged,
    ComponentFailed, ErrorOccurred, SystemShutdown, WorkflowCompleted,
    SessionStarted, SessionEnded
)
from src.core.config.loader import ConfigLoader
from src.core.security.sanitization import SecuritySanitizer
from src.core.security.authorization import AuthorizationManager
from src.core.error_handling import ErrorHandler, handle_exceptions
from src.core.health_check import HealthCheck
from src.observability.monitoring.metrics import MetricsCollector
from src.observability.monitoring.tracing import TraceManager
from src.observability.logging.config import get_logger

# Import skill registry and validator
from src.skills.skill_registry import SkillRegistry, SkillInterface, SkillMetadata, SkillType
from src.skills.skill_validator import SkillValidator, ValidationReport, ValidationSeverity
from src.skills.skill_factory import SkillFactory, SkillExecutionContext


# Keep existing enums from original plugin_manager.py
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


class SecurityLevel(Enum):
    """Plugin security levels."""
    UNTRUSTED = "untrusted"      # No special permissions
    SANDBOX = "sandbox"          # Limited sandbox environment
    TRUSTED = "trusted"          # Full system access
    SYSTEM = "system"            # System-level access
    VERIFIED = "verified"        # Cryptographically verified


# Keep existing dataclasses from original plugin_manager.py
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
    load_mode: str = "eager"
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
    
    # Validation results
    validation_report: Optional[ValidationReport] = None
    validation_passed: bool = False
    
    # Skill registry integration
    skill_id: Optional[str] = None
    skill_registered: bool = False
    
    # Sandbox information
    sandbox_enabled: bool = False
    sandbox_process: Optional[Any] = None
    
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
        self.plugin_id = plugin_id
        self.error_code = error_code
        super().__init__(message)


class PluginInterface(SkillInterface):
    """Extended interface that all plugins should implement."""
    
    def get_plugin_metadata(self) -> PluginMetadata:
        """Get plugin metadata."""
        pass
    
    async def on_install(self) -> None:
        """Called when plugin is installed."""
        pass
    
    async def on_uninstall(self) -> None:
        """Called when plugin is uninstalled."""
        pass
    
    async def on_enable(self) -> None:
        """Called when plugin is enabled."""
        pass
    
    async def on_disable(self) -> None:
        """Called when plugin is disabled."""
        pass


class PluginSandbox:
    """Enhanced sandbox for secure plugin execution."""
    
    def __init__(self, plugin_id: str, security_level: SecurityLevel, resource_limits: Dict[str, Any]):
        self.plugin_id = plugin_id
        self.security_level = security_level
        self.resource_limits = resource_limits
        self.logger = get_logger(f"PluginSandbox:{plugin_id}")
        self.process = None
        self.thread = None
        self._setup_sandbox()
    
    def _setup_sandbox(self):
        """Setup sandbox environment."""
        # Create isolated environment
        self.sandbox_dir = Path(tempfile.mkdtemp(prefix=f"plugin_sandbox_{self.plugin_id}_"))
        self.sandbox_dir.chmod(0o700)
        
        # Setup virtual environment if needed
        if self.security_level == SecurityLevel.UNTRUSTED:
            self._create_virtual_env()
    
    def _create_virtual_env(self):
        """Create isolated Python virtual environment."""
        venv_path = self.sandbox_dir / "venv"
        subprocess.run([sys.executable, "-m", "venv", str(venv_path)], check=True)
        self.python_path = venv_path / "bin" / "python"
    
    @contextmanager
    def resource_limited_execution(self):
        """Context manager for resource-limited execution."""
        if self.security_level in [SecurityLevel.UNTRUSTED, SecurityLevel.SANDBOX]:
            # Set resource limits (Unix only)
            if hasattr(resource, 'RLIMIT_AS'):
                # Memory limit
                memory_limit = int(self.resource_limits.get('memory_mb', 256) * 1024 * 1024)
                resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))
            
            if hasattr(resource, 'RLIMIT_CPU'):
                # CPU time limit
                cpu_limit = int(self.resource_limits.get('cpu_seconds', 30))
                resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit, cpu_limit))
        
        try:
            yield
        finally:
            pass
    
    def execute_in_sandbox(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function in sandboxed environment."""
        if self.security_level == SecurityLevel.UNTRUSTED:
            # Execute in separate process
            return self._execute_in_process(func, *args, **kwargs)
        elif self.security_level == SecurityLevel.SANDBOX:
            # Execute in thread with restrictions
            return self._execute_in_thread(func, *args, **kwargs)
        else:
            # Trusted execution
            return func(*args, **kwargs)
    
    def _execute_in_process(self, func: Callable, *args, **kwargs) -> Any:
        """Execute in isolated process."""
        # This is a simplified version - in production, use multiprocessing
        # with proper IPC and serialization
        import multiprocessing
        
        def wrapper():
            with self.resource_limited_execution():
                return func(*args, **kwargs)
        
        process = multiprocessing.Process(target=wrapper)
        process.start()
        process.join(timeout=self.resource_limits.get('timeout_seconds', 30))
        
        if process.is_alive():
            process.terminate()
            raise PluginError(f"Plugin {self.plugin_id} execution timed out")
        
        return None  # Would need proper IPC for return values
    
    def _execute_in_thread(self, func: Callable, *args, **kwargs) -> Any:
        """Execute in thread with monitoring."""
        result = [None]
        exception = [None]
        
        def wrapper():
            try:
                with self.resource_limited_execution():
                    result[0] = func(*args, **kwargs)
            except Exception as e:
                exception[0] = e
        
        thread = threading.Thread(target=wrapper)
        thread.start()
        thread.join(timeout=self.resource_limits.get('timeout_seconds', 30))
        
        if thread.is_alive():
            # Thread timeout - this is tricky to handle safely
            raise PluginError(f"Plugin {self.plugin_id} execution timed out")
        
        if exception[0]:
            raise exception[0]
        
        return result[0]
    
    def cleanup(self):
        """Cleanup sandbox resources."""
        if self.process:
            self.process.terminate()
        if self.sandbox_dir and self.sandbox_dir.exists():
            shutil.rmtree(self.sandbox_dir)


class PluginValidator:
    """Enhanced plugin validator with security checks."""
    
    def __init__(self, skill_validator: SkillValidator, security_sanitizer: SecuritySanitizer):
        self.skill_validator = skill_validator
        self.security_sanitizer = security_sanitizer
        self.logger = get_logger(__name__)
    
    async def validate_plugin(self, plugin_path: Path, metadata: PluginMetadata) -> ValidationReport:
        """Comprehensive plugin validation."""
        # Verify plugin integrity
        if not await self._verify_plugin_integrity(plugin_path, metadata):
            raise PluginError("Plugin integrity check failed")
        
        # Check plugin signature if required
        if metadata.security_level >= SecurityLevel.TRUSTED:
            if not await self._verify_plugin_signature(plugin_path, metadata):
                raise PluginError("Plugin signature verification failed")
        
        # Scan for security issues
        security_issues = await self._scan_for_security_issues(plugin_path)
        if security_issues:
            self.logger.warning(f"Security issues found in plugin {metadata.plugin_id}: {security_issues}")
        
        # If it's a skill plugin, use skill validator
        if metadata.plugin_type == PluginType.SKILL:
            # Load the plugin module temporarily for validation
            spec = importlib.util.spec_from_file_location(
                f"plugin_temp_{metadata.plugin_id}",
                plugin_path / "__init__.py"
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Find the main class
            main_class = getattr(module, metadata.main_class) if metadata.main_class else None
            if not main_class:
                # Try to find a class that implements SkillInterface
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if issubclass(obj, SkillInterface) and obj != SkillInterface:
                        main_class = obj
                        break
            
            if main_class:
                # Create skill metadata from plugin metadata
                skill_metadata = SkillMetadata(
                    skill_id=metadata.plugin_id,
                    name=metadata.name,
                    version=metadata.version,
                    description=metadata.description,
                    author=metadata.author,
                    skill_type=SkillType.CUSTOM,
                    capabilities=[],
                    dependencies=[dep.plugin_id for dep in metadata.dependencies],
                    tags=metadata.keywords,
                    security_level=metadata.security_level.value,
                    resource_requirements={
                        'memory_mb': metadata.memory_limit_mb,
                        'cpu_percent': metadata.cpu_limit_percent
                    }
                )
                
                # Run skill validation
                report = await self.skill_validator.validate_skill(
                    metadata.plugin_id,
                    main_class,
                    skill_metadata
                )
                
                return report
        
        # For non-skill plugins, create a basic validation report
        return self._create_basic_validation_report(metadata, security_issues)
    
    async def _verify_plugin_integrity(self, plugin_path: Path, metadata: PluginMetadata) -> bool:
        """Verify plugin file integrity."""
        if metadata.checksum:
            calculated_checksum = self._calculate_checksum(plugin_path)
            return calculated_checksum == metadata.checksum
        return True
    
    async def _verify_plugin_signature(self, plugin_path: Path, metadata: PluginMetadata) -> bool:
        """Verify plugin digital signature."""
        # Implement digital signature verification
        # This is a placeholder - implement actual signature verification
        return True
    
    async def _scan_for_security_issues(self, plugin_path: Path) -> List[str]:
        """Scan plugin for security issues."""
        issues = []
        
        # Check for dangerous patterns in code
        dangerous_patterns = [
            r'exec\s*\(',
            r'eval\s*\(',
            r'__import__\s*\(',
            r'compile\s*\(',
            r'subprocess\.',
            r'os\.system\s*\(',
            r'open\s*\(.*[\'"]\/etc\/',
            r'socket\.',
        ]
        
        for py_file in plugin_path.rglob("*.py"):
            try:
                content = py_file.read_text()
                for pattern in dangerous_patterns:
                    import re
                    if re.search(pattern, content):
                        issues.append(f"Dangerous pattern '{pattern}' found in {py_file.name}")
            except Exception as e:
                issues.append(f"Could not scan {py_file.name}: {str(e)}")
        
        return issues
    
    def _calculate_checksum(self, plugin_path: Path) -> str:
        """Calculate plugin checksum."""
        sha256_hash = hashlib.sha256()
        
        for file_path in sorted(plugin_path.rglob("*")):
            if file_path.is_file():
                with open(file_path, "rb") as f:
                    for byte_block in iter(lambda: f.read(4096), b""):
                        sha256_hash.update(byte_block)
        
        return sha256_hash.hexdigest()
    
    def _create_basic_validation_report(self, metadata: PluginMetadata, 
                                       security_issues: List[str]) -> ValidationReport:
        """Create a basic validation report for non-skill plugins."""
        from src.skills.skill_validator import ValidationResult, ValidationType
        
        report = ValidationReport(
            skill_id=metadata.plugin_id,
            validation_id=str(uuid.uuid4()),
            is_valid=len(security_issues) == 0,
            security_issues=len(security_issues)
        )
        
        # Add security validation results
        for issue in security_issues:
            report.results.append(ValidationResult(
                rule_id="security_scan",
                rule_name="Security Scan",
                validation_type=ValidationType.SECURITY,
                severity=ValidationSeverity.SECURITY,
                passed=False,
                message=issue
            ))
        
        return report


class PluginLoader:
    """Enhanced plugin loader with validation and sandboxing."""
    
    def __init__(self, logger, plugin_validator: PluginValidator):
        self.logger = logger
        self.plugin_validator = plugin_validator
        self._loaded_modules = {}
        
    async def load_plugin_from_path(self, plugin_path: Path, 
                                   validate: bool = True) -> Optional[Any]:
        """Load a plugin from the specified path with validation."""
        try:
            # Check if plugin.json exists
            metadata_path = plugin_path / "plugin.json"
            if not metadata_path.exists():
                self.logger.error(f"No plugin.json found in {plugin_path}")
                return None
            
            # Load metadata
            with open(metadata_path, 'r') as f:
                metadata_dict = json.load(f)
            
            metadata = self._parse_metadata(metadata_dict)
            
            # Validate plugin if requested
            if validate:
                validation_report = await self.plugin_validator.validate_plugin(plugin_path, metadata)
                if not validation_report.is_valid:
                    self.logger.error(f"Plugin {metadata.plugin_id} validation failed")
                    return None
            
            # Load the plugin module
            module_name = f"plugins.{metadata.plugin_id}"
            spec = importlib.util.spec_from_file_location(
                module_name,
                plugin_path / "__init__.py"
            )
            
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
                
                self._loaded_modules[metadata.plugin_id] = module
                
                self.logger.info(f"Successfully loaded plugin: {metadata.plugin_id}")
                return module
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to load plugin from {plugin_path}: {str(e)}")
            return None
    
    def _parse_metadata(self, metadata_dict: Dict[str, Any]) -> PluginMetadata:
        """Parse plugin metadata from dictionary."""
        # Convert string enums to enum values
        if 'plugin_type' in metadata_dict:
            metadata_dict['plugin_type'] = PluginType(metadata_dict['plugin_type'])
        if 'security_level' in metadata_dict:
            metadata_dict['security_level'] = SecurityLevel(metadata_dict['security_level'])
        
        # Convert dependencies
        if 'dependencies' in metadata_dict:
            metadata_dict['dependencies'] = [
                PluginDependency(**dep) if isinstance(dep, dict) else dep
                for dep in metadata_dict['dependencies']
            ]
        
        # Convert capabilities
        if 'provides' in metadata_dict:
            metadata_dict['provides'] = [
                PluginCapability(**cap) if isinstance(cap, dict) else cap
                for cap in metadata_dict['provides']
            ]
        
        return PluginMetadata(**metadata_dict)
    
    async def unload_plugin(self, plugin_id: str) -> None:
        """Unload a plugin and clean up resources."""
        if plugin_id in self._loaded_modules:
            module = self._loaded_modules[plugin_id]
            
            # Remove from sys.modules
            module_name = f"plugins.{plugin_id}"
            if module_name in sys.modules:
                del sys.modules[module_name]
            
            # Remove from loaded modules
            del self._loaded_modules[plugin_id]
            
            self.logger.info(f"Unloaded plugin: {plugin_id}")


class PluginSkillAdapter:
    """Adapter to convert plugins to skills for registry integration."""
    
    def __init__(self, skill_registry: SkillRegistry, skill_factory: SkillFactory):
        self.skill_registry = skill_registry
        self.skill_factory = skill_factory
        self.logger = get_logger(__name__)
    
    async def register_plugin_as_skill(self, plugin_info: PluginInfo) -> bool:
        """Register a plugin as a skill in the skill registry."""
        if plugin_info.metadata.plugin_type != PluginType.SKILL:
            return False
        
        try:
            # Get the plugin instance
            if not plugin_info.instance:
                return False
            
            # Create skill metadata
            skill_metadata = self._create_skill_metadata(plugin_info.metadata)
            
            # Register with skill registry
            success = await self.skill_registry.register_skill(
                plugin_info.metadata.plugin_id,
                type(plugin_info.instance),
                skill_metadata
            )
            
            if success:
                plugin_info.skill_id = plugin_info.metadata.plugin_id
                plugin_info.skill_registered = True
                self.logger.info(f"Registered plugin {plugin_info.metadata.plugin_id} as skill")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to register plugin as skill: {str(e)}")
            return False
    
    async def unregister_plugin_skill(self, plugin_info: PluginInfo) -> bool:
        """Unregister a plugin from the skill registry."""
        if not plugin_info.skill_registered or not plugin_info.skill_id:
            return False
        
        try:
            success = await self.skill_registry.unregister_skill(
                plugin_info.skill_id,
                reason="plugin_unloaded"
            )
            
            if success:
                plugin_info.skill_registered = False
                self.logger.info(f"Unregistered plugin skill {plugin_info.skill_id}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to unregister plugin skill: {str(e)}")
            return False
    
    def _create_skill_metadata(self, plugin_metadata: PluginMetadata) -> SkillMetadata:
        """Convert plugin metadata to skill metadata."""
        from src.skills.skill_registry import SkillCapability as SkillCap
        
        # Convert capabilities
        capabilities = []
        for cap in plugin_metadata.provides:
            capabilities.append(SkillCap(
                name=cap.name,
                description=cap.description or "",
                input_types=[],
                output_types=[]
            ))
        
        return SkillMetadata(
            skill_id=plugin_metadata.plugin_id,
            name=plugin_metadata.name,
            version=plugin_metadata.version,
            description=plugin_metadata.description,
            author=plugin_metadata.author,
            skill_type=SkillType.CUSTOM,
            capabilities=capabilities,
            dependencies=[dep.plugin_id for dep in plugin_metadata.dependencies],
            tags=plugin_metadata.keywords,
            min_system_version=plugin_metadata.min_system_version,
            max_system_version=plugin_metadata.max_system_version,
            configuration_schema=plugin_metadata.config_schema,
            security_level=plugin_metadata.security_level.value,
            resource_requirements={
                'memory_mb': plugin_metadata.memory_limit_mb,
                'cpu_percent': plugin_metadata.cpu_limit_percent,
                'network_access': plugin_metadata.network_access,
                'file_system_access': plugin_metadata.file_system_access
            }
        )


class EnhancedPluginManager:
    """
    Enhanced Plugin Manager with proper skill registry integration,
    validation, and sandboxing capabilities.
    """
    
    def __init__(self, container: Container):
        """Initialize the enhanced plugin manager."""
        self.container = container
        self.logger = get_logger(__name__)
        self.config = container.get(ConfigLoader)
        self.event_bus = container.get(EventBus)
        self.error_handler = container.get(ErrorHandler)
        self.metrics = container.get(MetricsCollector)
        self.tracer = container.get(TraceManager)
        self.health_check = container.get(HealthCheck)
        
        # Security components
        self.security_sanitizer = container.get(SecuritySanitizer)
        self.authz_manager = container.get(AuthorizationManager)
        
        # Skill integration
        self.skill_registry = container.get(SkillRegistry)
        self.skill_validator = container.get(SkillValidator)
        self.skill_factory = container.get(SkillFactory)
        
        # Plugin components
        self.plugin_validator = PluginValidator(self.skill_validator, self.security_sanitizer)
        self.plugin_loader = PluginLoader(self.logger, self.plugin_validator)
        self.skill_adapter = PluginSkillAdapter(self.skill_registry, self.skill_factory)
        
        # Plugin storage
        self.plugins: Dict[str, PluginInfo] = {}
        self.plugins_by_type: Dict[PluginType, Set[str]] = {pt: set() for pt in PluginType}
        self.plugin_sandboxes: Dict[str, PluginSandbox] = {}
        
        # Configuration
        self.plugin_directory = Path(self.config.get("plugins.directory", "plugins"))
        self.auto_load_plugins = self.config.get("plugins.auto_load", True)
        self.enable_sandboxing = self.config.get("plugins.enable_sandboxing", True)
        self.validation_required = self.config.get("plugins.require_validation", True)
        
        # Background tasks
        self._background_tasks = []
        self._shutdown_event = asyncio.Event()
        
        # Ensure plugin directory exists
        self.plugin_directory.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("Enhanced Plugin Manager initialized")
    
    async def initialize(self) -> None:
        """Initialize the plugin manager."""
        try:
            # Register health check
            await self.health_check.register_check("plugin_manager", self._health_check_callback)
            
            # Register event handlers
            await self._register_event_handlers()
            
            # Start background tasks
            self._background_tasks = [
                asyncio.create_task(self._plugin_health_monitor_loop()),
                asyncio.create_task(self._hot_reload_monitor_loop()),
                asyncio.create_task(self._performance_monitor_loop()),
                asyncio.create_task(self._cleanup_monitor_loop())
            ]
            
            # Auto-discover and load plugins if enabled
            if self.auto_load_plugins:
                await self.discover_plugins()
                
                # Load all validated plugins
                for plugin_id, plugin_info in self.plugins.items():
                    if plugin_info.validation_passed:
                        await self.load_plugin(plugin_id)
            
            self.logger.info("Plugin manager initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize plugin manager: {str(e)}")
            raise
    
    async def _register_event_handlers(self) -> None:
        """Register event handlers."""
        self.event_bus.subscribe(ComponentRegistered, self._handle_component_registered)
        self.event_bus.subscribe(ComponentFailed, self._handle_component_failed)
        self.event_bus.subscribe(SystemShutdown, self._handle_system_shutdown)
    
    async def discover_plugins(self) -> List[str]:
        """Discover all plugins in the plugin directory."""
        discovered_plugins = []
        
        try:
            # Scan plugin directory
            for item in self.plugin_directory.iterdir():
                if item.is_dir() and (item / "plugin.json").exists():
                    plugin_id = await self._discover_single_plugin(item)
                    if plugin_id:
                        discovered_plugins.append(plugin_id)
                elif item.suffix in ['.zip', '.tar.gz']:
                    plugin_id = await self._discover_plugin_archive(item)
                    if plugin_id:
                        discovered_plugins.append(plugin_id)
            
            self.logger.info(f"Discovered {len(discovered_plugins)} plugins")
            return discovered_plugins
            
        except Exception as e:
            self.logger.error(f"Error during plugin discovery: {str(e)}")
            return discovered_plugins
    
    async def _discover_single_plugin(self, plugin_path: Path) -> Optional[str]:
        """Discover a single plugin."""
        try:
            # Load plugin metadata
            metadata_path = plugin_path / "plugin.json"
            with open(metadata_path, 'r') as f:
                metadata_dict = json.load(f)
            
            metadata = self.plugin_loader._parse_metadata(metadata_dict)
            
            # Validate plugin if required
            validation_report = None
            validation_passed = True
            
            if self.validation_required:
                try:
                    validation_report = await self.plugin_validator.validate_plugin(
                        plugin_path, metadata
                    )
                    validation_passed = validation_report.is_valid
                except Exception as e:
                    self.logger.error(f"Plugin validation failed for {metadata.plugin_id}: {str(e)}")
                    validation_passed = False
            
            # Create plugin info
            plugin_info = PluginInfo(
                metadata=metadata,
                state=PluginState.DISCOVERED,
                installation_path=plugin_path,
                validation_report=validation_report,
                validation_passed=validation_passed
            )
            
            # Store plugin info
            self.plugins[metadata.plugin_id] = plugin_info
            self.plugins_by_type[metadata.plugin_type].add(metadata.plugin_id)
            
            # Fire event
            await self.event_bus.emit(ComponentRegistered(
                component_id=metadata.plugin_id,
                component_type="plugin",
                metadata={"plugin_type": metadata.plugin_type.value}
            ))
            
            self.logger.info(f"Discovered plugin: {metadata.plugin_id} (validated: {validation_passed})")
            return metadata.plugin_id
            
        except Exception as e:
            self.logger.error(f"Failed to discover plugin at {plugin_path}: {str(e)}")
            return None
    
    async def _discover_plugin_archive(self, archive_path: Path) -> Optional[str]:
        """Discover plugin from archive."""
        # Extract to temporary directory
        temp_dir = Path(tempfile.mkdtemp())
        
        try:
            # Extract archive
            if archive_path.suffix == '.zip':
                with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
            else:
                # Handle tar.gz
                import tarfile
                with tarfile.open(archive_path, 'r:gz') as tar_ref:
                    tar_ref.extractall(temp_dir)
            
            # Find plugin.json
            for root, dirs, files in os.walk(temp_dir):
                if "plugin.json" in files:
                    plugin_path = Path(root)
                    return await self._discover_single_plugin(plugin_path)
            
            return None
            
        finally:
            # Cleanup temp directory
            shutil.rmtree(temp_dir)
    
    async def load_plugin(self, plugin_id: str) -> None:
        """Load and initialize a plugin with proper sandboxing."""
        if plugin_id not in self.plugins:
            raise PluginError(f"Plugin {plugin_id} not found")
        
        plugin_info = self.plugins[plugin_id]
        
        if plugin_info.state >= PluginState.LOADED:
            self.logger.warning(f"Plugin {plugin_id} already loaded")
            return
        
        try:
            # Check validation status
            if self.validation_required and not plugin_info.validation_passed:
                raise PluginError(f"Plugin {plugin_id} failed validation")
            
            # Create sandbox if required
            if self.enable_sandboxing and plugin_info.metadata.security_level <= SecurityLevel.SANDBOX:
                sandbox = PluginSandbox(
                    plugin_id,
                    plugin_info.metadata.security_level,
                    {
                        'memory_mb': plugin_info.metadata.memory_limit_mb,
                        'cpu_seconds': 30,
                        'timeout_seconds': 30
                    }
                )
                self.plugin_sandboxes[plugin_id] = sandbox
                plugin_info.sandbox_enabled = True
            
            # Load the plugin module
            module = await self.plugin_loader.load_plugin_from_path(
                plugin_info.installation_path,
                validate=False  # Already validated during discovery
            )
            
            if not module:
                raise PluginError(f"Failed to load plugin module for {plugin_id}")
            
            plugin_info.module = module
            plugin_info.state = PluginState.LOADED
            
            # Find and instantiate main class
            main_class = None
            if plugin_info.metadata.main_class:
                main_class = getattr(module, plugin_info.metadata.main_class, None)
            else:
                # Try to find a class that implements PluginInterface
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if issubclass(obj, PluginInterface) and obj != PluginInterface:
                        main_class = obj
                        break
            
            if main_class:
                # Create instance
                if plugin_info.sandbox_enabled:
                    sandbox = self.plugin_sandboxes[plugin_id]
                    instance = sandbox.execute_in_sandbox(main_class)
                else:
                    instance = main_class()
                
                plugin_info.instance = instance
                
                # Initialize plugin
                config = plugin_info.metadata.default_config.copy()
                await instance.initialize(config)
                
                plugin_info.state = PluginState.INITIALIZED
                
                # Register as skill if applicable
                if plugin_info.metadata.plugin_type == PluginType.SKILL:
                    success = await self.skill_adapter.register_plugin_as_skill(plugin_info)
                    if success:
                        self.logger.info(f"Plugin {plugin_id} registered as skill")
                
                # Call on_install hook
                if hasattr(instance, 'on_install'):
                    await instance.on_install()
            
            # Update metrics
            self.metrics.increment("plugin_manager.plugins_loaded")
            
            self.logger.info(f"Successfully loaded plugin: {plugin_id}")
            
        except Exception as e:
            plugin_info.state = PluginState.ERROR
            plugin_info.last_error = e
            plugin_info.error_count += 1
            
            # Cleanup sandbox if created
            if plugin_id in self.plugin_sandboxes:
                self.plugin_sandboxes[plugin_id].cleanup()
                del self.plugin_sandboxes[plugin_id]
            
            self.logger.error(f"Failed to load plugin {plugin_id}: {str(e)}")
            raise
    
    async def enable_plugin(self, plugin_id: str) -> None:
        """Enable a loaded plugin."""
        if plugin_id not in self.plugins:
            raise PluginError(f"Plugin {plugin_id} not found")
        
        plugin_info = self.plugins[plugin_id]
        
        if plugin_info.state < PluginState.INITIALIZED:
            await self.load_plugin(plugin_id)
        
        if plugin_info.instance:
            # Call on_enable hook
            if hasattr(plugin_info.instance, 'on_enable'):
                await plugin_info.instance.on_enable()
            
            plugin_info.state = PluginState.ENABLED
            
            # Activate in skill registry if applicable
            if plugin_info.skill_registered:
                await self.skill_registry.update_skill_state(
                    plugin_info.skill_id,
                    SkillType.ACTIVE
                )
            
            self.logger.info(f"Enabled plugin: {plugin_id}")
    
    async def disable_plugin(self, plugin_id: str) -> None:
        """Disable a plugin."""
        if plugin_id not in self.plugins:
            raise PluginError(f"Plugin {plugin_id} not found")
        
        plugin_info = self.plugins[plugin_id]
        
        if plugin_info.instance:
            # Call on_disable hook
            if hasattr(plugin_info.instance, 'on_disable'):
                await plugin_info.instance.on_disable()
            
            plugin_info.state = PluginState.DISABLED
            
            # Deactivate in skill registry if applicable
            if plugin_info.skill_registered:
                await self.skill_registry.update_skill_state(
                    plugin_info.skill_id,
                    SkillType.INACTIVE
                )
            
            self.logger.info(f"Disabled plugin: {plugin_id}")
    
    async def unload_plugin(self, plugin_id: str, force: bool = False) -> None:
        """Unload a plugin and clean up resources."""
        if plugin_id not in self.plugins:
            raise PluginError(f"Plugin {plugin_id} not found")
        
        plugin_info = self.plugins[plugin_id]
        
        try:
            # Disable first if enabled
            if plugin_info.state == PluginState.ENABLED:
                await self.disable_plugin(plugin_id)
            
            # Unregister from skill registry
            if plugin_info.skill_registered:
                await self.skill_adapter.unregister_plugin_skill(plugin_info)
            
            # Call on_uninstall hook
            if plugin_info.instance and hasattr(plugin_info.instance, 'on_uninstall'):
                await plugin_info.instance.on_uninstall()
            
            # Cleanup instance
            if plugin_info.instance and hasattr(plugin_info.instance, 'cleanup'):
                await plugin_info.instance.cleanup()
            
            # Unload module
            await self.plugin_loader.unload_plugin(plugin_id)
            
            # Cleanup sandbox
            if plugin_id in self.plugin_sandboxes:
                self.plugin_sandboxes[plugin_id].cleanup()
                del self.plugin_sandboxes[plugin_id]
            
            # Update state
            plugin_info.state = PluginState.UNLOADED
            plugin_info.module = None
            plugin_info.instance = None
            
            # Fire event
            await self.event_bus.emit(ComponentUnregistered(
                component_id=plugin_id,
                component_type="plugin",
                reason="unloaded"
            ))
            
            self.logger.info(f"Unloaded plugin: {plugin_id}")
            
        except Exception as e:
            if not force:
                raise
            self.logger.error(f"Error unloading plugin {plugin_id}: {str(e)}")
    
    async def hot_reload_plugin(self, plugin_id: str) -> None:
        """Hot reload a plugin without full restart."""
        if plugin_id not in self.plugins:
            raise PluginError(f"Plugin {plugin_id} not found")
        
        plugin_info = self.plugins[plugin_id]
        original_state = plugin_info.state
        
        try:
            # Unload the plugin
            await self.unload_plugin(plugin_id, force=True)
            
            # Re-discover the plugin
            await self._discover_single_plugin(plugin_info.installation_path)
            
            # Reload to original state
            if original_state >= PluginState.LOADED:
                await self.load_plugin(plugin_id)
            
            if original_state == PluginState.ENABLED:
                await self.enable_plugin(plugin_id)
            
            self.logger.info(f"Hot reloaded plugin: {plugin_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to hot reload plugin {plugin_id}: {str(e)}")
            raise
    
    def get_plugin_info(self, plugin_id: str) -> Optional[PluginInfo]:
        """Get information about a specific plugin."""
        return self.plugins.get(plugin_id)
    
    def list_plugins(self, 
                    plugin_type: Optional[PluginType] = None,
                    state: Optional[PluginState] = None,
                    validated_only: bool = False) -> List[PluginInfo]:
        """List plugins with optional filters."""
        plugins = list(self.plugins.values())
        
        if plugin_type:
            plugins = [p for p in plugins if p.metadata.plugin_type == plugin_type]
        
        if state:
            plugins = [p for p in plugins if p.state == state]
        
        if validated_only:
            plugins = [p for p in plugins if p.validation_passed]
        
        return plugins
    
    def get_plugin_capabilities(self) -> Dict[str, List[str]]:
        """Get all capabilities provided by loaded plugins."""
        capabilities = {}
        
        for plugin_info in self.plugins.values():
            if plugin_info.state >= PluginState.LOADED:
                for capability in plugin_info.metadata.provides:
                    if capability.name not in capabilities:
                        capabilities[capability.name] = []
                    capabilities[capability.name].append(plugin_info.metadata.plugin_id)
        
        return capabilities
    
    async def get_plugin_status(self) -> Dict[str, Any]:
        """Get comprehensive plugin system status."""
        total_plugins = len(self.plugins)
        
        status = {
            "total_plugins": total_plugins,
            "plugins_by_type": {
                pt.value: len(self.plugins_by_type[pt])
                for pt in PluginType
            },
            "plugins_by_state": {},
            "sandboxed_plugins": len(self.plugin_sandboxes),
            "skill_plugins": sum(1 for p in self.plugins.values() if p.skill_registered),
            "validation_failures": sum(1 for p in self.plugins.values() if not p.validation_passed),
            "error_count": sum(p.error_count for p in self.plugins.values()),
            "total_activations": sum(p.activation_count for p in self.plugins.values())
        }
        
        # Count by state
        for plugin_info in self.plugins.values():
            state_name = plugin_info.state.value
            status["plugins_by_state"][state_name] = status["plugins_by_state"].get(state_name, 0) + 1
        
        return status
    
    async def _plugin_health_monitor_loop(self) -> None:
        """Monitor plugin health."""
        while not self._shutdown_event.is_set():
            try:
                for plugin_id, plugin_info in self.plugins.items():
                    if plugin_info.state == PluginState.ENABLED and plugin_info.instance:
                        try:
                            # Check plugin health
                            if hasattr(plugin_info.instance, 'health_check'):
                                health = await plugin_info.instance.health_check()
                                plugin_info.health_status = health.get('status', 'unknown')
                                plugin_info.last_health_check = datetime.now(timezone.utc)
                        except Exception as e:
                            plugin_info.health_status = 'error'
                            plugin_info.error_count += 1
                            self.logger.error(f"Health check failed for plugin {plugin_id}: {str(e)}")
                
                await asyncio.sleep(60)  # Check every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in plugin health monitor: {str(e)}")
                await asyncio.sleep(60)
    
    async def _hot_reload_monitor_loop(self) -> None:
        """Monitor for plugin changes and hot reload."""
        if not self.config.get("plugins.hot_reload", False):
            return
        
        plugin_mtimes = {}
        
        while not self._shutdown_event.is_set():
            try:
                for plugin_id, plugin_info in self.plugins.items():
                    if plugin_info.installation_path:
                        # Check if plugin files have changed
                        current_mtime = max(
                            f.stat().st_mtime
                            for f in plugin_info.installation_path.rglob("*.py")
                            if f.is_file()
                        )
                        
                        if plugin_id in plugin_mtimes and current_mtime > plugin_mtimes[plugin_id]:
                            self.logger.info(f"Detected changes in plugin {plugin_id}, hot reloading...")
                            await self.hot_reload_plugin(plugin_id)
                        
                        plugin_mtimes[plugin_id] = current_mtime
                
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in hot reload monitor: {str(e)}")
                await asyncio.sleep(5)
    
    async def _performance_monitor_loop(self) -> None:
        """Monitor plugin performance metrics."""
        while not self._shutdown_event.is_set():
            try:
                for plugin_id, plugin_info in self.plugins.items():
                    if plugin_info.state == PluginState.ENABLED:
                        # Collect performance metrics
                        if plugin_id in self.plugin_sandboxes:
                            # Get sandbox resource usage
                            # This would need actual implementation
                            pass
                
                await asyncio.sleep(30)  # Monitor every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in performance monitor: {str(e)}")
                await asyncio.sleep(30)
    
    async def _cleanup_monitor_loop(self) -> None:
        """Clean up stale plugin resources."""
        while not self._shutdown_event.is_set():
            try:
                # Clean up failed plugins
                for plugin_id, plugin_info in list(self.plugins.items()):
                    if plugin_info.state == PluginState.ERROR and plugin_info.error_count > 5:
                        self.logger.warning(f"Removing failed plugin {plugin_id} after {plugin_info.error_count} errors")
                        await self.unload_plugin(plugin_id, force=True)
                        del self.plugins[plugin_id]
                
                await asyncio.sleep(300)  # Clean up every 5 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in cleanup monitor: {str(e)}")
                await asyncio.sleep(300)
    
    async def _handle_component_registered(self, event) -> None:
        """Handle component registration events."""
        # Check if any plugins depend on this component
        component_id = event.component_id
        
        for plugin_id, plugin_info in self.plugins.items():
            if component_id in [dep.plugin_id for dep in plugin_info.metadata.dependencies]:
                if plugin_info.state == PluginState.DISCOVERED:
                    # Try to load the plugin now that dependency is available
                    self.logger.info(f"Dependency {component_id} available, loading plugin {plugin_id}")
                    await self.load_plugin(plugin_id)
    
    async def _handle_component_failed(self, event) -> None:
        """Handle component failure events."""
        component_id = event.component_id
        
        # Check if this is a plugin
        if component_id in self.plugins:
            plugin_info = self.plugins[component_id]
            plugin_info.state = PluginState.ERROR
            plugin_info.error_count += 1
    
    async def _handle_system_shutdown(self, event) -> None:
        """Handle system shutdown event."""
        self._shutdown_event.set()
        
        # Disable all plugins
        for plugin_id in list(self.plugins.keys()):
            try:
                await self.unload_plugin(plugin_id, force=True)
            except Exception as e:
                self.logger.error(f"Error unloading plugin {plugin_id} during shutdown: {str(e)}")
    
    async def _health_check_callback(self) -> Dict[str, Any]:
        """Health check callback."""
        try:
            total_plugins = len(self.plugins)
            enabled_plugins = sum(1 for p in self.plugins.values() if p.state == PluginState.ENABLED)
            failed_plugins = sum(1 for p in self.plugins.values() if p.state == PluginState.ERROR)
            
            health_status = "healthy"
            if failed_plugins > total_plugins * 0.2:  # More than 20% failed
                health_status = "degraded"
            elif failed_plugins > total_plugins * 0.5:  # More than 50% failed
                health_status = "unhealthy"
            
            return {
                "status": health_status,
                "total_plugins": total_plugins,
                "enabled_plugins": enabled_plugins,
                "failed_plugins": failed_plugins,
                "sandboxed_plugins": len(self.plugin_sandboxes)
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def shutdown(self) -> None:
        """Shutdown the plugin manager."""
        self.logger.info("Shutting down plugin manager...")
        
        # Signal shutdown
        self._shutdown_event.set()
        
        # Cancel background tasks
        for task in self._background_tasks:
            if not task.done():
                task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self._background_tasks, return_exceptions=True)
        
        # Unload all plugins
        for plugin_id in list(self.plugins.keys()):
            try:
                await self.unload_plugin(plugin_id, force=True)
            except Exception as e:
                self.logger.error(f"Error unloading plugin {plugin_id}: {str(e)}")
        
        # Cleanup sandboxes
        for sandbox in self.plugin_sandboxes.values():
            sandbox.cleanup()
        
        self.plugin_sandboxes.clear()
        self.plugins.clear()
        
        self.logger.info("Plugin manager shutdown complete")
