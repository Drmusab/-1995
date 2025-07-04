"""
Advanced Configuration Management System
Author: Drmusab
Last Modified: 2025-06-13 11:30:53 UTC

This module provides comprehensive configuration management for the AI assistant,
supporting multiple sources, dynamic reloading, validation, encryption, and
seamless integration with all core system components.
"""

from pathlib import Path
from typing import Optional, Dict, Any, List, Set, Callable, Type, Union, AsyncGenerator, TypeVar
import asyncio
import threading
import time
import os
import json
import re
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
from contextlib import asynccontextmanager
import logging
import hashlib
import weakref
from abc import ABC, abstractmethod
import uuid
import copy
from collections import defaultdict, ChainMap
from concurrent.futures import ThreadPoolExecutor
import tempfile
import shutil

# External imports
import yaml
import toml
import jsonschema
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import aiofiles
import aiohttp

# Core imports (will be available when integrated)
from src.core.events.event_bus import EventBus
from src.core.events.event_types import (
    ConfigurationLoaded, ConfigurationReloaded, ConfigurationChanged,
    ConfigurationValidationFailed, ConfigurationSourceAdded,
    ConfigurationSourceRemoved, ConfigurationEncrypted, ConfigurationDecrypted,
    ErrorOccurred, SystemStateChanged
)
from src.core.error_handling import ErrorHandler, handle_exceptions
from src.core.dependency_injection import Container
from src.core.health_check import HealthCheck
from src.core.security.encryption import EncryptionManager
from src.observability.logging.config import get_logger

# Type definitions
T = TypeVar('T')


class ConfigSource(Enum):
    """Configuration source types."""
    FILE = "file"
    ENVIRONMENT = "environment"
    REMOTE = "remote"
    DATABASE = "database"
    VAULT = "vault"
    CONSUL = "consul"
    ETCD = "etcd"
    AWS_PARAMETER_STORE = "aws_parameter_store"
    AZURE_KEY_VAULT = "azure_key_vault"
    GCP_SECRET_MANAGER = "gcp_secret_manager"


class ConfigFormat(Enum):
    """Configuration file formats."""
    YAML = "yaml"
    JSON = "json"
    TOML = "toml"
    INI = "ini"
    PROPERTIES = "properties"
    XML = "xml"


class ConfigPriority(Enum):
    """Configuration source priorities."""
    LOWEST = 0
    LOW = 1
    NORMAL = 2
    HIGH = 3
    HIGHEST = 4
    OVERRIDE = 5


class ReloadStrategy(Enum):
    """Configuration reload strategies."""
    MANUAL = "manual"
    FILE_WATCH = "file_watch"
    POLLING = "polling"
    EVENT_DRIVEN = "event_driven"
    ON_ACCESS = "on_access"


@dataclass
class ConfigSourceInfo:
    """Information about a configuration source."""
    source_id: str
    source_type: ConfigSource
    location: str  # file path, URL, database connection, etc.
    format: ConfigFormat = ConfigFormat.YAML
    priority: ConfigPriority = ConfigPriority.NORMAL
    reload_strategy: ReloadStrategy = ReloadStrategy.FILE_WATCH
    reload_interval: float = 30.0  # seconds
    encryption_enabled: bool = False
    validation_schema: Optional[Dict[str, Any]] = None
    environment_filter: Optional[str] = None  # only load for specific environments
    tags: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Runtime information
    last_loaded: Optional[datetime] = None
    last_modified: Optional[datetime] = None
    load_count: int = 0
    error_count: int = 0
    last_error: Optional[Exception] = None
    checksum: Optional[str] = None


@dataclass
class ConfigValue:
    """Wrapper for configuration values with metadata."""
    value: Any
    source_id: str
    priority: ConfigPriority
    loaded_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    encrypted: bool = False
    validation_passed: bool = True
    ttl: Optional[float] = None  # Time to live in seconds
    tags: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_expired(self) -> bool:
        """Check if the value has expired."""
        if self.ttl is None:
            return False
        
        elapsed = (datetime.now(timezone.utc) - self.loaded_at).total_seconds()
        return elapsed > self.ttl


@dataclass
class ConfigurationSnapshot:
    """Complete configuration snapshot."""
    snapshot_id: str
    timestamp: datetime
    environment: str
    sources: List[str]
    configuration: Dict[str, Any]
    checksum: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class ConfigError(Exception):
    """Custom exception for configuration operations."""
    
    def __init__(self, message: str, source_id: Optional[str] = None, 
                 error_code: Optional[str] = None):
        super().__init__(message)
        self.source_id = source_id
        self.error_code = error_code
        self.timestamp = datetime.now(timezone.utc)


class ConfigurationProvider(ABC):
    """Abstract base class for configuration providers."""
    
    @abstractmethod
    async def load_config(self, source_info: ConfigSourceInfo) -> Dict[str, Any]:
        """Load configuration from the source."""
        pass
    
    @abstractmethod
    def can_handle(self, source_type: ConfigSource) -> bool:
        """Check if this provider can handle the source type."""
        pass
    
    @abstractmethod
    async def watch_for_changes(self, source_info: ConfigSourceInfo, 
                               callback: Callable[[Dict[str, Any]], None]) -> None:
        """Watch for configuration changes."""
        pass


class FileConfigProvider(ConfigurationProvider):
    """Configuration provider for file-based sources."""
    
    def __init__(self, logger):
        self.logger = logger
        self._watchers: Dict[str, Observer] = {}
    
    def can_handle(self, source_type: ConfigSource) -> bool:
        """Check if this provider handles file sources."""
        return source_type == ConfigSource.FILE
    
    async def load_config(self, source_info: ConfigSourceInfo) -> Dict[str, Any]:
        """Load configuration from a file."""
        file_path = Path(source_info.location)
        
        if not file_path.exists():
            raise ConfigError(f"Configuration file not found: {file_path}", source_info.source_id)
        
        try:
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                content = await f.read()
            
            # Parse based on format
            if source_info.format == ConfigFormat.YAML:
                config = yaml.safe_load(content)
            elif source_info.format == ConfigFormat.JSON:
                config = json.loads(content)
            elif source_info.format == ConfigFormat.TOML:
                config = toml.loads(content)
            else:
                raise ConfigError(f"Unsupported format: {source_info.format}", source_info.source_id)
            
            return config or {}
            
        except Exception as e:
            raise ConfigError(f"Failed to load config from {file_path}: {str(e)}", source_info.source_id)
    
    async def watch_for_changes(self, source_info: ConfigSourceInfo, 
                               callback: Callable[[Dict[str, Any]], None]) -> None:
        """Watch file for changes."""
        if source_info.reload_strategy != ReloadStrategy.FILE_WATCH:
            return
        
        file_path = Path(source_info.location)
        if not file_path.exists():
            return
        
        class ConfigFileHandler(FileSystemEventHandler):
            def __init__(self, provider, source, cb):
                self.provider = provider
                self.source = source
                self.callback = cb
            
            def on_modified(self, event):
                if not event.is_directory and Path(event.src_path) == file_path:
                    asyncio.create_task(self._handle_change())
            
            async def _handle_change(self):
                try:
                    config = await self.provider.load_config(self.source)
                    self.callback(config)
                except Exception as e:
                    self.provider.logger.error(f"Error reloading config: {str(e)}")
        
        # Stop existing watcher
        if source_info.source_id in self._watchers:
            self._watchers[source_info.source_id].stop()
        
        # Start new watcher
        observer = Observer()
        handler = ConfigFileHandler(self, source_info, callback)
        observer.schedule(handler, str(file_path.parent), recursive=False)
        observer.start()
        
        self._watchers[source_info.source_id] = observer


class EnvironmentConfigProvider(ConfigurationProvider):
    """Configuration provider for environment variables."""
    
    def __init__(self, logger):
        self.logger = logger
    
    def can_handle(self, source_type: ConfigSource) -> bool:
        """Check if this provider handles environment sources."""
        return source_type == ConfigSource.ENVIRONMENT
    
    async def load_config(self, source_info: ConfigSourceInfo) -> Dict[str, Any]:
        """Load configuration from environment variables."""
        prefix = source_info.location or ""
        config = {}
        
        for key, value in os.environ.items():
            if prefix and not key.startswith(prefix):
                continue
            
            # Remove prefix and convert to nested structure
            config_key = key[len(prefix):] if prefix else key
            config_key = config_key.lstrip('_').lower()
            
            # Convert underscores to nested structure
            keys = config_key.split('_')
            current = config
            
            for i, k in enumerate(keys[:-1]):
                if k not in current:
                    current[k] = {}
                current = current[k]
            
            # Try to parse value as JSON, fallback to string
            try:
                current[keys[-1]] = json.loads(value)
            except (json.JSONDecodeError, ValueError):
                current[keys[-1]] = value
        
        return config
    
    async def watch_for_changes(self, source_info: ConfigSourceInfo, 
                               callback: Callable[[Dict[str, Any]], None]) -> None:
        """Environment variables don't support watching."""
        pass


class RemoteConfigProvider(ConfigurationProvider):
    """Configuration provider for remote HTTP sources."""
    
    def __init__(self, logger):
        self.logger = logger
        self._session: Optional[aiohttp.ClientSession] = None
    
    def can_handle(self, source_type: ConfigSource) -> bool:
        """Check if this provider handles remote sources."""
        return source_type == ConfigSource.REMOTE
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session
    
    async def load_config(self, source_info: ConfigSourceInfo) -> Dict[str, Any]:
        """Load configuration from remote URL."""
        session = await self._get_session()
        
        try:
            timeout = aiohttp.ClientTimeout(total=30)
            async with session.get(source_info.location, timeout=timeout) as response:
                response.raise_for_status()
                content = await response.text()
            
            # Parse based on format
            if source_info.format == ConfigFormat.YAML:
                config = yaml.safe_load(content)
            elif source_info.format == ConfigFormat.JSON:
                config = json.loads(content)
            elif source_info.format == ConfigFormat.TOML:
                config = toml.loads(content)
            else:
                raise ConfigError(f"Unsupported format: {source_info.format}", source_info.source_id)
            
            return config or {}
            
        except Exception as e:
            raise ConfigError(f"Failed to load remote config from {source_info.location}: {str(e)}", 
                            source_info.source_id)
    
    async def watch_for_changes(self, source_info: ConfigSourceInfo, 
                               callback: Callable[[Dict[str, Any]], None]) -> None:
        """Poll remote source for changes."""
        if source_info.reload_strategy != ReloadStrategy.POLLING:
            return
        
        async def poll_remote():
            last_checksum = None
            
            while True:
                try:
                    await asyncio.sleep(source_info.reload_interval)
                    
                    config = await self.load_config(source_info)
                    current_checksum = hashlib.md5(
                        json.dumps(config, sort_keys=True).encode()
                    ).hexdigest()
                    
                    if last_checksum is not None and current_checksum != last_checksum:
                        callback(config)
                    
                    last_checksum = current_checksum
                    
                except Exception as e:
                    self.logger.error(f"Error polling remote config: {str(e)}")
        
        asyncio.create_task(poll_remote())


class ConfigValidator:
    """Configuration validation using JSON Schema."""
    
    def __init__(self, logger):
        self.logger = logger
        self._schemas: Dict[str, Dict[str, Any]] = {}
    
    def register_schema(self, schema_id: str, schema: Dict[str, Any]) -> None:
        """Register a validation schema."""
        try:
            # Validate the schema itself
            jsonschema.Draft7Validator.check_schema(schema)
            self._schemas[schema_id] = schema
            self.logger.debug(f"Registered validation schema: {schema_id}")
        except Exception as e:
            raise ConfigError(f"Invalid schema {schema_id}: {str(e)}")
    
    def validate(self, config: Dict[str, Any], schema_id: Optional[str] = None,
                schema: Optional[Dict[str, Any]] = None) -> List[str]:
        """Validate configuration against schema."""
        errors = []
        
        try:
            validation_schema = None
            
            if schema:
                validation_schema = schema
            elif schema_id and schema_id in self._schemas:
                validation_schema = self._schemas[schema_id]
            
            if validation_schema:
                validator = jsonschema.Draft7Validator(validation_schema)
                for error in validator.iter_errors(config):
                    errors.append(f"{'.'.join(str(p) for p in error.path)}: {error.message}")
            
        except Exception as e:
            errors.append(f"Validation error: {str(e)}")
        
        return errors


class ConfigInterpolator:
    """Configuration value interpolation and variable substitution."""
    
    def __init__(self, logger):
        self.logger = logger
        self._variable_pattern = re.compile(r'\$\{([^}]+)\}')
        self._env_pattern = re.compile(r'\$\{env:([^}]+)\}')
        self._file_pattern = re.compile(r'\$\{file:([^}]+)\}')
    
    def interpolate(self, config: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Interpolate variables in configuration."""
        context = context or {}
        result = copy.deepcopy(config)
        
        # Add environment variables to context
        context.update({f"env:{k}": v for k, v in os.environ.items()})
        
        return self._interpolate_recursive(result, context)
    
    def _interpolate_recursive(self, obj: Any, context: Dict[str, Any]) -> Any:
        """Recursively interpolate variables."""
        if isinstance(obj, dict):
            return {k: self._interpolate_recursive(v, context) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._interpolate_recursive(item, context) for item in obj]
        elif isinstance(obj, str):
            return self._interpolate_string(obj, context)
        else:
            return obj
    
    def _interpolate_string(self, value: str, context: Dict[str, Any]) -> str:
        """Interpolate variables in a string value."""
        result = value
        
        # Environment variable interpolation
        for match in self._env_pattern.finditer(value):
            var_name = match.group(1)
            env_value = os.environ.get(var_name, f"${{{match.group(0)}}}")
            result = result.replace(match.group(0), env_value)
        
        # File content interpolation
        for match in self._file_pattern.finditer(value):
            file_path = match.group(1)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    file_content = f.read().strip()
                result = result.replace(match.group(0), file_content)
            except Exception as e:
                self.logger.warning(f"Failed to read file {file_path}: {str(e)}")
        
        # General variable interpolation
        for match in self._variable_pattern.finditer(value):
            var_name = match.group(1)
            if var_name in context:
                var_value = str(context[var_name])
                result = result.replace(match.group(0), var_value)
        
        return result


class ConfigCache:
    """Configuration caching with TTL support."""
    
    def __init__(self, logger, default_ttl: float = 300.0):
        self.logger = logger
        self.default_ttl = default_ttl
        self._cache: Dict[str, ConfigValue] = {}
        self._lock = threading.RLock()
    
    def get(self, key: str) -> Optional[ConfigValue]:
        """Get value from cache."""
        with self._lock:
            if key in self._cache:
                value = self._cache[key]
                if not value.is_expired():
                    return value
                else:
                    del self._cache[key]
            return None
    
    def set(self, key: str, value: ConfigValue) -> None:
        """Set value in cache."""
        with self._lock:
            self._cache[key] = value
    
    def invalidate(self, key: Optional[str] = None) -> None:
        """Invalidate cache entries."""
        with self._lock:
            if key:
                self._cache.pop(key, None)
            else:
                self._cache.clear()
    
    def cleanup_expired(self) -> int:
        """Remove expired entries and return count."""
        with self._lock:
            expired_keys = [
                key for key, value in self._cache.items()
                if value.is_expired()
            ]
            
            for key in expired_keys:
                del self._cache[key]
            
            return len(expired_keys)


class ConfigLoader:
    """
    Advanced Configuration Management System for the AI Assistant.
    
    This loader provides comprehensive configuration management including:
    - Multiple configuration sources (files, environment, remote)
    - Dynamic hot-reloading with file watching
    - Hierarchical configuration merging with priorities
    - Variable interpolation and environment substitution
    - JSON Schema validation
    - Encryption support for sensitive values
    - Configuration caching with TTL
    - Event-driven configuration updates
    - Environment-specific configurations
    - Configuration snapshots and rollback
    - Health monitoring and metrics
    """
    
    def __init__(self, environment: str = "development", container: Optional[Container] = None):
        """
        Initialize the configuration loader.
        
        Args:
            environment: Current environment (development, staging, production)
            container: Optional dependency injection container
        """
        self.environment = environment
        self.container = container
        self.logger = get_logger(__name__)
        
        # Core components
        self.event_bus: Optional[EventBus] = None
        self.error_handler: Optional[ErrorHandler] = None
        self.health_check: Optional[HealthCheck] = None
        self.encryption_manager: Optional[EncryptionManager] = None
        
        # Configuration management
        self._sources: Dict[str, ConfigSourceInfo] = {}
        self._providers: List[ConfigurationProvider] = []
        self._configurations: Dict[str, Dict[str, Any]] = {}
        self._merged_config: Dict[str, Any] = {}
        
        # Support components
        self._validator = ConfigValidator(self.logger)
        self._interpolator = ConfigInterpolator(self.logger)
        self._cache = ConfigCache(self.logger)
        
        # State management
        self._initialized = False
        self._loading_lock = asyncio.Lock()
        self._reload_tasks: Dict[str, asyncio.Task] = {}
        self._snapshots: List[ConfigurationSnapshot] = []
        
        # Configuration
        self._max_snapshots = 10
        self._auto_reload_enabled = True
        self._encryption_enabled = False
        self._validation_enabled = True
        self._caching_enabled = True
        
        # Performance tracking
        self._load_times: Dict[str, List[float]] = defaultdict(list)
        self._error_counts: Dict[str, int] = defaultdict(int)
        
        # Initialize providers
        self._setup_providers()
        
        # Try to get dependencies from container
        if container:
            try:
                self.event_bus = container.get(EventBus)
                self.error_handler = container.get(ErrorHandler)
                self.health_check = container.get(HealthCheck)
                self.encryption_manager = container.get(EncryptionManager)
                
                # Register health check
                self.health_check.register_component("config_loader", self._health_check_callback)
                
            except Exception as e:
                self.logger.warning(f"Could not get dependencies from container: {str(e)}")
        
        self.logger.info(f"ConfigLoader initialized for environment: {environment}")

    def _setup_providers(self) -> None:
        """Setup configuration providers."""
        self._providers = [
            FileConfigProvider(self.logger),
            EnvironmentConfigProvider(self.logger),
            RemoteConfigProvider(self.logger)
        ]
        
        self.logger.debug(f"Initialized {len(self._providers)} configuration providers")

    async def initialize(self) -> None:
        """Initialize the configuration loader."""
        if self._initialized:
            return
        
        async with self._loading_lock:
            try:
                self.logger.info("Initializing configuration loader...")
                
                # Register default schemas
                await self._register_default_schemas()
                
                # Load default configurations
                await self._load_default_configurations()
                
                # Start background tasks
                asyncio.create_task(self._cache_cleanup_loop())
                
                self._initialized = True
                
                # Emit initialization event
                if self.event_bus:
                    await self.event_bus.emit(ConfigurationLoaded(
                        environment=self.environment,
                        sources=list(self._sources.keys()),
                        total_keys=len(self._get_all_keys())
                    ))
                
                self.logger.info("Configuration loader initialized successfully")
                
            except Exception as e:
                self.logger.error(f"Failed to initialize configuration loader: {str(e)}")
                raise ConfigError(f"Initialization failed: {str(e)}")

    async def _register_default_schemas(self) -> None:
        """Register default validation schemas."""
        # Core system schema
        core_schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": {
                "environment": {"type": "string"},
                "debug": {"type": "boolean"},
                "logging": {
                    "type": "object",
                    "properties": {
                        "level": {"type": "string", "enum": ["DEBUG", "INFO", "WARNING", "ERROR"]},
                        "format": {"type": "string"}
                    }
                }
            }
        }
        
        self._validator.register_schema("core", core_schema)
        
        # AI Assistant specific schemas
        assistant_schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": {
                "assistant": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "version": {"type": "string"},
                        "max_sessions": {"type": "integer", "minimum": 1},
                        "session_timeout": {"type": "number", "minimum": 0}
                    }
                },
                "models": {
                    "type": "object",
                    "properties": {
                        "default_provider": {"type": "string"},
                        "providers": {"type": "object"}
                    }
                }
            }
        }
        
        self._validator.register_schema("assistant", assistant_schema)

    async def _load_default_configurations(self) -> None:
        """Load default configuration sources."""
        # Default configuration paths
        default_sources = [
            ("base_config", ConfigSource.FILE, "configs/base.yaml", ConfigFormat.YAML, ConfigPriority.LOW),
            ("env_config", ConfigSource.FILE, f"configs/environments/{self.environment}.yaml", 
             ConfigFormat.YAML, ConfigPriority.NORMAL),
            ("local_config", ConfigSource.FILE, "configs/local.yaml", ConfigFormat.YAML, ConfigPriority.HIGH),
            ("env_vars", ConfigSource.ENVIRONMENT, "AI_ASSISTANT_", ConfigFormat.JSON, ConfigPriority.HIGHEST)
        ]
        
        for source_id, source_type, location, format_type, priority in default_sources:
            try:
                await self.add_source(
                    source_id=source_id,
                    source_type=source_type,
                    location=location,
                    format_type=format_type,
                    priority=priority,
                    load_immediately=True
                )
            except Exception as e:
                # Don't fail if optional configs are missing
                if source_id in ["local_config"]:
                    self.logger.debug(f"Optional config {source_id} not found: {str(e)}")
                else:
                    self.logger.warning(f"Failed to load default config {source_id}: {str(e)}")

    @handle_exceptions
    async def add_source(
        self,
        source_id: str,
        source_type: ConfigSource,
        location: str,
        format_type: ConfigFormat = ConfigFormat.YAML,
        priority: ConfigPriority = ConfigPriority.NORMAL,
        reload_strategy: ReloadStrategy = ReloadStrategy.FILE_WATCH,
        reload_interval: float = 30.0,
        encryption_enabled: bool = False,
        validation_schema: Optional[str] = None,
        environment_filter: Optional[str] = None,
        load_immediately: bool = True,
        **kwargs
    ) -> None:
        """
        Add a configuration source.
        
        Args:
            source_id: Unique identifier for the source
            source_type: Type of configuration source
            location: Source location (file path, URL, etc.)
            format_type: Configuration format
            priority: Loading priority
            reload_strategy: How to handle reloading
            reload_interval: Reload interval for polling
            encryption_enabled: Whether values are encrypted
            validation_schema: Schema ID for validation
            environment_filter: Only load for specific environment
            load_immediately: Whether to load immediately
            **kwargs: Additional metadata
        """
        if source_id in self._sources:
            raise ConfigError(f"Configuration source {source_id} already exists")
        
        # Skip if environment filter doesn't match
        if environment_filter and environment_filter != self.environment:
            self.logger.debug(f"Skipping source {source_id} (environment filter: {environment_filter})")
            return
        
        # Create source info
        source_info = ConfigSourceInfo(
            source_id=source_id,
            source_type=source_type,
            location=location,
            format=format_type,
            priority=priority,
            reload_strategy=reload_strategy,
            reload_interval=reload_interval,
            encryption_enabled=encryption_enabled,
            validation_schema=validation_schema,
            environment_filter=environment_filter,
            metadata=kwargs
        )
        
        # Find appropriate provider
        provider = None
        for p in self._providers:
            if p.can_handle(source_type):
                provider = p
                break
        
        if not provider:
            raise ConfigError(f"No provider found for source type {source_type}")
        
        # Register source
        self._sources[source_id] = source_info
        
        # Load immediately if requested
        if load_immediately:
            await self._load_source(source_id)
        
        # Setup watching for changes
        if reload_strategy in [ReloadStrategy.FILE_WATCH, ReloadStrategy.POLLING]:
            await provider.watch_for_changes(source_info, 
                                           lambda config: asyncio.create_task(self._handle_source_change(source_id, config)))
        
        # Emit source added event
        if self.event_bus:
            await self.event_bus.emit(ConfigurationSourceAdded(
                source_id=source_id,
                source_type=source_type.value,
                location=location,
                priority=priority.value
            ))
        
        self.logger.info(f"Added configuration source: {source_id} ({source_type.value})")

    async def _load_source(self, source_id: str) -> None:
        """Load configuration from a specific source."""
        if source_id not in self._sources:
            raise ConfigError(f"Configuration source {source_id} not found")
        
        source_info = self._sources[source_id]
        start_time = time.time()
        
        try:
            # Find provider
            provider = None
            for p in self._providers:
                if p.can_handle(source_info.source_type):
                    provider = p
                    break
            
            if not provider:
                raise ConfigError(f"No provider for source type {source_info.source_type}")
            
            # Load configuration
            config = await provider.load_config(source_info)
            
            # Apply interpolation
            config = self._interpolator.interpolate(config, {
                'environment': self.environment,
                'source_id': source_id
            })
            
            # Decrypt if needed
            if source_info.encryption_enabled and self.encryption_manager:
                config = await self._decrypt_config(config)
            
            # Validate if schema is specified
            if self._validation_enabled and source_info.validation_schema:
                errors = self._validator.validate(config, source_info.validation_schema)
                if errors:
                    raise ConfigError(f"Validation failed for {source_id}: {'; '.join(errors)}")
            
            # Store configuration
            self._configurations[source_id] = config
            
            # Update source info
            source_info.last_loaded = datetime.now(timezone.utc)
            source_info.load_count += 1
            source_info.checksum = hashlib.md5(
                json.dumps(config, sort_keys=True).encode()
            ).hexdigest()
            
            # Merge configurations
            await self._merge_configurations()
            
            # Track performance
            load_time = time.time() - start_time
            self._load_times[source_id].append(load_time)
            if len(self._load_times[source_id]) > 100:
                self._load_times[source_id].pop(0)
            
            self.logger.debug(f"Loaded configuration from {source_id} in {load_time:.3f}s")
            
        except Exception as e:
            source_info.error_count += 1
            source_info.last_error = e
            self._error_counts[source_id] += 1
            
            self.logger.error(f"Failed to load configuration from {source_id}: {str(e)}")
            
            if self.event_bus:
                await self.event_bus.emit(ErrorOccurred(
                    component="config_loader",
                    error_type=type(e).__name__,
                    error_message=str(e),
                    severity="warning"
                ))
            
            raise ConfigError(f"Failed to load source {source_id}: {str(e)}", source_id)

    async def _merge_configurations(self) -> None:
        """Merge configurations from all sources based on priority."""
        # Sort sources by priority
        sorted_sources = sorted(
            self._sources.items(),
            key=lambda x: x[1].priority.value
        )
        
        # Merge configurations
        merged = {}
        
        for source_id, source_info in sorted_sources:
            if source_id in self._configurations:
                config = self._configurations[source_id]
                merged = self._deep_merge(merged, config)
        
        # Store merged configuration
        old_config = self._merged_config.copy()
        self._merged_config = merged
        
        # Create snapshot
        await self._create_snapshot()
        
        # Emit configuration changed event if there are changes
        if old_config != merged and self.event_bus:
            await self.event_bus.emit(ConfigurationChanged(
                environment=self.environment,
                changed_keys=self._get_changed_keys(old_config, merged),
                total_keys=len(self._get_all_keys())
            ))

    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries, with override taking precedence."""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = copy.deepcopy(value)
        
        return result

    def _get_changed_keys(self, old: Dict[str, Any], new: Dict[str, Any], prefix: str = "") -> List[str]:
        """Get list of changed configuration keys."""
        changed = []
        
        # Check for new or modified keys
        for key, value in new.items():
            full_key = f"{prefix}.{key}" if prefix else key
            
            if key not in old:
                changed.append(full_key)
            elif isinstance(value, dict) and isinstance(old[key], dict):
                changed.extend(self._get_changed_keys(old[key], value, full_key))
            elif value != old[key]:
                changed.append(full_key)
        
        # Check for removed keys
        for key in old.keys():
            if key not in new:
                full_key = f"{prefix}.{key}" if prefix else key
                changed.append(full_key)
        
        return changed

    def _get_all_keys(self, config: Optional[Dict[str, Any]] = None, prefix: str = "") -> List[str]:
        """Get all configuration keys as dot-notation paths."""
        if config is None:
            config = self._merged_config
        
        keys = []
        
        for key, value in config.items():
            full_key = f"{prefix}.{key}" if prefix else key
            keys.append(full_key)
            
            if isinstance(value, dict):
                keys.extend(self._get_all_keys(value, full_key))
        
        return keys

    async def _decrypt_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Decrypt encrypted configuration values."""
        if not self.encryption_manager:
            return config
        
        def decrypt_recursive(obj):
            if isinstance(obj, dict):
                result = {}
                for key, value in obj.items():
                    if key.endswith('_encrypted') and isinstance(value, str):
                        # Decrypt the value
                        try:
                            decrypted = self.encryption_manager.decrypt(value)
                            # Use the key without _encrypted suffix
                            result_key = key[:-10]  # Remove '_encrypted'
                            result[result_key] = decrypted
                        except Exception as e:
                            self.logger.error(f"Failed to decrypt {key}: {str(e)}")
                            result[key] = value
                    else:
                        result[key] = decrypt_recursive(value)
                return result
            elif isinstance(obj, list):
                return [decrypt_recursive(item) for item in obj]
            else:
                return obj
        
        return decrypt_recursive(config)

    async def _handle_source_change(self, source_id: str, new_config: Dict[str, Any]) -> None:
        """Handle configuration source change."""
        try:
            self.logger.info(f"Configuration change detected in source: {source_id}")
            
            # Reload the source
            await self._load_source(source_id)
            
            # Emit reload event
            if self.event_bus:
                await self.event_bus.emit(ConfigurationReloaded(
                    source_id=source_id,
                    environment=self.environment,
                    reload_time=datetime.now(timezone.utc)
                ))
            
        except Exception as e:
            self.logger.error(f"Failed to handle configuration change for {source_id}: {str(e)}")

    async def _create_snapshot(self) -> str:
        """Create a configuration snapshot."""
        snapshot_id = str(uuid.uuid4())
        
        snapshot = ConfigurationSnapshot(
            snapshot_id=snapshot_id,
            timestamp=datetime.now(timezone.utc),
            environment=self.environment,
            sources=list(self._sources.keys()),
            configuration=copy.deepcopy(self._merged_config),
            checksum=hashlib.md5(
                json.dumps(self._merged_config, sort_keys=True).encode()
            ).hexdigest()
        )
        
        self._snapshots.append(snapshot)
        
        # Keep only recent snapshots
        if len(self._snapshots) > self._max_snapshots:
            self._snapshots.pop(0)
        
        return snapshot_id

    def get(self, key: str, default: Any = None, use_cache: bool = True) -> Any:
        """
        Get a configuration value.
        
        Args:
            key: Configuration key in dot notation (e.g., 'database.host')
            default: Default value if key not found
            use_cache: Whether to use cached value
            
        Returns:
            Configuration value or default
        """
        # Check cache first
        if use_cache and self._caching_enabled:
            cached_value = self._cache.get(key)
            if cached_value:
                return cached_value.value
        
        # Navigate to the value
        value = self._merged_config
        
        try:
            for part in key.split('.'):
                if isinstance(value, dict) and part in value:
                    value = value[part]
                else:
                    value = default
                    break
            
            # Cache the value
            if use_cache and self._caching_enabled and value != default:
                config_value = ConfigValue(
                    value=value,
                    source_id="merged",
                    priority=ConfigPriority.NORMAL,
                    ttl=self._cache.default_ttl
                )
                self._cache.set(key, config_value)
            
            return value
            
        except Exception as e:
            self.logger.error(f"Error getting configuration key {key}: {str(e)}")
            return default

    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get an entire configuration section.
        
        Args:
            section: Section name in dot notation
            
        Returns:
            Configuration section as dictionary
        """
        value = self.get(section, {})
        return value if isinstance(value, dict) else {}

    def set(self, key: str, value: Any, source_id: str = "runtime") -> None:
        """
        Set a configuration value at runtime.
        
        Args:
            key: Configuration key in dot notation
            value: Value to set
            source_id: Source identifier for tracking
        """
        # Create runtime source if it doesn't exist
        if source_id not in self._configurations:
            self._configurations[source_id] = {}
        
        # Navigate and set the value
        config = self._configurations[source_id]
        parts = key.split('.')
        
        for part in parts[:-1]:
            if part not in config:
                config[part] = {}
            config = config[part]
        
        config[parts[-1]] = value
        
        # Invalidate cache
        self._cache.invalidate(key)
        
        # Re-merge configurations
        asyncio.create_task(self._merge_configurations())
        
        self.logger.debug(f"Set configuration {key} = {value}")

    def has(self, key: str) -> bool:
        """
        Check if a configuration key exists.
        
        Args:
            key: Configuration key in dot notation
            
        Returns:
            True if key exists, False otherwise
        """
        return self.get(key, object()) is not object()

    async def reload(self, source_id: Optional[str] = None) -> None:
        """
        Reload configuration from sources.
        
        Args:
            source_id: Specific source to reload, or None for all sources
        """
        if source_id:
            if source_id in self._sources:
                await self._load_source(source_id)
            else:
                raise ConfigError(f"Configuration source {source_id} not found")
        else:
            # Reload all sources
            for sid in list(self._sources.keys()):
                try:
                    await self._load_source(sid)
                except Exception as e:
                    self.logger.error(f"Failed to reload source {sid}: {str(e)}")

    async def remove_source(self, source_id: str) -> None:
        """
        Remove a configuration source.
        
        Args:
            source_id: Source identifier to remove
        """
        if source_id not in self._sources:
            raise ConfigError(f"Configuration source {source_id} not found")
        
        # Stop any reload tasks
        if source_id in self._reload_tasks:
            self._reload_tasks[source_id].cancel()
            del self._reload_tasks[source_id]
        
        # Remove from sources and configurations
        del self._sources[source_id]
        self._configurations.pop(source_id, None)
        
        # Re-merge configurations
        await self._merge_configurations()
        
        # Emit source removed event
        if self.event_bus:
            await self.event_bus.emit(ConfigurationSourceRemoved(
                source_id=source_id,
                environment=self.environment
            ))
        
        self.logger.info(f"Removed configuration source: {source_id}")

    def list_sources(self) -> List[Dict[str, Any]]:
        """List all configuration sources."""
        return [
            {
                'source_id': source_id,
                'source_type': info.source_type.value,
                'location': info.location,
                'priority': info.priority.value,
                'last_loaded': info.last_loaded.isoformat() if info.last_loaded else None,
                'load_count': info.load_count,
                'error_count': info.error_count,
                'checksum': info.checksum
            }
            for source_id, info in self._sources.items()
        ]

    def get_snapshot(self, snapshot_id: Optional[str] = None) -> Optional[ConfigurationSnapshot]:
        """
        Get a configuration snapshot.
        
        Args:
            snapshot_id: Specific snapshot ID, or None for latest
            
        Returns:
            Configuration snapshot or None if not found
        """
        if not self._snapshots:
            return None
        
        if snapshot_id is None:
            return self._snapshots[-1]  # Latest snapshot
        
        for snapshot in self._snapshots:
            if snapshot.snapshot_id == snapshot_id:
                return snapshot
        
        return None

    def list_snapshots(self) -> List[Dict[str, Any]]:
        """List all configuration snapshots."""
        return [
            {
                'snapshot_id': snapshot.snapshot_id,
                'timestamp': snapshot.timestamp.isoformat(),
                'environment': snapshot.environment,
                'sources': snapshot.sources,
                'checksum': snapshot.checksum
            }
            for snapshot in self._snapshots
        ]

    async def rollback_to_snapshot(self, snapshot_id: str) -> None:
        """
        Rollback configuration to a specific snapshot.
        
        Args:
            snapshot_id: Snapshot ID to rollback to
        """
        snapshot = self.get_snapshot(snapshot_id)
        if not snapshot:
            raise ConfigError(f"Snapshot {snapshot_id} not found")
        
        # Replace merged configuration
        old_config = self._merged_config.copy()
        self._merged_config = copy.deepcopy(snapshot.configuration)
        
        # Invalidate cache
        self._cache.invalidate()
        
        # Emit configuration changed event
        if self.event_bus:
            await self.event_bus.emit(ConfigurationChanged(
                environment=self.environment,
                changed_keys=self._get_changed_keys(old_config, self._merged_config),
                total_keys=len(self._get_all_keys())
            ))
        
        self.logger.info(f"Rolled back configuration to snapshot: {snapshot_id}")

    def get_metrics(self) -> Dict[str, Any]:
        """Get configuration loader metrics."""
        total_loads = sum(info.load_count for info in self._sources.values())
        total_errors = sum(info.error_count for info in self._sources.values())
        
        avg_load_times = {}
        for source_id, times in self._load_times.items():
            if times:
                avg_load_times[source_id] = sum(times) / len(times)
        
        return {
            'environment': self.environment,
            'initialized': self._initialized,
            'sources_count': len(self._sources),
            'total_loads': total_loads,
            'total_errors': total_errors,
            'error_rate': total_errors / max(total_loads, 1),
            'cache_size': len(self._cache._cache),
            'snapshots_count': len(self._snapshots),
            'average_load_times': avg_load_times,
            'configuration_keys': len(self._get_all_keys())
        }

    async def _cache_cleanup_loop(self) -> None:
        """Background task to clean up expired cache entries."""
        while True:
            try:
                await asyncio.sleep(60)  # Run every minute
                
                expired_count = self._cache.cleanup_expired()
                if expired_count > 0:
                    self.logger.debug(f"Cleaned up {expired_count} expired cache entries")
                
            except Exception as e:
                self.logger.error(f"Error during cache cleanup: {str(e)}")
