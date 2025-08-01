"""
Advanced Configuration Validation System
Author: Drmusab
Last Modified: 2025-01-13 11:21:00 UTC

This module provides comprehensive configuration validation for the AI assistant,
including real-time validation, security checks, performance impact analysis,
and seamless integration with all core system components.
"""

import inspect
import ipaddress
import json
import logging
import os
import re
import socket
import ssl
import subprocess
import threading
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Pattern, Set, Tuple, Type, Union
from urllib.parse import urlparse

import asyncio
import yaml

# Core imports
from src.core.config.loader import ConfigLoader
from src.core.dependency_injection import Container
from src.core.error_handling import ErrorHandler, handle_exceptions
from src.core.events.event_bus import EventBus
from src.core.events.event_types import (
    ComponentConfigurationUpdated,
    ConfigurationBackupCreated,
    ConfigurationChanged,
    ConfigurationReloaded,
    ConfigurationValidated,
    ConfigurationValidationFailed,
    EnvironmentConfigurationChanged,
    PerformanceImpactDetected,
    PluginConfigurationValidated,
    SecurityViolationDetected,
    SkillConfigurationValidated,
    WorkflowConfigurationValidated,
)
from src.core.health_check import HealthCheck
from src.core.security.authentication import AuthenticationManager
from src.core.security.authorization import AuthorizationManager
from src.core.security.encryption import EncryptionManager
from src.observability.logging.config import get_logger

# Observability
from src.observability.monitoring.metrics import MetricsCollector
from src.observability.monitoring.tracing import TraceManager


class ValidationLevel(Enum):
    """Configuration validation levels."""

    STRICT = "strict"  # Strict validation with no tolerance
    STANDARD = "standard"  # Standard validation with warnings
    PERMISSIVE = "permissive"  # Permissive validation for development
    CUSTOM = "custom"  # Custom validation rules


class ValidationType(Enum):
    """Types of validation checks."""

    SYNTAX = "syntax"  # Syntax and format validation
    SEMANTIC = "semantic"  # Semantic correctness
    SECURITY = "security"  # Security compliance
    PERFORMANCE = "performance"  # Performance impact
    COMPATIBILITY = "compatibility"  # Component compatibility
    RESOURCE = "resource"  # Resource availability
    NETWORK = "network"  # Network connectivity
    DEPENDENCY = "dependency"  # Dependency validation
    BUSINESS_LOGIC = "business_logic"  # Business rule validation
    INTEGRATION = "integration"  # Integration requirements


class ValidationSeverity(Enum):
    """Validation issue severity levels."""

    CRITICAL = "critical"  # Must be fixed, system cannot start
    ERROR = "error"  # Must be fixed for proper operation
    WARNING = "warning"  # Should be fixed but not critical
    INFO = "info"  # Informational, no action required
    DEBUG = "debug"  # Debug information


class ConfigSection(Enum):
    """Configuration sections to validate."""

    CORE = "core"
    DATABASE = "database"
    CACHE = "cache"
    LLM = "llm"
    SPEECH = "speech"
    VISION = "vision"
    MEMORY = "memory"
    LEARNING = "learning"
    SKILLS = "skills"
    PLUGINS = "plugins"
    WORKFLOWS = "workflows"
    API = "api"
    SECURITY = "security"
    MONITORING = "monitoring"
    NETWORKING = "networking"
    STORAGE = "storage"
    INTEGRATIONS = "integrations"
    ENVIRONMENT = "environment"


@dataclass
class ValidationRule:
    """Defines a configuration validation rule."""

    rule_id: str
    name: str
    description: str
    section: ConfigSection
    validation_type: ValidationType
    severity: ValidationSeverity

    # Rule configuration
    enabled: bool = True
    apply_to_environments: Set[str] = field(default_factory=lambda: {"all"})
    conditions: Dict[str, Any] = field(default_factory=dict)

    # Validation function
    validator_function: Optional[Callable] = None
    regex_pattern: Optional[str] = None
    allowed_values: Optional[List[Any]] = None
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    required_keys: Optional[List[str]] = None
    forbidden_keys: Optional[List[str]] = None

    # Dependencies
    depends_on: List[str] = field(default_factory=list)
    conflicts_with: List[str] = field(default_factory=list)

    # Metadata
    tags: Set[str] = field(default_factory=set)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: Optional[datetime] = None


@dataclass
class ValidationResult:
    """Result of a validation check."""

    rule_id: str
    section: str
    key_path: str
    severity: ValidationSeverity
    message: str

    # Context information
    actual_value: Any = None
    expected_value: Any = None
    suggestions: List[str] = field(default_factory=list)

    # Location information
    file_path: Optional[str] = None
    line_number: Optional[int] = None

    # Metadata
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    validation_type: Optional[ValidationType] = None
    tags: Set[str] = field(default_factory=set)


@dataclass
class ValidationReport:
    """Comprehensive validation report."""

    validation_id: str
    config_version: str
    environment: str

    # Results
    results: List[ValidationResult] = field(default_factory=list)
    total_issues: int = 0
    critical_issues: int = 0
    error_issues: int = 0
    warning_issues: int = 0
    info_issues: int = 0

    # Summary
    is_valid: bool = True
    can_start: bool = True
    recommendations: List[str] = field(default_factory=list)

    # Performance metrics
    validation_time: float = 0.0
    rules_applied: int = 0
    sections_validated: Set[str] = field(default_factory=set)

    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    validator_version: str = "1.0.0"


class ConfigValidationError(Exception):
    """Custom exception for configuration validation errors."""

    def __init__(
        self,
        message: str,
        section: Optional[str] = None,
        key_path: Optional[str] = None,
        validation_type: Optional[ValidationType] = None,
    ):
        super().__init__(message)
        self.section = section
        self.key_path = key_path
        self.validation_type = validation_type
        self.timestamp = datetime.now(timezone.utc)


class BaseValidator(ABC):
    """Abstract base class for configuration validators."""

    def __init__(self, logger):
        self.logger = logger

    @abstractmethod
    def validate(self, config: Dict[str, Any], context: Dict[str, Any]) -> List[ValidationResult]:
        """Validate configuration section."""
        pass

    @abstractmethod
    def get_supported_sections(self) -> List[ConfigSection]:
        """Get list of supported configuration sections."""
        pass

    def _create_result(
        self,
        rule_id: str,
        section: str,
        key_path: str,
        severity: ValidationSeverity,
        message: str,
        **kwargs,
    ) -> ValidationResult:
        """Helper method to create validation result."""
        return ValidationResult(
            rule_id=rule_id,
            section=section,
            key_path=key_path,
            severity=severity,
            message=message,
            **kwargs,
        )


class CoreValidator(BaseValidator):
    """Validator for core system configuration."""

    def get_supported_sections(self) -> List[ConfigSection]:
        return [ConfigSection.CORE, ConfigSection.ENVIRONMENT]

    def validate(self, config: Dict[str, Any], context: Dict[str, Any]) -> List[ValidationResult]:
        results = []

        # Validate core configuration
        if "core" in config:
            core_config = config["core"]
            results.extend(self._validate_core_section(core_config))

        # Validate environment configuration
        if "environment" in config:
            env_config = config["environment"]
            results.extend(self._validate_environment_section(env_config))

        return results

    def _validate_core_section(self, core_config: Dict[str, Any]) -> List[ValidationResult]:
        results = []

        # Check required core settings
        required_keys = ["debug", "log_level", "max_workers"]
        for key in required_keys:
            if key not in core_config:
                results.append(
                    self._create_result(
                        "core_001",
                        "core",
                        f"core.{key}",
                        ValidationSeverity.ERROR,
                        f"Required core setting '{key}' is missing",
                    )
                )

        # Validate log level
        if "log_level" in core_config:
            valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
            if core_config["log_level"] not in valid_levels:
                results.append(
                    self._create_result(
                        "core_002",
                        "core",
                        "core.log_level",
                        ValidationSeverity.ERROR,
                        f"Invalid log level '{core_config['log_level']}'. Must be one of: {valid_levels}",
                    )
                )

        # Validate max_workers
        if "max_workers" in core_config:
            max_workers = core_config["max_workers"]
            if not isinstance(max_workers, int) or max_workers < 1:
                results.append(
                    self._create_result(
                        "core_003",
                        "core",
                        "core.max_workers",
                        ValidationSeverity.ERROR,
                        "max_workers must be a positive integer",
                    )
                )
            elif max_workers > 100:
                results.append(
                    self._create_result(
                        "core_004",
                        "core",
                        "core.max_workers",
                        ValidationSeverity.WARNING,
                        f"max_workers value {max_workers} is very high and may impact performance",
                    )
                )

        return results

    def _validate_environment_section(self, env_config: Dict[str, Any]) -> List[ValidationResult]:
        results = []

        # Validate environment name
        if "name" in env_config:
            valid_envs = ["development", "staging", "production", "testing"]
            if env_config["name"] not in valid_envs:
                results.append(
                    self._create_result(
                        "env_001",
                        "environment",
                        "environment.name",
                        ValidationSeverity.WARNING,
                        f"Unusual environment name '{env_config['name']}'. Consider using: {valid_envs}",
                    )
                )

        return results


class DatabaseValidator(BaseValidator):
    """Validator for database configuration."""

    def get_supported_sections(self) -> List[ConfigSection]:
        return [ConfigSection.DATABASE]

    def validate(self, config: Dict[str, Any], context: Dict[str, Any]) -> List[ValidationResult]:
        results = []

        if "database" not in config:
            results.append(
                self._create_result(
                    "db_001",
                    "database",
                    "database",
                    ValidationSeverity.ERROR,
                    "Database configuration is missing",
                )
            )
            return results

        db_config = config["database"]

        # Validate connection string
        if "url" in db_config:
            results.extend(self._validate_db_url(db_config["url"]))

        # Validate connection pool settings
        if "pool" in db_config:
            results.extend(self._validate_pool_settings(db_config["pool"]))

        # Validate timeout settings
        if "timeout" in db_config:
            timeout = db_config["timeout"]
            if not isinstance(timeout, (int, float)) or timeout <= 0:
                results.append(
                    self._create_result(
                        "db_002",
                        "database",
                        "database.timeout",
                        ValidationSeverity.ERROR,
                        "Database timeout must be a positive number",
                    )
                )

        return results

    def _validate_db_url(self, url: str) -> List[ValidationResult]:
        results = []

        try:
            parsed = urlparse(url)

            # Check scheme
            valid_schemes = ["postgresql", "mysql", "sqlite", "mongodb"]
            if parsed.scheme not in valid_schemes:
                results.append(
                    self._create_result(
                        "db_003",
                        "database",
                        "database.url",
                        ValidationSeverity.ERROR,
                        f"Unsupported database scheme '{parsed.scheme}'. Supported: {valid_schemes}",
                    )
                )

            # Check for credentials in production
            environment = os.getenv("ENVIRONMENT", "development")
            if environment == "production" and (parsed.username or parsed.password):
                results.append(
                    self._create_result(
                        "db_004",
                        "database",
                        "database.url",
                        ValidationSeverity.WARNING,
                        "Database credentials should not be embedded in URL for production. Use environment variables.",
                    )
                )

        except Exception as e:
            results.append(
                self._create_result(
                    "db_005",
                    "database",
                    "database.url",
                    ValidationSeverity.ERROR,
                    f"Invalid database URL format: {str(e)}",
                )
            )

        return results

    def _validate_pool_settings(self, pool_config: Dict[str, Any]) -> List[ValidationResult]:
        results = []

        if "min_size" in pool_config and "max_size" in pool_config:
            min_size = pool_config["min_size"]
            max_size = pool_config["max_size"]

            if min_size > max_size:
                results.append(
                    self._create_result(
                        "db_006",
                        "database",
                        "database.pool",
                        ValidationSeverity.ERROR,
                        "Pool min_size cannot be greater than max_size",
                    )
                )

        return results


class LLMValidator(BaseValidator):
    """Validator for LLM configuration."""

    def get_supported_sections(self) -> List[ConfigSection]:
        return [ConfigSection.LLM]

    def validate(self, config: Dict[str, Any], context: Dict[str, Any]) -> List[ValidationResult]:
        results = []

        if "llm" not in config:
            results.append(
                self._create_result(
                    "llm_001",
                    "llm",
                    "llm",
                    ValidationSeverity.ERROR,
                    "LLM configuration is missing",
                )
            )
            return results

        llm_config = config["llm"]

        # Validate providers
        if "providers" in llm_config:
            results.extend(self._validate_providers(llm_config["providers"]))

        # Validate default model
        if "default_model" in llm_config:
            results.extend(self._validate_default_model(llm_config["default_model"], llm_config))

        # Validate rate limits
        if "rate_limits" in llm_config:
            results.extend(self._validate_rate_limits(llm_config["rate_limits"]))

        return results

    def _validate_providers(self, providers: Dict[str, Any]) -> List[ValidationResult]:
        results = []

        supported_providers = ["openai", "anthropic", "ollama", "deepseek", "huggingface"]

        for provider_name, provider_config in providers.items():
            if provider_name not in supported_providers:
                results.append(
                    self._create_result(
                        "llm_002",
                        "llm",
                        f"llm.providers.{provider_name}",
                        ValidationSeverity.WARNING,
                        f"Unknown provider '{provider_name}'. Supported: {supported_providers}",
                    )
                )

            # Validate API key for external providers
            if provider_name in ["openai", "anthropic"] and "api_key" not in provider_config:
                results.append(
                    self._create_result(
                        "llm_003",
                        "llm",
                        f"llm.providers.{provider_name}.api_key",
                        ValidationSeverity.ERROR,
                        f"API key is required for {provider_name} provider",
                    )
                )

            # Validate base URL for local providers
            if provider_name == "ollama" and "base_url" in provider_config:
                results.extend(
                    self._validate_url(
                        provider_config["base_url"], f"llm.providers.{provider_name}.base_url"
                    )
                )

        return results

    def _validate_default_model(
        self, default_model: str, llm_config: Dict[str, Any]
    ) -> List[ValidationResult]:
        results = []

        # Check if default model exists in any provider
        providers = llm_config.get("providers", {})
        model_found = False

        for provider_config in providers.values():
            if "models" in provider_config and default_model in provider_config["models"]:
                model_found = True
                break

        if not model_found:
            results.append(
                self._create_result(
                    "llm_004",
                    "llm",
                    "llm.default_model",
                    ValidationSeverity.ERROR,
                    f"Default model '{default_model}' not found in any provider configuration",
                )
            )

        return results

    def _validate_rate_limits(self, rate_limits: Dict[str, Any]) -> List[ValidationResult]:
        results = []

        for provider, limits in rate_limits.items():
            if "requests_per_minute" in limits:
                rpm = limits["requests_per_minute"]
                if not isinstance(rpm, int) or rpm <= 0:
                    results.append(
                        self._create_result(
                            "llm_005",
                            "llm",
                            f"llm.rate_limits.{provider}.requests_per_minute",
                            ValidationSeverity.ERROR,
                            "requests_per_minute must be a positive integer",
                        )
                    )

        return results

    def _validate_url(self, url: str, key_path: str) -> List[ValidationResult]:
        results = []

        try:
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                results.append(
                    self._create_result(
                        "url_001", "llm", key_path, ValidationSeverity.ERROR, "Invalid URL format"
                    )
                )
        except Exception as e:
            results.append(
                self._create_result(
                    "url_002",
                    "llm",
                    key_path,
                    ValidationSeverity.ERROR,
                    f"URL validation failed: {str(e)}",
                )
            )

        return results


class SecurityValidator(BaseValidator):
    """Validator for security configuration."""

    def get_supported_sections(self) -> List[ConfigSection]:
        return [ConfigSection.SECURITY]

    def validate(self, config: Dict[str, Any], context: Dict[str, Any]) -> List[ValidationResult]:
        results = []

        if "security" not in config:
            results.append(
                self._create_result(
                    "sec_001",
                    "security",
                    "security",
                    ValidationSeverity.WARNING,
                    "Security configuration is missing",
                )
            )
            return results

        security_config = config["security"]

        # Validate authentication settings
        if "authentication" in security_config:
            results.extend(self._validate_authentication(security_config["authentication"]))

        # Validate encryption settings
        if "encryption" in security_config:
            results.extend(self._validate_encryption(security_config["encryption"]))

        # Validate TLS/SSL settings
        if "tls" in security_config:
            results.extend(self._validate_tls(security_config["tls"]))

        return results

    def _validate_authentication(self, auth_config: Dict[str, Any]) -> List[ValidationResult]:
        results = []

        # Check for strong secret keys
        if "secret_key" in auth_config:
            secret_key = auth_config["secret_key"]
            if len(secret_key) < 32:
                results.append(
                    self._create_result(
                        "sec_002",
                        "security",
                        "security.authentication.secret_key",
                        ValidationSeverity.CRITICAL,
                        "Secret key must be at least 32 characters long",
                    )
                )

        # Validate session timeout
        if "session_timeout" in auth_config:
            timeout = auth_config["session_timeout"]
            if not isinstance(timeout, int) or timeout <= 0:
                results.append(
                    self._create_result(
                        "sec_003",
                        "security",
                        "security.authentication.session_timeout",
                        ValidationSeverity.ERROR,
                        "Session timeout must be a positive integer",
                    )
                )

        return results

    def _validate_encryption(self, encryption_config: Dict[str, Any]) -> List[ValidationResult]:
        results = []

        # Validate encryption algorithm
        if "algorithm" in encryption_config:
            algorithm = encryption_config["algorithm"]
            secure_algorithms = ["AES-256-GCM", "ChaCha20-Poly1305", "AES-256-CBC"]
            if algorithm not in secure_algorithms:
                results.append(
                    self._create_result(
                        "sec_004",
                        "security",
                        "security.encryption.algorithm",
                        ValidationSeverity.ERROR,
                        f"Weak encryption algorithm '{algorithm}'. Recommended: {secure_algorithms}",
                    )
                )

        return results

    def _validate_tls(self, tls_config: Dict[str, Any]) -> List[ValidationResult]:
        results = []

        # Validate minimum TLS version
        if "min_version" in tls_config:
            min_version = tls_config["min_version"]
            if min_version < "1.2":
                results.append(
                    self._create_result(
                        "sec_005",
                        "security",
                        "security.tls.min_version",
                        ValidationSeverity.CRITICAL,
                        "TLS version must be 1.2 or higher for security",
                    )
                )

        return results


class PerformanceValidator(BaseValidator):
    """Validator for performance-related configuration."""

    def get_supported_sections(self) -> List[ConfigSection]:
        return [ConfigSection.CORE, ConfigSection.MEMORY, ConfigSection.CACHE]

    def validate(self, config: Dict[str, Any], context: Dict[str, Any]) -> List[ValidationResult]:
        results = []

        # Validate memory settings
        if "memory" in config:
            results.extend(self._validate_memory_settings(config["memory"]))

        # Validate cache settings
        if "cache" in config:
            results.extend(self._validate_cache_settings(config["cache"]))

        # Cross-validate settings for performance impact
        results.extend(self._validate_performance_impact(config))

        return results

    def _validate_memory_settings(self, memory_config: Dict[str, Any]) -> List[ValidationResult]:
        results = []

        # Check memory limits
        if "max_memory_mb" in memory_config:
            max_memory = memory_config["max_memory_mb"]
            if max_memory > 8192:  # 8GB
                results.append(
                    self._create_result(
                        "perf_001",
                        "memory",
                        "memory.max_memory_mb",
                        ValidationSeverity.WARNING,
                        f"High memory limit {max_memory}MB may impact system performance",
                    )
                )

        return results

    def _validate_cache_settings(self, cache_config: Dict[str, Any]) -> List[ValidationResult]:
        results = []

        # Validate cache size
        if "max_size_mb" in cache_config:
            max_size = cache_config["max_size_mb"]
            if max_size > 2048:  # 2GB
                results.append(
                    self._create_result(
                        "perf_002",
                        "cache",
                        "cache.max_size_mb",
                        ValidationSeverity.WARNING,
                        f"Large cache size {max_size}MB may consume significant memory",
                    )
                )

        return results

    def _validate_performance_impact(self, config: Dict[str, Any]) -> List[ValidationResult]:
        results = []

        # Check if too many concurrent processes are configured
        total_workers = 0

        if "core" in config and "max_workers" in config["core"]:
            total_workers += config["core"]["max_workers"]

        if "api" in config and "workers" in config["api"]:
            total_workers += config["api"]["workers"]

        if total_workers > 50:
            results.append(
                self._create_result(
                    "perf_003",
                    "performance",
                    "global.total_workers",
                    ValidationSeverity.WARNING,
                    f"Total worker count {total_workers} is very high and may cause resource contention",
                )
            )

        return results


class NetworkValidator(BaseValidator):
    """Validator for network-related configuration."""

    def get_supported_sections(self) -> List[ConfigSection]:
        return [ConfigSection.NETWORKING, ConfigSection.API]

    def validate(self, config: Dict[str, Any], context: Dict[str, Any]) -> List[ValidationResult]:
        results = []

        # Validate API configuration
        if "api" in config:
            results.extend(self._validate_api_config(config["api"]))

        # Validate networking configuration
        if "networking" in config:
            results.extend(self._validate_networking_config(config["networking"]))

        return results

    def _validate_api_config(self, api_config: Dict[str, Any]) -> List[ValidationResult]:
        results = []

        # Validate host binding
        if "host" in api_config:
            host = api_config["host"]
            if host == "0.0.0.0":
                environment = os.getenv("ENVIRONMENT", "development")
                if environment == "production":
                    results.append(
                        self._create_result(
                            "net_001",
                            "api",
                            "api.host",
                            ValidationSeverity.WARNING,
                            "Binding to 0.0.0.0 in production may be a security risk",
                        )
                    )

        # Validate port
        if "port" in api_config:
            port = api_config["port"]
            if not isinstance(port, int) or port < 1 or port > 65535:
                results.append(
                    self._create_result(
                        "net_002",
                        "api",
                        "api.port",
                        ValidationSeverity.ERROR,
                        "Port must be a valid integer between 1 and 65535",
                    )
                )
            elif port < 1024:
                results.append(
                    self._create_result(
                        "net_003",
                        "api",
                        "api.port",
                        ValidationSeverity.WARNING,
                        "Using privileged port (<1024) may require elevated permissions",
                    )
                )

        return results

    def _validate_networking_config(
        self, networking_config: Dict[str, Any]
    ) -> List[ValidationResult]:
        results = []

        # Validate timeout settings
        if "timeout" in networking_config:
            timeout = networking_config["timeout"]
            if not isinstance(timeout, (int, float)) or timeout <= 0:
                results.append(
                    self._create_result(
                        "net_004",
                        "networking",
                        "networking.timeout",
                        ValidationSeverity.ERROR,
                        "Network timeout must be a positive number",
                    )
                )

        return results


class PluginValidator(BaseValidator):
    """Validator for plugin configuration."""

    def get_supported_sections(self) -> List[ConfigSection]:
        return [ConfigSection.PLUGINS]

    def validate(self, config: Dict[str, Any], context: Dict[str, Any]) -> List[ValidationResult]:
        results = []

        if "plugins" not in config:
            return results

        plugins_config = config["plugins"]

        # Validate plugin directories
        if "directories" in plugins_config:
            results.extend(self._validate_plugin_directories(plugins_config["directories"]))

        # Validate plugin security settings
        if "security_validation" in plugins_config:
            security_enabled = plugins_config["security_validation"]
            if not security_enabled:
                results.append(
                    self._create_result(
                        "plugin_001",
                        "plugins",
                        "plugins.security_validation",
                        ValidationSeverity.WARNING,
                        "Plugin security validation is disabled - this may be a security risk",
                    )
                )

        return results

    def _validate_plugin_directories(self, directories: List[str]) -> List[ValidationResult]:
        results = []

        for directory in directories:
            path = Path(directory)
            if not path.exists():
                results.append(
                    self._create_result(
                        "plugin_002",
                        "plugins",
                        f"plugins.directories.{directory}",
                        ValidationSeverity.WARNING,
                        f"Plugin directory '{directory}' does not exist",
                    )
                )
            elif not path.is_dir():
                results.append(
                    self._create_result(
                        "plugin_003",
                        "plugins",
                        f"plugins.directories.{directory}",
                        ValidationSeverity.ERROR,
                        f"Plugin directory '{directory}' is not a directory",
                    )
                )

        return results


class EnhancedConfigValidator:
    """
    Advanced Configuration Validation System for the AI Assistant.

    This validator provides comprehensive configuration validation including:
    - Multi-level validation (syntax, semantic, security, performance)
    - Real-time configuration monitoring
    - Environment-specific validation rules
    - Security compliance checking
    - Performance impact analysis
    - Integration with all core system components
    - Plugin and skill configuration validation
    - Event-driven validation notifications
    - Detailed reporting and recommendations
    """

    def __init__(self, container: Container):
        """
        Initialize the enhanced configuration validator.

        Args:
            container: Dependency injection container
        """
        self.container = container
        self.logger = get_logger(__name__)

        # Core services
        self.config_loader = container.get(ConfigLoader)
        self.event_bus = container.get(EventBus)
        self.error_handler = container.get(ErrorHandler)
        self.health_check = container.get(HealthCheck)

        # Security components
        try:
            self.auth_manager = container.get(AuthenticationManager)
            self.authz_manager = container.get(AuthorizationManager)
            self.encryption_manager = container.get(EncryptionManager)
        except Exception:
            self.auth_manager = None
            self.authz_manager = None
            self.encryption_manager = None

        # Observability
        self.metrics = container.get(MetricsCollector)
        self.tracer = container.get(TraceManager)

        # Validation components
        self._setup_validators()
        self._setup_rules()
        self._setup_monitoring()

        # State management
        self.validation_level = ValidationLevel.STANDARD
        self.current_environment = os.getenv("ENVIRONMENT", "development")
        self.validation_cache: Dict[str, ValidationReport] = {}
        self.rule_cache: Dict[str, ValidationRule] = {}

        # Real-time monitoring
        self.file_watchers: Dict[str, Any] = {}
        self.last_validation: Optional[datetime] = None
        self.validation_lock = asyncio.Lock()

        # Performance tracking
        self.validation_history: List[ValidationReport] = []
        self.validation_stats: Dict[str, Any] = {
            "total_validations": 0,
            "successful_validations": 0,
            "failed_validations": 0,
            "average_validation_time": 0.0,
        }

        # Configuration
        self.auto_fix_enabled = False
        self.backup_on_validation = True
        self.notify_on_changes = True

        # Register health check
        self.health_check.register_component("config_validator", self._health_check_callback)

        self.logger.info("EnhancedConfigValidator initialized successfully")

    def _setup_validators(self) -> None:
        """Setup section-specific validators."""
        self.validators: Dict[ConfigSection, BaseValidator] = {
            ConfigSection.CORE: CoreValidator(self.logger),
            ConfigSection.DATABASE: DatabaseValidator(self.logger),
            ConfigSection.LLM: LLMValidator(self.logger),
            ConfigSection.SECURITY: SecurityValidator(self.logger),
            ConfigSection.NETWORKING: NetworkValidator(self.logger),
            ConfigSection.PLUGINS: PluginValidator(self.logger),
        }

        # Add performance validator for multiple sections
        perf_validator = PerformanceValidator(self.logger)
        for section in perf_validator.get_supported_sections():
            if section not in self.validators:
                self.validators[section] = perf_validator

    def _setup_rules(self) -> None:
        """Setup validation rules."""
        self.rules: List[ValidationRule] = []

        # Load rules from configuration
        self._load_builtin_rules()
        self._load_custom_rules()

        # Build rule cache
        for rule in self.rules:
            self.rule_cache[rule.rule_id] = rule

    def _load_builtin_rules(self) -> None:
        """Load built-in validation rules."""
        # Core system rules
        self.rules.extend(
            [
                ValidationRule(
                    rule_id="global_001",
                    name="Required Configuration Sections",
                    description="Ensure all required configuration sections are present",
                    section=ConfigSection.CORE,
                    validation_type=ValidationType.SYNTAX,
                    severity=ValidationSeverity.CRITICAL,
                    required_keys=["core", "api", "logging"],
                ),
                ValidationRule(
                    rule_id="security_001",
                    name="Production Security Requirements",
                    description="Enforce security requirements in production environment",
                    section=ConfigSection.SECURITY,
                    validation_type=ValidationType.SECURITY,
                    severity=ValidationSeverity.CRITICAL,
                    apply_to_environments={"production"},
                    conditions={"tls_enabled": True, "authentication_required": True},
                ),
                ValidationRule(
                    rule_id="performance_001",
                    name="Resource Limits Validation",
                    description="Validate resource limits are within acceptable ranges",
                    section=ConfigSection.CORE,
                    validation_type=ValidationType.PERFORMANCE,
                    severity=ValidationSeverity.WARNING,
                    conditions={"max_memory_mb": {"max": 8192}, "max_workers": {"max": 50}},
                ),
            ]
        )

    def _load_custom_rules(self) -> None:
        """Load custom validation rules from configuration."""
        try:
            rules_config = self.config_loader.get("validation.custom_rules", [])
            for rule_config in rules_config:
                rule = ValidationRule(**rule_config)
                self.rules.append(rule)
        except Exception as e:
            self.logger.warning(f"Failed to load custom validation rules: {str(e)}")

    def _setup_monitoring(self) -> None:
        """Setup monitoring and metrics."""
        try:
            # Register validation metrics
            self.metrics.register_counter("config_validations_total")
            self.metrics.register_counter("config_validation_errors_total")
            self.metrics.register_histogram("config_validation_duration_seconds")
            self.metrics.register_gauge("config_validation_issues")
            self.metrics.register_counter("config_changes_detected")

        except Exception as e:
            self.logger.warning(f"Failed to setup monitoring: {str(e)}")

    async def initialize(self) -> None:
        """Initialize the configuration validator."""
        try:
            # Perform initial validation
            await self.validate_all_configurations()

            # Start file monitoring if enabled
            if self.notify_on_changes:
                await self._start_file_monitoring()

            # Start background tasks
            asyncio.create_task(self._validation_maintenance_loop())
            asyncio.create_task(self._metrics_update_loop())

            self.logger.info("ConfigValidator initialization completed")

        except Exception as e:
            self.logger.error(f"Failed to initialize ConfigValidator: {str(e)}")
            raise ConfigValidationError(f"Initialization failed: {str(e)}")

    @handle_exceptions
    async def validate_all_configurations(
        self, validation_level: Optional[ValidationLevel] = None
    ) -> ValidationReport:
        """
        Validate all system configurations.

        Args:
            validation_level: Override default validation level

        Returns:
            Comprehensive validation report
        """
        async with self.validation_lock:
            start_time = time.time()
            validation_id = f"full_validation_{int(start_time)}"

            # Use provided level or default
            level = validation_level or self.validation_level

            try:
                with self.tracer.trace("config_validation") as span:
                    span.set_attributes(
                        {
                            "validation_id": validation_id,
                            "validation_level": level.value,
                            "environment": self.current_environment,
                        }
                    )

                    # Get current configuration
                    config = self._get_full_configuration()

                    # Create validation context
                    context = {
                        "environment": self.current_environment,
                        "validation_level": level,
                        "timestamp": datetime.now(timezone.utc),
                    }

                    # Perform validation
                    report = ValidationReport(
                        validation_id=validation_id,
                        config_version=self._get_config_version(config),
                        environment=self.current_environment,
                    )

                    # Validate each section
                    all_results = []
                    for section, validator in self.validators.items():
                        try:
                            section_results = validator.validate(config, context)
                            all_results.extend(section_results)
                            report.sections_validated.add(section.value)

                        except Exception as e:
                            self.logger.error(
                                f"Validation failed for section {section.value}: {str(e)}"
                            )
                            all_results.append(
                                ValidationResult(
                                    rule_id="validator_error",
                                    section=section.value,
                                    key_path=f"{section.value}.*",
                                    severity=ValidationSeverity.ERROR,
                                    message=f"Validator error: {str(e)}",
                                )
                            )

                    # Apply custom rules
                    custom_results = await self._apply_custom_rules(config, context)
                    all_results.extend(custom_results)

                    # Analyze results
                    report.results = all_results
                    report = self._analyze_validation_results(report)

                    # Update performance metrics
                    validation_time = time.time() - start_time
                    report.validation_time = validation_time
                    report.rules_applied = len(self.rules)

                    # Cache report
                    self.validation_cache[validation_id] = report
                    self.validation_history.append(report)

                    # Update statistics
                    self._update_validation_stats(report)

                    # Emit validation event
                    if report.is_valid:
                        await self.event_bus.emit(
                            ConfigurationValidated(
                                validation_id=validation_id,
                                environment=self.current_environment,
                                validation_time=validation_time,
                                issues_found=report.total_issues,
                            )
                        )
                    else:
                        await self.event_bus.emit(
                            ConfigurationValidationFailed(
                                validation_id=validation_id,
                                environment=self.current_environment,
                                critical_issues=report.critical_issues,
                                error_issues=report.error_issues,
                            )
                        )

                    # Update metrics
                    self.metrics.increment("config_validations_total")
                    self.metrics.record("config_validation_duration_seconds", validation_time)
                    self.metrics.set("config_validation_issues", report.total_issues)

                    if not report.is_valid:
                        self.metrics.increment("config_validation_errors_total")

                    self.last_validation = datetime.now(timezone.utc)

                    self.logger.info(
                        f"Configuration validation completed: {validation_id} "
                        f"({report.total_issues} issues found in {validation_time:.2f}s)"
                    )

                    return report

            except Exception as e:
                validation_time = time.time() - start_time

                # Create error report
                error_report = ValidationReport(
                    validation_id=validation_id,
                    config_version="unknown",
                    environment=self.current_environment,
                    is_valid=False,
                    can_start=False,
                    validation_time=validation_time,
                )

                error_report.results.append(
                    ValidationResult(
                        rule_id="validation_system_error",
                        section="system",
                        key_path="validation",
                        severity=ValidationSeverity.CRITICAL,
                        message=f"Configuration validation system error: {str(e)}",
                    )
                )

                self.logger.error(f"Configuration validation failed: {str(e)}")
                return error_report

    async def _apply_custom_rules(
        self, config: Dict[str, Any], context: Dict[str, Any]
    ) -> List[ValidationResult]:
        """Apply custom validation rules."""
        results = []

        for rule in self.rules:
            # Check if rule applies to current environment
            if not self._rule_applies_to_environment(rule, context["environment"]):
                continue

            # Check if rule is enabled
            if not rule.enabled:
                continue

            try:
                # Apply rule
                rule_results = await self._apply_single_rule(rule, config, context)
                results.extend(rule_results)

            except Exception as e:
                self.logger.error(f"Failed to apply rule {rule.rule_id}: {str(e)}")
                results.append(
                    ValidationResult(
                        rule_id=rule.rule_id,
                        section=rule.section.value,
                        key_path="rule_application",
                        severity=ValidationSeverity.ERROR,
                        message=f"Rule application failed: {str(e)}",
                    )
                )

        return results

    def _rule_applies_to_environment(self, rule: ValidationRule, environment: str) -> bool:
        """Check if a rule applies to the current environment."""
        if "all" in rule.apply_to_environments:
            return True
        return environment in rule.apply_to_environments

    async def _apply_single_rule(
        self, rule: ValidationRule, config: Dict[str, Any], context: Dict[str, Any]
    ) -> List[ValidationResult]:
        """Apply a single validation rule."""
        results = []

        # Get section config
        section_name = rule.section.value
        section_config = config.get(section_name, {})

        # Apply rule based on type
        if rule.validator_function:
            # Custom validator function
            try:
                rule_result = await rule.validator_function(section_config, context)
                if rule_result:
                    results.append(rule_result)
            except Exception as e:
                results.append(
                    ValidationResult(
                        rule_id=rule.rule_id,
                        section=section_name,
                        key_path=section_name,
                        severity=rule.severity,
                        message=f"Custom validator failed: {str(e)}",
                    )
                )

        elif rule.required_keys:
            # Check required keys
            for key in rule.required_keys:
                if key not in section_config:
                    results.append(
                        ValidationResult(
                            rule_id=rule.rule_id,
                            section=section_name,
                            key_path=f"{section_name}.{key}",
                            severity=rule.severity,
                            message=f"Required key '{key}' is missing from {section_name} configuration",
                        )
                    )

        elif rule.forbidden_keys:
            # Check forbidden keys
            for key in rule.forbidden_keys:
                if key in section_config:
                    results.append(
                        ValidationResult(
                            rule_id=rule.rule_id,
                            section=section_name,
                            key_path=f"{section_name}.{key}",
                            severity=rule.severity,
                            message=f"Forbidden key '{key}' found in {section_name} configuration",
                        )
                    )

        return results

    def _get_full_configuration(self) -> Dict[str, Any]:
        """Get the complete current configuration."""
        try:
            return self.config_loader.get_all()
        except Exception as e:
            self.logger.error(f"Failed to get full configuration: {str(e)}")
            return {}

    def _get_config_version(self, config: Dict[str, Any]) -> str:
        """Get configuration version or generate one."""
        if "version" in config:
            return str(config["version"])

        # Generate version based on config hash
        config_str = json.dumps(config, sort_keys=True, default=str)
        import hashlib

        return hashlib.md5(config_str.encode()).hexdigest()[:8]

    def _analyze_validation_results(self, report: ValidationReport) -> ValidationReport:
        """Analyze validation results and update report."""
        # Count issues by severity
        for result in report.results:
            report.total_issues += 1

            if result.severity == ValidationSeverity.CRITICAL:
                report.critical_issues += 1
            elif result.severity == ValidationSeverity.ERROR:
                report.error_issues += 1
            elif result.severity == ValidationSeverity.WARNING:
                report.warning_issues += 1
            elif result.severity == ValidationSeverity.INFO:
                report.info_issues += 1

        # Determine if configuration is valid
        report.is_valid = report.critical_issues == 0 and report.error_issues == 0
        report.can_start = report.critical_issues == 0

        # Generate recommendations
        report.recommendations = self._generate_recommendations(report)

        return report

    def _generate_recommendations(self, report: ValidationReport) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []

        if report.critical_issues > 0:
            recommendations.append("Fix critical issues before starting the system")

        if report.error_issues > 0:
            recommendations.append("Address error-level issues for proper system operation")

        if report.warning_issues > 5:
            recommendations.append(
                "Consider addressing warning-level issues for optimal performance"
            )

        # Security-specific recommendations
        security_issues = [
            r for r in report.results if r.validation_type == ValidationType.SECURITY
        ]
        if security_issues:
            recommendations.append("Review and address security-related configuration issues")

        # Performance-specific recommendations
        performance_issues = [
            r for r in report.results if r.validation_type == ValidationType.PERFORMANCE
        ]
        if performance_issues:
            recommendations.append("Optimize configuration for better performance")

        return recommendations

    def _update_validation_stats(self, report: ValidationReport) -> None:
        """Update validation statistics."""
        self.validation_stats["total_validations"] += 1

        if report.is_valid:
            self.validation_stats["successful_validations"] += 1
        else:
            self.validation_stats["failed_validations"] += 1

        # Update average validation time
        total_time = (
            self.validation_stats["average_validation_time"]
            * (self.validation_stats["total_validations"] - 1)
            + report.validation_time
        )
        self.validation_stats["average_validation_time"] = (
            total_time / self.validation_stats["total_validations"]
        )

    @handle_exceptions
    async def validate_section(
        self, section: str, config_data: Optional[Dict[str, Any]] = None
    ) -> List[ValidationResult]:
        """
        Validate a specific configuration section.

        Args:
            section: Section name to validate
            config_data: Optional specific configuration data

        Returns:
            List of validation results
        """
        try:
            section_enum = ConfigSection(section.lower())
        except ValueError:
            raise ConfigValidationError(f"Unknown configuration section: {section}")

        if section_enum not in self.validators:
            raise ConfigValidationError(f"No validator available for section: {section}")

        # Get configuration data
        if config_data is None:
            config_data = self.config_loader.get(section, {})

        # Create context
        context = {
            "environment": self.current_environment,
            "validation_level": self.validation_level,
            "timestamp": datetime.now(timezone.utc),
        }

        # Validate section
        validator = self.validators[section_enum]
        results = validator.validate({section: config_data}, context)

        return results

    @handle_exceptions
    async def validate_plugin_config(
        self, plugin_id: str, plugin_config: Dict[str, Any]
    ) -> ValidationReport:
        """
        Validate plugin configuration.

        Args:
            plugin_id: Plugin identifier
            plugin_config: Plugin configuration

        Returns:
            Validation report for the plugin
        """
        validation_id = f"plugin_{plugin_id}_{int(time.time())}"

        report = ValidationReport(
            validation_id=validation_id,
            config_version="plugin",
            environment=self.current_environment,
        )

        # Basic plugin config validation
        required_keys = ["name", "version", "type"]
        for key in required_keys:
            if key not in plugin_config:
                report.results.append(
                    ValidationResult(
                        rule_id="plugin_config_001",
                        section="plugin",
                        key_path=f"plugin.{key}",
                        severity=ValidationSeverity.ERROR,
                        message=f"Required plugin configuration key '{key}' is missing",
                    )
                )

        # Validate plugin type
        if "type" in plugin_config:
            valid_types = ["skill", "processor", "integration", "ui_component"]
            if plugin_config["type"] not in valid_types:
                report.results.append(
                    ValidationResult(
                        rule_id="plugin_config_002",
                        section="plugin",
                        key_path="plugin.type",
                        severity=ValidationSeverity.ERROR,
                        message=f"Invalid plugin type '{plugin_config['type']}'. Valid types: {valid_types}",
                    )
                )

        # Analyze results
        report = self._analyze_validation_results(report)

        # Emit plugin validation event
        await self.event_bus.emit(
            PluginConfigurationValidated(
                plugin_id=plugin_id,
                validation_id=validation_id,
                is_valid=report.is_valid,
                issues_found=report.total_issues,
            )
        )

        return report

    @handle_exceptions
    async def validate_skill_config(
        self, skill_id: str, skill_config: Dict[str, Any]
    ) -> ValidationReport:
        """
        Validate skill configuration.

        Args:
            skill_id: Skill identifier
            skill_config: Skill configuration

        Returns:
            Validation report for the skill
        """
        validation_id = f"skill_{skill_id}_{int(time.time())}"

        report = ValidationReport(
            validation_id=validation_id,
            config_version="skill",
            environment=self.current_environment,
        )

        # Basic skill config validation
        required_keys = ["name", "version", "capabilities"]
        for key in required_keys:
            if key not in skill_config:
                report.results.append(
                    ValidationResult(
                        rule_id="skill_config_001",
                        section="skill",
                        key_path=f"skill.{key}",
                        severity=ValidationSeverity.ERROR,
                        message=f"Required skill configuration key '{key}' is missing",
                    )
                )

        # Validate capabilities
        if "capabilities" in skill_config:
            capabilities = skill_config["capabilities"]
            if not isinstance(capabilities, list) or not capabilities:
                report.results.append(
                    ValidationResult(
                        rule_id="skill_config_002",
                        section="skill",
                        key_path="skill.capabilities",
                        severity=ValidationSeverity.ERROR,
                        message="Skill must have at least one capability defined",
                    )
                )

        # Analyze results
        report = self._analyze_validation_results(report)

        # Emit skill validation event
        await self.event_bus.emit(
            SkillConfigurationValidated(
                skill_id=skill_id,
                validation_id=validation_id,
                is_valid=report.is_valid,
                issues_found=report.total_issues,
            )
        )

        return report

    async def _start_file_monitoring(self) -> None:
        """Start monitoring configuration files for changes."""
        try:
            config_files = [
                "configs/environments/development.yaml",
                "configs/environments/staging.yaml",
                "configs/environments/production.yaml",
                ".env",
            ]

            for config_file in config_files:
                if Path(config_file).exists():
                    # In a real implementation, you'd use a file watcher library
                    # like watchdog to monitor file changes
                    pass

        except Exception as e:
            self.logger.warning(f"Failed to start file monitoring: {str(e)}")

    async def _validation_maintenance_loop(self) -> None:
        """Background task for validation maintenance."""
        while True:
            try:
                # Clean up old validation reports
                if len(self.validation_history) > 100:
                    self.validation_history = self.validation_history[-50:]

                # Clean up validation cache
                if len(self.validation_cache) > 50:
                    oldest_keys = sorted(self.validation_cache.keys())[:25]
                    for key in oldest_keys:
                        del self.validation_cache[key]

                await asyncio.sleep(300)  # Run every 5 minutes

            except Exception as e:
                self.logger.error(f"Validation maintenance error: {str(e)}")
                await asyncio.sleep(300)

    async def _metrics_update_loop(self) -> None:
        """Background task for updating metrics."""
        while True:
            try:
                # Update validation statistics metrics
                self.metrics.set(
                    "config_validation_success_rate",
                    self.validation_stats["successful_validations"]
                    / max(self.validation_stats["total_validations"], 1),
                )

                self.metrics.set(
                    "config_validation_average_time",
                    self.validation_stats["average_validation_time"],
                )

                await asyncio.sleep(60)  # Update every minute

            except Exception as e:
                self.logger.error(f"Metrics update error: {str(e)}")
                await asyncio.sleep(60)

    def get_validation_report(self, validation_id: str) -> Optional[ValidationReport]:
        """Get a specific validation report."""
        return self.validation_cache.get(validation_id)

    def get_latest_validation_report(self) -> Optional[ValidationReport]:
        """Get the latest validation report."""
        if self.validation_history:
            return self.validation_history[-1]
        return None

    def get_validation_statistics(self) -> Dict[str, Any]:
        """Get validation statistics."""
        return {
            **self.validation_stats,
            "last_validation": self.last_validation.isoformat() if self.last_validation else None,
            "active_rules": len(self.rules),
            "cached_reports": len(self.validation_cache),
            "validation_level": self.validation_level.value,
            "environment": self.current_environment,
        }

    def set_validation_level(self, level: ValidationLevel) -> None:
        """Set the validation level."""
        self.validation_level = level
        self.logger.info(f"Validation level set to: {level.value}")

    def add_custom_rule(self, rule: ValidationRule) -> None:
        """Add a custom validation rule."""
        self.rules.append(rule)
        self.rule_cache[rule.rule_id] = rule
        self.logger.info(f"Added custom validation rule: {rule.rule_id}")

    def remove_rule(self, rule_id: str) -> bool:
        """Remove a validation rule."""
        if rule_id in self.rule_cache:
            rule = self.rule_cache[rule_id]
            self.rules.remove(rule)
            del self.rule_cache[rule_id]
            self.logger.info(f"Removed validation rule: {rule_id}")
            return True
        return False

    def enable_rule(self, rule_id: str) -> bool:
        """Enable a validation rule."""
        if rule_id in self.rule_cache:
            self.rule_cache[rule_id].enabled = True
            return True
        return False

    def disable_rule(self, rule_id: str) -> bool:
        """Disable a validation rule."""
        if rule_id in self.rule_cache:
            self.rule_cache[rule_id].enabled = False
            return True
        return False

    async def _health_check_callback(self) -> Dict[str, Any]:
        """Health check callback for the configuration validator."""
        try:
            return {
                "status": "healthy",
                "last_validation": (
                    self.last_validation.isoformat() if self.last_validation else None
                ),
                "validation_level": self.validation_level.value,
                "active_rules": len([r for r in self.rules if r.enabled]),
                "total_validations": self.validation_stats["total_validations"],
                "success_rate": self.validation_stats["successful_validations"]
                / max(self.validation_stats["total_validations"], 1),
            }
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    async def cleanup(self) -> None:
        """Cleanup validator resources."""
        try:
            # Cancel file watchers
            for watcher in self.file_watchers.values():
                if hasattr(watcher, "cancel"):
                    watcher.cancel()

            # Clear caches
            self.validation_cache.clear()
            self.rule_cache.clear()

            self.logger.info("ConfigValidator cleanup completed")

        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")

    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            if hasattr(self, "file_watchers") and self.file_watchers:
                self.logger.warning("ConfigValidator destroyed with active file watchers")
        except Exception:
            pass  # Ignore cleanup errors in destructor
