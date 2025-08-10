"""
Base Configuration Settings for AI Assistant (Legacy Compatibility Layer)
Author: Drmusab
Last Modified: 2025-01-13

This module provides backward compatibility for the AI assistant configuration system.
The actual configuration is now handled by the unified YAML-first configuration system
in src.core.config.unified_config, but this module maintains the original API for
existing code that depends on it.
"""

import importlib
import inspect
import logging
import multiprocessing
import os
import platform
import sys
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type, Union, TypeVar

import asyncio

# Import the new unified configuration system
from src.core.config.unified_config import UnifiedConfigManager, get_unified_config

# Backward compatibility imports
from src.core.config.loader import ConfigLoader
from src.core.config.yaml_loader import get_config, get_config_section, YamlConfigLoader

# Legacy imports for type hints and backward compatibility  
T = TypeVar('T')


# Keep original enums and dataclasses for backward compatibility
class Environment(Enum):
    """Application environment types."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class LogLevel(Enum):
    """Logging levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class ComponentLifecycle(Enum):
    """Component lifecycle management types."""
    SINGLETON = "singleton"
    TRANSIENT = "transient"
    SCOPED = "scoped"
    LAZY = "lazy"


# Legacy dataclasses for backward compatibility
@dataclass
class DatabaseConfig:
    """Database configuration settings."""
    url: str = "sqlite:///data/assistant.db"
    pool_size: int = 10
    max_overflow: int = 20
    pool_pre_ping: bool = True
    pool_recycle: int = 300
    echo: bool = False
    echo_pool: bool = False
    migration_dir: str = "migrations"
    backup_enabled: bool = True
    backup_interval: int = 3600


@dataclass
class CacheConfig:
    """Cache configuration settings."""
    redis_url: str = "redis://localhost:6379/0"
    default_ttl: int = 3600
    max_connections: int = 10
    socket_timeout: float = 5.0
    socket_connect_timeout: float = 5.0
    retry_on_timeout: bool = True
    health_check_interval: float = 30.0
    enabled: bool = True


@dataclass
class SecurityConfig:
    """Security configuration settings."""
    secret_key: str = field(default_factory=lambda: str(uuid.uuid4()))
    encryption_enabled: bool = True
    authentication_required: bool = True
    authorization_enabled: bool = True
    session_timeout: int = 3600
    max_login_attempts: int = 5
    password_min_length: int = 8
    jwt_algorithm: str = "HS256"
    jwt_expiration: int = 3600
    sanitization_enabled: bool = True
    audit_logging: bool = True


@dataclass
class MonitoringConfig:
    """Monitoring and observability configuration."""
    metrics_enabled: bool = True
    tracing_enabled: bool = True
    profiling_enabled: bool = False
    health_check_interval: float = 30.0
    metrics_port: int = 9090
    metrics_path: str = "/metrics"
    jaeger_endpoint: Optional[str] = None
    prometheus_pushgateway: Optional[str] = None
    log_sampling_rate: float = 1.0


@dataclass
class ProcessingConfig:
    """Processing pipeline configuration."""
    enable_speech_processing: bool = True
    enable_vision_processing: bool = True
    enable_multimodal_fusion: bool = True
    enable_reasoning: bool = True
    enable_learning: bool = True
    max_concurrent_requests: int = 10
    default_timeout: float = 30.0
    request_queue_size: int = 100
    default_quality: str = "balanced"
    adaptive_quality: bool = True
    default_voice: str = "neural"
    speech_model: str = "whisper-base"
    tts_model: str = "tacotron2"
    vision_model: str = "clip-vit-base"
    ocr_enabled: bool = True
    face_recognition_enabled: bool = False


@dataclass
class MemoryConfig:
    """Memory management configuration."""
    working_memory_size: int = 1000
    context_window_size: int = 4096
    episodic_memory_retention: int = 30
    semantic_memory_threshold: float = 0.7
    consolidation_interval: int = 3600
    vector_store_type: str = "faiss"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    memory_compression: bool = True


@dataclass
class LearningConfig:
    """Learning and adaptation configuration."""
    continual_learning_enabled: bool = True
    preference_learning_enabled: bool = True
    feedback_processing_enabled: bool = True
    model_adaptation_enabled: bool = False
    learning_rate: float = 0.001
    adaptation_threshold: float = 0.1
    feedback_weight: float = 0.5
    preference_weight: float = 0.3
    learning_update_interval: int = 300
    model_save_interval: int = 3600


@dataclass
class PluginConfig:
    """Plugin system configuration."""
    enabled: bool = True
    auto_discovery: bool = True
    hot_reload: bool = False
    security_validation: bool = True
    max_plugins: int = 100
    plugin_directories: List[str] = field(
        default_factory=lambda: ["plugins/", "src/plugins/", "data/plugins/"]
    )
    sandbox_enabled: bool = True
    plugin_timeout: float = 30.0


class BaseSettings:
    """
    Legacy Base Configuration Settings Wrapper.
    
    This class provides backward compatibility for the original BaseSettings API
    while delegating all functionality to the new unified configuration system.
    """

    def __init__(self, environment: Environment = Environment.DEVELOPMENT):
        """
        Initialize base settings.

        Args:
            environment: Application environment
        """
        self.environment = environment
        
        # Initialize the unified configuration manager
        self._unified_config = get_unified_config(environment.value)
        
        # Delegate to unified config for logger setup
        self.logger = self._unified_config.logger
        
        # Initialize legacy configuration objects from YAML
        self._init_legacy_configs()
        
        # Initialize dependency injection container (delegate to unified config)
        self.container = self._unified_config.get_container()
        
        self.logger.info(f"BaseSettings initialized for {environment.value} environment (using unified YAML configuration)")

    def _init_legacy_configs(self) -> None:
        """Initialize legacy configuration objects from the unified configuration."""
        # Create legacy config objects from YAML data
        db_config_data = self._unified_config.get_database_config()
        self.database = DatabaseConfig(**db_config_data)
        
        cache_config_data = self._unified_config.get_cache_config()
        self.cache = CacheConfig(**cache_config_data)
        
        security_config_data = self._unified_config.get_security_config()
        self.security = SecurityConfig(**security_config_data)
        
        monitoring_config_data = self._unified_config.get_monitoring_config()
        self.monitoring = MonitoringConfig(**monitoring_config_data)
        
        processing_config_data = self._unified_config.get_processing_config()
        self.processing = ProcessingConfig(**processing_config_data)
        
        memory_config_data = self._unified_config.get_memory_config()
        self.memory = MemoryConfig(**memory_config_data)
        
        learning_config_data = self._unified_config.get_learning_config()
        self.learning = LearningConfig(**learning_config_data)
        
        plugins_config_data = self._unified_config.get_plugins_config()
        self.plugins = PluginConfig(**plugins_config_data)
        
        # App information from YAML
        app_config = self._unified_config.get_app_config()
        self.app_name = app_config.get("name", "AI Assistant")
        self.app_version = app_config.get("version", "1.0.0")
        self.app_description = app_config.get("description", "Advanced AI Assistant with Multimodal Capabilities")
        
        # System information
        self.system_info = self._unified_config.get_system_info()

    @asynccontextmanager
    async def application_lifespan(self):
        """
        Application lifespan context manager for proper startup and shutdown.
        """
        try:
            # Initialize the unified configuration system
            await self._unified_config.initialize()
            yield
        finally:
            # Cleanup is handled by the unified configuration system
            pass

    async def initialize_system(self) -> None:
        """Initialize the complete AI assistant system."""
        await self._unified_config.initialize()

    async def cleanup_system(self) -> None:
        """Cleanup the AI assistant system."""
        # Cleanup is handled by the unified configuration system
        pass

    def get_component(self, component_type: Type[T]) -> T:
        """Get a component from the dependency injection container."""
        return self._unified_config.get_component(component_type)

    def get_config_dict(self) -> Dict[str, Any]:
        """Get complete configuration as dictionary."""
        return self._unified_config.get_config_dict()

    def validate_configuration(self) -> List[str]:
        """Validate the current configuration."""
        return self._unified_config.validate_configuration()

    def __str__(self) -> str:
        """String representation of settings."""
        return f"BaseSettings(environment={self.environment.value}, app={self.app_name})"

    def __repr__(self) -> str:
        """Detailed string representation."""
        return (
            f"BaseSettings("
            f"environment={self.environment.value}, "
            f"app_name='{self.app_name}', "
            f"version='{self.app_version}', "
            f"components_registered={len(self.container._services) if hasattr(self.container, '_services') else 0}"
            f")"
        )


# Environment-specific settings factories (maintaining backward compatibility)
def create_development_settings() -> BaseSettings:
    """Create development environment settings."""
    return BaseSettings(Environment.DEVELOPMENT)


def create_testing_settings() -> BaseSettings:
    """Create testing environment settings."""
    return BaseSettings(Environment.TESTING)


def create_staging_settings() -> BaseSettings:
    """Create staging environment settings."""
    return BaseSettings(Environment.STAGING)


def create_production_settings() -> BaseSettings:
    """Create production environment settings."""
    return BaseSettings(Environment.PRODUCTION)


# Global settings factory (maintaining backward compatibility)
def get_settings(environment: Optional[str] = None) -> BaseSettings:
    """
    Get settings for the specified environment.

    Args:
        environment: Environment name (defaults to ENVIRONMENT environment variable)

    Returns:
        BaseSettings instance for the environment
    """
    if environment is None:
        environment = os.getenv("ENVIRONMENT", "development").lower()

    env_mapping = {
        "development": Environment.DEVELOPMENT,
        "testing": Environment.TESTING,
        "staging": Environment.STAGING,
        "production": Environment.PRODUCTION,
    }

    env_enum = env_mapping.get(environment, Environment.DEVELOPMENT)
    return BaseSettings(env_enum)



