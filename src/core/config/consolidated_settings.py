"""
Consolidated Settings Module
Author: Drmusab
Last Modified: 2025-01-13

This module provides a single entry point for all configuration management,
consolidating and organizing all configuration-related functionality.
"""

# Core configuration
from .unified_config import (
    UnifiedConfigManager,
    get_unified_config,
    get_settings,
    create_development_settings,
    create_testing_settings,
    create_staging_settings,
    create_production_settings,
)

# YAML configuration loading
from .yaml_loader import (
    YamlConfigLoader,
    get_config_loader,
    load_config,
    get_config,
    get_config_section,
)

# Logging configuration
from .logging_config import (
    UnifiedLoggingConfig,
    get_logging_config,
    get_logger,
    setup_logging,
)

# Simple validation (the complex validator is available but not recommended)
from .validators.simple_validator import (
    ValidationLevel,
    ValidationType,
    ValidationSeverity,
    ValidationResult,
    EnhancedConfigValidator,
)

# Configuration types and enums for backward compatibility
from src.config_settings import (
    Environment,
    LogLevel,
    ComponentLifecycle,
    BaseSettings,
)

from src.core.performance_config import (
    PerformanceConfiguration,
    PerformanceThresholds,
    OptimizationSettings,
    get_default_performance_config,
    get_performance_recommendations,
)

from src.di_config import (
    ComponentConfiguration,
    create_configured_container,
)

__all__ = [
    # Core configuration
    "UnifiedConfigManager",
    "get_unified_config",
    "get_settings",
    "create_development_settings",
    "create_testing_settings",
    "create_staging_settings",
    "create_production_settings",
    
    # YAML loading
    "YamlConfigLoader",
    "get_config_loader",
    "load_config",
    "get_config",
    "get_config_section",
    
    # Logging
    "UnifiedLoggingConfig",
    "get_logging_config",
    "get_logger",
    "setup_logging",
    
    # Validation
    "ValidationLevel",
    "ValidationType", 
    "ValidationSeverity",
    "ValidationResult",
    "EnhancedConfigValidator",
    
    # Legacy compatibility
    "Environment",
    "LogLevel",
    "ComponentLifecycle",
    "BaseSettings",
    "PerformanceConfiguration",
    "PerformanceThresholds",
    "OptimizationSettings",
    "get_default_performance_config",
    "get_performance_recommendations",
    "ComponentConfiguration",
    "create_configured_container",
]


def get_all_config() -> dict:
    """Get all configuration as a dictionary."""
    unified_config = get_unified_config()
    return unified_config.get_config_dict()


def validate_all_config() -> list:
    """Validate all configuration and return any errors."""
    unified_config = get_unified_config()
    return unified_config.validate_configuration()


def initialize_config_system(environment: str = None) -> UnifiedConfigManager:
    """
    Initialize the complete configuration system.
    
    Args:
        environment: Environment name (development, production, testing)
        
    Returns:
        Initialized UnifiedConfigManager
    """
    # Initialize unified configuration
    unified_config = get_unified_config(environment)
    
    # Setup logging
    setup_logging()
    
    # Validate configuration
    errors = validate_all_config()
    if errors:
        logger = get_logger(__name__)
        logger.warning(f"Configuration validation warnings: {errors}")
    
    return unified_config