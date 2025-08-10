"""
Unified Configuration Management System
Author: Drmusab
Last Modified: 2025-01-13

This module provides a unified, YAML-first configuration system that consolidates
all configuration management for the AI assistant. It replaces multiple Python
configuration files with a single, comprehensive system.
"""

import os
import asyncio
import logging
from typing import Any, Dict, List, Optional, Type, TypeVar
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime, timezone

from src.core.config.yaml_loader import YamlConfigLoader, get_config_loader
from src.core.dependency_injection import Container, LifecycleScope

# Type definitions
T = TypeVar("T")


@dataclass
class ComponentRegistration:
    """Component registration information from YAML config."""
    
    name: str
    implementation: str
    lifecycle: str = "singleton"
    enabled: bool = True
    factory_params: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)


class UnifiedConfigManager:
    """
    Unified Configuration Management System.
    
    This class consolidates functionality from:
    - config_settings.py (BaseSettings)
    - di_config.py (ComponentConfiguration)  
    - observability/logging/config.py (LoggingConfig)
    - performance_config.py (PerformanceConfiguration)
    - config_validator.py validation rules
    
    All configuration is now YAML-first with Python providing the interface.
    """
    
    def __init__(self, environment: Optional[str] = None, config_dir: Optional[str] = None):
        """
        Initialize the unified configuration manager.
        
        Args:
            environment: Environment name (development, production, testing)
            config_dir: Directory containing configuration files (defaults to core/config)
        """
        self.environment = environment or os.getenv("ENVIRONMENT", "development")
        
        # Default to the organized config directory structure
        if config_dir is None:
            # Get the path to the core/config directory relative to this file
            current_dir = Path(__file__).parent
            config_dir = str(current_dir)
        
        self.config_dir = Path(config_dir)
        
        # Initialize YAML loader
        self.yaml_loader = get_config_loader(self.environment, str(self.config_dir))
        self._config = self.yaml_loader.load()
        
        # Initialize dependency injection container
        self.container = Container()
        
        # Track initialization
        self._initialized = False
        self._components_registered = False
        
        self.logger = self._setup_logging()
        self.logger.info(f"UnifiedConfigManager initialized for environment: {self.environment}")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging using YAML configuration."""
        logging_config = self.get_logging_config()
        
        # Configure root logger
        log_level = getattr(logging, logging_config.get("level", "INFO"))
        
        # Create logger
        logger = logging.getLogger(__name__)
        logger.setLevel(log_level)
        
        if not logger.handlers:
            # Create console handler if none exist
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    # Core Configuration Access Methods
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation."""
        return self.yaml_loader.get(key, default)
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """Get entire configuration section."""
        return self.yaml_loader.get_section(section)
    
    def get_all(self) -> Dict[str, Any]:
        """Get complete configuration."""
        return self._config
    
    # Legacy Configuration Interfaces (for backward compatibility)
    
    def get_app_config(self) -> Dict[str, Any]:
        """Get application configuration."""
        return self.get_section("app")
    
    def get_core_config(self) -> Dict[str, Any]:
        """Get core system configuration."""
        return self.get_section("core")
    
    def get_database_config(self) -> Dict[str, Any]:
        """Get database configuration."""
        storage_config = self.get("integrations.storage", {})
        db_config = storage_config.get("database", {})
        
        return {
            "url": db_config.get("url", "sqlite:///data/assistant.db"),
            "pool_size": db_config.get("pool_size", 10),
            "max_overflow": db_config.get("max_overflow", 20),
            "pool_pre_ping": True,
            "pool_recycle": db_config.get("pool_recycle", 300),
            "echo": db_config.get("echo", False),
            "echo_pool": False,
            "migration_dir": "migrations",
            "backup_enabled": True,
            "backup_interval": 3600,
        }
    
    def get_cache_config(self) -> Dict[str, Any]:
        """Get cache configuration."""
        cache_config = self.get("integrations.cache", {})
        redis_config = cache_config.get("redis", {})
        
        # Build Redis URL from components
        redis_url = f"redis://{redis_config.get('host', 'localhost')}:{redis_config.get('port', 6379)}/{redis_config.get('db', 0)}"
        
        return {
            "redis_url": redis_url,
            "default_ttl": cache_config.get("default_ttl", 3600),
            "max_connections": redis_config.get("max_connections", 10),
            "socket_timeout": redis_config.get("socket_timeout", 5.0),
            "socket_connect_timeout": redis_config.get("socket_connect_timeout", 5.0),
            "retry_on_timeout": redis_config.get("retry_on_timeout", True),
            "health_check_interval": redis_config.get("health_check_interval", 30.0),
            "enabled": cache_config.get("enabled", True),
        }
    
    def get_security_config(self) -> Dict[str, Any]:
        """Get security configuration."""
        security_config = self.get("security", {})
        auth_config = security_config.get("authentication", {})
        api_auth_config = self.get("api.rest.authentication", {})
        
        return {
            "secret_key": api_auth_config.get("jwt_secret", ""),
            "encryption_enabled": security_config.get("encryption", {}).get("enabled", True),
            "authentication_required": auth_config.get("enabled", True),
            "authorization_enabled": security_config.get("authorization", {}).get("enabled", True),
            "session_timeout": auth_config.get("session_timeout", 3600),
            "max_login_attempts": auth_config.get("max_login_attempts", 5),
            "password_min_length": 8,
            "jwt_algorithm": "HS256",
            "jwt_expiration": api_auth_config.get("token_expiry", 3600),
            "sanitization_enabled": security_config.get("sanitization", {}).get("enabled", True),
            "audit_logging": security_config.get("audit", {}).get("enabled", True),
        }
    
    def get_monitoring_config(self) -> Dict[str, Any]:
        """Get monitoring configuration."""
        observability_config = self.get("observability", {})
        metrics_config = observability_config.get("metrics", {})
        tracing_config = observability_config.get("tracing", {})
        health_config = observability_config.get("health_checks", {})
        
        return {
            "metrics_enabled": metrics_config.get("enabled", True),
            "tracing_enabled": tracing_config.get("enabled", True),
            "profiling_enabled": observability_config.get("profiling", {}).get("enabled", False),
            "health_check_interval": health_config.get("interval", 30.0),
            "metrics_port": metrics_config.get("port", 9090),
            "metrics_path": metrics_config.get("path", "/metrics"),
            "jaeger_endpoint": f"{tracing_config.get('agent_host', 'localhost')}:{tracing_config.get('agent_port', 6831)}",
            "prometheus_pushgateway": None,
            "log_sampling_rate": tracing_config.get("sampling_rate", 1.0),
        }
    
    def get_processing_config(self) -> Dict[str, Any]:
        """Get processing configuration."""
        core_config = self.get("core.engine", {})
        processing_config = self.get("processing", {})
        
        return {
            "enable_speech_processing": core_config.get("enable_speech_processing", True),
            "enable_vision_processing": core_config.get("enable_vision_processing", True),
            "enable_multimodal_fusion": core_config.get("enable_multimodal_fusion", True),
            "enable_reasoning": core_config.get("enable_reasoning", True),
            "enable_learning": core_config.get("enable_learning", True),
            "max_concurrent_requests": core_config.get("max_concurrent_requests", 10),
            "default_timeout": core_config.get("default_timeout_seconds", 30.0),
            "request_queue_size": 100,
            "default_quality": core_config.get("default_quality_level", "balanced"),
            "adaptive_quality": core_config.get("enable_adaptive_quality", True),
            "default_voice": "neural",
            "speech_model": processing_config.get("speech", {}).get("whisper_model", "base"),
            "tts_model": "tacotron2",
            "vision_model": "clip-vit-base",
            "ocr_enabled": True,
            "face_recognition_enabled": False,
        }
    
    def get_memory_config(self) -> Dict[str, Any]:
        """Get memory configuration."""
        memory_config = self.get("memory", {})
        working_memory = memory_config.get("working_memory", {})
        context_config = memory_config.get("context", {})
        semantic_memory = memory_config.get("semantic_memory", {})
        
        return {
            "working_memory_size": working_memory.get("capacity", 1000),
            "context_window_size": context_config.get("max_context_size", 4096),
            "episodic_memory_retention": memory_config.get("episodic_memory", {}).get("retention_days", 30),
            "semantic_memory_threshold": semantic_memory.get("similarity_threshold", 0.7),
            "consolidation_interval": memory_config.get("episodic_memory", {}).get("consolidation_interval", 3600),
            "vector_store_type": memory_config.get("vector_store", {}).get("backend", "faiss"),
            "embedding_model": semantic_memory.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2"),
            "memory_compression": working_memory.get("compression", True),
        }
    
    def get_learning_config(self) -> Dict[str, Any]:
        """Get learning configuration."""
        learning_config = self.get("learning", {})
        continual_learning = learning_config.get("continual_learning", {})
        preference_learning = learning_config.get("preference_learning", {})
        feedback_processing = learning_config.get("feedback_processing", {})
        model_adaptation = learning_config.get("model_adaptation", {})
        
        return {
            "continual_learning_enabled": continual_learning.get("enabled", True),
            "preference_learning_enabled": preference_learning.get("enabled", True),
            "feedback_processing_enabled": feedback_processing.get("enabled", True),
            "model_adaptation_enabled": model_adaptation.get("enabled", True),
            "learning_rate": continual_learning.get("learning_rate", 0.001),
            "adaptation_threshold": 0.1,
            "feedback_weight": 0.5,
            "preference_weight": preference_learning.get("preference_decay", 0.3),
            "learning_update_interval": 300,
            "model_save_interval": 3600,
        }
    
    def get_plugins_config(self) -> Dict[str, Any]:
        """Get plugins configuration."""
        plugins_config = self.get("core.plugins", {})
        
        return {
            "enabled": True,
            "auto_discovery": plugins_config.get("auto_discovery", True),
            "hot_reload": plugins_config.get("hot_reload", False),
            "security_validation": plugins_config.get("security_validation", True),
            "max_plugins": plugins_config.get("max_plugins", 100),
            "plugin_directories": plugins_config.get("directories", ["plugins/", "src/plugins/", "data/plugins/"]),
            "sandbox_enabled": plugins_config.get("sandbox_enabled", True),
            "plugin_timeout": plugins_config.get("plugin_timeout", 30.0),
        }
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration."""
        logging_config = self.get("observability.logging", {})
        
        return {
            "level": logging_config.get("level", "INFO"),
            "format": logging_config.get("format", "structured"),
            "handlers": logging_config.get("handlers", ["console", "file"]),
            "file_rotation": logging_config.get("file_rotation", True),
            "max_file_size": logging_config.get("max_file_size", "100MB"),
            "backup_count": logging_config.get("backup_count", 5),
            "structured_logging": logging_config.get("structured_logging", True),
            "request_logging": logging_config.get("request_logging", True),
            "performance_logging": logging_config.get("performance_logging", True),
            "component_levels": logging_config.get("component_levels", {}),
        }
    
    def get_performance_config(self) -> Dict[str, Any]:
        """Get performance configuration."""
        performance_config = self.get("performance", {})
        monitoring_config = self.get("performance_monitoring", {})
        
        return {
            # From performance section
            "max_memory_usage_gb": performance_config.get("max_memory_usage_gb", 4.0),
            "max_cpu_usage_percent": performance_config.get("max_cpu_usage_percent", 80.0),
            "max_concurrent_requests": performance_config.get("max_concurrent_requests", 50),
            "max_concurrent_workflows": performance_config.get("max_concurrent_workflows", 20),
            "max_concurrent_sessions": performance_config.get("max_concurrent_sessions", 500),
            
            # From performance_monitoring section
            "thresholds": monitoring_config.get("thresholds", {}),
            "optimizations": monitoring_config.get("optimizations", {}),
            "monitoring_enabled": monitoring_config.get("monitoring_enabled", True),
            "profiling_enabled": monitoring_config.get("profiling_enabled", False),
            "metrics_collection_interval": monitoring_config.get("metrics_collection_interval", 60),
            "performance_logging": monitoring_config.get("performance_logging", True),
        }
    
    def get_validation_config(self) -> Dict[str, Any]:
        """Get validation configuration."""
        return self.get_section("validation")
    
    # Dependency Injection Methods
    
    def get_component_registrations(self) -> Dict[str, List[ComponentRegistration]]:
        """Get component registrations from YAML configuration."""
        di_config = self.get("dependency_injection", {})
        components_config = di_config.get("components", {})
        
        registrations = {}
        
        for category_name, category_components in components_config.items():
            registrations[category_name] = []
            
            for component_name, component_config in category_components.items():
                registration = ComponentRegistration(
                    name=component_name,
                    implementation=component_config.get("implementation", ""),
                    lifecycle=component_config.get("lifecycle", "singleton"),
                    enabled=component_config.get("enabled", True),
                    factory_params=component_config.get("factory_params", {}),
                    dependencies=component_config.get("dependencies", [])
                )
                registrations[category_name].append(registration)
        
        return registrations
    
    async def register_components(self) -> None:
        """Register all components in the dependency injection container."""
        if self._components_registered:
            return
        
        registrations = self.get_component_registrations()
        
        for category_name, category_registrations in registrations.items():
            self.logger.debug(f"Registering {category_name} components...")
            
            for registration in category_registrations:
                if not registration.enabled:
                    self.logger.debug(f"Skipping disabled component: {registration.name}")
                    continue
                
                try:
                    await self._register_single_component(registration)
                except Exception as e:
                    self.logger.error(f"Failed to register component {registration.name}: {str(e)}")
        
        self._components_registered = True
        self.logger.info("All components registered successfully")
    
    async def _register_single_component(self, registration: ComponentRegistration) -> None:
        """Register a single component."""
        try:
            # Import the component class
            module_path, class_name = registration.implementation.rsplit(".", 1)
            module = __import__(module_path, fromlist=[class_name])
            component_class = getattr(module, class_name)
            
            # Determine lifecycle scope
            lifecycle_map = {
                "singleton": LifecycleScope.SINGLETON,
                "transient": LifecycleScope.TRANSIENT,
                "scoped": LifecycleScope.SCOPED,
            }
            scope = lifecycle_map.get(registration.lifecycle, LifecycleScope.SINGLETON)
            
            # Create factory function if needed
            if registration.factory_params:
                # Resolve parameter values from configuration
                resolved_params = self._resolve_parameters(registration.factory_params)
                
                def factory():
                    return component_class(**resolved_params)
                
                self.container.register(component_class, factory=factory, scope=scope)
            else:
                self.container.register(component_class, scope=scope)
            
            self.logger.debug(f"Registered component: {registration.name} -> {registration.implementation}")
            
        except Exception as e:
            self.logger.error(f"Failed to register component {registration.name}: {str(e)}")
            raise
    
    def _resolve_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve parameter values from configuration."""
        resolved = {}
        
        for key, value in params.items():
            if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                # Configuration reference
                config_key = value[2:-1]  # Remove ${ and }
                resolved[key] = self.get(config_key)
            else:
                resolved[key] = value
        
        return resolved
    
    def get_container(self) -> Container:
        """Get the dependency injection container."""
        return self.container
    
    def get_component(self, component_type: Type[T]) -> T:
        """Get a component from the dependency injection container."""
        return self.container.get(component_type)
    
    # System Information
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        import platform
        import multiprocessing
        
        return {
            "platform": platform.platform(),
            "architecture": platform.architecture(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "cpu_count": multiprocessing.cpu_count(),
            "hostname": platform.node(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        }
    
    def get_config_dict(self) -> Dict[str, Any]:
        """Get complete configuration as dictionary."""
        config_dict = self._config.copy()
        
        # Add computed system information
        config_dict["system"] = self.get_system_info()
        
        return config_dict
    
    def validate_configuration(self) -> List[str]:
        """Validate the current configuration."""
        errors = []
        validation_config = self.get_validation_config()
        
        if not validation_config:
            return errors
        
        # Basic validation rules
        rules = validation_config.get("rules", {})
        
        for rule_category, rule_list in rules.items():
            for rule in rule_list:
                if not rule.get("enabled", True):
                    continue
                
                try:
                    rule_errors = self._validate_rule(rule)
                    errors.extend(rule_errors)
                except Exception as e:
                    errors.append(f"Validation rule {rule.get('rule_id', 'unknown')} failed: {str(e)}")
        
        return errors
    
    def _validate_rule(self, rule: Dict[str, Any]) -> List[str]:
        """Validate a single rule."""
        errors = []
        rule_id = rule.get("rule_id", "unknown")
        section = rule.get("section", "")
        
        # Check required keys
        if "required_keys" in rule:
            for key in rule["required_keys"]:
                if not self.get(f"{section}.{key}" if section else key):
                    errors.append(f"Rule {rule_id}: Required key '{key}' missing from section '{section}'")
        
        # Check conditions
        if "conditions" in rule:
            conditions = rule["conditions"]
            section_config = self.get_section(section) if section else self._config
            
            for key, condition in conditions.items():
                value = section_config.get(key)
                
                if isinstance(condition, dict):
                    if "min" in condition and value is not None and value < condition["min"]:
                        errors.append(f"Rule {rule_id}: {key} value {value} is below minimum {condition['min']}")
                    
                    if "max" in condition and value is not None and value > condition["max"]:
                        errors.append(f"Rule {rule_id}: {key} value {value} exceeds maximum {condition['max']}")
                    
                    if "min_length" in condition and isinstance(value, str) and len(value) < condition["min_length"]:
                        errors.append(f"Rule {rule_id}: {key} length {len(value)} is below minimum {condition['min_length']}")
                
                elif condition != value:
                    errors.append(f"Rule {rule_id}: {key} value {value} does not match required value {condition}")
        
        return errors
    
    async def initialize(self) -> None:
        """Initialize the configuration system."""
        if self._initialized:
            return
        
        try:
            # Register components
            await self.register_components()
            
            # Validate configuration
            validation_errors = self.validate_configuration()
            if validation_errors:
                self.logger.warning(f"Configuration validation warnings: {validation_errors}")
            
            self._initialized = True
            self.logger.info("UnifiedConfigManager initialization completed")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize UnifiedConfigManager: {str(e)}")
            raise


# Global instance
_unified_config: Optional[UnifiedConfigManager] = None


def get_unified_config(environment: Optional[str] = None, config_dir: Optional[str] = None) -> UnifiedConfigManager:
    """Get the global unified configuration manager."""
    global _unified_config
    
    if _unified_config is None:
        _unified_config = UnifiedConfigManager(environment, config_dir)
    
    return _unified_config


# Backward compatibility functions
def get_settings(environment: Optional[str] = None) -> UnifiedConfigManager:
    """Get settings for the specified environment (backward compatibility)."""
    return get_unified_config(environment)


def create_development_settings() -> UnifiedConfigManager:
    """Create development environment settings."""
    return UnifiedConfigManager("development")


def create_testing_settings() -> UnifiedConfigManager:
    """Create testing environment settings."""
    return UnifiedConfigManager("testing")


def create_staging_settings() -> UnifiedConfigManager:
    """Create staging environment settings."""
    return UnifiedConfigManager("staging")


def create_production_settings() -> UnifiedConfigManager:
    """Create production environment settings."""
    return UnifiedConfigManager("production")