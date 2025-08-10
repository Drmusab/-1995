"""
YAML Configuration Loader for AI Assistant
Author: Drmusab
Last Modified: 2025-01-10

This module provides YAML-first configuration loading for the AI assistant,
replacing the previous Python-based configuration system while maintaining
backward compatibility.
"""

import os
import re
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field

try:
    from src.observability.logging.config import get_logger
except ImportError:
    # Fallback logger for testing
    import logging
    def get_logger(name):
        return logging.getLogger(name)


@dataclass
class YamlConfigLoader:
    """
    YAML-first configuration loader for the AI Assistant.
    
    This loader consolidates all configuration into YAML files with environment
    overrides and variable interpolation support.
    """
    
    def __init__(self, environment: Optional[str] = None, config_dir: str = "."):
        """
        Initialize the YAML configuration loader.
        
        Args:
            environment: Environment name (development, production, testing)
            config_dir: Directory containing configuration files
        """
        self.environment = environment or os.getenv("ENVIRONMENT", "development")
        self.config_dir = Path(config_dir)
        self.logger = get_logger(__name__)
        
        self._config: Dict[str, Any] = {}
        self._loaded = False
        
    def load(self) -> Dict[str, Any]:
        """
        Load configuration from YAML files.
        
        Returns:
            Complete configuration dictionary
        """
        if self._loaded:
            return self._config
            
        try:
            # Load base configuration
            base_config = self._load_yaml_file("config.yaml")
            if not base_config:
                raise FileNotFoundError("Base configuration file config.yaml not found")
                
            # Load environment-specific overrides
            env_config = self._load_yaml_file(f"config.{self.environment}.yaml")
            
            # Also try common abbreviations
            if not env_config:
                env_abbreviations = {
                    "development": "dev",
                    "production": "prod", 
                    "testing": "test"
                }
                
                if self.environment in env_abbreviations:
                    env_config = self._load_yaml_file(f"config.{env_abbreviations[self.environment]}.yaml")
            
            # Merge configurations (environment overrides base)
            self._config = self._deep_merge(base_config, env_config or {})
            
            # Apply environment variable interpolation
            self._config = self._interpolate_variables(self._config)
            
            # Set environment in config
            self._config.setdefault("app", {})["environment"] = self.environment
            
            self._loaded = True
            
            self.logger.info(f"Configuration loaded successfully for environment: {self.environment}")
            return self._config
            
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {str(e)}")
            raise
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation.
        
        Args:
            key: Configuration key in dot notation (e.g., 'database.url')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        if not self._loaded:
            self.load()
            
        return self._get_nested_value(self._config, key, default)
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get an entire configuration section.
        
        Args:
            section: Section name
            
        Returns:
            Configuration section as dictionary
        """
        return self.get(section, {})
    
    def _load_yaml_file(self, filename: str) -> Optional[Dict[str, Any]]:
        """
        Load a YAML configuration file.
        
        Args:
            filename: YAML filename
            
        Returns:
            Configuration dictionary or None if file doesn't exist
        """
        file_path = self.config_dir / filename
        
        if not file_path.exists():
            self.logger.debug(f"Configuration file not found: {file_path}")
            return None
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Parse YAML
            config = yaml.safe_load(content)
            
            self.logger.debug(f"Loaded configuration from: {file_path}")
            return config or {}
            
        except Exception as e:
            self.logger.error(f"Failed to load YAML file {file_path}: {str(e)}")
            raise
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deep merge two dictionaries.
        
        Args:
            base: Base configuration
            override: Override configuration
            
        Returns:
            Merged configuration
        """
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
                
        return result
    
    def _interpolate_variables(self, config: Any) -> Any:
        """
        Interpolate environment variables and other placeholders.
        
        Args:
            config: Configuration value (dict, list, str, etc.)
            
        Returns:
            Configuration with interpolated values
        """
        if isinstance(config, dict):
            return {k: self._interpolate_variables(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [self._interpolate_variables(item) for item in config]
        elif isinstance(config, str):
            return self._interpolate_string(config)
        else:
            return config
    
    def _interpolate_string(self, value: str) -> Union[str, int, float, bool]:
        """
        Interpolate variables in a string value.
        
        Args:
            value: String value with potential variables
            
        Returns:
            Interpolated value (converted to appropriate type)
        """
        # Environment variable pattern: ${env:VAR_NAME:default_value}
        env_pattern = re.compile(r'\$\{env:([^}:]+)(?::([^}]*))?\}')
        
        def replace_env_var(match):
            var_name = match.group(1)
            default_value = match.group(2) if match.group(2) is not None else ""
            return os.getenv(var_name, default_value)
        
        # Replace environment variables
        result = env_pattern.sub(replace_env_var, value)
        
        # Special variable replacements
        result = result.replace("${pid}", str(os.getpid()))
        
        # Try to convert to appropriate type
        return self._convert_type(result)
    
    def _convert_type(self, value: str) -> Union[str, int, float, bool]:
        """
        Convert string value to appropriate type.
        
        Args:
            value: String value
            
        Returns:
            Converted value
        """
        if value.lower() in ('true', 'yes', 'on'):
            return True
        elif value.lower() in ('false', 'no', 'off'):
            return False
        elif value.isdigit():
            return int(value)
        elif value.replace('.', '', 1).isdigit():
            return float(value)
        else:
            return value
    
    def _get_nested_value(self, config: Dict[str, Any], key: str, default: Any = None) -> Any:
        """
        Get nested value using dot notation.
        
        Args:
            config: Configuration dictionary
            key: Dot-notation key
            default: Default value
            
        Returns:
            Nested value or default
        """
        keys = key.split('.')
        current = config
        
        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return default
                
        return current


# Global configuration instance
_config_loader: Optional[YamlConfigLoader] = None


def get_config_loader(environment: Optional[str] = None, config_dir: str = ".") -> YamlConfigLoader:
    """
    Get the global configuration loader instance.
    
    Args:
        environment: Environment name
        config_dir: Configuration directory
        
    Returns:
        Configuration loader instance
    """
    global _config_loader
    
    if _config_loader is None:
        _config_loader = YamlConfigLoader(environment, config_dir)
        
    return _config_loader


def load_config(environment: Optional[str] = None, config_dir: str = ".") -> Dict[str, Any]:
    """
    Load configuration from YAML files.
    
    Args:
        environment: Environment name
        config_dir: Configuration directory
        
    Returns:
        Complete configuration dictionary
    """
    loader = get_config_loader(environment, config_dir)
    return loader.load()


def get_config(key: str, default: Any = None, environment: Optional[str] = None) -> Any:
    """
    Get a configuration value.
    
    Args:
        key: Configuration key in dot notation
        default: Default value
        environment: Environment name
        
    Returns:
        Configuration value or default
    """
    loader = get_config_loader(environment)
    return loader.get(key, default)


def get_config_section(section: str, environment: Optional[str] = None) -> Dict[str, Any]:
    """
    Get an entire configuration section.
    
    Args:
        section: Section name
        environment: Environment name
        
    Returns:
        Configuration section
    """
    loader = get_config_loader(environment)
    return loader.get_section(section)


# Compatibility layer for existing Python configuration classes
class ConfigCompat:
    """
    Compatibility layer to maintain existing Python configuration API
    while using YAML as the underlying source.
    """
    
    def __init__(self, environment: Optional[str] = None):
        self.environment = environment or os.getenv("ENVIRONMENT", "development")
        self._loader = get_config_loader(self.environment)
        self._config = self._loader.load()
        
    @property
    def DEBUG(self) -> bool:
        return self.environment == "development"
        
    @property
    def TESTING(self) -> bool:
        return self.environment == "testing"
        
    @property
    def app_name(self) -> str:
        return self._config.get("app", {}).get("name", "AI Assistant")
        
    @property
    def app_version(self) -> str:
        return self._config.get("app", {}).get("version", "1.0.0")
        
    @property
    def app_description(self) -> str:
        return self._config.get("app", {}).get("description", "Advanced AI Assistant")
        
    def get_database_config(self) -> Dict[str, Any]:
        """Get database configuration."""
        storage_config = self._config.get("integrations", {}).get("storage", {})
        db_config = storage_config.get("database", {})
        
        return {
            "url": db_config.get("url", "sqlite:///data/assistant.db"),
            "pool_size": db_config.get("pool_size", 10),
            "max_overflow": db_config.get("max_overflow", 20),
            "echo": db_config.get("echo", False),
            "auto_migrate": db_config.get("auto_migrate", False),
        }
        
    def get_cache_config(self) -> Dict[str, Any]:
        """Get cache configuration."""
        cache_config = self._config.get("integrations", {}).get("cache", {})
        redis_config = cache_config.get("redis", {})
        
        return {
            "redis_url": f"redis://{redis_config.get('host', 'localhost')}:{redis_config.get('port', 6379)}/{redis_config.get('db', 0)}",
            "enabled": cache_config.get("enabled", True),
            "default_ttl": cache_config.get("default_ttl", 3600),
            "max_connections": redis_config.get("max_connections", 10),
        }
        
    def get_security_config(self) -> Dict[str, Any]:
        """Get security configuration."""
        security_config = self._config.get("security", {})
        
        return {
            "authentication_enabled": security_config.get("authentication", {}).get("enabled", True),
            "authorization_enabled": security_config.get("authorization", {}).get("enabled", True),
            "encryption_enabled": security_config.get("encryption", {}).get("enabled", True),
            "jwt_secret": self._config.get("api", {}).get("rest", {}).get("authentication", {}).get("jwt_secret", ""),
        }
        
    def get_api_config(self) -> Dict[str, Any]:
        """Get API configuration."""
        api_config = self._config.get("api", {})
        rest_config = api_config.get("rest", {})
        
        return {
            "host": rest_config.get("host", "0.0.0.0"),
            "port": rest_config.get("port", 8000),
            "workers": rest_config.get("workers", 4),
            "debug": rest_config.get("debug", False),
            "reload": rest_config.get("reload", False),
        }
        
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration."""
        logging_config = self._config.get("observability", {}).get("logging", {})
        
        return {
            "level": logging_config.get("level", "INFO"),
            "format": logging_config.get("format", "structured"),
            "handlers": logging_config.get("handlers", ["console", "file"]),
            "component_levels": logging_config.get("component_levels", {}),
        }
        
    def get_component_config(self, component_name: str) -> Dict[str, Any]:
        """Get configuration for a specific component."""
        core_config = self._config.get("core", {})
        
        component_configs = {
            "engine": core_config.get("engine", {}),
            "component_manager": core_config.get("component_manager", {}),
            "workflow": core_config.get("workflow", {}),
            "sessions": core_config.get("sessions", {}),
            "interactions": core_config.get("interactions", {}),
            "plugins": core_config.get("plugins", {}),
            "memory": self._config.get("memory", {}),
            "learning": self._config.get("learning", {}),
            "skills": self._config.get("skills", {}),
            "api": self._config.get("api", {}),
            "security": self._config.get("security", {}),
            "observability": self._config.get("observability", {}),
        }
        
        return component_configs.get(component_name, {})


# Factory functions for backward compatibility
def create_development_config() -> ConfigCompat:
    """Create development configuration."""
    return ConfigCompat("development")


def create_production_config() -> ConfigCompat:
    """Create production configuration."""
    return ConfigCompat("production")


def create_testing_config() -> ConfigCompat:
    """Create testing configuration."""
    return ConfigCompat("testing")


def get_settings(environment: Optional[str] = None) -> ConfigCompat:
    """
    Get settings for the specified environment.
    
    Args:
        environment: Environment name
        
    Returns:
        Configuration instance
    """
    if environment is None:
        environment = os.getenv("ENVIRONMENT", "development")
        
    return ConfigCompat(environment)


# Export public interface
__all__ = [
    "YamlConfigLoader",
    "ConfigCompat",
    "get_config_loader",
    "load_config", 
    "get_config",
    "get_config_section",
    "create_development_config",
    "create_production_config", 
    "create_testing_config",
    "get_settings",
]