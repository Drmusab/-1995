"""
Simplified Logging Configuration (YAML-first approach)
Author: Drmusab
Last Modified: 2025-01-13

This module provides a simplified logging configuration that uses the unified
YAML configuration system, replacing the complex original logging system with
a much simpler approach while maintaining backward compatibility.
"""

import logging
import logging.handlers
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Import the unified configuration system
from src.core.config.unified_config import get_unified_config


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger instance using unified configuration.
    
    Args:
        name: Logger name (defaults to caller's module name)
        
    Returns:
        Configured logger instance
    """
    if name is None:
        import inspect
        frame = inspect.currentframe().f_back
        name = frame.f_globals.get("__name__", "unknown")
    
    logger = logging.getLogger(name)
    
    # Configure logger if not already configured
    if not logger.handlers:
        _configure_logger(logger, name)
    
    return logger


def setup_logging(
    level: str = "INFO",
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    log_file: Optional[str] = None,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
) -> None:
    """
    Setup basic logging configuration.
    
    Args:
        level: Log level
        format: Log format string
        log_file: Optional log file path
        max_bytes: Maximum log file size
        backup_count: Number of backup files
    """
    # Get logging configuration from unified config
    try:
        unified_config = get_unified_config()
        logging_config = unified_config.get_logging_config()
    except Exception:
        # Fallback to basic configuration
        logging_config = {
            "level": level,
            "format": format,
            "handlers": ["console"],
        }
    
    # Configure root logger
    root_logger = logging.getLogger()
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Set log level
    log_level = getattr(logging, logging_config.get("level", level))
    root_logger.setLevel(log_level)
    
    # Create formatter
    formatter = logging.Formatter(logging_config.get("format", format))
    
    # Add console handler
    if "console" in logging_config.get("handlers", ["console"]):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # Add file handler
    if "file" in logging_config.get("handlers", []) or log_file:
        file_path = log_file or "data/logs/application.log"
        
        # Create directory if needed
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            file_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8"
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def _configure_logger(logger: logging.Logger, name: str) -> None:
    """Configure a specific logger using unified configuration."""
    try:
        unified_config = get_unified_config()
        logging_config = unified_config.get_logging_config()
        
        # Set component-specific log level if configured
        component_levels = logging_config.get("component_levels", {})
        if name in component_levels:
            level = getattr(logging, component_levels[name])
            logger.setLevel(level)
        
    except Exception:
        # Fallback to basic configuration
        pass


# Backward compatibility classes and functions
class LogLevel:
    """Log level constants for backward compatibility."""
    TRACE = 5
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50


class LogFormat:
    """Log format constants for backward compatibility."""
    JSON = "json"
    STRUCTURED = "structured"
    CONSOLE = "console"
    COMPACT = "compact"


class LoggingConfig:
    """Legacy logging configuration class for backward compatibility."""
    
    def __init__(self):
        """Initialize with unified configuration."""
        try:
            unified_config = get_unified_config()
            self.config = unified_config.get_logging_config()
        except Exception:
            self.config = {
                "level": "INFO",
                "format": "structured",
                "handlers": ["console", "file"],
                "component_levels": {},
            }
    
    @property
    def level(self) -> str:
        return self.config.get("level", "INFO")
    
    @property
    def format_type(self) -> str:
        return self.config.get("format", "structured")
    
    @property
    def handlers(self) -> list:
        return self.config.get("handlers", ["console", "file"])


class ContextualLogger:
    """Simplified contextual logger for backward compatibility."""
    
    def __init__(self, logger: logging.Logger, context: Optional[Dict[str, Any]] = None):
        self.logger = logger
        self.context = context or {}
    
    def with_context(self, **kwargs) -> "ContextualLogger":
        """Create a new logger with additional context."""
        new_context = {**self.context, **kwargs}
        return ContextualLogger(self.logger, new_context)
    
    def debug(self, message: str, *args, **kwargs):
        """Log at DEBUG level."""
        self.logger.debug(message, *args, **kwargs)
    
    def info(self, message: str, *args, **kwargs):
        """Log at INFO level."""
        self.logger.info(message, *args, **kwargs)
    
    def warning(self, message: str, *args, **kwargs):
        """Log at WARNING level."""
        self.logger.warning(message, *args, **kwargs)
    
    def error(self, message: str, *args, **kwargs):
        """Log at ERROR level."""
        self.logger.error(message, *args, **kwargs)
    
    def critical(self, message: str, *args, **kwargs):
        """Log at CRITICAL level."""
        self.logger.critical(message, *args, **kwargs)
    
    def exception(self, message: str, *args, **kwargs):
        """Log an exception."""
        kwargs["exc_info"] = True
        self.error(message, *args, **kwargs)


def configure_logging(config: Optional[LoggingConfig] = None) -> None:
    """Configure the global logging system."""
    if config is None:
        config = LoggingConfig()
    
    setup_logging(
        level=config.level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


def with_log_context(**kwargs) -> ContextualLogger:
    """Create a logger with specific context."""
    logger = get_logger()
    return ContextualLogger(logger, kwargs)


def create_default_config() -> LoggingConfig:
    """Create default logging configuration."""
    return LoggingConfig()


def create_development_config() -> LoggingConfig:
    """Create development logging configuration."""
    return LoggingConfig()


def create_production_config() -> LoggingConfig:
    """Create production logging configuration."""
    return LoggingConfig()


# Legacy compatibility exports
__all__ = [
    "get_logger",
    "setup_logging",
    "configure_logging",
    "LogLevel",
    "LogFormat", 
    "LoggingConfig",
    "ContextualLogger",
    "with_log_context",
    "create_default_config",
    "create_development_config",
    "create_production_config",
]