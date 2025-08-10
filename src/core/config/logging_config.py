"""
Unified Logging Configuration
Author: Drmusab
Last Modified: 2025-01-13

This module provides a unified logging configuration that consolidates all logging
setup for the AI assistant using the YAML-first configuration approach.
"""

import logging
import logging.handlers
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from src.core.config.unified_config import get_unified_config


class UnifiedLoggingConfig:
    """
    Unified logging configuration manager.
    
    This class consolidates all logging configuration into a single, simple system
    that uses the YAML configuration files as the primary source of truth.
    """
    
    def __init__(self):
        self.unified_config = get_unified_config()
        self._loggers: Dict[str, logging.Logger] = {}
        self._configured = False
    
    def get_logger(self, name: Optional[str] = None) -> logging.Logger:
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
        
        if name not in self._loggers:
            self._loggers[name] = self._create_logger(name)
        
        return self._loggers[name]
    
    def _create_logger(self, name: str) -> logging.Logger:
        """Create and configure a logger."""
        logger = logging.getLogger(name)
        
        if not logger.handlers or not self._configured:
            self._configure_logger(logger, name)
        
        return logger
    
    def _configure_logger(self, logger: logging.Logger, name: str) -> None:
        """Configure a logger with settings from YAML configuration."""
        logging_config = self.unified_config.get_logging_config()
        
        # Set log level
        log_level = getattr(logging, logging_config.get("level", "INFO"))
        logger.setLevel(log_level)
        
        # Check for component-specific log levels
        component_levels = logging_config.get("component_levels", {})
        if name in component_levels:
            component_level = getattr(logging, component_levels[name])
            logger.setLevel(component_level)
        
        # Clear existing handlers if reconfiguring
        if logger.handlers:
            logger.handlers.clear()
        
        # Configure handlers based on YAML config
        handlers = logging_config.get("handlers", ["console"])
        
        for handler_name in handlers:
            if handler_name == "console":
                self._add_console_handler(logger, logging_config)
            elif handler_name == "file":
                self._add_file_handler(logger, logging_config)
    
    def _add_console_handler(self, logger: logging.Logger, config: Dict[str, Any]) -> None:
        """Add console handler to logger."""
        handler = logging.StreamHandler(sys.stdout)
        
        # Set formatter based on config
        format_type = config.get("format", "structured")
        if format_type == "json":
            formatter = logging.Formatter(
                '{"timestamp": "%(asctime)s", "level": "%(levelname)s", '
                '"component": "%(name)s", "message": "%(message)s"}'
            )
        elif format_type == "console":
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        else:  # structured
            formatter = logging.Formatter(
                'Time: %(asctime)s | Level: %(levelname)s | Component: %(name)s | Message: %(message)s'
            )
        
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    def _add_file_handler(self, logger: logging.Logger, config: Dict[str, Any]) -> None:
        """Add file handler to logger."""
        # Create logs directory if it doesn't exist
        logs_dir = Path("data/logs")
        logs_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = logs_dir / "assistant.log"
        
        # Use rotating file handler if rotation is enabled
        if config.get("file_rotation", True):
            max_bytes = self._parse_size(config.get("max_file_size", "100MB"))
            backup_count = config.get("backup_count", 5)
            
            handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=max_bytes,
                backupCount=backup_count
            )
        else:
            handler = logging.FileHandler(log_file)
        
        # Set formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    def _parse_size(self, size_str: str) -> int:
        """Parse size string like '100MB' into bytes."""
        size_str = size_str.upper()
        if size_str.endswith('KB'):
            return int(size_str[:-2]) * 1024
        elif size_str.endswith('MB'):
            return int(size_str[:-2]) * 1024 * 1024
        elif size_str.endswith('GB'):
            return int(size_str[:-2]) * 1024 * 1024 * 1024
        else:
            return int(size_str)
    
    def setup_logging(self) -> None:
        """Setup global logging configuration."""
        if self._configured:
            return
        
        # Configure root logger
        root_logger = logging.getLogger()
        self._configure_logger(root_logger, "root")
        
        self._configured = True


# Global logging config instance
_logging_config: Optional[UnifiedLoggingConfig] = None


def get_logging_config() -> UnifiedLoggingConfig:
    """Get the global unified logging configuration."""
    global _logging_config
    
    if _logging_config is None:
        _logging_config = UnifiedLoggingConfig()
    
    return _logging_config


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger instance using unified configuration.
    
    This is the main entry point for getting loggers throughout the application.
    
    Args:
        name: Logger name (defaults to caller's module name)
        
    Returns:
        Configured logger instance
    """
    return get_logging_config().get_logger(name)


def setup_logging() -> None:
    """Setup global logging configuration."""
    get_logging_config().setup_logging()


# For backward compatibility with the old logging system
def create_structured_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Create a structured logger (backward compatibility)."""
    return get_logger(name)


def create_console_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Create a console logger (backward compatibility)."""
    return get_logger(name)


def create_file_logger(name: str, filename: str, level: str = "INFO") -> logging.Logger:
    """Create a file logger (backward compatibility)."""
    return get_logger(name)