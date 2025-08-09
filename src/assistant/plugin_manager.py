"""
Enhanced Plugin Manager for AI Assistant System

This module manages plugins and extensions for the AI assistant system.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set
import uuid
import logging

from src.core.dependency_injection import Container
from src.core.events.event_bus import EventBus
from src.core.events.event_types import (
    PluginLoaded,
    PluginEnabled,
    PluginDisabled
)


class PluginStatus(Enum):
    """Plugin status enumeration."""
    UNLOADED = "unloaded"
    LOADED = "loaded"
    ENABLED = "enabled"
    DISABLED = "disabled"
    ERROR = "error"


@dataclass
class PluginInfo:
    """Information about a plugin."""
    plugin_id: str
    name: str
    version: str = "1.0.0"
    description: str = ""
    author: str = ""
    status: PluginStatus = PluginStatus.UNLOADED
    dependencies: List[str] = field(default_factory=list)
    capabilities: List[str] = field(default_factory=list)
    loaded_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class EnhancedPluginManager:
    """
    Enhanced plugin manager that provides comprehensive plugin lifecycle management
    with dependency resolution and security controls.
    """

    def __init__(self, container: Container):
        """Initialize the plugin manager."""
        self.container = container
        self.plugins: Dict[str, PluginInfo] = {}
        self.enabled_plugins: Set[str] = set()
        self.event_bus = container.get(EventBus) if container else None
        self.logger = logging.getLogger(__name__)

    async def initialize(self) -> None:
        """Initialize the plugin manager."""
        self.logger.info("Initializing Enhanced Plugin Manager")
        
        # Discover and load built-in plugins
        await self.discover_plugins()
        
        self.logger.info("Enhanced Plugin Manager initialized successfully")

    async def discover_plugins(self) -> List[str]:
        """
        Discover available plugins.
        
        Returns:
            List of discovered plugin IDs
        """
        # Stub implementation - would normally scan plugin directories
        discovered_plugins = [
            "core_skills_plugin",
            "nlp_processor_plugin", 
            "memory_enhancer_plugin",
            "api_extensions_plugin"
        ]
        
        for plugin_id in discovered_plugins:
            if plugin_id not in self.plugins:
                plugin_info = PluginInfo(
                    plugin_id=plugin_id,
                    name=plugin_id.replace("_", " ").title(),
                    description=f"Built-in {plugin_id} functionality",
                    author="AI Assistant System",
                    capabilities=["core_functionality"]
                )
                self.plugins[plugin_id] = plugin_info
        
        self.logger.info(f"Discovered {len(discovered_plugins)} plugins")
        return discovered_plugins

    async def load_plugin(self, plugin_id: str) -> bool:
        """
        Load a plugin.
        
        Args:
            plugin_id: Plugin identifier
            
        Returns:
            True if successful, False otherwise
        """
        plugin = self.plugins.get(plugin_id)
        if not plugin:
            self.logger.error(f"Plugin {plugin_id} not found")
            return False
        
        if plugin.status != PluginStatus.UNLOADED:
            self.logger.warning(f"Plugin {plugin_id} already loaded")
            return True
        
        try:
            # Check dependencies
            if not await self._check_dependencies(plugin_id):
                return False
            
            # Load plugin (stub implementation)
            await self._load_plugin_implementation(plugin_id)
            
            plugin.status = PluginStatus.LOADED
            plugin.loaded_at = datetime.now(timezone.utc)
            
            if self.event_bus:
                await self.event_bus.emit(
                    PluginLoaded(
                        plugin_id=plugin_id,
                        plugin_name=plugin.name,
                        version=plugin.version
                    )
                )
            
            self.logger.info(f"Loaded plugin {plugin_id}")
            return True
            
        except Exception as e:
            plugin.status = PluginStatus.ERROR
            self.logger.error(f"Failed to load plugin {plugin_id}: {e}")
            return False

    async def enable_plugin(self, plugin_id: str) -> bool:
        """
        Enable a plugin.
        
        Args:
            plugin_id: Plugin identifier
            
        Returns:
            True if successful, False otherwise
        """
        plugin = self.plugins.get(plugin_id)
        if not plugin:
            return False
        
        if plugin.status != PluginStatus.LOADED:
            # Try to load first
            if not await self.load_plugin(plugin_id):
                return False
        
        try:
            plugin.status = PluginStatus.ENABLED
            self.enabled_plugins.add(plugin_id)
            
            if self.event_bus:
                await self.event_bus.emit(
                    PluginEnabled(
                        plugin_id=plugin_id,
                        plugin_name=plugin.name
                    )
                )
            
            self.logger.info(f"Enabled plugin {plugin_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to enable plugin {plugin_id}: {e}")
            return False

    async def disable_plugin(self, plugin_id: str) -> bool:
        """
        Disable a plugin.
        
        Args:
            plugin_id: Plugin identifier
            
        Returns:
            True if successful, False otherwise
        """
        plugin = self.plugins.get(plugin_id)
        if not plugin:
            return False
        
        if plugin.status != PluginStatus.ENABLED:
            return True
        
        try:
            plugin.status = PluginStatus.DISABLED
            self.enabled_plugins.discard(plugin_id)
            
            if self.event_bus:
                await self.event_bus.emit(
                    PluginDisabled(
                        plugin_id=plugin_id,
                        plugin_name=plugin.name
                    )
                )
            
            self.logger.info(f"Disabled plugin {plugin_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to disable plugin {plugin_id}: {e}")
            return False

    async def _check_dependencies(self, plugin_id: str) -> bool:
        """Check if plugin dependencies are satisfied."""
        plugin = self.plugins.get(plugin_id)
        if not plugin:
            return False
        
        for dep_id in plugin.dependencies:
            dep_plugin = self.plugins.get(dep_id)
            if not dep_plugin or dep_plugin.status not in [PluginStatus.LOADED, PluginStatus.ENABLED]:
                self.logger.error(f"Plugin {plugin_id} dependency {dep_id} not satisfied")
                return False
        
        return True

    async def _load_plugin_implementation(self, plugin_id: str) -> None:
        """Load the actual plugin implementation (stub)."""
        # This would normally load the plugin code, validate it, etc.
        await asyncio.sleep(0.1)  # Simulate loading time

    def list_plugins(self) -> List[PluginInfo]:
        """List all plugins."""
        return list(self.plugins.values())

    def get_enabled_plugins(self) -> List[str]:
        """Get list of enabled plugin IDs."""
        return list(self.enabled_plugins)

    async def get_plugin_status(self) -> Dict[str, Any]:
        """Get plugin system status."""
        return {
            "total_plugins": len(self.plugins),
            "loaded_plugins": len([p for p in self.plugins.values() if p.status == PluginStatus.LOADED]),
            "enabled_plugins": len(self.enabled_plugins),
            "failed_plugins": len([p for p in self.plugins.values() if p.status == PluginStatus.ERROR]),
            "plugins": {
                plugin_id: {
                    "name": plugin.name,
                    "status": plugin.status.value,
                    "version": plugin.version
                }
                for plugin_id, plugin in self.plugins.items()
            }
        }

    async def shutdown(self) -> None:
        """Shutdown the plugin manager."""
        self.logger.info("Shutting down Enhanced Plugin Manager")
        
        # Disable all enabled plugins
        enabled_plugins = list(self.enabled_plugins)
        for plugin_id in enabled_plugins:
            await self.disable_plugin(plugin_id)
        
        self.plugins.clear()
        self.enabled_plugins.clear()
        
        self.logger.info("Enhanced Plugin Manager shutdown complete")