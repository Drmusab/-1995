"""
Enhanced Component Manager for AI Assistant System

This module provides centralized management of all system components,
handling initialization, lifecycle management, and health monitoring.
"""

import asyncio
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass
from enum import Enum
import logging

from src.core.dependency_injection import Container
from src.core.events.event_bus import EventBus
from src.core.events.event_types import (
    ComponentStarted,
    ComponentHealthChanged,
    SystemHealthCheck
)


class ComponentStatus(Enum):
    """Component status enumeration."""
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    RUNNING = "running"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class ComponentInfo:
    """Information about a managed component."""
    name: str
    instance: Any
    status: ComponentStatus = ComponentStatus.UNINITIALIZED
    dependencies: List[str] = None
    health_score: float = 1.0
    last_health_check: float = 0.0


class EnhancedComponentManager:
    """
    Enhanced component manager that provides centralized management
    of all system components with dependency resolution and health monitoring.
    """

    def __init__(self, container: Container):
        """Initialize the component manager."""
        self.container = container
        self.components: Dict[str, ComponentInfo] = {}
        self.initialization_order: List[str] = []
        self.event_bus = container.get(EventBus) if container else None
        self.logger = logging.getLogger(__name__)

    async def discover_components(self) -> List[str]:
        """
        Discover available components in the system.
        This is a stub implementation.
        """
        # In a real implementation, this would scan for available components
        discovered = [
            "memory_manager",
            "skill_registry", 
            "processing_pipeline",
            "reasoning_engine",
            "learning_engine"
        ]
        
        self.logger.info(f"Discovered {len(discovered)} components")
        return discovered

    async def initialize_all(self) -> None:
        """Initialize all discovered components."""
        # Stub implementation - in reality would initialize actual components
        for component_name in ["memory_manager", "skill_registry"]:
            self.components[component_name] = ComponentInfo(
                name=component_name,
                instance=MockComponent(component_name),
                status=ComponentStatus.RUNNING
            )
            
            if self.event_bus:
                await self.event_bus.emit(ComponentStarted(component_name=component_name))
                
        self.logger.info(f"Initialized {len(self.components)} components")

    def list_components(self) -> List[str]:
        """List all managed components."""
        return list(self.components.keys())

    def get_component_status(self) -> Dict[str, Any]:
        """Get status of all components."""
        return {
            "total_components": len(self.components),
            "running_components": sum(
                1 for c in self.components.values() 
                if c.status == ComponentStatus.RUNNING
            ),
            "failed_components": sum(
                1 for c in self.components.values() 
                if c.status == ComponentStatus.ERROR
            ),
            "components": {
                name: {
                    "status": info.status.value,
                    "health_score": info.health_score
                } 
                for name, info in self.components.items()
            }
        }

    async def shutdown_all(self) -> None:
        """Shutdown all components."""
        for component_name, component_info in self.components.items():
            try:
                if hasattr(component_info.instance, 'shutdown'):
                    await component_info.instance.shutdown()
                component_info.status = ComponentStatus.STOPPED
                self.logger.info(f"Shut down component: {component_name}")
            except Exception as e:
                self.logger.error(f"Error shutting down {component_name}: {e}")


class MockComponent:
    """Mock component for testing purposes."""
    
    def __init__(self, name: str):
        self.name = name
        self.initialized = False
    
    async def initialize(self):
        """Initialize the mock component."""
        self.initialized = True
        
    async def shutdown(self):
        """Shutdown the mock component."""
        self.initialized = False