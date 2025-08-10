"""
Dependency Injection Configuration (Legacy Compatibility Layer)
Author: Drmusab
Last Modified: 2025-01-13

This module provides backward compatibility for dependency injection configuration.
The actual DI configuration is now handled by the unified YAML-first configuration
system, but this module maintains the original API.
"""

from typing import Any, Dict, Type

import asyncio

# Import the new unified configuration system
from src.core.config.unified_config import UnifiedConfigManager, get_unified_config
from src.core.dependency_injection import Container, LifecycleScope
from src.observability.logging.config import get_logger


class ComponentConfiguration:
    """
    Legacy Component Configuration class for backward compatibility.
    
    This class now delegates all functionality to the unified configuration system
    while maintaining the original API.
    """

    def __init__(self):
        self.logger = get_logger(__name__)
        self._unified_config = get_unified_config()

    async def configure_container(self, container: Container) -> None:
        """Configure the dependency injection container with all components."""
        try:
            # Use the unified configuration system to register components
            await self._unified_config.register_components()
            
            self.logger.info("Dependency injection container configured successfully via unified config")
            
        except Exception as e:
            self.logger.error(f"Failed to configure DI container: {str(e)}")
            raise


async def create_configured_container() -> Container:
    """Create and configure a dependency injection container."""
    # Get the unified configuration system
    unified_config = get_unified_config()
    
    # Initialize the unified configuration system
    await unified_config.initialize()
    
    # Return the configured container
    return unified_config.get_container()


# Mock implementations for missing components (to allow testing)
class MockComponent:
    """Mock component for testing."""

    def __init__(self, container: Container):
        self.container = container
        self.logger = get_logger(self.__class__.__name__)

    async def initialize(self):
        """Initialize mock component."""
        pass

    async def shutdown(self):
        """Shutdown mock component."""
        pass


def create_mock_factories(container: Container) -> None:
    """Create mock factories for components that don't exist yet."""
    
    # Mock missing processing components
    missing_components = [
        "VisionProcessor",
        "ImageAnalyzer", 
        "MultimodalFusionStrategy",
        "EnhancedAudioPipeline",
        "EnhancedWhisperTranscriber",
        "EnhancedTextToSpeech",
        "EnhancedEmotionDetector",
        "EnhancedSpeakerRecognition",
        "TaskPlanner",
        "DecisionTree",
        "ContinualLearner",
        "PreferenceLearner",
        "FeedbackProcessor",
        "ModelAdapter",
    ]

    for component_name in missing_components:
        # Create mock class
        mock_class = type(component_name, (MockComponent,), {})

        # Register with container
        container.register(
            mock_class,
            factory=lambda cls=mock_class: cls(container),
            scope=LifecycleScope.SINGLETON,
        )
