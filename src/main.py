"""
Advanced AI Assistant Application Entry Point
Author: Drmusab
Last Modified: 2025-07-05 11:11:10 UTC

This module serves as the main entry point for the AI assistant system,
initializing all core components, handling startup/shutdown, and providing
the primary application interface.
"""

import argparse
import logging
import os
import signal
import sys
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import asyncio

from src.assistant.core import EnhancedComponentManager

# Assistant components
from src.assistant.core import (
    EngineState,
    CoreAssistantEngine,
    ModalityType,
    MultimodalInput,
    PriorityLevel,
    ProcessingContext,
    ProcessingMode,
    ProcessingResult,
)
from src.assistant.core import (
    InputModality,
    InteractionHandler,
    InteractionMode,
    OutputModality,
    UserMessage,
)
from src.assistant.core import EnhancedPluginManager
from src.assistant.core import EnhancedSessionManager
from src.assistant.core import WorkflowOrchestrator

# Core imports
from src.core.config.loader import ConfigLoader
from src.core.dependency_injection import Container
from src.core.error_handling import ErrorHandler, handle_exceptions
from src.core.events.event_bus import EventBus
from src.core.events.event_types import (
    ComponentHealthChanged,
    ErrorOccurred,
    SystemShutdownCompleted,
    SystemShutdownStarted,
    SystemStarted,
)
from src.core.health_check import HealthCheck
from src.observability.logging.config import get_logger, configure_logging

# Observability
from src.observability.monitoring.metrics import MetricsCollector
try:
    from src.observability.monitoring.tracing import TraceManager
except ImportError:
    TraceManager = None

# Security components (optional based on availability)
try:
    from src.core.security.authentication import AuthenticationManager
    from src.core.security.authorization import AuthorizationManager
    from src.core.security.sanitization import SecuritySanitizer

    SECURITY_AVAILABLE = True
except ImportError:
    SECURITY_AVAILABLE = False

# API components (optional based on availability)
try:
    from src.api.rest import setup_rest_api

    REST_API_AVAILABLE = True
except ImportError:
    REST_API_AVAILABLE = False

try:
    from src.api.websocket import setup_websocket_api

    WEBSOCKET_API_AVAILABLE = True
except ImportError:
    WEBSOCKET_API_AVAILABLE = False

try:
    from src.api.graphql import setup_graphql_api

    GRAPHQL_API_AVAILABLE = True
except ImportError:
    GRAPHQL_API_AVAILABLE = False


class AIAssistant:
    """
    Main AI Assistant application class that coordinates all components and provides
    the primary interface for interacting with the assistant.

    This class is responsible for:
    - Initializing all core components in the correct order
    - Managing the lifecycle of the assistant system
    - Providing APIs for external interaction
    - Handling graceful startup and shutdown procedures
    - Coordinating health checks and monitoring
    - Processing user input and generating responses
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the AI Assistant with optional configuration path.

        Args:
            config_path: Path to configuration file
        """
        # Setup initial logging
        configure_logging()
        self.logger = get_logger(__name__)
        self.logger.info("Initializing AI Assistant")

        # Setup dependency injection
        self.container = Container()

        # Load configuration
        self.config_loader = ConfigLoader(config_path)
        self.container.register(ConfigLoader, lambda: self.config_loader)

        # Base components
        self.event_bus = EventBus()
        self.error_handler = ErrorHandler(self.event_bus)
        self.health_check = HealthCheck()

        # Register core components
        self.container.register(EventBus, lambda: self.event_bus)
        self.container.register(ErrorHandler, lambda: self.error_handler)
        self.container.register(HealthCheck, lambda: self.health_check)

        # Setup observability
        self.metrics = MetricsCollector()
        self.tracer = TraceManager()
        self.container.register(MetricsCollector, lambda: self.metrics)
        self.container.register(TraceManager, lambda: self.tracer)

        # Initialize security components if available
        self._init_security_components()

        # Initialize assistant components (will be set during initialization)
        self.component_manager = None
        self.plugin_manager = None
        self.session_manager = None
        self.workflow_orchestrator = None
        self.core_engine = None
        self.interaction_handler = None

        # State tracking
        self.initialized = False
        self.shutting_down = False
        self.startup_time = None
        self.version = "1.0.0"

        # Register signal handlers
        self._register_signal_handlers()

        # API servers (will be set during initialization)
        self.api_servers = {}

        self.logger.info("AI Assistant instance created")

    def _init_security_components(self):
        """Initialize security components if available."""
        if SECURITY_AVAILABLE:
            self.auth_manager = AuthenticationManager()
            self.authz_manager = AuthorizationManager()
            self.security_sanitizer = SecuritySanitizer()

            self.container.register(AuthenticationManager, lambda: self.auth_manager)
            self.container.register(AuthorizationManager, lambda: self.authz_manager)
            self.container.register(SecuritySanitizer, lambda: self.security_sanitizer)

            self.logger.info("Security components initialized")

    @handle_exceptions
    async def initialize(self):
        """Initialize all assistant components in the correct dependency order."""
        if self.initialized:
            self.logger.warning("AI Assistant already initialized")
            return

        self.logger.info("Beginning AI Assistant initialization")
        start_time = datetime.now(timezone.utc)

        try:
            # 1. Initialize component manager first (manages other components)
            self.component_manager = EnhancedComponentManager(self.container)
            self.container.register(EnhancedComponentManager, lambda: self.component_manager)
            await self.component_manager.discover_components()
            await self.component_manager.initialize_all()

            # 2. Initialize session manager (needed for user sessions)
            self.session_manager = EnhancedSessionManager(self.container)
            self.container.register(EnhancedSessionManager, lambda: self.session_manager)
            await self.session_manager.initialize()

            # 3. Initialize workflow orchestrator (coordinates workflows)
            self.workflow_orchestrator = WorkflowOrchestrator(self.container)
            self.container.register(WorkflowOrchestrator, lambda: self.workflow_orchestrator)
            await self.workflow_orchestrator.initialize()

            # 4. Initialize plugin manager (extends functionality)
            self.plugin_manager = EnhancedPluginManager(self.container)
            self.container.register(EnhancedPluginManager, lambda: self.plugin_manager)
            await self.plugin_manager.initialize()

            # 5. Initialize core engine (processes inputs)
            self.core_engine = CoreAssistantEngine(self.container)
            self.container.register(CoreAssistantEngine, lambda: self.core_engine)
            await self.core_engine.initialize()

            # 6. Initialize interaction handler (manages user interactions)
            self.interaction_handler = InteractionHandler(self.container)
            self.container.register(InteractionHandler, lambda: self.interaction_handler)
            await self.interaction_handler.initialize()

            # 7. Initialize API interfaces based on configuration
            await self._initialize_apis()

            # 8. Start health check monitoring
            self.health_check.register_component("ai_assistant", self._health_check_callback)
            self._start_background_tasks()

            # Mark as initialized
            self.initialized = True
            self.startup_time = datetime.now(timezone.utc)
            initialization_time = (self.startup_time - start_time).total_seconds()

            # Emit system started event
            await self.event_bus.emit(
                SystemStarted(
                    version=self.version,
                    startup_time=initialization_time,
                    components_initialized=len(self.component_manager.list_components()),
                )
            )

            self.logger.info(f"AI Assistant initialization completed in {initialization_time:.2f}s")

        except Exception as e:
            self.logger.error(f"Failed to initialize AI Assistant: {str(e)}")
            self.logger.error(traceback.format_exc())
            # Clean up any initialized components
            await self.shutdown()
            raise

    async def _initialize_apis(self):
        """Initialize API interfaces based on configuration."""
        apis_enabled = self.config_loader.get("api.enabled", False)
        if not apis_enabled:
            self.logger.info("API interfaces disabled")
            return

        # Initialize REST API if available and enabled
        if REST_API_AVAILABLE and self.config_loader.get("api.rest.enabled", False):
            self.api_servers["rest"] = await setup_rest_api(self)
            self.logger.info("REST API initialized")

        # Initialize WebSocket API if available and enabled
        if WEBSOCKET_API_AVAILABLE and self.config_loader.get("api.websocket.enabled", False):
            self.api_servers["websocket"] = await setup_websocket_api(self)
            self.logger.info("WebSocket API initialized")

        # Initialize GraphQL API if available and enabled
        if GRAPHQL_API_AVAILABLE and self.config_loader.get("api.graphql.enabled", False):
            self.api_servers["graphql"] = await setup_graphql_api(self)
            self.logger.info("GraphQL API initialized")

    def _start_background_tasks(self):
        """Start background maintenance tasks."""
        # System health monitoring
        asyncio.create_task(self._health_monitoring_loop())

        # Plugin discovery (if configured for auto-discovery)
        if self.config_loader.get("plugins.auto_discovery_interval", 0) > 0:
            asyncio.create_task(self._plugin_discovery_loop())

    @handle_exceptions
    async def shutdown(self):
        """Gracefully shut down all assistant components in reverse dependency order."""
        if self.shutting_down:
            self.logger.warning("Shutdown already in progress")
            return

        self.shutting_down = True
        self.logger.info("Beginning AI Assistant shutdown")

        # Emit shutdown started event
        await self.event_bus.emit(SystemShutdownStarted())

        # 1. Shut down API servers first
        for api_name, server in self.api_servers.items():
            try:
                if hasattr(server, "shutdown"):
                    await server.shutdown()
                self.logger.info(f"Shut down {api_name} API")
            except Exception as e:
                self.logger.error(f"Error shutting down {api_name} API: {str(e)}")

        # 2. Shut down interaction handler
        if self.interaction_handler:
            try:
                await self.interaction_handler.cleanup()
                self.logger.info("Interaction handler shut down")
            except Exception as e:
                self.logger.error(f"Error shutting down interaction handler: {str(e)}")

        # 3. Shut down core engine
        if self.core_engine:
            try:
                await self.core_engine.shutdown()
                self.logger.info("Core engine shut down")
            except Exception as e:
                self.logger.error(f"Error shutting down core engine: {str(e)}")

        # 4. Shut down plugin manager
        if self.plugin_manager:
            try:
                await self.plugin_manager.shutdown()
                self.logger.info("Plugin manager shut down")
            except Exception as e:
                self.logger.error(f"Error shutting down plugin manager: {str(e)}")

        # 5. Shut down workflow orchestrator
        if self.workflow_orchestrator:
            try:
                await self.workflow_orchestrator.shutdown_all()
                self.logger.info("Workflow orchestrator shut down")
            except Exception as e:
                self.logger.error(f"Error shutting down workflow orchestrator: {str(e)}")

        # 6. Shut down session manager
        if self.session_manager:
            try:
                await self.session_manager.cleanup()
                self.logger.info("Session manager shut down")
            except Exception as e:
                self.logger.error(f"Error shutting down session manager: {str(e)}")

        # 7. Shut down component manager (last, as other components depend on it)
        if self.component_manager:
            try:
                await self.component_manager.shutdown_all()
                self.logger.info("Component manager shut down")
            except Exception as e:
                self.logger.error(f"Error shutting down component manager: {str(e)}")

        # Emit shutdown completed event
        await self.event_bus.emit(SystemShutdownCompleted())

        self.logger.info("AI Assistant shutdown completed")

    def _register_signal_handlers(self):
        """Register handlers for system signals."""
        if sys.platform != "win32":
            # Register SIGTERM and SIGINT handlers
            for sig in (signal.SIGTERM, signal.SIGINT):
                signal.signal(sig, self._handle_shutdown_signal)
        else:
            # Windows-specific handling
            signal.signal(signal.SIGINT, self._handle_shutdown_signal)

    def _handle_shutdown_signal(self, signum, frame):
        """Handle shutdown signals from the OS."""
        if self.shutting_down:
            self.logger.warning("Received additional shutdown signal, forcing exit")
            sys.exit(1)

        self.logger.info(f"Received shutdown signal {signum}, initiating graceful shutdown")
        # Schedule the shutdown coroutine to run in the event loop
        asyncio.create_task(self._shutdown_on_signal())

    async def _shutdown_on_signal(self):
        """Shutdown assistant in response to a signal."""
        try:
            await self.shutdown()
            # Allow time for cleanup to complete
            await asyncio.sleep(2)
        finally:
            # Stop the event loop
            loop = asyncio.get_running_loop()
            loop.stop()

    async def _health_monitoring_loop(self):
        """Background task for system health monitoring."""
        while not self.shutting_down:
            try:
                # Check component health
                health_result = await self.health_check.check_all()

                # Log health status
                healthy_components = sum(
                    1 for c in health_result["components"].values() if c.get("status") == "healthy"
                )
                total_components = len(health_result["components"])

                self.logger.debug(
                    f"Health check: {healthy_components}/{total_components} components healthy, "
                    f"overall status: {health_result['status']}"
                )

                # Update metrics
                if self.metrics:
                    self.metrics.set("system_health_score", health_result.get("health_score", 0))
                    self.metrics.set("healthy_components", healthy_components)

                await asyncio.sleep(60)  # Check every minute

            except Exception as e:
                self.logger.error(f"Error in health monitoring loop: {str(e)}")
                await asyncio.sleep(60)

    async def _plugin_discovery_loop(self):
        """Background task for plugin discovery."""
        interval = self.config_loader.get("plugins.auto_discovery_interval", 3600)

        while not self.shutting_down:
            try:
                await asyncio.sleep(interval)

                if self.plugin_manager:
                    self.logger.info("Running scheduled plugin discovery")
                    discovered = await self.plugin_manager.discover_plugins()
                    self.logger.info(f"Discovered {len(discovered)} new plugins")

            except Exception as e:
                self.logger.error(f"Error in plugin discovery loop: {str(e)}")

    async def _health_check_callback(self) -> Dict[str, Any]:
        """Health check callback for the AI Assistant."""
        try:
            if not self.initialized:
                return {"status": "initializing"}

            if self.shutting_down:
                return {"status": "shutting_down"}

            # Calculate uptime
            uptime = (
                (datetime.now(timezone.utc) - self.startup_time).total_seconds()
                if self.startup_time
                else 0
            )

            # Get component statuses
            component_status = "healthy"
            if self.component_manager:
                cm_status = self.component_manager.get_component_status()
                if cm_status.get("failed_components", 0) > 0:
                    component_status = "degraded"

            return {
                "status": component_status,
                "version": self.version,
                "uptime_seconds": uptime,
                "components_status": component_status,
                "api_servers_running": len(self.api_servers),
            }

        except Exception as e:
            return {"status": "error", "error": str(e)}

    @handle_exceptions
    async def process_text_input(
        self,
        text: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        context_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Process a text input from a user.

        Args:
            text: Text input from user
            session_id: Optional session identifier
            user_id: Optional user identifier
            context_data: Optional additional context data

        Returns:
            Processing result including response text
        """
        if not self.initialized:
            raise RuntimeError("AI Assistant not initialized")

        # Create or get session
        if not session_id:
            session_id = await self.session_manager.create_session(user_id)

        # Create interaction
        interaction_id = await self.interaction_handler.start_interaction(
            user_id=user_id,
            session_id=session_id,
            interaction_mode=InteractionMode.CONVERSATIONAL,
            input_modalities={InputModality.TEXT},
            output_modalities={OutputModality.TEXT},
        )

        # Create message
        message_id = str(uuid.uuid4())
        message = UserMessage(
            message_id=message_id,
            user_id=user_id,
            session_id=session_id,
            interaction_id=interaction_id,
            text=text,
            modality=InputModality.TEXT,
        )

        # Process message
        response = await self.interaction_handler.process_user_message(
            interaction_id=interaction_id, message=message
        )

        # End interaction
        await self.interaction_handler.end_interaction(interaction_id)

        return {
            "response_text": response.text,
            "response_id": response.response_id,
            "session_id": session_id,
            "interaction_id": interaction_id,
            "processing_time": response.processing_time,
            "confidence": response.confidence,
            "suggested_follow_ups": response.suggested_follow_ups,
        }

    @handle_exceptions
    async def process_multimodal_input(
        self,
        input_data: Dict[str, Any],
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        real_time: bool = False,
        streaming: bool = False,
    ) -> Dict[str, Any]:
        """
        Process multimodal input (text, audio, image, etc).

        Args:
            input_data: Dictionary containing multimodal input data
            session_id: Optional session identifier
            user_id: Optional user identifier
            real_time: Enable real-time processing
            streaming: Enable streaming response

        Returns:
            Processing result including multimodal response
        """
        if not self.initialized:
            raise RuntimeError("AI Assistant not initialized")

        # Create or get session
        if not session_id:
            session_id = await self.session_manager.create_session(user_id)

        # Determine input modalities
        input_modalities = set()
        if input_data.get("text"):
            input_modalities.add(InputModality.TEXT)
        if input_data.get("audio"):
            input_modalities.add(InputModality.SPEECH)
        if input_data.get("image"):
            input_modalities.add(InputModality.VISION)

        # Create interaction
        interaction_id = await self.interaction_handler.start_interaction(
            user_id=user_id,
            session_id=session_id,
            interaction_mode=InteractionMode.CONVERSATIONAL,
            input_modalities=input_modalities,
            output_modalities={OutputModality.TEXT, OutputModality.SPEECH},
        )

        # Create message
        message_id = str(uuid.uuid4())
        message = UserMessage(
            message_id=message_id,
            user_id=user_id,
            session_id=session_id,
            interaction_id=interaction_id,
            text=input_data.get("text"),
            audio_data=input_data.get("audio"),
            image_data=input_data.get("image"),
            modality=(
                InputModality.MULTIMODAL
                if len(input_modalities) > 1
                else next(iter(input_modalities))
            ),
        )

        # Process message
        response = await self.interaction_handler.process_user_message(
            interaction_id=interaction_id, message=message, real_time=real_time, streaming=streaming
        )

        # End interaction
        await self.interaction_handler.end_interaction(interaction_id)

        # Build response
        result = {
            "response_text": response.text,
            "response_id": response.response_id,
            "session_id": session_id,
            "interaction_id": interaction_id,
            "processing_time": response.processing_time,
            "confidence": response.confidence,
            "modalities": [m.value for m in response.modalities],
        }

        # Add audio if present
        if response.audio_data is not None:
            result["audio_data"] = response.audio_data

        # Add visual elements if present
        if response.visual_elements:
            result["visual_elements"] = response.visual_elements

        return result

    @handle_exceptions
    async def execute_workflow(
        self,
        workflow_id: str,
        input_data: Dict[str, Any],
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute a predefined workflow.

        Args:
            workflow_id: Workflow identifier
            input_data: Input data for the workflow
            session_id: Optional session identifier
            user_id: Optional user identifier

        Returns:
            Workflow execution result
        """
        if not self.initialized:
            raise RuntimeError("AI Assistant not initialized")

        # Create or get session
        if not session_id:
            session_id = await self.session_manager.create_session(user_id)

        # Execute workflow
        execution_id = await self.workflow_orchestrator.execute_workflow(
            workflow_id=workflow_id, input_data=input_data, session_id=session_id, user_id=user_id
        )

        # Get execution status
        status = await self.workflow_orchestrator.get_execution_status(execution_id)

        return status

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        if not self.initialized:
            return {"status": "not_initialized"}

        try:
            # Get engine status asynchronously
            engine_status = None
            if self.core_engine:
                loop = asyncio.get_event_loop()
                engine_status = asyncio.run_coroutine_threadsafe(
                    self.core_engine.get_engine_status(), loop
                ).result()

            # Get plugin status asynchronously
            plugin_status = None
            if self.plugin_manager:
                loop = asyncio.get_event_loop()
                plugin_status = asyncio.run_coroutine_threadsafe(
                    self.plugin_manager.get_plugin_status(), loop
                ).result()

            return {
                "status": "running" if not self.shutting_down else "shutting_down",
                "initialized": self.initialized,
                "version": self.version,
                "startup_time": self.startup_time.isoformat() if self.startup_time else None,
                "uptime_seconds": (
                    (datetime.now(timezone.utc) - self.startup_time).total_seconds()
                    if self.startup_time
                    else 0
                ),
                "api_servers": list(self.api_servers.keys()),
                "components": {
                    "component_manager": (
                        self.component_manager.get_component_status()
                        if self.component_manager
                        else None
                    ),
                    "core_engine": engine_status,
                    "plugin_manager": plugin_status,
                    "session_manager": (
                        self.session_manager.get_session_statistics()
                        if self.session_manager
                        else None
                    ),
                    "workflow_orchestrator": {
                        "active_executions": (
                            len(self.workflow_orchestrator.get_active_executions())
                            if self.workflow_orchestrator
                            else 0
                        ),
                        "available_workflows": (
                            len(self.workflow_orchestrator.list_workflows())
                            if self.workflow_orchestrator
                            else 0
                        ),
                    },
                    "interaction_handler": {
                        "active_interactions": (
                            len(self.interaction_handler.get_active_interactions())
                            if self.interaction_handler
                            else 0
                        )
                    },
                },
            }

        except Exception as e:
            self.logger.error(f"Error getting system status: {str(e)}")
            return {"status": "error", "error": str(e)}


async def async_main(args=None):
    """Async entry point for the AI Assistant."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Advanced AI Assistant")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--version", action="store_true", help="Show version and exit")

    parsed_args = parser.parse_args(args)

    # Show version if requested
    if parsed_args.version:
        print("AI Assistant v1.0.0")
        return

    # Set debug mode if requested
    if parsed_args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Create and initialize the assistant
    assistant = AIAssistant(config_path=parsed_args.config)

    try:
        await assistant.initialize()

        # Keep the application running until interrupted
        # In a real application, this would be replaced with a server loop
        # or integration with a larger application framework
        while True:
            await asyncio.sleep(1)

    except KeyboardInterrupt:
        print("\nShutdown requested...")
    except Exception as e:
        print(f"Error: {str(e)}")
        traceback.print_exc()
    finally:
        # Ensure assistant is shut down gracefully
        await assistant.shutdown()


def main(args=None):
    """Main entry point for the AI Assistant."""
    try:
        asyncio.run(async_main(args))
    except KeyboardInterrupt:
        print("\nShutdown completed")
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
