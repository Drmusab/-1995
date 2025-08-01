"""
Advanced GraphQL Resolvers for AI Assistant
Author: Drmusab
Last Modified: 2025-06-20 03:36:58 UTC

This module provides comprehensive GraphQL resolvers for the AI assistant,
integrating with all core system components including the core engine,
component manager, workflow orchestrator, session manager, interaction handler,
and plugin manager.
"""

import json
import logging
import traceback
import uuid
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional, Set, TypeVar, Union

import asyncio
import strawberry
from strawberry.permission import BasePermission
from strawberry.scalars import JSON
from strawberry.types import Info

# GraphQL schema types
from src.api.graphql.schema import (  # Core types; Enum types from schema; Schema classes
    AssistantResponseSchema,
    ComponentInfo,
    ComponentInfoSchema,
    ComponentMetadataSchema,
    ComponentPriorityEnum,
    ComponentStateEnum,
    ErrorResponse,
    ErrorResponseSchema,
    EventSchema,
    ExecutionModeEnum,
    HealthCheck,
    HealthCheckSchema,
    HealthStatusEnum,
    InteractionContextSchema,
    InteractionModeEnum,
    InteractionStateEnum,
    ModalityEnum,
    MultimodalInput,
    PluginInfo,
    PluginInfoSchema,
    PluginMetadataSchema,
    PluginStateEnum,
    PluginTypeEnum,
    PriorityEnum,
    ProcessingContextSchema,
    ProcessingModeEnum,
    ProcessingResult,
    ProcessingResultSchema,
    SecurityLevelEnum,
    SessionConfigurationSchema,
    SessionContextSchema,
    SessionInfo,
    SessionInfoSchema,
    SessionStateEnum,
    SessionTypeEnum,
    StatusEnum,
    StepStateEnum,
    SuccessResponse,
    SuccessResponseSchema,
    SystemStatusSchema,
    UserMessageSchema,
    UserProfileSchema,
    WorkflowDefinitionSchema,
    WorkflowExecution,
    WorkflowExecutionSchema,
    WorkflowStateEnum,
)
from src.assistant.component_manager import (
    ComponentPriority,
    ComponentState,
    EnhancedComponentManager,
)

# Assistant components
from src.assistant.core_engine import (
    EngineState,
    EnhancedCoreEngine,
)
from src.assistant.core_engine import MultimodalInput as CoreMultimodalInput
from src.assistant.core_engine import (
    PriorityLevel,
    ProcessingContext,
    ProcessingMode,
)
from src.assistant.core_engine import ProcessingResult as CoreProcessingResult
from src.assistant.interaction_handler import (
    InputModality,
    InteractionHandler,
    InteractionMode,
    InteractionState,
    OutputModality,
)
from src.assistant.plugin_manager import (
    EnhancedPluginManager,
    PluginState,
    PluginType,
    SecurityLevel,
)
from src.assistant.session_manager import (
    EnhancedSessionManager,
    SessionConfiguration,
    SessionPriority,
    SessionState,
    SessionType,
)
from src.assistant.workflow_orchestrator import (
    ExecutionMode,
    WorkflowOrchestrator,
    WorkflowPriority,
    WorkflowState,
)

# Core imports
from src.core.config.loader import ConfigLoader
from src.core.dependency_injection import Container
from src.core.error_handling import ErrorHandler, handle_exceptions
from src.core.events.event_bus import EventBus
from src.core.events.event_types import (
    GraphQLError,
    GraphQLMutationExecuted,
    GraphQLQueryExecuted,
    GraphQLSubscriptionStarted,
    SystemHealthChecked,
    UserAuthenticated,
    UserAuthorized,
)
from src.core.health_check import HealthCheck
from src.core.security.authentication import AuthenticationManager
from src.core.security.authorization import AuthorizationManager

# Learning and adaptation
from src.learning.continual_learning import ContinualLearner
from src.learning.feedback_processor import FeedbackProcessor
from src.learning.preference_learning import PreferenceLearner
from src.memory.core_memory.memory_manager import MemoryManager
from src.memory.operations.context_manager import ContextManager
from src.observability.logging.config import get_logger

# Observability
from src.observability.monitoring.metrics import MetricsCollector
from src.observability.monitoring.tracing import TraceManager
from src.skills.skill_factory import SkillFactory

# Skills and memory
from src.skills.skill_registry import SkillRegistry

# Type definitions
T = TypeVar("T")


class AuthenticationRequired(BasePermission):
    """Permission class requiring user authentication."""

    message = "Authentication required"

    def has_permission(self, source: Any, info: Info, **kwargs) -> bool:
        """Check if user is authenticated."""
        try:
            context = info.context
            if hasattr(context, "user_id") and context.user_id:
                return True

            # Check authorization header
            auth_header = context.request.headers.get("authorization")
            if auth_header and auth_header.startswith("Bearer "):
                token = auth_header[7:]
                # Validate token (implementation depends on auth system)
                return self._validate_token(token, context)

            return False
        except Exception:
            return False

    def _validate_token(self, token: str, context) -> bool:
        """Validate authentication token."""
        try:
            # Get auth manager from context
            auth_manager = context.auth_manager
            if auth_manager:
                user_info = auth_manager.validate_token(token)
                if user_info:
                    context.user_id = user_info.get("user_id")
                    context.user_info = user_info
                    return True
            return False
        except Exception:
            return False


class AdminRequired(BasePermission):
    """Permission class requiring admin privileges."""

    message = "Administrator privileges required"

    def has_permission(self, source: Any, info: Info, **kwargs) -> bool:
        """Check if user has admin privileges."""
        try:
            context = info.context
            if hasattr(context, "user_info") and context.user_info:
                return context.user_info.get("role") == "admin"
            return False
        except Exception:
            return False


@strawberry.input
class MultimodalInputInput:
    """Input type for multimodal processing requests."""

    text: Optional[str] = None
    modality_weights: Optional[JSON] = None
    processing_hints: Optional[JSON] = None
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    priority: Optional[str] = "normal"
    timeout_seconds: Optional[float] = 30.0


@strawberry.input
class WorkflowExecutionInput:
    """Input type for workflow execution requests."""

    workflow_id: str
    session_id: str
    input_data: JSON
    user_id: Optional[str] = None
    priority: Optional[str] = "normal"
    context: Optional[JSON] = None


@strawberry.input
class SessionCreationInput:
    """Input type for session creation requests."""

    user_id: Optional[str] = None
    session_type: Optional[str] = "interactive"
    priority: Optional[str] = "normal"
    max_idle_time: Optional[float] = 1800.0
    max_session_time: Optional[float] = 86400.0
    persist_context: Optional[bool] = True
    context_data: Optional[JSON] = None


@strawberry.input
class InteractionInput:
    """Input type for user interactions."""

    interaction_id: Optional[str] = None
    session_id: str
    message_text: Optional[str] = None
    modality: Optional[str] = "text"
    real_time: Optional[bool] = False
    streaming: Optional[bool] = False
    context: Optional[JSON] = None


@strawberry.input
class FeedbackInput:
    """Input type for user feedback."""

    interaction_id: str
    feedback_type: str
    feedback_data: JSON
    user_id: Optional[str] = None
    rating: Optional[int] = None
    comments: Optional[str] = None


class GraphQLContext:
    """Context object for GraphQL requests."""

    def __init__(self, request, container: Container):
        self.request = request
        self.container = container
        self.user_id: Optional[str] = None
        self.user_info: Optional[Dict[str, Any]] = None

        # Core components
        self.core_engine = container.get(EnhancedCoreEngine)
        self.component_manager = container.get(EnhancedComponentManager)
        self.workflow_orchestrator = container.get(WorkflowOrchestrator)
        self.session_manager = container.get(EnhancedSessionManager)
        self.interaction_handler = container.get(InteractionHandler)
        self.plugin_manager = container.get(EnhancedPluginManager)

        # Supporting components
        self.skill_registry = container.get(SkillRegistry)
        self.skill_factory = container.get(SkillFactory)
        self.memory_manager = container.get(MemoryManager)
        self.context_manager = container.get(ContextManager)

        # Learning systems
        self.continual_learner = container.get(ContinualLearner)
        self.preference_learner = container.get(PreferenceLearner)
        self.feedback_processor = container.get(FeedbackProcessor)

        # Core services
        self.config = container.get(ConfigLoader)
        self.event_bus = container.get(EventBus)
        self.error_handler = container.get(ErrorHandler)
        self.health_check = container.get(HealthCheck)
        self.auth_manager = container.get(AuthenticationManager)
        self.authz_manager = container.get(AuthorizationManager)

        # Observability
        self.metrics = container.get(MetricsCollector)
        self.tracer = container.get(TraceManager)
        self.logger = get_logger(__name__)


@strawberry.type
class Query:
    """GraphQL Query operations."""

    @strawberry.field
    async def system_status(self, info: Info) -> HealthCheck:
        """Get overall system health status."""
        try:
            context: GraphQLContext = info.context

            # Emit health check event
            await context.event_bus.emit(
                SystemHealthChecked(
                    check_type="graphql_query", requester=context.user_id or "anonymous"
                )
            )

            # Get system health
            health_status = await context.health_check.get_system_health()

            return HealthCheck(
                status=health_status.get("status", "unknown"),
                uptime_seconds=health_status.get("uptime_seconds", 0.0),
                version=health_status.get("version", "1.0.0"),
                checks=health_status.get("checks", {}),
                metrics=health_status.get("metrics", {}),
            )

        except Exception as e:
            context.logger.error(f"Error getting system status: {str(e)}")
            return HealthCheck(
                status="unhealthy",
                uptime_seconds=0.0,
                version="1.0.0",
                checks={"error": str(e)},
                metrics={},
            )

    @strawberry.field
    async def engine_status(self, info: Info) -> Dict[str, Any]:
        """Get core engine status."""
        try:
            context: GraphQLContext = info.context
            status = await context.core_engine.get_engine_status()
            return status
        except Exception as e:
            context.logger.error(f"Error getting engine status: {str(e)}")
            return {"error": str(e)}

    @strawberry.field
    async def components(self, info: Info, state: Optional[str] = None) -> List[ComponentInfo]:
        """List system components with optional state filtering."""
        try:
            context: GraphQLContext = info.context

            # Filter by state if provided
            state_filter = None
            if state:
                try:
                    state_filter = ComponentState(state)
                except ValueError:
                    pass

            components = context.component_manager.list_components(state_filter)

            result = []
            for component_id in components:
                component_status = context.component_manager.get_component_status(component_id)
                if component_status:
                    result.append(
                        ComponentInfo(
                            component_id=component_id,
                            state=component_status.get("state", "unknown"),
                            health_status=component_status.get("health_status", "unknown"),
                            error_count=component_status.get("error_count", 0),
                            performance_metrics=component_status.get("performance_metrics", {}),
                        )
                    )

            return result

        except Exception as e:
            context.logger.error(f"Error listing components: {str(e)}")
            return []

    @strawberry.field
    async def workflows(self, info: Info) -> List[Dict[str, Any]]:
        """List available workflows."""
        try:
            context: GraphQLContext = info.context
            workflows = context.workflow_orchestrator.list_workflows()
            return workflows
        except Exception as e:
            context.logger.error(f"Error listing workflows: {str(e)}")
            return []

    @strawberry.field
    async def active_workflows(self, info: Info) -> List[WorkflowExecution]:
        """List currently executing workflows."""
        try:
            context: GraphQLContext = info.context
            executions = context.workflow_orchestrator.get_active_executions()

            result = []
            for execution in executions:
                result.append(
                    WorkflowExecution(
                        execution_id=execution["execution_id"],
                        workflow_id=execution["workflow_id"],
                        session_id=execution["session_id"],
                        state=execution["state"],
                        execution_time=execution["execution_time"],
                        completed_steps=execution["completed_steps"],
                        current_steps=execution["current_steps"],
                    )
                )

            return result

        except Exception as e:
            context.logger.error(f"Error listing active workflows: {str(e)}")
            return []

    @strawberry.field
    async def workflow_status(self, info: Info, execution_id: str) -> Dict[str, Any]:
        """Get status of a specific workflow execution."""
        try:
            context: GraphQLContext = info.context
            status = await context.workflow_orchestrator.get_execution_status(execution_id)
            return status
        except Exception as e:
            context.logger.error(f"Error getting workflow status: {str(e)}")
            return {"error": str(e)}

    @strawberry.field
    async def sessions(self, info: Info, user_id: Optional[str] = None) -> List[SessionInfo]:
        """List sessions, optionally filtered by user."""
        try:
            context: GraphQLContext = info.context

            # Use current user if no user_id provided
            if not user_id and context.user_id:
                user_id = context.user_id

            sessions = context.session_manager.get_active_sessions()

            result = []
            for session in sessions:
                if not user_id or session.get("user_id") == user_id:
                    result.append(
                        SessionInfo(
                            session_id=session["session_id"],
                            state=session["state"],
                            interaction_count=session["interaction_count"],
                            memory_usage_mb=session["memory_usage_mb"],
                            health_score=session.get("health_score", 1.0),
                            last_activity=datetime.fromisoformat(session["last_activity"]),
                        )
                    )

            return result

        except Exception as e:
            context.logger.error(f"Error listing sessions: {str(e)}")
            return []

    @strawberry.field
    async def session_statistics(self, info: Info) -> Dict[str, Any]:
        """Get comprehensive session statistics."""
        try:
            context: GraphQLContext = info.context
            stats = context.session_manager.get_session_statistics()
            return stats
        except Exception as e:
            context.logger.error(f"Error getting session statistics: {str(e)}")
            return {"error": str(e)}

    @strawberry.field
    async def active_interactions(self, info: Info) -> List[Dict[str, Any]]:
        """List currently active interactions."""
        try:
            context: GraphQLContext = info.context
            interactions = context.interaction_handler.get_active_interactions()
            return interactions
        except Exception as e:
            context.logger.error(f"Error listing active interactions: {str(e)}")
            return []

    @strawberry.field
    async def plugins(
        self, info: Info, state: Optional[str] = None, plugin_type: Optional[str] = None
    ) -> List[PluginInfo]:
        """List plugins with optional filtering."""
        try:
            context: GraphQLContext = info.context

            # Convert string filters to enums
            state_filter = None
            type_filter = None

            if state:
                try:
                    state_filter = PluginState(state)
                except ValueError:
                    pass

            if plugin_type:
                try:
                    type_filter = PluginType(plugin_type)
                except ValueError:
                    pass

            plugins = context.plugin_manager.list_plugins(state_filter, type_filter)

            result = []
            for plugin in plugins:
                result.append(
                    PluginInfo(
                        plugin_id=plugin["plugin_id"],
                        name=plugin["name"],
                        version=plugin["version"],
                        state=plugin["state"],
                        plugin_type=plugin["type"],
                        load_time=plugin["load_time"],
                        memory_usage=plugin["memory_usage"],
                    )
                )

            return result

        except Exception as e:
            context.logger.error(f"Error listing plugins: {str(e)}")
            return []

    @strawberry.field
    async def plugin_capabilities(self, info: Info) -> Dict[str, List[str]]:
        """Get available plugin capabilities."""
        try:
            context: GraphQLContext = info.context
            capabilities = context.plugin_manager.get_plugin_capabilities()
            return capabilities
        except Exception as e:
            context.logger.error(f"Error getting plugin capabilities: {str(e)}")
            return {}

    @strawberry.field
    async def skills(self, info: Info) -> List[Dict[str, Any]]:
        """List available skills."""
        try:
            context: GraphQLContext = info.context
            skills = await context.skill_registry.list_skills()
            return skills
        except Exception as e:
            context.logger.error(f"Error listing skills: {str(e)}")
            return []

    @strawberry.field
    async def memory_stats(self, info: Info, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Get memory system statistics."""
        try:
            context: GraphQLContext = info.context
            stats = await context.memory_manager.get_memory_statistics(session_id)
            return stats
        except Exception as e:
            context.logger.error(f"Error getting memory stats: {str(e)}")
            return {"error": str(e)}


@strawberry.type
class Mutation:
    """GraphQL Mutation operations."""

    @strawberry.mutation
    async def process_multimodal_input(
        self, info: Info, input_data: MultimodalInputInput
    ) -> ProcessingResult:
        """Process multimodal input through the core engine."""
        try:
            context: GraphQLContext = info.context

            # Emit query event
            await context.event_bus.emit(
                GraphQLMutationExecuted(
                    mutation_name="process_multimodal_input",
                    user_id=context.user_id or "anonymous",
                    input_data=asdict(input_data),
                )
            )

            # Create core multimodal input
            core_input = CoreMultimodalInput(
                text=input_data.text,
                modality_weights=input_data.modality_weights or {},
                processing_hints=input_data.processing_hints or {},
            )

            # Create processing context
            processing_context = ProcessingContext(
                session_id=input_data.session_id or str(uuid.uuid4()),
                user_id=input_data.user_id or context.user_id,
                priority=getattr(PriorityLevel, input_data.priority.upper(), PriorityLevel.NORMAL),
                timeout_seconds=input_data.timeout_seconds,
            )

            # Process through core engine
            result = await context.core_engine.process_multimodal_input(
                core_input, processing_context
            )

            # Convert result to GraphQL type
            return ProcessingResult(
                success=result.success,
                request_id=result.request_id,
                session_id=result.session_id,
                processing_time=result.processing_time,
                response_text=result.response_text,
                overall_confidence=result.overall_confidence,
                errors=result.errors,
                component_timings=result.component_timings,
            )

        except Exception as e:
            context.logger.error(f"Error processing multimodal input: {str(e)}")

            await context.event_bus.emit(
                GraphQLError(
                    operation_type="mutation",
                    operation_name="process_multimodal_input",
                    error_message=str(e),
                    user_id=context.user_id or "anonymous",
                )
            )

            return ProcessingResult(
                success=False,
                request_id=str(uuid.uuid4()),
                session_id=input_data.session_id or str(uuid.uuid4()),
                processing_time=0.0,
                response_text=None,
                overall_confidence=0.0,
                errors=[str(e)],
                component_timings={},
            )

    @strawberry.mutation
    async def create_session(self, info: Info, input_data: SessionCreationInput) -> SessionInfo:
        """Create a new session."""
        try:
            context: GraphQLContext = info.context

            # Create session configuration
            config = SessionConfiguration(
                session_type=getattr(
                    SessionType, input_data.session_type.upper(), SessionType.INTERACTIVE
                ),
                priority=getattr(
                    SessionPriority, input_data.priority.upper(), SessionPriority.NORMAL
                ),
                max_idle_time=input_data.max_idle_time,
                max_session_time=input_data.max_session_time,
                persist_context=input_data.persist_context,
            )

            # Create session
            session_id = await context.session_manager.create_session(
                user_id=input_data.user_id or context.user_id,
                session_config=config,
                context_data=input_data.context_data,
            )

            # Get session info
            session_info = await context.session_manager.get_session(session_id)

            if session_info:
                return SessionInfo(
                    session_id=session_id,
                    state=session_info.state.value,
                    interaction_count=session_info.interaction_count,
                    memory_usage_mb=session_info.memory_usage_mb,
                    health_score=session_info.health_score,
                    last_activity=session_info.last_activity,
                )
            else:
                raise Exception("Failed to retrieve created session")

        except Exception as e:
            context.logger.error(f"Error creating session: {str(e)}")
            raise

    @strawberry.mutation
    async def end_session(
        self, info: Info, session_id: str, reason: Optional[str] = "user_ended"
    ) -> SuccessResponse:
        """End a session."""
        try:
            context: GraphQLContext = info.context

            await context.session_manager.end_session(session_id, reason)

            return SuccessResponse(
                success=True,
                message=f"Session {session_id} ended successfully",
                timestamp=datetime.now(timezone.utc),
            )

        except Exception as e:
            context.logger.error(f"Error ending session: {str(e)}")
            return SuccessResponse(
                success=False,
                message=f"Failed to end session: {str(e)}",
                timestamp=datetime.now(timezone.utc),
            )

    @strawberry.mutation
    async def execute_workflow(
        self, info: Info, input_data: WorkflowExecutionInput
    ) -> WorkflowExecution:
        """Execute a workflow."""
        try:
            context: GraphQLContext = info.context

            # Convert priority
            priority = getattr(
                WorkflowPriority, input_data.priority.upper(), WorkflowPriority.NORMAL
            )

            # Execute workflow
            execution_id = await context.workflow_orchestrator.execute_workflow(
                workflow_id=input_data.workflow_id,
                session_id=input_data.session_id,
                input_data=input_data.input_data,
                user_id=input_data.user_id or context.user_id,
                priority=priority,
                context=input_data.context,
            )

            # Get execution status
            status = await context.workflow_orchestrator.get_execution_status(execution_id)

            return WorkflowExecution(
                execution_id=execution_id,
                workflow_id=input_data.workflow_id,
                session_id=input_data.session_id,
                state=status["state"],
                execution_time=status["execution_time"],
                completed_steps=status["completed_steps"],
                current_steps=status["current_steps"],
            )

        except Exception as e:
            context.logger.error(f"Error executing workflow: {str(e)}")
            raise

    @strawberry.mutation
    async def cancel_workflow(self, info: Info, execution_id: str) -> SuccessResponse:
        """Cancel a running workflow."""
        try:
            context: GraphQLContext = info.context

            await context.workflow_orchestrator.cancel_execution(execution_id)

            return SuccessResponse(
                success=True,
                message=f"Workflow execution {execution_id} cancelled",
                timestamp=datetime.now(timezone.utc),
            )

        except Exception as e:
            context.logger.error(f"Error cancelling workflow: {str(e)}")
            return SuccessResponse(
                success=False,
                message=f"Failed to cancel workflow: {str(e)}",
                timestamp=datetime.now(timezone.utc),
            )

    @strawberry.mutation
    async def start_interaction(self, info: Info, input_data: InteractionInput) -> Dict[str, Any]:
        """Start a new user interaction."""
        try:
            context: GraphQLContext = info.context

            # Convert modality
            input_modality = getattr(InputModality, input_data.modality.upper(), InputModality.TEXT)

            # Start interaction
            interaction_id = await context.interaction_handler.start_interaction(
                user_id=context.user_id,
                session_id=input_data.session_id,
                interaction_mode=InteractionMode.CONVERSATIONAL,
                input_modalities={input_modality},
                output_modalities={OutputModality.TEXT},
            )

            return {
                "interaction_id": interaction_id,
                "session_id": input_data.session_id,
                "status": "started",
            }

        except Exception as e:
            context.logger.error(f"Error starting interaction: {str(e)}")
            raise

    @strawberry.mutation
    async def submit_feedback(self, info: Info, input_data: FeedbackInput) -> SuccessResponse:
        """Submit user feedback for an interaction."""
        try:
            context: GraphQLContext = info.context

            await context.interaction_handler.handle_user_feedback(
                interaction_id=input_data.interaction_id,
                feedback_type=input_data.feedback_type,
                feedback_data=input_data.feedback_data,
                user_id=input_data.user_id or context.user_id,
            )

            return SuccessResponse(
                success=True,
                message="Feedback submitted successfully",
                timestamp=datetime.now(timezone.utc),
            )

        except Exception as e:
            context.logger.error(f"Error submitting feedback: {str(e)}")
            return SuccessResponse(
                success=False,
                message=f"Failed to submit feedback: {str(e)}",
                timestamp=datetime.now(timezone.utc),
            )

    @strawberry.mutation(permission_classes=[AdminRequired])
    async def load_plugin(self, info: Info, plugin_id: str) -> SuccessResponse:
        """Load a plugin (admin only)."""
        try:
            context: GraphQLContext = info.context

            await context.plugin_manager.load_plugin(plugin_id)

            return SuccessResponse(
                success=True,
                message=f"Plugin {plugin_id} loaded successfully",
                timestamp=datetime.now(timezone.utc),
            )

        except Exception as e:
            context.logger.error(f"Error loading plugin: {str(e)}")
            return SuccessResponse(
                success=False,
                message=f"Failed to load plugin: {str(e)}",
                timestamp=datetime.now(timezone.utc),
            )

    @strawberry.mutation(permission_classes=[AdminRequired])
    async def unload_plugin(self, info: Info, plugin_id: str) -> SuccessResponse:
        """Unload a plugin (admin only)."""
        try:
            context: GraphQLContext = info.context

            await context.plugin_manager.unload_plugin(plugin_id)

            return SuccessResponse(
                success=True,
                message=f"Plugin {plugin_id} unloaded successfully",
                timestamp=datetime.now(timezone.utc),
            )

        except Exception as e:
            context.logger.error(f"Error unloading plugin: {str(e)}")
            return SuccessResponse(
                success=False,
                message=f"Failed to unload plugin: {str(e)}",
                timestamp=datetime.now(timezone.utc),
            )


@strawberry.type
class Subscription:
    """GraphQL Subscription operations for real-time updates."""

    @strawberry.subscription
    async def system_health_updates(self, info: Info) -> AsyncGenerator[HealthCheck, None]:
        """Subscribe to system health updates."""
        context: GraphQLContext = info.context

        # Emit subscription started event
        await context.event_bus.emit(
            GraphQLSubscriptionStarted(
                subscription_name="system_health_updates", user_id=context.user_id or "anonymous"
            )
        )

        try:
            # Subscribe to health check events
            async for event in context.event_bus.subscribe("system_health_changed"):
                health_data = event.data if hasattr(event, "data") else {}

                yield HealthCheck(
                    status=health_data.get("status", "unknown"),
                    uptime_seconds=health_data.get("uptime_seconds", 0.0),
                    version=health_data.get("version", "1.0.0"),
                    checks=health_data.get("checks", {}),
                    metrics=health_data.get("metrics", {}),
                )

        except Exception as e:
            context.logger.error(f"Error in system health subscription: {str(e)}")

    @strawberry.subscription
    async def workflow_updates(
        self, info: Info, execution_id: Optional[str] = None
    ) -> AsyncGenerator[WorkflowExecution, None]:
        """Subscribe to workflow execution updates."""
        context: GraphQLContext = info.context

        try:
            # Subscribe to workflow events
            async for event in context.event_bus.subscribe("workflow_*"):
                if (
                    execution_id
                    and hasattr(event, "execution_id")
                    and event.execution_id != execution_id
                ):
                    continue

                if hasattr(event, "execution_id"):
                    try:
                        status = await context.workflow_orchestrator.get_execution_status(
                            event.execution_id
                        )

                        yield WorkflowExecution(
                            execution_id=event.execution_id,
                            workflow_id=status["workflow_id"],
                            session_id=status["session_id"],
                            state=status["state"],
                            execution_time=status["execution_time"],
                            completed_steps=status["completed_steps"],
                            current_steps=status["current_steps"],
                        )
                    except Exception:
                        continue  # Skip if can't get status

        except Exception as e:
            context.logger.error(f"Error in workflow subscription: {str(e)}")

    @strawberry.subscription
    async def session_updates(
        self, info: Info, session_id: Optional[str] = None
    ) -> AsyncGenerator[SessionInfo, None]:
        """Subscribe to session updates."""
        context: GraphQLContext = info.context

        try:
            # Subscribe to session events
            async for event in context.event_bus.subscribe("session_*"):
                if session_id and hasattr(event, "session_id") and event.session_id != session_id:
                    continue

                if hasattr(event, "session_id"):
                    try:
                        sessions = context.session_manager.get_active_sessions()
                        session_data = next(
                            (s for s in sessions if s["session_id"] == event.session_id), None
                        )

                        if session_data:
                            yield SessionInfo(
                                session_id=session_data["session_id"],
                                state=session_data["state"],
                                interaction_count=session_data["interaction_count"],
                                memory_usage_mb=session_data["memory_usage_mb"],
                                health_score=session_data.get("health_score", 1.0),
                                last_activity=datetime.fromisoformat(session_data["last_activity"]),
                            )
                    except Exception:
                        continue  # Skip if can't get session data

        except Exception as e:
            context.logger.error(f"Error in session subscription: {str(e)}")

    @strawberry.subscription
    async def component_updates(self, info: Info) -> AsyncGenerator[ComponentInfo, None]:
        """Subscribe to component status updates."""
        context: GraphQLContext = info.context

        try:
            # Subscribe to component events
            async for event in context.event_bus.subscribe("component_*"):
                if hasattr(event, "component_id"):
                    try:
                        status = context.component_manager.get_component_status(event.component_id)

                        if status:
                            yield ComponentInfo(
                                component_id=event.component_id,
                                state=status.get("state", "unknown"),
                                health_status=status.get("health_status", "unknown"),
                                error_count=status.get("error_count", 0),
                                performance_metrics=status.get("performance_metrics", {}),
                            )
                    except Exception:
                        continue  # Skip if can't get component status

        except Exception as e:
            context.logger.error(f"Error in component subscription: {str(e)}")


# Main schema definition
schema = strawberry.Schema(query=Query, mutation=Mutation, subscription=Subscription)


class GraphQLResolver:
    """
    Main GraphQL resolver class that provides the interface for GraphQL operations.

    This class integrates with all core AI assistant components:
    - Core Engine for multimodal processing
    - Component Manager for system component management
    - Workflow Orchestrator for task execution
    - Session Manager for user session handling
    - Interaction Handler for user interactions
    - Plugin Manager for system extensions
    - Memory systems for context and knowledge management
    - Learning systems for adaptation and improvement
    """

    def __init__(self, container: Container):
        """
        Initialize the GraphQL resolver.

        Args:
            container: Dependency injection container with all system components
        """
        self.container = container
        self.logger = get_logger(__name__)

        # Core components
        self.core_engine = container.get(EnhancedCoreEngine)
        self.component_manager = container.get(EnhancedComponentManager)
        self.workflow_orchestrator = container.get(WorkflowOrchestrator)
        self.session_manager = container.get(EnhancedSessionManager)
        self.interaction_handler = container.get(InteractionHandler)
        self.plugin_manager = container.get(EnhancedPluginManager)

        # Observability
        self.metrics = container.get(MetricsCollector)
        self.event_bus = container.get(EventBus)

        # Setup metrics
        self._setup_metrics()

        self.logger.info("GraphQL resolver initialized successfully")

    def _setup_metrics(self) -> None:
        """Setup GraphQL-specific metrics."""
        try:
            self.metrics.register_counter("graphql_queries_total")
            self.metrics.register_counter("graphql_mutations_total")
            self.metrics.register_counter("graphql_subscriptions_total")
            self.metrics.register_counter("graphql_errors_total")
            self.metrics.register_histogram("graphql_query_duration_seconds")
            self.metrics.register_histogram("graphql_mutation_duration_seconds")

        except Exception as e:
            self.logger.warning(f"Failed to setup GraphQL metrics: {str(e)}")

    def create_context(self, request) -> GraphQLContext:
        """
        Create GraphQL context for a request.

        Args:
            request: HTTP request object

        Returns:
            GraphQL context with all system components
        """
        return GraphQLContext(request, self.container)

    def get_schema(self) -> strawberry.Schema:
        """
        Get the GraphQL schema.

        Returns:
            Strawberry GraphQL schema
        """
        return schema

    async def health_check(self) -> Dict[str, Any]:
        """
        Health check for the GraphQL resolver.

        Returns:
            Health status information
        """
        try:
            return {
                "status": "healthy",
                "resolver_active": True,
                "schema_valid": True,
                "component_count": len(
                    [
                        self.core_engine,
                        self.component_manager,
                        self.workflow_orchestrator,
                        self.session_manager,
                        self.interaction_handler,
                        self.plugin_manager,
                    ]
                ),
            }
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}


# Export the main resolver and schema for use in the API layer
__all__ = [
    "GraphQLResolver",
    "GraphQLContext",
    "schema",
    "Query",
    "Mutation",
    "Subscription",
    "AuthenticationRequired",
    "AdminRequired",
]
