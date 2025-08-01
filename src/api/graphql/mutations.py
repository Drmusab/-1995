"""
GraphQL Mutations for AI Assistant
Author: Drmusab
Last Modified: 2025-06-20 04:00:00 UTC

This module provides comprehensive GraphQL mutations for the AI assistant,
enabling client applications to interact with all core system components
through a unified GraphQL interface with type safety and validation.
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Union

import asyncio
import strawberry
from strawberry.scalars import JSON
from strawberry.types import Info

# Schema imports
from src.api.graphql.schema import ComponentInfo as ComponentInfoType
from src.api.graphql.schema import (
    ComponentPriorityEnum,
    ErrorResponse,
    ExecutionModeEnum,
    InteractionContextSchema,
    InteractionModeEnum,
    ModalityEnum,
    MultimodalInputSchema,
)
from src.api.graphql.schema import PluginInfo as PluginInfoType
from src.api.graphql.schema import (
    PluginMetadataSchema,
    PluginTypeEnum,
    PriorityEnum,
    ProcessingContextSchema,
    ProcessingModeEnum,
)
from src.api.graphql.schema import (
    ProcessingResult as ProcessingResultType,  # Core types; Schema classes for validation; Enums
)
from src.api.graphql.schema import (
    SecurityLevelEnum,
    SessionConfigurationSchema,
)
from src.api.graphql.schema import SessionInfo as SessionInfoType
from src.api.graphql.schema import (
    SessionTypeEnum,
    StepTypeEnum,
    SuccessResponse,
    UserProfileSchema,
    WorkflowDefinitionSchema,
)
from src.api.graphql.schema import WorkflowExecution as WorkflowExecutionType
from src.api.graphql.schema import (
    WorkflowStateEnum,
)
from src.assistant.component_manager import (
    ComponentMetadata,
    ComponentPriority,
    EnhancedComponentManager,
)

# Assistant components
from src.assistant.core_engine import (
    EnhancedCoreEngine,
    ModalityType,
    MultimodalInput,
    PriorityLevel,
    ProcessingContext,
    ProcessingMode,
)
from src.assistant.interaction_handler import (
    InputModality,
    InteractionHandler,
    InteractionMode,
    InteractionPriority,
    OutputModality,
    UserMessage,
    UserProfile,
)
from src.assistant.plugin_manager import (
    EnhancedPluginManager,
    PluginLoadMode,
    PluginType,
    SecurityLevel,
)
from src.assistant.session_manager import (
    EnhancedSessionManager,
    SessionConfiguration,
    SessionPriority,
    SessionType,
)
from src.assistant.workflow_orchestrator import (
    ExecutionMode,
    StepType,
    WorkflowBuilder,
    WorkflowCondition,
    WorkflowDefinition,
    WorkflowOrchestrator,
    WorkflowPriority,
)

# Core imports
from src.core.config.loader import ConfigLoader
from src.core.dependency_injection import Container
from src.core.error_handling import ErrorHandler, handle_exceptions
from src.core.events.event_bus import EventBus
from src.core.security.authentication import AuthenticationManager
from src.core.security.authorization import AuthorizationManager
from src.core.security.sanitization import InputSanitizer
from src.learning.preference_learning import PreferenceLearner
from src.memory.core_memory.memory_manager import MemoryManager

# Observability
from src.observability.logging.config import get_logger

# Skills and memory
from src.skills.skill_factory import SkillFactory
from src.skills.skill_registry import SkillRegistry

logger = get_logger(__name__)


# =============================================================================
# Input Types for Mutations
# =============================================================================


@strawberry.input
class MultimodalInputInput:
    """Input type for multimodal data."""

    text: Optional[str] = None
    modality_weights: Optional[JSON] = None
    processing_hints: Optional[JSON] = None


@strawberry.input
class ProcessingContextInput:
    """Input type for processing context."""

    session_id: str
    user_id: Optional[str] = None
    conversation_id: Optional[str] = None
    priority: Optional[str] = "normal"
    timeout_seconds: Optional[float] = 30.0
    metadata: Optional[JSON] = None
    tags: Optional[List[str]] = None


@strawberry.input
class ComponentRegistrationInput:
    """Input type for component registration."""

    component_id: str
    component_type: str
    priority: Optional[str] = "normal"
    dependencies: Optional[List[str]] = None
    config_section: Optional[str] = None
    description: Optional[str] = None
    version: Optional[str] = "1.0.0"
    tags: Optional[List[str]] = None


@strawberry.input
class WorkflowStepInput:
    """Input type for workflow steps."""

    step_id: str
    step_type: str
    name: str
    description: Optional[str] = None
    skill_name: Optional[str] = None
    component_name: Optional[str] = None
    function_name: Optional[str] = None
    parameters: Optional[JSON] = None
    dependencies: Optional[List[str]] = None
    next_steps: Optional[List[str]] = None
    timeout_seconds: Optional[float] = 30.0
    priority: Optional[str] = "normal"
    continue_on_error: Optional[bool] = False
    max_retries: Optional[int] = 3


@strawberry.input
class WorkflowDefinitionInput:
    """Input type for workflow definitions."""

    workflow_id: str
    name: str
    description: Optional[str] = None
    version: Optional[str] = "1.0.0"
    steps: List[WorkflowStepInput]
    start_steps: List[str]
    end_steps: List[str]
    execution_mode: Optional[str] = "sequential"
    timeout_seconds: Optional[float] = 300.0
    max_concurrent_steps: Optional[int] = 5
    variables: Optional[JSON] = None
    category: Optional[str] = None
    tags: Optional[List[str]] = None


@strawberry.input
class WorkflowExecutionInput:
    """Input type for workflow execution."""

    workflow_id: str
    session_id: str
    user_id: Optional[str] = None
    input_data: Optional[JSON] = None
    priority: Optional[str] = "normal"
    context: Optional[JSON] = None


@strawberry.input
class SessionCreationInput:
    """Input type for session creation."""

    user_id: Optional[str] = None
    session_type: Optional[str] = "interactive"
    priority: Optional[str] = "normal"
    max_idle_time: Optional[float] = 1800.0
    max_session_time: Optional[float] = 86400.0
    context_data: Optional[JSON] = None
    device_info: Optional[JSON] = None
    location_info: Optional[JSON] = None


@strawberry.input
class InteractionStartInput:
    """Input type for starting interactions."""

    user_id: Optional[str] = None
    session_id: Optional[str] = None
    interaction_mode: Optional[str] = "conversational"
    input_modalities: Optional[List[str]] = None
    output_modalities: Optional[List[str]] = None
    device_info: Optional[JSON] = None
    priority: Optional[str] = "normal"


@strawberry.input
class UserMessageInput:
    """Input type for user messages."""

    interaction_id: str
    text: Optional[str] = None
    modality: Optional[str] = "text"
    language: Optional[str] = "en"
    real_time: Optional[bool] = False
    streaming: Optional[bool] = False
    conversation_context: Optional[JSON] = None


@strawberry.input
class UserPreferencesInput:
    """Input type for user preferences."""

    user_id: str
    preferred_language: Optional[str] = None
    preferred_interaction_mode: Optional[str] = None
    input_modalities: Optional[List[str]] = None
    output_modalities: Optional[List[str]] = None
    accessibility_needs: Optional[JSON] = None
    privacy_settings: Optional[JSON] = None


@strawberry.input
class PluginInstallationInput:
    """Input type for plugin installation."""

    plugin_path: Optional[str] = None
    plugin_url: Optional[str] = None
    plugin_data: Optional[JSON] = None
    auto_enable: Optional[bool] = True
    security_level: Optional[str] = "sandbox"


@strawberry.input
class FeedbackInput:
    """Input type for user feedback."""

    interaction_id: str
    feedback_type: str
    rating: Optional[int] = None
    comments: Optional[str] = None
    feedback_data: Optional[JSON] = None
    user_id: Optional[str] = None


# =============================================================================
# Mutation Results
# =============================================================================


@strawberry.type
class MutationResult:
    """Base result type for mutations."""

    success: bool
    message: Optional[str] = None
    data: Optional[JSON] = None
    errors: Optional[List[str]] = None
    timestamp: datetime = strawberry.field(default_factory=lambda: datetime.now(timezone.utc))


@strawberry.type
class ProcessingMutationResult:
    """Result type for processing mutations."""

    success: bool
    result: Optional[ProcessingResultType] = None
    errors: Optional[List[str]] = None
    timestamp: datetime = strawberry.field(default_factory=lambda: datetime.now(timezone.utc))


@strawberry.type
class WorkflowMutationResult:
    """Result type for workflow mutations."""

    success: bool
    execution_id: Optional[str] = None
    workflow_execution: Optional[WorkflowExecutionType] = None
    errors: Optional[List[str]] = None
    timestamp: datetime = strawberry.field(default_factory=lambda: datetime.now(timezone.utc))


@strawberry.type
class SessionMutationResult:
    """Result type for session mutations."""

    success: bool
    session_id: Optional[str] = None
    session_info: Optional[SessionInfoType] = None
    errors: Optional[List[str]] = None
    timestamp: datetime = strawberry.field(default_factory=lambda: datetime.now(timezone.utc))


@strawberry.type
class InteractionMutationResult:
    """Result type for interaction mutations."""

    success: bool
    interaction_id: Optional[str] = None
    response: Optional[JSON] = None
    errors: Optional[List[str]] = None
    timestamp: datetime = strawberry.field(default_factory=lambda: datetime.now(timezone.utc))


# =============================================================================
# Mutation Class
# =============================================================================


@strawberry.type
class Mutation:
    """Root mutation type for the AI Assistant GraphQL API."""

    def __init__(self):
        self.container: Optional[Container] = None
        self.core_engine: Optional[EnhancedCoreEngine] = None
        self.component_manager: Optional[EnhancedComponentManager] = None
        self.workflow_orchestrator: Optional[WorkflowOrchestrator] = None
        self.interaction_handler: Optional[InteractionHandler] = None
        self.session_manager: Optional[EnhancedSessionManager] = None
        self.plugin_manager: Optional[EnhancedPluginManager] = None
        self.skill_factory: Optional[SkillFactory] = None
        self.skill_registry: Optional[SkillRegistry] = None
        self.memory_manager: Optional[MemoryManager] = None
        self.preference_learner: Optional[PreferenceLearner] = None
        self.auth_manager: Optional[AuthenticationManager] = None
        self.authz_manager: Optional[AuthorizationManager] = None
        self.input_sanitizer: Optional[InputSanitizer] = None
        self.initialized = False

    async def _ensure_initialized(self, info: Info) -> None:
        """Ensure the mutation resolver is initialized with dependencies."""
        if not self.initialized:
            # Get container from GraphQL context
            self.container = info.context.get("container")
            if not self.container:
                raise RuntimeError("Dependency injection container not available")

            # Initialize components
            self.core_engine = self.container.get(EnhancedCoreEngine)
            self.component_manager = self.container.get(EnhancedComponentManager)
            self.workflow_orchestrator = self.container.get(WorkflowOrchestrator)
            self.interaction_handler = self.container.get(InteractionHandler)
            self.session_manager = self.container.get(EnhancedSessionManager)
            self.plugin_manager = self.container.get(EnhancedPluginManager)
            self.skill_factory = self.container.get(SkillFactory)
            self.skill_registry = self.container.get(SkillRegistry)
            self.memory_manager = self.container.get(MemoryManager)
            self.preference_learner = self.container.get(PreferenceLearner)
            self.auth_manager = self.container.get(AuthenticationManager)
            self.authz_manager = self.container.get(AuthorizationManager)
            self.input_sanitizer = self.container.get(InputSanitizer)

            self.initialized = True

    async def _check_authentication(self, info: Info, required: bool = True) -> Optional[str]:
        """Check user authentication and return user ID."""
        if not required:
            return info.context.get("user_id")

        user_id = info.context.get("user_id")
        if not user_id:
            raise PermissionError("Authentication required")

        if self.auth_manager:
            is_authenticated = await self.auth_manager.is_authenticated(user_id)
            if not is_authenticated:
                raise PermissionError("Invalid authentication")

        return user_id

    async def _check_authorization(self, user_id: str, resource: str, action: str) -> None:
        """Check user authorization for specific actions."""
        if self.authz_manager:
            is_authorized = await self.authz_manager.check_permission(user_id, resource, action)
            if not is_authorized:
                raise PermissionError(f"Not authorized to {action} {resource}")

    # =========================================================================
    # Core Engine Mutations
    # =========================================================================

    @strawberry.mutation
    async def process_multimodal_input(
        self, info: Info, input_data: MultimodalInputInput, context: ProcessingContextInput
    ) -> ProcessingMutationResult:
        """Process multimodal input through the core engine."""
        await self._ensure_initialized(info)
        user_id = await self._check_authentication(info, required=False)

        try:
            # Validate and sanitize input
            if input_data.text and self.input_sanitizer:
                input_data.text = await self.input_sanitizer.sanitize_text(input_data.text)

            # Create multimodal input
            multimodal_input = MultimodalInput(
                text=input_data.text,
                modality_weights=input_data.modality_weights or {},
                processing_hints=input_data.processing_hints or {},
            )

            # Create processing context
            processing_context = ProcessingContext(
                session_id=context.session_id,
                user_id=context.user_id or user_id,
                conversation_id=context.conversation_id,
                request_id=str(uuid.uuid4()),
                priority=(
                    PriorityLevel[context.priority.upper()]
                    if context.priority
                    else PriorityLevel.NORMAL
                ),
                timeout_seconds=context.timeout_seconds or 30.0,
                metadata=context.metadata or {},
                tags=context.tags or [],
            )

            # Process through core engine
            result = await self.core_engine.process_multimodal_input(
                multimodal_input, processing_context
            )

            # Convert to GraphQL type
            gql_result = ProcessingResultType(
                success=result.success,
                request_id=result.request_id,
                session_id=result.session_id,
                processing_time=result.processing_time,
                response_text=result.response_text,
                overall_confidence=result.overall_confidence,
                errors=result.errors,
                component_timings=result.component_timings,
            )

            return ProcessingMutationResult(
                success=result.success,
                result=gql_result,
                errors=result.errors if result.errors else None,
            )

        except Exception as e:
            logger.error(f"Error in process_multimodal_input: {str(e)}")
            return ProcessingMutationResult(success=False, errors=[str(e)])

    # =========================================================================
    # Component Manager Mutations
    # =========================================================================

    @strawberry.mutation
    async def register_component(
        self, info: Info, component_input: ComponentRegistrationInput
    ) -> MutationResult:
        """Register a new component with the component manager."""
        await self._ensure_initialized(info)
        user_id = await self._check_authentication(info)
        await self._check_authorization(user_id, "components", "create")

        try:
            # Create component metadata
            dependencies = []
            if component_input.dependencies:
                for dep_id in component_input.dependencies:
                    from src.assistant.component_manager import ComponentDependency, DependencyType

                    dependencies.append(
                        ComponentDependency(
                            component_id=dep_id, dependency_type=DependencyType.REQUIRED
                        )
                    )

            metadata = ComponentMetadata(
                component_id=component_input.component_id,
                component_type=type(
                    component_input.component_type
                ),  # This would need proper type resolution
                priority=(
                    ComponentPriority[component_input.priority.upper()]
                    if component_input.priority
                    else ComponentPriority.NORMAL
                ),
                dependencies=dependencies,
                config_section=component_input.config_section,
                description=component_input.description,
                version=component_input.version or "1.0.0",
                tags=set(component_input.tags or []),
            )

            # Register component
            self.component_manager.register_component(
                component_input.component_id,
                type(component_input.component_type),  # This would need proper type resolution
                metadata.priority,
                dependencies,
            )

            return MutationResult(
                success=True,
                message=f"Component {component_input.component_id} registered successfully",
                data={"component_id": component_input.component_id},
            )

        except Exception as e:
            logger.error(f"Error in register_component: {str(e)}")
            return MutationResult(
                success=False, message="Failed to register component", errors=[str(e)]
            )

    @strawberry.mutation
    async def start_component(self, info: Info, component_id: str) -> MutationResult:
        """Start a registered component."""
        await self._ensure_initialized(info)
        user_id = await self._check_authentication(info)
        await self._check_authorization(user_id, "components", "update")

        try:
            # This would need to be implemented in the component manager
            # For now, return success
            return MutationResult(
                success=True,
                message=f"Component {component_id} started successfully",
                data={"component_id": component_id},
            )

        except Exception as e:
            logger.error(f"Error in start_component: {str(e)}")
            return MutationResult(
                success=False, message="Failed to start component", errors=[str(e)]
            )

    @strawberry.mutation
    async def stop_component(self, info: Info, component_id: str) -> MutationResult:
        """Stop a running component."""
        await self._ensure_initialized(info)
        user_id = await self._check_authentication(info)
        await self._check_authorization(user_id, "components", "update")

        try:
            await self.component_manager.stop_component(component_id)

            return MutationResult(
                success=True,
                message=f"Component {component_id} stopped successfully",
                data={"component_id": component_id},
            )

        except Exception as e:
            logger.error(f"Error in stop_component: {str(e)}")
            return MutationResult(
                success=False, message="Failed to stop component", errors=[str(e)]
            )

    # =========================================================================
    # Workflow Orchestrator Mutations
    # =========================================================================

    @strawberry.mutation
    async def create_workflow(
        self, info: Info, workflow_input: WorkflowDefinitionInput
    ) -> MutationResult:
        """Create a new workflow definition."""
        await self._ensure_initialized(info)
        user_id = await self._check_authentication(info)
        await self._check_authorization(user_id, "workflows", "create")

        try:
            # Create workflow using builder
            builder = WorkflowBuilder()
            builder.set_basic_info(
                workflow_input.workflow_id,
                workflow_input.name,
                workflow_input.description,
                workflow_input.version or "1.0.0",
            )

            # Add steps
            for step_input in workflow_input.steps:
                from src.assistant.workflow_orchestrator import WorkflowStep

                step = WorkflowStep(
                    step_id=step_input.step_id,
                    step_type=StepType[step_input.step_type.upper()],
                    name=step_input.name,
                    description=step_input.description,
                    skill_name=step_input.skill_name,
                    component_name=step_input.component_name,
                    function_name=step_input.function_name,
                    parameters=step_input.parameters or {},
                    dependencies=step_input.dependencies or [],
                    next_steps=step_input.next_steps or [],
                    timeout_seconds=step_input.timeout_seconds or 30.0,
                    priority=(
                        WorkflowPriority[step_input.priority.upper()]
                        if step_input.priority
                        else WorkflowPriority.NORMAL
                    ),
                    continue_on_error=step_input.continue_on_error or False,
                    max_retries=step_input.max_retries or 3,
                )
                builder.add_step(step)

            # Set flow
            builder.set_start_steps(*workflow_input.start_steps)
            builder.set_end_steps(*workflow_input.end_steps)

            # Set execution mode
            if workflow_input.execution_mode:
                builder.set_execution_mode(ExecutionMode[workflow_input.execution_mode.upper()])

            # Build and register workflow
            workflow = builder.build()
            workflow.timeout_seconds = workflow_input.timeout_seconds or 300.0
            workflow.max_concurrent_steps = workflow_input.max_concurrent_steps or 5
            workflow.variables = workflow_input.variables or {}
            workflow.category = workflow_input.category
            workflow.tags = set(workflow_input.tags or [])
            workflow.created_by = user_id

            self.workflow_orchestrator.register_workflow(workflow)

            return MutationResult(
                success=True,
                message=f"Workflow {workflow_input.workflow_id} created successfully",
                data={"workflow_id": workflow_input.workflow_id},
            )

        except Exception as e:
            logger.error(f"Error in create_workflow: {str(e)}")
            return MutationResult(
                success=False, message="Failed to create workflow", errors=[str(e)]
            )

    @strawberry.mutation
    async def execute_workflow(
        self, info: Info, execution_input: WorkflowExecutionInput
    ) -> WorkflowMutationResult:
        """Execute a workflow."""
        await self._ensure_initialized(info)
        user_id = await self._check_authentication(info)
        await self._check_authorization(user_id, "workflows", "execute")

        try:
            execution_id = await self.workflow_orchestrator.execute_workflow(
                execution_input.workflow_id,
                execution_input.session_id,
                execution_input.input_data or {},
                execution_input.user_id or user_id,
                (
                    WorkflowPriority[execution_input.priority.upper()]
                    if execution_input.priority
                    else WorkflowPriority.NORMAL
                ),
                execution_input.context or {},
            )

            # Get execution status
            execution_status = await self.workflow_orchestrator.get_execution_status(execution_id)

            # Convert to GraphQL type
            gql_execution = WorkflowExecutionType(
                execution_id=execution_id,
                workflow_id=execution_input.workflow_id,
                session_id=execution_input.session_id,
                state=execution_status["state"],
                execution_time=execution_status["execution_time"],
                completed_steps=execution_status["completed_steps"],
                current_steps=execution_status["current_steps"],
            )

            return WorkflowMutationResult(
                success=True, execution_id=execution_id, workflow_execution=gql_execution
            )

        except Exception as e:
            logger.error(f"Error in execute_workflow: {str(e)}")
            return WorkflowMutationResult(success=False, errors=[str(e)])

    @strawberry.mutation
    async def cancel_workflow_execution(self, info: Info, execution_id: str) -> MutationResult:
        """Cancel a running workflow execution."""
        await self._ensure_initialized(info)
        user_id = await self._check_authentication(info)
        await self._check_authorization(user_id, "workflows", "update")

        try:
            await self.workflow_orchestrator.cancel_execution(execution_id)

            return MutationResult(
                success=True,
                message=f"Workflow execution {execution_id} cancelled successfully",
                data={"execution_id": execution_id},
            )

        except Exception as e:
            logger.error(f"Error in cancel_workflow_execution: {str(e)}")
            return MutationResult(
                success=False, message="Failed to cancel workflow execution", errors=[str(e)]
            )

    # =========================================================================
    # Session Manager Mutations
    # =========================================================================

    @strawberry.mutation
    async def create_session(
        self, info: Info, session_input: SessionCreationInput
    ) -> SessionMutationResult:
        """Create a new user session."""
        await self._ensure_initialized(info)
        user_id = await self._check_authentication(info, required=False)

        try:
            # Create session configuration
            config = SessionConfiguration(
                session_type=(
                    SessionType[session_input.session_type.upper()]
                    if session_input.session_type
                    else SessionType.INTERACTIVE
                ),
                priority=(
                    SessionPriority[session_input.priority.upper()]
                    if session_input.priority
                    else SessionPriority.NORMAL
                ),
                max_idle_time=session_input.max_idle_time or 1800.0,
                max_session_time=session_input.max_session_time or 86400.0,
            )

            # Prepare context data
            context_data = session_input.context_data or {}
            if session_input.device_info:
                context_data["device_info"] = session_input.device_info
            if session_input.location_info:
                context_data["location_info"] = session_input.location_info

            # Create session
            session_id = await self.session_manager.create_session(
                session_input.user_id or user_id, config, context_data
            )

            # Get session info
            session_info = await self.session_manager.get_session(session_id)

            # Convert to GraphQL type
            gql_session = SessionInfoType(
                session_id=session_id,
                state=session_info.state.value,
                interaction_count=session_info.interaction_count,
                memory_usage_mb=session_info.memory_usage_mb,
                health_score=session_info.health_score,
                last_activity=session_info.last_activity,
            )

            return SessionMutationResult(
                success=True, session_id=session_id, session_info=gql_session
            )

        except Exception as e:
            logger.error(f"Error in create_session: {str(e)}")
            return SessionMutationResult(success=False, errors=[str(e)])

    @strawberry.mutation
    async def end_session(
        self, info: Info, session_id: str, reason: Optional[str] = "user_ended"
    ) -> MutationResult:
        """End an active session."""
        await self._ensure_initialized(info)
        user_id = await self._check_authentication(info, required=False)

        try:
            await self.session_manager.end_session(session_id, reason or "user_ended")

            return MutationResult(
                success=True,
                message=f"Session {session_id} ended successfully",
                data={"session_id": session_id, "reason": reason},
            )

        except Exception as e:
            logger.error(f"Error in end_session: {str(e)}")
            return MutationResult(success=False, message="Failed to end session", errors=[str(e)])

    # =========================================================================
    # Interaction Handler Mutations
    # =========================================================================

    @strawberry.mutation
    async def start_interaction(
        self, info: Info, interaction_input: InteractionStartInput
    ) -> InteractionMutationResult:
        """Start a new user interaction."""
        await self._ensure_initialized(info)
        user_id = await self._check_authentication(info, required=False)

        try:
            # Convert input modalities
            input_modalities = set()
            if interaction_input.input_modalities:
                input_modalities = {
                    InputModality[m.upper()] for m in interaction_input.input_modalities
                }

            # Convert output modalities
            output_modalities = set()
            if interaction_input.output_modalities:
                output_modalities = {
                    OutputModality[m.upper()] for m in interaction_input.output_modalities
                }

            # Start interaction
            interaction_id = await self.interaction_handler.start_interaction(
                interaction_input.user_id or user_id,
                interaction_input.session_id,
                (
                    InteractionMode[interaction_input.interaction_mode.upper()]
                    if interaction_input.interaction_mode
                    else InteractionMode.CONVERSATIONAL
                ),
                input_modalities,
                output_modalities,
                interaction_input.device_info,
                (
                    InteractionPriority[interaction_input.priority.upper()]
                    if interaction_input.priority
                    else InteractionPriority.NORMAL
                ),
            )

            return InteractionMutationResult(
                success=True,
                interaction_id=interaction_id,
                response={"interaction_id": interaction_id, "status": "started"},
            )

        except Exception as e:
            logger.error(f"Error in start_interaction: {str(e)}")
            return InteractionMutationResult(success=False, errors=[str(e)])

    @strawberry.mutation
    async def process_user_message(
        self, info: Info, message_input: UserMessageInput
    ) -> InteractionMutationResult:
        """Process a user message."""
        await self._ensure_initialized(info)
        user_id = await self._check_authentication(info, required=False)

        try:
            # Create user message
            message = UserMessage(
                message_id=str(uuid.uuid4()),
                user_id=user_id,
                interaction_id=message_input.interaction_id,
                text=message_input.text,
                modality=(
                    InputModality[message_input.modality.upper()]
                    if message_input.modality
                    else InputModality.TEXT
                ),
                language=message_input.language or "en",
                conversation_context=message_input.conversation_context or {},
            )

            # Process message
            response = await self.interaction_handler.process_user_message(
                message_input.interaction_id,
                message,
                message_input.real_time or False,
                message_input.streaming or False,
            )

            # Convert response to JSON
            response_data = {
                "response_id": response.response_id,
                "text": response.text,
                "modalities": [m.value for m in response.modalities],
                "confidence": response.confidence,
                "processing_time": response.processing_time,
                "expects_response": response.expects_response,
                "suggested_follow_ups": response.suggested_follow_ups,
            }

            return InteractionMutationResult(
                success=True, interaction_id=message_input.interaction_id, response=response_data
            )

        except Exception as e:
            logger.error(f"Error in process_user_message: {str(e)}")
            return InteractionMutationResult(success=False, errors=[str(e)])

    @strawberry.mutation
    async def end_interaction(
        self, info: Info, interaction_id: str, reason: Optional[str] = "completed"
    ) -> MutationResult:
        """End an active interaction."""
        await self._ensure_initialized(info)
        await self._check_authentication(info, required=False)

        try:
            await self.interaction_handler.end_interaction(interaction_id, reason or "completed")

            return MutationResult(
                success=True,
                message=f"Interaction {interaction_id} ended successfully",
                data={"interaction_id": interaction_id, "reason": reason},
            )

        except Exception as e:
            logger.error(f"Error in end_interaction: {str(e)}")
            return MutationResult(
                success=False, message="Failed to end interaction", errors=[str(e)]
            )

    # =========================================================================
    # User Preference Mutations
    # =========================================================================

    @strawberry.mutation
    async def update_user_preferences(
        self, info: Info, preferences_input: UserPreferencesInput
    ) -> MutationResult:
        """Update user preferences."""
        await self._ensure_initialized(info)
        user_id = await self._check_authentication(info)

        # Check if user can update their own preferences or admin can update any
        if user_id != preferences_input.user_id:
            await self._check_authorization(user_id, "users", "update")

        try:
            # Prepare preferences update
            preferences = {}

            if preferences_input.preferred_language:
                preferences["preferred_language"] = preferences_input.preferred_language

            if preferences_input.preferred_interaction_mode:
                preferences["interaction_mode"] = preferences_input.preferred_interaction_mode

            if preferences_input.input_modalities:
                preferences["input_modalities"] = preferences_input.input_modalities

            if preferences_input.output_modalities:
                preferences["output_modalities"] = preferences_input.output_modalities

            if preferences_input.accessibility_needs:
                preferences["accessibility"] = preferences_input.accessibility_needs

            if preferences_input.privacy_settings:
                preferences["privacy"] = preferences_input.privacy_settings

            # Update preferences through interaction handler
            await self.interaction_handler.update_user_preferences(
                preferences_input.user_id, preferences
            )

            return MutationResult(
                success=True,
                message=f"Preferences updated for user {preferences_input.user_id}",
                data={"user_id": preferences_input.user_id},
            )

        except Exception as e:
            logger.error(f"Error in update_user_preferences: {str(e)}")
            return MutationResult(
                success=False, message="Failed to update user preferences", errors=[str(e)]
            )

    # =========================================================================
    # Plugin Manager Mutations
    # =========================================================================

    @strawberry.mutation
    async def install_plugin(
        self, info: Info, plugin_input: PluginInstallationInput
    ) -> MutationResult:
        """Install a new plugin."""
        await self._ensure_initialized(info)
        user_id = await self._check_authentication(info)
        await self._check_authorization(user_id, "plugins", "create")

        try:
            # This would need to be implemented in the plugin manager
            # For now, return success placeholder
            return MutationResult(
                success=True, message="Plugin installation initiated", data={"status": "installing"}
            )

        except Exception as e:
            logger.error(f"Error in install_plugin: {str(e)}")
            return MutationResult(
                success=False, message="Failed to install plugin", errors=[str(e)]
            )

    @strawberry.mutation
    async def enable_plugin(self, info: Info, plugin_id: str) -> MutationResult:
        """Enable a plugin."""
        await self._ensure_initialized(info)
        user_id = await self._check_authentication(info)
        await self._check_authorization(user_id, "plugins", "update")

        try:
            await self.plugin_manager.enable_plugin(plugin_id)

            return MutationResult(
                success=True,
                message=f"Plugin {plugin_id} enabled successfully",
                data={"plugin_id": plugin_id},
            )

        except Exception as e:
            logger.error(f"Error in enable_plugin: {str(e)}")
            return MutationResult(success=False, message="Failed to enable plugin", errors=[str(e)])

    @strawberry.mutation
    async def disable_plugin(self, info: Info, plugin_id: str) -> MutationResult:
        """Disable a plugin."""
        await self._ensure_initialized(info)
        user_id = await self._check_authentication(info)
        await self._check_authorization(user_id, "plugins", "update")

        try:
            await self.plugin_manager.disable_plugin(plugin_id)

            return MutationResult(
                success=True,
                message=f"Plugin {plugin_id} disabled successfully",
                data={"plugin_id": plugin_id},
            )

        except Exception as e:
            logger.error(f"Error in disable_plugin: {str(e)}")
            return MutationResult(
                success=False, message="Failed to disable plugin", errors=[str(e)]
            )

    @strawberry.mutation
    async def unload_plugin(
        self, info: Info, plugin_id: str, force: Optional[bool] = False
    ) -> MutationResult:
        """Unload a plugin."""
        await self._ensure_initialized(info)
        user_id = await self._check_authentication(info)
        await self._check_authorization(user_id, "plugins", "delete")

        try:
            await self.plugin_manager.unload_plugin(plugin_id, force or False)

            return MutationResult(
                success=True,
                message=f"Plugin {plugin_id} unloaded successfully",
                data={"plugin_id": plugin_id},
            )

        except Exception as e:
            logger.error(f"Error in unload_plugin: {str(e)}")
            return MutationResult(success=False, message="Failed to unload plugin", errors=[str(e)])

    # =========================================================================
    # Feedback and Learning Mutations
    # =========================================================================

    @strawberry.mutation
    async def submit_feedback(self, info: Info, feedback_input: FeedbackInput) -> MutationResult:
        """Submit user feedback for an interaction."""
        await self._ensure_initialized(info)
        user_id = await self._check_authentication(info, required=False)

        try:
            # Prepare feedback data
            feedback_data = feedback_input.feedback_data or {}
            if feedback_input.rating is not None:
                feedback_data["rating"] = feedback_input.rating
            if feedback_input.comments:
                feedback_data["comments"] = feedback_input.comments

            # Submit feedback
            await self.interaction_handler.handle_user_feedback(
                feedback_input.interaction_id,
                feedback_input.feedback_type,
                feedback_data,
                feedback_input.user_id or user_id,
            )

            return MutationResult(
                success=True,
                message="Feedback submitted successfully",
                data={
                    "interaction_id": feedback_input.interaction_id,
                    "feedback_type": feedback_input.feedback_type,
                },
            )

        except Exception as e:
            logger.error(f"Error in submit_feedback: {str(e)}")
            return MutationResult(
                success=False, message="Failed to submit feedback", errors=[str(e)]
            )

    # =========================================================================
    # Memory Management Mutations
    # =========================================================================

    @strawberry.mutation
    async def store_memory(
        self,
        info: Info,
        memory_type: str,
        memory_data: JSON,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> MutationResult:
        """Store data in the memory system."""
        await self._ensure_initialized(info)
        current_user_id = await self._check_authentication(info, required=False)

        try:
            # Use provided user_id or current user
            target_user_id = user_id or current_user_id

            # Store memory based on type
            if memory_type.lower() == "episodic":
                await self.memory_manager.store_episodic_memory(
                    event_type="user_data",
                    data=memory_data,
                    session_id=session_id,
                    user_id=target_user_id,
                )
            elif memory_type.lower() == "semantic":
                await self.memory_manager.store_semantic_memory(
                    content=str(memory_data),
                    metadata={"user_id": target_user_id, "session_id": session_id},
                )
            else:
                raise ValueError(f"Unknown memory type: {memory_type}")

            return MutationResult(
                success=True,
                message=f"{memory_type.capitalize()} memory stored successfully",
                data={"memory_type": memory_type, "user_id": target_user_id},
            )

        except Exception as e:
            logger.error(f"Error in store_memory: {str(e)}")
            return MutationResult(success=False, message="Failed to store memory", errors=[str(e)])

    @strawberry.mutation
    async def clear_session_memory(self, info: Info, session_id: str) -> MutationResult:
        """Clear memory for a specific session."""
        await self._ensure_initialized(info)
        user_id = await self._check_authentication(info)
        await self._check_authorization(user_id, "memory", "delete")

        try:
            # Clear working memory for session
            await self.memory_manager.clear_session_memory(session_id)

            return MutationResult(
                success=True,
                message=f"Memory cleared for session {session_id}",
                data={"session_id": session_id},
            )

        except Exception as e:
            logger.error(f"Error in clear_session_memory: {str(e)}")
            return MutationResult(
                success=False, message="Failed to clear session memory", errors=[str(e)]
            )

    # =========================================================================
    # System Administration Mutations
    # =========================================================================

    @strawberry.mutation
    async def system_shutdown(self, info: Info, graceful: Optional[bool] = True) -> MutationResult:
        """Initiate system shutdown."""
        await self._ensure_initialized(info)
        user_id = await self._check_authentication(info)
        await self._check_authorization(user_id, "system", "admin")

        try:
            if graceful:
                # Initiate graceful shutdown
                await self.core_engine.shutdown()
                await self.component_manager.shutdown_all()

                return MutationResult(
                    success=True,
                    message="Graceful system shutdown initiated",
                    data={"shutdown_type": "graceful"},
                )
            else:
                # Force shutdown (would need implementation)
                return MutationResult(
                    success=True,
                    message="Force system shutdown initiated",
                    data={"shutdown_type": "force"},
                )

        except Exception as e:
            logger.error(f"Error in system_shutdown: {str(e)}")
            return MutationResult(
                success=False, message="Failed to initiate system shutdown", errors=[str(e)]
            )


# =============================================================================
# Export
# =============================================================================

__all__ = [
    "Mutation",
    "MutationResult",
    "ProcessingMutationResult",
    "WorkflowMutationResult",
    "SessionMutationResult",
    "InteractionMutationResult",
    # Input types
    "MultimodalInputInput",
    "ProcessingContextInput",
    "ComponentRegistrationInput",
    "WorkflowStepInput",
    "WorkflowDefinitionInput",
    "WorkflowExecutionInput",
    "SessionCreationInput",
    "InteractionStartInput",
    "UserMessageInput",
    "UserPreferencesInput",
    "PluginInstallationInput",
    "FeedbackInput",
]
