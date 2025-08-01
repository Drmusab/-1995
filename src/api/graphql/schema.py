"""
Comprehensive Schema System for AI Assistant
Author: Drmusab
Last Modified: 2025-01-20 03:31:48 UTC

This module provides unified schema definitions for all core system components,
ensuring type safety, validation, and consistency across APIs, databases, and
internal communications.
"""

import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, ForwardRef, List, Literal, Optional, Set, Union

import strawberry
from pydantic import (
    UUID4,
    BaseModel,
    ConfigDict,
    EmailStr,
    Field,
    HttpUrl,
    Json,
    SecretStr,
    root_validator,
    validator,
)
from pydantic.types import NonNegativeFloat, NonNegativeInt, PositiveFloat, PositiveInt
from strawberry.scalars import JSON

# =============================================================================
# Base Schema Classes
# =============================================================================


class TimestampMixin(BaseModel):
    """Mixin for timestamp fields."""

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: Optional[datetime] = None

    def update_timestamp(self):
        self.updated_at = datetime.now(timezone.utc)


class VersionedMixin(BaseModel):
    """Mixin for versioned entities."""

    version: PositiveInt = Field(default=1, description="Schema version")
    schema_version: str = Field(default="1.0.0", description="Schema version string")


class IdentifiableMixin(BaseModel):
    """Mixin for entities with unique identifiers."""

    id: UUID4 = Field(default_factory=uuid.uuid4, description="Unique identifier")

    class Config:
        json_encoders = {uuid.UUID: str}


# =============================================================================
# Enums
# =============================================================================


class StatusEnum(str, Enum):
    """Generic status enumeration."""

    ACTIVE = "active"
    INACTIVE = "inactive"
    PENDING = "pending"
    SUSPENDED = "suspended"
    ERROR = "error"
    TERMINATED = "terminated"


class PriorityEnum(str, Enum):
    """Priority levels."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class HealthStatusEnum(str, Enum):
    """Health status enumeration."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class ProcessingModeEnum(str, Enum):
    """Processing mode enumeration."""

    SYNCHRONOUS = "synchronous"
    ASYNCHRONOUS = "asynchronous"
    REAL_TIME = "real_time"
    BATCH = "batch"
    STREAMING = "streaming"
    INTERACTIVE = "interactive"


class ModalityEnum(str, Enum):
    """Input/output modality types."""

    TEXT = "text"
    SPEECH = "speech"
    VISION = "vision"
    AUDIO = "audio"
    GESTURE = "gesture"
    MULTIMODAL = "multimodal"
    HAPTIC = "haptic"
    VISUAL = "visual"


# =============================================================================
# Core Engine Schemas
# =============================================================================


class MultimodalInputSchema(BaseModel):
    """Schema for multimodal input data."""

    text: Optional[str] = Field(None, description="Text input")
    audio_data: Optional[bytes] = Field(None, description="Audio data")
    image_data: Optional[bytes] = Field(None, description="Image data")
    video_data: Optional[bytes] = Field(None, description="Video data")
    gesture_data: Optional[Dict[str, Any]] = Field(None, description="Gesture data")
    modality_weights: Dict[str, float] = Field(default_factory=dict, description="Modality weights")
    processing_hints: Dict[str, Any] = Field(default_factory=dict, description="Processing hints")

    @validator("modality_weights")
    def validate_weights(cls, v):
        if v and not all(0 <= weight <= 1 for weight in v.values()):
            raise ValueError("Modality weights must be between 0 and 1")
        return v


class ProcessingContextSchema(BaseModel, TimestampMixin):
    """Schema for processing context."""

    session_id: str = Field(..., description="Session identifier")
    user_id: Optional[str] = Field(None, description="User identifier")
    conversation_id: Optional[str] = Field(None, description="Conversation identifier")
    request_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()), description="Request identifier"
    )
    priority: PriorityEnum = Field(default=PriorityEnum.NORMAL, description="Processing priority")
    timeout_seconds: PositiveFloat = Field(default=30.0, description="Processing timeout")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    tags: List[str] = Field(default_factory=list, description="Context tags")


class ProcessingResultSchema(BaseModel, TimestampMixin):
    """Schema for processing results."""

    success: bool = Field(..., description="Processing success status")
    request_id: str = Field(..., description="Request identifier")
    session_id: str = Field(..., description="Session identifier")
    processing_time: NonNegativeFloat = Field(..., description="Processing time in seconds")

    # Core results
    response_text: Optional[str] = Field(None, description="Generated text response")
    synthesized_audio: Optional[bytes] = Field(None, description="Synthesized audio data")
    generated_image: Optional[bytes] = Field(None, description="Generated image data")

    # Analysis results
    transcription_result: Optional[Dict[str, Any]] = Field(
        None, description="Speech transcription result"
    )
    emotion_result: Optional[Dict[str, Any]] = Field(None, description="Emotion detection result")
    speaker_result: Optional[Dict[str, Any]] = Field(None, description="Speaker recognition result")
    vision_result: Optional[Dict[str, Any]] = Field(None, description="Vision processing result")
    intent_result: Optional[Dict[str, Any]] = Field(None, description="Intent detection result")
    entity_result: Optional[List[Dict[str, Any]]] = Field(
        None, description="Entity extraction result"
    )
    sentiment_result: Optional[Dict[str, Any]] = Field(
        None, description="Sentiment analysis result"
    )

    # Reasoning and planning
    reasoning_trace: Optional[List[Dict[str, Any]]] = Field(None, description="Reasoning trace")
    decision_path: Optional[List[str]] = Field(None, description="Decision path")
    executed_skills: List[str] = Field(default_factory=list, description="Executed skills")

    # Memory operations
    memory_updates: List[Dict[str, Any]] = Field(default_factory=list, description="Memory updates")
    retrieved_memories: List[Dict[str, Any]] = Field(
        default_factory=list, description="Retrieved memories"
    )

    # Learning and adaptation
    learning_updates: List[Dict[str, Any]] = Field(
        default_factory=list, description="Learning updates"
    )
    preference_updates: List[Dict[str, Any]] = Field(
        default_factory=list, description="Preference updates"
    )

    # Quality metrics
    overall_confidence: NonNegativeFloat = Field(
        default=0.0, ge=0.0, le=1.0, description="Overall confidence"
    )
    modality_confidences: Dict[str, float] = Field(
        default_factory=dict, description="Per-modality confidence"
    )
    quality_metrics: Dict[str, float] = Field(default_factory=dict, description="Quality metrics")

    # Error handling
    errors: List[str] = Field(default_factory=list, description="Processing errors")
    warnings: List[str] = Field(default_factory=list, description="Processing warnings")

    # Performance metrics
    component_timings: Dict[str, float] = Field(
        default_factory=dict, description="Component execution times"
    )
    memory_usage: Dict[str, float] = Field(
        default_factory=dict, description="Memory usage statistics"
    )

    # Engine metadata
    engine_version: str = Field(default="1.0.0", description="Engine version")


# =============================================================================
# Component Manager Schemas
# =============================================================================


class ComponentStateEnum(str, Enum):
    """Component state enumeration."""

    UNREGISTERED = "unregistered"
    REGISTERED = "registered"
    INITIALIZING = "initializing"
    INITIALIZED = "initialized"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"
    DISABLED = "disabled"


class ComponentPriorityEnum(str, Enum):
    """Component priority enumeration."""

    CRITICAL = "critical"
    ESSENTIAL = "essential"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"
    BACKGROUND = "background"


class DependencyTypeEnum(str, Enum):
    """Dependency type enumeration."""

    REQUIRED = "required"
    OPTIONAL = "optional"
    WEAK = "weak"
    CONDITIONAL = "conditional"


class ComponentDependencySchema(BaseModel):
    """Schema for component dependencies."""

    component_id: str = Field(..., description="Component identifier")
    dependency_type: DependencyTypeEnum = Field(
        default=DependencyTypeEnum.REQUIRED, description="Dependency type"
    )
    condition: Optional[str] = Field(None, description="Conditional dependency expression")
    timeout_seconds: PositiveFloat = Field(
        default=30.0, description="Dependency resolution timeout"
    )
    retry_count: NonNegativeInt = Field(default=3, description="Retry attempts")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Dependency metadata")


class ComponentMetadataSchema(BaseModel, TimestampMixin):
    """Schema for component metadata."""

    component_id: str = Field(..., description="Component identifier")
    component_type: str = Field(..., description="Component type")
    priority: ComponentPriorityEnum = Field(
        default=ComponentPriorityEnum.NORMAL, description="Component priority"
    )
    dependencies: List[ComponentDependencySchema] = Field(
        default_factory=list, description="Component dependencies"
    )
    provides: Set[str] = Field(default_factory=set, description="Provided capabilities")
    config_section: Optional[str] = Field(None, description="Configuration section")
    health_check_interval: PositiveFloat = Field(default=60.0, description="Health check interval")
    auto_restart: bool = Field(default=True, description="Auto-restart on failure")
    restart_max_attempts: PositiveInt = Field(default=3, description="Maximum restart attempts")
    restart_backoff_factor: PositiveFloat = Field(default=2.0, description="Restart backoff factor")
    shutdown_timeout: PositiveFloat = Field(default=30.0, description="Shutdown timeout")
    initialization_timeout: PositiveFloat = Field(
        default=60.0, description="Initialization timeout"
    )
    tags: Set[str] = Field(default_factory=set, description="Component tags")
    description: Optional[str] = Field(None, description="Component description")
    version: str = Field(default="1.0.0", description="Component version")


class ComponentInfoSchema(BaseModel, TimestampMixin):
    """Schema for component runtime information."""

    metadata: ComponentMetadataSchema = Field(..., description="Component metadata")
    state: ComponentStateEnum = Field(
        default=ComponentStateEnum.UNREGISTERED, description="Component state"
    )
    startup_time: Optional[datetime] = Field(None, description="Component startup time")
    last_health_check: Optional[datetime] = Field(None, description="Last health check time")
    health_status: HealthStatusEnum = Field(
        default=HealthStatusEnum.UNKNOWN, description="Health status"
    )
    error_count: NonNegativeInt = Field(default=0, description="Error count")
    restart_count: NonNegativeInt = Field(default=0, description="Restart count")
    last_error: Optional[str] = Field(None, description="Last error message")
    performance_metrics: Dict[str, Any] = Field(
        default_factory=dict, description="Performance metrics"
    )
    resource_usage: Dict[str, Any] = Field(default_factory=dict, description="Resource usage")


# =============================================================================
# Workflow Orchestrator Schemas
# =============================================================================


class WorkflowStateEnum(str, Enum):
    """Workflow execution states."""

    CREATED = "created"
    PLANNING = "planning"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    SUSPENDED = "suspended"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRY = "retry"


class StepStateEnum(str, Enum):
    """Step execution states."""

    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRY = "retry"
    BLOCKED = "blocked"


class ExecutionModeEnum(str, Enum):
    """Workflow execution modes."""

    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    CONDITIONAL = "conditional"
    ADAPTIVE = "adaptive"
    STREAMING = "streaming"
    BATCH = "batch"


class StepTypeEnum(str, Enum):
    """Workflow step types."""

    SKILL_EXECUTION = "skill_execution"
    DATA_PROCESSING = "data_processing"
    DECISION_POINT = "decision_point"
    PARALLEL_GATEWAY = "parallel_gateway"
    MERGE_GATEWAY = "merge_gateway"
    CONDITION_CHECK = "condition_check"
    USER_INTERACTION = "user_interaction"
    MEMORY_OPERATION = "memory_operation"
    EXTERNAL_API_CALL = "external_api_call"
    VALIDATION = "validation"
    TRANSFORMATION = "transformation"


class WorkflowConditionSchema(BaseModel):
    """Schema for workflow conditions."""

    condition_id: str = Field(..., description="Condition identifier")
    expression: str = Field(..., description="Condition expression")
    condition_type: str = Field(default="boolean", description="Condition type")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Condition parameters")
    description: Optional[str] = Field(None, description="Condition description")


class WorkflowStepSchema(BaseModel, TimestampMixin):
    """Schema for workflow steps."""

    step_id: str = Field(..., description="Step identifier")
    step_type: StepTypeEnum = Field(..., description="Step type")
    name: str = Field(..., description="Step name")
    description: Optional[str] = Field(None, description="Step description")

    # Execution configuration
    skill_name: Optional[str] = Field(None, description="Skill name for skill execution")
    component_name: Optional[str] = Field(None, description="Component name")
    function_name: Optional[str] = Field(None, description="Function name")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Step parameters")

    # Flow control
    dependencies: List[str] = Field(default_factory=list, description="Step dependencies")
    conditions: List[WorkflowConditionSchema] = Field(
        default_factory=list, description="Step conditions"
    )
    next_steps: List[str] = Field(default_factory=list, description="Next steps")

    # Error handling
    retry_count: NonNegativeInt = Field(default=0, description="Current retry count")
    max_retries: NonNegativeInt = Field(default=3, description="Maximum retries")
    retry_delay: PositiveFloat = Field(default=1.0, description="Retry delay")
    continue_on_error: bool = Field(default=False, description="Continue on error")
    fallback_steps: List[str] = Field(default_factory=list, description="Fallback steps")

    # Performance
    timeout_seconds: PositiveFloat = Field(default=30.0, description="Step timeout")
    priority: PriorityEnum = Field(default=PriorityEnum.NORMAL, description="Step priority")

    # State tracking
    state: StepStateEnum = Field(default=StepStateEnum.PENDING, description="Step state")
    start_time: Optional[datetime] = Field(None, description="Step start time")
    end_time: Optional[datetime] = Field(None, description="Step end time")
    execution_time: NonNegativeFloat = Field(default=0.0, description="Execution time")
    result: Optional[Any] = Field(None, description="Step result")
    error: Optional[str] = Field(None, description="Step error")

    # Metadata
    tags: Set[str] = Field(default_factory=set, description="Step tags")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Step metadata")


class WorkflowDefinitionSchema(BaseModel, TimestampMixin):
    """Schema for workflow definitions."""

    workflow_id: str = Field(..., description="Workflow identifier")
    name: str = Field(..., description="Workflow name")
    version: str = Field(default="1.0.0", description="Workflow version")
    description: Optional[str] = Field(None, description="Workflow description")

    # Steps and flow
    steps: Dict[str, WorkflowStepSchema] = Field(default_factory=dict, description="Workflow steps")
    start_steps: List[str] = Field(default_factory=list, description="Starting steps")
    end_steps: List[str] = Field(default_factory=list, description="Ending steps")

    # Execution configuration
    execution_mode: ExecutionModeEnum = Field(
        default=ExecutionModeEnum.SEQUENTIAL, description="Execution mode"
    )
    timeout_seconds: PositiveFloat = Field(default=300.0, description="Workflow timeout")
    max_concurrent_steps: PositiveInt = Field(default=5, description="Maximum concurrent steps")

    # Error handling
    error_handling_strategy: str = Field(
        default="stop_on_error", description="Error handling strategy"
    )
    global_retry_count: NonNegativeInt = Field(default=0, description="Global retry count")
    max_global_retries: NonNegativeInt = Field(default=3, description="Maximum global retries")

    # Context and variables
    input_schema: Dict[str, Any] = Field(default_factory=dict, description="Input schema")
    output_schema: Dict[str, Any] = Field(default_factory=dict, description="Output schema")
    variables: Dict[str, Any] = Field(default_factory=dict, description="Workflow variables")

    # Metadata
    created_by: Optional[str] = Field(None, description="Workflow creator")
    tags: Set[str] = Field(default_factory=set, description="Workflow tags")
    category: Optional[str] = Field(None, description="Workflow category")


class WorkflowExecutionSchema(BaseModel, TimestampMixin):
    """Schema for workflow execution instances."""

    execution_id: str = Field(..., description="Execution identifier")
    workflow_id: str = Field(..., description="Workflow identifier")
    session_id: str = Field(..., description="Session identifier")
    user_id: Optional[str] = Field(None, description="User identifier")

    # State management
    state: WorkflowStateEnum = Field(
        default=WorkflowStateEnum.CREATED, description="Execution state"
    )
    current_steps: Set[str] = Field(default_factory=set, description="Currently executing steps")
    completed_steps: Set[str] = Field(default_factory=set, description="Completed steps")
    failed_steps: Set[str] = Field(default_factory=set, description="Failed steps")

    # Input/Output
    input_data: Dict[str, Any] = Field(default_factory=dict, description="Input data")
    output_data: Dict[str, Any] = Field(default_factory=dict, description="Output data")
    intermediate_results: Dict[str, Any] = Field(
        default_factory=dict, description="Intermediate results"
    )

    # Context and variables
    context: Dict[str, Any] = Field(default_factory=dict, description="Execution context")
    variables: Dict[str, Any] = Field(default_factory=dict, description="Execution variables")

    # Execution tracking
    start_time: Optional[datetime] = Field(None, description="Execution start time")
    end_time: Optional[datetime] = Field(None, description="Execution end time")
    execution_time: NonNegativeFloat = Field(default=0.0, description="Total execution time")
    step_executions: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict, description="Step execution details"
    )

    # Error handling
    errors: List[str] = Field(default_factory=list, description="Execution errors")
    warnings: List[str] = Field(default_factory=list, description="Execution warnings")
    retry_count: NonNegativeInt = Field(default=0, description="Retry count")

    # Performance metrics
    performance_metrics: Dict[str, float] = Field(
        default_factory=dict, description="Performance metrics"
    )
    resource_usage: Dict[str, float] = Field(default_factory=dict, description="Resource usage")

    # Priority
    priority: PriorityEnum = Field(default=PriorityEnum.NORMAL, description="Execution priority")


# =============================================================================
# Interaction Handler Schemas
# =============================================================================


class InteractionStateEnum(str, Enum):
    """Interaction states."""

    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    RESPONDING = "responding"
    WAITING_FOR_INPUT = "waiting_for_input"
    STREAMING = "streaming"
    PAUSED = "paused"
    ENDED = "ended"
    ERROR = "error"


class InteractionModeEnum(str, Enum):
    """Interaction modes."""

    CONVERSATIONAL = "conversational"
    COMMAND = "command"
    TASK_ORIENTED = "task_oriented"
    EXPLORATORY = "exploratory"
    EDUCATIONAL = "educational"
    CREATIVE = "creative"
    ANALYTICAL = "analytical"
    REAL_TIME = "real_time"


class UserProfileSchema(BaseModel, TimestampMixin):
    """Schema for user profiles."""

    user_id: str = Field(..., description="User identifier")
    username: Optional[str] = Field(None, description="Username")
    display_name: Optional[str] = Field(None, description="Display name")
    email: Optional[EmailStr] = Field(None, description="Email address")

    # Preferences
    preferred_language: str = Field(default="en", description="Preferred language")
    preferred_voice: Optional[str] = Field(None, description="Preferred voice")
    preferred_interaction_mode: InteractionModeEnum = Field(
        default=InteractionModeEnum.CONVERSATIONAL, description="Preferred interaction mode"
    )
    preferred_input_modalities: Set[ModalityEnum] = Field(
        default_factory=lambda: {ModalityEnum.TEXT}, description="Preferred input modalities"
    )
    preferred_output_modalities: Set[ModalityEnum] = Field(
        default_factory=lambda: {ModalityEnum.TEXT}, description="Preferred output modalities"
    )

    # Accessibility settings
    accessibility_needs: Dict[str, Any] = Field(
        default_factory=dict, description="Accessibility needs"
    )
    font_size_multiplier: PositiveFloat = Field(default=1.0, description="Font size multiplier")
    high_contrast_mode: bool = Field(default=False, description="High contrast mode")
    screen_reader_compatible: bool = Field(default=False, description="Screen reader compatibility")

    # Behavioral patterns
    interaction_patterns: Dict[str, Any] = Field(
        default_factory=dict, description="Interaction patterns"
    )
    learning_style: Optional[str] = Field(None, description="Learning style")
    attention_span: Optional[PositiveFloat] = Field(None, description="Attention span")

    # Security and privacy
    privacy_settings: Dict[str, Any] = Field(default_factory=dict, description="Privacy settings")
    data_retention_preferences: Dict[str, Any] = Field(
        default_factory=dict, description="Data retention preferences"
    )

    # Statistics
    last_active: Optional[datetime] = Field(None, description="Last activity time")
    total_interactions: NonNegativeInt = Field(default=0, description="Total interactions")
    is_active: bool = Field(default=True, description="Account active status")


class InteractionContextSchema(BaseModel, TimestampMixin):
    """Schema for interaction context."""

    interaction_id: str = Field(..., description="Interaction identifier")
    session_id: str = Field(..., description="Session identifier")
    user_id: Optional[str] = Field(None, description="User identifier")
    conversation_id: Optional[str] = Field(None, description="Conversation identifier")

    # Interaction metadata
    interaction_mode: InteractionModeEnum = Field(
        default=InteractionModeEnum.CONVERSATIONAL, description="Interaction mode"
    )
    priority: PriorityEnum = Field(default=PriorityEnum.NORMAL, description="Interaction priority")
    input_modalities: Set[ModalityEnum] = Field(default_factory=set, description="Input modalities")
    output_modalities: Set[ModalityEnum] = Field(
        default_factory=set, description="Output modalities"
    )

    # Timing
    start_time: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), description="Start time"
    )
    last_activity: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), description="Last activity"
    )
    timeout_seconds: PositiveFloat = Field(default=300.0, description="Timeout seconds")

    # State
    state: InteractionStateEnum = Field(
        default=InteractionStateEnum.IDLE, description="Interaction state"
    )
    is_real_time: bool = Field(default=False, description="Real-time processing")
    is_streaming: bool = Field(default=False, description="Streaming mode")

    # Context data
    conversation_history: List[Dict[str, Any]] = Field(
        default_factory=list, description="Conversation history"
    )
    current_topic: Optional[str] = Field(None, description="Current topic")
    user_intent: Optional[str] = Field(None, description="User intent")
    emotional_state: Optional[str] = Field(None, description="Emotional state")

    # Technical context
    device_info: Dict[str, Any] = Field(default_factory=dict, description="Device information")
    network_info: Dict[str, Any] = Field(default_factory=dict, description="Network information")
    location_info: Dict[str, Any] = Field(default_factory=dict, description="Location information")

    # Processing hints
    quality_preference: str = Field(default="balanced", description="Quality preference")
    latency_requirements: Dict[str, float] = Field(
        default_factory=dict, description="Latency requirements"
    )

    # Metadata
    tags: Set[str] = Field(default_factory=set, description="Context tags")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class UserMessageSchema(BaseModel, TimestampMixin):
    """Schema for user messages."""

    message_id: str = Field(..., description="Message identifier")
    user_id: Optional[str] = Field(None, description="User identifier")
    session_id: Optional[str] = Field(None, description="Session identifier")
    interaction_id: Optional[str] = Field(None, description="Interaction identifier")

    # Message content
    text: Optional[str] = Field(None, description="Text content")
    audio_data: Optional[bytes] = Field(None, description="Audio data")
    image_data: Optional[bytes] = Field(None, description="Image data")
    video_data: Optional[bytes] = Field(None, description="Video data")
    gesture_data: Optional[Dict[str, Any]] = Field(None, description="Gesture data")

    # Message metadata
    modality: ModalityEnum = Field(default=ModalityEnum.TEXT, description="Input modality")
    language: str = Field(default="en", description="Message language")
    encoding: Optional[str] = Field(None, description="Content encoding")

    # Processing hints
    intent: Optional[str] = Field(None, description="Detected intent")
    entities: List[Dict[str, Any]] = Field(default_factory=list, description="Extracted entities")
    sentiment: Optional[str] = Field(None, description="Sentiment")
    confidence: NonNegativeFloat = Field(
        default=0.0, ge=0.0, le=1.0, description="Confidence score"
    )

    # Context
    reply_to: Optional[str] = Field(None, description="Reply to message ID")
    conversation_context: Dict[str, Any] = Field(
        default_factory=dict, description="Conversation context"
    )

    # Security
    is_sanitized: bool = Field(default=False, description="Content sanitized")
    risk_level: str = Field(default="low", description="Risk level")


class AssistantResponseSchema(BaseModel, TimestampMixin):
    """Schema for assistant responses."""

    response_id: str = Field(..., description="Response identifier")
    interaction_id: str = Field(..., description="Interaction identifier")
    user_id: Optional[str] = Field(None, description="User identifier")

    # Response content
    text: Optional[str] = Field(None, description="Text response")
    audio_data: Optional[bytes] = Field(None, description="Audio data")
    image_data: Optional[bytes] = Field(None, description="Image data")
    video_data: Optional[bytes] = Field(None, description="Video data")
    visual_elements: List[Dict[str, Any]] = Field(
        default_factory=list, description="Visual elements"
    )

    # Response metadata
    modalities: Set[ModalityEnum] = Field(default_factory=set, description="Output modalities")
    language: str = Field(default="en", description="Response language")

    # Response characteristics
    response_type: str = Field(default="answer", description="Response type")
    tone: Optional[str] = Field(None, description="Response tone")
    formality_level: str = Field(default="neutral", description="Formality level")
    confidence: NonNegativeFloat = Field(
        default=0.0, ge=0.0, le=1.0, description="Confidence score"
    )

    # Processing information
    processing_time: NonNegativeFloat = Field(default=0.0, description="Processing time")
    workflow_id: Optional[str] = Field(None, description="Workflow identifier")
    execution_id: Optional[str] = Field(None, description="Execution identifier")
    component_chain: List[str] = Field(default_factory=list, description="Component chain")

    # Context and reasoning
    reasoning_trace: List[Dict[str, Any]] = Field(
        default_factory=list, description="Reasoning trace"
    )
    sources: List[str] = Field(default_factory=list, description="Information sources")
    related_topics: List[str] = Field(default_factory=list, description="Related topics")

    # Interaction management
    expects_response: bool = Field(default=False, description="Expects user response")
    suggested_follow_ups: List[str] = Field(
        default_factory=list, description="Suggested follow-ups"
    )
    interaction_hints: Dict[str, Any] = Field(default_factory=dict, description="Interaction hints")


# =============================================================================
# Session Manager Schemas
# =============================================================================


class SessionStateEnum(str, Enum):
    """Session states."""

    INITIALIZING = "initializing"
    ACTIVE = "active"
    IDLE = "idle"
    PAUSED = "paused"
    SUSPENDED = "suspended"
    EXPIRING = "expiring"
    EXPIRED = "expired"
    TERMINATED = "terminated"
    ERROR = "error"
    MIGRATING = "migrating"


class SessionTypeEnum(str, Enum):
    """Session types."""

    INTERACTIVE = "interactive"
    BATCH = "batch"
    API = "api"
    BACKGROUND = "background"
    SYSTEM = "system"
    GUEST = "guest"
    AUTHENTICATED = "authenticated"


class SessionConfigurationSchema(BaseModel):
    """Schema for session configuration."""

    session_type: SessionTypeEnum = Field(
        default=SessionTypeEnum.INTERACTIVE, description="Session type"
    )
    priority: PriorityEnum = Field(default=PriorityEnum.NORMAL, description="Session priority")
    max_idle_time: PositiveFloat = Field(default=1800.0, description="Maximum idle time")
    max_session_time: PositiveFloat = Field(default=86400.0, description="Maximum session time")
    cleanup_on_expire: bool = Field(default=True, description="Cleanup on expiration")
    persist_context: bool = Field(default=True, description="Persist context")
    enable_clustering: bool = Field(default=False, description="Enable clustering")
    enable_backup: bool = Field(default=True, description="Enable backup")
    auto_save_interval: PositiveFloat = Field(default=300.0, description="Auto-save interval")
    context_window_size: PositiveInt = Field(default=4096, description="Context window size")
    memory_limit_mb: PositiveFloat = Field(default=512.0, description="Memory limit")
    cpu_limit_percent: PositiveFloat = Field(default=50.0, description="CPU limit")
    network_timeout: PositiveFloat = Field(default=30.0, description="Network timeout")
    encryption_enabled: bool = Field(default=True, description="Encryption enabled")
    compression_enabled: bool = Field(default=True, description="Compression enabled")
    audit_logging: bool = Field(default=True, description="Audit logging")
    analytics_enabled: bool = Field(default=True, description="Analytics enabled")


class SessionContextSchema(BaseModel, TimestampMixin):
    """Schema for session context."""

    session_id: str = Field(..., description="Session identifier")
    user_id: Optional[str] = Field(None, description="User identifier")

    # Session metadata
    last_activity: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), description="Last activity"
    )
    last_heartbeat: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), description="Last heartbeat"
    )

    # User information
    user_profile: Dict[str, Any] = Field(default_factory=dict, description="User profile")
    user_preferences: Dict[str, Any] = Field(default_factory=dict, description="User preferences")
    authentication_data: Dict[str, Any] = Field(
        default_factory=dict, description="Authentication data"
    )

    # Conversation state
    conversation_history: List[Dict[str, Any]] = Field(
        default_factory=list, description="Conversation history"
    )
    current_topic: Optional[str] = Field(None, description="Current topic")
    conversation_flow: List[str] = Field(default_factory=list, description="Conversation flow")

    # Processing context
    active_workflows: Set[str] = Field(default_factory=set, description="Active workflows")
    interaction_history: List[Dict[str, Any]] = Field(
        default_factory=list, description="Interaction history"
    )
    processing_queue: List[Dict[str, Any]] = Field(
        default_factory=list, description="Processing queue"
    )

    # Memory context
    working_memory_data: Dict[str, Any] = Field(
        default_factory=dict, description="Working memory data"
    )
    episodic_memories: List[str] = Field(default_factory=list, description="Episodic memories")
    semantic_context: Dict[str, Any] = Field(default_factory=dict, description="Semantic context")

    # Environment context
    device_info: Dict[str, Any] = Field(default_factory=dict, description="Device information")
    network_info: Dict[str, Any] = Field(default_factory=dict, description="Network information")
    location_info: Dict[str, Any] = Field(default_factory=dict, description="Location information")
    timezone_info: str = Field(default="UTC", description="Timezone information")

    # Technical context
    api_keys: Dict[str, SecretStr] = Field(default_factory=dict, description="API keys")
    feature_flags: Dict[str, bool] = Field(default_factory=dict, description="Feature flags")
    experiments: Dict[str, Any] = Field(default_factory=dict, description="Experiments")

    # Performance context
    performance_metrics: Dict[str, float] = Field(
        default_factory=dict, description="Performance metrics"
    )
    resource_usage: Dict[str, float] = Field(default_factory=dict, description="Resource usage")
    quality_settings: Dict[str, str] = Field(default_factory=dict, description="Quality settings")

    # Custom data
    custom_data: Dict[str, Any] = Field(default_factory=dict, description="Custom data")
    tags: Set[str] = Field(default_factory=set, description="Session tags")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class SessionInfoSchema(BaseModel, TimestampMixin):
    """Schema for session information."""

    session_id: str = Field(..., description="Session identifier")
    state: SessionStateEnum = Field(
        default=SessionStateEnum.INITIALIZING, description="Session state"
    )
    config: SessionConfigurationSchema = Field(
        default_factory=SessionConfigurationSchema, description="Session configuration"
    )
    context: SessionContextSchema = Field(..., description="Session context")

    # Lifecycle tracking
    started_at: Optional[datetime] = Field(None, description="Session start time")
    last_activity: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), description="Last activity"
    )
    expires_at: Optional[datetime] = Field(None, description="Expiration time")

    # Resource tracking
    memory_usage_mb: NonNegativeFloat = Field(default=0.0, description="Memory usage")
    cpu_usage_percent: NonNegativeFloat = Field(default=0.0, description="CPU usage")
    network_bytes_sent: NonNegativeInt = Field(default=0, description="Network bytes sent")
    network_bytes_received: NonNegativeInt = Field(default=0, description="Network bytes received")

    # Health and performance
    health_score: NonNegativeFloat = Field(default=1.0, ge=0.0, le=1.0, description="Health score")
    response_time_avg: NonNegativeFloat = Field(default=0.0, description="Average response time")
    error_count: NonNegativeInt = Field(default=0, description="Error count")
    warning_count: NonNegativeInt = Field(default=0, description="Warning count")

    # Clustering information
    cluster_node: Optional[str] = Field(None, description="Cluster node")
    primary_node: Optional[str] = Field(None, description="Primary node")
    replica_nodes: Set[str] = Field(default_factory=set, description="Replica nodes")

    # Version and consistency
    version: PositiveInt = Field(default=1, description="Session version")
    checksum: Optional[str] = Field(None, description="Data checksum")
    last_backup: Optional[datetime] = Field(None, description="Last backup time")

    # Statistics
    interaction_count: NonNegativeInt = Field(default=0, description="Interaction count")
    workflow_count: NonNegativeInt = Field(default=0, description="Workflow count")
    message_count: NonNegativeInt = Field(default=0, description="Message count")
    total_processing_time: NonNegativeFloat = Field(
        default=0.0, description="Total processing time"
    )


# =============================================================================
# Plugin Manager Schemas
# =============================================================================


class PluginStateEnum(str, Enum):
    """Plugin states."""

    DISCOVERED = "discovered"
    VALIDATED = "validated"
    LOADED = "loaded"
    INITIALIZED = "initialized"
    ENABLED = "enabled"
    DISABLED = "disabled"
    SUSPENDED = "suspended"
    ERROR = "error"
    UNLOADED = "unloaded"
    UPDATING = "updating"
    INSTALLING = "installing"
    UNINSTALLING = "uninstalling"


class PluginTypeEnum(str, Enum):
    """Plugin types."""

    SKILL = "skill"
    PROCESSOR = "processor"
    INTEGRATION = "integration"
    UI_COMPONENT = "ui_component"
    WORKFLOW_EXTENSION = "workflow_extension"
    MEMORY_PROVIDER = "memory_provider"
    LEARNING_MODULE = "learning_module"
    SECURITY_MODULE = "security_module"
    MIDDLEWARE = "middleware"
    UTILITY = "utility"
    THEME = "theme"
    LANGUAGE_PACK = "language_pack"


class PluginLoadModeEnum(str, Enum):
    """Plugin loading modes."""

    EAGER = "eager"
    LAZY = "lazy"
    ON_DEMAND = "on_demand"
    SCHEDULED = "scheduled"


class SecurityLevelEnum(str, Enum):
    """Security levels."""

    UNTRUSTED = "untrusted"
    SANDBOX = "sandbox"
    TRUSTED = "trusted"
    SYSTEM = "system"
    VERIFIED = "verified"


class PluginDependencySchema(BaseModel):
    """Schema for plugin dependencies."""

    plugin_id: str = Field(..., description="Plugin identifier")
    version_requirement: str = Field(default="*", description="Version requirement")
    optional: bool = Field(default=False, description="Optional dependency")
    auto_install: bool = Field(default=False, description="Auto-install")
    load_order: int = Field(default=0, description="Load order priority")


class PluginCapabilitySchema(BaseModel):
    """Schema for plugin capabilities."""

    name: str = Field(..., description="Capability name")
    version: str = Field(..., description="Capability version")
    interface: str = Field(..., description="Interface type")
    description: Optional[str] = Field(None, description="Capability description")
    category: Optional[str] = Field(None, description="Capability category")


class PluginMetadataSchema(BaseModel, TimestampMixin):
    """Schema for plugin metadata."""

    plugin_id: str = Field(..., description="Plugin identifier")
    name: str = Field(..., description="Plugin name")
    version: str = Field(..., description="Plugin version")
    description: str = Field(..., description="Plugin description")
    author: str = Field(..., description="Plugin author")

    # Plugin characteristics
    plugin_type: PluginTypeEnum = Field(..., description="Plugin type")
    load_mode: PluginLoadModeEnum = Field(default=PluginLoadModeEnum.EAGER, description="Load mode")
    security_level: SecurityLevelEnum = Field(
        default=SecurityLevelEnum.UNTRUSTED, description="Security level"
    )

    # Dependencies and capabilities
    dependencies: List[PluginDependencySchema] = Field(
        default_factory=list, description="Plugin dependencies"
    )
    system_dependencies: List[str] = Field(default_factory=list, description="System dependencies")
    provides: List[PluginCapabilitySchema] = Field(
        default_factory=list, description="Provided capabilities"
    )

    # Entry points
    main_class: Optional[str] = Field(None, description="Main class")
    entry_points: Dict[str, str] = Field(default_factory=dict, description="Entry points")

    # Configuration
    config_schema: Dict[str, Any] = Field(default_factory=dict, description="Configuration schema")
    default_config: Dict[str, Any] = Field(
        default_factory=dict, description="Default configuration"
    )

    # Resource requirements
    memory_limit_mb: PositiveFloat = Field(default=256.0, description="Memory limit")
    cpu_limit_percent: PositiveFloat = Field(default=10.0, description="CPU limit")
    network_access: bool = Field(default=False, description="Network access")
    file_system_access: bool = Field(default=False, description="File system access")

    # Lifecycle hooks
    install_hooks: List[str] = Field(default_factory=list, description="Install hooks")
    uninstall_hooks: List[str] = Field(default_factory=list, description="Uninstall hooks")

    # Metadata
    homepage: Optional[HttpUrl] = Field(None, description="Plugin homepage")
    repository: Optional[HttpUrl] = Field(None, description="Repository URL")
    license: Optional[str] = Field(None, description="License")
    keywords: List[str] = Field(default_factory=list, description="Keywords")
    categories: List[str] = Field(default_factory=list, description="Categories")

    # System integration
    api_version: str = Field(default="1.0.0", description="API version")
    min_system_version: str = Field(default="1.0.0", description="Minimum system version")
    max_system_version: Optional[str] = Field(None, description="Maximum system version")

    # Security
    signature: Optional[str] = Field(None, description="Digital signature")
    checksum: Optional[str] = Field(None, description="Content checksum")
    permissions: Set[str] = Field(default_factory=set, description="Required permissions")

    # Installation info
    installation_date: Optional[datetime] = Field(None, description="Installation date")
    update_date: Optional[datetime] = Field(None, description="Update date")
    source_url: Optional[HttpUrl] = Field(None, description="Source URL")


class PluginInfoSchema(BaseModel, TimestampMixin):
    """Schema for plugin runtime information."""

    metadata: PluginMetadataSchema = Field(..., description="Plugin metadata")
    state: PluginStateEnum = Field(default=PluginStateEnum.DISCOVERED, description="Plugin state")

    # Runtime data
    installation_path: Optional[str] = Field(None, description="Installation path")

    # Performance metrics
    load_time: NonNegativeFloat = Field(default=0.0, description="Load time")
    memory_usage: NonNegativeFloat = Field(default=0.0, description="Memory usage")
    cpu_usage: NonNegativeFloat = Field(default=0.0, description="CPU usage")
    error_count: NonNegativeInt = Field(default=0, description="Error count")
    warning_count: NonNegativeInt = Field(default=0, description="Warning count")

    # Health information
    last_health_check: Optional[datetime] = Field(None, description="Last health check")
    health_status: HealthStatusEnum = Field(
        default=HealthStatusEnum.UNKNOWN, description="Health status"
    )

    # Usage statistics
    activation_count: NonNegativeInt = Field(default=0, description="Activation count")
    last_used: Optional[datetime] = Field(None, description="Last used time")
    usage_statistics: Dict[str, Any] = Field(default_factory=dict, description="Usage statistics")

    # Error tracking
    last_error: Optional[str] = Field(None, description="Last error")
    error_history: List[Dict[str, Any]] = Field(default_factory=list, description="Error history")


# =============================================================================
# Common API Schemas
# =============================================================================


class ErrorResponseSchema(BaseModel):
    """Schema for error responses."""

    error: bool = Field(default=True, description="Error flag")
    message: str = Field(..., description="Error message")
    code: Optional[str] = Field(None, description="Error code")
    details: Optional[Dict[str, Any]] = Field(None, description="Error details")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), description="Error timestamp"
    )
    request_id: Optional[str] = Field(None, description="Request identifier")
    trace_id: Optional[str] = Field(None, description="Trace identifier")


class SuccessResponseSchema(BaseModel):
    """Schema for success responses."""

    success: bool = Field(default=True, description="Success flag")
    message: Optional[str] = Field(None, description="Success message")
    data: Optional[Any] = Field(None, description="Response data")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Response metadata")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), description="Response timestamp"
    )
    request_id: Optional[str] = Field(None, description="Request identifier")


class PaginationSchema(BaseModel):
    """Schema for pagination."""

    page: PositiveInt = Field(default=1, description="Page number")
    page_size: PositiveInt = Field(default=20, le=100, description="Page size")
    total_items: NonNegativeInt = Field(..., description="Total items")
    total_pages: NonNegativeInt = Field(..., description="Total pages")
    has_next: bool = Field(..., description="Has next page")
    has_previous: bool = Field(..., description="Has previous page")


class PaginatedResponseSchema(BaseModel):
    """Schema for paginated responses."""

    items: List[Any] = Field(..., description="Response items")
    pagination: PaginationSchema = Field(..., description="Pagination information")
    filters: Optional[Dict[str, Any]] = Field(None, description="Applied filters")
    sort: Optional[Dict[str, str]] = Field(None, description="Sort criteria")


# =============================================================================
# Health Check Schemas
# =============================================================================


class HealthCheckSchema(BaseModel, TimestampMixin):
    """Schema for health checks."""

    status: HealthStatusEnum = Field(..., description="Health status")
    message: Optional[str] = Field(None, description="Health message")
    checks: Dict[str, Any] = Field(default_factory=dict, description="Individual checks")
    metrics: Dict[str, float] = Field(default_factory=dict, description="Health metrics")
    uptime_seconds: NonNegativeFloat = Field(..., description="Uptime in seconds")
    version: str = Field(..., description="System version")


class SystemStatusSchema(BaseModel, TimestampMixin):
    """Schema for system status."""

    overall_status: HealthStatusEnum = Field(..., description="Overall system status")
    components: Dict[str, HealthCheckSchema] = Field(..., description="Component health")
    system_metrics: Dict[str, float] = Field(default_factory=dict, description="System metrics")
    active_sessions: NonNegativeInt = Field(..., description="Active sessions")
    active_workflows: NonNegativeInt = Field(..., description="Active workflows")
    memory_usage_mb: NonNegativeFloat = Field(..., description="Memory usage")
    cpu_usage_percent: NonNegativeFloat = Field(..., description="CPU usage")


# =============================================================================
# Event Schemas
# =============================================================================


class EventSchema(BaseModel, TimestampMixin):
    """Base schema for events."""

    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Event identifier")
    event_type: str = Field(..., description="Event type")
    source: str = Field(..., description="Event source")
    data: Dict[str, Any] = Field(..., description="Event data")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Event metadata")
    correlation_id: Optional[str] = Field(None, description="Correlation identifier")
    causation_id: Optional[str] = Field(None, description="Causation identifier")


# =============================================================================
# Strawberry GraphQL Type Definitions
# =============================================================================


@strawberry.type
class MultimodalInput:
    """GraphQL type for multimodal input."""

    text: Optional[str] = None
    modality_weights: JSON = strawberry.field(default_factory=dict)
    processing_hints: JSON = strawberry.field(default_factory=dict)


@strawberry.type
class ProcessingResult:
    """GraphQL type for processing results."""

    success: bool
    request_id: str
    session_id: str
    processing_time: float
    response_text: Optional[str] = None
    overall_confidence: float = 0.0
    errors: List[str] = strawberry.field(default_factory=list)
    component_timings: JSON = strawberry.field(default_factory=dict)


@strawberry.type
class ComponentInfo:
    """GraphQL type for component information."""

    component_id: str
    state: str
    health_status: str
    error_count: int
    performance_metrics: JSON = strawberry.field(default_factory=dict)


@strawberry.type
class WorkflowExecution:
    """GraphQL type for workflow execution."""

    execution_id: str
    workflow_id: str
    session_id: str
    state: str
    execution_time: float
    completed_steps: List[str] = strawberry.field(default_factory=list)
    current_steps: List[str] = strawberry.field(default_factory=list)


@strawberry.type
class SessionInfo:
    """GraphQL type for session information."""

    session_id: str
    state: str
    interaction_count: int
    memory_usage_mb: float
    health_score: float
    last_activity: datetime


@strawberry.type
class PluginInfo:
    """GraphQL type for plugin information."""

    plugin_id: str
    name: str
    version: str
    state: str
    plugin_type: str
    load_time: float
    memory_usage: float


@strawberry.type
class HealthCheck:
    """GraphQL type for health checks."""

    status: str
    uptime_seconds: float
    version: str
    checks: JSON = strawberry.field(default_factory=dict)
    metrics: JSON = strawberry.field(default_factory=dict)


@strawberry.type
class ErrorResponse:
    """GraphQL type for error responses."""

    error: bool = True
    message: str
    code: Optional[str] = None
    details: Optional[JSON] = None
    timestamp: datetime


@strawberry.type
class SuccessResponse:
    """GraphQL type for success responses."""

    success: bool = True
    message: Optional[str] = None
    data: Optional[JSON] = None
    timestamp: datetime


# =============================================================================
# Schema Validation Utilities
# =============================================================================


class SchemaValidator:
    """Utility class for schema validation."""

    @staticmethod
    def validate_multimodal_input(data: Dict[str, Any]) -> MultimodalInputSchema:
        """Validate multimodal input data."""
        return MultimodalInputSchema(**data)

    @staticmethod
    def validate_processing_context(data: Dict[str, Any]) -> ProcessingContextSchema:
        """Validate processing context data."""
        return ProcessingContextSchema(**data)

    @staticmethod
    def validate_workflow_definition(data: Dict[str, Any]) -> WorkflowDefinitionSchema:
        """Validate workflow definition data."""
        return WorkflowDefinitionSchema(**data)

    @staticmethod
    def validate_session_info(data: Dict[str, Any]) -> SessionInfoSchema:
        """Validate session information data."""
        return SessionInfoSchema(**data)

    @staticmethod
    def validate_plugin_metadata(data: Dict[str, Any]) -> PluginMetadataSchema:
        """Validate plugin metadata."""
        return PluginMetadataSchema(**data)


# =============================================================================
# Schema Migration Support
# =============================================================================


class SchemaMigration:
    """Base class for schema migrations."""

    from_version: str
    to_version: str

    def migrate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate data from old schema to new schema."""
        raise NotImplementedError

    def rollback(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Rollback data from new schema to old schema."""
        raise NotImplementedError


class SchemaRegistry:
    """Registry for schema versions and migrations."""

    def __init__(self):
        self.schemas: Dict[str, Dict[str, BaseModel]] = {}
        self.migrations: List[SchemaMigration] = []
        self.current_version = "1.0.0"

    def register_schema(self, version: str, schemas: Dict[str, BaseModel]) -> None:
        """Register a schema version."""
        self.schemas[version] = schemas

    def register_migration(self, migration: SchemaMigration) -> None:
        """Register a schema migration."""
        self.migrations.append(migration)

    def get_schema(self, name: str, version: Optional[str] = None) -> Optional[BaseModel]:
        """Get a schema by name and version."""
        version = version or self.current_version
        return self.schemas.get(version, {}).get(name)

    def migrate_data(
        self, data: Dict[str, Any], from_version: str, to_version: str
    ) -> Dict[str, Any]:
        """Migrate data between schema versions."""
        current_data = data
        current_version = from_version

        while current_version != to_version:
            migration = self._find_migration(current_version, to_version)
            if not migration:
                raise ValueError(f"No migration path from {current_version} to {to_version}")

            current_data = migration.migrate(current_data)
            current_version = migration.to_version

        return current_data

    def _find_migration(self, from_version: str, to_version: str) -> Optional[SchemaMigration]:
        """Find a migration between versions."""
        for migration in self.migrations:
            if migration.from_version == from_version:
                return migration
        return None


# =============================================================================
# Export All Schemas
# =============================================================================

__all__ = [
    # Base mixins
    "TimestampMixin",
    "VersionedMixin",
    "IdentifiableMixin",
    # Enums
    "StatusEnum",
    "PriorityEnum",
    "HealthStatusEnum",
    "ProcessingModeEnum",
    "ModalityEnum",
    "ComponentStateEnum",
    "ComponentPriorityEnum",
    "DependencyTypeEnum",
    "WorkflowStateEnum",
    "StepStateEnum",
    "ExecutionModeEnum",
    "StepTypeEnum",
    "InteractionStateEnum",
    "InteractionModeEnum",
    "SessionStateEnum",
    "SessionTypeEnum",
    "PluginStateEnum",
    "PluginTypeEnum",
    "PluginLoadModeEnum",
    "SecurityLevelEnum",
    # Core Engine Schemas
    "MultimodalInputSchema",
    "ProcessingContextSchema",
    "ProcessingResultSchema",
    # Component Manager Schemas
    "ComponentDependencySchema",
    "ComponentMetadataSchema",
    "ComponentInfoSchema",
    # Workflow Orchestrator Schemas
    "WorkflowConditionSchema",
    "WorkflowStepSchema",
    "WorkflowDefinitionSchema",
    "WorkflowExecutionSchema",
    # Interaction Handler Schemas
    "UserProfileSchema",
    "InteractionContextSchema",
    "UserMessageSchema",
    "AssistantResponseSchema",
    # Session Manager Schemas
    "SessionConfigurationSchema",
    "SessionContextSchema",
    "SessionInfoSchema",
    # Plugin Manager Schemas
    "PluginDependencySchema",
    "PluginCapabilitySchema",
    "PluginMetadataSchema",
    "PluginInfoSchema",
    # Common API Schemas
    "ErrorResponseSchema",
    "SuccessResponseSchema",
    "PaginationSchema",
    "PaginatedResponseSchema",
    # Health Check Schemas
    "HealthCheckSchema",
    "SystemStatusSchema",
    # Event Schemas
    "EventSchema",
    # GraphQL Types
    "MultimodalInput",
    "ProcessingResult",
    "ComponentInfo",
    "WorkflowExecution",
    "SessionInfo",
    "PluginInfo",
    "HealthCheck",
    "ErrorResponse",
    "SuccessResponse",
    # Utilities
    "SchemaValidator",
    "SchemaMigration",
    "SchemaRegistry",
]
