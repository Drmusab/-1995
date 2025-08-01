"""
OpenAPI schemas for the AI Assistant REST API.

This module defines Pydantic models for API request/response validation
and serialization, integrating with the core system components.
"""

from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from pydantic import BaseModel, Field, root_validator, validator
from pydantic.types import Json

# =============================================================================
# Base Models
# =============================================================================


class BaseSchema(BaseModel):
    """Base schema with common configuration."""

    class Config:
        use_enum_values = True
        validate_assignment = True
        arbitrary_types_allowed = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }


class TimestampMixin(BaseModel):
    """Mixin for models with timestamp fields."""

    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None


# =============================================================================
# Core System Schemas
# =============================================================================


class HealthStatus(str, Enum):
    """Health check status enumeration."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class HealthCheckResponse(BaseSchema):
    """Health check response schema."""

    status: HealthStatus
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    version: str
    components: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    uptime: timedelta

    class Config:
        json_encoders = {
            **BaseSchema.Config.json_encoders,
            timedelta: lambda v: int(v.total_seconds()),
        }


# =============================================================================
# Session Management Schemas
# =============================================================================


class SessionStatus(str, Enum):
    """Session status enumeration."""

    ACTIVE = "active"
    IDLE = "idle"
    EXPIRED = "expired"
    TERMINATED = "terminated"


class SessionContext(BaseSchema):
    """Session context schema."""

    user_id: Optional[str] = None
    preferences: Dict[str, Any] = Field(default_factory=dict)
    capabilities: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SessionRequest(BaseSchema):
    """Session creation request schema."""

    user_id: Optional[str] = None
    session_type: str = "interactive"
    context: Optional[SessionContext] = None
    ttl_seconds: Optional[int] = Field(default=3600, ge=60, le=86400)


class SessionResponse(BaseSchema, TimestampMixin):
    """Session response schema."""

    session_id: UUID
    status: SessionStatus
    context: SessionContext
    expires_at: Optional[datetime] = None


# =============================================================================
# Interaction Schemas
# =============================================================================


class MessageType(str, Enum):
    """Message type enumeration."""

    TEXT = "text"
    AUDIO = "audio"
    IMAGE = "image"
    VIDEO = "video"
    DOCUMENT = "document"
    MULTIMODAL = "multimodal"


class InteractionMode(str, Enum):
    """Interaction mode enumeration."""

    CHAT = "chat"
    VOICE = "voice"
    VISION = "vision"
    MULTIMODAL = "multimodal"


class MessageContent(BaseSchema):
    """Message content schema."""

    type: MessageType
    text: Optional[str] = None
    audio_url: Optional[str] = None
    image_url: Optional[str] = None
    video_url: Optional[str] = None
    document_url: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @root_validator
    def validate_content(cls, values):
        """Validate that appropriate content is provided for the message type."""
        msg_type = values.get("type")
        if msg_type == MessageType.TEXT and not values.get("text"):
            raise ValueError("Text content required for text messages")
        elif msg_type == MessageType.AUDIO and not values.get("audio_url"):
            raise ValueError("Audio URL required for audio messages")
        elif msg_type == MessageType.IMAGE and not values.get("image_url"):
            raise ValueError("Image URL required for image messages")
        return values


class InteractionRequest(BaseSchema):
    """Interaction request schema."""

    session_id: UUID
    content: MessageContent
    mode: InteractionMode = InteractionMode.CHAT
    context: Optional[Dict[str, Any]] = Field(default_factory=dict)
    stream: bool = False
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=None, ge=1, le=8192)


class InteractionResponse(BaseSchema, TimestampMixin):
    """Interaction response schema."""

    interaction_id: UUID
    session_id: UUID
    request: MessageContent
    response: MessageContent
    processing_time_ms: int
    confidence_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)


# =============================================================================
# Memory System Schemas
# =============================================================================


class MemoryType(str, Enum):
    """Memory type enumeration."""

    WORKING = "working"
    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"


class MemoryEntry(BaseSchema, TimestampMixin):
    """Memory entry schema."""

    memory_id: UUID
    type: MemoryType
    content: Dict[str, Any]
    importance: float = Field(ge=0.0, le=1.0)
    decay_rate: float = Field(default=0.1, ge=0.0, le=1.0)
    associations: List[UUID] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MemoryQuery(BaseSchema):
    """Memory query schema."""

    query: str
    memory_types: Optional[List[MemoryType]] = None
    tags: Optional[List[str]] = None
    limit: int = Field(default=10, ge=1, le=100)
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)


class MemoryQueryResponse(BaseSchema):
    """Memory query response schema."""

    results: List[MemoryEntry]
    total_count: int
    query_time_ms: int


# =============================================================================
# Skills System Schemas
# =============================================================================


class SkillType(str, Enum):
    """Skill type enumeration."""

    BUILTIN = "builtin"
    CUSTOM = "custom"
    EXTERNAL = "external"
    COMPOSED = "composed"


class SkillStatus(str, Enum):
    """Skill status enumeration."""

    ACTIVE = "active"
    INACTIVE = "inactive"
    DEPRECATED = "deprecated"
    EXPERIMENTAL = "experimental"


class SkillParameter(BaseSchema):
    """Skill parameter schema."""

    name: str
    type: str
    description: str
    required: bool = False
    default_value: Optional[Any] = None
    validation_rules: Optional[Dict[str, Any]] = None


class SkillDefinition(BaseSchema, TimestampMixin):
    """Skill definition schema."""

    skill_id: str
    name: str
    description: str
    version: str
    type: SkillType
    status: SkillStatus
    parameters: List[SkillParameter] = Field(default_factory=list)
    dependencies: List[str] = Field(default_factory=list)
    capabilities: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SkillExecutionRequest(BaseSchema):
    """Skill execution request schema."""

    skill_id: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    context: Optional[Dict[str, Any]] = Field(default_factory=dict)
    timeout_seconds: Optional[int] = Field(default=30, ge=1, le=300)


class SkillExecutionResponse(BaseSchema):
    """Skill execution response schema."""

    execution_id: UUID
    skill_id: str
    status: str
    result: Optional[Any] = None
    error: Optional[str] = None
    execution_time_ms: int
    metadata: Dict[str, Any] = Field(default_factory=dict)


# =============================================================================
# Workflow Orchestration Schemas
# =============================================================================


class WorkflowStatus(str, Enum):
    """Workflow status enumeration."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class WorkflowStep(BaseSchema):
    """Workflow step schema."""

    step_id: str
    skill_id: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    dependencies: List[str] = Field(default_factory=list)
    retry_count: int = Field(default=0, ge=0, le=5)
    timeout_seconds: int = Field(default=30, ge=1, le=300)


class WorkflowDefinition(BaseSchema, TimestampMixin):
    """Workflow definition schema."""

    workflow_id: str
    name: str
    description: str
    version: str
    steps: List[WorkflowStep]
    metadata: Dict[str, Any] = Field(default_factory=dict)


class WorkflowExecutionRequest(BaseSchema):
    """Workflow execution request schema."""

    workflow_id: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    context: Optional[Dict[str, Any]] = Field(default_factory=dict)


class WorkflowExecutionResponse(BaseSchema, TimestampMixin):
    """Workflow execution response schema."""

    execution_id: UUID
    workflow_id: str
    status: WorkflowStatus
    current_step: Optional[str] = None
    completed_steps: List[str] = Field(default_factory=list)
    failed_steps: List[str] = Field(default_factory=list)
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


# =============================================================================
# Plugin System Schemas
# =============================================================================


class PluginType(str, Enum):
    """Plugin type enumeration."""

    PROCESSOR = "processor"
    INTEGRATOR = "integrator"
    ENHANCER = "enhancer"
    ANALYZER = "analyzer"


class PluginStatus(str, Enum):
    """Plugin status enumeration."""

    INSTALLED = "installed"
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    UPDATING = "updating"


class PluginInfo(BaseSchema, TimestampMixin):
    """Plugin information schema."""

    plugin_id: str
    name: str
    description: str
    version: str
    author: str
    type: PluginType
    status: PluginStatus
    dependencies: List[str] = Field(default_factory=list)
    permissions: List[str] = Field(default_factory=list)
    configuration: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PluginInstallRequest(BaseSchema):
    """Plugin installation request schema."""

    plugin_id: str
    source: str  # URL, file path, or registry reference
    configuration: Optional[Dict[str, Any]] = Field(default_factory=dict)
    auto_activate: bool = True


class PluginInstallResponse(BaseSchema):
    """Plugin installation response schema."""

    plugin_id: str
    status: str
    message: str
    installed_version: Optional[str] = None


# =============================================================================
# Processing Schemas
# =============================================================================


class ProcessingMode(str, Enum):
    """Processing mode enumeration."""

    SYNC = "sync"
    ASYNC = "async"
    STREAM = "stream"
    BATCH = "batch"


class ProcessingRequest(BaseSchema):
    """Processing request schema."""

    input_data: Union[str, Dict[str, Any], List[Any]]
    processing_type: str
    mode: ProcessingMode = ProcessingMode.SYNC
    options: Dict[str, Any] = Field(default_factory=dict)
    timeout_seconds: Optional[int] = Field(default=60, ge=1, le=600)


class ProcessingResponse(BaseSchema):
    """Processing response schema."""

    processing_id: UUID
    status: str
    result: Optional[Any] = None
    error: Optional[str] = None
    processing_time_ms: int
    metadata: Dict[str, Any] = Field(default_factory=dict)


# =============================================================================
# Learning System Schemas
# =============================================================================


class LearningType(str, Enum):
    """Learning type enumeration."""

    SUPERVISED = "supervised"
    UNSUPERVISED = "unsupervised"
    REINFORCEMENT = "reinforcement"
    CONTINUAL = "continual"
    PREFERENCE = "preference"


class FeedbackType(str, Enum):
    """Feedback type enumeration."""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    CORRECTION = "correction"


class LearningData(BaseSchema):
    """Learning data schema."""

    input_data: Any
    target_output: Optional[Any] = None
    feedback: Optional[str] = None
    feedback_type: Optional[FeedbackType] = None
    context: Dict[str, Any] = Field(default_factory=dict)


class LearningRequest(BaseSchema):
    """Learning request schema."""

    learning_type: LearningType
    data: List[LearningData]
    model_id: Optional[str] = None
    configuration: Dict[str, Any] = Field(default_factory=dict)


class LearningResponse(BaseSchema):
    """Learning response schema."""

    learning_id: UUID
    status: str
    model_id: Optional[str] = None
    metrics: Dict[str, float] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


# =============================================================================
# Integration Schemas
# =============================================================================


class IntegrationType(str, Enum):
    """Integration type enumeration."""

    LLM = "llm"
    CACHE = "cache"
    STORAGE = "storage"
    EXTERNAL_API = "external_api"


class IntegrationStatus(str, Enum):
    """Integration status enumeration."""

    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    ERROR = "error"
    AUTHENTICATING = "authenticating"


class IntegrationConfig(BaseSchema):
    """Integration configuration schema."""

    integration_id: str
    type: IntegrationType
    name: str
    configuration: Dict[str, Any]
    credentials: Optional[Dict[str, str]] = None  # Sensitive data
    status: IntegrationStatus = IntegrationStatus.DISCONNECTED
    metadata: Dict[str, Any] = Field(default_factory=dict)


class IntegrationTestRequest(BaseSchema):
    """Integration test request schema."""

    integration_id: str
    test_type: str = "connectivity"
    parameters: Dict[str, Any] = Field(default_factory=dict)


class IntegrationTestResponse(BaseSchema):
    """Integration test response schema."""

    integration_id: str
    test_type: str
    success: bool
    message: str
    details: Dict[str, Any] = Field(default_factory=dict)
    test_time_ms: int


# =============================================================================
# Monitoring & Observability Schemas
# =============================================================================


class MetricType(str, Enum):
    """Metric type enumeration."""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class MetricData(BaseSchema):
    """Metric data schema."""

    name: str
    type: MetricType
    value: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    labels: Dict[str, str] = Field(default_factory=dict)
    description: Optional[str] = None


class LogLevel(str, Enum):
    """Log level enumeration."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class LogEntry(BaseSchema):
    """Log entry schema."""

    timestamp: datetime = Field(default_factory=datetime.utcnow)
    level: LogLevel
    message: str
    component: str
    session_id: Optional[UUID] = None
    user_id: Optional[str] = None
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


# =============================================================================
# Error Handling Schemas
# =============================================================================


class ErrorType(str, Enum):
    """Error type enumeration."""

    VALIDATION_ERROR = "validation_error"
    AUTHENTICATION_ERROR = "authentication_error"
    AUTHORIZATION_ERROR = "authorization_error"
    NOT_FOUND_ERROR = "not_found_error"
    CONFLICT_ERROR = "conflict_error"
    RATE_LIMIT_ERROR = "rate_limit_error"
    INTERNAL_ERROR = "internal_error"
    EXTERNAL_SERVICE_ERROR = "external_service_error"
    TIMEOUT_ERROR = "timeout_error"


class ErrorDetail(BaseSchema):
    """Error detail schema."""

    field: Optional[str] = None
    message: str
    code: Optional[str] = None


class ErrorResponse(BaseSchema):
    """Error response schema."""

    error_type: ErrorType
    message: str
    details: List[ErrorDetail] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    trace_id: Optional[str] = None
    request_id: Optional[str] = None


# =============================================================================
# Pagination & Filtering Schemas
# =============================================================================


class SortOrder(str, Enum):
    """Sort order enumeration."""

    ASC = "asc"
    DESC = "desc"


class PaginationRequest(BaseSchema):
    """Pagination request schema."""

    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=20, ge=1, le=100)
    sort_by: Optional[str] = None
    sort_order: SortOrder = SortOrder.DESC


class PaginationResponse(BaseSchema):
    """Pagination response schema."""

    page: int
    page_size: int
    total_count: int
    total_pages: int
    has_next: bool
    has_previous: bool


class FilterRequest(BaseSchema):
    """Filter request schema."""

    filters: Dict[str, Any] = Field(default_factory=dict)
    search_query: Optional[str] = None
    date_range: Optional[Dict[str, datetime]] = None


# =============================================================================
# Batch Operation Schemas
# =============================================================================


class BatchOperationType(str, Enum):
    """Batch operation type enumeration."""

    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    PROCESS = "process"


class BatchOperation(BaseSchema):
    """Batch operation schema."""

    operation_type: BatchOperationType
    data: Any
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BatchRequest(BaseSchema):
    """Batch request schema."""

    operations: List[BatchOperation]
    fail_on_error: bool = False
    timeout_seconds: Optional[int] = Field(default=300, ge=1, le=3600)


class BatchResult(BaseSchema):
    """Batch result schema."""

    operation_index: int
    success: bool
    result: Optional[Any] = None
    error: Optional[str] = None


class BatchResponse(BaseSchema):
    """Batch response schema."""

    batch_id: UUID
    total_operations: int
    successful_operations: int
    failed_operations: int
    results: List[BatchResult]
    processing_time_ms: int
    metadata: Dict[str, Any] = Field(default_factory=dict)


# =============================================================================
# Export Models (for easy importing)
# =============================================================================

__all__ = [
    # Base models
    "BaseSchema",
    "TimestampMixin",
    # Core system
    "HealthStatus",
    "HealthCheckResponse",
    # Session management
    "SessionStatus",
    "SessionContext",
    "SessionRequest",
    "SessionResponse",
    # Interactions
    "MessageType",
    "InteractionMode",
    "MessageContent",
    "InteractionRequest",
    "InteractionResponse",
    # Memory system
    "MemoryType",
    "MemoryEntry",
    "MemoryQuery",
    "MemoryQueryResponse",
    # Skills system
    "SkillType",
    "SkillStatus",
    "SkillParameter",
    "SkillDefinition",
    "SkillExecutionRequest",
    "SkillExecutionResponse",
    # Workflow orchestration
    "WorkflowStatus",
    "WorkflowStep",
    "WorkflowDefinition",
    "WorkflowExecutionRequest",
    "WorkflowExecutionResponse",
    # Plugin system
    "PluginType",
    "PluginStatus",
    "PluginInfo",
    "PluginInstallRequest",
    "PluginInstallResponse",
    # Processing
    "ProcessingMode",
    "ProcessingRequest",
    "ProcessingResponse",
    # Learning system
    "LearningType",
    "FeedbackType",
    "LearningData",
    "LearningRequest",
    "LearningResponse",
    # Integrations
    "IntegrationType",
    "IntegrationStatus",
    "IntegrationConfig",
    "IntegrationTestRequest",
    "IntegrationTestResponse",
    # Monitoring & observability
    "MetricType",
    "MetricData",
    "LogLevel",
    "LogEntry",
    # Error handling
    "ErrorType",
    "ErrorDetail",
    "ErrorResponse",
    # Pagination & filtering
    "SortOrder",
    "PaginationRequest",
    "PaginationResponse",
    "FilterRequest",
    # Batch operations
    "BatchOperationType",
    "BatchOperation",
    "BatchRequest",
    "BatchResult",
    "BatchResponse",
]
