"""
Comprehensive Event Type Definitions for AI Assistant System
Author: Drmusab
Last Modified: 2025-06-13 10:45:25 UTC

This module defines all event types used throughout the AI assistant system,
providing a centralized registry of events for component communication,
monitoring, and system coordination.
"""

from datetime import datetime, timezone
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass, field
from enum import Enum
import uuid


class EventCategory(Enum):
    """Categories of events in the system."""
    SYSTEM = "system"
    COMPONENT = "component"
    WORKFLOW = "workflow"
    SESSION = "session"
    USER = "user"
    PROCESSING = "processing"
    MEMORY = "memory"
    LEARNING = "learning"
    SECURITY = "security"
    PERFORMANCE = "performance"
    ERROR = "error"
    PLUGIN = "plugin"
    SKILL = "skill"
    INTEGRATION = "integration"
    API = "api"
    HEALTH = "health"
    AUDIT = "audit"


class EventPriority(Enum):
    """Event priority levels."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3
    EMERGENCY = 4


class EventSeverity(Enum):
    """Event severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    FATAL = "fatal"


@dataclass
class BaseEvent:
    """Base class for all system events."""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: str = field(init=False)
    category: EventCategory = field(init=False)
    priority: EventPriority = EventPriority.NORMAL
    severity: EventSeverity = EventSeverity.INFO
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    source_component: Optional[str] = None
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    correlation_id: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not hasattr(self, 'event_type'):
            self.event_type = self.__class__.__name__
        if not hasattr(self, 'category'):
            self.category = self._infer_category()
    
    def _infer_category(self) -> EventCategory:
        """Infer event category from event type."""
        event_name = self.__class__.__name__.lower()
        
        if any(x in event_name for x in ['system', 'startup', 'shutdown']):
            return EventCategory.SYSTEM
        elif any(x in event_name for x in ['component', 'health']):
            return EventCategory.COMPONENT
        elif any(x in event_name for x in ['workflow', 'step', 'task']):
            return EventCategory.WORKFLOW
        elif any(x in event_name for x in ['session', 'conversation']):
            return EventCategory.SESSION
        elif any(x in event_name for x in ['user', 'interaction']):
            return EventCategory.USER
        elif any(x in event_name for x in ['processing', 'modality', 'fusion']):
            return EventCategory.PROCESSING
        elif any(x in event_name for x in ['memory', 'retrieval', 'consolidation']):
            return EventCategory.MEMORY
        elif any(x in event_name for x in ['learning', 'adaptation', 'feedback']):
            return EventCategory.LEARNING
        elif any(x in event_name for x in ['auth', 'security', 'violation']):
            return EventCategory.SECURITY
        elif any(x in event_name for x in ['performance', 'metrics', 'monitoring']):
            return EventCategory.PERFORMANCE
        elif any(x in event_name for x in ['error', 'exception', 'failure']):
            return EventCategory.ERROR
        elif any(x in event_name for x in ['plugin']):
            return EventCategory.PLUGIN
        elif any(x in event_name for x in ['skill']):
            return EventCategory.SKILL
        else:
            return EventCategory.SYSTEM


# =============================================================================
# SYSTEM EVENTS
# =============================================================================

@dataclass
class SystemStarted(BaseEvent):
    """System startup completed event."""
    category: EventCategory = field(default=EventCategory.SYSTEM, init=False)
    startup_time: float = 0.0
    components_loaded: int = 0
    version: str = "1.0.0"
    environment: str = "development"
    node_id: Optional[str] = None


@dataclass
class SystemShutdownStarted(BaseEvent):
    """System shutdown initiated event."""
    category: EventCategory = field(default=EventCategory.SYSTEM, init=False)
    priority: EventPriority = EventPriority.HIGH
    reason: str = "manual"
    graceful: bool = True


@dataclass
class SystemShutdownCompleted(BaseEvent):
    """System shutdown completed event."""
    category: EventCategory = field(default=EventCategory.SYSTEM, init=False)
    priority: EventPriority = EventPriority.HIGH
    shutdown_time: float = 0.0
    components_stopped: int = 0


@dataclass
class SystemStateChanged(BaseEvent):
    """System state change event."""
    category: EventCategory = field(default=EventCategory.SYSTEM, init=False)
    old_state: str = ""
    new_state: str = ""
    reason: Optional[str] = None


@dataclass
class SystemResourceAlert(BaseEvent):
    """System resource usage alert."""
    category: EventCategory = field(default=EventCategory.SYSTEM, init=False)
    priority: EventPriority = EventPriority.HIGH
    severity: EventSeverity = EventSeverity.WARNING
    resource_type: str = ""  # cpu, memory, disk, network
    current_usage: float = 0.0
    threshold: float = 0.0
    unit: str = ""


# =============================================================================
# COMPONENT EVENTS
# =============================================================================

@dataclass
class ComponentRegistered(BaseEvent):
    """Component registration event."""
    category: EventCategory = field(default=EventCategory.COMPONENT, init=False)
    component_id: str = ""
    component_type: str = ""
    priority: int = 0


@dataclass
class ComponentInitialized(BaseEvent):
    """Component initialization completed event."""
    category: EventCategory = field(default=EventCategory.COMPONENT, init=False)
    component_id: str = ""
    initialization_time: float = 0.0


@dataclass
class ComponentStarted(BaseEvent):
    """Component started event."""
    category: EventCategory = field(default=EventCategory.COMPONENT, init=False)
    component_id: str = ""


@dataclass
class ComponentStopped(BaseEvent):
    """Component stopped event."""
    category: EventCategory = field(default=EventCategory.COMPONENT, init=False)
    component_id: str = ""
    reason: Optional[str] = None


@dataclass
class ComponentFailed(BaseEvent):
    """Component failure event."""
    category: EventCategory = field(default=EventCategory.COMPONENT, init=False)
    priority: EventPriority = EventPriority.HIGH
    severity: EventSeverity = EventSeverity.ERROR
    component_id: str = ""
    error_message: str = ""
    error_type: str = ""


@dataclass
class ComponentHealthChanged(BaseEvent):
    """Component health status change event."""
    category: EventCategory = field(default=EventCategory.COMPONENT, init=False)
    component: str = ""
    healthy: bool = True
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DependencyResolved(BaseEvent):
    """Dependency resolution event."""
    category: EventCategory = field(default=EventCategory.COMPONENT, init=False)
    component_id: str = ""
    dependency_id: str = ""
    resolution_time: float = 0.0


# =============================================================================
# ENGINE EVENTS
# =============================================================================

@dataclass
class EngineStarted(BaseEvent):
    """Core engine started event."""
    category: EventCategory = field(default=EventCategory.SYSTEM, init=False)
    engine_id: int = 0
    version: str = "1.0.0"
    components_loaded: int = 0
    startup_time: Optional[datetime] = None


@dataclass
class EngineShutdown(BaseEvent):
    """Core engine shutdown event."""
    category: EventCategory = field(default=EventCategory.SYSTEM, init=False)
    priority: EventPriority = EventPriority.HIGH
    engine_id: int = 0
    shutdown_time: Optional[datetime] = None
    uptime_seconds: float = 0.0


# =============================================================================
# WORKFLOW EVENTS
# =============================================================================

@dataclass
class WorkflowStarted(BaseEvent):
    """Workflow execution started event."""
    category: EventCategory = field(default=EventCategory.WORKFLOW, init=False)
    workflow_id: str = ""
    execution_id: str = ""
    workflow_name: str = ""


@dataclass
class WorkflowCompleted(BaseEvent):
    """Workflow execution completed event."""
    category: EventCategory = field(default=EventCategory.WORKFLOW, init=False)
    workflow_id: str = ""
    execution_id: str = ""
    execution_time: float = 0.0
    steps_completed: int = 0


@dataclass
class WorkflowFailed(BaseEvent):
    """Workflow execution failed event."""
    category: EventCategory = field(default=EventCategory.WORKFLOW, init=False)
    priority: EventPriority = EventPriority.HIGH
    severity: EventSeverity = EventSeverity.ERROR
    workflow_id: str = ""
    execution_id: str = ""
    error_message: str = ""
    execution_time: float = 0.0


@dataclass
class WorkflowPaused(BaseEvent):
    """Workflow execution paused event."""
    category: EventCategory = field(default=EventCategory.WORKFLOW, init=False)
    workflow_id: str = ""
    execution_id: str = ""


@dataclass
class WorkflowResumed(BaseEvent):
    """Workflow execution resumed event."""
    category: EventCategory = field(default=EventCategory.WORKFLOW, init=False)
    workflow_id: str = ""
    execution_id: str = ""


@dataclass
class WorkflowCancelled(BaseEvent):
    """Workflow execution cancelled event."""
    category: EventCategory = field(default=EventCategory.WORKFLOW, init=False)
    workflow_id: str = ""
    execution_id: str = ""


@dataclass
class WorkflowStepStarted(BaseEvent):
    """Workflow step started event."""
    category: EventCategory = field(default=EventCategory.WORKFLOW, init=False)
    workflow_id: str = ""
    execution_id: str = ""
    step_id: str = ""
    step_name: str = ""
    step_type: str = ""


@dataclass
class WorkflowStepCompleted(BaseEvent):
    """Workflow step completed event."""
    category: EventCategory = field(default=EventCategory.WORKFLOW, init=False)
    workflow_id: str = ""
    execution_id: str = ""
    step_id: str = ""
    step_name: str = ""
    execution_time: float = 0.0
    success: bool = True


@dataclass
class WorkflowStepFailed(BaseEvent):
    """Workflow step failed event."""
    category: EventCategory = field(default=EventCategory.WORKFLOW, init=False)
    priority: EventPriority = EventPriority.HIGH
    severity: EventSeverity = EventSeverity.ERROR
    workflow_id: str = ""
    execution_id: str = ""
    step_id: str = ""
    error_message: str = ""
    error_type: str = ""


@dataclass
class WorkflowStepSkipped(BaseEvent):
    """Workflow step skipped event."""
    category: EventCategory = field(default=EventCategory.WORKFLOW, init=False)
    workflow_id: str = ""
    execution_id: str = ""
    step_id: str = ""
    reason: str = ""


@dataclass
class WorkflowBranchingOccurred(BaseEvent):
    """Workflow branching event."""
    category: EventCategory = field(default=EventCategory.WORKFLOW, init=False)
    workflow_id: str = ""
    execution_id: str = ""
    branch_point: str = ""
    selected_branches: List[str] = field(default_factory=list)


@dataclass
class WorkflowMerged(BaseEvent):
    """Workflow merge event."""
    category: EventCategory = field(default=EventCategory.WORKFLOW, init=False)
    workflow_id: str = ""
    execution_id: str = ""
    merge_point: str = ""


@dataclass
class WorkflowAdapted(BaseEvent):
    """Workflow adaptation event."""
    category: EventCategory = field(default=EventCategory.WORKFLOW, init=False)
    workflow_id: str = ""
    execution_id: str = ""
    adaptation_reason: str = ""
    changes_made: List[str] = field(default_factory=list)


# =============================================================================
# SESSION EVENTS
# =============================================================================

@dataclass
class SessionStarted(BaseEvent):
    """Session started event."""
    category: EventCategory = field(default=EventCategory.SESSION, init=False)
    created_at: Optional[datetime] = None


@dataclass
class SessionEnded(BaseEvent):
    """Session ended event."""
    category: EventCategory = field(default=EventCategory.SESSION, init=False)
    duration: float = 0.0
    interaction_count: int = 0
    reason: str = "completed"


@dataclass
class SessionExpired(BaseEvent):
    """Session expired event."""
    category: EventCategory = field(default=EventCategory.SESSION, init=False)
    priority: EventPriority = EventPriority.NORMAL
    duration: float = 0.0


@dataclass
class SessionRestored(BaseEvent):
    """Session restored event."""
    category: EventCategory = field(default=EventCategory.SESSION, init=False)
    restored_from: str = ""  # backup, cache, etc.


@dataclass
class SessionContextUpdated(BaseEvent):
    """Session context updated event."""
    category: EventCategory = field(default=EventCategory.SESSION, init=False)
    updates: List[str] = field(default_factory=list)


@dataclass
class SessionStateChanged(BaseEvent):
    """Session state change event."""
    category: EventCategory = field(default=EventCategory.SESSION, init=False)
    old_state: str = ""
    new_state: str = ""


@dataclass
class SessionCleanupStarted(BaseEvent):
    """Session cleanup started event."""
    category: EventCategory = field(default=EventCategory.SESSION, init=False)
    cleanup_type: str = "expired"


@dataclass
class SessionCleanupCompleted(BaseEvent):
    """Session cleanup completed event."""
    category: EventCategory = field(default=EventCategory.SESSION, init=False)
    sessions_cleaned: int = 0
    cleanup_time: float = 0.0


@dataclass
class SessionMigrated(BaseEvent):
    """Session migrated event."""
    category: EventCategory = field(default=EventCategory.SESSION, init=False)
    from_node: str = ""
    to_node: str = ""
    migration_time: float = 0.0


@dataclass
class SessionClusteringStarted(BaseEvent):
    """Session clustering started event."""
    category: EventCategory = field(default=EventCategory.SESSION, init=False)
    cluster_strategy: str = ""


@dataclass
class SessionHealthCheckFailed(BaseEvent):
    """Session health check failed event."""
    category: EventCategory = field(default=EventCategory.SESSION, init=False)
    priority: EventPriority = EventPriority.HIGH
    severity: EventSeverity = EventSeverity.WARNING
    health_score: float = 0.0


# =============================================================================
# USER AND INTERACTION EVENTS
# =============================================================================

@dataclass
class UserInteractionStarted(BaseEvent):
    """User interaction started event."""
    category: EventCategory = field(default=EventCategory.USER, init=False)
    interaction_id: str = ""
    interaction_mode: str = ""
    input_modalities: List[str] = field(default_factory=list)
    output_modalities: List[str] = field(default_factory=list)


@dataclass
class UserInteractionCompleted(BaseEvent):
    """User interaction completed event."""
    category: EventCategory = field(default=EventCategory.USER, init=False)
    interaction_id: str = ""
    duration: float = 0.0
    message_count: int = 0
    reason: str = "completed"


@dataclass
class UserInteractionFailed(BaseEvent):
    """User interaction failed event."""
    category: EventCategory = field(default=EventCategory.USER, init=False)
    priority: EventPriority = EventPriority.HIGH
    severity: EventSeverity = EventSeverity.ERROR
    interaction_id: str = ""
    error_message: str = ""
    error_type: str = ""


@dataclass
class ConversationStarted(BaseEvent):
    """Conversation started event."""
    category: EventCategory = field(default=EventCategory.USER, init=False)
    conversation_id: str = ""


@dataclass
class ConversationEnded(BaseEvent):
    """Conversation ended event."""
    category: EventCategory = field(default=EventCategory.USER, init=False)
    conversation_id: str = ""
    duration: float = 0.0
    message_count: int = 0


@dataclass
class MessageReceived(BaseEvent):
    """Message received event."""
    category: EventCategory = field(default=EventCategory.USER, init=False)
    message_id: str = ""
    interaction_id: str = ""
    modality: str = ""
    content_preview: Optional[str] = None


@dataclass
class MessageSent(BaseEvent):
    """Message sent event."""
    category: EventCategory = field(default=EventCategory.USER, init=False)
    message_id: str = ""
    interaction_id: str = ""
    modalities: List[str] = field(default_factory=list)


@dataclass
class MessageProcessed(BaseEvent):
    """Message processed event."""
    category: EventCategory = field(default=EventCategory.USER, init=False)
    message_id: str = ""
    interaction_id: str = ""
    processing_time: float = 0.0
    confidence: float = 0.0
    success: bool = True


@dataclass
class UserJoinedSession(BaseEvent):
    """User joined session event."""
    category: EventCategory = field(default=EventCategory.USER, init=False)


@dataclass
class UserLeftSession(BaseEvent):
    """User left session event."""
    category: EventCategory = field(default=EventCategory.USER, init=False)
    session_duration: float = 0.0


@dataclass
class UserAuthenticated(BaseEvent):
    """User authentication event."""
    category: EventCategory = field(default=EventCategory.SECURITY, init=False)
    auth_method: str = ""
    success: bool = True


@dataclass
class UserAuthorized(BaseEvent):
    """User authorization event."""
    category: EventCategory = field(default=EventCategory.SECURITY, init=False)
    resource: str = ""
    action: str = ""
    success: bool = True


@dataclass
class UserLoggedOut(BaseEvent):
    """User logout event."""
    category: EventCategory = field(default=EventCategory.SECURITY, init=False)
    session_duration: float = 0.0


@dataclass
class UserPreferenceUpdated(BaseEvent):
    """User preference updated event."""
    category: EventCategory = field(default=EventCategory.USER, init=False)
    preferences: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# PROCESSING EVENTS
# =============================================================================

@dataclass
class ProcessingStarted(BaseEvent):
    """Processing started event."""
    category: EventCategory = field(default=EventCategory.PROCESSING, init=False)
    request_id: str = ""
    input_modalities: List[str] = field(default_factory=list)


@dataclass
class ProcessingCompleted(BaseEvent):
    """Processing completed event."""
    category: EventCategory = field(default=EventCategory.PROCESSING, init=False)
    request_id: str = ""
    processing_time: float = 0.0
    success: bool = True
    confidence: float = 0.0


@dataclass
class ProcessingError(BaseEvent):
    """Processing error event."""
    category: EventCategory = field(default=EventCategory.PROCESSING, init=False)
    priority: EventPriority = EventPriority.HIGH
    severity: EventSeverity = EventSeverity.ERROR
    request_id: str = ""
    error_type: str = ""
    error_message: str = ""


@dataclass
class ModalityProcessingStarted(BaseEvent):
    """Modality processing started event."""
    category: EventCategory = field(default=EventCategory.PROCESSING, init=False)
    request_id: str = ""
    modality: str = ""


@dataclass
class ModalityProcessingCompleted(BaseEvent):
    """Modality processing completed event."""
    category: EventCategory = field(default=EventCategory.PROCESSING, init=False)
    request_id: str = ""
    modality: str = ""
    success: bool = True


@dataclass
class ModalityDetected(BaseEvent):
    """Modality detection event."""
    category: EventCategory = field(default=EventCategory.PROCESSING, init=False)
    interaction_id: str = ""
    detected_modalities: List[str] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class FusionStarted(BaseEvent):
    """Multimodal fusion started event."""
    category: EventCategory = field(default=EventCategory.PROCESSING, init=False)
    request_id: str = ""
    modalities: List[str] = field(default_factory=list)


@dataclass
class FusionCompleted(BaseEvent):
    """Multimodal fusion completed event."""
    category: EventCategory = field(default=EventCategory.PROCESSING, init=False)
    request_id: str = ""
    fusion_confidence: float = 0.0


@dataclass
class InteractionModeChanged(BaseEvent):
    """Interaction mode changed event."""
    category: EventCategory = field(default=EventCategory.PROCESSING, init=False)
    interaction_id: str = ""
    old_mode: str = ""
    new_mode: str = ""


@dataclass
class RealTimeProcessingStarted(BaseEvent):
    """Real-time processing started event."""
    category: EventCategory = field(default=EventCategory.PROCESSING, init=False)
    interaction_id: str = ""


@dataclass
class RealTimeProcessingCompleted(BaseEvent):
    """Real-time processing completed event."""
    category: EventCategory = field(default=EventCategory.PROCESSING, init=False)
    interaction_id: str = ""
    processing_time: float = 0.0


@dataclass
class StreamingStarted(BaseEvent):
    """Streaming started event."""
    category: EventCategory = field(default=EventCategory.PROCESSING, init=False)
    interaction_id: str = ""


@dataclass
class StreamingCompleted(BaseEvent):
    """Streaming completed event."""
    category: EventCategory = field(default=EventCategory.PROCESSING, init=False)
    interaction_id: str = ""
    total_chunks: int = 0
    final_response_id: str = ""


# =============================================================================
# MEMORY EVENTS
# =============================================================================

@dataclass
class MemoryOperationStarted(BaseEvent):
    """Memory operation started event."""
    category: EventCategory = field(default=EventCategory.MEMORY, init=False)
    request_id: str = ""
    operation_type: str = ""


@dataclass
class MemoryOperationCompleted(BaseEvent):
    """Memory operation completed event."""
    category: EventCategory = field(default=EventCategory.MEMORY, init=False)
    request_id: str = ""
    operations_completed: List[str] = field(default_factory=list)


@dataclass
class MemoryRetrievalStarted(BaseEvent):
    """Memory retrieval started event."""
    category: EventCategory = field(default=EventCategory.MEMORY, init=False)
    query: str = ""
    memory_type: str = ""


@dataclass
class MemoryRetrievalCompleted(BaseEvent):
    """Memory retrieval completed event."""
    category: EventCategory = field(default=EventCategory.MEMORY, init=False)
    query: str = ""
    results_count: int = 0
    retrieval_time: float = 0.0


@dataclass
class MemoryConsolidationStarted(BaseEvent):
    """Memory consolidation started event."""
    category: EventCategory = field(default=EventCategory.MEMORY, init=False)
    consolidation_type: str = ""


@dataclass
class MemoryConsolidationCompleted(BaseEvent):
    """Memory consolidation completed event."""
    category: EventCategory = field(default=EventCategory.MEMORY, init=False)
    memories_consolidated: int = 0
    consolidation_time: float = 0.0


@dataclass
class ContextAdapted(BaseEvent):
    """Context adaptation event."""
    category: EventCategory = field(default=EventCategory.MEMORY, init=False)
    adaptation_type: str = ""
    context_changes: List[str] = field(default_factory=list)


# =============================================================================
# SKILL EVENTS
# =============================================================================

@dataclass
class SkillExecutionStarted(BaseEvent):
    """Skill execution started event."""
    category: EventCategory = field(default=EventCategory.SKILL, init=False)
    skill_id: str = ""
    skill_name: str = ""
    execution_id: str = ""


@dataclass
class SkillExecutionCompleted(BaseEvent):
    """Skill execution completed event."""
    category: EventCategory = field(default=EventCategory.SKILL, init=False)
    skill_id: str = ""
    skill_name: str = ""
    execution_id: str = ""
    execution_time: float = 0.0
    success: bool = True


@dataclass
class SkillExecutionFailed(BaseEvent):
    """Skill execution failed event."""
    category: EventCategory = field(default=EventCategory.SKILL, init=False)
    priority: EventPriority = EventPriority.HIGH
    severity: EventSeverity = EventSeverity.ERROR
    skill_id: str = ""
    skill_name: str = ""
    execution_id: str = ""
    error_message: str = ""
    error_type: str = ""


@dataclass
class SkillRegistered(BaseEvent):
    """Skill registered event."""
    category: EventCategory = field(default=EventCategory.SKILL, init=False)
    skill_id: str = ""
    skill_name: str = ""
    skill_type: str = ""


@dataclass
class SkillUnregistered(BaseEvent):
    """Skill unregistered event."""
    category: EventCategory = field(default=EventCategory.SKILL, init=False)
    skill_id: str = ""
    skill_name: str = ""


@dataclass
class SkillValidationStarted(BaseEvent):
    """Skill validation started event."""
    category: EventCategory = field(default=EventCategory.SKILL, init=False)
    skill_id: str = ""
    validation_type: str = ""


@dataclass
class SkillValidationCompleted(BaseEvent):
    """Skill validation completed event."""
    category: EventCategory = field(default=EventCategory.SKILL, init=False)
    skill_id: str = ""
    validation_success: bool = True
    issues_found: List[str] = field(default_factory=list)


# =============================================================================
# LEARNING EVENTS
# =============================================================================

@dataclass
class LearningEventOccurred(BaseEvent):
    """Learning event occurred."""
    category: EventCategory = field(default=EventCategory.LEARNING, init=False)
    event_type: str = ""
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FeedbackReceived(BaseEvent):
    """Feedback received event."""
    category: EventCategory = field(default=EventCategory.LEARNING, init=False)
    interaction_id: str = ""
    feedback_type: str = ""
    feedback_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelAdaptationStarted(BaseEvent):
    """Model adaptation started event."""
    category: EventCategory = field(default=EventCategory.LEARNING, init=False)
    model_id: str = ""
    adaptation_type: str = ""


@dataclass
class ModelAdaptationCompleted(BaseEvent):
    """Model adaptation completed event."""
    category: EventCategory = field(default=EventCategory.LEARNING, init=False)
    model_id: str = ""
    adaptation_success: bool = True
    adaptation_time: float = 0.0


@dataclass
class LearningMetricsUpdated(BaseEvent):
    """Learning metrics updated event."""
    category: EventCategory = field(default=EventCategory.LEARNING, init=False)
    metrics: Dict[str, float] = field(default_factory=dict)


# =============================================================================
# PLUGIN EVENTS
# =============================================================================

@dataclass
class PluginLoaded(BaseEvent):
    """Plugin loaded event."""
    category: EventCategory = field(default=EventCategory.PLUGIN, init=False)
    plugin_id: str = ""
    plugin_type: str = ""
    load_time: float = 0.0


@dataclass
class PluginUnloaded(BaseEvent):
    """Plugin unloaded event."""
    category: EventCategory = field(default=EventCategory.PLUGIN, init=False)
    plugin_id: str = ""
    plugin_type: str = ""


@dataclass
class PluginEnabled(BaseEvent):
    """Plugin enabled event."""
    category: EventCategory = field(default=EventCategory.PLUGIN, init=False)
    plugin_id: str = ""
    plugin_type: str = ""


@dataclass
class PluginDisabled(BaseEvent):
    """Plugin disabled event."""
    category: EventCategory = field(default=EventCategory.PLUGIN, init=False)
    plugin_id: str = ""
    plugin_type: str = ""


@dataclass
class PluginError(BaseEvent):
    """Plugin error event."""
    category: EventCategory = field(default=EventCategory.PLUGIN, init=False)
    priority: EventPriority = EventPriority.HIGH
    severity: EventSeverity = EventSeverity.ERROR
    plugin_id: str = ""
    error_message: str = ""
    error_type: str = ""


@dataclass
class PluginDependencyResolved(BaseEvent):
    """Plugin dependency resolved event."""
    category: EventCategory = field(default=EventCategory.PLUGIN, init=False)
    plugin_id: str = ""
    dependency_id: str = ""


@dataclass
class PluginHotReloaded(BaseEvent):
    """Plugin hot reloaded event."""
    category: EventCategory = field(default=EventCategory.PLUGIN, init=False)
    plugin_id: str = ""
    plugin_type: str = ""


@dataclass
class PluginSecurityViolation(BaseEvent):
    """Plugin security violation event."""
    category: EventCategory = field(default=EventCategory.PLUGIN, init=False)
    priority: EventPriority = EventPriority.CRITICAL
    severity: EventSeverity = EventSeverity.CRITICAL
    plugin_id: str = ""
    violation_type: str = ""
    violation_details: str = ""


@dataclass
class PluginPerformanceWarning(BaseEvent):
    """Plugin performance warning event."""
    category: EventCategory = field(default=EventCategory.PLUGIN, init=False)
    priority: EventPriority = EventPriority.HIGH
    severity: EventSeverity = EventSeverity.WARNING
    plugin_id: str = ""
    metric: str = ""
    value: float = 0.0
    limit: float = 0.0


# =============================================================================
# INTEGRATION EVENTS
# =============================================================================

@dataclass
class IntegrationConnected(BaseEvent):
    """Integration connected event."""
    category: EventCategory = field(default=EventCategory.INTEGRATION, init=False)
    integration_type: str = ""
    integration_name: str = ""


@dataclass
class IntegrationDisconnected(BaseEvent):
    """Integration disconnected event."""
    category: EventCategory = field(default=EventCategory.INTEGRATION, init=False)
    priority: EventPriority = EventPriority.HIGH
    integration_type: str = ""
    integration_name: str = ""
    reason: str = ""


@dataclass
class IntegrationError(BaseEvent):
    """Integration error event."""
    category: EventCategory = field(default=EventCategory.INTEGRATION, init=False)
    priority: EventPriority = EventPriority.HIGH
    severity: EventSeverity = EventSeverity.ERROR
    integration_type: str = ""
    integration_name: str = ""
    error_message: str = ""
    error_type: str = ""


@dataclass
class APICallStarted(BaseEvent):
    """API call started event."""
    category: EventCategory = field(default=EventCategory.API, init=False)
    api_name: str = ""
    endpoint: str = ""
    method: str = ""


@dataclass
class APICallCompleted(BaseEvent):
    """API call completed event."""
    category: EventCategory = field(default=EventCategory.API, init=False)
    api_name: str = ""
    endpoint: str = ""
    method: str = ""
    status_code: int = 200
    response_time: float = 0.0


@dataclass
class APICallFailed(BaseEvent):
    """API call failed event."""
    category: EventCategory = field(default=EventCategory.API, init=False)
    priority: EventPriority = EventPriority.HIGH
    severity: EventSeverity = EventSeverity.ERROR
    api_name: str = ""
    endpoint: str = ""
    method: str = ""
    status_code: int = 500
    error_message: str = ""


@dataclass
class RateLimitExceeded(BaseEvent):
    """Rate limit exceeded event."""
    category: EventCategory = field(default=EventCategory.API, init=False)
    priority: EventPriority = EventPriority.HIGH
    severity: EventSeverity = EventSeverity.WARNING
    api_name: str = ""
    limit: int = 0
    current_usage: int = 0
    reset_time: Optional[datetime] = None


# =============================================================================
# SECURITY EVENTS
# =============================================================================

@dataclass
class SecurityViolationDetected(BaseEvent):
    """Security violation detected event."""
    category: EventCategory = field(default=EventCategory.SECURITY, init=False)
    priority: EventPriority = EventPriority.CRITICAL
    severity: EventSeverity = EventSeverity.CRITICAL
    violation_type: str = ""
    source_ip: Optional[str] = None
    details: str = ""


@dataclass
class AuthenticationFailed(BaseEvent):
    """Authentication failed event."""
    category: EventCategory = field(default=EventCategory.SECURITY, init=False)
    priority: EventPriority = EventPriority.HIGH
    severity: EventSeverity = EventSeverity.WARNING
    auth_method: str = ""
    failure_reason: str = ""
    source_ip: Optional[str] = None


@dataclass
class AuthorizationFailed(BaseEvent):
    """Authorization failed event."""
    category: EventCategory = field(default=EventCategory.SECURITY, init=False)
    priority: EventPriority = EventPriority.HIGH
    severity: EventSeverity = EventSeverity.WARNING
    resource: str = ""
    action: str = ""
    failure_reason: str = ""


@dataclass
class SuspiciousActivityDetected(BaseEvent):
    """Suspicious activity detected event."""
    category: EventCategory = field(default=EventCategory.SECURITY, init=False)
    priority: EventPriority = EventPriority.HIGH
    severity: EventSeverity = EventSeverity.WARNING
    activity_type: str = ""
    confidence_score: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EncryptionEvent(BaseEvent):
    """Encryption operation event."""
    category: EventCategory = field(default=EventCategory.SECURITY, init=False)
    operation: str = ""  # encrypt, decrypt
    data_type: str = ""
    success: bool = True


# =============================================================================
# PERFORMANCE EVENTS
# =============================================================================

@dataclass
class PerformanceMetricUpdated(BaseEvent):
    """Performance metric updated event."""
    category: EventCategory = field(default=EventCategory.PERFORMANCE, init=False)
    metric_name: str = ""
    metric_value: float = 0.0
    metric_unit: str = ""
    previous_value: Optional[float] = None


@dataclass
class PerformanceThresholdExceeded(BaseEvent):
    """Performance threshold exceeded event."""
    category: EventCategory = field(default=EventCategory.PERFORMANCE, init=False)
    priority: EventPriority = EventPriority.HIGH
    severity: EventSeverity = EventSeverity.WARNING
    metric_name: str = ""
    current_value: float = 0.0
    threshold: float = 0.0
    threshold_type: str = "upper"  # upper, lower


@dataclass
class ResourceUsageHigh(BaseEvent):
    """High resource usage event."""
    category: EventCategory = field(default=EventCategory.PERFORMANCE, init=False)
    priority: EventPriority = EventPriority.HIGH
    severity: EventSeverity = EventSeverity.WARNING
    resource_type: str = ""
    usage_percentage: float = 0.0
    threshold_percentage: float = 0.0


@dataclass
class PerformanceOptimizationApplied(BaseEvent):
    """Performance optimization applied event."""
    category: EventCategory = field(default=EventCategory.PERFORMANCE, init=False)
    optimization_type: str = ""
    improvement_factor: float = 0.0
    affected_components: List[str] = field(default_factory=list)


# =============================================================================
# ERROR EVENTS
# =============================================================================

@dataclass
class ErrorOccurred(BaseEvent):
    """General error occurred event."""
    category: EventCategory = field(default=EventCategory.ERROR, init=False)
    priority: EventPriority = EventPriority.HIGH
    severity: EventSeverity = EventSeverity.ERROR
    component: str = ""
    error_type: str = ""
    error_message: str = ""


@dataclass
class ExceptionCaught(BaseEvent):
    """Exception caught event."""
    category: EventCategory = field(default=EventCategory.ERROR, init=False)
    priority: EventPriority = EventPriority.HIGH
    severity: EventSeverity = EventSeverity.ERROR
    component: str = ""
    exception_type: str = ""
    exception_message: str = ""
    stack_trace: Optional[str] = None


@dataclass
class ErrorRecoveryStarted(BaseEvent):
    """Error recovery started event."""
    category: EventCategory = field(default=EventCategory.ERROR, init=False)
    error_id: str = ""
    recovery_strategy: str = ""


@dataclass
class ErrorRecoveryCompleted(BaseEvent):
    """Error recovery completed event."""
    category: EventCategory = field(default=EventCategory.ERROR, init=False)
    error_id: str = ""
    recovery_success: bool = True
    recovery_time: float = 0.0


# =============================================================================
# HEALTH AND MONITORING EVENTS
# =============================================================================

@dataclass
class HealthCheckStarted(BaseEvent):
    """Health check started event."""
    category: EventCategory = field(default=EventCategory.HEALTH, init=False)
    check_type: str = ""
    target_component: str = ""


@dataclass
class HealthCheckCompleted(BaseEvent):
    """Health check completed event."""
    category: EventCategory = field(default=EventCategory.HEALTH, init=False)
    check_type: str = ""
    target_component: str = ""
    health_status: str = ""
    check_duration: float = 0.0


@dataclass
class HealthCheckFailed(BaseEvent):
    """Health check failed event."""
    category: EventCategory = field(default=EventCategory.HEALTH, init=False)
    priority: EventPriority = EventPriority.HIGH
    severity: EventSeverity = EventSeverity.WARNING
    check_type: str = ""
    target_component: str = ""
    failure_reason: str = ""


@dataclass
class AlertTriggered(BaseEvent):
    """Alert triggered event."""
    category: EventCategory = field(default=EventCategory.HEALTH, init=False)
    priority: EventPriority = EventPriority.HIGH
    alert_type: str = ""
    alert_message: str = ""
    trigger_condition: str = ""


@dataclass
class AlertResolved(BaseEvent):
    """Alert resolved event."""
    category: EventCategory = field(default=EventCategory.HEALTH, init=False)
    alert_type: str = ""
    resolution_time: float = 0.0
    resolution_method: str = ""


# =============================================================================
# AUDIT EVENTS
# =============================================================================

@dataclass
class AuditLogEntry(BaseEvent):
    """Audit log entry event."""
    category: EventCategory = field(default=EventCategory.AUDIT, init=False)
    action: str = ""
    resource: str = ""
    outcome: str = ""  # success, failure
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataAccessEvent(BaseEvent):
    """Data access event."""
    category: EventCategory = field(default=EventCategory.AUDIT, init=False)
    data_type: str = ""
    access_type: str = ""  # read, write, delete
    success: bool = True


@dataclass
class ConfigurationChanged(BaseEvent):
    """Configuration changed event."""
    category: EventCategory = field(default=EventCategory.AUDIT, init=False)
    config_section: str = ""
    changes: Dict[str, Any] = field(default_factory=dict)
    changed_by: Optional[str] = None


# =============================================================================
# EVENT REGISTRY
# =============================================================================

EVENT_TYPES = {
    # System Events
    "SystemStarted": SystemStarted,
    "SystemShutdownStarted": SystemShutdownStarted,
    "SystemShutdownCompleted": SystemShutdownCompleted,
    "SystemStateChanged": SystemStateChanged,
    "SystemResourceAlert": SystemResourceAlert,
    
    # Component Events
    "ComponentRegistered": ComponentRegistered,
    "ComponentInitialized": ComponentInitialized,
    "ComponentStarted": ComponentStarted,
    "ComponentStopped": ComponentStopped,
    "ComponentFailed": ComponentFailed,
    "ComponentHealthChanged": ComponentHealthChanged,
    "DependencyResolved": DependencyResolved,
    
    # Engine Events
    "EngineStarted": EngineStarted,
    "EngineShutdown": EngineShutdown,
    
    # Workflow Events
    "WorkflowStarted": WorkflowStarted,
    "WorkflowCompleted": WorkflowCompleted,
    "WorkflowFailed": WorkflowFailed,
    "WorkflowPaused": WorkflowPaused,
    "WorkflowResumed": WorkflowResumed,
    "WorkflowCancelled": WorkflowCancelled,
    "WorkflowStepStarted": WorkflowStepStarted,
    "WorkflowStepCompleted": WorkflowStepCompleted,
    "WorkflowStepFailed": WorkflowStepFailed,
    "WorkflowStepSkipped": WorkflowStepSkipped,
    "WorkflowBranchingOccurred": WorkflowBranchingOccurred,
    "WorkflowMerged": WorkflowMerged,
    "WorkflowAdapted": WorkflowAdapted,
    
    # Session Events
    "SessionStarted": SessionStarted,
    "SessionEnded": SessionEnded,
    "SessionExpired": SessionExpired,
    "SessionRestored": SessionRestored,
    "SessionContextUpdated": SessionContextUpdated,
    "SessionStateChanged": SessionStateChanged,
    "SessionCleanupStarted": SessionCleanupStarted,
    "SessionCleanupCompleted": SessionCleanupCompleted,
    "SessionMigrated": SessionMigrated,
    "SessionClusteringStarted": SessionClusteringStarted,
    "SessionHealthCheckFailed": SessionHealthCheckFailed,
    
    # User and Interaction Events
    "UserInteractionStarted": UserInteractionStarted,
    "UserInteractionCompleted": UserInteractionCompleted,
    "UserInteractionFailed": UserInteractionFailed,
    "ConversationStarted": ConversationStarted,
    "ConversationEnded": ConversationEnded,
    "MessageReceived": MessageReceived,
    "MessageSent": MessageSent,
    "MessageProcessed": MessageProcessed,
    "UserJoinedSession": UserJoinedSession,
    "UserLeftSession": UserLeftSession,
    "UserAuthenticated": UserAuthenticated,
    "UserAuthorized": UserAuthorized,
    "UserLoggedOut": UserLoggedOut,
    "UserPreferenceUpdated": UserPreferenceUpdated,
    
    # Processing Events
    "ProcessingStarted": ProcessingStarted,
    "ProcessingCompleted": ProcessingCompleted,
    "ProcessingError": ProcessingError,
    "ModalityProcessingStarted": ModalityProcessingStarted,
    "ModalityProcessingCompleted": ModalityProcessingCompleted,
    "ModalityDetected": ModalityDetected,
    "FusionStarted": FusionStarted,
    "FusionCompleted": FusionCompleted,
    "InteractionModeChanged": InteractionModeChanged,
    "RealTimeProcessingStarted": RealTimeProcessingStarted,
    "RealTimeProcessingCompleted": RealTimeProcessingCompleted,
    "StreamingStarted": StreamingStarted,
    "StreamingCompleted": StreamingCompleted,
    
    # Memory Events
    "MemoryOperationStarted": MemoryOperationStarted,
    "MemoryOperationCompleted": MemoryOperationCompleted,
    "MemoryRetrievalStarted": MemoryRetrievalStarted,
    "MemoryRetrievalCompleted": MemoryRetrievalCompleted,
    "MemoryConsolidationStarted": MemoryConsolidationStarted,
    "MemoryConsolidationCompleted": MemoryConsolidationCompleted,
    "ContextAdapted": ContextAdapted,
    
    # Skill Events
    "SkillExecutionStarted": SkillExecutionStarted,
    "SkillExecutionCompleted": SkillExecutionCompleted,
    "SkillExecutionFailed": SkillExecutionFailed,
    "SkillRegistered": SkillRegistered,
    "SkillUnregistered": SkillUnregistered,
    "SkillValidationStarted": SkillValidationStarted,
    "SkillValidationCompleted": SkillValidationCompleted,
    
    # Learning Events
    "LearningEventOccurred": LearningEventOccurred,
    "FeedbackReceived": FeedbackReceived,
    "ModelAdaptationStarted": ModelAdaptationStarted,
    "ModelAdaptationCompleted": ModelAdaptationCompleted,
    "LearningMetricsUpdated": LearningMetricsUpdated,
    
    # Plugin Events
    "PluginLoaded": PluginLoaded,
    "PluginUnloaded": PluginUnloaded,
    "PluginEnabled": PluginEnabled,
    "PluginDisabled": PluginDisabled,
    "PluginError": PluginError,
    "PluginDependencyResolved": PluginDependencyResolved,
    "PluginHotReloaded": PluginHotReloaded,
    "PluginSecurityViolation": PluginSecurityViolation,
    "PluginPerformanceWarning": PluginPerformanceWarning,
    
    # Integration Events
    "IntegrationConnected": IntegrationConnected,
    "IntegrationDisconnected": IntegrationDisconnected,
    "IntegrationError": IntegrationError,
    "APICallStarted": APICallStarted,
    "APICallCompleted": APICallCompleted,
    "APICallFailed": APICallFailed,
    "RateLimitExceeded": RateLimitExceeded,
    
    # Security Events
    "SecurityViolationDetected": SecurityViolationDetected,
    "AuthenticationFailed": AuthenticationFailed,
    "AuthorizationFailed": AuthorizationFailed,
    "SuspiciousActivityDetected": SuspiciousActivityDetected,
    "EncryptionEvent": EncryptionEvent,
    
    # Performance Events
    "PerformanceMetricUpdated": PerformanceMetricUpdated,
    "PerformanceThresholdExceeded": PerformanceThresholdExceeded,
    "ResourceUsageHigh": ResourceUsageHigh,
    "PerformanceOptimizationApplied": PerformanceOptimizationApplied,
    
    # Error Events
    "ErrorOccurred": ErrorOccurred,
    "ExceptionCaught": ExceptionCaught,
    "ErrorRecoveryStarted": ErrorRecoveryStarted,
    "ErrorRecoveryCompleted": ErrorRecoveryCompleted,
    
    # Health and Monitoring Events
    "HealthCheckStarted": HealthCheckStarted,
    "HealthCheckCompleted": HealthCheckCompleted,
    "HealthCheckFailed": HealthCheckFailed,
    "AlertTriggered": AlertTriggered,
    "AlertResolved": AlertResolved,
    
    # Audit Events
    "AuditLogEntry": AuditLogEntry,
    "DataAccessEvent": DataAccessEvent,
    "ConfigurationChanged": ConfigurationChanged,
}


def get_event_type(event_name: str) -> Optional[type]:
    """Get event type class by name."""
    return EVENT_TYPES.get(event_name)


def list_event_types(category: Optional[EventCategory] = None) -> List[str]:
    """List all event types, optionally filtered by category."""
    if category is None:
        return list(EVENT_TYPES.keys())
    
    filtered_events = []
    for event_name, event_class in EVENT_TYPES.items():
        try:
            # Create a temporary instance to check category
            temp_instance = event_class()
            if temp_instance.category == category:
                filtered_events.append(event_name)
        except Exception:
            continue
    
    return filtered_events


def create_event(event_type: str, **kwargs) -> Optional[BaseEvent]:
    """Create an event instance by type name."""
    event_class = EVENT_TYPES.get(event_type)
    if event_class:
        return event_class(**kwargs)
    return None


# Export all event types for easy importing
__all__ = [
    "BaseEvent",
    "EventCategory",
    "EventPriority",
    "EventSeverity",
    "EVENT_TYPES",
    "get_event_type",
    "list_event_types",
    "create_event",
] + list(EVENT_TYPES.keys())
