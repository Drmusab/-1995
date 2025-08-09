"""
Consolidated Core Assistant System

This module contains all the core components of the AI assistant system,
consolidated into a single file for simplified management and deployment.
It includes:
- Component lifecycle management
- Multimodal input processing (text, speech, vision)
- Session management with automatic cleanup
- Multi-modal user interaction handling
- Plugin discovery and lifecycle management
- Complex workflow execution and orchestration
- Centralized error handling and recovery
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import uuid

from src.core.dependency_injection import Container
from src.core.error_handling import (
    AIAssistantError,
    ErrorCategory,
    ErrorSeverity,
    error_handler
)
from src.core.events.event_bus import EventBus
from src.core.events.event_types import (
    SessionStarted,
    SessionEnded,
    SessionCleanupStarted,
    TaskCompleted,
    SkillExecuted,
    MemoryUpdated,
    LearningEventOccurred,
    SystemHealthCheck,
    ComponentStarted,
    ComponentHealthChanged,
    ProcessingStarted,
    ProcessingCompleted,
    EngineStarted,
    UserInteractionStarted,
    UserInteractionCompleted,
    PluginLoaded,
    PluginEnabled,
    PluginDisabled,
    WorkflowStarted,
    WorkflowCompleted,
    WorkflowStepStarted,
    WorkflowStepCompleted
)
from src.integrations.model_inference_coordinator import ModelInferenceCoordinator
from src.memory.core_memory.memory_manager import MemoryManager
from src.processing.natural_language.bilingual_manager import BilingualManager, Language
from src.processing.multimodal.fusion_strategies import MultimodalFusionStrategy
from src.reasoning.logic_engine import LogicEngine
from src.reasoning.planning.task_planner import TaskPlanner
from src.skills.skill_registry import SkillRegistry
from src.learning.continual_learning import ContinualLearningEngine
from src.learning.preference_learning import PreferenceLearningEngine
from src.observability.logging import get_logger
from src.observability.monitoring.metrics import MetricsCollector


# =============================================================================
# ENUMS AND STATUS DEFINITIONS
# =============================================================================

class AssistantState(Enum):
    """States of the assistant lifecycle."""
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    READY = "ready"
    PROCESSING = "processing"
    LEARNING = "learning"
    ERROR = "error"
    SHUTTING_DOWN = "shutting_down"
    SHUTDOWN = "shutdown"


class ProcessingMode(Enum):
    """Processing modes for different types of interactions."""
    CONVERSATIONAL = "conversational"
    TASK_ORIENTED = "task_oriented"
    LEARNING = "learning"
    CREATIVE = "creative"
    ANALYTICAL = "analytical"
    MULTIMODAL = "multimodal"
    BATCH = "batch"
    REAL_TIME = "real_time"
    STREAMING = "streaming"


class ComponentStatus(Enum):
    """Component status enumeration."""
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    RUNNING = "running"
    STOPPED = "stopped"
    ERROR = "error"


class EngineState(Enum):
    """Engine state enumeration."""
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    READY = "ready"
    PROCESSING = "processing"
    ERROR = "error"
    SHUTDOWN = "shutdown"


class ModalityType(Enum):
    """Input modality types."""
    TEXT = "text"
    SPEECH = "speech"
    VISION = "vision"
    MULTIMODAL = "multimodal"


class InputModality(Enum):
    """Input modality types for interactions."""
    TEXT = "text"
    SPEECH = "speech"
    VISION = "vision"
    GESTURE = "gesture"
    MULTIMODAL = "multimodal"


class OutputModality(Enum):
    """Output modality types."""
    TEXT = "text"
    SPEECH = "speech"
    VISUAL = "visual"
    HAPTIC = "haptic"


class InteractionMode(Enum):
    """Interaction mode types."""
    CONVERSATIONAL = "conversational"
    COMMAND = "command"
    WORKFLOW = "workflow"
    COLLABORATIVE = "collaborative"


class PriorityLevel(Enum):
    """Processing priority levels."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


class PluginStatus(Enum):
    """Plugin status enumeration."""
    UNLOADED = "unloaded"
    LOADED = "loaded"
    ENABLED = "enabled"
    DISABLED = "disabled"
    ERROR = "error"


class WorkflowStatus(Enum):
    """Workflow execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class StepStatus(Enum):
    """Workflow step status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


# =============================================================================
# DATA CLASSES AND MODELS
# =============================================================================

@dataclass
class AssistantContext:
    """Context information for the current assistant session."""
    session_id: str
    user_id: Optional[str] = None
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    active_skills: Set[str] = field(default_factory=set)
    processing_mode: ProcessingMode = ProcessingMode.CONVERSATIONAL
    language_preference: Language = Language.ARABIC
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_interaction: Optional[datetime] = None


@dataclass
class ProcessingRequest:
    """Request for processing user input."""
    input_data: Union[str, Dict[str, Any]]
    input_type: str  # "text", "speech", "vision", "multimodal"
    context: AssistantContext
    metadata: Dict[str, Any] = field(default_factory=dict)
    priority: int = 1
    timeout: Optional[float] = None
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class ProcessingResponse:
    """Response from processing user input."""
    response_data: Union[str, Dict[str, Any]]
    response_type: str
    confidence: float
    skills_used: List[str]
    reasoning_path: Optional[List[Dict[str, Any]]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    processing_time: float = 0.0


@dataclass
class ComponentInfo:
    """Information about a managed component."""
    name: str
    instance: Any
    status: ComponentStatus = ComponentStatus.UNINITIALIZED
    dependencies: List[str] = field(default_factory=list)
    health_score: float = 1.0
    last_health_check: float = 0.0


@dataclass
class MultimodalInput:
    """Represents multimodal input data."""
    input_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    text: Optional[str] = None
    audio_data: Optional[bytes] = None
    image_data: Optional[bytes] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ProcessingContext:
    """Processing context information."""
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    preferences: Dict[str, Any] = field(default_factory=dict)
    mode: ProcessingMode = ProcessingMode.BATCH
    priority: PriorityLevel = PriorityLevel.NORMAL


@dataclass
class ProcessingResult:
    """Result of processing operation."""
    result_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    success: bool = True
    response_text: Optional[str] = None
    response_audio: Optional[bytes] = None
    response_data: Dict[str, Any] = field(default_factory=dict)
    processing_time: float = 0.0
    confidence: float = 1.0
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SessionInfo:
    """Information about a user session."""
    session_id: str
    user_id: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_activity: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    context: Dict[str, Any] = field(default_factory=dict)
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    preferences: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_active: bool = True


@dataclass
class UserMessage:
    """Represents a user message."""
    message_id: str
    user_id: Optional[str]
    session_id: str
    interaction_id: str
    text: Optional[str] = None
    audio_data: Optional[bytes] = None
    image_data: Optional[bytes] = None
    modality: InputModality = InputModality.TEXT
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AssistantResponse:
    """Represents an assistant response."""
    response_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    text: Optional[str] = None
    audio_data: Optional[bytes] = None
    visual_elements: Optional[Dict[str, Any]] = None
    modalities: Set[OutputModality] = field(default_factory=set)
    confidence: float = 1.0
    processing_time: float = 0.0
    suggested_follow_ups: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InteractionInfo:
    """Information about an active interaction."""
    interaction_id: str
    user_id: Optional[str]
    session_id: str
    mode: InteractionMode
    input_modalities: Set[InputModality]
    output_modalities: Set[OutputModality]
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_activity: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    message_count: int = 0
    context: Dict[str, Any] = field(default_factory=dict)


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


@dataclass
class WorkflowStep:
    """Represents a workflow step."""
    step_id: str
    name: str
    step_type: str  # "skill", "api_call", "condition", "loop", etc.
    parameters: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    timeout: Optional[float] = None
    retry_count: int = 0
    max_retries: int = 3


@dataclass
class WorkflowDefinition:
    """Defines a workflow."""
    workflow_id: str
    name: str
    description: str = ""
    version: str = "1.0.0"
    steps: List[WorkflowStep] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowExecution:
    """Represents a workflow execution instance."""
    execution_id: str
    workflow_id: str
    status: WorkflowStatus = WorkflowStatus.PENDING
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    input_data: Dict[str, Any] = field(default_factory=dict)
    output_data: Dict[str, Any] = field(default_factory=dict)
    step_results: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# COMPONENT CLASSES
# =============================================================================

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
        self.logger = logging.getLogger(f"{__name__}.ComponentManager")

    async def discover_components(self) -> List[str]:
        """Discover available components in the system."""
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


class EnhancedSessionManager:
    """
    Enhanced session manager that provides comprehensive session management
    with context persistence and automatic cleanup.
    """

    def __init__(self, container: Container):
        """Initialize the session manager."""
        self.container = container
        self.sessions: Dict[str, SessionInfo] = {}
        self.user_sessions: Dict[str, Set[str]] = {}  # user_id -> session_ids
        self.event_bus = container.get(EventBus) if container else None
        self.logger = logging.getLogger(f"{__name__}.SessionManager")
        
        # Configuration
        self.max_session_age = timedelta(hours=24)
        self.max_inactive_time = timedelta(hours=2)
        self.cleanup_interval = timedelta(minutes=30)
        
        # Background task
        self._cleanup_task: Optional[asyncio.Task] = None

    async def initialize(self) -> None:
        """Initialize the session manager."""
        self.logger.info("Initializing Enhanced Session Manager")
        
        # Start background cleanup task
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        self.logger.info("Enhanced Session Manager initialized successfully")

    async def create_session(
        self,
        user_id: Optional[str] = None,
        initial_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a new session."""
        session_id = str(uuid.uuid4())
        
        session_info = SessionInfo(
            session_id=session_id,
            user_id=user_id,
            context=initial_context or {},
        )
        
        self.sessions[session_id] = session_info
        
        # Track user sessions
        if user_id:
            if user_id not in self.user_sessions:
                self.user_sessions[user_id] = set()
            self.user_sessions[user_id].add(session_id)
        
        # Emit event
        if self.event_bus:
            await self.event_bus.emit(
                SessionStarted(
                    session_id=session_id,
                    user_id=user_id
                )
            )
        
        self.logger.info(f"Created session {session_id} for user {user_id}")
        return session_id

    async def get_session(self, session_id: str) -> Optional[SessionInfo]:
        """Get session information."""
        session = self.sessions.get(session_id)
        if session and session.is_active:
            # Update last activity
            session.last_activity = datetime.now(timezone.utc)
            return session
        return None

    async def update_session_context(
        self,
        session_id: str,
        context_update: Dict[str, Any]
    ) -> bool:
        """Update session context."""
        session = self.sessions.get(session_id)
        if session and session.is_active:
            session.context.update(context_update)
            session.last_activity = datetime.now(timezone.utc)
            return True
        return False

    async def add_conversation_entry(
        self,
        session_id: str,
        entry: Dict[str, Any]
    ) -> bool:
        """Add an entry to the conversation history."""
        session = self.sessions.get(session_id)
        if session and session.is_active:
            entry_with_timestamp = {
                **entry,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            session.conversation_history.append(entry_with_timestamp)
            session.last_activity = datetime.now(timezone.utc)
            
            # Limit conversation history size
            max_history = 100
            if len(session.conversation_history) > max_history:
                session.conversation_history = session.conversation_history[-max_history:]
            
            return True
        return False

    async def end_session(self, session_id: str) -> bool:
        """End a session."""
        session = self.sessions.get(session_id)
        if session:
            session.is_active = False
            
            # Remove from user sessions
            if session.user_id and session.user_id in self.user_sessions:
                self.user_sessions[session.user_id].discard(session_id)
                if not self.user_sessions[session.user_id]:
                    del self.user_sessions[session.user_id]
            
            # Emit event
            if self.event_bus:
                await self.event_bus.emit(
                    SessionEnded(
                        session_id=session_id,
                        user_id=session.user_id,
                        duration=(datetime.now(timezone.utc) - session.created_at).total_seconds()
                    )
                )
            
            self.logger.info(f"Ended session {session_id}")
            return True
        return False

    async def get_user_sessions(self, user_id: str) -> List[SessionInfo]:
        """Get all active sessions for a user."""
        session_ids = self.user_sessions.get(user_id, set())
        sessions = []
        
        for session_id in session_ids:
            session = self.sessions.get(session_id)
            if session and session.is_active:
                sessions.append(session)
        
        return sessions

    def get_session_statistics(self) -> Dict[str, Any]:
        """Get session statistics."""
        active_sessions = [s for s in self.sessions.values() if s.is_active]
        
        return {
            "total_sessions": len(self.sessions),
            "active_sessions": len(active_sessions),
            "unique_users": len(self.user_sessions),
            "average_session_duration": self._calculate_average_session_duration(),
        }

    def _calculate_average_session_duration(self) -> float:
        """Calculate average session duration in seconds."""
        if not self.sessions:
            return 0.0
            
        total_duration = 0.0
        count = 0
        now = datetime.now(timezone.utc)
        
        for session in self.sessions.values():
            if session.is_active:
                duration = (now - session.created_at).total_seconds()
            else:
                duration = (session.last_activity - session.created_at).total_seconds()
            
            total_duration += duration
            count += 1
        
        return total_duration / count if count > 0 else 0.0

    async def _cleanup_loop(self) -> None:
        """Background task for cleaning up expired sessions."""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval.total_seconds())
                await self._cleanup_expired_sessions()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in session cleanup: {e}")

    async def _cleanup_expired_sessions(self) -> None:
        """Clean up expired sessions."""
        now = datetime.now(timezone.utc)
        expired_sessions = []
        
        for session_id, session in self.sessions.items():
            if not session.is_active:
                continue
                
            # Check if session is too old
            if now - session.created_at > self.max_session_age:
                expired_sessions.append(session_id)
                continue
                
            # Check if session is inactive
            if now - session.last_activity > self.max_inactive_time:
                expired_sessions.append(session_id)
        
        if expired_sessions:
            if self.event_bus:
                await self.event_bus.emit(
                    SessionCleanupStarted(count=len(expired_sessions))
                )
            
            for session_id in expired_sessions:
                await self.end_session(session_id)
                del self.sessions[session_id]
            
            self.logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")

    async def cleanup(self) -> None:
        """Cleanup the session manager."""
        self.logger.info("Cleaning up Enhanced Session Manager")
        
        # Cancel cleanup task
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # End all active sessions
        active_session_ids = [
            session_id for session_id, session in self.sessions.items()
            if session.is_active
        ]
        
        for session_id in active_session_ids:
            await self.end_session(session_id)
        
        self.logger.info("Enhanced Session Manager cleanup complete")


class InteractionHandler:
    """
    Manages user interactions across different modalities and provides
    a unified interface for processing user input and generating responses.
    """

    def __init__(self, container: Container):
        """Initialize the interaction handler."""
        self.container = container
        self.active_interactions: Dict[str, InteractionInfo] = {}
        self.event_bus = container.get(EventBus) if container else None
        self.logger = logging.getLogger(f"{__name__}.InteractionHandler")

    async def initialize(self) -> None:
        """Initialize the interaction handler."""
        self.logger.info("Initializing Interaction Handler")
        self.logger.info("Interaction Handler initialized successfully")

    async def start_interaction(
        self,
        user_id: Optional[str],
        session_id: str,
        interaction_mode: InteractionMode,
        input_modalities: Set[InputModality],
        output_modalities: Set[OutputModality],
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Start a new interaction."""
        interaction_id = str(uuid.uuid4())
        
        interaction_info = InteractionInfo(
            interaction_id=interaction_id,
            user_id=user_id,
            session_id=session_id,
            mode=interaction_mode,
            input_modalities=input_modalities,
            output_modalities=output_modalities,
            context=context or {}
        )
        
        self.active_interactions[interaction_id] = interaction_info
        
        if self.event_bus:
            await self.event_bus.emit(
                UserInteractionStarted(
                    interaction_id=interaction_id,
                    user_id=user_id,
                    session_id=session_id
                )
            )
        
        self.logger.info(f"Started interaction {interaction_id}")
        return interaction_id

    async def process_user_message(
        self,
        interaction_id: str,
        message: UserMessage,
        real_time: bool = False,
        streaming: bool = False
    ) -> AssistantResponse:
        """Process a user message and generate response."""
        interaction = self.active_interactions.get(interaction_id)
        if not interaction:
            raise ValueError(f"Interaction {interaction_id} not found")
        
        start_time = datetime.now(timezone.utc)
        
        try:
            # Update interaction info
            interaction.last_activity = datetime.now(timezone.utc)
            interaction.message_count += 1
            
            # Process message (stub implementation)
            response_text = await self._generate_response(message, interaction)
            
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            response = AssistantResponse(
                text=response_text,
                modalities={OutputModality.TEXT},
                processing_time=processing_time,
                confidence=0.95
            )
            
            # Add some suggested follow-ups
            response.suggested_follow_ups = [
                "Can you tell me more about that?",
                "What else can I help you with?",
                "Would you like to explore related topics?"
            ]
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error processing message: {e}")
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            return AssistantResponse(
                text="I apologize, but I encountered an error processing your request. Please try again.",
                modalities={OutputModality.TEXT},
                processing_time=processing_time,
                confidence=0.0
            )

    async def _generate_response(
        self,
        message: UserMessage,
        interaction: InteractionInfo
    ) -> str:
        """Generate response text (stub implementation)."""
        if message.text:
            if "hello" in message.text.lower():
                return "Hello! How can I assist you today?"
            elif "help" in message.text.lower():
                return "I'm here to help! You can ask me questions, request tasks, or just have a conversation."
            elif "?" in message.text:
                return f"That's an interesting question about '{message.text}'. Let me think about that..."
            else:
                return f"I understand you mentioned '{message.text}'. How would you like me to help with that?"
        else:
            return "I received your message. How can I assist you today?"

    async def end_interaction(self, interaction_id: str) -> bool:
        """End an interaction."""
        interaction = self.active_interactions.get(interaction_id)
        if not interaction:
            return False
        
        duration = (datetime.now(timezone.utc) - interaction.started_at).total_seconds()
        
        if self.event_bus:
            await self.event_bus.emit(
                UserInteractionCompleted(
                    interaction_id=interaction_id,
                    user_id=interaction.user_id,
                    session_id=interaction.session_id,
                    duration=duration,
                    message_count=interaction.message_count
                )
            )
        
        del self.active_interactions[interaction_id]
        self.logger.info(f"Ended interaction {interaction_id}")
        return True

    def get_active_interactions(self) -> List[str]:
        """Get list of active interaction IDs."""
        return list(self.active_interactions.keys())

    async def cleanup(self) -> None:
        """Cleanup the interaction handler."""
        self.logger.info("Cleaning up Interaction Handler")
        
        # End all active interactions
        interaction_ids = list(self.active_interactions.keys())
        for interaction_id in interaction_ids:
            await self.end_interaction(interaction_id)
        
        self.logger.info("Interaction Handler cleanup complete")


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
        self.logger = logging.getLogger(f"{__name__}.PluginManager")

    async def initialize(self) -> None:
        """Initialize the plugin manager."""
        self.logger.info("Initializing Enhanced Plugin Manager")
        
        # Discover and load built-in plugins
        await self.discover_plugins()
        
        self.logger.info("Enhanced Plugin Manager initialized successfully")

    async def discover_plugins(self) -> List[str]:
        """Discover available plugins."""
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
        """Load a plugin."""
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
        """Enable a plugin."""
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
        """Disable a plugin."""
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


class WorkflowOrchestrator:
    """
    Orchestrates workflow execution with support for complex control flows,
    error handling, and monitoring.
    """

    def __init__(self, container: Container):
        """Initialize the workflow orchestrator."""
        self.container = container
        self.workflows: Dict[str, WorkflowDefinition] = {}
        self.executions: Dict[str, WorkflowExecution] = {}
        self.active_executions: Set[str] = set()
        self.event_bus = container.get(EventBus) if container else None
        self.logger = logging.getLogger(f"{__name__}.WorkflowOrchestrator")

    async def initialize(self) -> None:
        """Initialize the workflow orchestrator."""
        self.logger.info("Initializing Workflow Orchestrator")
        
        # Register built-in workflows
        await self._register_builtin_workflows()
        
        self.logger.info("Workflow Orchestrator initialized successfully")

    async def _register_builtin_workflows(self) -> None:
        """Register built-in workflows."""
        # Simple greeting workflow
        greeting_workflow = WorkflowDefinition(
            workflow_id="greeting_workflow",
            name="Greeting Workflow",
            description="Simple greeting and introduction workflow",
            steps=[
                WorkflowStep(
                    step_id="greet_user",
                    name="Greet User",
                    step_type="skill",
                    parameters={"skill_name": "greeting_skill"}
                ),
                WorkflowStep(
                    step_id="get_user_info",
                    name="Get User Information",
                    step_type="skill",
                    parameters={"skill_name": "user_info_skill"},
                    dependencies=["greet_user"]
                )
            ]
        )
        
        # Task management workflow
        task_workflow = WorkflowDefinition(
            workflow_id="task_management",
            name="Task Management Workflow",
            description="Workflow for managing user tasks and reminders",
            steps=[
                WorkflowStep(
                    step_id="parse_task",
                    name="Parse Task Request",
                    step_type="nlp",
                    parameters={"parser_type": "task_parser"}
                ),
                WorkflowStep(
                    step_id="create_task",
                    name="Create Task",
                    step_type="skill",
                    parameters={"skill_name": "task_creator"},
                    dependencies=["parse_task"]
                ),
                WorkflowStep(
                    step_id="schedule_reminder",
                    name="Schedule Reminder",
                    step_type="skill",
                    parameters={"skill_name": "reminder_scheduler"},
                    dependencies=["create_task"]
                )
            ]
        )
        
        self.workflows["greeting_workflow"] = greeting_workflow
        self.workflows["task_management"] = task_workflow
        
        self.logger.info(f"Registered {len(self.workflows)} built-in workflows")

    async def execute_workflow(
        self,
        workflow_id: str,
        input_data: Dict[str, Any],
        session_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> str:
        """Execute a workflow."""
        workflow = self.workflows.get(workflow_id)
        if not workflow:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        execution_id = str(uuid.uuid4())
        
        execution = WorkflowExecution(
            execution_id=execution_id,
            workflow_id=workflow_id,
            session_id=session_id,
            user_id=user_id,
            input_data=input_data,
            started_at=datetime.now(timezone.utc)
        )
        
        self.executions[execution_id] = execution
        self.active_executions.add(execution_id)
        
        if self.event_bus:
            await self.event_bus.emit(
                WorkflowStarted(
                    workflow_id=workflow_id,
                    execution_id=execution_id,
                    user_id=user_id
                )
            )
        
        # Start execution in background
        asyncio.create_task(self._execute_workflow(execution_id))
        
        self.logger.info(f"Started workflow execution {execution_id}")
        return execution_id

    async def _execute_workflow(self, execution_id: str) -> None:
        """Execute a workflow instance."""
        execution = self.executions.get(execution_id)
        workflow = self.workflows.get(execution.workflow_id) if execution else None
        
        if not execution or not workflow:
            return
        
        try:
            execution.status = WorkflowStatus.RUNNING
            
            # Execute steps in dependency order
            for step in workflow.steps:
                if execution.status != WorkflowStatus.RUNNING:
                    break
                
                await self._execute_step(execution_id, step)
            
            # Mark as completed
            execution.status = WorkflowStatus.COMPLETED
            execution.completed_at = datetime.now(timezone.utc)
            
            if self.event_bus:
                await self.event_bus.emit(
                    WorkflowCompleted(
                        workflow_id=workflow.workflow_id,
                        execution_id=execution_id,
                        success=True,
                        duration=(execution.completed_at - execution.started_at).total_seconds()
                    )
                )
            
            self.logger.info(f"Workflow execution {execution_id} completed successfully")
            
        except Exception as e:
            execution.status = WorkflowStatus.FAILED
            execution.error_message = str(e)
            execution.completed_at = datetime.now(timezone.utc)
            
            self.logger.error(f"Workflow execution {execution_id} failed: {e}")
            
        finally:
            self.active_executions.discard(execution_id)

    async def _execute_step(self, execution_id: str, step: WorkflowStep) -> None:
        """Execute a workflow step."""
        execution = self.executions[execution_id]
        
        if self.event_bus:
            await self.event_bus.emit(
                WorkflowStepStarted(
                    execution_id=execution_id,
                    step_id=step.step_id,
                    step_name=step.name
                )
            )
        
        try:
            # Check dependencies
            for dep_step_id in step.dependencies:
                if dep_step_id not in execution.step_results:
                    raise RuntimeError(f"Dependency {dep_step_id} not satisfied")
            
            # Execute step based on type
            result = await self._execute_step_by_type(step, execution)
            
            # Store result
            execution.step_results[step.step_id] = result
            
            if self.event_bus:
                await self.event_bus.emit(
                    WorkflowStepCompleted(
                        execution_id=execution_id,
                        step_id=step.step_id,
                        step_name=step.name,
                        success=True
                    )
                )
            
            self.logger.debug(f"Step {step.step_id} completed successfully")
            
        except Exception as e:
            execution.step_results[step.step_id] = {"error": str(e)}
            self.logger.error(f"Step {step.step_id} failed: {e}")
            raise

    async def _execute_step_by_type(
        self,
        step: WorkflowStep,
        execution: WorkflowExecution
    ) -> Dict[str, Any]:
        """Execute a step based on its type."""
        if step.step_type == "skill":
            return await self._execute_skill_step(step, execution)
        elif step.step_type == "nlp":
            return await self._execute_nlp_step(step, execution)
        elif step.step_type == "api_call":
            return await self._execute_api_step(step, execution)
        else:
            return {"result": f"Executed {step.step_type} step", "success": True}

    async def _execute_skill_step(
        self,
        step: WorkflowStep,
        execution: WorkflowExecution
    ) -> Dict[str, Any]:
        """Execute a skill step."""
        skill_name = step.parameters.get("skill_name", "unknown")
        await asyncio.sleep(0.1)  # Simulate processing
        return {
            "skill": skill_name,
            "result": f"Executed skill {skill_name}",
            "success": True
        }

    async def _execute_nlp_step(
        self,
        step: WorkflowStep,
        execution: WorkflowExecution
    ) -> Dict[str, Any]:
        """Execute an NLP step."""
        await asyncio.sleep(0.1)  # Simulate processing
        return {
            "parsed_intent": "task_creation",
            "entities": {"task": "example task"},
            "success": True
        }

    async def _execute_api_step(
        self,
        step: WorkflowStep,
        execution: WorkflowExecution
    ) -> Dict[str, Any]:
        """Execute an API call step."""
        await asyncio.sleep(0.1)  # Simulate API call
        return {
            "api_response": {"status": "success"},
            "success": True
        }

    async def get_execution_status(self, execution_id: str) -> Dict[str, Any]:
        """Get workflow execution status."""
        execution = self.executions.get(execution_id)
        if not execution:
            return {"error": "Execution not found"}
        
        return {
            "execution_id": execution_id,
            "workflow_id": execution.workflow_id,
            "status": execution.status.value,
            "started_at": execution.started_at.isoformat() if execution.started_at else None,
            "completed_at": execution.completed_at.isoformat() if execution.completed_at else None,
            "step_results": execution.step_results,
            "output_data": execution.output_data,
            "error_message": execution.error_message
        }

    def list_workflows(self) -> List[str]:
        """List available workflows."""
        return list(self.workflows.keys())

    def get_active_executions(self) -> List[str]:
        """Get list of active execution IDs."""
        return list(self.active_executions)

    async def shutdown_all(self) -> None:
        """Shutdown the workflow orchestrator."""
        self.logger.info("Shutting down Workflow Orchestrator")
        
        # Cancel active executions
        for execution_id in list(self.active_executions):
            execution = self.executions.get(execution_id)
            if execution:
                execution.status = WorkflowStatus.CANCELLED
        
        self.active_executions.clear()
        
        self.logger.info("Workflow Orchestrator shutdown complete")


# =============================================================================
# MAIN CORE ASSISTANT ENGINE
# =============================================================================

class CoreAssistantEngine:
    """
    Consolidated core orchestration engine for the AI assistant.
    
    This class coordinates all major subsystems including:
    - Component lifecycle management and health monitoring
    - Natural language processing and multimodal processing
    - Session management with automatic cleanup
    - Multi-modal user interaction handling
    - Plugin discovery and lifecycle management
    - Workflow execution and orchestration
    - Memory management and skill execution
    - Learning and adaptation capabilities
    - Reasoning and planning
    """
    
    def __init__(self, container: Container):
        """Initialize the consolidated core assistant engine."""
        self.container = container
        self.logger = get_logger(__name__)
        
        # Core state
        self.state = AssistantState.UNINITIALIZED
        self.active_contexts: Dict[str, AssistantContext] = {}
        
        # Initialize all component managers
        self.component_manager = EnhancedComponentManager(container)
        self.session_manager = EnhancedSessionManager(container)
        self.interaction_handler = InteractionHandler(container)
        self.plugin_manager = EnhancedPluginManager(container)
        self.workflow_orchestrator = WorkflowOrchestrator(container)
        
        # Component references (will be injected)
        self.event_bus: Optional[EventBus] = None
        self.model_coordinator: Optional[ModelInferenceCoordinator] = None
        self.memory_manager: Optional[MemoryManager] = None
        self.bilingual_manager: Optional[BilingualManager] = None
        self.skill_registry: Optional[SkillRegistry] = None
        self.logic_engine: Optional[LogicEngine] = None
        self.task_planner: Optional[TaskPlanner] = None
        self.learning_engine: Optional[ContinualLearningEngine] = None
        self.preference_engine: Optional[PreferenceLearningEngine] = None
        self.metrics_collector: Optional[MetricsCollector] = None
        self.multimodal_fusion: Optional[MultimodalFusionStrategy] = None
        
        # Performance tracking
        self.request_count = 0
        self.total_processing_time = 0.0
        self.skill_execution_stats: Dict[str, Dict[str, Any]] = {}
        
        # Configuration
        self.max_concurrent_requests = 10
        self.default_timeout = 30.0
        self.enable_learning = True
        self.enable_multimodal = True
        self.enable_workflows = True
        
    async def initialize(self) -> None:
        """Initialize the consolidated assistant engine and all its components."""
        try:
            self.state = AssistantState.INITIALIZING
            self.logger.info("Initializing Consolidated Core Assistant Engine")
            
            # Inject dependencies
            await self._inject_dependencies()
            
            # Initialize all component managers
            await self._initialize_component_managers()
            
            # Initialize subsystems
            await self._initialize_subsystems()
            
            # Register event handlers
            self._register_event_handlers()
            
            # Perform health check
            health_status = await self._perform_health_check()
            if not health_status["healthy"]:
                raise AIAssistantError(
                    "Health check failed during initialization",
                    ErrorCategory.SYSTEM,
                    ErrorSeverity.HIGH,
                    {"health_status": health_status}
                )
            
            self.state = AssistantState.READY
            self.logger.info("Consolidated Core Assistant Engine initialized successfully")
            
            # Emit initialization event
            await self.event_bus.emit(SystemHealthCheck(
                component="ConsolidatedCoreAssistantEngine",
                status="initialized",
                details={"state": self.state.value}
            ))
            
        except Exception as e:
            self.state = AssistantState.ERROR
            self.logger.error(f"Failed to initialize consolidated assistant engine: {str(e)}")
            raise
    
    async def _inject_dependencies(self) -> None:
        """Inject required dependencies from the container."""
        self.event_bus = await self.container.get(EventBus)
        self.model_coordinator = await self.container.get(ModelInferenceCoordinator)
        self.memory_manager = await self.container.get(MemoryManager)
        self.bilingual_manager = await self.container.get(BilingualManager)
        self.skill_registry = await self.container.get(SkillRegistry)
        self.logic_engine = await self.container.get(LogicEngine)
        self.task_planner = await self.container.get(TaskPlanner)
        self.learning_engine = await self.container.get(ContinualLearningEngine)
        self.preference_engine = await self.container.get(PreferenceLearningEngine)
        self.metrics_collector = await self.container.get(MetricsCollector)
        self.multimodal_fusion = await self.container.get(MultimodalFusionStrategy)
    
    async def _initialize_component_managers(self) -> None:
        """Initialize all component managers."""
        await self.component_manager.initialize()
        await self.session_manager.initialize()
        await self.interaction_handler.initialize()
        await self.plugin_manager.initialize()
        await self.workflow_orchestrator.initialize()
    
    async def _initialize_subsystems(self) -> None:
        """Initialize all subsystems in the correct order."""
        # Initialize memory manager
        if hasattr(self.memory_manager, 'initialize'):
            await self.memory_manager.initialize()
        
        # Initialize skill registry
        if hasattr(self.skill_registry, 'initialize'):
            await self.skill_registry.initialize()
        
        # Initialize learning engines
        if self.enable_learning:
            if hasattr(self.learning_engine, 'initialize'):
                await self.learning_engine.initialize()
            if hasattr(self.preference_engine, 'initialize'):
                await self.preference_engine.initialize()
    
    def _register_event_handlers(self) -> None:
        """Register handlers for system events."""
        self.event_bus.subscribe(SessionStarted, self._handle_session_started)
        self.event_bus.subscribe(SessionEnded, self._handle_session_ended)
        self.event_bus.subscribe(TaskCompleted, self._handle_task_completed)
        self.event_bus.subscribe(SkillExecuted, self._handle_skill_executed)
        self.event_bus.subscribe(WorkflowStarted, self._handle_workflow_started)
        self.event_bus.subscribe(WorkflowCompleted, self._handle_workflow_completed)
    
    async def create_session(
        self,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AssistantContext:
        """Create a new comprehensive assistant session."""
        # Create session through session manager
        session_id = await self.session_manager.create_session(
            user_id=user_id,
            initial_context=metadata or {}
        )
        
        # Create assistant context
        context = AssistantContext(
            session_id=session_id,
            user_id=user_id,
            metadata=metadata or {}
        )
        
        self.active_contexts[session_id] = context
        
        # Initialize session in memory manager
        await self.memory_manager.initialize_session(session_id)
        
        self.logger.info(f"Created comprehensive session: {session_id}")
        return context
    
    async def process_input(
        self,
        request: ProcessingRequest
    ) -> ProcessingResponse:
        """
        Process user input through the comprehensive assistant pipeline.
        
        This method integrates all components for complete processing including:
        - Multimodal input processing
        - Session management 
        - Interaction handling
        - Workflow execution (if applicable)
        - Plugin utilization
        """
        start_time = asyncio.get_event_loop().time()
        
        try:
            self.state = AssistantState.PROCESSING
            self.request_count += 1
            
            # Update session context
            request.context.last_interaction = datetime.now(timezone.utc)
            await self.session_manager.update_session_context(
                request.context.session_id,
                {"last_request": request.request_id}
            )
            
            # Start interaction handling
            interaction_id = await self.interaction_handler.start_interaction(
                user_id=request.context.user_id,
                session_id=request.context.session_id,
                interaction_mode=InteractionMode.CONVERSATIONAL,
                input_modalities={InputModality.TEXT},
                output_modalities={OutputModality.TEXT}
            )
            
            # Log request
            self.logger.info(
                f"Processing request {request.request_id} "
                f"for session {request.context.session_id} "
                f"with interaction {interaction_id}"
            )
            
            # Check if this should trigger a workflow
            workflow_id = await self._determine_workflow(request)
            if workflow_id and self.enable_workflows:
                response = await self._process_with_workflow(request, workflow_id)
            else:
                # Route based on input type for standard processing
                if request.input_type == "text":
                    response = await self._process_text_input(request)
                elif request.input_type == "multimodal":
                    response = await self._process_multimodal_input(request)
                elif request.input_type == "speech":
                    response = await self._process_speech_input(request)
                elif request.input_type == "vision":
                    response = await self._process_vision_input(request)
                else:
                    raise ValueError(f"Unsupported input type: {request.input_type}")
            
            # Post-processing
            response = await self._post_process_response(response, request)
            
            # Update metrics
            processing_time = asyncio.get_event_loop().time() - start_time
            response.processing_time = processing_time
            self.total_processing_time += processing_time
            
            # Track metrics
            await self.metrics_collector.record_metric(
                "assistant.request.processed",
                1,
                {
                    "input_type": request.input_type,
                    "session_id": request.context.session_id,
                    "processing_mode": request.context.processing_mode.value,
                    "interaction_id": interaction_id
                }
            )
            
            # End interaction
            await self.interaction_handler.end_interaction(interaction_id)
            
            self.state = AssistantState.READY
            return response
            
        except Exception as e:
            self.state = AssistantState.ERROR
            self.logger.error(
                f"Error processing request {request.request_id}: {str(e)}"
            )
            return ProcessingResponse(
                response_data=f"I apologize, but I encountered an error: {str(e)}",
                response_type="error",
                confidence=0.0,
                skills_used=[],
                metadata={"error": str(e)}
            )
    
    async def _determine_workflow(self, request: ProcessingRequest) -> Optional[str]:
        """Determine if the request should trigger a workflow."""
        if isinstance(request.input_data, str):
            text = request.input_data.lower()
            
            # Simple workflow triggers
            if any(word in text for word in ["task", "remind", "schedule"]):
                return "task_management"
            elif any(word in text for word in ["hello", "hi", "greet"]):
                return "greeting_workflow"
        
        return None
    
    async def _process_with_workflow(
        self,
        request: ProcessingRequest,
        workflow_id: str
    ) -> ProcessingResponse:
        """Process request using workflow orchestration."""
        try:
            # Execute workflow
            execution_id = await self.workflow_orchestrator.execute_workflow(
                workflow_id=workflow_id,
                input_data={"user_input": request.input_data},
                session_id=request.context.session_id,
                user_id=request.context.user_id
            )
            
            # Wait for completion (simplified - in real implementation would be async)
            await asyncio.sleep(1.0)  # Give workflow time to complete
            
            # Get execution status
            status = await self.workflow_orchestrator.get_execution_status(execution_id)
            
            if status.get("status") == "completed":
                return ProcessingResponse(
                    response_data=f"I've successfully completed the {workflow_id} workflow for you.",
                    response_type="workflow_result",
                    confidence=0.95,
                    skills_used=[workflow_id],
                    metadata={"workflow_execution": execution_id, "workflow_status": status}
                )
            else:
                return ProcessingResponse(
                    response_data=f"I'm working on your request using the {workflow_id} workflow.",
                    response_type="workflow_progress",
                    confidence=0.8,
                    skills_used=[workflow_id],
                    metadata={"workflow_execution": execution_id, "workflow_status": status}
                )
                
        except Exception as e:
            self.logger.error(f"Workflow processing failed: {e}")
            # Fall back to standard processing
            return await self._process_text_input(request)
    
    async def _process_text_input(
        self,
        request: ProcessingRequest
    ) -> ProcessingResponse:
        """Process text input through the NLP pipeline."""
        text = request.input_data if isinstance(request.input_data, str) else str(request.input_data)
        
        # Language detection and context building
        language_context = self.bilingual_manager.build_language_context(text)
        
        # Update conversation history in session
        await self.session_manager.add_conversation_entry(
            request.context.session_id,
            {
                "role": "user",
                "content": text,
                "language": language_context.user_query_language.value
            }
        )
        
        # Memory storage
        await self.memory_manager.store_memory(
            data={
                "type": "user_input",
                "content": text,
                "language": language_context.user_query_language.value,
                "session_id": request.context.session_id
            },
            memory_type="working",
            session_id=request.context.session_id
        )
        
        # Intent and entity extraction
        intent_data = {"intent": "general_conversation", "confidence": 0.8}
        entities = []
        
        # Reasoning and planning
        reasoning_result = await self.logic_engine.reason(
            context={
                "user_input": text,
                "intent": intent_data,
                "entities": entities,
                "conversation_history": request.context.conversation_history
            }
        )
        
        # Task planning
        if reasoning_result.get("requires_planning"):
            task_plan = await self.task_planner.create_plan(
                goal=reasoning_result.get("goal"),
                context=request.context
            )
        else:
            task_plan = None
        
        # Skill selection and execution with plugin support
        selected_skills = await self._select_skills_with_plugins(
            intent_data,
            entities,
            request.context
        )
        
        skill_results = []
        for skill_id in selected_skills:
            result = await self._execute_skill(
                skill_id,
                {
                    "input": text,
                    "context": request.context,
                    "language_context": language_context
                }
            )
            skill_results.append(result)
        
        # Generate response using model coordinator
        model_response = await self.model_coordinator.generate_response(
            prompt=self._build_prompt(text, request.context, skill_results),
            context={
                "session_id": request.context.session_id,
                "language": language_context.user_query_language.value,
                "skills_used": selected_skills
            }
        )
        
        # Format response based on language preference
        formatted_response = self.bilingual_manager.process_response(
            model_response["text"],
            language_context
        )
        
        # Update conversation history in session
        await self.session_manager.add_conversation_entry(
            request.context.session_id,
            {
                "role": "assistant",
                "content": formatted_response,
                "skills_used": selected_skills
            }
        )
        
        # Store in memory
        await self.memory_manager.store_memory(
            data={
                "type": "assistant_response",
                "content": formatted_response,
                "skills_used": selected_skills,
                "session_id": request.context.session_id
            },
            memory_type="working",
            session_id=request.context.session_id
        )
        
        return ProcessingResponse(
            response_data=formatted_response,
            response_type="text",
            confidence=model_response.get("confidence", 0.8),
            skills_used=selected_skills,
            reasoning_path=reasoning_result.get("path"),
            metadata={
                "language": language_context.user_query_language.value,
                "model_used": model_response.get("model"),
                "task_plan": task_plan
            }
        )
    
    async def _process_multimodal_input(
        self,
        request: ProcessingRequest
    ) -> ProcessingResponse:
        """Process multimodal input using fusion strategies."""
        if not self.enable_multimodal:
            return await self._process_text_input(request)
        
        # Extract modalities
        modalities = request.input_data if isinstance(request.input_data, dict) else {"text": request.input_data}
        
        # Apply multimodal fusion
        fused_representation = await self.multimodal_fusion.fuse(modalities)
        
        # Process fused representation
        text_representation = fused_representation.get("text", str(modalities))
        
        # Create new request with fused data
        text_request = ProcessingRequest(
            input_data=text_representation,
            input_type="text",
            context=request.context,
            metadata={**request.metadata, "original_modalities": list(modalities.keys())}
        )
        
        return await self._process_text_input(text_request)
    
    async def _process_speech_input(
        self,
        request: ProcessingRequest
    ) -> ProcessingResponse:
        """Process speech input through speech-to-text pipeline."""
        # This would involve speech processing
        return await self._process_text_input(request)
    
    async def _process_vision_input(
        self,
        request: ProcessingRequest
    ) -> ProcessingResponse:
        """Process vision input through vision pipeline."""
        # This would involve vision processing
        return await self._process_text_input(request)
    
    async def _select_skills_with_plugins(
        self,
        intent_data: Dict[str, Any],
        entities: List[Dict[str, Any]],
        context: AssistantContext
    ) -> List[str]:
        """Select appropriate skills including plugin-provided skills."""
        # Get available skills from registry
        available_skills = await self.skill_registry.get_available_skills()
        
        # Get enabled plugins that might provide additional skills
        enabled_plugins = self.plugin_manager.get_enabled_plugins()
        
        # Filter by intent
        relevant_skills = []
        for skill_id, skill_info in available_skills.items():
            if intent_data["intent"] in skill_info.get("supported_intents", []):
                relevant_skills.append(skill_id)
        
        # Add plugin-enhanced skills
        for plugin_id in enabled_plugins:
            # This would normally query plugin for additional skills
            relevant_skills.append(f"plugin_{plugin_id}_skill")
        
        # If no specific skills found, use general skills
        if not relevant_skills:
            relevant_skills = ["core_skills", "productivity_skills"]
        
        return relevant_skills[:3]  # Limit to top 3 skills
    
    async def _execute_skill(
        self,
        skill_id: str,
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a specific skill with enhanced tracking."""
        try:
            # Check if this is a plugin skill
            if skill_id.startswith("plugin_"):
                # Handle plugin skill execution
                return await self._execute_plugin_skill(skill_id, input_data)
            
            skill = await self.skill_registry.get_skill(skill_id)
            if not skill:
                return {"error": f"Skill {skill_id} not found"}
            
            result = await skill.execute(input_data)
            
            # Track execution statistics
            if skill_id not in self.skill_execution_stats:
                self.skill_execution_stats[skill_id] = {
                    "executions": 0,
                    "successes": 0,
                    "failures": 0
                }
            
            self.skill_execution_stats[skill_id]["executions"] += 1
            if result.get("success", True):
                self.skill_execution_stats[skill_id]["successes"] += 1
            else:
                self.skill_execution_stats[skill_id]["failures"] += 1
            
            # Emit skill executed event
            await self.event_bus.emit(SkillExecuted(
                skill_id=skill_id,
                session_id=input_data["context"].session_id,
                success=result.get("success", True)
            ))
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error executing skill {skill_id}: {str(e)}")
            return {"error": str(e), "success": False}
    
    async def _execute_plugin_skill(
        self,
        skill_id: str,
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a plugin-provided skill."""
        # Extract plugin ID from skill ID
        plugin_id = skill_id.replace("plugin_", "").replace("_skill", "")
        
        # Check if plugin is enabled
        if plugin_id not in self.plugin_manager.get_enabled_plugins():
            return {"error": f"Plugin {plugin_id} not enabled", "success": False}
        
        # Simulate plugin skill execution
        await asyncio.sleep(0.1)
        return {
            "result": f"Executed plugin skill from {plugin_id}",
            "plugin_id": plugin_id,
            "success": True
        }
    
    def _build_prompt(
        self,
        user_input: str,
        context: AssistantContext,
        skill_results: List[Dict[str, Any]]
    ) -> str:
        """Build enhanced prompt for model inference."""
        # Build conversation context
        conversation = "\n".join([
            f"{msg['role']}: {msg['content']}"
            for msg in context.conversation_history[-5:]  # Last 5 messages
        ])
        
        # Build skill context
        skill_context = ""
        if skill_results:
            skill_context = "\nRelevant information from skills:\n"
            for result in skill_results:
                if result.get("output"):
                    skill_context += f"- {result['output']}\n"
        
        # Add plugin information
        enabled_plugins = self.plugin_manager.get_enabled_plugins()
        plugin_context = ""
        if enabled_plugins:
            plugin_context = f"\nEnabled plugins: {', '.join(enabled_plugins)}\n"
        
        # Construct enhanced prompt
        prompt = f"""Conversation context:
{conversation}

Current user input: {user_input}
{skill_context}{plugin_context}
Please provide a helpful and contextual response in {context.language_preference.value}."""
        
        return prompt
    
    async def _post_process_response(
        self,
        response: ProcessingResponse,
        request: ProcessingRequest
    ) -> ProcessingResponse:
        """Post-process the response with enhanced capabilities."""
        # Learn from interaction if enabled
        if self.enable_learning:
            await self._learn_from_interaction(request, response)
        
        # Update preferences
        await self.preference_engine.update_preferences(
            user_id=request.context.user_id or "anonymous",
            interaction_data={
                "input_type": request.input_type,
                "processing_mode": request.context.processing_mode.value,
                "language": request.context.language_preference.value,
                "skills_used": response.skills_used
            }
        )
        
        return response
    
    async def _learn_from_interaction(
        self,
        request: ProcessingRequest,
        response: ProcessingResponse
    ) -> None:
        """Learn from the interaction for future improvements."""
        learning_data = {
            "input": request.input_data,
            "output": response.response_data,
            "context": {
                "session_id": request.context.session_id,
                "processing_mode": request.context.processing_mode.value,
                "skills_used": response.skills_used,
                "components_used": {
                    "session_manager": True,
                    "interaction_handler": True,
                    "plugin_manager": len(self.plugin_manager.get_enabled_plugins()) > 0,
                    "workflow_orchestrator": self.enable_workflows
                }
            },
            "metadata": {
                **request.metadata,
                **response.metadata
            }
        }
        
        # Submit to learning engine
        await self.learning_engine.learn_from_interaction(learning_data)
        
        # Emit learning event
        await self.event_bus.emit(LearningEventOccurred(
            event_type="enhanced_interaction",
            data=learning_data,
            session_id=request.context.session_id
        ))
    
    async def end_session(self, session_id: str) -> None:
        """End a comprehensive assistant session."""
        if session_id not in self.active_contexts:
            self.logger.warning(f"Attempted to end non-existent session: {session_id}")
            return
        
        context = self.active_contexts[session_id]
        
        # End session through session manager
        await self.session_manager.end_session(session_id)
        
        # Clean up session in memory manager
        await self.memory_manager.cleanup_session(session_id)
        
        # Remove from active contexts
        del self.active_contexts[session_id]
        
        self.logger.info(f"Ended comprehensive session: {session_id}")
    
    async def _perform_health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check on all components."""
        health_status = {
            "healthy": True,
            "components": {},
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        # Check core components
        core_components = {
            "event_bus": self.event_bus,
            "model_coordinator": self.model_coordinator,
            "memory_manager": self.memory_manager,
            "skill_registry": self.skill_registry,
            "logic_engine": self.logic_engine
        }
        
        # Check component managers
        manager_components = {
            "component_manager": self.component_manager,
            "session_manager": self.session_manager,
            "interaction_handler": self.interaction_handler,
            "plugin_manager": self.plugin_manager,
            "workflow_orchestrator": self.workflow_orchestrator
        }
        
        all_components = {**core_components, **manager_components}
        
        for name, component in all_components.items():
            try:
                if component and hasattr(component, 'health_check'):
                    component_health = await component.health_check()
                    health_status["components"][name] = component_health
                    if not component_health.get("healthy", True):
                        health_status["healthy"] = False
                else:
                    health_status["components"][name] = {"healthy": True}
            except Exception as e:
                health_status["components"][name] = {
                    "healthy": False,
                    "error": str(e)
                }
                health_status["healthy"] = False
        
        # Add component manager specific status
        health_status["component_manager_status"] = self.component_manager.get_component_status()
        health_status["session_statistics"] = self.session_manager.get_session_statistics()
        health_status["plugin_status"] = await self.plugin_manager.get_plugin_status()
        health_status["active_workflows"] = self.workflow_orchestrator.get_active_executions()
        
        return health_status
    
    async def get_status(self) -> Dict[str, Any]:
        """Get comprehensive current status of the assistant engine."""
        return {
            "state": self.state.value,
            "active_sessions": len(self.active_contexts),
            "total_requests": self.request_count,
            "average_processing_time": (
                self.total_processing_time / self.request_count 
                if self.request_count > 0 else 0
            ),
            "skill_stats": self.skill_execution_stats,
            "component_manager": self.component_manager.get_component_status(),
            "session_manager": self.session_manager.get_session_statistics(),
            "active_interactions": self.interaction_handler.get_active_interactions(),
            "plugin_manager": await self.plugin_manager.get_plugin_status(),
            "workflow_manager": {
                "available_workflows": self.workflow_orchestrator.list_workflows(),
                "active_executions": self.workflow_orchestrator.get_active_executions()
            },
            "capabilities": {
                "multimodal_processing": self.enable_multimodal,
                "learning_enabled": self.enable_learning,
                "workflow_orchestration": self.enable_workflows,
                "plugin_system": True,
                "session_management": True,
                "interaction_handling": True
            },
            "health": await self._perform_health_check()
        }
    
    # Enhanced Event Handlers
    async def _handle_session_started(self, event: SessionStarted) -> None:
        """Handle session started event."""
        self.logger.info(f"Enhanced session started: {event.session_id}")
    
    async def _handle_session_ended(self, event: SessionEnded) -> None:
        """Handle session ended event."""
        self.logger.info(f"Enhanced session ended: {event.session_id}, duration: {event.duration}s")
    
    async def _handle_task_completed(self, event: TaskCompleted) -> None:
        """Handle task completed event."""
        self.logger.info(f"Enhanced task completed: {event.task_id}")
    
    async def _handle_skill_executed(self, event: SkillExecuted) -> None:
        """Handle skill executed event."""
        self.logger.debug(f"Enhanced skill executed: {event.skill_id}")
    
    async def _handle_workflow_started(self, event: WorkflowStarted) -> None:
        """Handle workflow started event."""
        self.logger.info(f"Workflow started: {event.workflow_id}, execution: {event.execution_id}")
    
    async def _handle_workflow_completed(self, event: WorkflowCompleted) -> None:
        """Handle workflow completed event."""
        self.logger.info(f"Workflow completed: {event.workflow_id}, success: {event.success}")
    
    async def shutdown(self) -> None:
        """Shutdown the consolidated assistant engine gracefully."""
        self.logger.info("Shutting down Consolidated Core Assistant Engine")
        self.state = AssistantState.SHUTTING_DOWN
        
        # End all active sessions
        session_ids = list(self.active_contexts.keys())
        for session_id in session_ids:
            await self.end_session(session_id)
        
        # Shutdown all component managers
        await self.workflow_orchestrator.shutdown_all()
        await self.plugin_manager.shutdown()
        await self.interaction_handler.cleanup()
        await self.session_manager.cleanup()
        await self.component_manager.shutdown_all()
        
        # Shutdown core subsystems
        for component in [
            self.memory_manager,
            self.skill_registry,
            self.learning_engine,
            self.preference_engine
        ]:
            if component and hasattr(component, 'shutdown'):
                await component.shutdown()
        
        self.state = AssistantState.SHUTDOWN
        self.logger.info("Consolidated Core Assistant Engine shutdown complete")
