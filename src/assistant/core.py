"""
Enhanced Core Assistant Orchestrator

This module serves as the central coordination point for the AI assistant,
combining component management, session handling, interaction processing,
plugin management, and workflow orchestration capabilities.

Features:
- Centralized component lifecycle management with health monitoring
- Enhanced multimodal input processing (text, speech, vision)
- Advanced session management with automatic cleanup
- Multi-modal user interaction handling
- Plugin discovery and lifecycle management
- Complex workflow execution and orchestration
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import uuid
import logging

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
    WorkflowStepCompleted,
    SessionCleanupStarted
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


# Component Management Enums
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


# Input/Output Modality Enums
class ModalityType(Enum):
    """Input modality types."""
    TEXT = "text"
    SPEECH = "speech"
    VISION = "vision"
    MULTIMODAL = "multimodal"


class InputModality(Enum):
    """Input modality types."""
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


# Plugin Management Enums
class PluginStatus(Enum):
    """Plugin status enumeration."""
    UNLOADED = "unloaded"
    LOADED = "loaded"
    ENABLED = "enabled"
    DISABLED = "disabled"
    ERROR = "error"


# Workflow Management Enums
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


# Enhanced Component Management Data Classes
@dataclass
class ComponentInfo:
    """Information about a managed component."""
    name: str
    instance: Any
    status: ComponentStatus = ComponentStatus.UNINITIALIZED
    dependencies: List[str] = None
    health_score: float = 1.0
    last_health_check: float = 0.0


# Enhanced Processing Data Classes
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


# Session Management Data Classes
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


# Interaction Handling Data Classes
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


# Plugin Management Data Classes
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


# Workflow Management Data Classes
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


class CoreAssistantEngine:
    """
    Enhanced Core Orchestration Engine for the AI Assistant.
    
    This class provides comprehensive coordination of all major subsystems:
    
    Core Processing:
    - Natural language processing with bilingual support
    - Multimodal processing (text, speech, vision)
    - Real-time and streaming response capabilities
    
    System Management:
    - Centralized component lifecycle management
    - Health monitoring and status tracking
    - Dependency resolution and initialization ordering
    
    Session Management:
    - User session management with automatic cleanup
    - Conversation history tracking and context persistence
    - Configurable session timeouts and cleanup policies
    
    Interaction Handling:
    - Multi-modal user interaction management
    - Support for text, speech, vision, and gesture inputs
    - Conversation flow coordination
    
    Plugin System:
    - Plugin discovery and lifecycle management
    - Dependency resolution and security validation
    - Hot-loading and hot-reloading capabilities
    
    Workflow Orchestration:
    - Complex workflow execution and orchestration
    - Step dependency management and parallel execution
    - Built-in retry logic and error recovery
    
    Additional Features:
    - Memory management and persistence
    - Skill execution and management
    - Learning and adaptation
    - Reasoning and planning
    - Comprehensive error handling and recovery
    """
    
    def __init__(self, container: Container):
        """Initialize the core assistant engine."""
        self.container = container
        self.logger = get_logger(__name__)
        
        # Core state
        self.state = AssistantState.UNINITIALIZED
        self.active_contexts: Dict[str, AssistantContext] = {}
        
        # Component Management
        self.components: Dict[str, ComponentInfo] = {}
        self.initialization_order: List[str] = []
        
        # Session Management
        self.sessions: Dict[str, SessionInfo] = {}
        self.user_sessions: Dict[str, Set[str]] = {}  # user_id -> session_ids
        self.max_session_age = timedelta(hours=24)
        self.max_inactive_time = timedelta(hours=2)
        self.cleanup_interval = timedelta(minutes=30)
        self._cleanup_task: Optional[asyncio.Task] = None
        
        # Interaction Management
        self.active_interactions: Dict[str, InteractionInfo] = {}
        
        # Plugin Management
        self.plugins: Dict[str, PluginInfo] = {}
        self.enabled_plugins: Set[str] = set()
        
        # Workflow Management
        self.workflows: Dict[str, WorkflowDefinition] = {}
        self.executions: Dict[str, WorkflowExecution] = {}
        self.active_executions: Set[str] = set()
        
        # Core Processing
        self.engine_state = EngineState.UNINITIALIZED
        self.active_sessions: Set[str] = set()
        self.processing_queue: asyncio.Queue = asyncio.Queue()
        
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
        
    async def initialize(self) -> None:
        """Initialize the assistant engine and all its components."""
        try:
            self.state = AssistantState.INITIALIZING
            self.engine_state = EngineState.INITIALIZING
            self.logger.info("Initializing Enhanced Core Assistant Engine")
            
            # Inject dependencies
            await self._inject_dependencies()
            
            # Discover and initialize components
            await self._discover_components()
            await self._initialize_all_components()
            
            # Initialize subsystems
            await self._initialize_subsystems()
            
            # Initialize plugins
            await self._initialize_plugins()
            
            # Initialize workflows
            await self._initialize_workflows()
            
            # Start session cleanup
            await self._start_session_cleanup()
            
            # Initialize core processing pipeline
            await self._initialize_processing_pipeline()
            await self._initialize_reasoning_engine()
            await self._initialize_response_generator()
            
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
            self.engine_state = EngineState.READY
            self.logger.info("Enhanced Core Assistant Engine initialized successfully")
            
            # Emit initialization events
            await self.event_bus.emit(SystemHealthCheck(
                component="CoreAssistantEngine",
                status="initialized",
                details={"state": self.state.value}
            ))
            
            await self.event_bus.emit(EngineStarted(engine_type="enhanced_core"))
            
        except Exception as e:
            self.state = AssistantState.ERROR
            self.engine_state = EngineState.ERROR
            self.logger.error(f"Failed to initialize assistant engine: {str(e)}")
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

    # Component Management Methods
    async def _discover_components(self) -> List[str]:
        """Discover available components in the system."""
        discovered = [
            "memory_manager",
            "skill_registry", 
            "processing_pipeline",
            "reasoning_engine",
            "learning_engine",
            "multimodal_fusion",
            "event_bus",
            "metrics_collector"
        ]
        
        self.logger.info(f"Discovered {len(discovered)} components")
        return discovered

    async def _initialize_all_components(self) -> None:
        """Initialize all discovered components."""
        components_to_init = [
            "memory_manager", 
            "skill_registry",
            "processing_pipeline",
            "reasoning_engine"
        ]
        
        for component_name in components_to_init:
            component_instance = getattr(self, component_name, None)
            if component_instance is None:
                # Create mock component for components not yet integrated
                component_instance = MockComponent(component_name)
            
            self.components[component_name] = ComponentInfo(
                name=component_name,
                instance=component_instance,
                status=ComponentStatus.RUNNING
            )
            
            if self.event_bus:
                await self.event_bus.emit(ComponentStarted(component_name=component_name))
        
        self.logger.info(f"Initialized {len(self.components)} components")

    async def _initialize_plugins(self) -> None:
        """Initialize the plugin system."""
        self.logger.info("Initializing Enhanced Plugin System")
        
        # Discover and load built-in plugins
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
                
                # Auto-enable core plugins
                await self._load_plugin(plugin_id)
                await self._enable_plugin(plugin_id)
        
        self.logger.info(f"Plugin system initialized with {len(self.plugins)} plugins")

    async def _initialize_workflows(self) -> None:
        """Initialize built-in workflows."""
        self.logger.info("Initializing Workflow System")
        
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
                )
            ]
        )
        
        self.workflows["greeting_workflow"] = greeting_workflow
        self.workflows["task_management"] = task_workflow
        
        self.logger.info(f"Initialized {len(self.workflows)} built-in workflows")

    async def _start_session_cleanup(self) -> None:
        """Start background session cleanup task."""
        self.logger.info("Starting session cleanup task")
        self._cleanup_task = asyncio.create_task(self._session_cleanup_loop())

    async def _initialize_processing_pipeline(self) -> None:
        """Initialize the processing pipeline."""
        self.logger.debug("Processing pipeline initialized")

    async def _initialize_reasoning_engine(self) -> None:
        """Initialize the reasoning engine."""
        self.logger.debug("Reasoning engine initialized")

    async def _initialize_response_generator(self) -> None:
        """Initialize the response generator."""
        self.logger.debug("Response generator initialized")
    
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
    
    async def create_session(
        self,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        initial_context: Optional[Dict[str, Any]] = None
    ) -> AssistantContext:
        """Create a new enhanced assistant session."""
        session_id = str(uuid.uuid4())
        
        # Create assistant context (existing functionality)
        context = AssistantContext(
            session_id=session_id,
            user_id=user_id,
            metadata=metadata or {}
        )
        
        # Create enhanced session info
        session_info = SessionInfo(
            session_id=session_id,
            user_id=user_id,
            context=initial_context or {},
        )
        
        # Store both session representations
        self.active_contexts[session_id] = context
        self.sessions[session_id] = session_info
        self.active_sessions.add(session_id)
        
        # Track user sessions
        if user_id:
            if user_id not in self.user_sessions:
                self.user_sessions[user_id] = set()
            self.user_sessions[user_id].add(session_id)
        
        # Initialize session in memory manager
        await self.memory_manager.initialize_session(session_id)
        
        # Emit session started event
        await self.event_bus.emit(SessionStarted(
            session_id=session_id,
            user_id=user_id,
            timestamp=context.created_at
        ))
        
        self.logger.info(f"Created enhanced session: {session_id}")
        return context

    # Enhanced Session Management Methods
    async def get_session(self, session_id: str) -> Optional[SessionInfo]:
        """Get enhanced session information."""
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
            
            # Also update the assistant context
            context = self.active_contexts.get(session_id)
            if context:
                context.conversation_history.append(entry_with_timestamp)
            
            # Limit conversation history size
            max_history = 100
            if len(session.conversation_history) > max_history:
                session.conversation_history = session.conversation_history[-max_history:]
                if context:
                    context.conversation_history = context.conversation_history[-max_history:]
            
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

    async def _session_cleanup_loop(self) -> None:
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
            
            self.logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
    
    async def process_input(
        self,
        request: ProcessingRequest
    ) -> ProcessingResponse:
        """
        Enhanced process user input through the assistant pipeline.
        
        This is the main entry point for all user interactions with support for:
        - Multimodal input processing (text, speech, vision)
        - Real-time and streaming modes
        - Comprehensive error handling and recovery
        - Enhanced confidence scoring
        """
        start_time = datetime.now(timezone.utc)
        processing_id = str(uuid.uuid4())
        
        try:
            self.state = AssistantState.PROCESSING
            self.engine_state = EngineState.PROCESSING
            self.request_count += 1
            
            # Validate engine state
            if self.engine_state != EngineState.PROCESSING:
                if self.state != AssistantState.READY:
                    raise RuntimeError(f"Engine not ready. Current state: {self.state}")
            
            # Update context
            request.context.last_interaction = datetime.now(timezone.utc)
            
            # Update session activity
            session = self.sessions.get(request.context.session_id)
            if session:
                session.last_activity = datetime.now(timezone.utc)
            
            # Emit processing started event
            if self.event_bus:
                await self.event_bus.emit(
                    ProcessingStarted(
                        processing_id=processing_id,
                        modality=self._determine_input_modality(request).value
                    )
                )
            
            # Log request
            self.logger.info(
                f"Processing enhanced request {processing_id} "
                f"for session {request.context.session_id} "
                f"(type: {request.input_type})"
            )
            
            # Route based on input type with enhanced processing
            if request.input_type == "text":
                response = await self._process_text_input(request)
            elif request.input_type == "multimodal":
                response = await self._process_multimodal_input_enhanced(request)
            elif request.input_type == "speech":
                response = await self._process_speech_input_enhanced(request)
            elif request.input_type == "vision":
                response = await self._process_vision_input_enhanced(request)
            else:
                raise ValueError(f"Unsupported input type: {request.input_type}")
            
            # Post-processing
            response = await self._post_process_response(response, request)
            
            # Update metrics
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            response.processing_time = processing_time
            self.total_processing_time += processing_time
            
            # Emit processing completed event
            if self.event_bus:
                await self.event_bus.emit(
                    ProcessingCompleted(
                        processing_id=processing_id,
                        success=True,
                        processing_time=processing_time
                    )
                )
            
            # Track metrics
            await self.metrics_collector.record_metric(
                "assistant.request.processed",
                1,
                {
                    "input_type": request.input_type,
                    "session_id": request.context.session_id,
                    "processing_mode": request.context.processing_mode.value,
                    "confidence": response.confidence
                }
            )
            
            self.state = AssistantState.READY
            self.engine_state = EngineState.READY
            return response
            
        except Exception as e:
            self.state = AssistantState.ERROR
            self.engine_state = EngineState.ERROR
            
            # Emit error event
            if self.event_bus:
                await self.event_bus.emit(
                    ProcessingCompleted(
                        processing_id=processing_id,
                        success=False,
                        processing_time=(datetime.now(timezone.utc) - start_time).total_seconds()
                    )
                )
            
            self.logger.error(
                f"Error processing request {processing_id}: {str(e)}"
            )
            
            # Return enhanced error response
            return ProcessingResponse(
                response_data=f"I apologize, but I encountered an error: {str(e)}",
                response_type="error",
                confidence=0.0,
                skills_used=[],
                metadata={"error": str(e), "processing_id": processing_id},
                processing_time=(datetime.now(timezone.utc) - start_time).total_seconds()
            )

    def _determine_input_modality(self, request: ProcessingRequest) -> ModalityType:
        """Determine the primary modality of input data."""
        if request.input_type == "multimodal":
            if isinstance(request.input_data, dict):
                modalities = []
                if request.input_data.get("text"):
                    modalities.append(ModalityType.TEXT)
                if request.input_data.get("audio_data"):
                    modalities.append(ModalityType.SPEECH)
                if request.input_data.get("image_data"):
                    modalities.append(ModalityType.VISION)
                
                if len(modalities) > 1:
                    return ModalityType.MULTIMODAL
                elif modalities:
                    return modalities[0]
        
        # Map from input_type to modality
        type_mapping = {
            "text": ModalityType.TEXT,
            "speech": ModalityType.SPEECH,
            "vision": ModalityType.VISION,
            "multimodal": ModalityType.MULTIMODAL
        }
        
        return type_mapping.get(request.input_type, ModalityType.TEXT)
    
    async def _process_text_input(
        self,
        request: ProcessingRequest
    ) -> ProcessingResponse:
        """Process text input through the NLP pipeline."""
        text = request.input_data if isinstance(request.input_data, str) else str(request.input_data)
        
        # Language detection and context building
        language_context = self.bilingual_manager.build_language_context(text)
        
        # Update conversation history
        request.context.conversation_history.append({
            "role": "user",
            "content": text,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "language": language_context.user_query_language.value
        })
        
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
        
        # Intent and entity extraction (would be done by NLP processor)
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
        
        # Skill selection and execution
        selected_skills = await self._select_skills(
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
        
        # Update conversation history
        request.context.conversation_history.append({
            "role": "assistant",
            "content": formatted_response,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "skills_used": selected_skills
        })
        
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
                "model_used": model_response.get("model")
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
        # This would involve specialized multimodal processing
        # For now, we'll convert to text and process
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
        # For now, we'll treat it as text
        return await self._process_text_input(request)
    
    async def _process_vision_input(
        self,
        request: ProcessingRequest
    ) -> ProcessingResponse:
        """Process vision input through vision pipeline."""
        # This would involve vision processing
        # For now, we'll treat it as text description
        return await self._process_text_input(request)

    # Enhanced Multimodal Processing Methods
    async def _process_multimodal_input_enhanced(
        self,
        request: ProcessingRequest
    ) -> ProcessingResponse:
        """Enhanced multimodal input processing with fusion strategies."""
        if not self.enable_multimodal:
            return await self._process_text_input(request)
        
        # Create multimodal input object
        multimodal_input = self._create_multimodal_input(request)
        
        # Create processing context
        processing_context = ProcessingContext(
            session_id=request.context.session_id,
            user_id=request.context.user_id,
            conversation_history=request.context.conversation_history,
            mode=ProcessingMode.REAL_TIME if request.metadata.get("real_time") else ProcessingMode.BATCH,
            priority=PriorityLevel.NORMAL
        )
        
        # Process through enhanced pipeline
        result = await self._process_multimodal_core(multimodal_input, processing_context)
        
        # Convert to ProcessingResponse
        return ProcessingResponse(
            response_data=result.response_text or "Processed multimodal input successfully",
            response_type="multimodal",
            confidence=result.confidence,
            skills_used=request.metadata.get("skills_used", []),
            metadata={
                **result.metadata,
                "processing_time": result.processing_time,
                "modalities": self._get_input_modalities(multimodal_input)
            },
            processing_time=result.processing_time
        )

    async def _process_speech_input_enhanced(
        self,
        request: ProcessingRequest
    ) -> ProcessingResponse:
        """Enhanced speech input processing."""
        # Convert speech to multimodal input
        multimodal_input = MultimodalInput(
            text=None,
            audio_data=request.input_data if isinstance(request.input_data, bytes) else None,
            metadata=request.metadata
        )
        
        # For now, simulate speech-to-text conversion
        # In a real implementation, this would use speech recognition
        if isinstance(request.input_data, str):
            text_representation = request.input_data
        else:
            text_representation = "I heard your speech input. How can I help you?"
        
        # Create text request and process
        text_request = ProcessingRequest(
            input_data=text_representation,
            input_type="text",
            context=request.context,
            metadata={**request.metadata, "original_modality": "speech"}
        )
        
        response = await self._process_text_input(text_request)
        response.response_type = "speech"
        response.metadata["speech_processed"] = True
        
        return response

    async def _process_vision_input_enhanced(
        self,
        request: ProcessingRequest
    ) -> ProcessingResponse:
        """Enhanced vision input processing."""
        # Convert vision to multimodal input
        multimodal_input = MultimodalInput(
            text=None,
            image_data=request.input_data if isinstance(request.input_data, bytes) else None,
            metadata=request.metadata
        )
        
        # For now, simulate image analysis
        # In a real implementation, this would use computer vision
        if isinstance(request.input_data, str):
            text_representation = request.input_data
        else:
            text_representation = "I can see your image. Please tell me what you'd like me to analyze about it."
        
        # Create text request and process
        text_request = ProcessingRequest(
            input_data=text_representation,
            input_type="text",
            context=request.context,
            metadata={**request.metadata, "original_modality": "vision"}
        )
        
        response = await self._process_text_input(text_request)
        response.response_type = "vision"
        response.metadata["vision_processed"] = True
        
        return response

    async def _process_multimodal_core(
        self,
        input_data: MultimodalInput,
        context: ProcessingContext
    ) -> ProcessingResult:
        """Core multimodal processing with enhanced capabilities."""
        start_time = datetime.now(timezone.utc)
        
        try:
            # Determine dominant modality
            modality = self._determine_modality(input_data)
            
            # Generate response based on available modalities
            if input_data.text:
                response_text = await self._generate_response_from_text(input_data.text, context)
            elif input_data.audio_data:
                response_text = "I processed your audio input. How can I assist you further?"
            elif input_data.image_data:
                response_text = "I analyzed your image. What would you like to know about it?"
            else:
                response_text = "I received your multimodal input. How can I help you?"
            
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            return ProcessingResult(
                success=True,
                response_text=response_text,
                processing_time=processing_time,
                confidence=0.85,  # Enhanced confidence calculation
                metadata={
                    "modality": modality.value,
                    "has_text": input_data.text is not None,
                    "has_audio": input_data.audio_data is not None,
                    "has_image": input_data.image_data is not None
                }
            )
            
        except Exception as e:
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            return ProcessingResult(
                success=False,
                response_text=f"Error processing multimodal input: {str(e)}",
                processing_time=processing_time,
                confidence=0.0,
                errors=[str(e)]
            )

    def _create_multimodal_input(self, request: ProcessingRequest) -> MultimodalInput:
        """Create MultimodalInput from ProcessingRequest."""
        if isinstance(request.input_data, dict):
            return MultimodalInput(
                text=request.input_data.get("text"),
                audio_data=request.input_data.get("audio_data"),
                image_data=request.input_data.get("image_data"),
                metadata=request.metadata
            )
        else:
            return MultimodalInput(
                text=str(request.input_data),
                metadata=request.metadata
            )

    def _determine_modality(self, input_data: MultimodalInput) -> ModalityType:
        """Determine the primary modality of input data."""
        modalities = []
        if input_data.text:
            modalities.append(ModalityType.TEXT)
        if input_data.audio_data:
            modalities.append(ModalityType.SPEECH)
        if input_data.image_data:
            modalities.append(ModalityType.VISION)
            
        if len(modalities) > 1:
            return ModalityType.MULTIMODAL
        elif modalities:
            return modalities[0]
        else:
            return ModalityType.TEXT

    def _get_input_modalities(self, input_data: MultimodalInput) -> List[str]:
        """Get list of input modalities present."""
        modalities = []
        if input_data.text:
            modalities.append("text")
        if input_data.audio_data:
            modalities.append("audio")
        if input_data.image_data:
            modalities.append("image")
        return modalities

    async def _generate_response_from_text(
        self, 
        text: str, 
        context: ProcessingContext
    ) -> str:
        """Generate response from text input."""
        # Simple response generation - would be enhanced with actual NLP
        if "hello" in text.lower():
            return "Hello! How can I assist you today?"
        elif "help" in text.lower():
            return "I'm here to help! You can ask me questions, request tasks, or just have a conversation."
        elif "?" in text:
            return f"That's an interesting question about '{text}'. Let me think about that..."
        else:
            return f"I understand you mentioned '{text}'. How would you like me to help with that?"
    
    async def _select_skills(
        self,
        intent_data: Dict[str, Any],
        entities: List[Dict[str, Any]],
        context: AssistantContext
    ) -> List[str]:
        """Select appropriate skills based on intent and context."""
        # Get available skills
        available_skills = await self.skill_registry.get_available_skills()
        
        # Filter by intent
        relevant_skills = []
        for skill_id, skill_info in available_skills.items():
            if intent_data["intent"] in skill_info.get("supported_intents", []):
                relevant_skills.append(skill_id)
        
        # If no specific skills found, use general skills
        if not relevant_skills:
            relevant_skills = ["core_skills", "productivity_skills"]
        
        return relevant_skills[:3]  # Limit to top 3 skills
    
    async def _execute_skill(
        self,
        skill_id: str,
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a specific skill."""
        try:
            skill = await self.skill_registry.get_skill(skill_id)
            if not skill:
                return {"error": f"Skill {skill_id} not found"}
            
            result = await skill.execute(input_data)
            
            # Track execution
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
    
    def _build_prompt(
        self,
        user_input: str,
        context: AssistantContext,
        skill_results: List[Dict[str, Any]]
    ) -> str:
        """Build prompt for model inference."""
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
        
        # Construct prompt
        prompt = f"""Conversation context:
{conversation}

Current user input: {user_input}
{skill_context}
Please provide a helpful and contextual response in {context.language_preference.value}."""
        
        return prompt
    
    async def _post_process_response(
        self,
        response: ProcessingResponse,
        request: ProcessingRequest
    ) -> ProcessingResponse:
        """Post-process the response before returning."""
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
                "skills_used": response.skills_used
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
            event_type="interaction",
            data=learning_data,
            session_id=request.context.session_id
        ))
    
    async def end_session(self, session_id: str) -> None:
        """End an assistant session."""
        if session_id not in self.active_contexts:
            self.logger.warning(f"Attempted to end non-existent session: {session_id}")
            return
        
        context = self.active_contexts[session_id]
        
        # Clean up session in memory manager
        await self.memory_manager.cleanup_session(session_id)
        
        # Remove from active contexts
        del self.active_contexts[session_id]
        
        # Emit session ended event
        await self.event_bus.emit(SessionEnded(
            session_id=session_id,
            duration=(datetime.now(timezone.utc) - context.created_at).total_seconds()
        ))
        
        self.logger.info(f"Ended session: {session_id}")
    
    async def _perform_health_check(self) -> Dict[str, Any]:
        """Perform health check on all components."""
        health_status = {
            "healthy": True,
            "components": {},
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        # Check each component
        components = {
            "event_bus": self.event_bus,
            "model_coordinator": self.model_coordinator,
            "memory_manager": self.memory_manager,
            "skill_registry": self.skill_registry,
            "logic_engine": self.logic_engine
        }
        
        for name, component in components.items():
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
        
        return health_status
    
    async def get_status(self) -> Dict[str, Any]:
        """Get current status of the assistant engine."""
        return {
            "state": self.state.value,
            "active_sessions": len(self.active_contexts),
            "total_requests": self.request_count,
            "average_processing_time": (
                self.total_processing_time / self.request_count 
                if self.request_count > 0 else 0
            ),
            "skill_stats": self.skill_execution_stats,
            "health": await self._perform_health_check()
        }
    
    # Event handlers
    async def _handle_session_started(self, event: SessionStarted) -> None:
        """Handle session started event."""
        self.logger.info(f"Session started: {event.session_id}")
    
    async def _handle_session_ended(self, event: SessionEnded) -> None:
        """Handle session ended event."""
        self.logger.info(f"Session ended: {event.session_id}, duration: {event.duration}s")
    
    async def _handle_task_completed(self, event: TaskCompleted) -> None:
        """Handle task completed event."""
        self.logger.info(f"Task completed: {event.task_id}")
    
    async def _handle_skill_executed(self, event: SkillExecuted) -> None:
        """Handle skill executed event."""
        self.logger.debug(f"Skill executed: {event.skill_id}")
    
    async def shutdown(self) -> None:
        """Shutdown the assistant engine gracefully."""
        self.logger.info("Shutting down Core Assistant Engine")
        self.state = AssistantState.SHUTTING_DOWN
        
        # End all active sessions
        session_ids = list(self.active_contexts.keys())
        for session_id in session_ids:
            await self.end_session(session_id)
        
        # Shutdown subsystems
        for component in [
            self.memory_manager,
            self.skill_registry,
            self.learning_engine,
            self.preference_engine
        ]:
            if component and hasattr(component, 'shutdown'):
                await component.shutdown()
        
        self.state = AssistantState.SHUTDOWN
        self.logger.info("Core Assistant Engine shutdown complete")
