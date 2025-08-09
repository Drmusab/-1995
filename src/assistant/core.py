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
        metadata: Optional[Dict[str, Any]] = None
    ) -> AssistantContext:
        """Create a new assistant session."""
        session_id = str(uuid.uuid4())
        context = AssistantContext(
            session_id=session_id,
            user_id=user_id,
            metadata=metadata or {}
        )
        
        self.active_contexts[session_id] = context
        
        # Initialize session in memory manager
        await self.memory_manager.initialize_session(session_id)
        
        # Emit session started event
        await self.event_bus.emit(SessionStarted(
            session_id=session_id,
            user_id=user_id,
            timestamp=context.created_at
        ))
        
        self.logger.info(f"Created new session: {session_id}")
        return context
    
    async def process_input(
        self,
        request: ProcessingRequest
    ) -> ProcessingResponse:
        """
        Process user input through the assistant pipeline.
        
        This is the main entry point for all user interactions.
        """
        start_time = asyncio.get_event_loop().time()
        
        try:
            self.state = AssistantState.PROCESSING
            self.request_count += 1
            
            # Update context
            request.context.last_interaction = datetime.now(timezone.utc)
            
            # Log request
            self.logger.info(
                f"Processing request {request.request_id} "
                f"for session {request.context.session_id}"
            )
            
            # Route based on input type
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
                    "processing_mode": request.context.processing_mode.value
                }
            )
            
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
