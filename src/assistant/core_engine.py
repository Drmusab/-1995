"""
Advanced AI Assistant Core Engine with Memory Integration
Author: Drmusab
Last Modified: 2025-07-20 13:45:00 UTC

This module provides the main processing pipeline and orchestration engine for the
AI assistant, integrating all subsystems including speech processing, vision,
natural language understanding, memory, learning, and multimodal fusion.

Merged functionality from core_engine.py and core_engine_memory.py to provide
a unified memory-enhanced processing pipeline.
"""

from pathlib import Path
from typing import Optional, Dict, Any, Union, List, Callable, AsyncGenerator, TypeVar, Generic
import asyncio
import logging
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from enum import Enum
import threading
from contextlib import asynccontextmanager
import uuid
import json
import hashlib
from concurrent.futures import ThreadPoolExecutor
import traceback

import numpy as np
import torch

# Core imports
from src.core.config.loader import ConfigLoader
from src.core.events.event_bus import EventBus
from src.core.events.event_types import (
    EngineStarted, EngineShutdown, ProcessingStarted, ProcessingCompleted,
    ProcessingError, ModalityProcessingStarted, ModalityProcessingCompleted,
    FusionStarted, FusionCompleted, MemoryOperationStarted, MemoryOperationCompleted,
    SkillExecutionStarted, SkillExecutionCompleted, LearningEventOccurred,
    UserInteractionStarted, UserInteractionCompleted, SessionStarted, SessionEnded,
    ErrorOccurred, SystemStateChanged, ComponentHealthChanged,
    MessageReceived, MessageProcessed, MemoryRetrievalRequested, MemoryItemStored
)
from src.core.error_handling import ErrorHandler, handle_exceptions
from src.core.dependency_injection import Container
from src.core.health_check import HealthCheck
from src.core.security.authentication import AuthenticationManager
from src.core.security.authorization import AuthorizationManager

# Processing components
from src.processing.speech.audio_pipeline import (
    EnhancedAudioPipeline, AudioPipelineRequest, AudioPipelineResult,
    PipelineMode, WorkflowType, QualityLevel
)
from src.processing.speech.speech_to_text import (
    EnhancedWhisperTranscriber, TranscriptionRequest, TranscriptionResult,
    TranscriptionQuality, AudioSource
)
from src.processing.speech.text_to_speech import (
    EnhancedTextToSpeech, SynthesisRequest, SynthesisResult,
    VoiceQuality, SpeakingStyle, SynthesisMode
)
from src.processing.speech.emotion_detection import (
    EnhancedEmotionDetector, EmotionDetectionRequest, EmotionResult,
    AnalysisMode, FeatureSet
)
from src.processing.speech.speaker_recognition import (
    EnhancedSpeakerRecognition, SpeakerRecognitionRequest, SpeakerRecognitionResult,
    ProcessingMode, VerificationMode
)
from src.processing.vision.vision_processor import VisionProcessor
from src.processing.vision.image_analyzer import ImageAnalyzer
from src.processing.natural_language.intent_manager import IntentManager
from src.processing.natural_language.language_chain import LanguageChain
from src.processing.natural_language.sentiment_analyzer import SentimentAnalyzer
from src.processing.natural_language.entity_extractor import EntityExtractor
from src.processing.multimodal.fusion_strategies import MultimodalFusionStrategy

# Reasoning and planning
from src.reasoning.logic_engine import LogicEngine
from src.reasoning.knowledge_graph import KnowledgeGraph
from src.reasoning.planning.task_planner import TaskPlanner
from src.reasoning.decision_making.decision_tree import DecisionTree

# Skills management
from src.skills.skill_factory import SkillFactory
from src.skills.skill_registry import SkillRegistry
from src.skills.skill_validator import SkillValidator

# Memory systems - Enhanced integration
from src.memory.core_memory.memory_manager import MemoryManager
from src.memory.core_memory.base_memory import MemoryType
from src.memory.operations.context_manager import (
    ContextManager, MemoryContextManager, ContextType, ContextPriority
)
from src.memory.operations.retrieval import (
    MemoryRetriever, RetrievalRequest, RetrievalResult,
    RetrievalStrategy, MemoryRetrievalMode
)
from src.memory.storage.vector_store import VectorStore
from src.memory.core_memory.memory_types import WorkingMemory, EpisodicMemory, SemanticMemory

# Session memory integration
from src.assistant.session_memory_integrator import SessionMemoryIntegrator

# Learning and adaptation
from src.learning.continual_learning import ContinualLearner
from src.learning.preference_learning import PreferenceLearner
from src.learning.feedback_processor import FeedbackProcessor
from src.learning.model_adaptation import ModelAdapter

# Integrations
from src.integrations.llm.model_router import ModelRouter
from src.integrations.cache.cache_strategy import CacheStrategy
from src.integrations.storage.database import DatabaseManager

# Observability
from src.observability.monitoring.metrics import MetricsCollector
from src.observability.monitoring.tracing import TraceManager
from src.observability.logging.config import get_logger

# Session and workflow management
from src.assistant.session_manager import SessionManager
from src.assistant.workflow_orchestrator import WorkflowOrchestrator
from src.assistant.component_manager import EnhancedComponentManager, ComponentPriority, ComponentDependency, DependencyType
from src.assistant.interaction_handler import InteractionHandler


# Type definitions
T = TypeVar('T')


class EngineState(Enum):
    """Core engine operational states."""
    INITIALIZING = "initializing"
    READY = "ready"
    PROCESSING = "processing"
    LEARNING = "learning"
    MAINTENANCE = "maintenance"
    SHUTTING_DOWN = "shutting_down"
    ERROR = "error"
    SUSPENDED = "suspended"


class ProcessingMode(Enum):
    """Processing modes for the core engine."""
    SYNCHRONOUS = "synchronous"
    ASYNCHRONOUS = "asynchronous"
    REAL_TIME = "real_time"
    BATCH = "batch"
    STREAMING = "streaming"
    INTERACTIVE = "interactive"
    MEMORY_ENHANCED = "memory_enhanced"  # NEW: Memory-enhanced processing mode


class ModalityType(Enum):
    """Types of input/output modalities."""
    TEXT = "text"
    SPEECH = "speech"
    VISION = "vision"
    AUDIO = "audio"
    GESTURE = "gesture"
    MULTIMODAL = "multimodal"


class PriorityLevel(Enum):
    """Processing priority levels."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3
    EMERGENCY = 4


@dataclass
class ProcessingContext:
    """Context for processing requests."""
    session_id: str
    user_id: Optional[str] = None
    conversation_id: Optional[str] = None
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    priority: PriorityLevel = PriorityLevel.NORMAL
    timeout_seconds: float = 30.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    memory_enhanced: bool = True  # NEW: Enable memory enhancement by default


@dataclass
class MultimodalInput:
    """Container for multimodal input data."""
    text: Optional[str] = None
    audio: Optional[np.ndarray] = None
    image: Optional[np.ndarray] = None
    video: Optional[np.ndarray] = None
    gesture_data: Optional[Dict[str, Any]] = None
    context: Optional[ProcessingContext] = None
    modality_weights: Dict[str, float] = field(default_factory=dict)
    processing_hints: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryContext:
    """Enhanced memory context for processing."""
    session_id: str
    context_elements: List[Dict[str, Any]] = field(default_factory=list)
    entities: List[Dict[str, Any]] = field(default_factory=list)
    semantic_memories: List[Dict[str, Any]] = field(default_factory=list)
    episodic_memories: List[Dict[str, Any]] = field(default_factory=list)
    working_memory_state: Dict[str, Any] = field(default_factory=dict)
    retrieval_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessingResult:
    """Comprehensive processing result from the core engine."""
    success: bool
    request_id: str
    session_id: str
    processing_time: float
    
    # Core results
    response_text: Optional[str] = None
    synthesized_audio: Optional[np.ndarray] = None
    generated_image: Optional[np.ndarray] = None
    
    # Intermediate results
    transcription_result: Optional[TranscriptionResult] = None
    emotion_result: Optional[EmotionResult] = None
    speaker_result: Optional[SpeakerRecognitionResult] = None
    vision_result: Optional[Dict[str, Any]] = None
    intent_result: Optional[Dict[str, Any]] = None
    entity_result: Optional[List[Dict[str, Any]]] = None
    sentiment_result: Optional[Dict[str, Any]] = None
    
    # Memory context
    memory_context: Optional[MemoryContext] = None
    memory_enhanced: bool = False
    
    # Reasoning and planning
    reasoning_trace: Optional[List[Dict[str, Any]]] = None
    decision_path: Optional[List[str]] = None
    executed_skills: List[str] = field(default_factory=list)
    
    # Memory operations
    memory_updates: List[Dict[str, Any]] = field(default_factory=list)
    retrieved_memories: List[Dict[str, Any]] = field(default_factory=list)
    
    # Learning and adaptation
    learning_updates: List[Dict[str, Any]] = field(default_factory=list)
    preference_updates: List[Dict[str, Any]] = field(default_factory=list)
    
    # Quality and confidence metrics
    overall_confidence: float = 0.0
    modality_confidences: Dict[str, float] = field(default_factory=dict)
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Error handling
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Performance metrics
    component_timings: Dict[str, float] = field(default_factory=dict)
    memory_usage: Dict[str, float] = field(default_factory=dict)
    
    # Metadata
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    engine_version: str = "2.0.0"  # Updated version


@dataclass
class EngineConfiguration:
    """Configuration for the core engine."""
    # Processing settings
    default_processing_mode: ProcessingMode = ProcessingMode.ASYNCHRONOUS
    max_concurrent_requests: int = 10
    default_timeout_seconds: float = 30.0
    enable_real_time_processing: bool = True
    
    # Component settings
    enable_speech_processing: bool = True
    enable_vision_processing: bool = True
    enable_multimodal_fusion: bool = True
    enable_reasoning: bool = True
    enable_learning: bool = True
    enable_memory_enhancement: bool = True  # NEW: Memory enhancement setting
    
    # Quality settings
    default_quality_level: str = "balanced"
    adaptive_quality: bool = True
    quality_monitoring: bool = True
    
    # Memory settings
    working_memory_size: int = 1000
    context_window_size: int = 4096
    memory_consolidation_interval: int = 3600
    memory_retrieval_limit: int = 10  # NEW: Limit for memory retrieval
    min_memory_relevance: float = 0.7  # NEW: Minimum relevance threshold
    
    # Caching settings
    enable_response_caching: bool = True
    enable_component_caching: bool = True
    cache_ttl_seconds: int = 3600
    
    # Security settings
    require_authentication: bool = True
    enable_authorization: bool = True
    audit_logging: bool = True
    
    # Performance settings
    enable_performance_monitoring: bool = True
    enable_profiling: bool = False
    gc_interval_seconds: int = 300


class CoreEngineError(Exception):
    """Custom exception for core engine operations."""
    
    def __init__(self, message: str, error_code: Optional[str] = None, component: Optional[str] = None):
        super().__init__(message)
        self.error_code = error_code
        self.component = component
        self.timestamp = datetime.now(timezone.utc)


class EnhancedCoreEngine:
    """
    Advanced AI Assistant Core Engine with Memory Integration.
    
    This engine coordinates all AI assistant capabilities including:
    - Multimodal input processing (speech, vision, text, gestures)
    - Natural language understanding and generation
    - Speech-to-text and text-to-speech conversion
    - Emotion detection and speaker recognition
    - Visual processing and scene understanding
    - Reasoning, planning, and decision making
    - Memory management and context awareness (ENHANCED)
    - Skill execution and workflow orchestration
    - Learning and adaptation
    - Multimodal fusion and output generation
    
    Features:
    - Asynchronous and real-time processing
    - Component health monitoring and auto-recovery
    - Intelligent caching and performance optimization
    - Security and authentication integration
    - Comprehensive observability and metrics
    - Event-driven architecture
    - Session and conversation management
    - Adaptive quality and performance tuning
    - Enhanced memory integration for context-aware responses
    """

    def __init__(self, container: Container, config: Optional[EngineConfiguration] = None):
        """
        Initialize the enhanced core engine.
        
        Args:
            container: Dependency injection container
            config: Engine configuration (uses defaults if None)
        """
        self.container = container
        self.config = config or EngineConfiguration()
        self.logger = get_logger(__name__)
        
        # Engine state management
        self.state = EngineState.INITIALIZING
        self.startup_time: Optional[datetime] = None
        self.shutdown_time: Optional[datetime] = None
        self.processing_queue = asyncio.Queue()
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.engine_lock = threading.Lock()
        
        # Initialize core components
        self._setup_core_services()
        
        # Set up component manager
        self.component_manager = container.get(EnhancedComponentManager)
        
        # Memory integration components
        self.memory_integrator = None  # Will be initialized after components
        self.memory_retriever = None
        self.memory_context_manager = None
        
        # Register with health check system
        self.health_check.register_component("core_engine", self._health_check_callback)
        
        # Setup threading
        self._setup_threading()
        
        self.logger.info("EnhancedCoreEngine initialized successfully with memory integration")

    def _setup_core_services(self) -> None:
        """Setup core services and utilities."""
        self.config_loader = self.container.get(ConfigLoader)
        self.event_bus = self.container.get(EventBus)
        self.error_handler = self.container.get(ErrorHandler)
        self.health_check = self.container.get(HealthCheck)
        
        # Session and workflow management
        self.session_manager = self.container.get(SessionManager)
        self.workflow_orchestrator = self.container.get(WorkflowOrchestrator)
        self.interaction_handler = self.container.get(InteractionHandler)
        
        # Monitoring and observability
        self._setup_monitoring()

    def _setup_monitoring(self) -> None:
        """Setup monitoring and observability."""
        self.metrics = self.container.get(MetricsCollector)
        self.tracer = self.container.get(TraceManager)
        
        # Register core engine metrics
        self.metrics.register_counter("engine_requests_total")
        self.metrics.register_histogram("engine_processing_duration_seconds")
        self.metrics.register_gauge("engine_active_sessions")
        self.metrics.register_counter("engine_errors_total")
        self.metrics.register_gauge("engine_component_health")
        
        # Memory-specific metrics
        self.metrics.register_counter("memory_enhanced_responses")
        self.metrics.register_histogram("memory_retrieval_time")
        self.metrics.register_counter("memory_context_updates")

    def _setup_threading(self) -> None:
        """Setup threading and concurrency."""
        self.thread_pool = ThreadPoolExecutor(
            max_workers=self.config.max_concurrent_requests,
            thread_name_prefix="core_engine"
        )
        self.processing_semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)

    async def initialize(self) -> None:
        """Initialize the core engine and all components."""
        try:
            with self.tracer.trace("engine_initialization") as span:
                self.logger.info("Starting core engine initialization...")
                
                # Register all components with the component manager
                await self._register_components()
                
                # Initialize all components through the component manager
                await self.component_manager.initialize_all()
                
                # Get component references after initialization
                await self._get_component_references()
                
                # Initialize memory integration components
                await self._initialize_memory_integration()
                
                # Register event handlers
                await self._register_event_handlers()
                
                # Start background tasks
                await self._start_background_tasks()
                
                # Mark as ready
                self.state = EngineState.READY
                self.startup_time = datetime.now(timezone.utc)
                
                # Emit startup event
                await self.event_bus.emit(EngineStarted(
                    engine_id=id(self),
                    version=self.config.engine_version,
                    components_loaded=len(self.component_manager._components),
                    startup_time=self.startup_time
                ))
                
                self.logger.info(f"Core engine initialized successfully at {self.startup_time}")
                
        except Exception as e:
            self.state = EngineState.ERROR
            self.logger.error(f"Failed to initialize core engine: {str(e)}")
            await self.event_bus.emit(ErrorOccurred(
                component="core_engine",
                error_type=type(e).__name__,
                error_message=str(e),
                severity="critical"
            ))
            raise CoreEngineError(f"Engine initialization failed: {str(e)}") from e

    async def _register_components(self) -> None:
        """Register all components with the component manager."""
        self.logger.info("Registering components with component manager...")
        
        # Register core components
        self._register_core_components()
        
        # Register processing components
        self._register_processing_components()
        
        # Register reasoning components
        self._register_reasoning_components()
        
        # Register memory systems
        self._register_memory_systems()
        
        # Register learning systems
        self._register_learning_systems()

    def _register_core_components(self) -> None:
        """Register core service components."""
        # Session Manager
        self.component_manager.register_component(
            "session_manager",
            SessionManager,
            priority=ComponentPriority.ESSENTIAL,
            config_section="session"
        )
        
        # Memory Integrator - NEW
        self.component_manager.register_component(
            "session_memory_integrator",
            SessionMemoryIntegrator,
            priority=ComponentPriority.ESSENTIAL,
            dependencies=[
                ComponentDependency("session_manager", DependencyType.REQUIRED),
                ComponentDependency("memory_manager", DependencyType.REQUIRED)
            ],
            config_section="memory.integration"
        )
        
        # Workflow Orchestrator
        self.component_manager.register_component(
            "workflow_orchestrator",
            WorkflowOrchestrator,
            priority=ComponentPriority.ESSENTIAL,
            dependencies=[
                ComponentDependency("session_manager", DependencyType.REQUIRED)
            ],
            config_section="workflows"
        )
        
        # Interaction Handler
        self.component_manager.register_component(
            "interaction_handler",
            InteractionHandler,
            priority=ComponentPriority.HIGH,
            dependencies=[
                ComponentDependency("session_manager", DependencyType.REQUIRED),
                ComponentDependency("workflow_orchestrator", DependencyType.REQUIRED)
            ],
            config_section="interactions"
        )

    def _register_processing_components(self) -> None:
        """Register processing components."""
        # Speech processing components
        if self.config.enable_speech_processing:
            self.component_manager.register_component(
                "audio_pipeline",
                EnhancedAudioPipeline,
                priority=ComponentPriority.NORMAL,
                config_section="processing.speech.audio_pipeline"
            )
            
            self.component_manager.register_component(
                "speech_to_text",
                EnhancedWhisperTranscriber,
                priority=ComponentPriority.NORMAL,
                dependencies=[
                    ComponentDependency("audio_pipeline", DependencyType.OPTIONAL)
                ],
                config_section="processing.speech.speech_to_text"
            )
            
            self.component_manager.register_component(
                "text_to_speech",
                EnhancedTextToSpeech,
                priority=ComponentPriority.NORMAL,
                config_section="processing.speech.text_to_speech"
            )
            
            self.component_manager.register_component(
                "emotion_detector",
                EnhancedEmotionDetector,
                priority=ComponentPriority.NORMAL,
                dependencies=[
                    ComponentDependency("audio_pipeline", DependencyType.OPTIONAL)
                ],
                config_section="processing.speech.emotion_detection"
            )
            
            self.component_manager.register_component(
                "speaker_recognition",
                EnhancedSpeakerRecognition,
                priority=ComponentPriority.NORMAL,
                dependencies=[
                    ComponentDependency("audio_pipeline", DependencyType.OPTIONAL)
                ],
                config_section="processing.speech.speaker_recognition"
            )
        
        # Vision processing components
        if self.config.enable_vision_processing:
            self.component_manager.register_component(
                "vision_processor",
                VisionProcessor,
                priority=ComponentPriority.NORMAL,
                config_section="processing.vision.processor"
            )
            
            self.component_manager.register_component(
                "image_analyzer",
                ImageAnalyzer,
                priority=ComponentPriority.NORMAL,
                dependencies=[
                    ComponentDependency("vision_processor", DependencyType.REQUIRED)
                ],
                config_section="processing.vision.analyzer"
            )
        
        # Natural language processing components
        self.component_manager.register_component(
            "intent_manager",
            IntentManager,
            priority=ComponentPriority.HIGH,
            config_section="processing.nlp.intent"
        )
        
        self.component_manager.register_component(
            "language_chain",
            LanguageChain,
            priority=ComponentPriority.HIGH,
            dependencies=[
                ComponentDependency("intent_manager", DependencyType.OPTIONAL),
                ComponentDependency("model_router", DependencyType.REQUIRED)
            ],
            config_section="processing.nlp.language_chain"
        )
        
        self.component_manager.register_component(
            "sentiment_analyzer",
            SentimentAnalyzer,
            priority=ComponentPriority.NORMAL,
            config_section="processing.nlp.sentiment"
        )
        
        self.component_manager.register_component(
            "entity_extractor",
            EntityExtractor,
            priority=ComponentPriority.NORMAL,
            config_section="processing.nlp.entity"
        )
        
        # Multimodal fusion
        if self.config.enable_multimodal_fusion:
            self.component_manager.register_component(
                "fusion_strategy",
                MultimodalFusionStrategy,
                priority=ComponentPriority.HIGH,
                config_section="processing.multimodal.fusion"
            )
        
        # LLM integration
        self.component_manager.register_component(
            "model_router",
            ModelRouter,
            priority=ComponentPriority.ESSENTIAL,
            config_section="integrations.llm.router"
        )

    def _register_reasoning_components(self) -> None:
        """Register reasoning and planning components."""
        if self.config.enable_reasoning:
            self.component_manager.register_component(
                "logic_engine",
                LogicEngine,
                priority=ComponentPriority.HIGH,
                dependencies=[
                    ComponentDependency("model_router", DependencyType.REQUIRED)
                ],
                config_section="reasoning.logic"
            )
            
            self.component_manager.register_component(
                "knowledge_graph",
                KnowledgeGraph,
                priority=ComponentPriority.HIGH,
                config_section="reasoning.knowledge"
            )
            
            self.component_manager.register_component(
                "task_planner",
                TaskPlanner,
                priority=ComponentPriority.HIGH,
                dependencies=[
                    ComponentDependency("logic_engine", DependencyType.REQUIRED),
                    ComponentDependency("knowledge_graph", DependencyType.OPTIONAL)
                ],
                config_section="reasoning.planning"
            )
            
            self.component_manager.register_component(
                "decision_tree",
                DecisionTree,
                priority=ComponentPriority.HIGH,
                dependencies=[
                    ComponentDependency("logic_engine", DependencyType.REQUIRED)
                ],
                config_section="reasoning.decision"
            )

    def _register_memory_systems(self) -> None:
        """Register memory management systems."""
        self.component_manager.register_component(
            "vector_store",
            VectorStore,
            priority=ComponentPriority.ESSENTIAL,
            config_section="memory.vector_store"
        )
        
        self.component_manager.register_component(
            "working_memory",
            WorkingMemory,
            priority=ComponentPriority.HIGH,
            dependencies=[
                ComponentDependency("vector_store", DependencyType.REQUIRED)
            ],
            config_section="memory.working"
        )
        
        self.component_manager.register_component(
            "episodic_memory",
            EpisodicMemory,
            priority=ComponentPriority.HIGH,
            dependencies=[
                ComponentDependency("vector_store", DependencyType.REQUIRED)
            ],
            config_section="memory.episodic"
        )
        
        self.component_manager.register_component(
            "semantic_memory",
            SemanticMemory,
            priority=ComponentPriority.HIGH,
            dependencies=[
                ComponentDependency("vector_store", DependencyType.REQUIRED)
            ],
            config_section="memory.semantic"
        )
        
        # Memory Context Manager - Enhanced
        self.component_manager.register_component(
            "memory_context_manager",
            MemoryContextManager,
            priority=ComponentPriority.HIGH,
            dependencies=[
                ComponentDependency("working_memory", DependencyType.REQUIRED),
                ComponentDependency("episodic_memory", DependencyType.REQUIRED),
                ComponentDependency("semantic_memory", DependencyType.REQUIRED)
            ],
            config_section="memory.context"
        )
        
        # Memory Retriever - NEW
        self.component_manager.register_component(
            "memory_retriever",
            MemoryRetriever,
            priority=ComponentPriority.HIGH,
            dependencies=[
                ComponentDependency("memory_manager", DependencyType.REQUIRED)
            ],
            config_section="memory.retrieval"
        )
        
        self.component_manager.register_component(
            "context_manager",
            ContextManager,
            priority=ComponentPriority.HIGH,
            dependencies=[
                ComponentDependency("working_memory", DependencyType.REQUIRED),
                ComponentDependency("episodic_memory", DependencyType.REQUIRED),
                ComponentDependency("semantic_memory", DependencyType.REQUIRED)
            ],
            config_section="memory.context"
        )
        
        self.component_manager.register_component(
            "memory_manager",
            MemoryManager,
            priority=ComponentPriority.HIGH,
            dependencies=[
                ComponentDependency("working_memory", DependencyType.REQUIRED),
                ComponentDependency("episodic_memory", DependencyType.REQUIRED),
                ComponentDependency("semantic_memory", DependencyType.REQUIRED),
                ComponentDependency("context_manager", DependencyType.REQUIRED)
            ],
            config_section="memory.manager"
        )

    def _register_learning_systems(self) -> None:
        """Register learning and adaptation systems."""
        if self.config.enable_learning:
            self.component_manager.register_component(
                "feedback_processor",
                FeedbackProcessor,
                priority=ComponentPriority.NORMAL,
                config_section="learning.feedback"
            )
            
            self.component_manager.register_component(
                "preference_learner",
                PreferenceLearner,
                priority=ComponentPriority.NORMAL,
                dependencies=[
                    ComponentDependency("feedback_processor", DependencyType.REQUIRED),
                    ComponentDependency("memory_manager", DependencyType.REQUIRED)
                ],
                config_section="learning.preferences"
            )
            
            self.component_manager.register_component(
                "model_adapter",
                ModelAdapter,
                priority=ComponentPriority.NORMAL,
                dependencies=[
                    ComponentDependency("model_router", DependencyType.REQUIRED)
                ],
                config_section="learning.adaptation"
            )
            
            self.component_manager.register_component(
                "continual_learner",
                ContinualLearner,
                priority=ComponentPriority.NORMAL,
                dependencies=[
                    ComponentDependency("feedback_processor", DependencyType.REQUIRED),
                    ComponentDependency("preference_learner", DependencyType.OPTIONAL),
                    ComponentDependency("model_adapter", DependencyType.REQUIRED),
                    ComponentDependency("memory_manager", DependencyType.REQUIRED)
                ],
                config_section="learning.continual"
            )

    async def _get_component_references(self) -> None:
        """Get references to initialized components."""
        self.logger.info("Getting component references...")
        
        # Session and workflow management
        self.session_manager = await self.component_manager.get_component("session_manager")
        self.workflow_orchestrator = await self.component_manager.get_component("workflow_orchestrator")
        self.interaction_handler = await self.component_manager.get_component("interaction_handler")
        
        # Processing components
        if self.config.enable_speech_processing:
            try:
                self.audio_pipeline = await self.component_manager.get_component("audio_pipeline")
                self.speech_to_text = await self.component_manager.get_component("speech_to_text")
                self.text_to_speech = await self.component_manager.get_component("text_to_speech")
                self.emotion_detector = await self.component_manager.get_component("emotion_detector")
                self.speaker_recognition = await self.component_manager.get_component("speaker_recognition")
            except Exception as e:
                self.logger.warning(f"Some speech components failed to initialize: {str(e)}")
        
        # Vision processing
        if self.config.enable_vision_processing:
            try:
                self.vision_processor = await self.component_manager.get_component("vision_processor")
                self.image_analyzer = await self.component_manager.get_component("image_analyzer")
            except Exception as e:
                self.logger.warning(f"Some vision components failed to initialize: {str(e)}")
        
        # NLP components
        try:
            self.intent_manager = await self.component_manager.get_component("intent_manager")
            self.language_chain = await self.component_manager.get_component("language_chain")
            self.sentiment_analyzer = await self.component_manager.get_component("sentiment_analyzer")
            self.entity_extractor = await self.component_manager.get_component("entity_extractor")
        except Exception as e:
            self.logger.warning(f"Some NLP components failed to initialize: {str(e)}")
        
        # Multimodal fusion
        if self.config.enable_multimodal_fusion:
            try:
                self.fusion_strategy = await self.component_manager.get_component("fusion_strategy")
            except Exception as e:
                self.logger.warning(f"Fusion strategy failed to initialize: {str(e)}")
        
        # LLM integration
        try:
            self.model_router = await self.component_manager.get_component("model_router")
        except Exception as e:
            self.logger.error(f"Model router failed to initialize: {str(e)}")
            raise CoreEngineError("Failed to initialize model router, which is required") from e
        
        # Reasoning components
        if self.config.enable_reasoning:
            try:
                self.logic_engine = await self.component_manager.get_component("logic_engine")
                self.knowledge_graph = await self.component_manager.get_component("knowledge_graph")
                self.task_planner = await self.component_manager.get_component("task_planner")
                self.decision_tree = await self.component_manager.get_component("decision_tree")
            except Exception as e:
                self.logger.warning(f"Some reasoning components failed to initialize: {str(e)}")
        
        # Memory systems
        try:
            self.memory_manager = await self.component_manager.get_component("memory_manager")
            self.context_manager = await self.component_manager.get_component("context_manager")
            self.vector_store = await self.component_manager.get_component("vector_store")
            self.working_memory = await self.component_manager.get_component("working_memory")
            self.episodic_memory = await self.component_manager.get_component("episodic_memory")
            self.semantic_memory = await self.component_manager.get_component("semantic_memory")
        except Exception as e:
            self.logger.error(f"Memory systems failed to initialize: {str(e)}")
            raise CoreEngineError("Failed to initialize memory systems, which are required") from e
        
        # Learning systems
        if self.config.enable_learning:
            try:
                self.continual_learner = await self.component_manager.get_component("continual_learner")
                self.preference_learner = await self.component_manager.get_component("preference_learner")
                self.feedback_processor = await self.component_manager.get_component("feedback_processor")
                self.model_adapter = await self.component_manager.get_component("model_adapter")
            except Exception as e:
                self.logger.warning(f"Some learning components failed to initialize: {str(e)}")

    async def _initialize_memory_integration(self) -> None:
        """Initialize memory integration components."""
        if self.config.enable_memory_enhancement:
            try:
                # Get memory integration components
                self.memory_integrator = await self.component_manager.get_component("session_memory_integrator")
                self.memory_retriever = await self.component_manager.get_component("memory_retriever")
                self.memory_context_manager = await self.component_manager.get_component("memory_context_manager")
                
                self.logger.info("Memory integration components initialized successfully")
            except Exception as e:
                self.logger.warning(f"Memory integration components failed to initialize: {str(e)}")
                self.config.enable_memory_enhancement = False

    async def _register_event_handlers(self) -> None:
        """Register event handlers for system events."""
        # Component health monitoring
        self.event_bus.subscribe("component_health_changed", self._handle_component_health_change)
        
        # Error handling
        self.event_bus.subscribe("error_occurred", self._handle_error_event)
        
        # Performance monitoring
        self.event_bus.subscribe("performance_threshold_exceeded", self._handle_performance_event)
        
        # Learning events
        if self.config.enable_learning:
            self.event_bus.subscribe("learning_event_occurred", self._handle_learning_event)

    async def _start_background_tasks(self) -> None:
        """Start background maintenance tasks."""
        # Performance monitoring task
        if self.config.enable_performance_monitoring:
            asyncio.create_task(self._performance_monitoring_loop())
        
        # Memory consolidation task
        asyncio.create_task(self._memory_consolidation_loop())
        
        # Component health monitoring task
        asyncio.create_task(self._health_monitoring_loop())
        
        # Learning update task
        if self.config.enable_learning:
            asyncio.create_task(self._learning_update_loop())

    @handle_exceptions
    async def process_multimodal_input(
        self,
        input_data: MultimodalInput,
        context: Optional[ProcessingContext] = None
    ) -> ProcessingResult:
        """
        Process multimodal input through the complete AI assistant pipeline.
        
        Args:
            input_data: Multimodal input data
            context: Processing context
            
        Returns:
            Comprehensive processing result
        """
        start_time = datetime.now(timezone.utc)
        
        # Setup processing context
        if context is None:
            context = ProcessingContext(
                session_id=str(uuid.uuid4()),
                request_id=str(uuid.uuid4())
            )
        
        # Validate engine state
        if self.state not in [EngineState.READY, EngineState.PROCESSING]:
            raise CoreEngineError(f"Engine not ready for processing. Current state: {self.state}")
        
        async with self.processing_semaphore:
            try:
                with self.tracer.trace("multimodal_processing") as span:
                    span.set_attributes({
                        "session_id": context.session_id,
                        "request_id": context.request_id,
                        "user_id": context.user_id or "anonymous",
                        "has_text": input_data.text is not None,
                        "has_audio": input_data.audio is not None,
                        "has_image": input_data.image is not None,
                        "has_video": input_data.video is not None,
                        "memory_enhanced": context.memory_enhanced
                    })
                    
                    # Update engine state
                    self.state = EngineState.PROCESSING
                    
                    # Emit processing started event
                    await self.event_bus.emit(ProcessingStarted(
                        session_id=context.session_id,
                        request_id=context.request_id,
                        input_modalities=[k for k, v in asdict(input_data).items() if v is not None and k != 'context']
                    ))
                    
                    # Initialize result container
                    result = ProcessingResult(
                        success=False,
                        request_id=context.request_id,
                        session_id=context.session_id,
                        processing_time=0.0
                    )
                    
                    # Get memory context if enabled
                    memory_context = None
                    if context.memory_enhanced and self.config.enable_memory_enhancement:
                        memory_context = await self._get_memory_context(
                            context.session_id,
                            input_data.text or "",
                            context.user_id
                        )
                        result.memory_context = memory_context
                        result.memory_enhanced = True
                    
                    # Process each modality
                    modality_results = await self._process_modalities(input_data, context, result)
                    
                    # Perform multimodal fusion
                    if self.config.enable_multimodal_fusion and len(modality_results) > 1:
                        fusion_result = await self._perform_multimodal_fusion(
                            modality_results, input_data, context, result
                        )
                        result = self._merge_fusion_result(result, fusion_result)
                    
                    # Reasoning and planning
                    if self.config.enable_reasoning:
                        reasoning_result = await self._perform_reasoning(input_data, context, result)
                        result = self._merge_reasoning_result(result, reasoning_result)
                    
                    # Memory operations
                    memory_result = await self._perform_memory_operations(input_data, context, result)
                    result = self._merge_memory_result(result, memory_result)
                    
                    # Generate response with memory enhancement
                    response_result = await self._generate_response(input_data, context, result, memory_context)
                    result = self._merge_response_result(result, response_result)
                    
                    # Extract and store knowledge from response
                    if self.config.enable_memory_enhancement:
                        await self._extract_and_store_knowledge(
                            result.response_text or "",
                            context.session_id,
                            context.user_id
                        )
                    
                    # Learning and adaptation
                    if self.config.enable_learning:
                        await self._perform_learning_updates(input_data, context, result)
                    
                    # Finalize result
                    processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
                    result.processing_time = processing_time
                    result.success = True
                    result.overall_confidence = self._calculate_overall_confidence(result)
                    
                    # Update metrics
                    self.metrics.increment("engine_requests_total")
                    self.metrics.record("engine_processing_duration_seconds", processing_time)
                    if result.memory_enhanced:
                        self.metrics.increment("memory_enhanced_responses")
                    
                    # Emit completion event
                    await self.event_bus.emit(ProcessingCompleted(
                        session_id=context.session_id,
                        request_id=context.request_id,
                        processing_time=processing_time,
                        success=True,
                        confidence=result.overall_confidence
                    ))
                    
                    self.logger.info(
                        f"Multimodal processing completed for session {context.session_id} "
                        f"in {processing_time:.2f}s with confidence {result.overall_confidence:.2f}"
                        f" (memory_enhanced: {result.memory_enhanced})"
                    )
                    
                    return result
                    
            except Exception as e:
                # Handle processing error
                processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
                
                error_result = ProcessingResult(
                    success=False,
                    request_id=context.request_id,
                    session_id=context.session_id,
                    processing_time=processing_time,
                    errors=[str(e)]
                )
                
                self.metrics.increment("engine_errors_total")
                
                await self.event_bus.emit(ProcessingError(
                    session_id=context.session_id,
                    request_id=context.request_id,
                    error_type=type(e).__name__,
                    error_message=str(e)
                ))
                
                self.logger.error(f"Processing failed for session {context.session_id}: {str(e)}")
                return error_result
            
            finally:
                # Reset engine state
                self.state = EngineState.READY

    @handle_exceptions
    async def process_message(
        self,
        message: str,
        session_id: str,
        user_id: Optional[str] = None,
        context_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a user message with memory-enhanced context.
        
        This is a convenience method that wraps process_multimodal_input
        for simple text message processing.
        
        Args:
            message: The user message
            session_id: Session identifier
            user_id: Optional user identifier
            context_data: Optional additional context
            
        Returns:
            Response with memory-enhanced context
        """
        # Create processing context
        context = ProcessingContext(
            session_id=session_id,
            user_id=user_id,
            memory_enhanced=True,
            metadata=context_data or {}
        )
        
        # Create multimodal input
        input_data = MultimodalInput(
            text=message,
            context=context
        )
        
        # Process through main pipeline
        result = await self.process_multimodal_input(input_data, context)
        
        # Format response for compatibility
        response = {
            "text": result.response_text,
            "session_id": session_id,
            "success": result.success,
            "memory_enhanced": result.memory_enhanced,
            "confidence": result.overall_confidence,
            "processing_time": result.processing_time,
            "context_used": len(result.memory_context.context_elements) if result.memory_context else 0,
            "trace_id": result.request_id
        }
        
        # Add intent and entities if available
        if result.intent_result:
            response["intent"] = result.intent_result
        if result.entity_result:
            response["entities"] = result.entity_result
        
        return response

    async def _get_memory_context(
        self, 
        session_id: str, 
        query: str,
        user_id: Optional[str] = None
    ) -> MemoryContext:
        """
        Get memory context for message processing.
        
        Args:
            session_id: Session identifier
            query: User query/message
            user_id: Optional user identifier
            
        Returns:
            Memory context
        """
        try:
            start_time = asyncio.get_event_loop().time()
            
            # Get context from memory context manager
            context_dict = await self.memory_context_manager.get_context_dict(session_id)
            
            # Emit memory retrieval event
            await self.event_bus.emit(MemoryRetrievalRequested(
                session_id=session_id,
                query=query,
                context_id=context_dict.get("context_id"),
                timestamp=datetime.now(timezone.utc)
            ))
            
            # Get semantic memories related to the query
            semantic_request = RetrievalRequest(
                query=query,
                session_id=session_id,
                memory_types=[MemoryType.SEMANTIC],
                strategy=RetrievalStrategy.SEMANTIC,
                mode=MemoryRetrievalMode.CONTEXTUAL,
                max_results=self.config.memory_retrieval_limit,
                min_relevance=self.config.min_memory_relevance
            )
            
            semantic_results = await self.memory_retriever.retrieve(semantic_request)
            
            # Get recent episodic memories
            episodic_request = RetrievalRequest(
                query=query,
                session_id=session_id,
                memory_types=[MemoryType.EPISODIC],
                strategy=RetrievalStrategy.RECENCY,
                max_results=min(5, self.config.memory_retrieval_limit // 2)
            )
            
            episodic_results = await self.memory_retriever.retrieve(episodic_request)
            
            # Get working memory state
            working_memory_state = await self.working_memory.get_state(session_id)
            
            # Create memory context
            memory_context = MemoryContext(
                session_id=session_id,
                context_elements=context_dict.get("elements", []),
                entities=context_dict.get("entities", []),
                semantic_memories=[
                    {
                        "content": memory.content,
                        "relevance": score,
                        "memory_id": memory.memory_id
                    }
                    for memory, score in semantic_results.items
                ],
                episodic_memories=[
                    {
                        "content": memory.content,
                        "memory_id": memory.memory_id,
                        "timestamp": memory.timestamp.isoformat() if hasattr(memory, 'timestamp') else None
                    }
                    for memory in episodic_results.memories
                ],
                working_memory_state=working_memory_state or {},
                retrieval_metadata={
                    "retrieval_time": asyncio.get_event_loop().time() - start_time,
                    "semantic_count": len(semantic_results.items),
                    "episodic_count": len(episodic_results.memories),
                    "total_elements": len(context_dict.get("elements", []))
                }
            )
            
            # Record retrieval time
            self.metrics.record("memory_retrieval_time", memory_context.retrieval_metadata["retrieval_time"])
            
            return memory_context
            
        except Exception as e:
            self.logger.error(f"Error getting memory context: {str(e)}")
            # Return empty context on error
            return MemoryContext(session_id=session_id)

    async def _process_modalities(
        self,
        input_data: MultimodalInput,
        context: ProcessingContext,
        result: ProcessingResult
    ) -> Dict[str, Any]:
        """Process individual modalities concurrently."""
        modality_tasks = []
        modality_results = {}
        
        # Speech processing
        if input_data.audio is not None and self.config.enable_speech_processing:
            modality_tasks.append(
                self._process_speech_modality(input_data.audio, context, result)
            )
        
        # Vision processing
        if input_data.image is not None and self.config.enable_vision_processing:
            modality_tasks.append(
                self._process_vision_modality(input_data.image, context, result)
            )
        
        # Text processing
        if input_data.text is not None:
            modality_tasks.append(
                self._process_text_modality(input_data.text, context, result)
            )
        
        # Execute modality processing concurrently
        if modality_tasks:
            task_results = await asyncio.gather(*modality_tasks, return_exceptions=True)
            
            # Collect successful results
            for i, task_result in enumerate(task_results):
                if isinstance(task_result, Exception):
                    self.logger.error(f"Modality processing failed: {str(task_result)}")
                    result.errors.append(f"Modality processing error: {str(task_result)}")
                else:
                    modality_results.update(task_result)
        
        return modality_results

    async def _process_speech_modality(
        self,
        audio: np.ndarray,
        context: ProcessingContext,
        result: ProcessingResult
    ) -> Dict[str, Any]:
        """Process speech/audio input."""
        speech_results = {}
        
        try:
            await self.event_bus.emit(ModalityProcessingStarted(
                session_id=context.session_id,
                modality="speech",
                request_id=context.request_id
            ))
            
            # Speech-to-text transcription
            if hasattr(self, 'speech_to_text'):
                transcription_request = TranscriptionRequest(
                    audio_source=AudioSource.BUFFER,
                    session_id=context.session_id,
                    user_id=context.user_id,
                    quality=TranscriptionQuality.BALANCED
                )
                
                transcription_result = await self.speech_to_text.transcribe(
                    audio, transcription_request
                )
                result.transcription_result = transcription_result
                speech_results['transcription'] = transcription_result
            
            # Emotion detection
            if hasattr(self, 'emotion_detector'):
                emotion_request = EmotionDetectionRequest(
                    session_id=context.session_id,
                    user_id=context.user_id,
                    analysis_mode=AnalysisMode.REAL_TIME
                )
                
                emotion_result = await self.emotion_detector.detect_emotion(
                    audio, emotion_request
                )
                result.emotion_result = emotion_result
                speech_results['emotion'] = emotion_result
            
            # Speaker recognition
            if hasattr(self, 'speaker_recognition'):
                speaker_request = SpeakerRecognitionRequest(
                    session_id=context.session_id,
                    user_id=context.user_id,
                    mode=ProcessingMode.REAL_TIME
                )
                
                speaker_result = await self.speaker_recognition.recognize_speaker(
                    audio, speaker_request
                )
                result.speaker_result = speaker_result
                speech_results['speaker'] = speaker_result
            
            await self.event_bus.emit(ModalityProcessingCompleted(
                session_id=context.session_id,
                modality="speech",
                request_id=context.request_id,
                success=True
            ))
            
            return speech_results
            
        except Exception as e:
            self.logger.error(f"Speech processing failed: {str(e)}")
            result.errors.append(f"Speech processing error: {str(e)}")
            return {}

    async def _process_vision_modality(
        self,
        image: np.ndarray,
        context: ProcessingContext,
        result: ProcessingResult
    ) -> Dict[str, Any]:
        """Process vision/image input."""
        vision_results = {}
        
        try:
            await self.event_bus.emit(ModalityProcessingStarted(
                session_id=context.session_id,
                modality="vision",
                request_id=context.request_id
            ))
            
            # Image analysis
            if hasattr(self, 'image_analyzer'):
                analysis_result = await self.image_analyzer.analyze_image(image)
                result.vision_result = analysis_result
                vision_results['analysis'] = analysis_result
            
            # Vision processing
            if hasattr(self, 'vision_processor'):
                processing_result = await self.vision_processor.process_image(image)
                vision_results['processing'] = processing_result
            
            await self.event_bus.emit(ModalityProcessingCompleted(
                session_id=context.session_id,
                modality="vision",
                request_id=context.request_id,
                success=True
            ))
            
            return vision_results
            
        except Exception as e:
            self.logger.error(f"Vision processing failed: {str(e)}")
            result.errors.append(f"Vision processing error: {str(e)}")
            return {}

    async def _process_text_modality(
        self,
        text: str,
        context: ProcessingContext,
        result: ProcessingResult
    ) -> Dict[str, Any]:
        """Process text input."""
        text_results = {}
        
        try:
            await self.event_bus.emit(ModalityProcessingStarted(
                session_id=context.session_id,
                modality="text",
                request_id=context.request_id
            ))
            
            # Use memory context for enhanced processing
            memory_hints = {}
            if result.memory_context:
                memory_hints = {
                    "context_elements": result.memory_context.context_elements,
                    "entities": result.memory_context.entities
                }
            
            # Intent recognition with memory context
            intent_result = await self.intent_manager.detect_intent(text, context=memory_hints)
            result.intent_result = intent_result
            text_results['intent'] = intent_result
            
            # Entity extraction
            entity_result = await self.entity_extractor.extract(text)
            result.entity_result = entity_result
            text_results['entities'] = entity_result
            
            # Update memory context with new entities
            if result.memory_context and entity_result:
                for entity in entity_result:
                    await self.memory_context_manager.add_entity(
                        session_id=context.session_id,
                        entity_id=str(uuid.uuid4()),
                        name=entity.get("text", ""),
                        entity_type=entity.get("type", "unknown")
                    )
                self.metrics.increment("memory_context_updates")
            
            # Sentiment analysis
            sentiment_result = await self.sentiment_analyzer.analyze(text)
            result.sentiment_result = sentiment_result
            text_results['sentiment'] = sentiment_result
            
            await self.event_bus.emit(ModalityProcessingCompleted(
                session_id=context.session_id,
                modality="text",
                request_id=context.request_id,
                success=True
            ))
            
            return text_results
            
        except Exception as e:
            self.logger.error(f"Text processing failed: {str(e)}")
            result.errors.append(f"Text processing error: {str(e)}")
            return {}

    async def _perform_multimodal_fusion(
        self,
        modality_results: Dict[str, Any],
        input_data: MultimodalInput,
        context: ProcessingContext,
        result: ProcessingResult
    ) -> Dict[str, Any]:
        """Perform multimodal fusion of processing results."""
        try:
            await self.event_bus.emit(FusionStarted(
                session_id=context.session_id,
                modalities=list(modality_results.keys()),
                request_id=context.request_id
            ))
            
            fusion_result = await self.fusion_strategy.fuse_modalities(
                modality_results,
                weights=input_data.modality_weights
            )
            
            await self.event_bus.emit(FusionCompleted(
                session_id=context.session_id,
                fusion_confidence=fusion_result.get('confidence', 0.0),
                request_id=context.request_id
            ))
            
            return fusion_result
            
        except Exception as e:
            self.logger.error(f"Multimodal fusion failed: {str(e)}")
            result.errors.append(f"Fusion error: {str(e)}")
            return {}

    async def _perform_reasoning(
        self,
        input_data: MultimodalInput,
        context: ProcessingContext,
        result: ProcessingResult
    ) -> Dict[str, Any]:
        """Perform reasoning and planning operations."""
        reasoning_results = {}
        
        try:
            # Include memory context in reasoning
            reasoning_context = result.__dict__.copy()
            if result.memory_context:
                reasoning_context['memory_context'] = asdict(result.memory_context)
            
            # Logic reasoning
            if hasattr(self, 'logic_engine'):
                logic_result = await self.logic_engine.reason(reasoning_context)
                reasoning_results['logic'] = logic_result
            
            # Task planning
            if hasattr(self, 'task_planner'):
                planning_result = await self.task_planner.plan(reasoning_context)
                reasoning_results['planning'] = planning_result
            
            # Decision making
            if hasattr(self, 'decision_tree'):
                decision_result = await self.decision_tree.decide(reasoning_context)
                reasoning_results['decision'] = decision_result
            
            return reasoning_results
            
        except Exception as e:
            self.logger.error(f"Reasoning failed: {str(e)}")
            result.errors.append(f"Reasoning error: {str(e)}")
            return {}

    async def _perform_memory_operations(
        self,
        input_data: MultimodalInput,
        context: ProcessingContext,
        result: ProcessingResult
    ) -> Dict[str, Any]:
        """Perform memory storage and retrieval operations."""
        memory_results = {}
        
        try:
            await self.event_bus.emit(MemoryOperationStarted(
                session_id=context.session_id,
                operation_type="store_and_retrieve",
                request_id=context.request_id
            ))
            
            # Store current interaction in episodic memory
            episodic_data = {
                'session_id': context.session_id,
                'user_id': context.user_id,
                'input_data': {
                    'text': input_data.text,
                    'has_audio': input_data.audio is not None,
                    'has_image': input_data.image is not None
                },
                'processing_result': {
                    'intent': result.intent_result,
                    'entities': result.entity_result,
                    'sentiment': result.sentiment_result,
                    'confidence': result.overall_confidence
                },
                'timestamp': context.timestamp,
                'memory_enhanced': result.memory_enhanced
            }
            
            memory_item = await self.episodic_memory.store(episodic_data)
            
            # Emit memory stored event
            await self.event_bus.emit(MemoryItemStored(
                memory_id=memory_item.get('memory_id', ''),
                memory_type=MemoryType.EPISODIC,
                session_id=context.session_id,
                content_preview=str(episodic_data)[:100]
            ))
            
        except Exception as e:
            self.logger.error(f"Failed to store episodic memory: {e}")
            await self.event_bus.emit(ErrorOccurred(
                error_type="MemoryStorageError",
                error_message=str(e),
                component="core_engine",
                session_id=context.session_id if context else None
            ))
