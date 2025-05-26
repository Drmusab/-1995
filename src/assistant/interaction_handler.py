"""
Advanced Interaction Handler for AI Assistant
Author: Drmusab
Last Modified: 2025-05-26 15:16:37 UTC

This module provides comprehensive user interaction handling for the AI assistant,
managing multimodal input/output, conversation flow, context awareness, user
authentication, real-time processing, and seamless integration with all core components.
"""

from pathlib import Path
from typing import Optional, Dict, Any, List, Set, Callable, Type, Union, AsyncGenerator, TypeVar
import asyncio
import threading
import time
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
from contextlib import asynccontextmanager
import uuid
import json
import hashlib
from collections import defaultdict, deque
import weakref
from abc import ABC, abstractmethod
import logging
import inspect
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch

# Core imports
from src.core.config.loader import ConfigLoader
from src.core.events.event_bus import EventBus
from src.core.events.event_types import (
    UserInteractionStarted, UserInteractionCompleted, UserInteractionFailed,
    SessionStarted, SessionEnded, SessionContextUpdated, ConversationStarted,
    ConversationEnded, MessageReceived, MessageSent, MessageProcessed,
    ModalityDetected, InteractionModeChanged, UserAuthenticated, UserAuthorized,
    ErrorOccurred, SystemStateChanged, ComponentHealthChanged, FeedbackReceived,
    UserPreferenceUpdated, ContextAdapted, RealTimeProcessingStarted,
    RealTimeProcessingCompleted, StreamingStarted, StreamingCompleted
)
from src.core.error_handling import ErrorHandler, handle_exceptions
from src.core.dependency_injection import Container
from src.core.health_check import HealthCheck
from src.core.security.authentication import AuthenticationManager
from src.core.security.authorization import AuthorizationManager
from src.core.security.sanitization import InputSanitizer

# Assistant components
from src.assistant.core_engine import (
    EnhancedCoreEngine, MultimodalInput, ProcessingContext, ProcessingResult,
    EngineState, ProcessingMode, ModalityType, PriorityLevel
)
from src.assistant.component_manager import ComponentManager
from src.assistant.workflow_orchestrator import WorkflowOrchestrator, WorkflowPriority
from src.assistant.session_manager import SessionManager

# Processing components
from src.processing.natural_language.intent_manager import IntentManager
from src.processing.natural_language.language_chain import LanguageChain
from src.processing.natural_language.sentiment_analyzer import SentimentAnalyzer
from src.processing.speech.speech_to_text import TranscriptionRequest, TranscriptionQuality
from src.processing.speech.text_to_speech import SynthesisRequest, VoiceQuality
from src.processing.speech.emotion_detection import EmotionDetectionRequest, AnalysisMode
from src.processing.vision.image_analyzer import ImageAnalyzer
from src.processing.vision.vision_processor import VisionProcessor

# Memory systems
from src.memory.core_memory.memory_manager import MemoryManager
from src.memory.operations.context_manager import ContextManager
from src.memory.core_memory.memory_types import WorkingMemory, EpisodicMemory

# Learning and adaptation
from src.learning.continual_learning import ContinualLearner
from src.learning.preference_learning import PreferenceLearner
from src.learning.feedback_processor import FeedbackProcessor

# Observability
from src.observability.monitoring.metrics import MetricsCollector
from src.observability.monitoring.tracing import TraceManager
from src.observability.logging.config import get_logger

# Type definitions
T = TypeVar('T')


class InteractionState(Enum):
    """States of user interaction sessions."""
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    RESPONDING = "responding"
    WAITING_FOR_INPUT = "waiting_for_input"
    STREAMING = "streaming"
    PAUSED = "paused"
    ENDED = "ended"
    ERROR = "error"


class InteractionMode(Enum):
    """Interaction modes for different user experiences."""
    CONVERSATIONAL = "conversational"      # Natural conversation flow
    COMMAND = "command"                    # Direct command execution
    TASK_ORIENTED = "task_oriented"        # Structured task completion
    EXPLORATORY = "exploratory"            # Open-ended exploration
    EDUCATIONAL = "educational"            # Learning-focused interaction
    CREATIVE = "creative"                  # Creative collaboration
    ANALYTICAL = "analytical"              # Data analysis and insights
    REAL_TIME = "real_time"               # Real-time processing mode


class InputModality(Enum):
    """Types of input modalities."""
    TEXT = "text"
    SPEECH = "speech"
    VISION = "vision"
    GESTURE = "gesture"
    TOUCH = "touch"
    MULTIMODAL = "multimodal"


class OutputModality(Enum):
    """Types of output modalities."""
    TEXT = "text"
    SPEECH = "speech"
    VISUAL = "visual"
    HAPTIC = "haptic"
    MULTIMODAL = "multimodal"


class InteractionPriority(Enum):
    """Priority levels for interaction processing."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    URGENT = 3
    CRITICAL = 4


@dataclass
class UserProfile:
    """User profile information for personalization."""
    user_id: str
    username: Optional[str] = None
    display_name: Optional[str] = None
    email: Optional[str] = None
    
    # Preferences
    preferred_language: str = "en"
    preferred_voice: Optional[str] = None
    preferred_interaction_mode: InteractionMode = InteractionMode.CONVERSATIONAL
    preferred_input_modalities: Set[InputModality] = field(default_factory=lambda: {InputModality.TEXT})
    preferred_output_modalities: Set[OutputModality] = field(default_factory=lambda: {OutputModality.TEXT})
    
    # Accessibility settings
    accessibility_needs: Dict[str, Any] = field(default_factory=dict)
    font_size_multiplier: float = 1.0
    high_contrast_mode: bool = False
    screen_reader_compatible: bool = False
    
    # Behavioral patterns
    interaction_patterns: Dict[str, Any] = field(default_factory=dict)
    learning_style: Optional[str] = None
    attention_span: Optional[float] = None
    
    # Security and privacy
    privacy_settings: Dict[str, Any] = field(default_factory=dict)
    data_retention_preferences: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_active: Optional[datetime] = None
    total_interactions: int = 0
    is_active: bool = True


@dataclass
class InteractionContext:
    """Context information for an interaction."""
    interaction_id: str
    session_id: str
    user_id: Optional[str] = None
    conversation_id: Optional[str] = None
    
    # Interaction metadata
    interaction_mode: InteractionMode = InteractionMode.CONVERSATIONAL
    priority: InteractionPriority = InteractionPriority.NORMAL
    input_modalities: Set[InputModality] = field(default_factory=set)
    output_modalities: Set[OutputModality] = field(default_factory=set)
    
    # Timing
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_activity: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    timeout_seconds: float = 300.0
    
    # State
    state: InteractionState = InteractionState.IDLE
    is_real_time: bool = False
    is_streaming: bool = False
    
    # Context data
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    current_topic: Optional[str] = None
    user_intent: Optional[str] = None
    emotional_state: Optional[str] = None
    
    # Technical context
    device_info: Dict[str, Any] = field(default_factory=dict)
    network_info: Dict[str, Any] = field(default_factory=dict)
    location_info: Dict[str, Any] = field(default_factory=dict)
    
    # Processing hints
    quality_preference: str = "balanced"  # fast, balanced, quality
    latency_requirements: Dict[str, float] = field(default_factory=dict)
    
    # Metadata
    tags: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UserMessage:
    """Represents a user message in various modalities."""
    message_id: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    interaction_id: Optional[str] = None
    
    # Message content
    text: Optional[str] = None
    audio_data: Optional[np.ndarray] = None
    image_data: Optional[np.ndarray] = None
    video_data: Optional[np.ndarray] = None
    gesture_data: Optional[Dict[str, Any]] = None
    
    # Message metadata
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    modality: InputModality = InputModality.TEXT
    language: str = "en"
    encoding: Optional[str] = None
    
    # Processing hints
    intent: Optional[str] = None
    entities: List[Dict[str, Any]] = field(default_factory=list)
    sentiment: Optional[str] = None
    confidence: float = 0.0
    
    # Context
    reply_to: Optional[str] = None
    conversation_context: Dict[str, Any] = field(default_factory=dict)
    
    # Security
    is_sanitized: bool = False
    risk_level: str = "low"


@dataclass
class AssistantResponse:
    """Represents an assistant response in various modalities."""
    response_id: str
    interaction_id: str
    user_id: Optional[str] = None
    
    # Response content
    text: Optional[str] = None
    audio_data: Optional[np.ndarray] = None
    image_data: Optional[np.ndarray] = None
    video_data: Optional[np.ndarray] = None
    visual_elements: List[Dict[str, Any]] = field(default_factory=list)
    
    # Response metadata
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    modalities: Set[OutputModality] = field(default_factory=set)
    language: str = "en"
    
    # Response characteristics
    response_type: str = "answer"  # answer, question, clarification, action, etc.
    tone: Optional[str] = None
    formality_level: str = "neutral"
    confidence: float = 0.0
    
    # Processing information
    processing_time: float = 0.0
    workflow_id: Optional[str] = None
    execution_id: Optional[str] = None
    component_chain: List[str] = field(default_factory=list)
    
    # Context and reasoning
    reasoning_trace: List[Dict[str, Any]] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)
    related_topics: List[str] = field(default_factory=list)
    
    # Interaction management
    expects_response: bool = False
    suggested_follow_ups: List[str] = field(default_factory=list)
    interaction_hints: Dict[str, Any] = field(default_factory=dict)


class InteractionError(Exception):
    """Custom exception for interaction handling errors."""
    
    def __init__(self, message: str, interaction_id: Optional[str] = None, 
                 error_code: Optional[str] = None, user_id: Optional[str] = None):
        super().__init__(message)
        self.interaction_id = interaction_id
        self.error_code = error_code
        self.user_id = user_id
        self.timestamp = datetime.now(timezone.utc)


class InputProcessor(ABC):
    """Abstract base class for input processors."""
    
    @abstractmethod
    async def process(self, message: UserMessage, context: InteractionContext) -> Dict[str, Any]:
        """Process user input and return processed data."""
        pass
    
    @abstractmethod
    def can_process(self, modality: InputModality) -> bool:
        """Check if this processor can handle the given modality."""
        pass


class OutputGenerator(ABC):
    """Abstract base class for output generators."""
    
    @abstractmethod
    async def generate(self, content: str, modalities: Set[OutputModality], 
                      context: InteractionContext) -> AssistantResponse:
        """Generate response in specified modalities."""
        pass
    
    @abstractmethod
    def can_generate(self, modality: OutputModality) -> bool:
        """Check if this generator can handle the given modality."""
        pass


class TextInputProcessor(InputProcessor):
    """Processor for text input."""
    
    def __init__(self, intent_manager: IntentManager, sentiment_analyzer: SentimentAnalyzer,
                 input_sanitizer: InputSanitizer):
        self.intent_manager = intent_manager
        self.sentiment_analyzer = sentiment_analyzer
        self.input_sanitizer = input_sanitizer
        self.logger = get_logger(__name__)
    
    def can_process(self, modality: InputModality) -> bool:
        return modality == InputModality.TEXT
    
    async def process(self, message: UserMessage, context: InteractionContext) -> Dict[str, Any]:
        """Process text input."""
        if not message.text:
            raise InteractionError("No text content in message", message.message_id)
        
        # Sanitize input
        if not message.is_sanitized:
            sanitized_text = await self.input_sanitizer.sanitize_text(message.text)
            message.text = sanitized_text
            message.is_sanitized = True
        
        # Detect intent
        intent_result = await self.intent_manager.detect_intent(message.text)
        
        # Analyze sentiment
        sentiment_result = await self.sentiment_analyzer.analyze(message.text)
        
        return {
            "processed_text": message.text,
            "intent": intent_result,
            "sentiment": sentiment_result,
            "language_detected": "en",  # Would use language detection
            "confidence": max(intent_result.get("confidence", 0.0), 
                            sentiment_result.get("confidence", 0.0))
        }


class SpeechInputProcessor(InputProcessor):
    """Processor for speech input."""
    
    def __init__(self, core_engine: 'EnhancedCoreEngine'):
        self.core_engine = core_engine
        self.logger = get_logger(__name__)
    
    def can_process(self, modality: InputModality) -> bool:
        return modality == InputModality.SPEECH
    
    async def process(self, message: UserMessage, context: InteractionContext) -> Dict[str, Any]:
        """Process speech input."""
        if message.audio_data is None:
            raise InteractionError("No audio data in message", message.message_id)
        
        # Create transcription request
        transcription_request = TranscriptionRequest(
            session_id=context.session_id,
            user_id=context.user_id,
            quality=TranscriptionQuality.BALANCED
        )
        
        # Transcribe audio
        transcription_result = await self.core_engine.speech_to_text.transcribe(
            message.audio_data, transcription_request
        )
        
        # Detect emotion
        emotion_request = EmotionDetectionRequest(
            session_id=context.session_id,
            user_id=context.user_id,
            analysis_mode=AnalysisMode.REAL_TIME
        )
        
        emotion_result = await self.core_engine.emotion_detector.detect_emotion(
            message.audio_data, emotion_request
        )
        
        return {
            "transcription": transcription_result,
            "emotion": emotion_result,
            "audio_features": {},  # Would extract audio features
            "confidence": transcription_result.confidence
        }


class VisionInputProcessor(InputProcessor):
    """Processor for vision input."""
    
    def __init__(self, image_analyzer: ImageAnalyzer, vision_processor: VisionProcessor):
        self.image_analyzer = image_analyzer
        self.vision_processor = vision_processor
        self.logger = get_logger(__name__)
    
    def can_process(self, modality: InputModality) -> bool:
        return modality == InputModality.VISION
    
    async def process(self, message: UserMessage, context: InteractionContext) -> Dict[str, Any]:
        """Process vision input."""
        if message.image_data is None:
            raise InteractionError("No image data in message", message.message_id)
        
        # Analyze image
        analysis_result = await self.image_analyzer.analyze_image(message.image_data)
        
        # Process with vision pipeline
        processing_result = await self.vision_processor.process_image(message.image_data)
        
        return {
            "analysis": analysis_result,
            "processing": processing_result,
            "objects_detected": analysis_result.get("objects", []),
            "scene_description": analysis_result.get("description", ""),
            "confidence": analysis_result.get("confidence", 0.0)
        }


class MultimodalOutputGenerator(OutputGenerator):
    """Generator for multimodal output."""
    
    def __init__(self, core_engine: 'EnhancedCoreEngine', language_chain: LanguageChain):
        self.core_engine = core_engine
        self.language_chain = language_chain
        self.logger = get_logger(__name__)
    
    def can_generate(self, modality: OutputModality) -> bool:
        return True  # Can generate all modalities
    
    async def generate(self, content: str, modalities: Set[OutputModality], 
                      context: InteractionContext) -> AssistantResponse:
        """Generate multimodal response."""
        response = AssistantResponse(
            response_id=str(uuid.uuid4()),
            interaction_id=context.interaction_id,
            user_id=context.user_id,
            modalities=modalities
        )
        
        # Generate text response
        if OutputModality.TEXT in modalities:
            response.text = content
        
        # Generate speech response
        if OutputModality.SPEECH in modalities and content:
            synthesis_request = SynthesisRequest(
                text=content,
                session_id=context.session_id,
                user_id=context.user_id,
                quality=VoiceQuality.BALANCED
            )
            
            synthesis_result = await self.core_engine.text_to_speech.synthesize(synthesis_request)
            response.audio_data = synthesis_result.audio_data
        
        # Generate visual elements if needed
        if OutputModality.VISUAL in modalities:
            # Would generate charts, diagrams, etc.
            response.visual_elements = []
        
        return response


class InteractionHandler:
    """
    Advanced Interaction Handler for the AI Assistant.
    
    This handler manages all aspects of user interaction including:
    - Multimodal input processing (text, speech, vision, gestures)
    - Context-aware conversation management
    - Real-time and streaming processing
    - User authentication and authorization
    - Personalization and adaptation
    - Session and conversation lifecycle
    - Integration with core engine and workflow orchestrator
    - Performance monitoring and optimization
    - Error handling and recovery
    - Security and privacy compliance
    """
    
    def __init__(self, container: Container):
        """
        Initialize the interaction handler.
        
        Args:
            container: Dependency injection container
        """
        self.container = container
        self.logger = get_logger(__name__)
        
        # Core services
        self.config = container.get(ConfigLoader)
        self.event_bus = container.get(EventBus)
        self.error_handler = container.get(ErrorHandler)
        self.health_check = container.get(HealthCheck)
        
        # Core components
        self.core_engine = container.get(EnhancedCoreEngine)
        self.component_manager = container.get(ComponentManager)
        self.workflow_orchestrator = container.get(WorkflowOrchestrator)
        self.session_manager = container.get(SessionManager)
        
        # Processing components
        self.intent_manager = container.get(IntentManager)
        self.language_chain = container.get(LanguageChain)
        self.sentiment_analyzer = container.get(SentimentAnalyzer)
        self.image_analyzer = container.get(ImageAnalyzer)
        self.vision_processor = container.get(VisionProcessor)
        
        # Memory systems
        self.memory_manager = container.get(MemoryManager)
        self.context_manager = container.get(ContextManager)
        self.working_memory = container.get(WorkingMemory)
        self.episodic_memory = container.get(EpisodicMemory)
        
        # Learning and adaptation
        self.continual_learner = container.get(ContinualLearner)
        self.preference_learner = container.get(PreferenceLearner)
        self.feedback_processor = container.get(FeedbackProcessor)
        
        # Security
        self.auth_manager = container.get(AuthenticationManager)
        self.authz_manager = container.get(AuthorizationManager)
        self.input_sanitizer = container.get(InputSanitizer)
        
        # Observability
        self.metrics = container.get(MetricsCollector)
        self.tracer = container.get(TraceManager)
        
        # State management
        self.active_interactions: Dict[str, InteractionContext] = {}
        self.user_profiles: Dict[str, UserProfile] = {}
        self.conversation_histories: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        # Processing infrastructure
        self.input_processors: List[InputProcessor] = []
        self.output_generators: List[OutputGenerator] = []
        self.interaction_semaphore = asyncio.Semaphore(50)  # Max concurrent interactions
        
        # Performance tracking
        self.interaction_stats: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.response_times: deque = deque(maxlen=1000)
        
        # Configuration
        self.max_interaction_duration = self.config.get("interactions.max_duration", 1800.0)
        self.default_timeout = self.config.get("interactions.default_timeout", 300.0)
        self.enable_real_time = self.config.get("interactions.enable_real_time", True)
        self.enable_streaming = self.config.get("interactions.enable_streaming", True)
        self.max_conversation_history = self.config.get("interactions.max_history", 100)
        
        # Initialize components
        self._setup_processors()
        self._setup_monitoring()
        
        # Register health check
        self.health_check.register_component("interaction_handler", self._health_check_callback)
        
        self.logger.info("InteractionHandler initialized successfully")

    def _setup_processors(self) -> None:
        """Setup input processors and output generators."""
        try:
            # Input processors
            self.input_processors = [
                TextInputProcessor(self.intent_manager, self.sentiment_analyzer, self.input_sanitizer),
                SpeechInputProcessor(self.core_engine),
                VisionInputProcessor(self.image_analyzer, self.vision_processor)
            ]
            
            # Output generators
            self.output_generators = [
                MultimodalOutputGenerator(self.core_engine, self.language_chain)
            ]
            
            self.logger.info(f"Initialized {len(self.input_processors)} input processors and "
                           f"{len(self.output_generators)} output generators")
            
        except Exception as e:
            self.logger.error(f"Failed to setup processors: {str(e)}")

    def _setup_monitoring(self) -> None:
        """Setup monitoring and metrics."""
        try:
            # Register interaction metrics
            self.metrics.register_counter("interactions_total")
            self.metrics.register_counter("interactions_successful")
            self.metrics.register_counter("interactions_failed")
            self.metrics.register_histogram("interaction_duration_seconds")
            self.metrics.register_histogram("response_generation_time_seconds")
            self.metrics.register_gauge("active_interactions")
            self.metrics.register_counter("user_messages_total")
            self.metrics.register_counter("assistant_responses_total")
            
        except Exception as e:
            self.logger.warning(f"Failed to setup monitoring: {str(e)}")

    async def initialize(self) -> None:
        """Initialize the interaction handler."""
        try:
            # Load user profiles
            await self._load_user_profiles()
            
            # Start background tasks
            asyncio.create_task(self._interaction_cleanup_loop())
            asyncio.create_task(self._performance_monitoring_loop())
            asyncio.create_task(self._context_adaptation_loop())
            
            # Register event handlers
            await self._register_event_handlers()
            
            self.logger.info("InteractionHandler initialization completed")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize InteractionHandler: {str(e)}")
            raise InteractionError(f"Initialization failed: {str(e)}")

    async def _register_event_handlers(self) -> None:
        """Register event handlers for system events."""
        # User feedback events
        self.event_bus.subscribe("feedback_received", self._handle_user_feedback)
        
        # Session events
        self.event_bus.subscribe("session_ended", self._handle_session_ended)
        
        # Component health events
        self.event_bus.subscribe("component_health_changed", self._handle_component_health_change)

    @handle_exceptions
    async def start_interaction(
        self,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        interaction_mode: InteractionMode = InteractionMode.CONVERSATIONAL,
        input_modalities: Optional[Set[InputModality]] = None,
        output_modalities: Optional[Set[OutputModality]] = None,
        device_info: Optional[Dict[str, Any]] = None,
        priority: InteractionPriority = InteractionPriority.NORMAL
    ) -> str:
        """
        Start a new user interaction.
        
        Args:
            user_id: Optional user identifier
            session_id: Optional session identifier
            interaction_mode: Mode of interaction
            input_modalities: Preferred input modalities
            output_modalities: Preferred output modalities
            device_info: Device information
            priority: Interaction priority
            
        Returns:
            Interaction ID
        """
        interaction_id = str(uuid.uuid4())
        
        # Create or get session
        if not session_id:
            session_id = await self.session_manager.create_session(user_id)
        
        # Get user profile
        user_profile = await self._get_or_create_user_profile(user_id)
        
        # Create interaction context
        context = InteractionContext(
            interaction_id=interaction_id,
            session_id=session_id,
            user_id=user_id,
            interaction_mode=interaction_mode,
            priority=priority,
            input_modalities=input_modalities or user_profile.preferred_input_modalities,
            output_modalities=output_modalities or user_profile.preferred_output_modalities,
            device_info=device_info or {},
            timeout_seconds=self.default_timeout
        )
        
        # Store active interaction
        self.active_interactions[interaction_id] = context
        
        # Initialize conversation if needed
        if interaction_mode == InteractionMode.CONVERSATIONAL:
            conversation_id = str(uuid.uuid4())
            context.conversation_id = conversation_id
            
            await self.event_bus.emit(ConversationStarted(
                conversation_id=conversation_id,
                session_id=session_id,
                user_id=user_id
            ))
        
        # Emit interaction started event
        await self.event_bus.emit(UserInteractionStarted(
            interaction_id=interaction_id,
            session_id=session_id,
            user_id=user_id,
            interaction_mode=interaction_mode.value,
            input_modalities=[m.value for m in context.input_modalities],
            output_modalities=[m.value for m in context.output_modalities]
        ))
        
        # Update metrics
        self.metrics.increment("interactions_total")
        self.metrics.set("active_interactions", len(self.active_interactions))
        
        self.logger.info(f"Started interaction: {interaction_id} for user: {user_id}")
        return interaction_id

    @handle_exceptions
    async def process_user_message(
        self,
        interaction_id: str,
        message: UserMessage,
        real_time: bool = False,
        streaming: bool = False
    ) -> AssistantResponse:
        """
        Process a user message and generate a response.
        
        Args:
            interaction_id: Interaction identifier
            message: User message to process
            real_time: Enable real-time processing
            streaming: Enable streaming response
            
        Returns:
            Assistant response
        """
        async with self.interaction_semaphore:
            start_time = time.time()
            
            # Get interaction context
            if interaction_id not in self.active_interactions:
                raise InteractionError(f"Interaction {interaction_id} not found")
            
            context = self.active_interactions[interaction_id]
            context.last_activity = datetime.now(timezone.utc)
            context.state = InteractionState.PROCESSING
            context.is_real_time = real_time
            context.is_streaming = streaming
            
            try:
                with self.tracer.trace("user_message_processing") as span:
                    span.set_attributes({
                        "interaction_id": interaction_id,
                        "user_id": context.user_id or "anonymous",
                        "session_id": context.session_id,
                        "modality": message.modality.value,
                        "real_time": real_time,
                        "streaming": streaming
                    })
                    
                    # Authenticate and authorize if needed
                    if context.user_id:
                        await self._verify_user_authorization(context.user_id, interaction_id)
                    
                    # Emit message received event
                    await self.event_bus.emit(MessageReceived(
                        message_id=message.message_id,
                        interaction_id=interaction_id,
                        session_id=context.session_id,
                        user_id=context.user_id,
                        modality=message.modality.value,
                        content_preview=message.text[:100] if message.text else None
                    ))
                    
                    # Process input through appropriate processor
                    processed_input = await self._process_user_input(message, context)
                    
                    # Update context with processed information
                    await self._update_interaction_context(context, message, processed_input)
                    
                    # Determine processing approach
                    if real_time:
                        response = await self._process_real_time(processed_input, context)
                    elif streaming:
                        response = await self._process_streaming(processed_input, context)
                    else:
                        response = await self._process_standard(processed_input, context)
                    
                    # Store interaction in memory
                    await self._store_interaction_memory(context, message, response)
                    
                    # Update conversation history
                    await self._update_conversation_history(context, message, response)
                    
                    # Learn from interaction
                    await self._learn_from_interaction(context, message, response)
                    
                    # Calculate metrics
                    processing_time = time.time() - start_time
                    response.processing_time = processing_time
                    self.response_times.append(processing_time)
                    
                    # Update context state
                    context.state = InteractionState.RESPONDING
                    
                    # Emit message processed event
                    await self.event_bus.emit(MessageProcessed(
                        message_id=message.message_id,
                        interaction_id=interaction_id,
                        processing_time=processing_time,
                        confidence=response.confidence,
                        success=True
                    ))
                    
                    # Update metrics
                    self.metrics.increment("user_messages_total")
                    self.metrics.increment("assistant_responses_total")
                    self.metrics.record("response_generation_time_seconds", processing_time)
                    
                    self.logger.info(
                        f"Processed message {message.message_id} in {processing_time:.2f}s "
                        f"with confidence {response.confidence:.2f}"
                    )
                    
                    return response
                    
            except Exception as e:
                # Handle processing error
                processing_time = time.time() - start_time
                context.state = InteractionState.ERROR
                
                error_response = AssistantResponse(
                    response_id=str(uuid.uuid4()),
                    interaction_id=interaction_id,
                    user_id=context.user_id,
                    text="I apologize, but I encountered an error processing your message. Please try again.",
                    processing_time=processing_time,
                    confidence=0.0
                )
                
                await self.event_bus.emit(UserInteractionFailed(
                    interaction_id=interaction_id,
                    session_id=context.session_id,
                    user_id=context.user_id,
                    error_message=str(e),
                    error_type=type(e).__name__
                ))
                
                self.metrics.increment("interactions_failed")
                self.logger.error(f"Error processing message {message.message_id}: {str(e)}")
                
                return error_response

    async def _process_user_input(
        self,
        message: UserMessage,
        context: InteractionContext
    ) -> Dict[str, Any]:
        """Process user input through appropriate processors."""
        # Find appropriate processor
        processor = None
        for proc in self.input_processors:
            if proc.can_process(message.modality):
                processor = proc
                break
        
        if not processor:
            raise InteractionError(f"No processor found for modality {message.modality}")
        
        # Process input
        processed_data = await processor.process(message, context)
        
        # Detect modality if multimodal
        if message.modality == InputModality.MULTIMODAL:
            await self.event_bus.emit(ModalityDetected(
                interaction_id=context.interaction_id,
                detected_modalities=list(context.input_modalities),
                confidence=processed_data.get("confidence", 0.0)
            ))
        
        return processed_data

    async def _update_interaction_context(
        self,
        context: InteractionContext,
        message: UserMessage,
        processed_input: Dict[str, Any]
    ) -> None:
        """Update interaction context with new information."""
        # Update intent
        if "intent" in processed_input:
            intent_data = processed_input["intent"]
            context.user_intent = intent_data.get("intent")
        
        # Update emotional state
        if "emotion" in processed_input:
            emotion_data = processed_input["emotion"]
            context.emotional_state = emotion_data.get("dominant_emotion")
        elif "sentiment" in processed_input:
            sentiment_data = processed_input["sentiment"]
            context.emotional_state = sentiment_data.get("sentiment")
        
        # Update topic if conversation mode
        if context.interaction_mode == InteractionMode.CONVERSATIONAL:
            # Would use topic modeling to detect topic shifts
            pass
        
        # Update last activity
        context.last_activity = datetime.now(timezone.utc)

    async def _process_standard(
        self,
        processed_input: Dict[str, Any],
        context: InteractionContext
    ) -> AssistantResponse:
        """Process input through standard pipeline."""
        # Create multimodal input for core engine
        multimodal_input = MultimodalInput(
            text=processed_input.get("processed_text"),
            context=ProcessingContext(
                session_id=context.session_id,
                user_id=context.user_id,
                priority=PriorityLevel(context.priority.value)
            )
        )
        
        # Process through core engine
        processing_result = await self.core_engine.process_multimodal_input(
            multimodal_input,
            multimodal_input.context
        )
        
        # Generate response
        return await self._generate_response(processing_result, context)

    async def _process_real_time(
        self,
        processed_input: Dict[str, Any],
        context: InteractionContext
    ) -> AssistantResponse:
        """Process input with real-time optimizations."""
        await self.event_bus.emit(RealTimeProcessingStarted(
            interaction_id=context.interaction_id,
            session_id=context.session_id
        ))
        
        # Use faster processing pipeline
        multimodal_input = MultimodalInput(
            text=processed_input.get("processed_text"),
            context=ProcessingContext(
                session_id=context.session_id,
                user_id=context.user_id,
                priority=PriorityLevel.HIGH,
                timeout_seconds=10.0  # Shorter timeout for real-time
            )
        )
        
        processing_result = await self.core_engine.process_multimodal_input(
            multimodal_input,
            multimodal_input.context
        )
        
        response = await self._generate_response(processing_result, context, fast_mode=True)
        
        await self.event_bus.emit(RealTimeProcessingCompleted(
            interaction_id=context.interaction_id,
            processing_time=response.processing_time
        ))
        
        return response

    async def _process_streaming(
        self,
        processed_input: Dict[str, Any],
        context: InteractionContext
    ) -> AssistantResponse:
        """Process input with streaming response generation."""
        await self.event_bus.emit(StreamingStarted(
            interaction_id=context.interaction_id,
            session_id=context.session_id
        ))
        
        # For streaming, we would yield partial responses
        # This is a simplified version that returns a complete response
        response = await self._process_standard(processed_input, context)
        
        await self.event_bus.emit(StreamingCompleted(
            interaction_id=context.interaction_id,
            total_chunks=1,
            final_response_id=response.response_id
        ))
        
        return response

    async def _generate_response(
        self,
        processing_result: ProcessingResult,
        context: InteractionContext,
        fast_mode: bool = False
    ) -> AssistantResponse:
        """Generate assistant response from processing result."""
        # Find appropriate output generator
        generator = None
        for gen in self.output_generators:
            if all(gen.can_generate(modality) for modality in context.output_modalities):
                generator = gen
                break
        
        if not generator:
            raise InteractionError("No suitable output generator found")
        
        # Generate response
        response = await generator.generate(
            processing_result.response_text or "I understand.",
            context.output_modalities,
            context
        )
        
        # Add processing information
        response.processing_time = processing_result.processing_time
        response.confidence = processing_result.overall_confidence
        response.component_chain = list(processing_result.component_timings.keys())
        response.reasoning_trace = processing_result.reasoning_trace or []
        
        # Set response characteristics based on context
        if context.emotional_state:
            response.tone = self._adapt_tone_to_emotion(context.emotional_state)
        
        if context.interaction_mode == InteractionMode.EDUCATIONAL:
            response.suggested_follow_ups = self._generate_educational_follow_ups(processing_result)
        
        return response

    def _adapt_tone_to_emotion(self, emotional_state: str) -> str:
        """Adapt response tone based on user's emotional state."""
        emotion_tone_map = {
            "happy": "enthusiastic",
            "sad": "empathetic",
            "angry": "calm",
            "frustrated": "patient",
            "excited": "encouraging",
            "confused": "clarifying",
            "neutral": "neutral"
        }
        return emotion_tone_map.get(emotional_state.lower(), "neutral")

    def _generate_educational_follow_ups(self, processing_result: ProcessingResult) -> List[str]:
        """Generate educational follow-up suggestions."""
        # This would analyze the response and suggest related learning topics
        return [
            "Would you like to learn more about this topic?",
            "Do you have any questions about what I just explained?",
            "Would you like to see some examples?"
        ]

    async def _store_interaction_memory(
        self,
        context: InteractionContext,
        message: UserMessage,
        response: AssistantResponse
    ) -> None:
        """Store interaction in memory systems."""
        try:
            # Store in episodic memory
            interaction_data = {
                "interaction_id": context.interaction_id,
                "session_id": context.session_id,
                "user_id": context.user_id,
                "user_message": {
                    "text": message.text,
                    "modality": message.modality.value,
                    "timestamp": message.timestamp.isoformat()
                },
                "assistant_response": {
                    "text": response.text,
                    "modalities": [m.value for m in response.modalities],
                    "confidence": response.confidence,
                    "timestamp": response.timestamp.isoformat()
                },
                "context": {
                    "interaction_mode": context.interaction_mode.value,
                    "emotional_state": context.emotional_state,
                    "user_intent": context.user_intent
                }
            }
            
            await self.episodic_memory.store(interaction_data)
            
            # Update working memory
            await self.working_memory.update(context.session_id, {
                "last_interaction": interaction_data,
                "context": context.__dict__
            })
            
        except Exception as e:
            self.logger.warning(f"Failed to store interaction memory: {str(e)}")

    async def _update_conversation_history(
        self,
        context: InteractionContext,
        message: UserMessage,
        response: AssistantResponse
    ) -> None:
        """Update conversation history."""
        if context.conversation_id:
            history_entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "user_message": {
                    "text": message.text,
                    "modality": message.modality.value
                },
                "assistant_response": {
                    "text": response.text,
                    "modalities": [m.value for m in response.modalities],
                    "confidence": response.confidence
                }
            }
            
            conversation_history = self.conversation_histories[context.conversation_id]
            conversation_history.append(history_entry)
            
            # Limit history size
            if len(conversation_history) > self.max_conversation_history:
                conversation_history.pop(0)
            
            context.conversation_history = conversation_history

    async def _learn_from_interaction(
        self,
        context: InteractionContext,
        message: UserMessage,
        response: AssistantResponse
    ) -> None:
        """Learn from the interaction for future improvements."""
        try:
            learning_data = {
                "interaction_context": context.__dict__,
                "user_input": {
                    "text": message.text,
                    "modality": message.modality.value,
                    "intent": context.user_intent,
                    "emotional_state": context.emotional_state
                },
                "response_quality": {
                    "confidence": response.confidence,
                    "processing_time": response.processing_time,
                    "modalities": [m.value for m in response.modalities]
                },
                "context_factors": {
                    "interaction_mode": context.interaction_mode.value,
                    "device_info": context.device_info,
                    "time_of_day": context.start_time.hour
                }
            }
            
            # Update continual learning
            await self.continual_learner.learn_from_interaction(learning_data)
            
            # Update user preferences
            if context.user_id:
                await self.preference_learner.update_from_interaction(
                    context.user_id, learning_data
                )
            
        except Exception as e:
            self.logger.warning(f"Failed to learn from interaction: {str(e)}")

    @handle_exceptions
    async def end_interaction(self, interaction_id: str, reason: str = "completed") -> None:
        """
        End an active interaction.
        
        Args:
            interaction_id: Interaction identifier
            reason: Reason for ending the interaction
        """
        if interaction_id not in self.active_interactions:
            raise InteractionError(f"Interaction {interaction_id} not found")
        
        context = self.active_interactions[interaction_id]
        context.state = InteractionState.ENDED
        
        # Calculate interaction duration
        duration = (datetime.now(timezone.utc) - context.start_time).total_seconds()
        
        # Emit interaction completed event
        await self.event_bus.emit(UserInteractionCompleted(
            interaction_id=interaction_id,
            session_id=context.session_id,
            user_id=context.user_id,
            duration=duration,
            message_count=len(context.conversation_history),
            reason=reason
        ))
        
        # End conversation if applicable
        if context.conversation_id:
            await self.event_bus.emit(ConversationEnded(
                conversation_id=context.conversation_id,
                session_id=context.session_id,
                user_id=context.user_id,
                duration=duration,
                message_count=len(context.conversation_history)
            ))
        
        # Update metrics
        self.metrics.increment("interactions_successful")
        self.metrics.record("interaction_duration_seconds", duration)
        self.metrics.set("active_interactions", len(self.active_interactions) - 1)
        
        # Remove from active interactions
        del self.active_interactions[interaction_id]
        
        self.logger.info(f"Ended interaction: {interaction_id} after {duration:.2f}s")

    @handle_exceptions
    async def handle_user_feedback(
        self,
        interaction_id: str,
        feedback_type: str,
        feedback_data: Dict[str, Any],
        user_id: Optional[str] = None
    ) -> None:
        """
        Handle user feedback for an interaction.
        
        Args:
            interaction_id: Interaction identifier
            feedback_type: Type of feedback (rating, correction, etc.)
            feedback_data: Feedback data
            user_id: Optional user identifier
        """
        # Process feedback
        await self.feedback_processor.process_feedback(
            interaction_id=interaction_id,
            feedback_type=feedback_type,
            feedback_data=feedback_data,
            user_id=user_id
        )
        
        # Emit feedback event
        await self.event_bus.emit(FeedbackReceived(
            interaction_id=interaction_id,
            user_id=user_id,
            feedback_type=feedback_type,
            feedback_data=feedback_data
        ))
        
        self.logger.info(f"Received {feedback_type} feedback for interaction {interaction_id}")

    async def _get_or_create_user_profile(self, user_id: Optional[str]) -> UserProfile:
        """Get existing user profile or create a new one."""
        if not user_id:
            # Return default anonymous profile
            return UserProfile(user_id="anonymous")
        
        if user_id in self.user_profiles:
            profile = self.user_profiles[user_id]
            profile.last_active = datetime.now(timezone.utc)
            profile.total_interactions += 1
            return profile
        
        # Create new profile
        profile = UserProfile(user_id=user_id)
        
        # Load preferences if available
        if hasattr(self.preference_learner, 'get_user_preferences'):
            try:
                preferences = await self.preference_learner.get_user_preferences(user_id)
                if preferences:
                    # Update profile with learned preferences
                    profile.preferred_interaction_mode = InteractionMode(
                        preferences.get("interaction_mode", "conversational")
                    )
                    profile.interaction_patterns = preferences.get("patterns", {})
            except Exception as e:
                self.logger.warning(f"Failed to load preferences for user {user_id}: {str(e)}")
        
        self.user_profiles[user_id] = profile
        return profile

    async def _verify_user_authorization(self, user_id: str, interaction_id: str) -> None:
        """Verify user authorization for the interaction."""
        try:
            # Check if user is authenticated
            is_authenticated = await self.auth_manager.is_authenticated(user_id)
            if not is_authenticated:
                raise InteractionError(f"User {user_id} not authenticated", interaction_id)
            
            # Check authorization for interaction
            is_authorized = await self.authz_manager.check_permission(
                user_id, "interaction", "create"
            )
            if not is_authorized:
                raise InteractionError(f"User {user_id} not authorized for interactions", interaction_id)
            
        except Exception as e:
            self.logger.error(f"Authorization failed for user {user_id}: {str(e)}")
            raise

    async def _load_user_profiles(self) -> None:
        """Load user profiles from storage."""
        try:
            # Would load from database or file storage
            # For now, start with empty profiles
            self.user_profiles = {}
            self.logger.info("User profiles loaded")
        except Exception as e:
            self.logger.warning(f"Failed to load user profiles: {str(e)}")

    async def _interaction_cleanup_loop(self) -> None:
        """Background task to clean up expired interactions."""
        while True:
            try:
                current_time = datetime.now(timezone.utc)
                expired_interactions = []
                
                for interaction_id, context in self.active_interactions.items():
                    # Check for timeout
                    time_since_activity = (current_time - context.last_activity).total_seconds()
                    
                    if time_since_activity > context.timeout_seconds:
                        expired_interactions.append(interaction_id)
                    elif (current_time - context.start_time).total_seconds() > self.max_interaction_duration:
                        expired_interactions.append(interaction_id)
                
                # Clean up expired interactions
                for interaction_id in expired_interactions:
                    try:
                        await self.end_interaction(interaction_id, "timeout")
                    except Exception as e:
                        self.logger.error(f"Failed to cleanup interaction {interaction_id}: {str(e)}")
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Error in interaction cleanup loop: {str(e)}")
                await asyncio.sleep(60)

    async def _performance_monitoring_loop(self) -> None:
        """Background task for performance monitoring."""
        while True:
            try:
                # Update active interactions metric
                self.metrics.set("active_interactions", len(self.active_interactions))
                
                # Calculate average response time
                if self.response_times:
                    avg_response_time = sum(self.response_times) / len(self.response_times)
                    self.metrics.set("average_response_time_seconds", avg_response_time)
                
                await asyncio.sleep(30)  # Monitor every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Error in performance monitoring: {str(e)}")
                await asyncio.sleep(30)

    async def _context_adaptation_loop(self) -> None:
        """Background task for context adaptation."""
        while True:
            try:
                # Adapt interaction contexts based on learned patterns
                for interaction_id, context in self.active_interactions.items():
                    if context.user_id and context.interaction_mode == InteractionMode.CONVERSATIONAL:
                        # Check if we should adapt the interaction based on patterns
                        user_profile = self.user_profiles.get(context.user_id)
                        if user_profile and user_profile.interaction_patterns:
                            # Would implement adaptation logic here
                            pass
                
                await asyncio.sleep(300)  # Adapt every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Error in context adaptation: {str(e)}")
                await asyncio.sleep(300)

    async def _handle_user_feedback(self, event) -> None:
        """Handle user feedback events."""
        try:
            # Update learning systems with feedback
            await self.continual_learner.incorporate_feedback(event.feedback_data)
            
            # Update user preferences if applicable
            if event.user_id:
                await self.preference_learner.update_from_feedback(
                    event.user_id, event.feedback_data
                )
            
        except Exception as e:
            self.logger.error(f"Error handling user feedback: {str(e)}")

    async def _handle_session_ended(self, event) -> None:
        """Handle session ended events."""
        try:
            # End any active interactions for the session
            session_interactions = [
                interaction_id for interaction_id, context in self.active_interactions.items()
                if context.session_id == event.session_id
            ]
            
            for interaction_id in session_interactions:
                await self.end_interaction(interaction_id, "session_ended")
            
        except Exception as e:
            self.logger.error(f"Error handling session ended: {str(e)}")

    async def _handle_component_health_change(self, event) -> None:
        """Handle component health change events."""
        if not event.healthy:
            # Adapt processing to work around unhealthy components
            self.logger.warning(f"Component {event.component} is unhealthy, adapting interactions")

    def get_interaction_status(self, interaction_id: str) -> Dict[str, Any]:
        """Get status information for an interaction."""
        if interaction_id not in self.active_interactions:
            raise InteractionError(f"Interaction {interaction_id} not found")
        
        context = self.active_interactions[interaction_id]
        
        return {
            "interaction_id": interaction_id,
            "session_id": context.session_id,
            "user_id": context.user_id,
            "state": context.state.value,
            "interaction_mode": context.interaction_mode.value,
            "start_time": context.start_time.isoformat(),
            "last_activity": context.last_activity.isoformat(),
            "duration": (datetime.now(timezone.utc) - context.start_time).total_seconds(),
            "message_count": len(context.conversation_history),
            "input_modalities": [m.value for m in context.input_modalities],
            "output_modalities": [m.value for m in context.output_modalities],
            "current_topic": context.current_topic,
            "user_intent": context.user_intent,
            "emotional_state": context.emotional_state
        }

    def get_active_interactions(self) -> List[Dict[str, Any]]:
        """Get list of all active interactions."""
        return [
            self.get_interaction_status(interaction_id)
            for interaction_id in self.active_interactions.keys()
        ]

    def get_user_profile(self, user_id: str) -> Dict[str, Any]:
        """Get user profile information."""
        if user_id not in self.user_profiles:
            raise InteractionError(f"User profile {user_id} not found")
        
        profile = self.user_profiles[user_id]
        return asdict(profile)

    async def update_user_preferences(
        self,
        user_id: str,
        preferences: Dict[str, Any]
    ) -> None:
        """Update user preferences."""
        if user_id not in self.user_profiles:
            await self._get_or_create_user_profile(user_id)
        
        profile = self.user_profiles[user_id]
        
        # Update preferences
        if "interaction_mode" in preferences:
            profile.preferred_interaction_mode = InteractionMode(preferences["interaction_mode"])
        
        if "input_modalities" in preferences:
            profile.preferred_input_modalities = {
                InputModality(m) for m in preferences["input_modalities"]
            }
        
        if "output_modalities" in preferences:
            profile.preferred_output_modalities = {
                OutputModality(m) for m in preferences["output_modalities"]
            }
        
        if "language" in preferences:
            profile.preferred_language = preferences["language"]
        
        # Update accessibility settings
        if "accessibility" in preferences:
            profile.accessibility_needs.update(preferences["accessibility"])
        
        # Emit preference update event
        await self.event_bus.emit(UserPreferenceUpdated(
            user_id=user_id,
            preferences=preferences
        ))
        
        self.logger.info(f"Updated preferences for user {user_id}")

    async def _health_check_callback(self) -> Dict[str, Any]:
        """Health check callback for the interaction handler."""
        try:
            active_count = len(self.active_interactions)
            avg_response_time = (
                sum(self.response_times) / len(self.response_times)
                if self.response_times else 0.0
            )
            
            return {
                "status": "healthy",
                "active_interactions": active_count,
                "average_response_time": avg_response_time,
                "total_user_profiles": len(self.user_profiles),
                "processors_count": len(self.input_processors),
                "generators_count": len(self.output_generators)
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }

    async def cleanup(self) -> None:
        """Cleanup resources and end all active interactions."""
        try:
            # End all active interactions
            active_interaction_ids = list(self.active_interactions.keys())
            for interaction_id in active_interaction_ids:
                await self.end_interaction(interaction_id, "system_shutdown")
            
            # Save user profiles
            # Would save to persistent storage
            
            self.logger.info("InteractionHandler cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")

    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            if hasattr(self, 'active_interactions') and self.active_interactions:
                self.logger.warning("InteractionHandler destroyed with active interactions")
        except Exception:
            pass  # Ignore cleanup errors in destructor
