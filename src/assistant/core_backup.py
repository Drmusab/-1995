"""
Core Assistant Orchestrator

This module serves as the central coordination point for the AI assistant,
managing the flow of information between different components and ensuring
proper integration with all subsystems.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
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
    TaskCompleted,
    SkillExecuted,
    MemoryUpdated,
    LearningEventOccurred,
    SystemHealthCheck
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


class CoreAssistantEngine:
    """
    Core orchestration engine for the AI assistant.
    
    This class coordinates all major subsystems including:
    - Natural language processing
    - Multimodal processing
    - Memory management
    - Skill execution
    - Learning and adaptation
    - Reasoning and planning
    """
    
    def __init__(self, container: Container):
        """Initialize the core assistant engine."""
        self.container = container
        self.logger = get_logger(__name__)
        
        # Core state
        self.state = AssistantState.UNINITIALIZED
        self.active_contexts: Dict[str, AssistantContext] = {}
        
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
            self.logger.info("Initializing Core Assistant Engine")
            
            # Inject dependencies
            await self._inject_dependencies()
            
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
            self.logger.info("Core Assistant Engine initialized successfully")
            
            # Emit initialization event
            await self.event_bus.emit(SystemHealthCheck(
                component="CoreAssistantEngine",
                status="initialized",
                details={"state": self.state.value}
            ))
            
        except Exception as e:
            self.state = AssistantState.ERROR
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
