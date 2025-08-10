"""
Response generation logic for the AI assistant.

This module handles the creation of contextual, personality-aware responses
across multiple modalities (text, speech, visual) with advanced formatting
and adaptation capabilities.
"""

import asyncio
import logging
import uuid
import json
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable
from dataclasses import dataclass, field
from collections import defaultdict
import re
import numpy as np

from src.core.events.event_bus import EventBus
from src.core.events.event_types import Event, EventType
from src.core.error_handling import (
    ValidationError,
    ResponseGenerationError
)
from src.core.config.loader import ConfigLoader
from src.core.security.sanitization import Sanitizer
from src.integrations.model_inference_coordinator import ModelInferenceCoordinator
from src.integrations.cache.cache_strategy import CacheStrategy
from src.assistant.personality import PersonalityManager
from src.assistant.context_manager import ContextManager, ContextType
from src.memory.core_memory.memory_manager import MemoryManager
from src.processing.natural_language.bilingual_manager import BilingualManager
from src.processing.speech.text_to_speech import TextToSpeech
from src.processing.multimodal.fusion_strategies import FusionStrategy
from src.reasoning.inference_engine import InferenceEngine
from src.learning.preference_learning import PreferenceLearner

logger = logging.getLogger(__name__)


class ResponseType(Enum):
    """Types of responses the system can generate."""
    TEXT = auto()
    SPEECH = auto()
    VISUAL = auto()
    MULTIMODAL = auto()
    ACTION = auto()
    CLARIFICATION = auto()
    ERROR = auto()
    SUGGESTION = auto()
    CONFIRMATION = auto()


class ResponseTone(Enum):
    """Tone variations for responses."""
    PROFESSIONAL = auto()
    CASUAL = auto()
    FRIENDLY = auto()
    EMPATHETIC = auto()
    EDUCATIONAL = auto()
    MOTIVATIONAL = auto()
    HUMOROUS = auto()
    SERIOUS = auto()
    SUPPORTIVE = auto()


class ResponseFormat(Enum):
    """Response formatting options."""
    PLAIN_TEXT = auto()
    MARKDOWN = auto()
    HTML = auto()
    JSON = auto()
    STRUCTURED = auto()
    CONVERSATIONAL = auto()
    BULLET_POINTS = auto()
    NUMBERED_LIST = auto()


@dataclass
class ResponseTemplate:
    """Template for response generation."""
    template_id: str
    name: str
    pattern: str
    variables: List[str]
    tone: ResponseTone
    format: ResponseFormat
    examples: List[str] = field(default_factory=list)
    constraints: Dict[str, Any] = field(default_factory=dict)
    
    def fill(self, variables: Dict[str, Any]) -> str:
        """Fill template with provided variables."""
        result = self.pattern
        for var_name, var_value in variables.items():
            placeholder = f"{{{var_name}}}"
            if placeholder in result:
                result = result.replace(placeholder, str(var_value))
        return result


@dataclass
class ResponseComponent:
    """Component of a multi-part response."""
    component_id: str
    type: ResponseType
    content: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    priority: int = 1
    optional: bool = False
    
    def is_valid(self) -> bool:
        """Check if component is valid."""
        return self.content is not None and self.type is not None


@dataclass
class GeneratedResponse:
    """Complete generated response with all components."""
    response_id: str
    session_id: str
    timestamp: datetime
    primary_content: str
    response_type: ResponseType
    components: List[ResponseComponent] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    tone: Optional[ResponseTone] = None
    format: ResponseFormat = ResponseFormat.PLAIN_TEXT
    language: str = "en"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary format."""
        return {
            "response_id": self.response_id,
            "session_id": self.session_id,
            "timestamp": self.timestamp.isoformat(),
            "content": self.primary_content,
            "type": self.response_type.name,
            "components": [
                {
                    "id": comp.component_id,
                    "type": comp.type.name,
                    "content": comp.content,
                    "metadata": comp.metadata
                }
                for comp in self.components
            ],
            "metadata": self.metadata,
            "confidence": self.confidence,
            "tone": self.tone.name if self.tone else None,
            "format": self.format.name,
            "language": self.language
        }


class ResponseGenerator:
    """
    Generates contextual, personality-aware responses for the AI assistant.
    
    This class orchestrates response creation across different modalities,
    applying appropriate formatting, tone, and personalization.
    """
    
    def __init__(
        self,
        event_bus: EventBus,
        model_coordinator: ModelInferenceCoordinator,
        personality_manager: PersonalityManager,
        context_manager: ContextManager,
        memory_manager: MemoryManager,
        bilingual_manager: BilingualManager,
        text_to_speech: TextToSpeech,
        inference_engine: InferenceEngine,
        preference_learner: PreferenceLearner,
        cache_strategy: CacheStrategy,
        sanitizer: Sanitizer,
        config_loader: ConfigLoader
    ):
        """Initialize the response generator with required dependencies."""
        self.event_bus = event_bus
        self.model_coordinator = model_coordinator
        self.personality_manager = personality_manager
        self.context_manager = context_manager
        self.memory_manager = memory_manager
        self.bilingual_manager = bilingual_manager
        self.text_to_speech = text_to_speech
        self.inference_engine = inference_engine
        self.preference_learner = preference_learner
        self.cache = cache_strategy
        self.sanitizer = sanitizer
        self.config = config_loader.get_config("response_generation")
        
        # Response templates
        self.templates: Dict[str, ResponseTemplate] = {}
        self._load_response_templates()
        
        # Generation strategies
        self.generation_strategies: Dict[ResponseType, Callable] = {
            ResponseType.TEXT: self._generate_text_response,
            ResponseType.SPEECH: self._generate_speech_response,
            ResponseType.VISUAL: self._generate_visual_response,
            ResponseType.MULTIMODAL: self._generate_multimodal_response,
            ResponseType.ACTION: self._generate_action_response,
            ResponseType.CLARIFICATION: self._generate_clarification_response,
            ResponseType.ERROR: self._generate_error_response,
            ResponseType.SUGGESTION: self._generate_suggestion_response,
            ResponseType.CONFIRMATION: self._generate_confirmation_response
        }
        
        # Configuration
        self.max_response_length = self.config.get("max_response_length", 2000)
        self.default_tone = ResponseTone[self.config.get("default_tone", "FRIENDLY")]
        self.enable_caching = self.config.get("enable_caching", True)
        self.cache_ttl = self.config.get("cache_ttl", 300)
        self.confidence_threshold = self.config.get("confidence_threshold", 0.7)
        
        # Response enhancement pipeline
        self.enhancement_pipeline = [
            self._apply_personality,
            self._apply_context_awareness,
            self._apply_user_preferences,
            self._apply_formatting,
            self._apply_safety_checks
        ]
        
        # Subscribe to events
        self._subscribe_to_events()
        
        logger.info("ResponseGenerator initialized successfully")
    
    def _subscribe_to_events(self) -> None:
        """Subscribe to relevant system events."""
        self.event_bus.subscribe(EventType.CONTEXT_UPDATED, self._handle_context_update)
        self.event_bus.subscribe(EventType.PREFERENCE_LEARNED, self._handle_preference_update)
        self.event_bus.subscribe(EventType.PERSONALITY_ADJUSTED, self._handle_personality_update)
    
    def _load_response_templates(self) -> None:
        """Load predefined response templates."""
        template_config = self.config.get("templates", {})
        
        # Default templates
        default_templates = [
            ResponseTemplate(
                template_id="greeting",
                name="Greeting Template",
                pattern="Hello {user_name}! {greeting_message}",
                variables=["user_name", "greeting_message"],
                tone=ResponseTone.FRIENDLY,
                format=ResponseFormat.CONVERSATIONAL,
                examples=["Hello John! How can I help you today?"]
            ),
            ResponseTemplate(
                template_id="clarification",
                name="Clarification Template",
                pattern="I'm not quite sure I understand. Did you mean {options}?",
                variables=["options"],
                tone=ResponseTone.PROFESSIONAL,
                format=ResponseFormat.CONVERSATIONAL
            ),
            ResponseTemplate(
                template_id="error",
                name="Error Template",
                pattern="I apologize, but {error_context}. {suggestion}",
                variables=["error_context", "suggestion"],
                tone=ResponseTone.EMPATHETIC,
                format=ResponseFormat.CONVERSATIONAL
            ),
            ResponseTemplate(
                template_id="confirmation",
                name="Confirmation Template",
                pattern="Got it! I'll {action}. {details}",
                variables=["action", "details"],
                tone=ResponseTone.PROFESSIONAL,
                format=ResponseFormat.CONVERSATIONAL
            )
        ]
        
        # Add default templates
        for template in default_templates:
            self.templates[template.template_id] = template
        
        # Load custom templates from config
        for template_id, template_data in template_config.items():
            try:
                template = ResponseTemplate(
                    template_id=template_id,
                    name=template_data.get("name", template_id),
                    pattern=template_data["pattern"],
                    variables=template_data.get("variables", []),
                    tone=ResponseTone[template_data.get("tone", "FRIENDLY")],
                    format=ResponseFormat[template_data.get("format", "CONVERSATIONAL")],
                    examples=template_data.get("examples", []),
                    constraints=template_data.get("constraints", {})
                )
                self.templates[template_id] = template
            except Exception as e:
                logger.error(f"Failed to load template {template_id}: {str(e)}")
    
    async def generate_response(
        self,
        session_id: str,
        input_data: Dict[str, Any],
        response_type: Optional[ResponseType] = None,
        tone: Optional[ResponseTone] = None,
        format: Optional[ResponseFormat] = None,
        constraints: Optional[Dict[str, Any]] = None
    ) -> GeneratedResponse:
        """
        Generate a response based on input and context.
        
        Args:
            session_id: Session identifier
            input_data: Input data including intent, entities, context
            response_type: Type of response to generate
            tone: Desired tone for the response
            format: Desired format for the response
            constraints: Additional constraints for generation
            
        Returns:
            Generated response with all components
        """
        try:
            # Determine response type if not specified
            if not response_type:
                response_type = await self._determine_response_type(
                    session_id, input_data
                )
            
            # Check cache if enabled
            if self.enable_caching:
                cached_response = await self._check_cache(
                    session_id, input_data, response_type
                )
                if cached_response:
                    return cached_response
            
            # Get current context
            context = await self.context_manager.get_current_context(session_id)
            multimodal_context = await self.context_manager.get_multimodal_context(
                session_id
            )
            
            # Get user preferences
            user_preferences = await self._get_user_preferences(session_id)
            
            # Determine tone and format
            if not tone:
                tone = await self._determine_tone(
                    input_data, context, user_preferences
                )
            if not format:
                format = self._determine_format(
                    response_type, user_preferences
                )
            
            # Build generation context
            generation_context = {
                "session_id": session_id,
                "input": input_data,
                "context": context,
                "multimodal_context": multimodal_context,
                "user_preferences": user_preferences,
                "tone": tone,
                "format": format,
                "constraints": constraints or {},
                "personality": await self.personality_manager.get_current_traits(
                    session_id
                )
            }
            
            # Generate response using appropriate strategy
            strategy = self.generation_strategies.get(response_type)
            if not strategy:
                raise ResponseGenerationError(
                    f"No strategy for response type: {response_type}"
                )
            
            raw_response = await strategy(generation_context)
            
            # Enhance response through pipeline
            enhanced_response = await self._enhance_response(
                raw_response, generation_context
            )
            
            # Create final response
            response = GeneratedResponse(
                response_id=str(uuid.uuid4()),
                session_id=session_id,
                timestamp=datetime.now(timezone.utc),
                primary_content=enhanced_response["content"],
                response_type=response_type,
                components=enhanced_response.get("components", []),
                metadata=enhanced_response.get("metadata", {}),
                confidence=enhanced_response.get("confidence", 1.0),
                tone=tone,
                format=format,
                language=enhanced_response.get("language", "en")
            )
            
            # Cache response
            if self.enable_caching:
                await self._cache_response(response, input_data)
            
            # Emit response generated event
            await self.event_bus.emit(Event(
                type=EventType.RESPONSE_GENERATED,
                data={
                    "session_id": session_id,
                    "response_id": response.response_id,
                    "response_type": response_type.name,
                    "confidence": response.confidence
                }
            ))
            
            return response
            
        except Exception as e:
            logger.error(f"Failed to generate response: {str(e)}")
            # Generate error response
            return await self._generate_fallback_response(
                session_id, str(e), response_type
            )
    
    async def _determine_response_type(
        self,
        session_id: str,
        input_data: Dict[str, Any]
    ) -> ResponseType:
        """Determine appropriate response type based on input."""
        intent = input_data.get("intent", {}).get("name")
        
        # Check for specific intents
        if intent == "clarification_needed":
            return ResponseType.CLARIFICATION
        elif intent == "action_request":
            return ResponseType.ACTION
        elif intent == "suggestion_request":
            return ResponseType.SUGGESTION
        elif intent == "confirmation_request":
            return ResponseType.CONFIRMATION
        elif "error" in input_data:
            return ResponseType.ERROR
        
        # Check modality preferences
        if input_data.get("prefer_speech"):
            return ResponseType.SPEECH
        elif input_data.get("visual_context"):
            return ResponseType.MULTIMODAL
        
        # Default to text
        return ResponseType.TEXT
    
    async def _determine_tone(
        self,
        input_data: Dict[str, Any],
        context: Dict[str, Any],
        preferences: Dict[str, Any]
    ) -> ResponseTone:
        """Determine appropriate tone for response."""
        # Check user preferences
        preferred_tone = preferences.get("preferred_tone")
        if preferred_tone:
            try:
                return ResponseTone[preferred_tone]
            except KeyError:
                pass
        
        # Check emotional context
        emotional_context = context.get(ContextType.EMOTIONAL.name, {})
        if emotional_context:
            emotion = emotional_context.get("latest", {}).get("overall_emotion")
            if emotion in ["sad", "frustrated", "anxious"]:
                return ResponseTone.EMPATHETIC
            elif emotion in ["happy", "excited"]:
                return ResponseTone.FRIENDLY
        
        # Check task context
        task_context = context.get(ContextType.TASK.name, {})
        if task_context:
            task_type = task_context.get("latest", {}).get("task_type")
            if task_type == "professional":
                return ResponseTone.PROFESSIONAL
            elif task_type == "learning":
                return ResponseTone.EDUCATIONAL
        
        # Check time of day
        temporal_context = context.get(ContextType.TEMPORAL.name, {})
        if temporal_context:
            time_of_day = temporal_context.get("latest", {}).get("time_of_day")
            if time_of_day in ["morning", "afternoon"]:
                return ResponseTone.MOTIVATIONAL
            elif time_of_day in ["evening", "night"]:
                return ResponseTone.CASUAL
        
        # Default tone
        return self.default_tone
    
    def _determine_format(
        self,
        response_type: ResponseType,
        preferences: Dict[str, Any]
    ) -> ResponseFormat:
        """Determine appropriate format for response."""
        # Check user preferences
        preferred_format = preferences.get("preferred_format")
        if preferred_format:
            try:
                return ResponseFormat[preferred_format]
            except KeyError:
                pass
        
        # Format based on response type
        format_mapping = {
            ResponseType.TEXT: ResponseFormat.CONVERSATIONAL,
            ResponseType.ACTION: ResponseFormat.STRUCTURED,
            ResponseType.SUGGESTION: ResponseFormat.BULLET_POINTS,
            ResponseType.ERROR: ResponseFormat.CONVERSATIONAL,
            ResponseType.CLARIFICATION: ResponseFormat.CONVERSATIONAL
        }
        
        return format_mapping.get(response_type, ResponseFormat.PLAIN_TEXT)
    
    async def _generate_text_response(
        self,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate text-based response."""
        # Build prompt
        prompt = await self._build_text_prompt(context)
        
        # Select model based on context
        model_selection = await self.model_coordinator.select_model(
            task_type="text_generation",
            context=context
        )
        
        # Generate response
        model_response = await self.model_coordinator.generate(
            model=model_selection["model"],
            prompt=prompt,
            parameters={
                "temperature": self._get_temperature_for_tone(context["tone"]),
                "max_tokens": min(
                    self.max_response_length,
                    context.get("constraints", {}).get("max_length", self.max_response_length)
                ),
                "top_p": 0.9
            }
        )
        
        # Post-process text
        processed_text = await self._post_process_text(
            model_response["text"],
            context
        )
        
        return {
            "content": processed_text,
            "confidence": model_response.get("confidence", 0.9),
            "metadata": {
                "model": model_selection["model"],
                "prompt_tokens": model_response.get("prompt_tokens"),
                "completion_tokens": model_response.get("completion_tokens")
            }
        }
    
    async def _generate_speech_response(
        self,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate speech-based response."""
        # First generate text
        text_response = await self._generate_text_response(context)
        
        # Convert to speech
        speech_config = {
            "voice": context["user_preferences"].get("preferred_voice", "default"),
            "speed": context["user_preferences"].get("speech_speed", 1.0),
            "pitch": context["user_preferences"].get("speech_pitch", 1.0),
            "emotion": context["tone"].name.lower()
        }
        
        audio_data = await self.text_to_speech.synthesize(
            text_response["content"],
            **speech_config
        )
        
        # Create response with both text and audio
        components = [
            ResponseComponent(
                component_id=str(uuid.uuid4()),
                type=ResponseType.TEXT,
                content=text_response["content"],
                metadata={"role": "transcript"}
            ),
            ResponseComponent(
                component_id=str(uuid.uuid4()),
                type=ResponseType.SPEECH,
                content=audio_data,
                metadata={
                    "format": "audio/wav",
                    "duration": audio_data.get("duration"),
                    "voice": speech_config["voice"]
                }
            )
        ]
        
        return {
            "content": text_response["content"],
            "components": components,
            "confidence": text_response["confidence"],
            "metadata": {
                **text_response["metadata"],
                "speech_config": speech_config
            }
        }
    
    async def _generate_visual_response(
        self,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate visual response (charts, diagrams, etc.)."""
        visual_type = context["input"].get("visual_type", "chart")
        visual_data = context["input"].get("visual_data", {})
        
        # Generate visual description
        description_prompt = f"Generate a description for a {visual_type} showing: {visual_data}"
        
        description_response = await self.model_coordinator.generate(
            model="text_generation",
            prompt=description_prompt,
            parameters={"temperature": 0.3, "max_tokens": 200}
        )
        
        # Create visual component (placeholder for actual visual generation)
        visual_component = ResponseComponent(
            component_id=str(uuid.uuid4()),
            type=ResponseType.VISUAL,
            content={
                "type": visual_type,
                "data": visual_data,
                "description": description_response["text"]
            },
            metadata={"format": "application/json"}
        )
        
        return {
            "content": description_response["text"],
            "components": [visual_component],
            "confidence": 0.85,
            "metadata": {"visual_type": visual_type}
        }
    
    async def _generate_multimodal_response(
        self,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate response with multiple modalities."""
        components = []
        
        # Generate text component
        text_response = await self._generate_text_response(context)
        components.append(
            ResponseComponent(
                component_id=str(uuid.uuid4()),
                type=ResponseType.TEXT,
                content=text_response["content"],
                priority=1
            )
        )
        
        # Add speech if requested
        if context["user_preferences"].get("include_speech", False):
            speech_response = await self._generate_speech_response(context)
            components.extend(speech_response["components"])
        
        # Add visual if relevant
        if context["input"].get("visual_data"):
            visual_response = await self._generate_visual_response(context)
            components.extend(visual_response["components"])
        
        return {
            "content": text_response["content"],
            "components": components,
            "confidence": text_response["confidence"],
            "metadata": {
                "modalities": [comp.type.name for comp in components]
            }
        }
    
    async def _generate_action_response(
        self,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate response for action requests."""
        action = context["input"].get("action", {})
        
        # Use template if available
        if "confirmation" in self.templates:
            template = self.templates["confirmation"]
            content = template.fill({
                "action": action.get("name", "perform that action"),
                "details": action.get("details", "")
            })
        else:
            content = f"I'll {action.get('name', 'help with that')}. {action.get('details', '')}"
        
        # Create action component
        action_component = ResponseComponent(
            component_id=str(uuid.uuid4()),
            type=ResponseType.ACTION,
            content={
                "action": action.get("name"),
                "parameters": action.get("parameters", {}),
                "status": "pending"
            },
            metadata={"executable": True}
        )
        
        return {
            "content": content,
            "components": [action_component],
            "confidence": 0.95,
            "metadata": {"action_type": action.get("name")}
        }
    
    async def _generate_clarification_response(
        self,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate clarification request."""
        ambiguity = context["input"].get("ambiguity", {})
        options = ambiguity.get("options", [])
        
        # Use template
        if "clarification" in self.templates:
            template = self.templates["clarification"]
            options_text = " or ".join([f'"{opt}"' for opt in options[:3]])
            content = template.fill({"options": options_text})
        else:
            content = f"Could you clarify what you mean? {' or '.join(options)}"
        
        # Add suggestions component
        suggestions_component = ResponseComponent(
            component_id=str(uuid.uuid4()),
            type=ResponseType.SUGGESTION,
            content=options,
            metadata={"type": "clarification_options"}
        )
        
        return {
            "content": content,
            "components": [suggestions_component],
            "confidence": 0.7,
            "metadata": {"ambiguity_type": ambiguity.get("type")}
        }
    
    async def _generate_error_response(
        self,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate error response."""
        error = context["input"].get("error", {})
        
        # Use template
        if "error" in self.templates:
            template = self.templates["error"]
            content = template.fill({
                "error_context": error.get("message", "something went wrong"),
                "suggestion": error.get("suggestion", "Please try again.")
            })
        else:
            content = f"I apologize, but {error.get('message', 'an error occurred')}. {error.get('suggestion', '')}"
        
        return {
            "content": content,
            "confidence": 1.0,
            "metadata": {
                "error_type": error.get("type"),
                "error_code": error.get("code")
            }
        }
    
    async def _generate_suggestion_response(
        self,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate suggestion response."""
        suggestions = context["input"].get("suggestions", [])
        
        # Format suggestions
        if context["format"] == ResponseFormat.BULLET_POINTS:
            content = "Here are my suggestions:\n"
            content += "\n".join([f"• {s}" for s in suggestions])
        elif context["format"] == ResponseFormat.NUMBERED_LIST:
            content = "Here are my suggestions:\n"
            content += "\n".join([f"{i+1}. {s}" for i, s in enumerate(suggestions)])
        else:
            content = f"I suggest: {', '.join(suggestions)}"
        
        # Create suggestions component
        suggestions_component = ResponseComponent(
            component_id=str(uuid.uuid4()),
            type=ResponseType.SUGGESTION,
            content=suggestions,
            metadata={"interactive": True}
        )
        
        return {
            "content": content,
            "components": [suggestions_component],
            "confidence": 0.85,
            "metadata": {"suggestion_count": len(suggestions)}
        }
    
    async def _generate_confirmation_response(
        self,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate confirmation response."""
        confirmation = context["input"].get("confirmation", {})
        
        # Use appropriate tone
        if context["tone"] == ResponseTone.CASUAL:
            content = f"Sure thing! {confirmation.get('message', '')}"
        elif context["tone"] == ResponseTone.PROFESSIONAL:
            content = f"Confirmed. {confirmation.get('message', '')}"
        else:
            content = f"Got it! {confirmation.get('message', '')}"
        
        return {
            "content": content,
            "confidence": 1.0,
            "metadata": {"confirmation_type": confirmation.get("type")}
        }
    
    async def _build_text_prompt(self, context: Dict[str, Any]) -> str:
        """Build prompt for text generation."""
        prompt_parts = []
        
        # Add personality traits
        personality = context.get("personality", {})
        if personality:
            traits = [f"{k}: {v}" for k, v in personality.items()]
            prompt_parts.append(f"Assistant personality: {', '.join(traits)}")
        
        # Add context
        if context.get("context"):
            relevant_context = self._extract_relevant_context(context["context"])
            if relevant_context:
                prompt_parts.append(f"Context: {relevant_context}")
        
        # Add conversation history
        if context.get("input", {}).get("conversation_history"):
            history = context["input"]["conversation_history"][-3:]  # Last 3 turns
            prompt_parts.append("Recent conversation:")
            for turn in history:
                prompt_parts.append(f"{turn['role']}: {turn['content']}")
        
        # Add user input
        user_input = context["input"].get("user_input", "")
        prompt_parts.append(f"User: {user_input}")
        
        # Add tone instruction
        tone_instruction = self._get_tone_instruction(context["tone"])
        prompt_parts.append(f"Respond in a {tone_instruction} tone:")
        
        # Add format instruction if needed
        if context["format"] != ResponseFormat.PLAIN_TEXT:
            format_instruction = self._get_format_instruction(context["format"])
            prompt_parts.append(format_instruction)
        
        return "\n\n".join(prompt_parts)
    
    def _extract_relevant_context(self, context: Dict[str, Any]) -> str:
        """Extract relevant context information for prompt."""
        relevant_parts = []
        
        # Task context
        if ContextType.TASK.name in context:
            task_data = context[ContextType.TASK.name].get("latest", {})
            if task_data.get("current_task"):
                relevant_parts.append(f"Current task: {task_data['current_task']}")
        
        # Emotional context
        if ContextType.EMOTIONAL.name in context:
            emotion_data = context[ContextType.EMOTIONAL.name].get("latest", {})
            if emotion_data.get("overall_emotion"):
                relevant_parts.append(f"User emotion: {emotion_data['overall_emotion']}")
        
        # User preferences
        if ContextType.USER_PREFERENCE.name in context:
            pref_data = context[ContextType.USER_PREFERENCE.name].get("latest", {})
            if pref_data:
                relevant_parts.append(f"User preferences: {pref_data}")
        
        return "; ".join(relevant_parts)
    
    def _get_tone_instruction(self, tone: ResponseTone) -> str:
        """Get instruction for specific tone."""
        tone_instructions = {
            ResponseTone.PROFESSIONAL: "professional and formal",
            ResponseTone.CASUAL: "casual and relaxed",
            ResponseTone.FRIENDLY: "warm and friendly",
            ResponseTone.EMPATHETIC: "understanding and empathetic",
            ResponseTone.EDUCATIONAL: "clear and educational",
            ResponseTone.MOTIVATIONAL: "encouraging and motivational",
            ResponseTone.HUMOROUS: "light-hearted and humorous",
            ResponseTone.SERIOUS: "serious and focused",
            ResponseTone.SUPPORTIVE: "supportive and reassuring"
        }
        return tone_instructions.get(tone, "helpful")
    
    def _get_format_instruction(self, format: ResponseFormat) -> str:
        """Get instruction for specific format."""
        format_instructions = {
            ResponseFormat.MARKDOWN: "Format the response using Markdown syntax.",
            ResponseFormat.HTML: "Format the response using HTML tags.",
            ResponseFormat.JSON: "Structure the response as valid JSON.",
            ResponseFormat.BULLET_POINTS: "Format the response as bullet points.",
            ResponseFormat.NUMBERED_LIST: "Format the response as a numbered list.",
            ResponseFormat.STRUCTURED: "Provide a well-structured response with clear sections."
        }
        return format_instructions.get(format, "")
    
    def _get_temperature_for_tone(self, tone: ResponseTone) -> float:
        """Get temperature setting for tone."""
        temperature_map = {
            ResponseTone.PROFESSIONAL: 0.3,
            ResponseTone.CASUAL: 0.8,
            ResponseTone.FRIENDLY: 0.7,
            ResponseTone.EMPATHETIC: 0.6,
            ResponseTone.EDUCATIONAL: 0.4,
            ResponseTone.MOTIVATIONAL: 0.7,
            ResponseTone.HUMOROUS: 0.9,
            ResponseTone.SERIOUS: 0.2,
            ResponseTone.SUPPORTIVE: 0.6
        }
        return temperature_map.get(tone, 0.7)
    
    async def _post_process_text(
        self,
        text: str,
        context: Dict[str, Any]
    ) -> str:
        """Post-process generated text."""
        # Remove any unwanted artifacts
        text = self._clean_text(text)
        
        # Apply format-specific processing
        if context["format"] == ResponseFormat.MARKDOWN:
            text = self._ensure_valid_markdown(text)
        elif context["format"] == ResponseFormat.HTML:
            text = self._ensure_valid_html(text)
        elif context["format"] == ResponseFormat.JSON:
            text = self._ensure_valid_json(text)
        
        # Ensure appropriate length
        max_length = context.get("constraints", {}).get("max_length", self.max_response_length)
        if len(text) > max_length:
            text = self._truncate_intelligently(text, max_length)
        
        # Ensure completeness
        text = self._ensure_complete_sentences(text)
        
        return text
    
    def _clean_text(self, text: str) -> str:
        """Clean generated text."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove incomplete sentences at the end
        if text and text[-1] not in '.!?':
            last_period = text.rfind('.')
            if last_period > len(text) * 0.8:  # If we have most of the content
                text = text[:last_period + 1]
        
        return text
    
    def _ensure_valid_markdown(self, text: str) -> str:
        """Ensure text is valid Markdown."""
        # Basic Markdown validation and fixing
        # Ensure lists are properly formatted
        text = re.sub(r'^- ', '- ', text, flags=re.MULTILINE)
        text = re.sub(r'^\* ', '* ', text, flags=re.MULTILINE)
        
        # Ensure headers have space after #
        text = re.sub(r'^(#{1,6})([^ ])', r'\1 \2', text, flags=re.MULTILINE)
        
        return text
    
    def _ensure_valid_html(self, text: str) -> str:
        """Ensure text contains valid HTML."""
        # Basic HTML validation
        # This is simplified - in production, use proper HTML parser
        if not text.strip().startswith('<'):
            text = f"<p>{text}</p>"
        
        return text
    
    def _ensure_valid_json(self, text: str) -> str:
        """Ensure text is valid JSON."""
        try:
            # Try to parse and re-serialize
            data = json.loads(text)
            return json.dumps(data, indent=2)
        except json.JSONDecodeError:
            # If not valid JSON, wrap in object
            return json.dumps({"response": text}, indent=2)
    
    def _truncate_intelligently(self, text: str, max_length: int) -> str:
        """Truncate text intelligently at sentence boundaries."""
        if len(text) <= max_length:
            return text
        
        # Find last complete sentence within limit
        sentences = re.split(r'(?<=[.!?])\s+', text)
        truncated = ""
        
        for sentence in sentences:
            if len(truncated) + len(sentence) + 1 <= max_length:
                truncated += sentence + " "
            else:
                break
        
        return truncated.strip()
    
    def _ensure_complete_sentences(self, text: str) -> str:
        """Ensure text ends with complete sentences."""
        if not text:
            return text
        
        # Check if text ends with proper punctuation
        if text[-1] not in '.!?':
            # Try to complete the sentence
            if '.' in text:
                # Find last complete sentence
                last_period = text.rfind('.')
                if last_period > 0:
                    return text[:last_period + 1]
            
            # Otherwise, add period
            text += '.'
        
        return text
    
    async def _enhance_response(
        self,
        response: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Enhance response through pipeline."""
        enhanced = response.copy()
        
        for enhancer in self.enhancement_pipeline:
            enhanced = await enhancer(enhanced, context)
        
        return enhanced
    
    async def _apply_personality(
        self,
        response: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply personality traits to response."""
        personality = context.get("personality", {})
        
        if not personality:
            return response
        
        # Adjust response based on personality traits
        content = response["content"]
        
        # Apply trait-specific modifications
        if personality.get("humor", 0) > 0.7 and context["tone"] != ResponseTone.SERIOUS:
            # Add light humor if appropriate
            content = await self._add_humor(content, context)
        
        if personality.get("formality", 0) < 0.3:
            # Make more casual
            content = self._make_casual(content)
        
        if personality.get("verbosity", 0) > 0.7:
            # Add more detail
            content = await self._add_detail(content, context)
        
        response["content"] = content
        return response
    
    async def _apply_context_awareness(
        self,
        response: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply context awareness to response."""
        # Add time-aware greetings
        temporal_context = context["context"].get(ContextType.TEMPORAL.name, {})
        if temporal_context:
            time_of_day = temporal_context.get("latest", {}).get("time_of_day")
            if time_of_day and "Hello" in response["content"]:
                greeting = self._get_time_appropriate_greeting(time_of_day)
                response["content"] = response["content"].replace("Hello", greeting)
        
        # Add activity-aware references
        task_context = context["context"].get(ContextType.TASK.name, {})
        if task_context:
            current_task = task_context.get("latest", {}).get("current_task")
            if current_task:
                response["metadata"]["referenced_task"] = current_task
        
        return response
    
    async def _apply_user_preferences(
        self,
        response: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply user preferences to response."""
        preferences = context.get("user_preferences", {})
        
        # Apply language preferences
        if preferences.get("preferred_language") != "en":
            response = await self._translate_response(
                response,
                preferences["preferred_language"]
            )
        
        # Apply length preferences
        if preferences.get("response_length") == "concise":
            response["content"] = self._make_concise(response["content"])
        elif preferences.get("response_length") == "detailed":
            response["content"] = await self._make_detailed(
                response["content"],
                context
            )
        
        # Apply vocabulary preferences
        if preferences.get("vocabulary_level"):
            response["content"] = self._adjust_vocabulary(
                response["content"],
                preferences["vocabulary_level"]
            )
        
        return response
    
    async def _apply_formatting(
        self,
        response: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply formatting to response."""
        format = context["format"]
        content = response["content"]
        
        # Apply format-specific transformations
        if format == ResponseFormat.MARKDOWN:
            content = self._format_as_markdown(content)
        elif format == ResponseFormat.HTML:
            content = self._format_as_html(content)
        elif format == ResponseFormat.BULLET_POINTS:
            content = self._format_as_bullets(content)
        elif format == ResponseFormat.NUMBERED_LIST:
            content = self._format_as_numbered_list(content)
        
        response["content"] = content
        return response
    
    async def _apply_safety_checks(
        self,
        response: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply safety checks to response."""
        # Sanitize content
        response["content"] = await self.sanitizer.sanitize_output(
            response["content"]
        )
        
        # Check for sensitive information
        if self._contains_sensitive_info(response["content"]):
            response["content"] = self._redact_sensitive_info(response["content"])
            response["metadata"]["redacted"] = True
        
        # Ensure response is appropriate
        if not self._is_appropriate(response["content"], context):
            response = await self._generate_fallback_response(
                context["session_id"],
                "Response failed safety check",
                ResponseType.TEXT
            )
        
        return response
    
    async def _add_humor(self, content: str, context: Dict[str, Any]) -> str:
        """Add appropriate humor to content."""
        # Simple humor injection (would be more sophisticated in production)
        humor_phrases = [
            "By the way, ",
            "Fun fact: ",
            "Here's something interesting: "
        ]
        
        if len(content) > 50 and np.random.random() > 0.7:
            humor_prompt = f"Add a light, appropriate joke or pun related to: {content[:100]}"
            humor_response = await self.model_coordinator.generate(
                model="text_generation",
                prompt=humor_prompt,
                parameters={"temperature": 0.9, "max_tokens": 50}
            )
            
            if humor_response.get("text"):
                content += f" {np.random.choice(humor_phrases)}{humor_response['text']}"
        
        return content
    
    def _make_casual(self, content: str) -> str:
        """Make content more casual."""
        replacements = {
            "I would": "I'd",
            "It is": "It's",
            "You are": "You're",
            "Cannot": "Can't",
            "Will not": "Won't",
            "Therefore": "So",
            "However": "But"
        }
        
        for formal, casual in replacements.items():
            content = content.replace(formal, casual)
            content = content.replace(formal.lower(), casual.lower())
        
        return content
    
    async def _add_detail(self, content: str, context: Dict[str, Any]) -> str:
        """Add more detail to content."""
        if len(content) < 100:  # Only expand short responses
            detail_prompt = f"Expand on this with more helpful details: {content}"
            detail_response = await self.model_coordinator.generate(
                model="text_generation",
                prompt=detail_prompt,
                parameters={"temperature": 0.5, "max_tokens": 200}
            )
            
            if detail_response.get("text"):
                content += f" {detail_response['text']}"
        
        return content
    
    def _get_time_appropriate_greeting(self, time_of_day: str) -> str:
        """Get greeting appropriate for time of day."""
        greetings = {
            "morning": "Good morning",
            "afternoon": "Good afternoon",
            "evening": "Good evening",
            "night": "Good evening"
        }
        return greetings.get(time_of_day, "Hello")
    
    async def _translate_response(
        self,
        response: Dict[str, Any],
        target_language: str
    ) -> Dict[str, Any]:
        """Translate response to target language."""
        # Use bilingual manager for translation
        translated_content = await self.bilingual_manager.translate(
            response["content"],
            target_language=target_language
        )
        
        response["content"] = translated_content
        response["language"] = target_language
        
        # Translate components if present
        if response.get("components"):
            for component in response["components"]:
                if component.type == ResponseType.TEXT:
                    component.content = await self.bilingual_manager.translate(
                        component.content,
                        target_language=target_language
                    )
        
        return response
    
    def _make_concise(self, content: str) -> str:
        """Make content more concise."""
        # Remove filler words and phrases
        filler_patterns = [
            r'\b(basically|actually|really|very|quite|just)\b',
            r'\b(in fact|as a matter of fact|to be honest)\b',
            r'\b(you know|I mean|I think)\b'
        ]
        
        for pattern in filler_patterns:
            content = re.sub(pattern, '', content, flags=re.IGNORECASE)
        
        # Clean up extra spaces
        content = re.sub(r'\s+', ' ', content).strip()
        
        # Keep only essential sentences
        sentences = re.split(r'(?<=[.!?])\s+', content)
        if len(sentences) > 3:
            # Keep first and last sentence plus most important middle one
            essential = [sentences[0], sentences[-1]]
            content = ' '.join(essential)
        
        return content
    
    async def _make_detailed(
        self,
        content: str,
        context: Dict[str, Any]
    ) -> str:
        """Make content more detailed."""
        # Add examples or elaboration
        if "example" not in content.lower():
            example_prompt = f"Provide a relevant example for: {content}"
            example_response = await self.model_coordinator.generate(
                model="text_generation",
                prompt=example_prompt,
                parameters={"temperature": 0.5, "max_tokens": 100}
            )
            
            if example_response.get("text"):
                content += f" For example, {example_response['text']}"
        
        return content
    
    def _adjust_vocabulary(self, content: str, level: str) -> str:
        """Adjust vocabulary level."""
        if level == "simple":
            # Replace complex words with simpler alternatives
            replacements = {
                "utilize": "use",
                "implement": "do",
                "commence": "start",
                "terminate": "end",
                "demonstrate": "show",
                "facilitate": "help"
            }
            
            for complex, simple in replacements.items():
                content = re.sub(
                    rf'\b{complex}\b',
                    simple,
                    content,
                    flags=re.IGNORECASE
                )
        
        return content
    
    def _format_as_markdown(self, content: str) -> str:
        """Format content as Markdown."""
        # Add basic Markdown formatting
        lines = content.split('\n')
        formatted = []
        
        for line in lines:
            if line.strip():
                # Make first line a header if it's short
                if len(formatted) == 0 and len(line) < 50:
                    formatted.append(f"## {line}")
                else:
                    formatted.append(line)
        
        return '\n\n'.join(formatted)
    
    def _format_as_html(self, content: str) -> str:
        """Format content as HTML."""
        # Basic HTML formatting
        paragraphs = content.split('\n\n')
        html_parts = []
        
        for para in paragraphs:
            if para.strip():
                html_parts.append(f"<p>{para.strip()}</p>")
        
        return '\n'.join(html_parts)
    
    def _format_as_bullets(self, content: str) -> str:
        """Format content as bullet points."""
        # Split into sentences and format as bullets
        sentences = re.split(r'(?<=[.!?])\s+', content)
        
        if len(sentences) > 1:
            bullets = [f"• {sent}" for sent in sentences if sent.strip()]
            return '\n'.join(bullets)
        
        return f"• {content}"
    
    def _format_as_numbered_list(self, content: str) -> str:
        """Format content as numbered list."""
        # Split into sentences and format as numbered list
        sentences = re.split(r'(?<=[.!?])\s+', content)
        
        if len(sentences) > 1:
            numbered = [
                f"{i+1}. {sent}"
                for i, sent in enumerate(sentences)
                if sent.strip()
            ]
            return '\n'.join(numbered)
        
        return f"1. {content}"
    
    def _contains_sensitive_info(self, content: str) -> bool:
        """Check if content contains sensitive information."""
        # Check for patterns that might indicate sensitive info
        sensitive_patterns = [
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
            r'\b\d{16}\b',  # Credit card
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
            r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'  # Phone
        ]
        
        for pattern in sensitive_patterns:
            if re.search(pattern, content):
                return True
        
        return False
    
    def _redact_sensitive_info(self, content: str) -> str:
        """Redact sensitive information from content."""
        # Redact various sensitive patterns
        redactions = {
            r'\b\d{3}-\d{2}-\d{4}\b': '[SSN REDACTED]',
            r'\b\d{16}\b': '[CARD NUMBER REDACTED]',
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b': '[EMAIL REDACTED]',
            r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b': '[PHONE REDACTED]'
        }
        
        for pattern, replacement in redactions.items():
            content = re.sub(pattern, replacement, content)
        
        return content
    
    def _is_appropriate(self, content: str, context: Dict[str, Any]) -> bool:
        """Check if content is appropriate."""
        # Simple appropriateness check (would be more sophisticated in production)
        inappropriate_terms = self.config.get("inappropriate_terms", [])
        
        content_lower = content.lower()
        for term in inappropriate_terms:
            if term.lower() in content_lower:
                return False
        
        return True
    
    async def _generate_fallback_response(
        self,
        session_id: str,
        error_message: str,
        response_type: Optional[ResponseType]
    ) -> GeneratedResponse:
        """Generate fallback response for errors."""
        content = "I apologize, but I'm having trouble generating a response. Please try again or rephrase your request."
        
        if error_message:
            logger.error(f"Fallback response triggered: {error_message}")
        
        return GeneratedResponse(
            response_id=str(uuid.uuid4()),
            session_id=session_id,
            timestamp=datetime.now(timezone.utc),
            primary_content=content,
            response_type=response_type or ResponseType.ERROR,
            confidence=0.5,
            tone=ResponseTone.EMPATHETIC,
            format=ResponseFormat.CONVERSATIONAL,
            metadata={"fallback": True, "error": error_message}
        )
    
    async def _check_cache(
        self,
        session_id: str,
        input_data: Dict[str, Any],
        response_type: ResponseType
    ) -> Optional[GeneratedResponse]:
        """Check cache for existing response."""
        # Create cache key from input
        cache_key = self._create_cache_key(session_id, input_data, response_type)
        
        cached_data = await self.cache.get(cache_key)
        if cached_data:
            # Reconstruct response from cached data
            return GeneratedResponse(**cached_data)
        
        return None
    
    async def _cache_response(
        self,
        response: GeneratedResponse,
        input_data: Dict[str, Any]
    ) -> None:
        """Cache generated response."""
        cache_key = self._create_cache_key(
            response.session_id,
            input_data,
            response.response_type
        )
        
        # Don't cache certain response types
        if response.response_type in [ResponseType.ERROR, ResponseType.CLARIFICATION]:
            return
        
        await self.cache.set(
            cache_key,
            response.to_dict(),
            ttl=self.cache_ttl
        )
    
    def _create_cache_key(
        self,
        session_id: str,
        input_data: Dict[str, Any],
        response_type: ResponseType
    ) -> str:
        """Create cache key from input."""
        # Create deterministic key from relevant input parts
        key_parts = [
            session_id,
            response_type.name,
            input_data.get("intent", {}).get("name", ""),
            str(sorted(input_data.get("entities", [])))
        ]
        
        # Add user input if present
        if "user_input" in input_data:
            # Use first 100 chars of user input
            key_parts.append(input_data["user_input"][:100])
        
        return ":".join(key_parts)
    
    async def _get_user_preferences(self, session_id: str) -> Dict[str, Any]:
        """Get user preferences for session."""
        try:
            # Get from preference learner
            preferences = await self.preference_learner.get_preferences(session_id)
            return preferences or {}
        except Exception as e:
            logger.warning(f"Failed to get user preferences: {str(e)}")
            return {}
    
    async def _handle_context_update(self, event: Event) -> None:
        """Handle context update events."""
        # Context updates might affect response generation
        # Could trigger cache invalidation or preference updates
        pass
    
    async def _handle_preference_update(self, event: Event) -> None:
        """Handle preference update events."""
        # Update any cached preferences
        session_id = event.data.get("session_id")
        if session_id:
            # Invalidate relevant caches
            await self.cache.invalidate_pattern(f"*{session_id}*")
    
    async def _handle_personality_update(self, event: Event) -> None:
        """Handle personality update events."""
        # Personality changes affect response generation
        session_id = event.data.get("session_id")
        if session_id:
            # Invalidate response caches for this session
            await self.cache.invalidate_pattern(f"*{session_id}*")
    
    async def regenerate_response(
        self,
        response_id: str,
        modifications: Optional[Dict[str, Any]] = None
    ) -> GeneratedResponse:
        """Regenerate a previous response with modifications."""
        # This would retrieve the original context and regenerate
        # with any requested modifications
        raise NotImplementedError("Response regeneration not yet implemented")
    
    async def get_response_variations(
        self,
        session_id: str,
        input_data: Dict[str, Any],
        num_variations: int = 3
    ) -> List[GeneratedResponse]:
        """Generate multiple response variations."""
        variations = []
        
        # Generate variations with different tones
        tones = [ResponseTone.FRIENDLY, ResponseTone.PROFESSIONAL, ResponseTone.CASUAL]
        
        for i, tone in enumerate(tones[:num_variations]):
            response = await self.generate_response(
                session_id,
                input_data,
                tone=tone
            )
            variations.append(response)
        
        return variations
