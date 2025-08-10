"""
Conversation flow management for the AI assistant.

This module handles conversation state, flow control, turn-taking,
and coordination between different components during a conversation.
"""

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, field

from src.core.events.event_bus import EventBus
from src.core.events.event_types import Event, EventType
from src.core.error_handling import (
    ConversationError,
    ValidationError
)
from src.core.config.loader import ConfigLoader
from src.memory.core_memory.memory_manager import MemoryManager
from src.processing.natural_language.intent_manager import IntentManager
from src.integrations.model_inference_coordinator import ModelInferenceCoordinator
from src.skills.skill_registry import SkillRegistry
from src.learning.feedback_processor import FeedbackProcessor

logger = logging.getLogger(__name__)


class ConversationState(Enum):
    """Enumeration of possible conversation states."""
    IDLE = auto()
    LISTENING = auto()
    PROCESSING = auto()
    RESPONDING = auto()
    WAITING_FOR_INPUT = auto()
    ERROR = auto()
    TERMINATED = auto()


class TurnType(Enum):
    """Types of conversation turns."""
    USER = auto()
    ASSISTANT = auto()
    SYSTEM = auto()
    SKILL = auto()


@dataclass
class ConversationTurn:
    """Represents a single turn in a conversation."""
    turn_id: str
    turn_type: TurnType
    content: Any  # Can be text, audio, image, or multimodal
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    intent: Optional[str] = None
    entities: List[Dict[str, Any]] = field(default_factory=list)
    skill_invoked: Optional[str] = None
    response_time_ms: Optional[float] = None


@dataclass
class ConversationContext:
    """Maintains the context of an ongoing conversation."""
    session_id: str
    user_id: str
    start_time: datetime
    turns: List[ConversationTurn] = field(default_factory=list)
    active_skills: Set[str] = field(default_factory=set)
    topic_stack: List[str] = field(default_factory=list)
    emotional_state: Dict[str, float] = field(default_factory=dict)
    language: str = "en"
    modality: str = "text"
    custom_data: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def turn_count(self) -> int:
        """Get the number of turns in the conversation."""
        return len(self.turns)
    
    @property
    def last_user_turn(self) -> Optional[ConversationTurn]:
        """Get the last user turn."""
        for turn in reversed(self.turns):
            if turn.turn_type == TurnType.USER:
                return turn
        return None
    
    @property
    def last_assistant_turn(self) -> Optional[ConversationTurn]:
        """Get the last assistant turn."""
        for turn in reversed(self.turns):
            if turn.turn_type == TurnType.ASSISTANT:
                return turn
        return None


class ConversationManager:
    """
    Manages conversation flow and state for the AI assistant.
    
    This class orchestrates the entire conversation lifecycle, coordinating
    between various components like memory, skills, and response generation.
    """
    
    def __init__(
        self,
        event_bus: EventBus,
        memory_manager: MemoryManager,
        intent_manager: IntentManager,
        model_coordinator: ModelInferenceCoordinator,
        skill_registry: SkillRegistry,
        feedback_processor: FeedbackProcessor,
        config_loader: ConfigLoader
    ):
        """Initialize the conversation manager with required dependencies."""
        self.event_bus = event_bus
        self.memory_manager = memory_manager
        self.intent_manager = intent_manager
        self.model_coordinator = model_coordinator
        self.skill_registry = skill_registry
        self.feedback_processor = feedback_processor
        self.config = config_loader.get_config("conversation")
        
        # Conversation management
        self.active_conversations: Dict[str, ConversationContext] = {}
        self.conversation_states: Dict[str, ConversationState] = {}
        
        # Configuration
        self.max_conversation_length = self.config.get("max_turns", 100)
        self.context_window_size = self.config.get("context_window", 10)
        self.idle_timeout_seconds = self.config.get("idle_timeout", 300)
        self.max_concurrent_conversations = self.config.get("max_concurrent", 100)
        
        # Turn management
        self.turn_timeout_seconds = self.config.get("turn_timeout", 30)
        self.max_retries = self.config.get("max_retries", 3)
        
        # Response strategies
        self.clarification_threshold = self.config.get("clarification_threshold", 0.3)
        self.confidence_threshold = self.config.get("confidence_threshold", 0.7)
        
        # Subscribe to relevant events
        self._subscribe_to_events()
        
        logger.info("ConversationManager initialized successfully")
    
    def _subscribe_to_events(self) -> None:
        """Subscribe to relevant system events."""
        self.event_bus.subscribe(EventType.USER_INPUT, self._handle_user_input_event)
        self.event_bus.subscribe(EventType.SKILL_COMPLETED, self._handle_skill_completed)
        self.event_bus.subscribe(EventType.MEMORY_UPDATED, self._handle_memory_update)
        self.event_bus.subscribe(EventType.ERROR_OCCURRED, self._handle_error_event)
    
    async def start_conversation(
        self,
        user_id: str,
        initial_input: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Start a new conversation session.
        
        Args:
            user_id: Unique identifier for the user
            initial_input: Optional initial input to process
            metadata: Optional metadata for the conversation
            
        Returns:
            Session ID for the new conversation
            
        Raises:
            ConversationError: If unable to start conversation
        """
        try:
            # Check concurrent conversation limit
            if len(self.active_conversations) >= self.max_concurrent_conversations:
                raise ConversationError("Maximum concurrent conversations reached")
            
            # Create new session
            session_id = str(uuid.uuid4())
            context = ConversationContext(
                session_id=session_id,
                user_id=user_id,
                start_time=datetime.now(timezone.utc),
                custom_data=metadata or {}
            )
            
            # Initialize conversation state
            self.active_conversations[session_id] = context
            self.conversation_states[session_id] = ConversationState.IDLE
            
            # Initialize memory for this conversation
            await self.memory_manager.initialize_session(session_id, user_id)
            
            # Emit conversation started event
            await self.event_bus.emit(Event(
                type=EventType.CONVERSATION_STARTED,
                data={
                    "session_id": session_id,
                    "user_id": user_id,
                    "timestamp": context.start_time
                }
            ))
            
            # Process initial input if provided
            if initial_input:
                await self.process_input(session_id, initial_input)
            
            logger.info(f"Started conversation {session_id} for user {user_id}")
            return session_id
            
        except Exception as e:
            logger.error(f"Failed to start conversation: {str(e)}")
            raise ConversationError(f"Failed to start conversation: {str(e)}")
    
    async def process_input(
        self,
        session_id: str,
        user_input: Any,
        input_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process user input and generate a response.
        
        Args:
            session_id: Conversation session ID
            user_input: User input (text, audio, image, etc.)
            input_metadata: Optional metadata about the input
            
        Returns:
            Response dictionary containing the assistant's response
            
        Raises:
            ConversationError: If processing fails
        """
        context = self._get_conversation_context(session_id)
        if not context:
            raise ConversationError(f"Invalid session ID: {session_id}")
        
        # Update conversation state
        self.conversation_states[session_id] = ConversationState.PROCESSING
        
        try:
            # Create user turn
            user_turn = ConversationTurn(
                turn_id=str(uuid.uuid4()),
                turn_type=TurnType.USER,
                content=user_input,
                timestamp=datetime.now(timezone.utc),
                metadata=input_metadata or {}
            )
            
            # Add to conversation history
            context.turns.append(user_turn)
            
            # Process through pipeline
            response = await self._process_conversation_turn(
                context, user_turn
            )
            
            # Update state
            self.conversation_states[session_id] = ConversationState.IDLE
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing input: {str(e)}")
            self.conversation_states[session_id] = ConversationState.ERROR
            raise ConversationError(f"Failed to process input: {str(e)}")
    
    async def _process_conversation_turn(
        self,
        context: ConversationContext,
        user_turn: ConversationTurn
    ) -> Dict[str, Any]:
        """Process a single conversation turn through the full pipeline."""
        start_time = datetime.now(timezone.utc)
        
        # 1. Update working memory with current turn
        await self._update_working_memory(context, user_turn)
        
        # 2. Analyze intent and extract entities
        intent_result = await self.intent_manager.analyze(
            user_turn.content,
            context=self._build_intent_context(context)
        )
        
        user_turn.intent = intent_result.get("intent")
        user_turn.entities = intent_result.get("entities", [])
        
        # 3. Check if clarification is needed
        if intent_result.get("confidence", 1.0) < self.clarification_threshold:
            return await self._request_clarification(context, intent_result)
        
        # 4. Route to appropriate skill if needed
        skill_response = None
        if self._should_invoke_skill(intent_result):
            skill_response = await self._invoke_skill(
                context, user_turn, intent_result
            )
        
        # 5. Generate response
        response = await self._generate_response(
            context, user_turn, intent_result, skill_response
        )
        
        # 6. Create assistant turn
        assistant_turn = ConversationTurn(
            turn_id=str(uuid.uuid4()),
            turn_type=TurnType.ASSISTANT,
            content=response["content"],
            timestamp=datetime.now(timezone.utc),
            metadata=response.get("metadata", {}),
            skill_invoked=skill_response.get("skill_name") if skill_response else None,
            response_time_ms=(
                datetime.now(timezone.utc) - start_time
            ).total_seconds() * 1000
        )
        
        context.turns.append(assistant_turn)
        
        # 7. Update memory with assistant response
        await self._update_long_term_memory(context, user_turn, assistant_turn)
        
        # 8. Emit response event
        await self.event_bus.emit(Event(
            type=EventType.RESPONSE_GENERATED,
            data={
                "session_id": context.session_id,
                "turn_id": assistant_turn.turn_id,
                "response": response
            }
        ))
        
        return response
    
    async def _update_working_memory(
        self,
        context: ConversationContext,
        turn: ConversationTurn
    ) -> None:
        """Update working memory with the current turn."""
        working_memory = self.memory_manager.get_working_memory(context.session_id)
        
        # Add current turn to working memory
        await working_memory.add_item({
            "type": "conversation_turn",
            "turn_id": turn.turn_id,
            "content": turn.content,
            "timestamp": turn.timestamp,
            "turn_type": turn.turn_type.name
        })
        
        # Add recent context
        recent_turns = self._get_recent_turns(context, self.context_window_size)
        await working_memory.update_context({
            "recent_turns": [
                {
                    "type": t.turn_type.name,
                    "content": t.content,
                    "timestamp": t.timestamp.isoformat()
                }
                for t in recent_turns
            ],
            "active_topic": context.topic_stack[-1] if context.topic_stack else None,
            "active_skills": list(context.active_skills)
        })
    
    async def _update_long_term_memory(
        self,
        context: ConversationContext,
        user_turn: ConversationTurn,
        assistant_turn: ConversationTurn
    ) -> None:
        """Update long-term memory with conversation information."""
        # Store in episodic memory
        await self.memory_manager.store_episodic_memory(
            session_id=context.session_id,
            memory_data={
                "type": "conversation_exchange",
                "user_input": user_turn.content,
                "assistant_response": assistant_turn.content,
                "intent": user_turn.intent,
                "entities": user_turn.entities,
                "timestamp": user_turn.timestamp,
                "response_time_ms": assistant_turn.response_time_ms
            }
        )
        
        # Extract and store any semantic information
        if user_turn.entities:
            for entity in user_turn.entities:
                await self.memory_manager.store_semantic_memory(
                    session_id=context.session_id,
                    concept=entity.get("type", "unknown"),
                    data=entity
                )
    
    def _build_intent_context(self, context: ConversationContext) -> Dict[str, Any]:
        """Build context for intent analysis."""
        recent_turns = self._get_recent_turns(context, 5)
        
        return {
            "conversation_history": [
                {
                    "type": turn.turn_type.name,
                    "content": turn.content,
                    "intent": turn.intent
                }
                for turn in recent_turns
            ],
            "active_topic": context.topic_stack[-1] if context.topic_stack else None,
            "user_id": context.user_id,
            "language": context.language,
            "emotional_state": context.emotional_state
        }
    
    def _should_invoke_skill(self, intent_result: Dict[str, Any]) -> bool:
        """Determine if a skill should be invoked based on intent."""
        intent = intent_result.get("intent")
        confidence = intent_result.get("confidence", 0)
        
        # Check if intent maps to a skill
        if intent and confidence >= self.confidence_threshold:
            return self.skill_registry.has_skill_for_intent(intent)
        
        return False
    
    async def _invoke_skill(
        self,
        context: ConversationContext,
        user_turn: ConversationTurn,
        intent_result: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Invoke the appropriate skill based on intent."""
        try:
            intent = intent_result["intent"]
            skill_name = self.skill_registry.get_skill_for_intent(intent)
            
            if not skill_name:
                return None
            
            # Add skill to active skills
            context.active_skills.add(skill_name)
            
            # Get skill instance
            skill = self.skill_registry.get_skill(skill_name)
            
            # Prepare skill context
            skill_context = {
                "session_id": context.session_id,
                "user_id": context.user_id,
                "intent": intent,
                "entities": intent_result.get("entities", []),
                "conversation_context": self._build_intent_context(context),
                "user_input": user_turn.content
            }
            
            # Execute skill
            skill_result = await skill.execute(skill_context)
            
            # Emit skill execution event
            await self.event_bus.emit(Event(
                type=EventType.SKILL_EXECUTED,
                data={
                    "session_id": context.session_id,
                    "skill_name": skill_name,
                    "result": skill_result
                }
            ))
            
            return {
                "skill_name": skill_name,
                "result": skill_result
            }
            
        except Exception as e:
            logger.error(f"Error invoking skill: {str(e)}")
            return None
        finally:
            # Remove skill from active skills
            if skill_name in context.active_skills:
                context.active_skills.remove(skill_name)
    
    async def _generate_response(
        self,
        context: ConversationContext,
        user_turn: ConversationTurn,
        intent_result: Dict[str, Any],
        skill_response: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate response using the model inference coordinator."""
        # Build prompt context
        prompt_context = {
            "user_input": user_turn.content,
            "intent": intent_result.get("intent"),
            "entities": intent_result.get("entities", []),
            "conversation_history": self._format_conversation_history(context),
            "skill_response": skill_response,
            "user_preferences": await self._get_user_preferences(context.user_id),
            "emotional_context": context.emotional_state
        }
        
        # Select appropriate model based on context
        model_selection = await self.model_coordinator.select_model(
            task_type="conversation",
            context=prompt_context
        )
        
        # Generate response
        response = await self.model_coordinator.generate(
            model=model_selection["model"],
            prompt=self._build_response_prompt(prompt_context),
            parameters=model_selection.get("parameters", {})
        )
        
        # Post-process response
        processed_response = await self._post_process_response(
            response, context, skill_response
        )
        
        return processed_response
    
    async def _request_clarification(
        self,
        context: ConversationContext,
        intent_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate a clarification request when intent is unclear."""
        ambiguous_intents = intent_result.get("possible_intents", [])
        
        clarification_prompt = self._build_clarification_prompt(
            context, ambiguous_intents
        )
        
        response = await self.model_coordinator.generate(
            model="conversation",
            prompt=clarification_prompt,
            parameters={"temperature": 0.7, "max_tokens": 150}
        )
        
        return {
            "content": response["text"],
            "type": "clarification",
            "metadata": {
                "ambiguous_intents": ambiguous_intents,
                "original_confidence": intent_result.get("confidence", 0)
            }
        }
    
    def _build_response_prompt(self, context: Dict[str, Any]) -> str:
        """Build prompt for response generation."""
        prompt_parts = []
        
        # Add conversation history
        if context.get("conversation_history"):
            prompt_parts.append("Conversation history:")
            prompt_parts.append(context["conversation_history"])
            prompt_parts.append("")
        
        # Add current user input
        prompt_parts.append(f"User: {context['user_input']}")
        
        # Add intent and entities
        if context.get("intent"):
            prompt_parts.append(f"Detected intent: {context['intent']}")
        
        if context.get("entities"):
            prompt_parts.append(f"Entities: {context['entities']}")
        
        # Add skill response if available
        if context.get("skill_response"):
            skill_result = context["skill_response"]["result"]
            prompt_parts.append(f"\nSkill response: {skill_result}")
        
        # Add instruction
        prompt_parts.append("\nGenerate a helpful and contextual response:")
        
        return "\n".join(str(part) for part in prompt_parts)
    
    def _build_clarification_prompt(
        self,
        context: ConversationContext,
        ambiguous_intents: List[str]
    ) -> str:
        """Build prompt for clarification request."""
        recent_context = self._format_conversation_history(context, max_turns=3)
        
        prompt = f"""Recent conversation:
{recent_context}

The user's intent is unclear. Possible intents: {', '.join(ambiguous_intents)}

Generate a polite clarification question to understand what the user wants:"""
        
        return prompt
    
    async def _post_process_response(
        self,
        response: Dict[str, Any],
        context: ConversationContext,
        skill_response: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Post-process the generated response."""
        processed = {
            "content": response.get("text", ""),
            "type": "response",
            "metadata": {
                "model": response.get("model"),
                "confidence": response.get("confidence", 1.0)
            }
        }
        
        # Add skill-specific metadata
        if skill_response:
            processed["metadata"]["skill"] = skill_response["skill_name"]
            processed["metadata"]["skill_data"] = skill_response.get("result", {})
        
        # Add any multimedia content
        if response.get("multimedia"):
            processed["multimedia"] = response["multimedia"]
        
        # Add suggested actions
        if response.get("actions"):
            processed["actions"] = response["actions"]
        
        return processed
    
    def _format_conversation_history(
        self,
        context: ConversationContext,
        max_turns: Optional[int] = None
    ) -> str:
        """Format conversation history as a string."""
        turns_to_include = self._get_recent_turns(
            context, max_turns or self.context_window_size
        )
        
        formatted_turns = []
        for turn in turns_to_include:
            prefix = {
                TurnType.USER: "User",
                TurnType.ASSISTANT: "Assistant",
                TurnType.SYSTEM: "System",
                TurnType.SKILL: "Skill"
            }.get(turn.turn_type, "Unknown")
            
            formatted_turns.append(f"{prefix}: {turn.content}")
        
        return "\n".join(formatted_turns)
    
    def _get_recent_turns(
        self,
        context: ConversationContext,
        n: int
    ) -> List[ConversationTurn]:
        """Get the n most recent turns from the conversation."""
        return context.turns[-n:] if len(context.turns) > n else context.turns
    
    async def _get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """Retrieve user preferences from memory."""
        try:
            preferences = await self.memory_manager.get_user_preferences(user_id)
            return preferences or {}
        except Exception as e:
            logger.warning(f"Failed to retrieve user preferences: {str(e)}")
            return {}
    
    def _get_conversation_context(self, session_id: str) -> Optional[ConversationContext]:
        """Get conversation context by session ID."""
        return self.active_conversations.get(session_id)
    
    async def end_conversation(self, session_id: str) -> None:
        """
        End a conversation session.
        
        Args:
            session_id: Session ID to end
            
        Raises:
            ConversationError: If session doesn't exist
        """
        context = self._get_conversation_context(session_id)
        if not context:
            raise ConversationError(f"Invalid session ID: {session_id}")
        
        try:
            # Update conversation state
            self.conversation_states[session_id] = ConversationState.TERMINATED
            
            # Save conversation summary
            await self._save_conversation_summary(context)
            
            # Clean up memory
            await self.memory_manager.cleanup_session(session_id)
            
            # Emit conversation ended event
            await self.event_bus.emit(Event(
                type=EventType.CONVERSATION_ENDED,
                data={
                    "session_id": session_id,
                    "user_id": context.user_id,
                    "duration_seconds": (
                        datetime.now(timezone.utc) - context.start_time
                    ).total_seconds(),
                    "turn_count": context.turn_count
                }
            ))
            
            # Remove from active conversations
            del self.active_conversations[session_id]
            del self.conversation_states[session_id]
            
            logger.info(f"Ended conversation {session_id}")
            
        except Exception as e:
            logger.error(f"Error ending conversation: {str(e)}")
            raise ConversationError(f"Failed to end conversation: {str(e)}")
    
    async def _save_conversation_summary(self, context: ConversationContext) -> None:
        """Save a summary of the conversation for future reference."""
        try:
            # Generate summary using model
            summary_prompt = f"""Summarize this conversation in a few sentences:
{self._format_conversation_history(context)}

Summary:"""
            
            summary_response = await self.model_coordinator.generate(
                model="summarization",
                prompt=summary_prompt,
                parameters={"temperature": 0.3, "max_tokens": 200}
            )
            
            # Store summary in long-term memory
            await self.memory_manager.store_conversation_summary(
                session_id=context.session_id,
                user_id=context.user_id,
                summary={
                    "text": summary_response["text"],
                    "start_time": context.start_time,
                    "end_time": datetime.now(timezone.utc),
                    "turn_count": context.turn_count,
                    "topics": context.topic_stack,
                    "skills_used": list(context.active_skills)
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to save conversation summary: {str(e)}")
    
    async def get_conversation_state(self, session_id: str) -> ConversationState:
        """Get the current state of a conversation."""
        return self.conversation_states.get(session_id, ConversationState.TERMINATED)
    
    async def get_conversation_history(
        self,
        session_id: str,
        max_turns: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get conversation history for a session."""
        context = self._get_conversation_context(session_id)
        if not context:
            raise ConversationError(f"Invalid session ID: {session_id}")
        
        turns = context.turns[-max_turns:] if max_turns else context.turns
        
        return [
            {
                "turn_id": turn.turn_id,
                "type": turn.turn_type.name,
                "content": turn.content,
                "timestamp": turn.timestamp.isoformat(),
                "intent": turn.intent,
                "entities": turn.entities,
                "metadata": turn.metadata
            }
            for turn in turns
        ]
    
    async def provide_feedback(
        self,
        session_id: str,
        turn_id: str,
        feedback: Dict[str, Any]
    ) -> None:
        """Process user feedback for a specific turn."""
        context = self._get_conversation_context(session_id)
        if not context:
            raise ConversationError(f"Invalid session ID: {session_id}")
        
        # Find the turn
        turn = next((t for t in context.turns if t.turn_id == turn_id), None)
        if not turn:
            raise ConversationError(f"Invalid turn ID: {turn_id}")
        
        # Process feedback
        await self.feedback_processor.process_feedback({
            "session_id": session_id,
            "turn_id": turn_id,
            "turn_type": turn.turn_type.name,
            "feedback": feedback,
            "context": {
                "intent": turn.intent,
                "content": turn.content,
                "timestamp": turn.timestamp
            }
        })
    
    async def _handle_user_input_event(self, event: Event) -> None:
        """Handle user input events from the event bus."""
        try:
            session_id = event.data.get("session_id")
            user_input = event.data.get("input")
            
            if session_id and user_input:
                await self.process_input(session_id, user_input)
                
        except Exception as e:
            logger.error(f"Error handling user input event: {str(e)}")
    
    async def _handle_skill_completed(self, event: Event) -> None:
        """Handle skill completion events."""
        try:
            session_id = event.data.get("session_id")
            skill_name = event.data.get("skill_name")
            
            context = self._get_conversation_context(session_id)
            if context and skill_name in context.active_skills:
                context.active_skills.remove(skill_name)
                
        except Exception as e:
            logger.error(f"Error handling skill completed event: {str(e)}")
    
    async def _handle_memory_update(self, event: Event) -> None:
        """Handle memory update events."""
        # This can be used to update conversation context based on memory changes
        pass
    
    async def _handle_error_event(self, event: Event) -> None:
        """Handle error events."""
        try:
            session_id = event.data.get("session_id")
            if session_id and session_id in self.conversation_states:
                self.conversation_states[session_id] = ConversationState.ERROR
                
        except Exception as e:
            logger.error(f"Error handling error event: {str(e)}")
    
    async def cleanup_idle_conversations(self) -> None:
        """Clean up conversations that have been idle for too long."""
        current_time = datetime.now(timezone.utc)
        sessions_to_clean = []
        
        for session_id, context in self.active_conversations.items():
            if context.turns:
                last_activity = context.turns[-1].timestamp
                idle_duration = (current_time - last_activity).total_seconds()
                
                if idle_duration > self.idle_timeout_seconds:
                    sessions_to_clean.append(session_id)
        
        # Clean up idle sessions
        for session_id in sessions_to_clean:
            try:
                await self.end_conversation(session_id)
                logger.info(f"Cleaned up idle conversation: {session_id}")
            except Exception as e:
                logger.error(f"Error cleaning up session {session_id}: {str(e)}")
    
    async def get_active_conversations(self) -> List[Dict[str, Any]]:
        """Get list of all active conversations."""
        return [
            {
                "session_id": context.session_id,
                "user_id": context.user_id,
                "state": self.conversation_states.get(
                    context.session_id, ConversationState.IDLE
                ).name,
                "start_time": context.start_time.isoformat(),
                "turn_count": context.turn_count,
                "last_activity": (
                    context.turns[-1].timestamp.isoformat()
                    if context.turns else context.start_time.isoformat()
                )
            }
            for context in self.active_conversations.values()
        ]
