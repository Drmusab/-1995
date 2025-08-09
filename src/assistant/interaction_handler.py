"""
Interaction Handler for AI Assistant System

This module manages user interactions across different modalities and channels.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set
import uuid
import logging

from src.core.dependency_injection import Container
from src.core.events.event_bus import EventBus
from src.core.events.event_types import (
    UserInteractionStarted,
    UserInteractionCompleted
)


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
        self.logger = logging.getLogger(__name__)

    async def initialize(self) -> None:
        """Initialize the interaction handler."""
        self.logger.info("Initializing Interaction Handler")
        # Stub initialization
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
        """
        Start a new interaction.
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            interaction_mode: Mode of interaction
            input_modalities: Supported input modalities
            output_modalities: Supported output modalities
            context: Optional interaction context
            
        Returns:
            Interaction identifier
        """
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
        """
        Process a user message and generate response.
        
        Args:
            interaction_id: Interaction identifier
            message: User message to process
            real_time: Enable real-time processing
            streaming: Enable streaming response
            
        Returns:
            Assistant response
        """
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
        # Simple response generation based on input
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
        """
        End an interaction.
        
        Args:
            interaction_id: Interaction identifier
            
        Returns:
            True if successful, False if interaction not found
        """
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